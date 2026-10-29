"""下载示例文档（markdown 源码）。

- 索引地址、并发数、超时与重试次数来自 `[download]`；
- 只重试瞬时错误（超时、429、5xx），404/403 直接失败；
- 路径镜像官方 URL 结构，并校验不会逃出目标目录；
- 单文件先写临时文件再原子替换，避免半截 markdown。
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import pathlib
import re
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from fastrag.config import Config
from fastrag.errors import FastragError

HTML_PREFIXES = (b"<!doctype", b"<html")
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _page_pattern(base_url: str) -> re.Pattern[str]:
    root = re.escape(base_url.rstrip("/")) + "/oss/python/"
    return re.compile(root + r"[^)\s\"'<>]+")


@dataclass(frozen=True, slots=True)
class DownloadSummary:
    succeeded: int
    total_bytes: int
    failed: tuple[tuple[str, str], ...]

    @property
    def attempted(self) -> int:
        return self.succeeded + len(self.failed)


def fetch(url: str, config: Config) -> bytes:
    """带重试地抓取 URL；只对瞬时错误退避重试。"""
    settings = config.download
    attempts = settings.retries + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": settings.user_agent})
            with urllib.request.urlopen(request, timeout=settings.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUS:
                raise FastragError(f"下载失败 HTTP {exc.code}: {url}") from None
            last_error = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(1 + attempt)
    raise FastragError(f"下载失败（重试 {attempts} 次）: {url}: {last_error}")


def collect_urls(config: Config) -> list[str]:
    """从官方索引里取出 oss/python 文档 URL（去重、去锚点、剔除排除项）。"""
    settings = config.download
    try:
        index_text = fetch(settings.index_url, config).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FastragError(f"索引不是 UTF-8 文本: {settings.index_url}") from exc

    urls: set[str] = set()
    for match in _page_pattern(settings.base_url).findall(index_text):
        url = match.split("#", 1)[0].rstrip(".,;")
        if any(part in url for part in settings.exclude):
            continue
        if not url.endswith(".md"):
            url = f"{url.rstrip('/')}.md"
        urls.add(url)
    return sorted(urls)


def _safe_destination(url: str, out_dir: pathlib.Path, base_url: str) -> pathlib.Path:
    rel = url.removeprefix(base_url)
    if rel == url or rel.startswith(("http://", "https://")):
        raise FastragError(f"URL 不在允许的站点下: {url}")
    root = out_dir.resolve()
    dest = (root / rel).resolve()
    if not dest.is_relative_to(root):
        raise FastragError(f"URL 解析出的路径越界: {url}")
    return dest


def download(url: str, out_dir: pathlib.Path, config: Config) -> int:
    """下载单个页面（markdown），返回字节数。"""
    settings = config.download
    dest = _safe_destination(url, out_dir, settings.base_url)
    content = fetch(url, config)

    head = content[:512].lstrip().lower()
    if not content.strip() or head.startswith(HTML_PREFIXES):
        raise FastragError("返回内容不是 markdown（可能是 HTML 页面或空响应）")

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content)
        os.replace(temp, dest)
    except BaseException:
        pathlib.Path(temp).unlink(missing_ok=True)
        raise
    return len(content)


def download_docs(config: Config) -> DownloadSummary:
    """并发下载全部文档，单个失败不影响其他文件。"""
    out_dir = config.paths.docs_dir
    urls = collect_urls(config)
    succeeded, total_bytes = 0, 0
    failed: list[tuple[str, str]] = []
    with cf.ThreadPoolExecutor(config.download.concurrency) as executor:
        futures = {executor.submit(download, url, out_dir, config): url for url in urls}
        for future in cf.as_completed(futures):
            url = futures[future]
            try:
                total_bytes += future.result()
                succeeded += 1
            except Exception as exc:  # noqa: BLE001 - 单个 URL 失败继续下载其余
                failed.append((url, f"{type(exc).__name__}: {exc}"))
    return DownloadSummary(succeeded, total_bytes, tuple(failed))
