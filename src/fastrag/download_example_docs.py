"""下载 LangChain 官方 Python 文档到 data/langchain-docs/。

目标目录来自 config/fastrag.toml 的 paths.docs_dir。
只抓 oss/python 下的文档（LangChain / LangGraph 等库的使用知识），
排除 contributing 等非库内容。

索引里给出的都是 `.md` 地址（直接返回 markdown 源码），必须原样下载；
如果去掉 `.md` 去请求，拿到的是网页 HTML，写进 .md 文件就会变成一堆标签。

用法:
    uv run python -m fastrag.download_example_docs
"""

import concurrent.futures as cf
import pathlib
import re
import time
import urllib.request

from fastrag.config import load_config

INDEX_URL = (
    "https://docs.langchain.com/_llms/agent-development-lifecycle/build/python.txt"
)
BASE = "https://docs.langchain.com/"
PAGE_PATTERN = re.compile(r"https://docs\.langchain\.com/oss/python/[^)\s\"'<>]+")
EXCLUDE = ("/contributing/",)
USER_AGENT = "fastrag-docs-downloader/0.1"
MAX_RETRIES = 3
HTML_PREFIXES = (b"<!doctype", b"<html")


def fetch(url: str) -> bytes:
    """带重试地抓取 URL。"""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=30) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1 + attempt)
    assert last_error is not None
    raise last_error


def collect_urls() -> list[str]:
    """从官方索引里取出 oss/python 文档 URL（去重、去锚点、剔除排除项）。"""
    index_text = fetch(INDEX_URL).decode("utf-8")

    urls = set()
    for match in PAGE_PATTERN.findall(index_text):
        url = match.split("#", 1)[0].rstrip(".,;")
        if any(part in url for part in EXCLUDE):
            continue
        # 索引里的链接已经带 .md（markdown 源码地址），缺了才补上
        if not url.endswith(".md"):
            url = f"{url.rstrip('/')}.md"
        urls.add(url)
    return sorted(urls)


def download(url: str, out_dir: pathlib.Path) -> int:
    """下载单个页面（markdown），路径镜像官方 URL 结构，返回字节数。"""
    rel = url.removeprefix(BASE)
    dest = out_dir / rel
    content = fetch(url)

    head = content[:512].lstrip().lower()
    if not content.strip() or head.startswith(HTML_PREFIXES):
        raise ValueError("返回内容不是 markdown（可能是 HTML 页面或空响应）")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return len(content)


def main() -> None:
    out_dir = load_config().paths.docs_dir
    urls = collect_urls()
    succeeded, total, failed = 0, 0, []
    with cf.ThreadPoolExecutor(12) as executor:
        futures = {executor.submit(download, url, out_dir): url for url in urls}
        for future in cf.as_completed(futures):
            url = futures[future]
            try:
                total += future.result()
                succeeded += 1
            except Exception as exc:  # noqa: BLE001
                failed.append((url, repr(exc)[:100]))

    print(f"成功 {succeeded}/{len(urls)}，共 {total / 1024 / 1024:.2f} MB -> {out_dir}")
    for url, error in failed:
        print(f"  失败 {url} {error}")


if __name__ == "__main__":
    main()
