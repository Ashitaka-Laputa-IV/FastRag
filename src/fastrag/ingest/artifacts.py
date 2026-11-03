"""派生产物的通用工具：manifest 比对与原子写入。

`chunks.jsonl` 是唯一上游，向量与 BM25 索引都记录它的全字节 sha256。
把「manifest 比对 + 原子落盘」收在一处，新增派生产物时不必再复刻一套逻辑。
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ArtifactKey:
    """一份派生产物的身份：类型 + 上游指纹 + 构建参数。"""

    kind: str
    source_digest: str
    params: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_digest": self.source_digest,
            "params": dict(self.params),
        }


def read_manifest(directory: pathlib.Path, name: str = "manifest.json") -> dict | None:
    """读取 manifest；不存在或损坏时返回 None，由调用方决定重建。"""
    path = directory / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def matches(
    directory: pathlib.Path, key: ArtifactKey, name: str = "manifest.json"
) -> bool:
    manifest = read_manifest(directory, name)
    if manifest is None:
        return False
    return (
        manifest.get("kind") == key.kind
        and manifest.get("source_digest") == key.source_digest
        and manifest.get("params") == dict(key.params)
    )


@contextlib.contextmanager
def atomic_path(path: pathlib.Path, suffix: str = "") -> Iterator[pathlib.Path]:
    """在目标目录建临时文件，正常退出时原子替换目标；失败则清理。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=suffix, dir=path.parent
    )
    os.close(fd)
    temp_path = pathlib.Path(temp)
    try:
        yield temp_path
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def write_atomic(path: pathlib.Path, data: bytes) -> None:
    """先写同目录临时文件再 rename，避免崩溃留下半截文件。"""
    with atomic_path(path) as temp:
        temp.write_bytes(data)


def write_json_atomic(path: pathlib.Path, data: Mapping[str, Any]) -> None:
    """原子写 JSON（含行尾换行）。"""
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    write_atomic(path, text.encode("utf-8"))


def write_manifest(
    directory: pathlib.Path,
    key: ArtifactKey,
    extra: Mapping[str, Any] | None = None,
    name: str = "manifest.json",
) -> None:
    payload = key.as_dict()
    if extra:
        payload.update(extra)
    write_json_atomic(directory / name, payload)
