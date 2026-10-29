"""派生产物的通用工具：manifest 比对与原子写入。

`chunks.jsonl` 是唯一上游，向量与 BM25 索引都记录它的全字节 sha256。
把「manifest 比对 + 原子落盘」收在一处，新增派生产物时不必再复刻一套逻辑。
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


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


def matches(directory: pathlib.Path, key: ArtifactKey, name: str = "manifest.json") -> bool:
    manifest = read_manifest(directory, name)
    if manifest is None:
        return False
    return (
        manifest.get("kind") == key.kind
        and manifest.get("source_digest") == key.source_digest
        and manifest.get("params") == dict(key.params)
    )


def write_json_atomic(path: pathlib.Path, data: Mapping[str, Any]) -> None:
    """先写同目录临时文件再 rename，避免半截 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(temp, path)
    except BaseException:
        pathlib.Path(temp).unlink(missing_ok=True)
        raise


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


def write_npy_atomic(path: pathlib.Path, array: np.ndarray) -> None:
    """先写临时 `.npy` 再 rename，避免崩溃留下半截矩阵。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        with open(temp, "wb") as fh:
            np.save(fh, array)
        os.replace(temp, path)
    except BaseException:
        pathlib.Path(temp).unlink(missing_ok=True)
        raise
