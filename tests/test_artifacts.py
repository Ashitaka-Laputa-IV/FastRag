"""原子写入：内容完整且不残留临时文件。"""

from __future__ import annotations

import json

from fastrag.ingest.artifacts import write_atomic, write_json_atomic


def test_write_atomic_replaces_without_temp_files(tmp_path):
    path = tmp_path / "nested" / "out.bin"
    write_atomic(path, b"ok")
    assert path.read_bytes() == b"ok"
    assert [item.name for item in path.parent.iterdir()] == ["out.bin"]


def test_write_json_atomic_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    write_json_atomic(path, {"kind": "demo", "值": [1, 2]})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "kind": "demo",
        "值": [1, 2],
    }
