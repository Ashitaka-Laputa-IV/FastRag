"""`.env` 读取：引号、行尾注释与不覆盖已有环境变量。"""

from __future__ import annotations

import os

from fastrag.config import load_env


def test_load_env_parses_and_keeps_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('A="x#y"  # 注释\nB=来自文件\nexport C=2\n', encoding="utf-8")
    monkeypatch.setenv("B", "已有值")
    monkeypatch.setenv("A", "旧值")

    load_env(env)

    assert os.environ["A"] == "旧值"
    assert os.environ["B"] == "已有值"
    assert os.environ["C"] == "2"


def test_load_env_missing_file_is_noop(tmp_path):
    load_env(tmp_path / "missing.env")
