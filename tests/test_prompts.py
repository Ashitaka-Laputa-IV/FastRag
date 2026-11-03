"""提示词模板：mustache 占位符与字面量花括号。"""

from __future__ import annotations

import pytest

from fastrag.errors import FastragError
from fastrag.llm import prompts as prompts_module


@pytest.fixture
def override(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts_module, "home", lambda: tmp_path)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "demo.txt").write_text(
        'JSON: {"a": 1}\n问题: {{question}}\n', encoding="utf-8"
    )


def test_render_keeps_literal_braces(override):
    text = prompts_module.render("demo", question="你好")
    assert '{"a": 1}' in text
    assert "问题: 你好" in text


def test_missing_variable_raises(override):
    with pytest.raises(FastragError, match="question"):
        prompts_module.render("demo", other="x")
