"""``safe_text.escape_label`` の試験（画面に出す文字を無害化する土台）."""

from __future__ import annotations

import pytest

from shokoban.safe_text import MARKDOWN_CHARACTERS, escape_label

DANGEROUS_FORMS = ("![", "](http", "<img", "**", "|", "`")


def has_unescaped(text: str, fragment: str) -> bool:
    """Markdown の意味を持つ形（バックスラッシュで無害化されていない形）で残っているか."""
    start = 0
    while True:
        index = text.find(fragment, start)
        if index < 0:
            return False
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return True
        start = index + 1


def test_every_markdown_character_is_escaped() -> None:
    for character in MARKDOWN_CHARACTERS:
        assert escape_label(character) == "\\" + character


def test_plain_text_is_left_alone() -> None:
    assert escape_label("やまだ商事 2026年11月") == "やまだ商事 2026年11月"
    assert escape_label("") == ""


def test_newlines_become_spaces() -> None:
    assert escape_label("1行目\n2行目\r\n3行目") == "1行目 2行目  3行目"


def test_image_html_and_link_payloads_are_neutralized() -> None:
    payloads = (
        "![x](http://127.0.0.1:9/p?v=1)",
        "<img src=http://127.0.0.1:9/q>",
        "[link](http://127.0.0.1:9/r)",
        "# 見出し",
        "**強い** _斜体_ `code`",
        "| a | b |",
        "~取り消し~ :colon:",
        "波 { } 括弧",
    )
    for payload in payloads:
        escaped = escape_label(payload)
        for forbidden in DANGEROUS_FORMS:
            assert not has_unescaped(escaped, forbidden), (payload, escaped)
        # 元の文字は消えていない（バックスラッシュを外せば元の文字に戻る）。
        assert escaped != payload
        assert escaped.replace("\\", "").replace("\u200b", "") == payload  # 自動リンク切りの見えない区切りは除いて比べる


def test_escape_label_adds_one_backslash_per_markdown_character() -> None:
    payload = "![x](http://127.0.0.1:9/p?v=1)"
    assert escape_label(payload).count("\\") == sum(
        1 for character in payload if character in MARKDOWN_CHARACTERS
    )


def test_escape_label_handles_a_literal_backslash() -> None:
    assert escape_label("a\\b") == "a\\\\b"
    assert escape_label("\\<img") == "\\\\\\<img"


def test_escape_label_rejects_non_strings() -> None:
    for value in (None, 3, b"bytes", ["list"]):
        with pytest.raises(TypeError):
            escape_label(value)  # type: ignore[arg-type]


def test_escape_label_breaks_bare_url_autolinks() -> None:
    # 画像記法を無害化しても、素の URL が自動リンクになることがある（通信は起きない）ので、それも崩す
    for payload in ("http://127.0.0.1:9/p", "HTTPS://example.invalid/x", "www.example.invalid", "WWW.example.invalid"):
        out = escape_label(payload)
        assert "://" not in out
        assert "www." not in out.lower().replace("\\", "")
