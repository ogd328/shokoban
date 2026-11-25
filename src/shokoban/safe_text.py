"""画面のラベルに出す文字から、Markdown の意味を落とす.

Streamlit の見出し・ボタン・チェック・選択・折りたたみのラベルは、Markdown と
して解釈される。手元AIの答え・お客様名・題・見出し・根拠のような外から来た
文字をそのまま入れると、``![...](http://...)`` の画像記法でブラウザが勝手に
通信してしまう。

本文（答え・打ち合わせの本文・メモ・根拠・使った記録）は ``st.text``・
``st.code``・``st.text_area(disabled=True)`` で出し、この関数は通さない。
ラベルに入れざるを得ない外から来た文字だけを、ここで無害化する。
"""

from __future__ import annotations

# CommonMark がバックスラッシュで無害化できる ASCII 記号のうち、画面の見た目や
# 画像・リンク・HTML・見出し・表に意味を持つもの全部。
MARKDOWN_CHARACTERS = frozenset("\\`*_{}[]()#+-.!|<>~:")

__all__ = ["MARKDOWN_CHARACTERS", "escape_label"]


def escape_label(text: str) -> str:
    """ラベル用の文字列にする（Markdown の記号を全部バックスラッシュで無害化）.

    改行・復帰は空白1つにする（ラベルは1行の約束）。文字そのものは消さないので、
    元の文字は読める。二重に通すとバックスラッシュが画面に見えるので、画面の
    表示はどこか1か所でだけ通す。
    """
    if type(text) is not str:
        raise TypeError("escape_label は文字列だけを受け付けます")
    parts: list[str] = []
    # URL の自動リンク（http://… や www.…）を切るため、見えない区切りを挟む
    text = text.replace("://", ":\u200b//").replace("www.", "www\u200b.").replace("WWW.", "WWW\u200b.")
    for char in text:
        if char in ("\n", "\r"):
            parts.append(" ")
        elif char in MARKDOWN_CHARACTERS:
            parts.append("\\" + char)
        else:
            parts.append(char)
    return "".join(parts)
