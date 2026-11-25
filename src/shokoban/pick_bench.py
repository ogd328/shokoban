"""5欄の名前・文の分け方・行の振り分けの決まり（``assemble_r3`` などの土台）.

このモジュールが持っているもの:

- 5欄の名前（``FIELDS``）、1行の長さ（``MAX_SPAN_CHARS``）と欄ごとの行数
  （``MAX_ROWS_PER_FIELD``）の上限、行を落とした理由のコード（``REASON_*``）。
  欄の名前は ``assemble_r3`` と画面側（``gui``・``mac_app``）、上限と理由コードは
  ``assemble_r3`` と ``cloud_memo`` が使う。
- ``split_sentences``: 文字起こしを「。」「！」「？」と改行で文に分ける
  （同じ入力なら同じ結果）。``name_review`` が使う。
- 名前（敬称付き）・3桁以上の数字・電話・メールの形と、数字を残してよい作業の量の
  単位（``_NAME_RE`` など）。``mask_r3`` と ``mask_stage5`` が使う。
- 合意・先送り・依頼の言葉と、検査を通った1行（``_Row``）を ``合意なし``・
  ``依頼なし``・``先送り`` の決まりで欄に振り分ける ``_route_rows``。
  ``assemble_r3`` が、読む役が選んだ欄に当てた場合の結果を記録するのに使う。

名前に ``bench`` が付いているのは歴史的な名前。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


FIELDS: tuple[str, ...] = (
    "困りごと", "頼まれていること", "こちらがやること", "決まったこと", "まだ決まっていないこと",
)

MAX_SPAN_CHARS = 120
MAX_ROWS_PER_FIELD = 8

# 落とした理由（共有の欄には出さない。理由コードだけ dropped に残す）。
REASON_SHAPE = "形崩れ"
REASON_NUMBER = "番号違い"
REASON_NOT_PART = "部分でない"
REASON_TOO_LONG = "長すぎ"
REASON_NO_AGREEMENT = "合意なし"
REASON_NO_REQUEST = "依頼なし"
REASON_DEFERRED = "先送り"
REASON_ROWS = "行数"


# 合意の言葉（足し引きしてよい。広げすぎると合意でない文まで決まったことにしてしまう）。
AGREEMENT_PHRASES: tuple[str, ...] = (
    "それでお願いします", "それでいきましょう", "それで行きましょう", "それでいこう",
    "それで進めて", "じゃあそれで", "ではそれで", "それで決まり", "決まりですね",
    "決定で", "確定で", "それで大丈夫です", "お願いします、それで",
)

# 先送りの言葉（「今度」は先送りでない文にも当たりやすいので、足し引きは慎重に）。
DEFERRAL_PHRASES: tuple[str, ...] = (
    "あとで", "後で", "次回", "今度", "落ち着いたら",
    "後回し", "そのうち", "いずれ", "来年以降", "来期",
)

# 依頼の言葉（足し引きしてよい）。
# 【こちらがやること】に残せるのは、根拠の文そのものにこの言葉がある行だけ。
REQUEST_PHRASES: tuple[str, ...] = (
    "てもらえますか", "お願いしたい", "お願いできますか",
)


DEFERRAL_MARK = "（先送り）"


# ---------------------------------------------------------------- 文の分け方

_TERMINALS = "。！？!?"


def split_sentences(text: str) -> list[str]:
    """「。」「！」「？」「!」「?」と改行で文に分ける.

    区切り文字はその文に含め、空白だけの文は捨てる。同じ入力なら同じ結果。
    """
    if type(text) is not str:
        raise ValueError("text は文字列で指定してください")

    sentences: list[str] = []
    buffer: list[str] = []
    index = 0
    length = len(text)

    def flush() -> None:
        sentence = "".join(buffer).strip()
        if sentence:
            sentences.append(sentence)
        buffer.clear()

    while index < length:
        char = text[index]
        if char in "\r\n":
            flush()
            index += 1
            continue
        buffer.append(char)
        index += 1
        if char in _TERMINALS:
            # 「！？」「！！」のような連続は同じ文の末尾として扱う。
            while index < length and text[index] in _TERMINALS:
                buffer.append(text[index])
                index += 1
            flush()
    flush()
    return sentences


# ---------------------------------------------------------------- 名前・数字・連絡先の形

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"0\d{1,4}[-‐‑ー()（）]\d{1,4}[-‐‑ー()（）]\d{3,4}|0\d{9,10}")
# 敬称付きの名前（2文字以上の名前＋さん/様/殿）。「田中さん」のような短い名前も伏せられるよう、
# 3文字以上に限らず2文字から当てる。
_NAME_RE = re.compile(r"[一-龥々ァ-ヶA-Za-z0-9][一-龥々ァ-ヶA-Za-z0-9・ー\-]{1,20}(?:さん|様|殿)")
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d{3,}")
# 3桁以上の数字でも、作業の量の単位が直後に付くものは伏せない。
_QUANTITY_UNITS = ("ページ", "部", "枚", "回", "本", "投稿")


# ---------------------------------------------------------------- 振り分け

@dataclass
class _Row:
    """検査を通った1行（欄の振り分け前）."""

    field_ai: str
    field: str
    sent: int
    span_masked: str
    rules: list[str]
    deferred: bool = False


def _near(sentences: Sequence[str], sent: int, phrases: Sequence[str]) -> bool:
    """文 ``sent`` かその前後の文（まとまりの境目をまたぐ）に言葉があるか."""
    for number in (sent - 1, sent, sent + 1):
        if 1 <= number <= len(sentences):
            text = sentences[number - 1]
            if any(phrase in text for phrase in phrases):
                return True
    return False


def _in_sentence(sentences: Sequence[str], sent: int, phrases: Sequence[str]) -> bool:
    """文 ``sent`` そのものに言葉があるか（前後の文は見ない）."""
    if not 1 <= sent <= len(sentences):
        return False
    text = sentences[sent - 1]
    return any(phrase in text for phrase in phrases)


def _route_rows(rows: Sequence[_Row], sentences: Sequence[str]) -> None:
    """合意なし → 依頼なし → 先送り の順に欄を振り分ける（その場で書き換える）.

    - 【決まったこと】で前後に合意の言葉が無い行は【まだ決まっていないこと】へ。
    - 【こちらがやること】で根拠の文そのものに依頼の言葉が無い行は
      【まだ決まっていないこと】へ（前後の文は見ない）。
    - 前後に先送りの言葉がある行は印を付け、【決まったこと】【こちらがやること】
      の行は【まだ決まっていないこと】へ移す。【困りごと】【頼まれていること】は
      欄を変えない。依頼なしで未決へ移った行も、前後に先送り語があれば印が付く。
    """
    for row in rows:
        if row.field == "決まったこと" and not _near(
            sentences, row.sent, AGREEMENT_PHRASES
        ):
            row.field = "まだ決まっていないこと"
            row.rules.append(REASON_NO_AGREEMENT)
        if row.field == "こちらがやること" and not _in_sentence(
            sentences, row.sent, REQUEST_PHRASES
        ):
            row.field = "まだ決まっていないこと"
            row.rules.append(REASON_NO_REQUEST)
        if _near(sentences, row.sent, DEFERRAL_PHRASES):
            row.deferred = True
            if row.field in ("決まったこと", "こちらがやること"):
                row.field = "まだ決まっていないこと"
            row.rules.append(REASON_DEFERRED)