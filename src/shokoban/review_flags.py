"""送る前の確認画面の「目立たせ」: 伏せたあとの文のうち、漏れていそうな文を選ぶ.

考え方:

- 特定できるもの（名前・連絡先・金額など）は自動の伏せで漏れ0を目指し、文脈から推し量れる型
  （体調・家族の事情・手元の数）は自動の伏せだけでは取りこぼしうる。
- クラウドに送る前に、人が伏せたあとの文を見る。画面は漏れていそうな文を目立たせる。
  目立たせる文は **1回の打ち合わせで全体の1割以内**。

目立たせは伏せないので、伏せる規則（``mask_r3``）より広く当てる。文ごとに手がかりの点を足し、
点の高い順に、全体の1割（切り上げ）まで選ぶ。点が ``MIN_SCORE`` に届かない文は選ばない。

見るのは伏せたあとの文（クラウドに出す文字列）だけ。元の文は使わない（画面の外に出さない）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from .mask_r3 import _FAMILY_WORD_RE
from .name_candidates import _APPOSITIVE_BAD_RE, _APPOSITIVE_NOT, _READING_NOT
from .stage5_r3_types import MASKED_SENTENCE

CAP_RATIO = 0.1  # 1回の打ち合わせで全体の1割以内
MIN_SCORE = 2

# 一部だけ伏せた家族・体調の札。札の周りに、推し量れる言い方が残りやすい。
_PARTIAL_TAGS_RE = re.compile(r"〈(?:家族の事情|体調の事情)〉")
# 体・受診・休みの言い方（伏せる規則より広い。目立たせるだけ）。
_HEALTH_HINT_RE = re.compile(
    r"体(?!験|制|系|裁|育館)|からだ|身体|調子|具合|痛|疲れ|寝(?:込|られ|れ)|眠|熱(?!気|心|意|量|い|く)|薬|医|病|診|院(?!長)|"
    r"(?<![理教学])科(?!目|学)|"
    r"検(?:査|診)|治|手術|入院|退院|通院|予約[^。]{0,8}(?:午前|午後|朝|時|日|先生)|先生に|休(?:ませ|養|んで|み(?:を|が)?(?:もら|とら|取))|"
    r"しんど|つら|きつ|無理(?:が|は|を)?(?:でき|きか|きかん)|気分|落ち込|メンタル|妊娠|出産|産休"
)
# 身内・暮らしの言い方。
# 「子ども」「学校」は、教室・保育などのお客様では仕事の話に毎文出るので、目立たせの手がかりにしない。
_FAMILY_HINT_RE = re.compile(
    _FAMILY_WORD_RE.pattern.replace("子ども|子供|", "")
    + r"|親族|身内|実家|施設|面会|見舞|介護|送迎|迎え|保証|葬|法事|相続|離婚|別居|同居|受験"
    r"|ばあ(?:ちゃん|さん)|じい(?:ちゃん|さん)|年寄り|ケアマネ|ヘルパー|デイ(?:サービス)?|ショート"
)
# 手元・在庫・残りの数。
_STOCK_HINT_RE = re.compile(
    r"[0-9０-９一二三四五六七八九十百千万]+\s*(?:枚|個|部|冊|本|箱|ケース|セット|点|件|袋|束|キロ|kg)"
    r"[^。]{0,8}(?:あり|ある|あっ|残|余|たま|溜ま|眠|手元|在庫)"
    r"|(?:残り|在庫|手元|手持ち|ストック)[^。]{0,8}[0-9０-９一二三四五六七八九十百千万]"
)
# 数の言い直し（前の文で数を伏せたのに、続く文に伏せていない数が残る）。
_COUNT_RE = re.compile(r"[0-9０-９一二三四五六七八九十百千万]+\s*(?:枚|個|部|冊|本|箱|ケース|セット|点|件|袋|束|キロ|kg)")

# 〈名前〉〈社名〉のすぐ後のかな・カタカナ（読みの言い直し「〈名前〉、たかなし、」）。
_NAME_TAIL_RE = re.compile(r"〈(?:名前|社名)〉(?:です|でございます|と申します|は)?[、,，]\s*([ぁ-ん]{2,8}|[ァ-ヶー]{2,8})(?=[、。,，\s]|です|って|$)")
# 伏せていないカタカナ・英字の固有名詞らしい語（3字以上で、よくある語でないもの）は弱い手がかり。
_PROPER_LIKE_RE = re.compile(r"(?<![ァ-ヶー])([ァ-ヶ][ァ-ヶー]{2,})(?:さん|様|ちゃん|くん|社|店|屋)")
_COMMON_KATAKANA = frozenset(
    {"パート", "アルバイト", "バイト", "スタッフ", "オーナー", "マスター", "シェフ", "ドライバー", "ママ", "パパ",
     "ケアマネ", "ヘルパー", "デザイナー", "カメラマン", "ライター", "メーカー", "ユーザー", "リピーター", "ファン"}
)


_GREETINGS = frozenset(
    {"こんにちは", "こんばんは", "おはよう", "おはようございます", "ありがとう", "ありがとうございます",
     "いらっしゃい", "いらっしゃいませ", "はじめまして", "おつかれさま", "おつかれさまです", "ごめんなさい",
     "すみません", "どうぞ", "ようこそ", "おひさしぶり", "ひさしぶり"}
)


def _is_phrase(word: str) -> bool:
    """名前の読みでなく、語尾・言い回し・つなぎ・あいさつの言葉（名前の候補出しと同じ線）."""
    return (
        word in _GREETINGS
        or word in _APPOSITIVE_NOT
        or word in _READING_NOT
        or _APPOSITIVE_BAD_RE.search(word) is not None
        or word.startswith("お")
    )


@dataclass(frozen=True)
class Flag:
    sent: int  # 1始まり
    score: int
    reasons: tuple[str, ...]
    ok_phrases: tuple[str, ...] = ()  # 「問題なし」で覚える言い回しの候補（名前の手がかりは含めない）


# 「問題なし」で覚える言い回しの長さ（当たった語の前後、句読点まで）。短すぎると別の文まで消える。
OK_PHRASE_MAX = 16
_PUNCT = "、。,，！？!?\n "
# 「問題なし」で消せる手がかり（名前の後のかなは、特定できるものの手がかりなので消せない）。
_SUPPRESSIBLE = (
    (_PARTIAL_TAGS_RE, "家族・体調を一部だけ伏せた"),
    (_FAMILY_HINT_RE, "身内・暮らしの言い方"),
    (_HEALTH_HINT_RE, "体・受診の言い方"),
    (_STOCK_HINT_RE, "手元・残りの数"),
)


def ok_phrase(text: str, start: int, end: int) -> str:
    """当たった語を含む、前後の句読点までの言い回し（最大 ``OK_PHRASE_MAX`` 字）."""
    left = start
    while left > 0 and text[left - 1] not in _PUNCT and end - (left - 1) <= OK_PHRASE_MAX:
        left -= 1
    right = end
    while right < len(text) and text[right] not in _PUNCT and (right + 1) - left <= OK_PHRASE_MAX:
        right += 1
    return text[left:right]


def _ok_spans(text: str, ok: frozenset[str]) -> list[tuple[int, int]]:
    spans = []
    for phrase in ok:
        position = text.find(phrase)
        while phrase and position >= 0:
            spans.append((position, position + len(phrase)))
            position = text.find(phrase, position + 1)
    return spans


def score_sentence(text: str, ok: frozenset[str] = frozenset()) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    """1文の点・理由（画面に出す短い言葉）・「問題なし」で覚える言い回しの候補.

    ``ok`` は覚えた言い回し（全体共通）。当たった語が覚えた言い回しの中にだけあるときは数えない。
    """
    if text == MASKED_SENTENCE:
        return 0, (), ()
    spans = _ok_spans(text, ok)
    score = 0
    reasons: list[str] = []
    phrases: list[str] = []
    for pattern, reason in _SUPPRESSIBLE:
        live = [
            m for m in pattern.finditer(text)
            if not any(a <= m.start() and m.end() <= b for a, b in spans)
        ]
        if live:
            score += 2
            reasons.append(reason)
            phrases.append(ok_phrase(text, live[0].start(), live[0].end()))
    if any(not _is_phrase(m.group(1)) for m in _NAME_TAIL_RE.finditer(text)):
        score += 5  # 特定できるもの（自動の伏せで0にしたもの）は最優先で目立たせる。「問題なし」にはできない
        reasons.append("名前の後のかな（読み）")
    if any(m.group(1) not in _COMMON_KATAKANA for m in _PROPER_LIKE_RE.finditer(text)):
        score += 2
        reasons.append("伏せていない呼び名")
    return score, tuple(reasons), tuple(dict.fromkeys(phrases))


def flag_sentences(
    texts: Sequence[str], *, cap_ratio: float = CAP_RATIO, ok: frozenset[str] = frozenset()
) -> list[Flag]:
    """伏せたあとの文の並びから、目立たせる文を選ぶ（点の高い順、全体の1割まで。文番号の順で返す）."""
    if type(texts) is str:
        raise ValueError("texts は文の並びで指定してください")
    scored = []
    for number, text in enumerate(texts, start=1):
        score, reasons, phrases = score_sentence(str(text), ok)
        previous = str(texts[number - 2]) if number >= 2 else ""
        if "〈数字〉" in previous and _COUNT_RE.search(str(text)):
            score += 3  # 伏せた数をそのまま言い直すのは漏れそのもの（上限があると、この文が切られて漏れが残りうる）
            reasons = reasons + ("前の文で伏せた数の言い直し",)
        if score >= MIN_SCORE:
            scored.append(Flag(number, score, reasons, phrases))
    cap = math.ceil(len(texts) * cap_ratio) if texts else 0
    scored.sort(key=lambda flag: (-flag.score, flag.sent))
    chosen = scored[:cap]
    return sorted(chosen, key=lambda flag: flag.sent)
