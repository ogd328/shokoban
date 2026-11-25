"""伏せる処理（``mask_r3``）と5欄の組み立て（``assemble_r3``）が使う、手元の部品.

このモジュールが持っているもの:

- 伏せた跡の札（〈社名〉〈名前〉〈連絡先〉〈金額〉〈数字〉〈体調の事情〉）と、
  伏せた種類の名前（``KIND_*``）。
- 金額の幅の札: ``parse_money_value`` が「15万」「19万9千」のような金額を数に読み、
  ``money_band_label`` が ``MONEY_BANDS`` の表で幅の札（〈金額：10万円台〉など）にする。
- 節の分け方: ``split_clauses`` が文を読点と「ので・から・けど・けれど」で節に分ける。
- 残った数の検査: ``leftover_types`` が、置き換えたあとの文に社名・名前・連絡先・
  3桁以上の算用数字が残っていないかを調べる。
- 日付の言い換え: ``relative_date_suffixes`` / ``add_date_suffixes`` が、「明日」
  「来週火曜」などの相対の日付に、打ち合わせの日からの実日付（「（11月17日）」）を添える。
  年は書かない。

``mask_r3`` が伏せる処理の中で、``assemble_r3`` が5欄の行に実日付を添えるときに使う。
この部品は socket / http / urllib / requests / subprocess を import しない
（読む役の起動は ``reader_stage5`` 側）。
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta
from typing import Sequence

from .pick_bench import _EMAIL_RE, _NAME_RE, _NUMBER_RE, _PHONE_RE, _QUANTITY_UNITS

# ---------------------------------------------------------------- 札と種類の名前

MASK_COMPANY = "〈社名〉"
MASK_NAME = "〈名前〉"
MASK_CONTACT = "〈連絡先〉"
MASK_MONEY = "〈金額〉"
MASK_NUMBER = "〈数字〉"
MASK_HEALTH = "〈体調の事情〉"

# 金額の幅の札。各行は「この金額未満ならこの札」で、小さい幅から順に見る。
MONEY_BANDS: tuple[tuple[int | None, str], ...] = (
    (10_000, "〈金額：1万円未満〉"),
    (100_000, "〈金額：数万円〉"),
    (200_000, "〈金額：10万円台〉"),
    (500_000, "〈金額：20〜40万円台〉"),
    (1_000_000, "〈金額：50〜90万円台〉"),
    (None, "〈金額：百万円以上〉"),
)

# 伏せた種類の名前（mask_log にはこの名前だけを書く。元の語は書かない）。
KIND_COMPANY = "社名"
KIND_NAME = "名前"
KIND_CONTACT = "連絡先"
KIND_MONEY = "金額"
KIND_NUMBER = "数字"
KIND_HEALTH = "体調"
KIND_NET = "網"


# 節の切れ目になる語（直後に来たら切る）。「が」「て」「し」では切らない。
_CLAUSE_ENDERS: tuple[str, ...] = ("けれども", "けれど", "けど", "ので", "から")
_CLAUSE_SEPARATOR = "、"


# ---------------------------------------------------------------- 金額

_KANJI_DIGITS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_BIG_UNITS = {"万": 10_000, "億": 100_000_000}


def parse_money_value(body: str) -> int | None:
    """金額の数字の部分を数として読む（読めなければ ``None``）.

    「15万」「19万9千」「三百万」「1,500万」のような形を読む。「数十万」
    「何万」のように数として読めない形は ``None``（幅なしの〈金額〉）。
    """
    if type(body) is not str or not body.strip():
        return None
    text = unicodedata.normalize("NFKC", body)
    total = 0
    section = 0
    current = 0
    seen = False
    index = 0
    while index < len(text):
        char = text[index]
        if char in " ,":
            index += 1
            continue
        if char.isdigit():
            end = index
            while end < len(text) and text[end].isdigit():
                end += 1
            current = current * (10 ** (end - index)) + int(text[index:end])
            seen = True
            index = end
            continue
        if char in _SMALL_UNITS:
            section += (current if current else 1) * _SMALL_UNITS[char]
            current = 0
            seen = True
        elif char in _BIG_UNITS:
            value = section + current
            if value == 0:
                value = 1
            total += value * _BIG_UNITS[char]
            section = 0
            current = 0
            seen = True
        elif char in _KANJI_DIGITS:
            current = _KANJI_DIGITS[char]
            seen = True
        else:  # 「数」「何」など、数として読めない形
            return None
        index += 1
    if not seen:
        return None
    return total + section + current


def money_band_label(value: int | None) -> str:
    """金額を、``MONEY_BANDS`` の表の札に変える（読めなければ幅なしの〈金額〉）."""
    if value is None:
        return MASK_MONEY
    for upper, label in MONEY_BANDS:
        if upper is None or value < upper:
            return label
    return MASK_MONEY  # pragma: no cover - 表の最後は None なので到達しない


# ---------------------------------------------------------------- 数字・連絡先


_URL_RE = re.compile(
    r"(?:https?|ftp)://[^\s、。」』）\)]+"
    r"|www\.[A-Za-z0-9][A-Za-z0-9.\-]*"
)
# SNS の宛先（@example）は、宛先ごと伏せる。
_HANDLE_RE = re.compile(r"@[A-Za-z0-9_.\-]+")


# ---------------------------------------------------------------- 節の分け方

def _ending_at(text: str, index: int) -> int | None:
    for ender in _CLAUSE_ENDERS:
        if text.startswith(ender, index):
            return index + len(ender)
    return None


def split_clauses(text: str) -> list[tuple[int, int, str]]:
    """節に分け、``(始まり, 終わり, 節の文字)`` を返す.

    区切りは読点「、」と、直後に来る「ので・から・けど・けれど」。
    「が」「て」「し」では切らない（「私が作る」「作って送る」を壊さないため）。
    """
    if type(text) is not str:
        raise ValueError("text は文字列で指定してください")
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == _CLAUSE_SEPARATOR:
            spans.append((start, index))
            start = index + 1
            index += 1
            continue
        end = _ending_at(text, index)
        if end is not None:
            spans.append((start, end))
            start = end
            index = end
            continue
        index += 1
    spans.append((start, len(text)))

    clauses: list[tuple[int, int, str]] = []
    for begin, end in spans:
        raw = text[begin:end]
        clause = raw.strip()
        if not clause:
            continue
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw) - len(raw.rstrip())
        clauses.append((begin + lead, end - trail, clause))
    return clauses


# ---------------------------------------------------------------- 残った数・型の検査

def _has_leftover_number(text: str) -> bool:
    for match in _NUMBER_RE.finditer(text):
        if text.startswith(_QUANTITY_UNITS, match.end()):
            continue
        return True
    return False


def leftover_types(
    text: str, *, company_name: str = "", aliases: Sequence[str] = ()
) -> tuple[str, ...]:
    """置き換え漏れの型（社名・名前・連絡先・3桁以上の算用数字）を返す.

    ``mask_r3`` が、置き換えたあとの文にまだ残っている型を調べるのに使う
    （残っていればその文を丸ごと伏せ、種類は ``網`` と記録する）。作業の量の
    単位（ページ・部・枚・回・本・投稿）が直後に付く数字は残してよいので、漏れに数えない。
    """
    found: list[str] = []
    terms = [term for term in (company_name, *aliases) if type(term) is str and term]
    if any(term in text for term in terms):
        found.append(KIND_COMPANY)
    if _NAME_RE.search(text):
        found.append(KIND_NAME)
    if (
        _EMAIL_RE.search(text)
        or _PHONE_RE.search(text)
        or _URL_RE.search(text)
        or "@" in text
    ):
        found.append(KIND_CONTACT)
    if _has_leftover_number(text):
        found.append(KIND_NUMBER)
    return tuple(found)


# ---------------------------------------------------------------- 相対の日付

_WEEKDAY_INDEX = {"月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6}
_RELATIVE_DATE_RE = re.compile(
    r"明後日|明日|(再来週|来週|今週)([月火水木金土日])曜(?:日)?|来月|月末"
)


def _day_suffix(day: date, meeting_year: int) -> str:
    prefix = "翌年" if day.year != meeting_year else ""
    return f"（{prefix}{day.month}月{day.day}日）"


def _month_suffix(year: int, month: int, meeting_year: int) -> str:
    prefix = "翌年" if year != meeting_year else ""
    return f"（{prefix}{month}月）"


def relative_date_suffixes(text: str, meeting_date: date) -> tuple[str, ...]:
    """相対の日付を「（11月17日）」「（12月）」の形に変える（年を書かない）.

    打ち合わせの年と計算した年が違うときだけ「（翌年1月17日）」「（翌年1月）」。
    3桁以上の算用数字を出さないため、年は書かない。
    """
    if type(text) is not str:
        raise ValueError("text は文字列で指定してください")
    if not isinstance(meeting_date, date):
        raise ValueError("meeting_date は日付で指定してください")

    monday = meeting_date - timedelta(days=meeting_date.weekday())
    suffixes: list[str] = []
    for match in _RELATIVE_DATE_RE.finditer(text):
        token = match.group(0)
        if token == "明日":
            suffix = _day_suffix(meeting_date + timedelta(days=1), meeting_date.year)
        elif token == "明後日":
            suffix = _day_suffix(meeting_date + timedelta(days=2), meeting_date.year)
        elif match.group(2):
            offset = {"今週": 0, "来週": 7, "再来週": 14}[match.group(1)]
            day = monday + timedelta(days=offset + _WEEKDAY_INDEX[match.group(2)])
            suffix = _day_suffix(day, meeting_date.year)
        elif token == "来月":
            year, month = (
                (meeting_date.year + 1, 1)
                if meeting_date.month == 12
                else (meeting_date.year, meeting_date.month + 1)
            )
            suffix = _month_suffix(year, month, meeting_date.year)
        else:  # 月末
            suffix = _month_suffix(meeting_date.year, meeting_date.month, meeting_date.year)
        if suffix not in suffixes:
            suffixes.append(suffix)
    return tuple(suffixes)


def add_date_suffixes(text: str, meeting_date: date) -> str:
    """相対の日付の実日付を、行末に足した文字列を返す."""
    return text + "".join(relative_date_suffixes(text, meeting_date))
