"""送る前の確認: 伏せたあとの全文を人が見て、確認済みにする.

考え方:

- 特定できるもの（名前・連絡先・金額など）は自動の伏せで漏れ0を目指し、文脈から推し量れる型は
  自動の伏せだけでは取りこぼしうる。
- **クラウドに送る前に、人が伏せたあとの文を見る**。画面は漏れていそうな文を目立たせる
  （``review_flags``。1回の打ち合わせで全体の1割以内）。

このモジュールは Streamlit に依存しない（画面は ``gui/shokoban_mac.py`` の「送る前の確認」）。

- 伏せ方は送るときと同じ: 名前の辞書の伏せる語＋まだ確認されていない候補（``terms_to_mask``）を
  ``mask_r3(extra_terms=...)`` に足し、区間の札付けは手元の12B（使えないときは偽物＝区間はすべて
  伏せる。漏れの側に倒さない）。
- 人は文ごとに「さらに伏せる」を選べる（その文は〈この文は伏せました〉にする）。
- 「確認済み」で、送る文字列の sha256 を ``<保存先>/send-approvals/<customer_id>/<record_id>.json``
  （0600）に書く。本文・名前は書かない（文番号・件数・指紋だけ）。送る側は ``check_approval`` で、
  送る文字列が確認したものと同じときだけ送る。

架空データ（``synthetic``）では常に使える。本番データ（``real``）では ``SHOKOBAN_REVIEW_REAL=1``
のときだけ開く（名前の確認の画面と同じ。それ以外は何も読まない）。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .mac_app import MacAppError, MacAppPaths, _check_customer_id, _ensure_private_dir, store_root_for_mode
from .mask_r3 import dry_tagger, mask_r3
from .name_candidates import Candidate, dict_terms_to_mask, terms_to_mask
from .name_dict import NameDict
from .name_review import NameReviewError, require_review_mode
from .review_flags import flag_sentences
from .stage5_r3_types import MASKED_SENTENCE, MaskedR3

APPROVAL_DIR = "send-approvals"
# 「問題なし」で覚えた言い回し（全体共通。目立たせにだけ効き、伏せ方は変えない）。
OK_DIR = "review-ok"
OK_FILE = "common.json"
OK_VERSION = 1
APPROVAL_VERSION = 1


class SendReviewError(RuntimeError):
    """送る前の確認ができない（画面がこの文言を出して止まる）."""


# 短い相づち（「はい。」「なるほど。」）は、画面で初めから畳む。
# 伏せた札（〈…〉）がある文・目立たせた文は畳まない。
SHORT_MAX_CHARS = 12


def is_short_filler(text: str, flagged: bool) -> bool:
    return not flagged and "〈" not in text and len(text.strip()) <= SHORT_MAX_CHARS


@dataclass(frozen=True)
class SendRow:
    sent: int  # 1始まり
    text: str  # 伏せたあとの文（送る文字列）
    flagged: bool
    score: int = 0
    reasons: tuple[str, ...] = ()
    ok_phrases: tuple[str, ...] = ()  # 「問題なし」で覚える言い回しの候補

    @property
    def short(self) -> bool:
        """画面で畳む短い相づちか."""
        return is_short_filler(self.text, self.flagged)


@dataclass(frozen=True)
class SendReview:
    rows: tuple[SendRow, ...]
    tagger_model: str
    tagger_note: str | None = None  # 札付けを使えなかったとき（区間はすべて伏せた）
    masked: MaskedR3 | None = None  # この画面の元の伏せた結果（クラウドに送る中身の元）

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def flagged(self) -> int:
        return sum(1 for row in self.rows if row.flagged)

    @property
    def short(self) -> int:
        return sum(1 for row in self.rows if row.short)


def prepare_send_review(
    sentences: Sequence[str],
    *,
    name_dict: NameDict | None,
    candidates: Sequence[Candidate],
    company_name: str = "",
    aliases: Sequence[str] = (),
    tagger: Callable[[str], Sequence[str]] = dry_tagger,
    tagger_model: str = "dry",
    tagger_note: str | None = None,
    ok_phrases: Iterable[str] = (),
) -> SendReview:
    """送るときと同じ伏せ方で伏せ、目立たせる文を選ぶ（``ok_phrases`` は「問題なし」で覚えた言い回し）."""
    if type(sentences) is str:
        raise SendReviewError("文は並びで渡してください。")
    texts = [str(item) for item in sentences]
    if not texts:
        raise SendReviewError("文がありません。")
    result = mask_r3(
        texts,
        tagger=tagger,
        tagger_model=tagger_model,
        company_name=company_name,
        aliases=aliases,
        extra_terms=terms_to_mask(name_dict, candidates),
        dict_terms=dict_terms_to_mask(name_dict),
    )
    records = sorted(result.records, key=lambda record: record.sent)
    masked = [record.text for record in records]
    flags = {flag.sent: flag for flag in flag_sentences(masked, ok=frozenset(ok_phrases))}
    rows = tuple(
        SendRow(
            sent=record.sent,
            text=record.text,
            flagged=record.sent in flags,
            score=flags[record.sent].score if record.sent in flags else 0,
            reasons=flags[record.sent].reasons if record.sent in flags else (),
            ok_phrases=flags[record.sent].ok_phrases if record.sent in flags else (),
        )
        for record in records
    )
    return SendReview(
        rows=rows,
        tagger_model=tagger_model,
        tagger_note=tagger_note,
        masked=replace(result, records=tuple(records)),
    )


def final_texts(review: SendReview, hide: Iterable[int] = ()) -> list[str]:
    """人が「さらに伏せる」を選んだ文を〈この文は伏せました〉にした、送る文字列の並び."""
    hidden = {int(number) for number in hide}
    unknown = hidden - {row.sent for row in review.rows}
    if unknown:
        raise SendReviewError("ない文番号が選ばれています。")
    return [MASKED_SENTENCE if row.sent in hidden else row.text for row in review.rows]


def final_masked(review: SendReview, hide: Iterable[int] = ()) -> MaskedR3:
    """人が「さらに伏せる」を選んだ文を伏せた、クラウドに送る中身全体（``MaskedR3``）.

    伏せた文は ``final_texts`` と同じ〈この文は伏せました〉にし、その文の元の文に由来する
    手がかりを ``reader_payload()`` に残さない: ``hint``（公開した部分から付いた型）を空に、
    ``region_reply``（応答型の文という印）を偽に、``state`` を ``伏せた`` にする。
    区間の番号と区間の札（品書きの語だけ）は文の中身ではないのでそのまま。伏せていない文は変えない。
    """
    if review.masked is None:
        raise SendReviewError("送る中身が用意されていません。もう一度「伏せて確かめる」を押してください。")
    hidden = {int(number) for number in hide}
    unknown = hidden - {row.sent for row in review.rows}
    if unknown:
        raise SendReviewError("ない文番号が選ばれています。")
    records = tuple(
        replace(record, text=MASKED_SENTENCE, state="伏せた", hint=(), region_reply=False)
        if record.sent in hidden
        else record
        for record in review.masked.records
    )
    return replace(review.masked, records=records)


def payload_digest(masked: MaskedR3) -> str:
    """クラウドに送る中身全体（``reader_payload()``）の指紋（キー順を固定した JSON の sha256）."""
    payload = json.dumps(masked.reader_payload(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def texts_digest(texts: Sequence[str]) -> str:
    """送る文字列の並びの指紋（文の区切りも含めて比べる）."""
    if type(texts) is str:
        raise SendReviewError("文は並びで渡してください。")
    payload = json.dumps([str(text) for text in texts], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_record_id(record_id: str) -> str:
    value = str(record_id).strip()
    if not value or "/" in value or "\\" in value or value in {".", ".."} or value.startswith("."):
        raise SendReviewError("打ち合わせの番号が正しくありません。")
    return value


def approval_path(paths: MacAppPaths, mode: str, customer_id: str, record_id: str) -> Path:
    """確認済みの記録の場所（保存先はデータモードで決まる）."""
    try:
        require_review_mode(mode)
        root = store_root_for_mode(paths, mode)
    except (NameReviewError, MacAppError) as exc:
        raise SendReviewError(str(exc) or "保存先を決められません。") from exc
    if not _check_customer_id(customer_id):
        raise SendReviewError("お客様の識別子が不正です。")
    return root / APPROVAL_DIR / customer_id / f"{_check_record_id(record_id)}.json"


def save_approval(
    path: Path,
    review: SendReview,
    hide: Iterable[int] = (),
    *,
    reviewed_at: datetime,
    read_all: bool = False,
) -> dict[str, Any]:
    """確認済みを書く（0600）。本文・名前は書かない.

    ``read_all`` は、画面で「伏せたあとの全文を読んだ」を付けたこと。偽なら書かずに断る。
    送る中身全体の指紋（``payload_sha256``）も書く。
    """
    if read_all is not True:
        raise SendReviewError("伏せたあとの全文を読んだ確認が必要です。")
    hidden = sorted({int(number) for number in hide})
    texts = final_texts(review, hidden)
    payload_sha256 = payload_digest(final_masked(review, hidden))
    record = {
        "version": APPROVAL_VERSION,
        "reviewed_at": reviewed_at.isoformat(),
        "sha256": texts_digest(texts),
        "payload_sha256": payload_sha256,
        "read_all": True,
        "sentences": len(texts),
        "flagged": [row.sent for row in review.rows if row.flagged],
        "hidden_by_person": hidden,
        "tagger_model": review.tagger_model,
    }
    target = Path(path)
    for directory in (target.parent.parent, target.parent):  # send-approvals/ と お客様のフォルダ
        _ensure_private_dir(directory)
    tmp = target.with_name(f".{target.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, target)
    os.chmod(target, 0o600)
    return record


def load_approval(path: Path) -> Mapping[str, Any] | None:
    target = Path(path)
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SendReviewError("確認済みの記録を読めません。") from exc
    if type(data) is not dict or data.get("version") != APPROVAL_VERSION or type(data.get("sha256")) is not str:
        raise SendReviewError("確認済みの記録の形が正しくありません。")
    return data


def check_approval(path: Path, texts: Sequence[str]) -> bool:
    """送る文字列が、人が確認したものと同じなら真（送る側の関所）."""
    data = load_approval(path)
    return data is not None and data["sha256"] == texts_digest(texts)


def check_payload_approval(path: Path, masked: MaskedR3) -> bool:
    """クラウドに送る中身全体が、全文を読んで確認したものと同じなら真（送る側の関所）."""
    data = load_approval(path)
    return (
        data is not None
        and data.get("read_all") is True
        and data.get("payload_sha256") == payload_digest(masked)
    )


# ---------------------------------------------------------------- 「問題なし」で覚えた言い回し（全体共通）

def ok_phrases_path(paths: MacAppPaths, mode: str) -> Path:
    try:
        require_review_mode(mode)
        root = store_root_for_mode(paths, mode)
    except (NameReviewError, MacAppError) as exc:
        raise SendReviewError(str(exc) or "保存先を決められません。") from exc
    return root / OK_DIR / OK_FILE


def load_ok_phrases(path: Path) -> frozenset[str]:
    target = Path(path)
    if not target.is_file():
        return frozenset()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SendReviewError("「問題なし」の記録を読めません。") from exc
    phrases = data.get("phrases") if type(data) is dict else None
    if data.get("version") != OK_VERSION or type(phrases) is not list or any(type(p) is not str or not p for p in phrases):
        raise SendReviewError("「問題なし」の記録の形が正しくありません。")
    return frozenset(phrases)


def add_ok_phrases(path: Path, phrases: Iterable[str]) -> int:
    """言い回しを足して書く（0600）。足した数を返す。名前・社名の札だけの言い回しは足さない."""
    new = [str(p) for p in phrases if str(p).strip() and len(str(p)) >= 3]
    if not new:
        return 0
    target = Path(path)
    current = sorted(load_ok_phrases(target))
    merged = list(dict.fromkeys(current + new))
    added = len(merged) - len(current)
    _ensure_private_dir(target.parent)
    tmp = target.with_name(f".{target.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"version": OK_VERSION, "phrases": merged}, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, target)
    os.chmod(target, 0o600)
    return added


# ---------------------------------------------------------------- 「さらに伏せる」の語を辞書へ

@dataclass(frozen=True)
class HideWord:
    sent: int
    term: str
    kind: str  # 人名 / 社名・店名・団体
    common: bool = False  # 全体共通の辞書へ


def check_hide_words(review: SendReview, words: Iterable[HideWord]) -> list[HideWord]:
    """入れる語を確かめる: その文に表示されている語だけ・札を含まない・辞書の語の形（1〜20字）."""
    from .name_dict import NameDictError, check_kind, check_term

    texts = {row.sent: row.text for row in review.rows}
    checked: list[HideWord] = []
    for word in words:
        try:
            term = check_term(word.term)
            kind = check_kind(word.kind)
        except NameDictError as exc:
            raise SendReviewError("辞書に入れる語が正しくありません。") from exc
        if "〈" in term or "〉" in term:
            raise SendReviewError("札（〈…〉）は辞書に入れられません。")
        if word.sent not in texts or term not in texts[word.sent]:
            raise SendReviewError("辞書に入れる語は、その文に出ている語にしてください。")
        checked.append(HideWord(word.sent, term, kind, bool(word.common)))
    return checked


def save_hide_words(dict_path: Path, common_path: Path, words: Sequence[HideWord]) -> tuple[int, int]:
    """確かめた語を、伏せる語として辞書に書く（お客様ごと・全体共通）。``(お客様ごと, 共通)`` の数."""
    from .name_review import apply_review

    own = {w.term: w.kind for w in words if not w.common}
    common = {w.term: w.kind for w in words if w.common}
    try:
        a = apply_review(dict_path, own).masked if own else 0
        b = apply_review(common_path, common).masked if common else 0
    except NameReviewError as exc:
        raise SendReviewError(str(exc)) from exc
    return a, b
