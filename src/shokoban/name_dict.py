"""お客様ごとの名前・社名の辞書.

辞書は2つの並びを持つ。

- 伏せる語（``mask``）: ``{"term", "kind"}``。kind は ``人名`` か ``社名・店名・団体``。
- 伏せない語（``pass``）: 名前に見えるが一般の語（人が「伏せない」と決めた語）。

辞書はお客様の情報なので、**手元の保存先だけに置き、git・クラウドに入れない**。このモジュール
は置き場所の既定を持たない（呼ぶ側が必ずパスを渡す）。読み書きは JSON、書き込みは同じ
フォルダの一時ファイルに書いて ``os.replace`` で置き換える（途中で止まっても壊れない）。

import 時に socket / http / subprocess を引き込まない。
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

KIND_PERSON = "人名"
KIND_ORG = "社名・店名・団体"
MASK_KINDS: tuple[str, ...] = (KIND_PERSON, KIND_ORG)

TERM_MIN_CHARS = 1
TERM_MAX_CHARS = 20
DICT_VERSION = 1


class NameDictError(ValueError):
    """辞書の形・中身が正しくないとき."""


@dataclass(frozen=True)
class MaskEntry:
    term: str
    kind: str


def check_term(term: Any) -> str:
    """語の型・長さ・改行を検査し、前後の空白を落とした語を返す（正しくなければ ``NameDictError``）."""
    if type(term) is not str:
        raise NameDictError("語が文字列ではありません")
    value = term.strip()
    if not TERM_MIN_CHARS <= len(value) <= TERM_MAX_CHARS:
        raise NameDictError(f"語の長さは{TERM_MIN_CHARS}〜{TERM_MAX_CHARS}文字にしてください")
    if "\n" in value or "\r" in value:
        raise NameDictError("語に改行は入れられません")
    return value


def check_kind(kind: Any) -> str:
    if type(kind) is not str or kind not in MASK_KINDS:
        raise NameDictError("kind は 人名 か 社名・店名・団体 にしてください")
    return kind


class NameDict:
    """伏せる語と伏せない語。同じ語が両方に入ることはない（後から入れた方が前の方を置き換える）."""

    def __init__(
        self,
        mask: Iterable[MaskEntry] = (),
        pass_terms: Iterable[str] = (),
    ) -> None:
        self._mask: dict[str, str] = {}
        self._pass: dict[str, None] = {}
        for entry in mask:
            if type(entry) is not MaskEntry:
                raise NameDictError("mask の項目の型が正しくありません")
            term = check_term(entry.term)
            if term in self._mask:
                raise NameDictError("伏せる語に同じ語が重なっています")
            self._mask[term] = check_kind(entry.kind)
        for item in pass_terms:
            term = check_term(item)
            if term in self._pass:
                raise NameDictError("伏せない語に同じ語が重なっています")
            self._pass[term] = None
        if set(self._mask) & set(self._pass):
            raise NameDictError("同じ語が、伏せる語と伏せない語の両方に入っています")

    # ---- 読み出し
    @property
    def mask(self) -> tuple[MaskEntry, ...]:
        return tuple(MaskEntry(term, kind) for term, kind in self._mask.items())

    @property
    def pass_terms(self) -> tuple[str, ...]:
        return tuple(self._pass)

    def mask_terms(self) -> tuple[tuple[str, str], ...]:
        """伏せる語を ``(語, kind)`` の並びで返す."""
        return tuple(self._mask.items())

    def knows(self, term: str) -> bool:
        """伏せる語か伏せない語のどちらかに入っている語か."""
        return term in self._mask or term in self._pass

    # ---- 書き換え
    def add_mask(self, term: Any, kind: Any) -> None:
        value, checked = check_term(term), check_kind(kind)
        self._pass.pop(value, None)
        self._mask[value] = checked

    def add_pass(self, term: Any) -> None:
        value = check_term(term)
        self._mask.pop(value, None)
        self._pass[value] = None

    def remove(self, term: Any) -> bool:
        """どちらの並びからも外す（外したら True）."""
        value = check_term(term)
        found = value in self._mask or value in self._pass
        self._mask.pop(value, None)
        self._pass.pop(value, None)
        return found

    # ---- JSON
    def to_json(self) -> dict[str, Any]:
        return {
            "version": DICT_VERSION,
            "mask": [{"term": term, "kind": kind} for term, kind in self._mask.items()],
            "pass": list(self._pass),
        }

    @classmethod
    def from_json(cls, payload: Any) -> "NameDict":
        if type(payload) is not dict:
            raise NameDictError("辞書のルートがオブジェクトではありません")
        if payload.get("version", DICT_VERSION) != DICT_VERSION:
            raise NameDictError("辞書の version が合いません")
        unknown = set(payload) - {"version", "mask", "pass"}
        if unknown:
            raise NameDictError("辞書に知らない項目があります")
        raw_mask = payload.get("mask", [])
        raw_pass = payload.get("pass", [])
        if type(raw_mask) is not list or type(raw_pass) is not list:
            raise NameDictError("mask と pass は配列にしてください")
        entries: list[MaskEntry] = []
        for item in raw_mask:
            if type(item) is not dict or set(item) != {"term", "kind"}:
                raise NameDictError("mask の項目は term と kind だけのオブジェクトにしてください")
            entries.append(MaskEntry(check_term(item["term"]), check_kind(item["kind"])))
        return cls(entries, [check_term(item) for item in raw_pass])


def empty_dict() -> NameDict:
    """1回目の打ち合わせ用の空の辞書."""
    return NameDict()


def load_name_dict(path: Any, *, missing_ok: bool = False) -> NameDict:
    """JSON の辞書を読む。ファイルが無いとき、``missing_ok`` なら空の辞書、でなければ ``FileNotFoundError``."""
    target = Path(path)
    if not target.exists():
        if missing_ok:
            return empty_dict()
        raise FileNotFoundError(f"辞書がありません: {target.name}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NameDictError(f"辞書を読めません: {target.name}") from exc
    return NameDict.from_json(payload)


def save_name_dict(path: Any, name_dict: NameDict) -> Path:
    """辞書を JSON で保存する（0600。同じフォルダの一時ファイル→``os.replace``）.

    置き場所のフォルダは作らない（無ければ ``FileNotFoundError``）。
    """
    if type(name_dict) is not NameDict:
        raise NameDictError("NameDict を渡してください")
    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"辞書を置くフォルダがありません: {target.parent.name}")
    text = json.dumps(name_dict.to_json(), ensure_ascii=False, indent=2) + "\n"
    temporary = target.parent / f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return target


def merge_name_dicts(common: NameDict, customer: NameDict, *, common_pass: bool = True) -> NameDict:
    """全体共通の辞書とお客様ごとの辞書を合わせる（全体共通とお客様ごとの2段）.

    伏せる語は両方の和（同じ語ならお客様ごとの kind）。**どちらかで伏せる語なら伏せる**
    （伏せる側が必ず勝つ）。伏せない語は、お客様ごとの分と、``common_pass`` が真なら共通の分
    （伏せる語に入った語は除く）。
    """
    masks: dict[str, str] = dict(common.mask_terms())
    masks.update(dict(customer.mask_terms()))
    passes = list(customer.pass_terms) + (list(common.pass_terms) if common_pass else [])
    merged_pass = [term for term in dict.fromkeys(passes) if term not in masks]
    return NameDict([MaskEntry(term, kind) for term, kind in masks.items()], merged_pass)
