"""伏せる側（``mask_r3``）と、読む役・組み立て側の間の決まった形.

両方の側はこの形だけを受け渡しに使う。読む役（クラウド）に渡るのは ``reader_payload()`` の
中身だけなので、送る前の確認（``send_review``）や送る中身の指紋も、この全部を見る。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RecordState = Literal["公開", "一部", "伏せた"]

MASKED_SENTENCE = "〈この文は伏せました〉"
# 家族の事情（離婚・介護・相続・家族間の揉め事など）の節の置き換えの札。
MASK_FAMILY = "〈家族の事情〉"
OMITTED = "〈省略〉"
REGION_TAG_EMPTY = "〈伏せた区間〉"
REGION_TAG_PREFIX = "〈伏せた区間：困りごと＝"

TROUBLE_MENU: tuple[str, ...] = (
    "費用の見通しへの不安",
    "人手が足りない",
    "集客",
    "時間が足りない",
    "やり方がわからない",
)
MAX_MENU_PICKS = 2

HINT_NAMES: tuple[str, ...] = ("依頼の型", "引き受けの型", "同意の型", "先送りの型")


@dataclass(frozen=True)
class MaskedRecord:
    sent: int  # 元の文番号（1始まり・連番）
    text: str  # クラウドに出す文字列（伏せた後）。伏せた文は MASKED_SENTENCE
    state: RecordState
    region: int | None = None  # 区間の番号（区間外は None）
    region_reply: bool = False  # 区間内で出した安全な応答型の文（根拠に使えない）
    hint: tuple[str, ...] = ()  # HINT_NAMES のどれか（公開した部分からだけ）


@dataclass(frozen=True)
class Region:
    region: int
    start: int  # 区間の最初の文番号（payload では from）
    end: int  # 区間の最後の文番号（payload では to）
    tag: str  # 〈伏せた区間：困りごと＝…〉または〈伏せた区間〉


@dataclass(frozen=True)
class MaskLogEntry:
    sent: int
    kinds: tuple[str, ...]  # 種類の名前だけ（元の語・手がかりの語は書かない）
    route: str  # 経路名
    risk_p: float | None = None  # 判定用モデルの p（判定器 off は None。文字列は残さない）


@dataclass(frozen=True)
class TaggerInfo:
    model: str  # 実際に使った札付け役（"dry" や "gemma-4-12b-ja"）
    ok: int = 0
    fail: int = 0


@dataclass(frozen=True)
class MaskedR3:
    records: tuple[MaskedRecord, ...]
    regions: tuple[Region, ...] = ()
    mask_log: tuple[MaskLogEntry, ...] = ()
    tagger: TaggerInfo = field(default_factory=lambda: TaggerInfo(model="dry"))

    def reader_payload(self) -> dict[str, Any]:
        """読む役（クラウド）に渡す全部。これ以外は渡さない."""
        return {
            "records": [
                {
                    "sent": r.sent,
                    "text": r.text,
                    "hint": list(r.hint),
                    "region": r.region,
                    "region_reply": r.region_reply,
                }
                for r in self.records
            ],
            "regions": [
                {"region": g.region, "from": g.start, "to": g.end, "tag": g.tag}
                for g in self.regions
            ],
        }

    def to_json(self) -> dict[str, Any]:
        """辞書の形にする（読む役に渡す全部に、文ごとの ``state``・``mask_log``・``tagger`` を足したもの）."""
        payload = self.reader_payload()
        for out, r in zip(payload["records"], self.records):
            out["state"] = r.state
        payload["mask_log"] = [
            {"sent": e.sent, "kinds": list(e.kinds), "route": e.route}
            | ({} if e.risk_p is None else {"risk_p": e.risk_p})
            for e in self.mask_log
        ]
        payload["tagger"] = {
            "model": self.tagger.model,
            "ok": self.tagger.ok,
            "fail": self.tagger.fail,
        }
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "MaskedR3":
        """``to_json`` の形から戻す."""
        records = tuple(
            MaskedRecord(
                sent=int(r["sent"]),
                text=str(r["text"]),
                state=r.get("state", "公開"),
                region=r.get("region"),
                region_reply=bool(r.get("region_reply", False)),
                hint=tuple(r.get("hint", ())),
            )
            for r in payload["records"]
        )
        regions = tuple(
            Region(region=int(g["region"]), start=int(g["from"]), end=int(g["to"]), tag=str(g["tag"]))
            for g in payload.get("regions", [])
        )
        mask_log = tuple(
            MaskLogEntry(
                sent=int(e["sent"]),
                kinds=tuple(e.get("kinds", ())),
                route=str(e.get("route", "")),
                risk_p=e.get("risk_p"),
            )
            for e in payload.get("mask_log", [])
        )
        tag = payload.get("tagger") or {}
        tagger = TaggerInfo(
            model=str(tag.get("model", "dry")),
            ok=int(tag.get("ok", 0)),
            fail=int(tag.get("fail", 0)),
        )
        return cls(records=records, regions=regions, mask_log=mask_log, tagger=tagger)


def region_tag(picks: tuple[str, ...] | list[str]) -> str:
    """品書きの語だけで札を組む。品書き外・3個以上・重複は札なし（〈伏せた区間〉）."""
    items = list(picks)
    if (
        not items
        or len(items) > MAX_MENU_PICKS
        or len(set(items)) != len(items)
        or any(item not in TROUBLE_MENU for item in items)
    ):
        return REGION_TAG_EMPTY
    return REGION_TAG_PREFIX + "、".join(items) + "〉"
