"""読む役・組み立て・runner の試験で使う、偽の ``MaskedR3`` と返事の作り方.

``mask_r3``（伏せる部品）は使わない。受け渡しの形 ``stage5_r3_types.MaskedR3`` を
そのまま偽の値で作る。文はすべて架空の短い文。
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from shokoban.reader_stage5 import ReaderReply
from shokoban.stage5_r3_types import (
    MASKED_SENTENCE,
    MaskedR3,
    MaskedRecord,
    MaskLogEntry,
    Region,
    TaggerInfo,
)

# 10文の架空の打ち合わせ（番号は1始まり）。
BASE: tuple[str, ...] = (
    "来月のイベント用にチラシをお願いできますか。",  # 1 お客様 -> 支援側 の依頼
    "はい、チラシは私が作ります。",  # 2 支援側の引き受け
    "写真は私のほうで用意しますね。",  # 3 お客様の申し出
    "助かります、それでお願いします。",  # 4 支援側の同意
    "印刷は印刷会社に頼もうと思ってます。",  # 5 第三者への依頼
    "ホームページの更新は来月にしましょう。",  # 6 提案
    "そうですね、一旦保留にしておきましょう。",  # 7 保留
    "人手が足りないのが悩みです。",  # 8 困りごと
    "やっぱりチラシの話はなしにしてください。",  # 9 撤回
    "分かりました。",  # 10
)


def make_masked(
    sentences: Sequence[str] = BASE,
    *,
    hidden: Sequence[int] = (),
    replies: Sequence[int] = (),
    regions: Sequence[tuple[int, int, int, str]] = (),
    tagger_model: str = "dry",
) -> MaskedR3:
    """文の並びから偽の ``MaskedR3`` を作る.

    ``hidden``: 伏せた文の番号／``replies``: 区間内の安全な応答型（根拠に使えない）の番号／
    ``regions``: ``(区間番号, from, to, 札)``。
    """
    region_of: dict[int, int] = {}
    for number, start, end, _ in regions:
        for sent in range(start, end + 1):
            region_of[sent] = number
    records = []
    for sent, text in enumerate(sentences, start=1):
        is_hidden = sent in hidden
        records.append(
            MaskedRecord(
                sent=sent,
                text=MASKED_SENTENCE if is_hidden else text,
                state="伏せた" if is_hidden else "公開",
                region=region_of.get(sent),
                region_reply=sent in replies,
            )
        )
    return MaskedR3(
        records=tuple(records),
        regions=tuple(Region(n, a, b, tag) for n, a, b, tag in regions),
        mask_log=(MaskLogEntry(sent=1, kinds=("内情",), route="区間外"),),
        tagger=TaggerInfo(model=tagger_model, ok=len(regions), fail=0),
    )


def ev(
    etype: str,
    speaker: str,
    sent: int,
    span: str,
    to: str | None = "不明",
) -> dict[str, Any]:
    """根拠1件。``to=None`` なら ``to`` を書かない."""
    row: dict[str, Any] = {"type": etype, "speaker": speaker, "sent": sent, "span": span}
    if to is not None:
        row["to"] = to
    return row


def work(
    sent: int,
    span: str,
    *,
    who: str = "支援側",
    state: str = "引き受けた",
    requested: bool = False,
    evidence: Sequence[dict[str, Any]] = (),
    topic: int = 1,
    rank: int = 1,
) -> dict[str, Any]:
    return {
        "kind": "作業",
        "sent": sent,
        "span": span,
        "who": who,
        "state": state,
        "requested": requested,
        "evidence": list(evidence),
        "topic": topic,
        "rank": rank,
    }


def decision(
    sent: int,
    span: str,
    *,
    state: str = "合意",
    evidence: Sequence[dict[str, Any]] = (),
    topic: int = 1,
    rank: int = 1,
) -> dict[str, Any]:
    return {
        "kind": "決めごと",
        "sent": sent,
        "span": span,
        "state": state,
        "evidence": list(evidence),
        "topic": topic,
        "rank": rank,
    }


def plain(kind: str, sent: int, span: str, *, topic: int = 1, rank: int = 1) -> dict[str, Any]:
    """困りごと・未決（who・state・根拠を持たない）."""
    return {"kind": kind, "sent": sent, "span": span, "topic": topic, "rank": rank}


def reply(
    *items: dict[str, Any],
    speakers: Sequence[dict[str, Any]] | None = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {"items": list(items)}
    if speakers is not None:
        payload["speakers"] = list(speakers)
    return payload


def reader_reply(*items: dict[str, Any], model: str = "fake", **kwargs: Any) -> ReaderReply:
    return ReaderReply(
        text=json.dumps(reply(*items, **kwargs), ensure_ascii=False), model=model
    )
