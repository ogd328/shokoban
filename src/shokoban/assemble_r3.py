"""読む役の返事の点検と、5欄の組み立て（手元・AIなし）.

流れは:

- 形の点検: 読む役の返事の項目ごとに、型・必須・許す値・``sent`` と ``span`` を
  調べる。違えば ``形崩れ`` などで落とし、``dropped`` と ``counts`` に残す。
- 根拠の点検: 根拠を1件ずつ調べ、無効な根拠を除いた残りで欄を決める。
  有効な ``撤回`` があれば、欄の表より先に ``まだ決まっていないこと`` へ振り分ける。
- 欄の表: 上から順に、最初に当たった行で欄を決める（読む役は欄を選ばない）。
- 行数: こちらがやること・決まったこと・頼まれていることは削らない。
  困りごと・未決の kind の行だけ ``rank`` 順に各欄8行まで（困りごとは
  ``topic`` ごとの最良の1行を先に入れる）。
- 作業・決めごとの任意の ``detail``（対象の引用。3個まで）を点検し、
  だめなものだけ落として記録する。
- お客様の弱い同意（「かも」「できそう」「助か」）と、頼む言葉の無い質問の形の
  依頼・申し出は根拠にしない。作業の任意の ``name``（短い作業名）を点検して表示に使う
  （欄の振り分けには使わない）。

``pick_bench`` にある機械の決まり（合意なし・依頼なし・先送りの語）は欄の決定には当てない。
同じ判定を ``old_rules`` に記録するだけ（``pick_bench._route_rows`` をそのまま使う）。

入口は ``assemble_r3()``。読む役の返事を渡すのは ``cloud_memo``、画面（``gui``）は欄の
名前と話し手の印の関数を使う。この module は socket / http / subprocess を import しない
（読む役の起動は ``reader_stage5``）。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Sequence

from .mask_stage5 import add_date_suffixes
from .pick_bench import (
    DEFERRAL_MARK,
    FIELDS,
    MAX_ROWS_PER_FIELD,
    MAX_SPAN_CHARS,
    REASON_NOT_PART,
    REASON_NUMBER,
    REASON_ROWS,
    REASON_SHAPE,
    REASON_TOO_LONG,
    _Row,
    _route_rows,
)
from .stage5_r3_types import REGION_TAG_EMPTY, MaskedR3

# ---------------------------------------------------------------- 値と理由

KIND_PATH = "道筋"  # 支援側（こちら）が話した今後の進め方・段階・最終の姿・提案
KINDS: tuple[str, ...] = ("作業", "決めごと", "困りごと", "未決", KIND_PATH)
WHO_VALUES: tuple[str, ...] = ("支援側", "お客様", "不明")
WORK_STATES: tuple[str, ...] = ("引き受けた", "頼まれただけ", "提案中", "保留")
DECISION_STATES: tuple[str, ...] = ("合意", "片方の提案", "保留")
EVIDENCE_TYPES: tuple[str, ...] = ("依頼", "引き受け", "申し出", "同意", "保留", "撤回")
SPEAKER_VALUES: tuple[str, ...] = ("お客様", "支援側", "不明")
TO_VALUES: tuple[str, ...] = ("お客様", "支援側", "第三者", "不明")
ASK_TYPES = ("依頼", "申し出")
ACCEPT_TYPES = ("引き受け", "同意")
EVIDENCE_WINDOW = 5  # 内容の文から前後5文以内（探す範囲の制限）
# 作業で、お客様から支援側への依頼の根拠だけは前後5文の外でもよい（頼みが
# 打ち合わせの前半、引き受けが後半にあることが多い）。遠い依頼は「頼まれていること」
# の判定（ask_to_ours）にだけ使い、双方の合意・申し出への返事には使わない。

# 欄の名前（pick_bench.FIELDS と同じ並び）。
F_TROUBLE, F_ASKED, F_OURS, F_DECIDED, F_UNDECIDED = FIELDS
# 道筋の欄は5欄（FIELDS）に入れない（数え方が変わるため）。``columns`` と
# ``shared``（いちばん上）にだけ足す。``items``・``counts`` の意味は変えない。
F_PATH = "こちらから話した道筋・提案"
SHARED_FIELDS: tuple[str, ...] = (F_PATH, *FIELDS)
SPEAKER_SELF = "支援側"
SPEAKER_SELF_LABEL = "こちら"  # 表示では「支援側」を「こちら」と言い換える
SPEAKER_UNKNOWN = "不明"

REASON_HIDDEN = "伏せた文"
REASON_TAG_ONLY = "札だけ"
REASON_SPEAKER_SHAPE = "話し手の形"
REASON_SPEAKER_CONFLICT = "話し手と矛盾"
REASON_REGION_REPLY = "応答型"
REASON_TOO_FAR = "範囲外"
REASON_NO_ASK_BASIS = "依頼の根拠なし"
REASON_TOO_MANY = "多すぎ"
# detail（対象の引用）。evidence の理由と数が混ざらないよう "detail" を頭に付ける。
MAX_DETAIL = 3  # 仕様・数量・期限を別々の文から添えられるように3個まで
# detail は、項目の文から前後15文以内（根拠の5文より広い。期限・仕様は離れた文にあることが多い）。
DETAIL_WINDOW = 15
REASON_DETAIL_SHAPE = "detail形"  # 形崩れ（shape_errors）には数えない
REASON_DETAIL_NOT_ALLOWED = "detail不可"  # 困りごと・未決には許さない
REASON_DETAIL_COUNT = "detail個数"  # 3個目以降
DETAIL_PREFIX = "detail"
NOTE_OFFER_AGREED = "申し出と同意で確定"
# 確定に使わないお客様の発話。
REASON_WEAK_AGREE = "弱い同意"  # 「いいかも」「できそう」「助かる」は反応で、受諾ではない
REASON_QUESTION = "質問の形"  # 「〜要りますよね」「〜できるんですか」は頼みではない
WEAK_AGREE_RE = re.compile(r"(かも|できそう|助か)")
QUESTION_END_RE = re.compile(r"(か|かね|よね|かな|[？?])$")
# 頼む言葉と、提案の言葉（「そこでどうですか」「いかがですか」は質問の形でも申し出）。
REQUEST_WORD_RE = re.compile(
    r"(もら|いただ|頂|くれ|お願い|ください|下さい|ほしい|欲しい|たい|どう|いかが)"
)
# 支援側が自分から言い出した作業に、お客様が後からお礼・喜びを返しただけ（「作ってもらえるのは
# 助かります」「送っていただけるのはありがたい」）は、頼みではなく反応。支援側への依頼の根拠にしない。
# 頼む言い方（「お願いします」「〜してもらえますか」「〜してほしい」）があれば依頼のまま。
THANKS_RE = re.compile(
    r"ありがた|ありがと|助か|嬉し|うれし|素晴らし|すばらし|心強|楽しみ|湧き|わき|安心|"
    r"(?:もら|いただ|頂)(?:え|け)る(?:と|のは|なら)$"
)
STRONG_REQUEST_RE = re.compile(
    r"お願い|ください|下さい|ほしい|欲しい|頼み|頼む|"
    r"(?:もら|いただ|頂)(?:え|け)(?:ます|ません)?(?:か|[？?])|(?:て|で)(?:くれ|ちょうだい)|たい(?!へん|ない)"
)
# 作業名（name）。読む役の短い自由文を、手元で言葉の出どころまで点検する。
NAME_MAX_CHARS = 40
REASON_NAME_SHAPE = "name形"
REASON_NAME_NOT_ALLOWED = "name不可"  # 作業以外
REASON_NAME_WORD = "name言葉"  # 記録の近くに無い言葉
REASON_NAME_FORBIDDEN = "name禁止文字"  # 数字・札
NAME_RUN_RE = re.compile(r"[\u4e00-\u9fff々〆ヵヶ]+|[ァ-ヴー]+|[A-Za-zＡ-Ｚａ-ｚ]+")
NAME_DIGITS_RE = re.compile(r"[0-9０-９]+")  # 数字は記録の近くに同じ並びがあるときだけ
NAME_FORBIDDEN_RE = re.compile(r"[〈〉<>]")  # 札は書かない
# 動作と、作業の名前によく使う言葉（記録に無くても使ってよい）。
NAME_WORDS = frozenset(
    """作成 制作 撮影 取得 設定 準備 公開 確認 送付 提示 提出 検討 整理 試作 指導
    説明 掲載 更新 手配 用意 修正 変更 記録 共有 相談 連絡 見直 調整 案内 導入 登録
    管理 印刷 配布 発注 対応 支援 運用 試験 選定 収集 集計 比較 一覧 資料 原稿 内容
    方法 費用 日程 候補 作業 担当 方針 手順 書式 作り 直し 撮り 打ち 合わせ 受け 渡し
    追加 標準 初期 月額 分野 希望 紹介 事前 使用 許可 必要 計算 整備 状況 仕組 入力
    再訪 発信 受付 申込 掲示 告知 発送 返信 回答 報告 提案 改善 統一 分類 保存 移行
    メモ ページ データ リスト フォーム デザイン テンプレート""".split()
)
NAME_AFFIXES = frozenset("的用案例書表等別化内外先後前各新再仮初全旧済中枠欄版面申")  # 作業名を漢字のひと続きに切り分けるときに使う、決めた1文字
NOTE_TROUBLE_MERGED = "困りごと統合"  # 同じ文・同じ区間を指す困りごとを1行に
TROUBLE_RESERVED_SLOTS = 0  # 困りごとの8行のうち、品書きの札付きの区間用に空ける行数（今は空けない）

# 未決へ回した理由。
U_NO_BASIS = "根拠なし"
U_PARTY_UNKNOWN = "当事者不明"
U_WITHDRAWN = "撤回"
U_PROPOSED = "提案中"
U_ONE_SIDED = "片方の提案"
U_CUSTOMER_OPEN = "お客様の未確定"
U_HOLD = "保留"
U_UNASSIGNED = "担当未確定"

UNASSIGNED_MARK = "〈担当未確定〉"

_TAG_RE = re.compile(r"〈[^〉]*〉")


# ---------------------------------------------------------------- 形の道具


def _is_int(value: Any) -> bool:
    return type(value) is int  # bool は int の子なので ``type`` で見る


def _is_blank_char(char: str) -> bool:
    return char.isspace() or unicodedata.category(char)[0] in "PZ"


def _has_content(span: str, text: str) -> bool:
    """引用 ``span`` が、札の外に中身のある文字を1つでも含むか.

    〈内情〉〈省略〉などの札だけ、札の途中だけ、句読点だけの引用は ``False``。
    """
    tags = [m.span() for m in _TAG_RE.finditer(text)]
    start = text.find(span)
    while start != -1:
        for index in range(start, start + len(span)):
            if any(a <= index < b for a, b in tags):
                continue
            if _is_blank_char(text[index]):
                continue
            return True
        start = text.find(span, start + 1)
    return False


@dataclass(frozen=True)
class _Ctx:
    texts: dict[int, str]
    states: dict[int, str]
    replies: dict[int, bool]
    regions: dict[int, str]  # 区間の番号 -> 札
    total: int
    region_ranges: dict[int, tuple[int, int]] = field(default_factory=dict)  # 区間の番号 -> (from, to)


def _context(masked: MaskedR3) -> _Ctx:
    return _Ctx(
        texts={r.sent: r.text for r in masked.records},
        states={r.sent: r.state for r in masked.records},
        replies={r.sent: bool(r.region_reply) for r in masked.records},
        regions={g.region: g.tag for g in masked.regions},
        total=max((r.sent for r in masked.records), default=0),
        region_ranges={g.region: (g.start, g.end) for g in masked.regions},
    )


def _check_quote(sent: Any, span: Any, ctx: _Ctx) -> tuple[str | None, str]:
    """``sent`` と ``span`` の点検（理由と、前後の空白を除いた span）."""
    if not _is_int(sent) or not isinstance(span, str):
        return REASON_SHAPE, ""
    if sent not in ctx.texts:
        return REASON_NUMBER, ""
    if ctx.states[sent] == "伏せた":
        return REASON_HIDDEN, ""
    cleaned = span.strip()
    text = ctx.texts[sent]
    if not cleaned or cleaned not in text:
        return REASON_NOT_PART, ""
    if len(cleaned) > MAX_SPAN_CHARS:
        return REASON_TOO_LONG, ""
    if not _has_content(cleaned, text):
        return REASON_TAG_ONLY, ""
    return None, cleaned


# ---------------------------------------------------------------- データ


@dataclass
class _Evidence:
    type: str
    speaker: str
    to: str
    sent: int
    span: str
    far: bool = False  # 前後5文の外の依頼（ask_to_ours にだけ使う）


@dataclass
class _Item:
    index: int
    kind: str
    sent: int | None
    region: int | None
    span: str
    who: str
    state: str
    requested: bool
    evidence: list[_Evidence]
    topic: int
    rank: int
    detail: tuple[tuple[int, str], ...] = ()  # 対象の引用（文番号, 引用）
    name: str = ""  # 作業名（点検を通ったものだけ。表示にだけ使う）
    step: int | None = None  # 道筋だけ。段階の番号
    final: bool = False  # 道筋だけ。最後に目指す姿
    speaker: str = SPEAKER_UNKNOWN  # 行の文を話した側（お客様 / 支援側 / 不明）


@dataclass
class _Row5:
    field: str
    kind: str
    sent: int | None
    region: int | None
    span: str
    who: str
    state: str
    requested: bool
    topic: int
    rank: int
    order: int
    reason: str = ""
    deferred: bool = False
    unassigned: bool = False
    detail: tuple[tuple[int, str], ...] = ()
    name: str = ""
    step: int | None = None
    final: bool = False
    speaker: str = SPEAKER_UNKNOWN

    def key(self) -> tuple[Any, ...]:
        """同じ欄の中で1行にまとめる条件（文番号・引用・kind・who・state）."""
        return (self.sent, self.region, self.span, self.kind, self.who, self.state)

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "kind": self.kind,
            "sent": self.sent,
            "region": self.region,
            "span": self.span,
            "who": self.who,
            "state": self.state,
            "requested": self.requested,
            "topic": self.topic,
            "rank": self.rank,
            "reason": self.reason,
            "deferred": self.deferred,
            "unassigned": self.unassigned,
            "detail": [{"sent": n, "span": t} for n, t in self.detail],
            "name": self.name,
            "step": self.step,
            "final": self.final,
            "speaker": self.speaker,
        }


def _drop(
    dropped: list[dict[str, Any]],
    *,
    scope: str,
    reason: str,
    sent: Any = None,
    index: int | None = None,
    field: str | None = None,
) -> None:
    entry: dict[str, Any] = {
        "scope": scope,
        "sent": sent if _is_int(sent) else None,
        "reason": reason,
    }
    if index is not None:
        entry["index"] = index
    if field is not None:
        entry["field"] = field
    dropped.append(entry)


# ---------------------------------------------------------------- speakers


def _clean_speakers(
    raw: Any, ctx: _Ctx, dropped: list[dict[str, Any]]
) -> list[tuple[int, int, str]]:
    """範囲外・逆順・重なり・形違いの話し手は無視して記録する（``話し手の形``）."""
    if raw is None:
        return []
    if type(raw) is not list:
        _drop(dropped, scope="speakers", reason=REASON_SPEAKER_SHAPE)
        return []
    accepted: list[tuple[int, int, str]] = []
    for index, entry in enumerate(raw):
        ok = (
            type(entry) is dict
            and _is_int(entry.get("from"))
            and _is_int(entry.get("to"))
            and entry.get("who") in SPEAKER_VALUES
        )
        if ok:
            start, end, who = entry["from"], entry["to"], entry["who"]
            ok = (
                1 <= start <= end <= ctx.total
                and all(end < a or start > b for a, b, _ in accepted)
            )
        if not ok:
            _drop(dropped, scope="speakers", reason=REASON_SPEAKER_SHAPE, index=index)
            continue
        accepted.append((start, end, who))
    return accepted


def _speaker_at(speakers: Sequence[tuple[int, int, str]], sent: int) -> str | None:
    for start, end, who in speakers:
        if start <= sent <= end:
            return who
    return None


def _item_speaker(
    item: _Item, ctx: _Ctx, speakers: Sequence[tuple[int, int, str]]
) -> str:
    """行の文を話した側（お客様 / 支援側 / 不明）を決める.

    1. 行の元の文（``sent``）と同じ文を指す有効な根拠の ``speaker``（不明は除く）。
       複数あって食い違うときは使わない。ほかの文の根拠は使わない。
    2. ``speakers`` の範囲（文の範囲。区間の札だけの行は、区間の範囲の話し手が1種類だけのとき
       その話し手。範囲の話し手が不明を含めて2種類以上、または範囲の外だけなら不明）。
    3. 無ければ「不明」。
    """
    if item.sent is not None:
        said = {e.speaker for e in item.evidence if e.sent == item.sent and e.speaker != SPEAKER_UNKNOWN}
        if len(said) == 1:
            return next(iter(said))
        return _speaker_at(speakers, item.sent) or SPEAKER_UNKNOWN
    if item.region is not None and item.region in ctx.region_ranges:
        start, end = ctx.region_ranges[item.region]
        kinds = {who for n in range(start, end + 1) if (who := _speaker_at(speakers, n)) is not None}
        if len(kinds) == 1:
            return next(iter(kinds))
    return SPEAKER_UNKNOWN


# ---------------------------------------------------------------- 項目と根拠


def _parse_evidence(
    raw: Any,
    *,
    item_index: int,
    item_sent: int,
    ctx: _Ctx,
    speakers: Sequence[tuple[int, int, str]],
    dropped: list[dict[str, Any]],
    kind: str = "",
) -> _Evidence | None:
    """根拠1件の形と有効さを点検する。無効なら ``dropped`` に残して ``None``."""

    def bad(reason: str, sent: Any = None) -> None:
        _drop(dropped, scope="evidence", reason=reason, sent=sent, index=item_index)

    if type(raw) is not dict:
        bad(REASON_SHAPE)
        return None
    etype = raw.get("type")
    speaker = raw.get("speaker")
    sent = raw.get("sent")
    span = raw.get("span")
    if (
        etype not in EVIDENCE_TYPES
        or speaker not in SPEAKER_VALUES
        or not _is_int(sent)
        or type(span) is not str
    ):
        bad(REASON_SHAPE)
        return None
    if etype in ASK_TYPES:
        to = raw.get("to")
        if to not in TO_VALUES:  # 依頼・申し出は to が必須
            bad(REASON_SHAPE, sent)
            return None
    else:
        to = raw.get("to", "不明")
        if to not in TO_VALUES:
            bad(REASON_SHAPE, sent)
            return None

    # 引用・位置・種類の点検（無効な根拠は除く）。
    reason, cleaned = _check_quote(sent, span, ctx)
    far = False
    if reason is None:
        if ctx.replies[sent]:
            reason = REASON_REGION_REPLY
        elif abs(sent - item_sent) > EVIDENCE_WINDOW and not (
            kind == "作業" and etype == "依頼" and speaker == "お客様" and to == "支援側"
        ):
            reason = REASON_TOO_FAR
        else:
            far = abs(sent - item_sent) > EVIDENCE_WINDOW
            actual = _speaker_at(speakers, sent)
            if (
                actual is not None
                and actual != "不明"
                and speaker != "不明"
                and actual != speaker
            ):
                reason = REASON_SPEAKER_CONFLICT
    # お客様の弱い同意・頼む言葉の無い質問は、確定の根拠にしない。
    if reason is None and speaker == "お客様":
        if etype in ACCEPT_TYPES and WEAK_AGREE_RE.search(cleaned):
            reason = REASON_WEAK_AGREE
        elif (
            etype in ASK_TYPES
            and QUESTION_END_RE.search(cleaned.rstrip("。、. 　"))
            and not REQUEST_WORD_RE.search(cleaned)
        ):
            reason = REASON_QUESTION
    if reason is not None:
        bad(reason, sent)
        return None
    return _Evidence(
        type=etype, speaker=speaker, to=to, sent=sent, span=cleaned, far=far
    )


def _parse_detail(
    raw: dict[str, Any],
    *,
    kind: str,
    item_index: int,
    item_sent: int | None,
    ctx: _Ctx,
    dropped: list[dict[str, Any]],
    item_span: str | None = None,
) -> tuple[tuple[int, str], ...]:
    """項目の ``detail``（対象の引用）を点検する.

    作業・決めごとだけ。だめな ``detail`` は落として ``dropped``（scope ``detail``）に
    理由を残し、項目は残す。型違い・形違いも detail だけを落とす（項目の形崩れにしない）。
    点検は根拠の引用と同じ:位置・伏せた文・札だけ・120字以内・``region_reply``・
    項目の文から前後15文以内（``DETAIL_WINDOW``）。3個まで（``MAX_DETAIL``）。
    """
    if "detail" not in raw:
        return ()
    value = raw["detail"]

    def bad(reason: str, sent: Any = None) -> None:
        _drop(dropped, scope="detail", reason=reason, sent=sent, index=item_index)

    if kind not in ("作業", "決めごと") or item_sent is None:
        bad(REASON_DETAIL_NOT_ALLOWED, item_sent)
        return ()
    if value is None:
        return ()
    if type(value) is not list:
        bad(REASON_DETAIL_SHAPE, item_sent)
        return ()
    accepted: list[tuple[int, str]] = []
    for entry in value:
        if len(accepted) >= MAX_DETAIL:
            bad(REASON_DETAIL_COUNT)
            continue
        if type(entry) is not dict:
            bad(REASON_DETAIL_SHAPE)
            continue
        sent, span = entry.get("sent"), entry.get("span")
        reason, cleaned = _check_quote(sent, span, ctx)
        if reason == REASON_SHAPE:
            bad(REASON_DETAIL_SHAPE)
            continue
        if reason is None:
            if ctx.replies[sent]:
                reason = REASON_REGION_REPLY
            elif abs(sent - item_sent) > DETAIL_WINDOW:
                reason = REASON_TOO_FAR
        if reason is not None:
            bad(DETAIL_PREFIX + reason, sent)
            continue
        if (sent, cleaned) not in accepted:
            accepted.append((sent, cleaned))
    return _dedupe_detail(accepted, item_span)


def _dedupe_detail(
    accepted: Sequence[tuple[int, str]], item_span: str | None
) -> tuple[tuple[int, str], ...]:
    """同じ中身の繰り返しを外す（冗長な引用を減らす）.

    項目の引用に含まれる detail、同じ文のほかの detail に含まれる detail は出さない
    （落とすのは表示の重なりだけ。``dropped`` には残さない）。
    """
    kept: list[tuple[int, str]] = []
    for sent, span in accepted:
        if item_span and span in item_span:
            continue
        if any(other_sent == sent and span in other and span != other for other_sent, other in accepted):
            continue
        if (sent, span) not in kept:
            kept.append((sent, span))
    return tuple(kept)


def _parse_item(
    raw: Any,
    index: int,
    ctx: _Ctx,
    speakers: Sequence[tuple[int, int, str]],
    dropped: list[dict[str, Any]],
) -> _Item | None:
    def bad(reason: str, sent: Any = None) -> None:
        _drop(dropped, scope="item", reason=reason, sent=sent, index=index)

    if type(raw) is not dict:
        bad(REASON_SHAPE)
        return None
    kind = raw.get("kind")
    if kind not in KINDS:
        bad(REASON_SHAPE)
        return None
    topic, rank = raw.get("topic"), raw.get("rank")
    if not _is_int(topic) or not _is_int(rank):
        bad(REASON_SHAPE, raw.get("sent"))
        return None

    who = "不明"
    state = ""
    requested = False
    path_step: int | None = None
    path_final = False
    if kind == "作業":
        who, state, requested = raw.get("who"), raw.get("state"), raw.get("requested")
        if (
            who not in WHO_VALUES
            or state not in WORK_STATES
            or type(requested) is not bool
        ):
            bad(REASON_SHAPE, raw.get("sent"))
            return None
    elif kind == "決めごと":
        state = raw.get("state")
        if state not in DECISION_STATES or "who" in raw:
            bad(REASON_SHAPE, raw.get("sent"))
            return None
    elif kind == KIND_PATH:
        # step は整数（1以上）か null、final は真偽（null は偽）。型違いは項目の形崩れ。
        # who・state・requested・evidence は読む役が付けても無視する（落とすと道筋そのものが消える）。
        step = raw.get("step")
        final = raw.get("final")
        if final is None:
            final = False
        if (step is not None and (not _is_int(step) or step < 1)) or type(final) is not bool:
            bad(REASON_SHAPE, raw.get("sent"))
            return None
        path_step, path_final = step, final
    elif "who" in raw or "state" in raw:  # 困りごと・未決
        bad(REASON_SHAPE, raw.get("sent"))
        return None

    # 区間の札は、困りごとで ``region`` だけを書いたときだけ。
    has_region = "region" in raw
    if has_region:
        region = raw.get("region")
        if (
            kind != "困りごと"
            or "sent" in raw
            or "span" in raw
            or not _is_int(region)
        ):
            bad(REASON_SHAPE)
            return None
        if region not in ctx.regions:
            bad(REASON_NUMBER)
            return None
        _parse_detail(
            raw, kind=kind, item_index=index, item_sent=None, ctx=ctx, dropped=dropped
        )
        return _Item(
            index=index,
            kind=kind,
            sent=None,
            region=region,
            span=ctx.regions[region],
            who="不明",
            state="",
            requested=False,
            evidence=[],
            topic=topic,
            rank=rank,
        )

    sent = raw.get("sent")
    reason, span = _check_quote(sent, raw.get("span"), ctx)
    if reason is not None:
        bad(reason, sent)
        return None

    evidence: list[_Evidence] = []
    if kind in ("作業", "決めごと"):
        raw_evidence = raw.get("evidence", [])
        if type(raw_evidence) is not list:
            bad(REASON_SHAPE, sent)
            return None
        for entry in raw_evidence:
            parsed = _parse_evidence(
                entry,
                item_index=index,
                item_sent=sent,
                ctx=ctx,
                speakers=speakers,
                dropped=dropped,
                kind=kind,
            )
            if parsed is not None:
                evidence.append(parsed)
    detail = _parse_detail(
        raw, kind=kind, item_index=index, item_sent=sent, ctx=ctx, dropped=dropped, item_span=span
    )
    name = _parse_name(
        raw,
        kind=kind,
        item_index=index,
        near=_name_context(sent, evidence, detail, ctx),
        dropped=dropped,
    )
    return _Item(
        index=index,
        kind=kind,
        sent=sent,
        region=None,
        span=span,
        who=who,
        state=state,
        requested=requested,
        evidence=evidence,
        topic=topic,
        rank=rank,
        detail=detail,
        name=name,
        step=path_step,
        final=path_final,
    )


def _name_context(
    item_sent: int,
    evidence: Sequence[_Evidence],
    detail: Sequence[tuple[int, str]],
    ctx: _Ctx,
) -> str:
    """作業名の言葉を探す範囲: 記録の公開した文の全部（伏せた文・``region_reply`` は除く）.

    対象の言葉は前後5文の外にあることが多い（「チラシ」の話の
    途中の「初稿を出します」など）。作り話を防ぐには、記録に無い言葉を落とせば足りる。
    ``item_sent``・``evidence``・``detail`` は使わない（呼び出しの形を保つため残す）。
    """
    del item_sent, evidence, detail
    return "\n".join(
        ctx.texts[n]
        for n in sorted(ctx.texts)
        if ctx.states[n] != "伏せた" and not ctx.replies[n]
    )


def _run_is_grounded(run: str, near: str) -> bool:
    """漢字・カタカナ・英字のひと続きが、記録の近くの言葉・決めた言葉で組めるか.

    1文字だけのひと続き（「作る」の「作」など）は見ない。2文字以上は、2文字以上の
    かけら（記録の近くにある・``NAME_WORDS`` にある）と、決めた1文字（``NAME_AFFIXES``）
    だけで、頭から終わりまで切り分けられること。
    """
    if len(run) <= 1:
        return True
    ok = [False] * (len(run) + 1)
    ok[0] = True
    for end in range(1, len(run) + 1):
        for start in range(end):
            if not ok[start]:
                continue
            piece = run[start:end]
            if (len(piece) >= 2 and (piece in near or piece in NAME_WORDS)) or (
                len(piece) == 1 and piece in NAME_AFFIXES
            ):
                ok[end] = True
                break
    return ok[len(run)]


def _parse_name(
    raw: dict[str, Any],
    *,
    kind: str,
    item_index: int,
    near: str,
    dropped: list[dict[str, Any]],
) -> str:
    """作業の任意の ``name``（短い作業名）を点検する.

    だめなら name だけを落として記録し、項目は残す（表示は引用だけに戻る）。
    """
    if "name" not in raw or raw["name"] in (None, ""):
        return ""
    sent = raw.get("sent")

    def bad(reason: str) -> str:
        _drop(dropped, scope="name", reason=reason, sent=sent, index=item_index)
        return ""

    value = raw["name"]
    if kind != "作業":
        return bad(REASON_NAME_NOT_ALLOWED)
    if type(value) is not str:
        return bad(REASON_NAME_SHAPE)
    name = value.strip()
    if not name or len(name) > NAME_MAX_CHARS or "\n" in name:
        return bad(REASON_NAME_SHAPE)
    if NAME_FORBIDDEN_RE.search(name):
        return bad(REASON_NAME_FORBIDDEN)
    for run in NAME_RUN_RE.findall(name):
        if not _run_is_grounded(run, near):
            return bad(REASON_NAME_WORD)
    for digits in NAME_DIGITS_RE.findall(name):
        if digits not in near:
            return bad(REASON_NAME_WORD)
    return name


# ---------------------------------------------------------------- 欄の表


@dataclass
class _Facts:
    ask_to_ours: bool  # 支援側への依頼（依頼・お客様・to 支援側）
    ours_accept: bool  # 支援側の引き受け（引き受け/同意・支援側）
    customer_accept: bool  # お客様の引き受け（引き受け/同意・お客様）
    mutual: bool  # 双方の合意
    withdrawn: bool
    hold: bool
    party_unknown: bool
    offer_agreed: str  # 支援側の申し出＋お客様の返事があれば返事の型（無ければ ""）


def _offer_agreed(evidence: Sequence[_Evidence]) -> str:
    """支援側の申し出（to お客様）に、お客様が後の文で返事をしているか.

    返事は、お客様の ``同意``・``引き受け``（``to`` が第三者のものは除く）か、
    支援側への ``依頼``（「お願いします」）。申し出より前の文の返事・第三者への返事・
    支援側の相づちは数えない。返事の型を返す（``同意``・``引き受け``・``依頼``。無ければ ``""``）。
    """
    offers = [
        e.sent
        for e in evidence
        if e.type == "申し出" and e.speaker == "支援側" and e.to == "お客様"
    ]
    if not offers:
        return ""
    first = min(offers)
    for e in evidence:
        if e.speaker != "お客様" or e.sent <= first:
            continue
        if e.type in ACCEPT_TYPES and e.to != "第三者":
            return e.type
        if e.type == "依頼" and e.to == "支援側":
            return "依頼"
    return ""


def _facts(all_evidence: Sequence[_Evidence]) -> _Facts:
    parties = ("お客様", "支援側")
    # 遠い依頼は支援側への依頼の有無にだけ使う。
    far_ask = any(e.far for e in all_evidence)
    evidence = [e for e in all_evidence if not e.far]
    asks = [e for e in evidence if e.type in ASK_TYPES]
    accepts = [e for e in evidence if e.type in ACCEPT_TYPES]
    mutual = any(
        a.speaker in parties
        and b.speaker in parties
        and a.speaker != b.speaker
        and a.to == b.speaker
        for a in asks
        for b in accepts
    )
    unknown = any(e.speaker == "不明" for e in asks + accepts) or any(
        e.to == "不明" for e in asks
    )
    ours_first = min(
        (e.sent for e in all_evidence if e.speaker == "支援側" and e.type in ("申し出", "引き受け")),
        default=None,
    )

    def thanks_only(e: _Evidence) -> bool:
        return (
            ours_first is not None
            and e.sent > ours_first
            and THANKS_RE.search(e.span) is not None
            and STRONG_REQUEST_RE.search(e.span) is None
        )

    return _Facts(
        ask_to_ours=any(e.far and not thanks_only(e) for e in all_evidence)
        or any(
            e.type == "依頼"
            and e.speaker == "お客様"
            and e.to == "支援側"
            and not thanks_only(e)
            for e in asks
        ),
        ours_accept=any(e.speaker == "支援側" for e in accepts),
        customer_accept=any(e.speaker == "お客様" for e in accepts),
        mutual=mutual,
        withdrawn=any(e.type == "撤回" for e in evidence),
        hold=any(e.type == "保留" for e in evidence),
        party_unknown=unknown,
        offer_agreed=_offer_agreed(evidence),
    )


def _make_row(item: _Item, field: str, order: int, **extra: Any) -> _Row5:
    return _Row5(
        field=field,
        kind=item.kind,
        sent=item.sent,
        region=item.region,
        span=item.span,
        who=item.who,
        state=item.state,
        requested=item.requested,
        topic=item.topic,
        rank=item.rank,
        order=order,
        detail=item.detail,
        name=item.name,
        step=item.step,
        final=item.final,
        speaker=item.speaker,
        **extra,
    )


NOTE_BORROWED_ASK = "ほかの作業の頼み"
_INTERIM_RE = re.compile(r"構成案|試作|試案|たたき台|叩き台|ラフ|モック|デモ|ページ案|画面案|原案|素案")
_EXPLICIT_ASK_RE = re.compile(
    r"お願い|ください|下さい|ほしい|欲しい|頼み|頼む|(?:もら|いただ|頂)(?:え|け)(?:ます|ません)?(?:か|[？?])"
)


def _is_customer_ask(e: _Evidence) -> bool:
    return e.type == "依頼" and e.speaker == "お客様" and e.to == "支援側"


def _drop_borrowed_asks(items: list[_Item], notes: list[dict[str, Any]]) -> None:
    """支援側が自分から切り出した作業に、ほかの作業の頼みを借りた根拠を外す.

    お客様が「ページを作ってほしい」と頼んだ文（その文から出た作業がある）を、後で支援側が
    自分から「構成案を作っておきます」と切り出した別の作業の依頼の根拠にも使うと、構成案まで
    「頼まれていること」に出る（依頼の向きの取り違え）。頼みの文から出た作業が別に
    あり、この作業は支援側が自分の文で切り出した（引き受け・申し出がこの作業の文にある）ときは、
    その頼みはこの作業の根拠にしない。
    """
    anchored = {
        item.sent
        for item in items
        if item.kind == "作業"
        and item.sent is not None
        and any(_is_customer_ask(e) and e.sent == item.sent for e in item.evidence)
    }
    for item in items:
        if item.kind != "作業" or item.who != "支援側" or item.sent is None:
            continue
        # 途中の成果物（構成案・試作・たたき台など）だけ。本制作そのもの（料金表ページ・予約フォーム）は
        # 前の頼みの言い直しのことがあるので外さない。
        if not _INTERIM_RE.search(item.name + " " + item.span):
            continue
        self_started = any(
            e.speaker == "支援側" and e.type in ("申し出", "引き受け") and e.sent == item.sent
            for e in item.evidence
        )
        if not self_started:
            continue
        borrowed = [
            e
            for e in item.evidence
            if _is_customer_ask(e)
            and (
                (e.sent < item.sent and e.sent in anchored)
                # 切り出した後のお客様の中身の希望（「〜ははっきり書いときたい」）は、頼む言葉が無ければ反応。
                or (e.sent > item.sent and _EXPLICIT_ASK_RE.search(e.span) is None)
            )
        ]
        if not borrowed:
            continue
        item.evidence = [e for e in item.evidence if e not in borrowed]
        notes.append(
            {
                "reason": NOTE_BORROWED_ASK,
                "sent": item.sent,
                "index": item.index,
                "asks": [e.sent for e in borrowed],
            }
        )


# 先送りの印は、保留の発話がその項目と同じ事柄を指すときだけ。保留の引用と、項目の引用・名前・
# detail が、漢字・カタカナの2字の並びを1つも共有しないなら、別の事柄（決裁待ち・社内で話す・講座の開催など）の
# 保留を借りたとみて印を付けない（欄は「まだ決まっていないこと」のまま）。
_GRAM_CHARS = re.compile(r"[一-龥々ァ-ヶーA-Za-z]{2,}")


def _grams(text: str) -> set[str]:
    out: set[str] = set()
    for run in _GRAM_CHARS.findall(text):
        out.update(run[i : i + 2] for i in range(len(run) - 1))
    return out


# 保留の言い方そのもの（「一旦保留にしておきましょう」は事柄を名指ししていない。前後の項目を指す）。
_HOLD_WORDS_RE = re.compile(r"一旦|保留|先送り|後回し|見送り|持ち越し|検討|判断|置いて|その件|この件|あの件")


def _hold_names_the_item(item: _Item) -> bool:
    own = _grams(item.span) | _grams(item.name) | set().union(*(_grams(t) for _, t in item.detail))
    for e in item.evidence:
        if e.type != "保留":
            continue
        named = _grams(_HOLD_WORDS_RE.sub("、", e.span))
        if not named or named & own:
            return True
    return False


def _route(item: _Item, order: int, notes: list[dict[str, Any]]) -> list[_Row5]:
    """欄の表（上から順に、最初に当たった行）。1件が複数の欄に出ることがある."""
    # 道筋は専用の欄へ（5欄の表には入れない。1件が複数の欄に出ることもない）。
    if item.kind == KIND_PATH:
        return [_make_row(item, F_PATH, order)]
    # 困りごと・未決は、読む役の kind のまま。
    if item.kind == "困りごと":
        return [_make_row(item, F_TROUBLE, order)]
    if item.kind == "未決":
        return [_make_row(item, F_UNDECIDED, order)]

    facts = _facts(item.evidence)
    # 有効な撤回は、表より先にまだ決まっていないことへ。
    if facts.withdrawn:
        return [_make_row(item, F_UNDECIDED, order, reason=U_WITHDRAWN)]
    # 保留。保留の根拠があるときだけ（先送り）の印。
    if item.state == "保留":
        return [
            _make_row(
                item, F_UNDECIDED, order, reason=U_HOLD,
                deferred=facts.hold and _hold_names_the_item(item),
            )
        ]

    if item.kind == "作業":
        if item.requested and not facts.ask_to_ours:
            notes.append(
                {"reason": REASON_NO_ASK_BASIS, "sent": item.sent, "index": item.index}
            )
        # 担当が不明。
        if item.who == "不明":
            return [
                _make_row(
                    item, F_UNDECIDED, order, reason=U_UNASSIGNED, unassigned=True
                )
            ]
        # 支援側が引き受けた作業（引き受けの根拠がある）。
        if item.who == "支援側" and item.state == "引き受けた" and facts.ours_accept:
            rows = [_make_row(item, F_OURS, order)]
            if item.requested and facts.ask_to_ours:
                rows.append(_make_row(item, F_ASKED, order))
            return rows
        # 支援側の申し出にお客様の返事がそろう＝双方の合意で、作業する
        # 人が支援側。state が 提案中 などで返ってきても、根拠がそろえば「引き受けた作業」と同じ扱い。
        # 保留の根拠（撤回は表の前で振り分け済み）があるときは当てない。
        if item.who == "支援側" and facts.offer_agreed and not facts.hold:
            notes.append(
                {
                    "reason": NOTE_OFFER_AGREED,
                    "sent": item.sent,
                    "index": item.index,
                    "state": item.state,
                    "via": facts.offer_agreed,
                }
            )
            rows = [_make_row(item, F_OURS, order)]
            if item.requested and facts.ask_to_ours:
                rows.append(_make_row(item, F_ASKED, order))
            return rows
        # 支援側への依頼だけがある作業（まだ引き受けていない）。
        if item.who == "支援側" and item.state == "頼まれただけ" and facts.ask_to_ours:
            return [_make_row(item, F_ASKED, order)]
        # お客様がやる作業で、双方が合意したもの。
        if (
            item.who == "お客様"
            and item.state == "引き受けた"
            and facts.mutual
            and facts.customer_accept
        ):
            return [_make_row(item, F_DECIDED, order)]
    elif item.kind == "決めごと":
        # 双方が合意した決めごと。
        if item.state == "合意" and facts.mutual:
            return [_make_row(item, F_DECIDED, order)]

    # ここまでのどれにも当たらないものは、理由を付けてまだ決まっていないことへ。
    # 支援側がやる作業で、requested かつ支援側への依頼が
    # 有効なのに、まだ引き受けがそろわない（提案で返しただけ・引き受けの根拠が足りない）
    # ものは、頼まれていることにも出す（まだ決まっていないことにも残す）。
    asked: list[_Row5] = []
    if (
        item.kind == "作業"
        and item.who == "支援側"
        and item.requested
        and facts.ask_to_ours
    ):
        asked = [_make_row(item, F_ASKED, order)]
    if item.state == "提案中":
        reason = U_PROPOSED
    elif item.state == "片方の提案":
        reason = U_ONE_SIDED
    elif facts.party_unknown:
        reason = U_PARTY_UNKNOWN
    elif item.kind == "作業" and item.who == "お客様" and item.state == "頼まれただけ":
        reason = U_CUSTOMER_OPEN
    else:
        reason = U_NO_BASIS
    return asked + [_make_row(item, F_UNDECIDED, order, reason=reason)]


# ---------------------------------------------------------------- 行数


def _merge_trouble_rows(
    column: list[_Row5],
    region_of: dict[int, int | None],
    notes: list[dict[str, Any]],
) -> list[_Row5]:
    """困りごとの行のうち、同じ文・同じ区間を指す行を1行にまとめる.

    - 同じ文番号を指す行（引用が違っても）は、``rank`` の小さい行1つに。
    - 同じ区間の札を指す行は1つに。
    - 区間の中の文（体調の節の応答など）を指す行と、同じ区間の札の行があれば、
      中身のある文の行に寄せる（札だけの行は消す）。
    ``困りごと``・``未決`` 以外（表の3・4・9行目で未決へ移した行）には触れない。
    """

    def better(a: _Row5, b: _Row5) -> _Row5:
        return a if (a.rank, a.order) <= (b.rank, b.order) else b

    merged: list[_Row5] = []
    by_sent: dict[int, int] = {}
    by_region: dict[int, int] = {}
    for row in column:
        if row.kind != "困りごと":
            merged.append(row)
            continue
        slot = by_sent.get(row.sent) if row.sent is not None else by_region.get(row.region)
        if slot is None:
            merged.append(row)
            if row.sent is not None:
                by_sent[row.sent] = len(merged) - 1
            elif row.region is not None:
                by_region[row.region] = len(merged) - 1
            continue
        keep = better(merged[slot], row)
        notes.append({"reason": NOTE_TROUBLE_MERGED, "sent": row.sent, "region": row.region})
        merged[slot] = keep
    # 区間の札の行と、その区間の中の文の行が両方あるとき、札の行を消す。
    in_region = {
        region_of[row.sent]
        for row in merged
        if row.kind == "困りごと" and row.sent is not None and region_of.get(row.sent) is not None
    }
    result: list[_Row5] = []
    for row in merged:
        if row.kind == "困りごと" and row.region is not None and row.region in in_region:
            notes.append({"reason": NOTE_TROUBLE_MERGED, "sent": None, "region": row.region})
            continue
        result.append(row)
    return result


def _pick_trouble_rows(limited: list[_Row5]) -> tuple[list[_Row5], list[_Row5]]:
    """困りごとの行から、8行を選ぶ. 返すのは（残す行, 落とす行）.

    選ぶ順:
    1. ``topic`` ごとに、最も良い行を1行ずつ先に入れる（話題が8個を超えたら、その1行
       どうしを ``rank`` 順に8個まで）。
    2. 残りの枠を、残りの行で ``rank`` 順に埋める。
    「区間の札だけの行は文の行より後ろ」という決まりとの組み合わせ:
    - 話題の「最も良い行」は、段（文の行 → 品書きの札付きの区間 → 〈伏せた区間〉だけ）、
      ``rank`` の順で最初の行。同じ話題に文の行があれば、``rank`` が悪くても文の行が代表。
    - 〈伏せた区間〉だけの行（札に情報が無い）は、話題の代表には使わない。2.の枠が
      余ったときだけ入る。
    - 2.も 段 → ``rank`` の順。
    ``TROUBLE_RESERVED_SLOTS`` が1以上なら、品書きの札付きの区間のために、その行数
    だけ先に空ける（今は0）。
    残した行は、読む役の順位の順に並べる。
    """

    def position(row: _Row5) -> tuple[int, int, int]:
        return (row.rank, row.sent if row.sent is not None else 10**9, row.order)

    def tier(row: _Row5) -> int:
        if row.region is None:
            return 0
        return 2 if row.span == REGION_TAG_EMPTY else 1

    ordered = sorted(limited, key=lambda r: (tier(r), *position(r)))
    tagged = [r for r in ordered if tier(r) == 1]
    reserved = tagged[:TROUBLE_RESERVED_SLOTS]
    pool = [r for r in ordered if all(r is not x for x in reserved)]
    slots = MAX_ROWS_PER_FIELD - len(reserved)

    best: dict[int, _Row5] = {}
    for row in pool:  # ordered は 段 → rank の順なので、話題ごとの最初の行が代表
        if tier(row) < 2:
            best.setdefault(row.topic, row)
    leaders = sorted(best.values(), key=position)[:slots]
    leader_ids = {id(r) for r in leaders}
    fill = [r for r in pool if id(r) not in leader_ids][: slots - len(leaders)]
    chosen = leaders + fill + reserved
    chosen_ids = {id(r) for r in chosen}
    rest = [r for r in ordered if id(r) not in chosen_ids]
    return sorted(chosen, key=position), sorted(rest, key=position)


def _limit_rows(
    rows: list[_Row5],
    dropped: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    region_of: dict[int, int | None] | None = None,
) -> dict[str, list[_Row5]]:
    by_field: dict[str, list[_Row5]] = {field: [] for field in SHARED_FIELDS}
    seen: set[tuple[str, tuple[Any, ...]]] = set()
    for row in rows:  # 同じ欄の中で、文番号・引用・kind・who・state が同じ行だけ1行に
        if row.field == F_TROUBLE and row.kind == "困りごと":
            by_field[row.field].append(row)  # 困りごとは ``_merge_trouble_rows`` で順位の良い行に
            continue
        key = (row.field, row.key())
        if key in seen:
            continue
        seen.add(key)
        by_field[row.field].append(row)

    def by_position(row: _Row5) -> tuple[int, int]:
        return (row.sent if row.sent is not None else 10**9, row.order)

    for field in FIELDS:
        column = by_field[field]
        if field in (F_ASKED, F_OURS, F_DECIDED):
            by_field[field] = sorted(column, key=by_position)
            if len(column) > MAX_ROWS_PER_FIELD:
                notes.append(
                    {"reason": REASON_TOO_MANY, "field": field, "count": len(column)}
                )
            continue
        if field == F_TROUBLE:
            column = _merge_trouble_rows(column, region_of or {}, notes)
        # 困りごと・未決の kind の行だけ rank の小さい順に8行まで（困りごとは区間の札を後ろに）。
        # 欄の表で未決へ移した行（kind が作業・決めごと）は別枠で削らない。
        limited = [r for r in column if r.kind in ("困りごと", "未決")]
        moved = [r for r in column if r.kind not in ("困りごと", "未決")]
        if field == F_TROUBLE:
            kept, cut = _pick_trouble_rows(limited)
        else:
            limited.sort(key=lambda r: (r.rank, *by_position(r)))
            kept, cut = limited[:MAX_ROWS_PER_FIELD], limited[MAX_ROWS_PER_FIELD:]
        for row in cut:
            _drop(dropped, scope="line", reason=REASON_ROWS, sent=row.sent, field=field)
        by_field[field] = kept + sorted(moved, key=by_position)

    # 道筋は削らない。段階の小さい順 → 段階なし → 最後（同じなら文の番号順）。
    def path_position(row: _Row5) -> tuple[int, int, int, int]:
        group = 2 if row.final else (0 if row.step is not None else 1)
        return (group, row.step or 0, *by_position(row))

    path_rows = sorted(by_field[F_PATH], key=path_position)
    if len(path_rows) > MAX_ROWS_PER_FIELD:
        notes.append({"reason": REASON_TOO_MANY, "field": F_PATH, "count": len(path_rows)})
    by_field[F_PATH] = path_rows
    return by_field


def _line(row: _Row5, meeting_date: date | None) -> str:
    text = row.span
    if meeting_date is not None and row.region is None:
        text = add_date_suffixes(text, meeting_date)
    if row.name:  # 「作業名「引用」」
        text = f"{row.name}「{text}」"
    marks = (DEFERRAL_MARK if row.deferred else "") + (
        UNASSIGNED_MARK if row.unassigned else ""
    )
    if row.detail:  # 「引用（対象: detail1／detail2）」
        parts = [
            add_date_suffixes(span, meeting_date) if meeting_date is not None else span
            for _, span in row.detail
        ]
        text += f"（対象: {'／'.join(parts)}）"
    if row.kind == KIND_PATH:  # 「ステップ1: …」「最後: …」
        if row.final:
            text = f"最後{f'（ステップ{row.step}）' if row.step is not None else ''}: {text}"
        elif row.step is not None:
            text = f"ステップ{row.step}: {text}"
    return f"- {speaker_mark(row.speaker)}{marks}{text}"


def speaker_label(speaker: str) -> str:
    """表示用の話し手（「支援側」は「こちら」）。決まった3つ以外は「不明」."""
    if speaker == SPEAKER_SELF:
        return SPEAKER_SELF_LABEL
    return speaker if speaker == "お客様" else SPEAKER_UNKNOWN


def speaker_mark(speaker: str) -> str:
    """行頭の話し手の印（``[お客様] ``・``[こちら] ``・``[不明] ``）."""
    return f"[{speaker_label(speaker)}] "


def build_shared_r3(
    by_field: dict[str, list[_Row5]], meeting_date: date | None = None
) -> str:
    """本文を組む（道筋の欄がいちばん上、続けて5欄。行の無い欄は「なし」）."""
    lines: list[str] = []
    for field in SHARED_FIELDS:
        lines.append(f"【{field}】")
        rows = by_field[field]
        if rows:
            lines.extend(_line(row, meeting_date) for row in rows)
        else:
            lines.append("なし")
    return "\n".join(lines)


# ---------------------------------------------------------------- old_rules


def _old_rules(
    by_field: dict[str, list[_Row5]], ctx: _Ctx
) -> list[dict[str, Any]]:
    """``pick_bench`` の機械の決まりを当てたらどうなるかを記録するだけ（欄は変えない）.

    ここで手元にあるのは元の文ではなく伏せた後の文だけなので、
    伏せた後の ``text`` を使う。文字列は記録に残さない（欄・文番号・規則名だけ）。
    """
    texts = [ctx.texts.get(number, "") for number in range(1, ctx.total + 1)]
    rows = [
        (row, _Row(field_ai=row.field, field=row.field, sent=row.sent, span_masked=row.span, rules=[]))
        for field in FIELDS
        for row in by_field[field]
        if row.sent is not None
    ]
    _route_rows([old for _, old in rows], texts)
    return [
        {
            "sent": row.sent,
            "kind": row.kind,
            "field": row.field,
            "old_field": old.field,
            "rules": list(old.rules),
        }
        for row, old in rows
        if old.rules
    ]


# ---------------------------------------------------------------- 入口


def assemble_r3(
    masked: MaskedR3,
    reply: dict[str, Any],
    *,
    meeting_date: date | None = None,
) -> dict[str, Any]:
    """読む役の返事（JSON として読んだ辞書）から5欄を組む.

    返すのは ``columns``（欄ごとの行。道筋の欄 ``F_PATH`` を先頭に足した）・
    ``items``（5欄だけの平らな一覧。道筋は入れない）・``field_counts``（欄ごとの行数。
    道筋を含む）・``shared``（本文。道筋の欄がいちばん上）・``dropped``（落とした理由。
    文字列は残さない）・``counts``（落とした理由ごとの件数）・``shape_errors``（形崩れの件数）
    ・``speakers``（有効な話し手）・``notes``・``old_rules``。
    """
    ctx = _context(masked)
    dropped: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    speakers = _clean_speakers(reply.get("speakers"), ctx, dropped)

    raw_items = reply.get("items")
    if type(raw_items) is not list:
        raw_items = []
        _drop(dropped, scope="reply", reason=REASON_SHAPE)
    items = [
        parsed
        for index, raw in enumerate(raw_items)
        if (parsed := _parse_item(raw, index, ctx, speakers, dropped)) is not None
    ]

    for parsed_item in items:  # 行の文を話した側の印
        parsed_item.speaker = _item_speaker(parsed_item, ctx, speakers)
    _drop_borrowed_asks(items, notes)
    rows: list[_Row5] = []
    for order, item in enumerate(items):
        rows.extend(_route(item, order, notes))
    by_field = _limit_rows(
        rows,
        dropped,
        notes,
        {r.sent: r.region for r in masked.records},
    )

    counts: dict[str, int] = {}
    for entry in dropped:
        counts[entry["reason"]] = counts.get(entry["reason"], 0) + 1
    columns = {
        field: [row.as_dict() for row in by_field[field]] for field in SHARED_FIELDS
    }
    return {
        "columns": columns,
        # 5欄だけの平らな一覧（道筋は入れない）
        "items": [row for field in FIELDS for row in columns[field]],
        "field_counts": {field: len(columns[field]) for field in SHARED_FIELDS},
        "shared": build_shared_r3(by_field, meeting_date),
        "dropped": dropped,
        "counts": counts,
        "shape_errors": counts.get(REASON_SHAPE, 0),
        "speakers": [
            {"from": a, "to": b, "who": who} for a, b, who in speakers
        ],
        "notes": notes,
        "old_rules": _old_rules(by_field, ctx),
    }


def empty_assembly(reason: str) -> dict[str, Any]:
    """返事が使えないとき（全体の形崩れ・読む役の失敗）の空の組み立て結果."""
    dropped = [{"scope": "reply", "sent": None, "reason": reason}]
    return {
        "columns": {field: [] for field in SHARED_FIELDS},
        "items": [],
        "field_counts": {field: 0 for field in SHARED_FIELDS},
        "shared": "\n".join(f"【{field}】\nなし" for field in SHARED_FIELDS),
        "dropped": dropped,
        "counts": {reason: 1},
        "shape_errors": 1 if reason == REASON_SHAPE else 0,
        "speakers": [],
        "notes": [],
        "old_rules": [],
    }


