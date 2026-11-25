"""読む役で5欄を作る.

**読む役に送るのは、このモジュールだけ。** 送る前の確認（``send_review``）で、人が伏せたあとの
全文を読んで確認済みにした中身（``reader_payload()`` 全体の指紋 ``payload_sha256``）と、送る中身が
同じときだけ送る（``make_memo`` の関所）。読む役は引数で受け取る（テストでは偽物・スパイ）。

- 読む役は環境変数 ``SHOKOBAN_CLOUD_READER`` で選ぶ。未設定・空は「止めてあります」（送らない）、
  ``claude`` は Claude Code CLI（``claude -p``。モデルは ``SHOKOBAN_CLAUDE_MODEL``、既定 ``opus``。
  ここでは作るだけで、呼ぶのは ``make_memo`` のとき）、``opus`` は ``claude`` の古い呼び名、
  ``local`` は手元の OpenAI 互換サーバー（``SHOKOBAN_READER_URL`` 既定 ``http://127.0.0.1:1234/v1``、
  ``SHOKOBAN_READER_MODEL`` 必須。127.0.0.1・localhost・::1 以外は断る）、
  ``fake`` は手元だけで返す偽物（架空データでの画面の試用）。ほかの値は誤りとして断る。
- 保存先は確認済みの記録と同じ保存先の ``cloud-memos/<お客様>/<打ち合わせ>.json``（0600・フォルダ0700）。
- 5欄は伏せたあとの文だけから組むので、元の文・名前は入らない。人が直した版は ``edited`` に足す。
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .assemble_r3 import KIND_PATH, assemble_r3, empty_assembly
from .mac_app import MacAppError, _ensure_private_dir
from .pick_bench import MAX_SPAN_CHARS, REASON_SHAPE
from .reader_r3 import ReaderR3ShapeError, build_reader_prompt_r3, parse_reader_reply_r3, prompt_payload_r3
from .reader_local import DEFAULT_LOCAL_URL, make_local_reader
from .reader_stage5 import ReaderReply, ReaderStage5Error, make_claude_reader
from .send_review import SendReviewError, approval_path, check_payload_approval, payload_digest
from .stage5_r3_types import MASKED_SENTENCE, MaskedR3

READER_ENV = "SHOKOBAN_CLOUD_READER"
READER_OFF_LABEL = "止めてあります"
READER_CLAUDE = "claude"
READER_OPUS = "opus"  # claude の古い呼び名
READER_LOCAL = "local"
READER_FAKE = "fake"
CLAUDE_MODEL_ENV = "SHOKOBAN_CLAUDE_MODEL"
DEFAULT_CLAUDE_MODEL = "opus"
READER_URL_ENV = "SHOKOBAN_READER_URL"
READER_MODEL_ENV = "SHOKOBAN_READER_MODEL"
OPUS_TIMEOUT = 1800  # 読む役1回の待ち時間の上限（秒。30分）
MEMO_DIR = "cloud-memos"
MEMO_VERSION = 1

Reader = Callable[[str], Any]


class CloudMemoError(RuntimeError):
    """5欄を作れない（画面がこの文言を出して止まる。中身は含めない）."""


def fake_cloud_reader(prompt: str) -> ReaderReply:
    """手元だけで返す偽の読む役（外へは何も送らない）。形は必ず通る ``speakers`` / ``items``.

    伏せていない先頭の文から、未決と困りごとを1つずつ返す。伏せていない最後の文は、支援側（こちら）が
    話した道筋（ステップ1）として1つ返す（架空データで画面を試すため）。
    """
    payload = prompt_payload_r3(prompt)
    usable = [
        record
        for record in payload["records"]
        if type(record.get("sent")) is int
        and str(record.get("text") or "").strip()
        and record.get("text") != MASKED_SENTENCE
    ]
    items: list[dict[str, Any]] = []
    for rank, (record, kind) in enumerate(zip(usable, ("未決", "困りごと")), start=1):
        text = str(record["text"]).strip()
        items.append(
            {"kind": kind, "sent": record["sent"], "span": text[:MAX_SPAN_CHARS], "topic": rank, "rank": rank}
        )
    speakers: list[dict[str, Any]] = []
    if usable:
        last = usable[-1]
        text = str(last["text"]).strip()
        items.append(
            {
                "kind": KIND_PATH,
                "sent": last["sent"],
                "span": text[:MAX_SPAN_CHARS],
                "step": 1,
                "final": False,
                "topic": 3,
                "rank": 3,
            }
        )
        speakers.append({"from": last["sent"], "to": last["sent"], "who": "支援側"})
    return ReaderReply(
        text=json.dumps({"speakers": speakers, "items": items}, ensure_ascii=False), model=READER_FAKE
    )


def cloud_reader_from_env() -> tuple[Reader | None, str]:
    """環境変数から読む役を選ぶ。``(読む役 or None, 表示用の名前)``.

    未設定・空は ``(None, "止めてあります")``。``claude``（``opus``）・``local`` は読む役を作るだけ
    （呼ばない）。表示用の名前は、claude は ``claude (<モデル>)``、local は ``local (<モデル>)``。
    """
    value = os.environ.get(READER_ENV, "").strip()
    if not value:
        return None, READER_OFF_LABEL
    if value == READER_FAKE:
        return fake_cloud_reader, READER_FAKE
    if value in (READER_CLAUDE, READER_OPUS):
        model = os.environ.get(CLAUDE_MODEL_ENV, "").strip() or DEFAULT_CLAUDE_MODEL
        return make_claude_reader(model=model, timeout=OPUS_TIMEOUT), f"{READER_CLAUDE} ({model})"
    if value == READER_LOCAL:
        model = os.environ.get(READER_MODEL_ENV, "").strip()
        if not model:
            raise CloudMemoError(f"{READER_ENV}=local には {READER_MODEL_ENV}（モデル名）を指定してください。")
        url = os.environ.get(READER_URL_ENV, "").strip() or DEFAULT_LOCAL_URL
        try:
            reader = make_local_reader(base_url=url, model=model, timeout=OPUS_TIMEOUT)
        except ReaderStage5Error as exc:
            raise CloudMemoError(f"{READER_URL_ENV} が使えません: {exc}") from exc
        return reader, f"{READER_LOCAL} ({model})"
    raise CloudMemoError(f"{READER_ENV} の値が正しくありません（空・fake・claude・opus・local のどれか）。")


def reader_is_local(reader_name: str) -> bool:
    """表示用の名前が手元の読む役（``local``）のものか."""
    return str(reader_name).startswith(READER_LOCAL)


def memo_path(paths: Any, mode: str, customer_id: str, record_id: str) -> Path:
    """5欄の保存先（確認済みの記録と同じ保存先の ``cloud-memos/``。番号の検査も同じ）."""
    root = approval_path(paths, mode, customer_id, record_id).parents[2]
    return root / MEMO_DIR / customer_id / f"{record_id.strip()}.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def make_memo(
    masked: MaskedR3,
    *,
    approval_file: Path,
    reader: Reader,
    meeting_date: date | None = None,
    created_at: datetime | None = None,
    clock: Callable[[], datetime] = _now,
) -> dict[str, Any]:
    """確認済みの中身だけを読む役に渡し、5欄を作って返す（保存は ``save_memo``）.

    ``approval_file`` の確認済み（全文を読んだ・送る中身全体の指紋が同じ）でなければ、読む役を
    呼ばずに断る。読む役の失敗は種類と文だけの ``CloudMemoError``、返事の形崩れは空の5欄（理由つき）。
    """
    if not isinstance(masked, MaskedR3):
        raise CloudMemoError("送る中身が MaskedR3 ではありません。")
    if not callable(reader):
        raise CloudMemoError("読む役が呼び出し可能ではありません。")
    try:
        approved = check_payload_approval(Path(approval_file), masked)
    except SendReviewError as exc:
        raise CloudMemoError(str(exc)) from exc
    if not approved:
        raise CloudMemoError("この内容は確認済みではありません。送る前の確認で、全文を読んで確認済みにしてください。")

    prompt = build_reader_prompt_r3(masked)
    try:
        reply = reader(prompt)
    except Exception as exc:  # 中身は含めず、例外の種類と文だけ
        raise CloudMemoError(f"読む役が失敗しました: {type(exc).__name__}: {exc}") from exc

    model = str(getattr(reply, "model", "") or "")
    raw = getattr(reply, "text", None)
    reason = None
    try:
        parsed = parse_reader_reply_r3(raw)
    except ReaderR3ShapeError:
        assembled = empty_assembly(REASON_SHAPE)
        reason = REASON_SHAPE
    else:
        assembled = assemble_r3(masked, parsed, meeting_date=meeting_date)
    stamp = created_at if created_at is not None else clock()
    return {
        "version": MEMO_VERSION,
        "shared": assembled["shared"],
        "columns": assembled["columns"],
        "counts": assembled["counts"],
        "field_counts": assembled["field_counts"],
        "reason": reason,
        "reader_model": model,
        "payload_sha256": payload_digest(masked),
        "created_at": stamp.isoformat(),
    }


def _write_private(path: Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    try:
        for directory in (target.parent.parent, target.parent):  # cloud-memos/ とお客様のフォルダ
            _ensure_private_dir(directory)
    except MacAppError as exc:
        raise CloudMemoError(str(exc)) from exc
    tmp = target.with_name(f".{target.name}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, target)
    os.chmod(target, 0o600)


def save_memo(path: Path, memo: Mapping[str, Any]) -> None:
    """5欄を書く（0600・フォルダ0700）."""
    if type(memo) is not dict and not isinstance(memo, Mapping):
        raise CloudMemoError("5欄の形が正しくありません。")
    _write_private(path, dict(memo))


def load_memo(path: Path) -> dict[str, Any] | None:
    """保存した5欄を読む（無ければ None）."""
    target = Path(path)
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CloudMemoError("保存した5欄を読めません。") from exc
    if type(data) is not dict or data.get("version") != MEMO_VERSION or type(data.get("shared")) is not str:
        raise CloudMemoError("保存した5欄の形が正しくありません。")
    return data


_HEADING_RE = re.compile(r"^【(.+)】\s*$")
_BULLET_RE = re.compile(r"^(?:-|・)\s*")


def split_memo_sections(text: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """``shared``（人が直した版を含む）を、``【欄名】`` の見出しごとに分ける（画面用）.

    返すのは ``(最初の見出しより前の行, [(欄名, 項目の行)])``。項目の行は、行頭の ``- ``・``・`` を
    外した空でない行（「なし」だけの行は項目に数えない）。見出しが1つも無い・読めない文字でも
    例外にせず、そのまま前置きの行として返す（前に保存した版や、人が自由に直した版も出せるように）。
    本文は Markdown として解釈しない前提で、文字は書き換えない。
    """
    preface: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    for raw in str(text).splitlines():
        line = raw.rstrip()
        heading = _HEADING_RE.match(line)
        if heading:
            sections.append((heading.group(1), []))
            continue
        body = _BULLET_RE.sub("", line.strip())
        if not body:
            continue
        if sections:
            if body != "なし":
                sections[-1][1].append(body)
        else:
            preface.append(line.strip())
    return preface, sections


def save_edited(path: Path, text: str) -> dict[str, Any]:
    """人が直した版を ``edited`` に書く（ほかの項目は変えない）。5欄が無ければ断る."""
    if type(text) is not str:
        raise CloudMemoError("直した文は文字列で渡してください。")
    memo = load_memo(path)
    if memo is None:
        raise CloudMemoError("直す元の5欄がありません。")
    memo["edited"] = text
    _write_private(path, memo)
    return memo


# ---------------------------------------------------------------- 共有フォルダに渡す（share/）

SHARE_MEMO_DIR = "進行メモ"
SHARE_README = "README.md"
_SHARE_README_TEXT = """# 進行メモ（書庫番の5欄）

書庫番が作った、打ち合わせごとの進行メモです。ほかのエージェントが読んでよいのは、このフォルダ（`share/`）だけです。

- 場所: `進行メモ/<共有用の呼び名>/<打ち合わせの日>_<記録の番号>.md`。1つの打ち合わせに1ファイル。直して渡し直すと上書きされます。
- 中身: 人が送る前の確認で全文を読んで確認済みにした、伏せたあとの文だけから作った6つの欄です。
  - こちらから話した道筋・提案（ステップの順。「最後:」は最後に目指す姿）
  - 困りごと／頼まれていること／こちらがやること／決まったこと／まだ決まっていないこと
- 名前・社名・連絡先・金額・数などは〈名前〉〈社名〉〈数字〉のような札に置き換えてあります。札の中身を推し量ったり、元の記録を探したりしないでください。元の記録はこの道具を動かしている PC の中だけにあり、読めないように止めてあります。
- 行の頭の [お客様]・[こちら]・[不明] は、その文を話した側の推定です（音声からの聞き分けではありません）。
- 各行は打ち合わせの言葉をそのまま引いたものなので、話し言葉のままです。
- 「人が直した版: はい」のファイルは、人が直したものです。
"""


def share_memo_path(share_dir: Path, label: str, record_id: str, meeting_date: date | None) -> Path:
    """渡す先 ``<share>/進行メモ/<呼び名>/<日>_<記録の番号の末尾8字>.md``（作らない）."""
    from .progress_status import ProgressStatusError, validate_share_label

    try:
        checked = validate_share_label(label)
    except ProgressStatusError as exc:
        raise CloudMemoError(str(exc)) from exc
    rid = str(record_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", rid):
        raise CloudMemoError("記録の番号が正しくありません。")
    day = meeting_date.isoformat() if isinstance(meeting_date, date) else "日付なし"
    return Path(share_dir) / SHARE_MEMO_DIR / checked / f"{day}_{rid[-8:]}.md"


def render_share_memo(label: str, memo: Mapping[str, Any], *, meeting_date: date | None, shared_at: datetime) -> str:
    """渡すファイルの中身（見出し・作った日時・読む役・直した版か・5欄の本文）."""
    edited = memo.get("edited")
    is_edited = type(edited) is str and edited != memo.get("shared")
    body = edited if is_edited else memo.get("shared")
    if type(body) is not str:
        raise CloudMemoError("5欄の形が正しくありません。")
    day = meeting_date.isoformat() if isinstance(meeting_date, date) else "日付なし"
    lines = [
        f"# {label} の進行メモ（{day} の打ち合わせ）",
        "",
        f"- 渡した日時: {shared_at.isoformat(timespec='minutes')}",
        f"- 5欄を作った日時: {str(memo.get('created_at') or '不明')[:16]}",
        f"- 読む役: {memo.get('reader_model') or '不明'}",
        f"- 人が直した版: {'はい' if is_edited else 'いいえ'}",
        "- 読み方: 同じフォルダの上の README.md",
        "",
        body.rstrip(),
        "",
    ]
    return "\n".join(lines)


def share_memo(
    share_dir: Path,
    label: str,
    memo: Mapping[str, Any],
    *,
    record_id: str,
    meeting_date: date | None,
    shared_at: datetime,
) -> Path:
    """確認済みの5欄（直した版があればそちら）を ``share/進行メモ/`` に書く（フォルダ0700・ファイル0600）.

    呼ぶ側が、確認済みで、5欄が今の送る中身から作ったものであることを確かめてから呼ぶ。
    """
    from .progress_status import _write_private_text

    target = share_memo_path(share_dir, label, record_id, meeting_date)
    text = render_share_memo(label, memo, meeting_date=meeting_date, shared_at=shared_at)
    root = Path(share_dir)
    try:
        for directory in (root, root / SHARE_MEMO_DIR, target.parent):
            _ensure_private_dir(directory)
    except MacAppError as exc:
        raise CloudMemoError(str(exc)) from exc
    readme = root / SHARE_MEMO_DIR / SHARE_README
    if not readme.is_file() or readme.read_text(encoding="utf-8") != _SHARE_README_TEXT:
        _write_private_text(readme.parent, readme, _SHARE_README_TEXT)
    _write_private_text(target.parent, target, text)
    return target


# ---------------------------------------------------------------- 読む役の月の回数（普段使いの上限）

USAGE_FILE = "reader-usage.json"
MONTHLY_LIMIT_ENV = "SHOKOBAN_READER_MONTHLY_LIMIT"
# 普段使いで足りるように多めにした既定（読む役の呼び出しは1か月200回まで）
DEFAULT_MONTHLY_LIMIT = 200


def monthly_limit() -> int:
    """1か月の上限（環境変数で変えられる。正の整数でなければ既定の200）."""
    raw = os.environ.get(MONTHLY_LIMIT_ENV, "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_MONTHLY_LIMIT


def usage_path(paths: Any, mode: str) -> Path:
    """回数の記録 ``<保存先>/cloud-memos/reader-usage.json``（作らない）."""
    from .mac_app import store_root_for_mode

    return store_root_for_mode(paths, mode) / MEMO_DIR / USAGE_FILE


def load_usage(path: Path) -> dict[str, int]:
    """月（YYYY-MM）→ 呼んだ回数。無ければ空。壊れていたら止める（数え直せないまま呼ばない）."""
    target = Path(path)
    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CloudMemoError("読む役の回数の記録を読めません。") from exc
    if type(data) is not dict or not all(type(k) is str and type(v) is int and v >= 0 for k, v in data.items()):
        raise CloudMemoError("読む役の回数の記録の形が正しくありません。")
    return data


def reserve_reader_call(path: Path, month: str, limit: int) -> int:
    """呼ぶ前に1回分を数える（失敗しても1回と数える。お金は呼んだ時点でかかるため）.

    今月がもう上限なら数えずに ``CloudMemoError``。返すのは数えたあとの今月の回数。
    """
    from .progress_status import _write_private_text

    if not re.fullmatch(r"\d{4}-\d{2}", str(month)):
        raise CloudMemoError("月の形が正しくありません。")
    usage = load_usage(path)
    used = usage.get(month, 0)
    if used >= limit:
        raise CloudMemoError(f"今月の読む役の回数が上限（{limit}回）に届いています。設定の上限（SHOKOBAN_READER_MONTHLY_LIMIT）を確かめてください。")
    usage[month] = used + 1
    target = Path(path)
    try:
        _ensure_private_dir(target.parent)
    except MacAppError as exc:
        raise CloudMemoError(str(exc)) from exc
    _write_private_text(target.parent, target, json.dumps(usage, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return used + 1
