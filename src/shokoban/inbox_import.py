"""受け取りフォルダのテキストを、企業ごとの案件ファイルの下書きとして取り込む.

受け取りフォルダ（inbox）に届いた「YYYY-MM-DD 名前.txt|.md」だけを対象にする。
取り込んだものは下書き（draft）として保存し、確定はしない。確定は人が画面で行う。

取り込む本文の sha256 を取り込み元として残し、同じ内容を二度取り込まない
（移動だけ失敗した場合の再実行でも下書きを増やさない）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .customer_store import (
    CustomerStore,
    CustomerStoreError,
    RecordValidationError,
    normalize_customer_name,
)

INBOX_FILE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}) (.+)\.(txt|md)$")
ARCHIVE_DIR_NAME = "取り込み済み"
MAX_FILE_BYTES = 5 * 1024 * 1024
SUMMARY_MAX_CHARS = 120
JST = timezone(timedelta(hours=9), "JST")

STATUS_IMPORTED = "imported"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"

JOURNAL_NAME = ".import-journal.jsonl"
JOURNAL_WRITE_NOTE = (
    "取り込みの記録帳に書けませんでした（同じ本文かは本文の照合だけで見ます）"
)

SOURCE_REF_PREFIX = "inbox:"
SOURCE_HASH_PREFIX = "inbox-sha256:"
SOURCE_REF_NOTE = (
    "CustomerStore は source_refs に不透明ID以外を許可しないため、"
    "取り込み元（inbox:ファイル名）を案件ファイルに記録できませんでした"
)
ALREADY_IMPORTED_REASON = "前回取り込み済みの内容です（移動だけやり直しました）"
NAME_FILTER_REASON = "架空データの期間は「架空顧客」で始まる名前だけ取り込みます"

__all__ = [
    "ALREADY_IMPORTED_REASON",
    "ARCHIVE_DIR_NAME",
    "JOURNAL_NAME",
    "MAX_FILE_BYTES",
    "STATUS_FAILED",
    "STATUS_IMPORTED",
    "STATUS_SKIPPED",
    "SUMMARY_MAX_CHARS",
    "ImportResult",
    "InboxImportError",
    "import_inbox",
]


class InboxImportError(RuntimeError):
    """Raised when the inbox or the customer store cannot be used at all."""


@dataclass(frozen=True, slots=True)
class ImportResult:
    """受け取りフォルダの1件分の結果."""

    source_name: str
    status: str
    reason: str = ""
    customer_id: str | None = None
    meeting_id: str | None = None
    archived_to: Path | None = None
    checked_at: str = ""
    notes: tuple[str, ...] = ()


def import_inbox(
    store: CustomerStore,
    inbox: Path,
    *,
    now: datetime,
    accept_name: Callable[[str], bool] | None = None,
) -> list[ImportResult]:
    """受け取りフォルダを1回走査し、対象のテキストを下書きとして取り込む.

    取り込めたファイルだけを ``inbox/取り込み済み/`` へ移す。移せなかったものは
    そのまま残す。同じ本文を取り込み済みなら新しく下書きを作らず、移動だけを
    やり直す（移動失敗の後の再実行でも下書きは増えない）。

    ``accept_name`` を渡すと、名前部分（日付の後ろ・拡張子の前）で False を
    返したファイルは、動かさず読まずに見送る（理由は ``NAME_FILTER_REASON``）。
    """
    root = Path(inbox)
    if root.is_symlink() or not root.is_dir():
        raise InboxImportError("受け取りフォルダが見つかりません")
    try:
        store.validate_schema_manifest()
    except Exception as exc:
        raise InboxImportError("顧客ストアが初期化されていません") from exc

    checked_at = _checked_at(now)
    results: list[ImportResult] = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if entry.name in (ARCHIVE_DIR_NAME, JOURNAL_NAME):
            continue
        results.append(_import_entry(store, root, entry, checked_at, accept_name))
    return results


def _checked_at(now: datetime) -> str:
    if now.tzinfo is None:
        now = now.replace(tzinfo=JST)
    return now.astimezone(JST).isoformat()


def _import_entry(
    store: CustomerStore,
    inbox: Path,
    entry: Path,
    checked_at: str,
    accept_name: Callable[[str], bool] | None = None,
) -> ImportResult:
    name = entry.name
    reason, day, customer_name = _inspect_entry(entry)
    if reason:
        return ImportResult(source_name=name, status=STATUS_SKIPPED, reason=reason, checked_at=checked_at)
    assert day is not None

    if accept_name is not None and not accept_name(customer_name):
        return ImportResult(
            source_name=name,
            status=STATUS_SKIPPED,
            reason=NAME_FILTER_REASON,
            checked_at=checked_at,
        )

    try:
        raw = entry.read_bytes()
    except OSError:
        return ImportResult(
            source_name=name,
            status=STATUS_FAILED,
            reason="本文を読み込めませんでした",
            checked_at=checked_at,
        )
    try:
        text = _decode_text(raw)
    except UnicodeDecodeError:
        return ImportResult(
            source_name=name,
            status=STATUS_FAILED,
            reason="UTF-8のテキストとして読めません",
            checked_at=checked_at,
        )

    try:
        customer_id = store.resolve_customer(customer_name)
        if customer_id is None:
            customer_id = store.create_customer(display_name=customer_name)["customer_id"]
    except (CustomerStoreError, RecordValidationError):
        return ImportResult(
            source_name=name,
            status=STATUS_FAILED,
            reason="顧客を特定または作成できませんでした",
            checked_at=checked_at,
        )

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        previous = _journal_previous(inbox, store, customer_id, digest)
        if previous is None:
            previous = _find_previous_import(store, customer_id, digest)
    except (CustomerStoreError, RecordValidationError):
        return ImportResult(
            source_name=name,
            status=STATUS_FAILED,
            reason="前回の取り込みを確認できませんでした",
            customer_id=customer_id,
            checked_at=checked_at,
        )

    if previous is not None:
        # 前回の下書き（または確定済み）が同じ入力で残っている。新しく作らず、
        # 失敗した移動だけをやり直す。
        previous_id = previous.get("record_id")
        if previous.get("source") != "journal" and type(previous_id) is str:
            # 記録帳より前の取り込みを本文の照合で拾った時は、以後の編集に備えて
            # 記録帳にも残す（書けなくても今までどおり本文の照合で足りる）。
            _append_journal(
                inbox,
                digest=digest,
                source_name=name,
                customer_id=customer_id,
                meeting_id=previous_id,
                at=checked_at,
            )
        try:
            archived_to = _move_to_archive(entry, inbox / ARCHIVE_DIR_NAME, name, raw)
        except OSError:
            return ImportResult(
                source_name=name,
                status=STATUS_FAILED,
                reason=(
                    "前回取り込み済みの内容です"
                    f"（移動だけやり直しましたが、取り込み済みへ移動できませんでした: 下書き {previous_id}）"
                ),
                customer_id=customer_id,
                meeting_id=previous_id,
                checked_at=checked_at,
            )
        return ImportResult(
            source_name=name,
            status=STATUS_IMPORTED,
            reason=ALREADY_IMPORTED_REASON,
            customer_id=customer_id,
            meeting_id=previous_id,
            archived_to=archived_to,
            checked_at=checked_at,
        )

    occurred_at = datetime.combine(day, time(0, 0), tzinfo=JST).isoformat()
    try:
        meeting, notes = _create_draft(
            store,
            customer_id,
            filename=name,
            digest=digest,
            occurred_at=occurred_at,
            title=f"{day.isoformat()} 取り込み",
            summary=_summarize(text),
            details=text,
        )
    except (CustomerStoreError, RecordValidationError):
        return ImportResult(
            source_name=name,
            status=STATUS_FAILED,
            reason="下書きの作成に失敗しました",
            customer_id=customer_id,
            checked_at=checked_at,
        )

    meeting_id = meeting["record_id"]
    notes = list(notes)
    # 下書きを作った直後・移動の前に、本文と切り離した不変の取り込みIDを残す。
    # 以後、本文（details）を直しても同じ入力は新しく作らない。
    if not _append_journal(
        inbox,
        digest=digest,
        source_name=name,
        customer_id=customer_id,
        meeting_id=meeting_id,
        at=checked_at,
    ):
        notes.append(JOURNAL_WRITE_NOTE)
    try:
        archived_to = _move_to_archive(entry, inbox / ARCHIVE_DIR_NAME, name, raw)
    except OSError:
        return ImportResult(
            source_name=name,
            status=STATUS_FAILED,
            reason=f"下書きは作成しましたが取り込み済みへ移動できませんでした（下書き: {meeting_id}）",
            customer_id=customer_id,
            meeting_id=meeting_id,
            checked_at=checked_at,
            notes=tuple(notes),
        )

    return ImportResult(
        source_name=name,
        status=STATUS_IMPORTED,
        customer_id=customer_id,
        meeting_id=meeting_id,
        archived_to=archived_to,
        checked_at=checked_at,
        notes=tuple(notes),
    )


def _inspect_entry(entry: Path) -> tuple[str, date | None, str]:
    """対象外なら理由を、対象なら日付と顧客名を返す."""
    name = entry.name
    if entry.is_symlink():
        return "シンボリックリンクは対象外です", None, ""
    if entry.is_dir():
        return "フォルダは対象外です", None, ""
    if name.startswith("."):
        return "「.」で始まるファイルは対象外です", None, ""
    if name.endswith(".part"):
        return "書きかけのファイル（.part）は対象外です", None, ""
    matched = INBOX_FILE_PATTERN.fullmatch(name)
    if matched is None:
        return "ファイル名が「YYYY-MM-DD 名前.txt」の形になっていません", None, ""
    try:
        day = date.fromisoformat(matched.group(1))
    except ValueError:
        return "ファイル名の日付が正しくありません", None, ""
    customer_name = matched.group(2).strip()
    if not normalize_customer_name(customer_name):
        return "ファイル名の顧客名が空です", None, ""
    if not entry.is_file():
        return "通常ファイルではありません", None, ""
    try:
        size = entry.stat().st_size
    except OSError:
        return "ファイルの情報を取得できません", None, ""
    if size > MAX_FILE_BYTES:
        return "5MBを超えるファイルは対象外です", None, ""
    return "", day, customer_name


def _decode_text(raw: bytes) -> str:
    """UTF-8 として読み、改行を LF にそろえる."""
    text = raw.decode("utf-8")
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _summarize(text: str) -> str:
    return text.strip()[:SUMMARY_MAX_CHARS]


def _journal_path(inbox: Path) -> Path:
    """取り込みの記録帳（``.`` 始まりなので取り込み対象にならない）."""
    return Path(inbox) / JOURNAL_NAME


def _read_journal(inbox: Path) -> list[dict[str, str]]:
    """記録帳の読める行だけを返す（無い・壊れた行・形の違う行は飛ばす）."""
    path = _journal_path(inbox)
    try:
        info = os.lstat(path)
    except OSError:
        return []
    if not stat.S_ISREG(info.st_mode):
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    entries: list[dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if type(payload) is not dict:
            continue
        digest = payload.get("sha256")
        customer_id = payload.get("customer_id")
        meeting_id = payload.get("meeting_id")
        if not (
            type(digest) is str
            and digest
            and type(customer_id) is str
            and customer_id
            and type(meeting_id) is str
            and meeting_id
        ):
            continue
        source_name = payload.get("source_name")
        at = payload.get("at")
        entries.append(
            {
                "sha256": digest,
                "source_name": source_name if type(source_name) is str else "",
                "customer_id": customer_id,
                "meeting_id": meeting_id,
                "at": at if type(at) is str else "",
            }
        )
    return entries


def _append_journal(
    inbox: Path,
    *,
    digest: str,
    source_name: str,
    customer_id: str,
    meeting_id: str,
    at: str,
) -> bool:
    """記録帳に1行追記して fsync する（0600・追記のみ）。書けたら True."""
    path = _journal_path(inbox)
    payload = {
        "sha256": digest,
        "source_name": source_name,
        "customer_id": customer_id,
        "meeting_id": meeting_id,
        "at": at,
    }
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    data = line.encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError:
        return False
    try:
        os.fchmod(fd, 0o600)
        if os.write(fd, data) != len(data):
            return False
        os.fsync(fd)
    except OSError:
        return False
    finally:
        os.close(fd)
    return True


def _journal_previous(
    inbox: Path, store: CustomerStore, customer_id: str, digest: str
) -> dict[str, Any] | None:
    """記録帳にある同じ入力（sha256）の取り込み先を返す.

    その ``meeting_id`` がまだ店にあれば（下書き・確定・本文を直した後でも）
    同じ入力とみなす。店から消えていれば、本文の照合に任せる（None）。
    """
    candidates = [
        entry
        for entry in _read_journal(inbox)
        if entry["sha256"] == digest and entry["customer_id"] == customer_id
    ]
    if not candidates:
        return None
    alive = {
        meeting.get("record_id")
        for meeting in store.list_meetings(customer_id)
    }
    for entry in reversed(candidates):
        if entry["meeting_id"] in alive:
            return {"record_id": entry["meeting_id"], "source": "journal"}
    return None


def _find_previous_import(
    store: CustomerStore, customer_id: str, digest: str
) -> dict[str, Any] | None:
    """同じ内容を取り込み済みの打ち合わせ（下書き・確定とも）を探す.

    第一は ``source_refs`` の ``inbox-sha256:<hex>``。ただし CustomerStore は
    不透明ID以外の source_refs を拒否するため、保存できなかった場合は本文
    （details）の sha256 を計算し直して照合する。どちらも同じ本文に対する
    同じハッシュなので、移動失敗からの再実行で下書きを増やさない。
    """
    marker = f"{SOURCE_HASH_PREFIX}{digest}"
    for meeting in store.list_meetings(customer_id):
        refs = meeting.get("source_refs")
        if type(refs) is list and marker in refs:
            return meeting
        details = meeting.get("details")
        if type(details) is str and hashlib.sha256(details.encode("utf-8")).hexdigest() == digest:
            return meeting
    return None


def _source_refs(filename: str, digest: str) -> list[str]:
    """下書きに残す取り込み元（ファイル名と本文の sha256）."""
    return [f"{SOURCE_REF_PREFIX}{filename}", f"{SOURCE_HASH_PREFIX}{digest}"]


def _create_draft(
    store: CustomerStore,
    customer_id: str,
    *,
    filename: str,
    digest: str,
    occurred_at: str,
    title: str,
    summary: str,
    details: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """下書きを作る。取り込み元を source_refs に残せたかも返す."""
    try:
        meeting = store.create_meeting_draft(
            customer_id,
            occurred_at=occurred_at,
            title=title,
            summary=summary,
            details=details,
            source_refs=_source_refs(filename, digest),
        )
    except RecordValidationError as exc:
        if "source_refs" not in str(exc):
            raise
        meeting = store.create_meeting_draft(
            customer_id,
            occurred_at=occurred_at,
            title=title,
            summary=summary,
            details=details,
            source_refs=[],
        )
        return meeting, (SOURCE_REF_NOTE,)
    return meeting, ()


def _move_to_archive(source: Path, archive: Path, filename: str, data: bytes) -> Path:
    """``取り込み済み`` へ移す。同名があれば「 (2)」などを付ける."""
    if archive.is_symlink():
        raise OSError("取り込み済みフォルダがシンボリックリンクです")
    archive.mkdir(mode=0o700, parents=True, exist_ok=True)
    stem, suffix = os.path.splitext(filename)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for index in range(1, 1000):
        candidate = archive / (filename if index == 1 else f"{stem} ({index}){suffix}")
        try:
            fd = os.open(candidate, flags, 0o600)
        except FileExistsError:
            continue
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            candidate.unlink(missing_ok=True)
            raise
        try:
            source.unlink()
        except OSError:
            candidate.unlink(missing_ok=True)
            raise
        return candidate
    raise OSError("取り込み済みフォルダに空き名が見つかりません")
