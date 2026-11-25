"""書庫番の画面（実用版）が使う保存先・手元AI・下書きの一覧.

``SHOKOBAN_WORK_DIR`` があるときだけ実用版として動く。保存先は
作業フォルダの決まった場所（store-synthetic / inbox / outbox / outbox-held /
runtime.json）に固定し、一時フォルダへは逃げない。

このモジュールは Streamlit に依存しない。画面はここで作った値を表示するだけ。
"""

from __future__ import annotations

import json
import os
import re
import secrets
import stat
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from . import llm_runtime
from .customer_store import CustomerStore
from .llm_runtime import LLMRuntimeError, RuntimeConfig
from .local_llm import LocalLLMSettings
from .progress_status import (
    EVIDENCE_MAX_CHARS,
    FIELDS,
    SOURCES,
    ProgressStatus,
    ProgressStatusError,
    validate_share_label,
    validate_status,
)

WORK_DIR_ENV = "SHOKOBAN_WORK_DIR"
DATA_MODE_ENV = "SHOKOBAN_DATA_MODE"
DATA_MODES = ("real", "synthetic")
SYNTHETIC_NAME_PREFIX = "架空顧客"
LOCAL_LLM_PROVIDER = "lmstudio"

__all__ = [
    "DATA_MODES",
    "DATA_MODE_ENV",
    "LOCAL_LLM_PROVIDER",
    "SYNTHETIC_NAME_PREFIX",
    "WORK_DIR_ENV",
    "MacAppError",
    "MacAppPaths",
    "ensure_inbox",
    "list_draft_meetings",
    "load_share_labels",
    "load_status_record",
    "mac_app_paths_from_env",
    "prepare_local_llm",
    "runtime_status",
    "save_share_label",
    "save_status_record",
    "share_dir_for_mode",
    "status_local_dir_for_mode",
    "status_record_path",
    "store_root_for_mode",
    "synthetic_inbox_filter",
]


class MacAppError(RuntimeError):
    """実用版の保存先を安全に決められないときに送出する."""


@dataclass(frozen=True)
class MacAppPaths:
    """実用版の保存先（すべて絶対パス）."""

    work_dir: Path
    store_root: Path
    inbox: Path
    outbox: Path
    held: Path
    runtime_config: Path


def mac_app_paths_from_env(
    environ: Mapping[str, str] = os.environ,
) -> MacAppPaths | None:
    """``SHOKOBAN_WORK_DIR`` から保存先を決める（無ければ None）.

    空文字・相対パス・存在しない場所・ディレクトリでない場所・
    シンボリックリンクは受け付けない（一時フォルダに逃げない）。
    """
    raw = environ.get(WORK_DIR_ENV)
    if raw is None:
        return None
    if type(raw) is not str or not raw.strip():
        raise MacAppError(f"{WORK_DIR_ENV} が空です")

    work_dir = Path(raw)
    if not work_dir.is_absolute():
        raise MacAppError(f"{WORK_DIR_ENV} は絶対パスで指定してください")
    if work_dir.is_symlink():
        raise MacAppError(f"{WORK_DIR_ENV} にシンボリックリンクは使えません")
    if not work_dir.exists():
        raise MacAppError(f"{WORK_DIR_ENV} のフォルダがありません: {work_dir}")
    if not work_dir.is_dir():
        raise MacAppError(f"{WORK_DIR_ENV} がディレクトリではありません: {work_dir}")

    return MacAppPaths(
        work_dir=work_dir,
        store_root=work_dir / "store-synthetic",
        inbox=work_dir / "inbox",
        outbox=work_dir / "outbox",
        held=work_dir / "outbox-held",
        runtime_config=work_dir / "runtime.json",
    )


def synthetic_inbox_filter(name_part: str) -> bool:
    """受け取りフォルダの名前部分が「架空顧客」で始まる時だけ True."""
    return type(name_part) is str and name_part.startswith(SYNTHETIC_NAME_PREFIX)


def ensure_inbox(paths: MacAppPaths) -> Path:
    """受け取りフォルダ（inbox）が無ければ 0700 で作って返す.

    初めて使う人が空の作業フォルダから始められるようにするため。シンボリックリンク・
    フォルダでないものが在れば ``MacAppError``（置き換えない）。
    """
    _ensure_private_dir(paths.inbox)
    return paths.inbox


def store_root_for_mode(paths: MacAppPaths, mode: str) -> Path:
    """データモードに応じた保存先を返す（real → store、synthetic → store-synthetic）.

    ``SHOKOBAN_DATA_MODE`` が real か synthetic 以外なら ``MacAppError``。
    """
    if not isinstance(paths, MacAppPaths):
        raise MacAppError("保存先の指定が不正です")
    if mode == "real":
        return paths.work_dir / "store"
    if mode == "synthetic":
        return paths.store_root
    raise MacAppError(f"データモードが不正です: {mode!r}")


def prepare_local_llm(
    cfg: RuntimeConfig,
    alias: str | None,
    *,
    ensure: Callable[[RuntimeConfig, str | None], str] = llm_runtime.ensure_running,
) -> LocalLLMSettings:
    """手元AIを（必要なら起こして）使い、その別名の設定を返す.

    LLMRuntimeError はそのまま上げる。画面がメッセージを表示して止まる。
    """
    if not isinstance(cfg, RuntimeConfig):
        raise LLMRuntimeError("手元AIの設定が不正です")
    model = ensure(cfg, alias)
    return LocalLLMSettings(provider=LOCAL_LLM_PROVIDER, model=model)


def _read_alias(alias_path: Path) -> str | None:
    """状態ファイルの別名を読む（通常ファイル以外・読めない場合は None）."""
    try:
        info = os.lstat(alias_path)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    try:
        alias = alias_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return alias or None


def runtime_status(cfg: RuntimeConfig) -> dict[str, Any]:
    """手元AIの状態（実プロセスは起動しない）."""
    if not isinstance(cfg, RuntimeConfig):
        raise LLMRuntimeError("手元AIの設定が不正です")
    ledger = llm_runtime.current_ledger(cfg)
    return {
        "running": ledger is not None,
        "alias": ledger.alias if ledger is not None else None,
        "idle_seconds": llm_runtime.idle_seconds(cfg),
    }


def list_draft_meetings(store: CustomerStore, *, include_confirmed: bool = False) -> list[dict[str, Any]]:
    """全顧客の下書き（draft）の打ち合わせを、occurred_at の新しい順で返す.

    ``include_confirmed`` が真なら確定済み（confirmed）も入れる（本番データで確定済みの打ち合わせも見直せるように。記録は読むだけで変えない）。
    """
    statuses = ("draft", "confirmed") if include_confirmed else ("draft",)
    drafts: list[dict[str, Any]] = []
    for customer in store.list_customers():
        for meeting in store.list_meetings(customer["customer_id"]):
            if meeting.get("meeting_status") not in statuses:
                continue
            drafts.append(
                {
                    "customer_id": customer["customer_id"],
                    "display_name": customer["display_name"],
                    "record_id": meeting["record_id"],
                    "revision": meeting["revision"],
                    "occurred_at": meeting["occurred_at"],
                    "title": meeting["title"],
                    "summary": meeting["summary"],
                    "details": meeting["details"],
                    "meeting_status": meeting.get("meeting_status"),
                }
            )
    drafts.sort(key=lambda draft: str(draft["occurred_at"]), reverse=True)
    return drafts


_CUSTOMER_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,80}")


def share_dir_for_mode(paths: MacAppPaths, mode: str) -> Path:
    """共有ファイルの置き場（real → share、synthetic → share-synthetic）."""
    if not isinstance(paths, MacAppPaths):
        raise MacAppError("保存先の指定が不正です")
    if mode == "real":
        return paths.work_dir / "share"
    if mode == "synthetic":
        return paths.work_dir / "share-synthetic"
    raise MacAppError(f"データモードが不正です: {mode!r}")


def status_local_dir_for_mode(paths: MacAppPaths, mode: str) -> Path:
    """根拠と修正前の値を置く場所（ほかのエージェントは読まない）.

    real → ``status-local``、synthetic → ``status-local-synthetic``。
    """
    if not isinstance(paths, MacAppPaths):
        raise MacAppError("保存先の指定が不正です")
    if mode == "real":
        return paths.work_dir / "status-local"
    if mode == "synthetic":
        return paths.work_dir / "status-local-synthetic"
    raise MacAppError(f"データモードが不正です: {mode!r}")


def _check_customer_id(customer_id: Any) -> bool:
    return (
        type(customer_id) is str
        and _CUSTOMER_ID_RE.fullmatch(customer_id) is not None
    )


def status_record_path(
    paths: MacAppPaths, mode: str, customer_id: str
) -> Path:
    """``<status_local>/<customer_id>.json`` の場所を返す."""
    if not isinstance(paths, MacAppPaths):
        raise MacAppError("保存先の指定が不正です")
    if not _check_customer_id(customer_id):
        raise MacAppError("お客様の識別子が不正です")
    return status_local_dir_for_mode(paths, mode) / f"{customer_id}.json"


def _ensure_private_dir(directory: Path) -> None:
    if directory.is_symlink():
        raise MacAppError("保存先のフォルダがシンボリックリンクです")
    if directory.exists():
        if not directory.is_dir():
            raise MacAppError("保存先がディレクトリではありません")
    else:
        directory.mkdir(parents=True, mode=0o700)
    os.chmod(directory, 0o700)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """通常ファイルの JSON オブジェクトだけを読む（無ければ None）."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if type(payload) is dict else None


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    """一時ファイル → os.replace で 0600 の JSON を書く."""
    _ensure_private_dir(path.parent)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temp = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(text)
        os.replace(temp, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            temp.unlink()
        except OSError:
            pass
        raise


def load_share_labels(path: Path) -> dict[str, str]:
    """``share-labels.json`` を読む（customer_id → 共有用の呼び名）.

    無い・壊れている・規則に合わない呼び名は捨てる。未設定の間は共有しない。
    """
    payload = _read_json_object(Path(path))
    if payload is None:
        return {}
    labels: dict[str, str] = {}
    for customer_id, label in payload.items():
        if type(customer_id) is not str or type(label) is not str:
            continue
        try:
            labels[customer_id] = validate_share_label(label)
        except ProgressStatusError:
            continue
    return labels


def _label_key(label: str) -> str:
    """呼び名の取り違えを見つけるための正規化（NFKC＋大文字小文字を無視）."""
    return unicodedata.normalize("NFKC", label).casefold()


def save_share_label(path: Path, customer_id: str, label: str) -> None:
    """共有用の呼び名を 0600 で保存する（ほかの顧客の分は残す）.

    別のお客様が同じ呼び名（全角・大文字小文字の違いを含む）を使っていたら、
    取り違えの元なので ``MacAppError`` にして保存しない。同じお客様の付け直しは
    そのまま通す。
    """
    if not _check_customer_id(customer_id):
        raise MacAppError("お客様の識別子が不正です")
    normalized = validate_share_label(label)
    target = Path(path)
    labels = load_share_labels(target)
    key = _label_key(normalized)
    for other_id, other_label in labels.items():
        if other_id == customer_id:
            continue
        if _label_key(other_label) == key:
            raise MacAppError("その呼び名は別のお客様が使っています。")
    labels[customer_id] = normalized
    _write_private_json(target, labels)


def load_status_record(path: Path) -> dict[str, Any] | None:
    """``<customer_id>.json`` を読む（規則に合わなければ None）."""
    payload = _read_json_object(Path(path))
    if payload is None:
        return None
    raw = payload.get("status")
    if type(raw) is not dict:
        return None
    stage = raw.get("stage")
    next_task = raw.get("next_task")
    due = raw.get("due")
    # 保存形式のキーは "waiting" のまま（既にある手元の記録を読めるように）。
    mover = raw.get("waiting")
    try:
        status = validate_status(
            ProgressStatus(
                stage=stage if type(stage) is str else "",
                next_task=next_task if type(next_task) is str else "",
                due=due if type(due) is str else "",
                mover=mover if type(mover) is str else "",
            )
        )
    except ProgressStatusError:
        return None
    raw_evidence = payload.get("evidence")
    evidence = {field: "" for field in FIELDS}
    if type(raw_evidence) is dict:
        for field in FIELDS:
            value = raw_evidence.get(field)
            if type(value) is str and len(value) <= EVIDENCE_MAX_CHARS:
                evidence[field] = value
    inferred = None
    raw_inferred = payload.get("inferred")
    if type(raw_inferred) is dict:
        stage_i = raw_inferred.get("stage")
        next_task_i = raw_inferred.get("next_task")
        due_i = raw_inferred.get("due")
        mover_i = raw_inferred.get("waiting")
        try:
            inferred = validate_status(
                ProgressStatus(
                    stage=stage_i if type(stage_i) is str else "",
                    next_task=next_task_i if type(next_task_i) is str else "",
                    due=due_i if type(due_i) is str else "",
                    mover=mover_i if type(mover_i) is str else "",
                )
            )
        except ProgressStatusError:
            inferred = None
    updated = payload.get("updated")
    source = payload.get("source")
    return {
        "status": status,
        "evidence": evidence,
        "inferred": inferred,
        "raw_ok": payload.get("raw_ok") is True,
        "updated": updated if type(updated) is str else "",
        "source": source if source in SOURCES else "",
    }


def save_status_record(
    path: Path,
    status: ProgressStatus,
    evidence: Mapping[str, str],
    *,
    raw_ok: bool,
    updated: datetime,
    source: str,
    inferred: ProgressStatus | None = None,
) -> Path:
    """結果と根拠を 0600 の JSON に保存する（根拠は手元にだけ残す）.

    ``inferred`` には、人が直す前の手元AIの値を入れる。修正して共有
    しても、元の値はこのファイル（ほかのエージェントは読まない）に残る。
    """
    checked = validate_status(status)
    checked_inferred = validate_status(inferred) if inferred is not None else None
    if source not in SOURCES:
        raise MacAppError("出どころが不正です")
    if not isinstance(updated, datetime):
        raise MacAppError("更新日時が不正です")
    if not isinstance(evidence, Mapping):
        raise MacAppError("根拠が辞書ではありません")
    cleaned: dict[str, str] = {}
    for field in FIELDS:
        value = evidence.get(field, "")
        if type(value) is str and len(value) <= EVIDENCE_MAX_CHARS:
            cleaned[field] = value
        else:
            cleaned[field] = ""
    inferred_payload = (
        {
            "stage": checked_inferred.stage,
            "next_task": checked_inferred.next_task,
            "due": checked_inferred.due,
            "waiting": checked_inferred.mover,
        }
        if checked_inferred is not None
        else None
    )
    payload = {
        "status": {
            "stage": checked.stage,
            "next_task": checked.next_task,
            "due": checked.due,
            "waiting": checked.mover,
        },
        "evidence": cleaned,
        "inferred": inferred_payload,
        "raw_ok": bool(raw_ok),
        "updated": updated.isoformat(timespec="seconds"),
        "source": source,
    }
    target = Path(path)
    _write_private_json(target, payload)
    return target
