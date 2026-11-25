"""Local-only customer records and staging lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import sys
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

from .audit import AuditLogger, opaque_entity_ref


SCHEMA_VERSION = 2
CLEAN_ROOT_CONFIRMATION = "CREATE SCHEMA V2"
RETENTION_DAYS = {"immediate": 0, "7d": 7, "30d": 30, "90d": 90}
ARTIFACT_DEFAULT_RETENTION = {"audio": "immediate", "transcript": "30d"}
PROPOSAL_PAYLOAD_RETENTION = {"pending": 30, "rejected": 7, "accepted": 30}
CONSULTATION_RESPONSE_MAX_CODE_POINTS = 50_000
CONSULTATION_RESPONSE_MAX_UTF8_BYTES = 100_000
CONSULTATION_PROVIDER_KINDS = {
    "chatgpt_manual",
    "claude_manual",
    "gemini_manual",
    "copilot_manual",
    "other_external_ai_manual",
}

_ID_PREFIX = {
    "customer": "cus",
    "information": "inf",
    "menu": "mnu",
    "menu_event": "mev",
    "meeting": "mtg",
    "proposal": "prp",
    "artifact": "art",
    "consultation": "con",
}
_ID_RE = re.compile(r"^(?:cus|inf|mnu|mev|mtg|prp|art|con)_[0-9a-f]{32}$")
_CHANGE_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SOURCE_ID_RE = _ID_RE
_MENU_TRANSITIONS = {
    None: {"proposed", "active"},
    "proposed": {"active", "cancelled"},
    "active": {"paused", "completed", "cancelled"},
    "paused": {"active", "completed", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}
_RECORD_FIELDS = {
    "customer": {
        "display_name": str,
        "aliases": list,
        "occupation": str,
        "industry": str,
        "status": str,
    },
    "information": {
        "kind": str,
        "observed_at": str,
        "source_refs": list,
        "information_status": str,
        "supersedes": (str, type(None)),
        "origin": str,
    },
    "menu": {"menu_kind": str, "name": str},
    "menu_event": {
        "menu_id": str,
        "from_status": (str, type(None)),
        "to_status": str,
        "effective_at": str,
        "event_sequence": int,
        "source_refs": list,
    },
    "meeting": {
        "occurred_at": str,
        "meeting_status": str,
        "title": str,
        "summary": str,
        "details": str,
        "decisions": list,
        "next_actions": list,
        "source_refs": list,
        "confirmed_by": (str, type(None)),
        "confirmed_at": (str, type(None)),
    },
    "proposal": {
        "provider": str,
        "model": str,
        "model_identifier": str,
        "prompt_contract_version": str,
        "prompt_contract_hash": str,
        "input_digest": str,
        "source_refs": list,
        "payload_hash": str,
        "proposal_status": str,
        "reviewed_by": (str, type(None)),
        "reviewed_at": (str, type(None)),
        "human_edited": bool,
        "applied_record_id": (str, type(None)),
    },
    "consultation": {
        "provider_kind": str,
        "response_received_at": str,
        "inbound_artifact_ref": str,
        "inbound_response_digest": str,
        "outbound_brief_ref": type(None),
        "outbound_brief_digest": type(None),
        "supersedes": (str, type(None)),
        "ingest_change_id": str,
        "recorded_by_ref": str,
        "recorded_at": str,
    },
}
_COMMON_FIELDS = {
    "schema_version": int,
    "record_type": str,
    "record_id": str,
    "customer_id": str,
    "revision": int,
    "created_at": str,
    "updated_at": str,
    "record_digest": str,
}


class CustomerStoreError(RuntimeError):
    """Raised when a customer record operation fails closed."""


class RecordValidationError(CustomerStoreError):
    pass


class AliasConflictError(CustomerStoreError):
    pass


class RevisionConflictError(CustomerStoreError):
    pass


def _local_actor() -> str:
    if os.name == "nt":
        import ctypes

        size = ctypes.c_ulong(256)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not ctypes.windll.advapi32.GetUserNameW(buffer, ctypes.byref(size)):
            raise CustomerStoreError("local OS identity is unavailable")
        actor = buffer.value.strip()
    else:
        import pwd

        actor = pwd.getpwuid(os.geteuid()).pw_name.strip()
    if not actor:
        raise CustomerStoreError("local OS identity is unavailable")
    return actor


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_record_id(record_type: str) -> str:
    try:
        prefix = _ID_PREFIX[record_type]
    except KeyError as exc:
        raise RecordValidationError("unknown record type") from exc
    return f"{prefix}_{uuid.uuid4().hex}"


def normalize_customer_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return " ".join(normalized.split())


def normalize_consultation_response_text(value: str) -> str:
    if type(value) is not str:
        raise RecordValidationError("consultation response must be text")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        raise RecordValidationError("consultation response is empty")
    if len(normalized) > CONSULTATION_RESPONSE_MAX_CODE_POINTS:
        raise RecordValidationError("consultation response exceeds code point limit")

    for character in normalized:
        code_point = ord(character)
        if character not in {"\n", "\t"} and unicodedata.category(character) == "Cc":
            raise RecordValidationError("consultation response contains a control character")
        if 0x202A <= code_point <= 0x202E or 0x2066 <= code_point <= 0x2069:
            raise RecordValidationError("consultation response contains a bidi control")
        if 0x200B <= code_point <= 0x200F or code_point == 0xFEFF:
            raise RecordValidationError("consultation response contains a zero-width character")

    try:
        encoded = normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RecordValidationError("consultation response is not valid UTF-8 text") from exc
    if len(encoded) > CONSULTATION_RESPONSE_MAX_UTF8_BYTES:
        raise RecordValidationError("consultation response exceeds UTF-8 byte limit")
    return normalized


def validate_consultation_artifact(
    metadata: dict[str, Any],
    payload: bytes,
    *,
    expected_customer_id: str | None = None,
    expected_consultation_id: str | None = None,
    expected_confirmed: bool | None = None,
) -> None:
    fields = {
        "schema_version": int,
        "kind": str,
        "artifact_id": str,
        "consultation_id": str,
        "customer_id": str,
        "provider_kind": str,
        "ingest_change_id": str,
        "recorded_at": str,
        "delete_after": str,
        "payload_bytes": int,
        "payload_hash": str,
        "canonical_confirmed": bool,
    }
    if type(metadata) is not dict or set(metadata) != set(fields):
        raise RecordValidationError("consultation artifact metadata does not match schema")
    for key, expected_type in fields.items():
        if type(metadata[key]) is not expected_type:
            raise RecordValidationError(f"invalid consultation artifact field type: {key}")
    if type(payload) is not bytes:
        raise RecordValidationError("consultation artifact payload must be bytes")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise RecordValidationError("unsupported consultation artifact schema")
    if metadata["kind"] != "consultation_response":
        raise RecordValidationError("invalid consultation artifact kind")
    if not re.fullmatch(r"art_[0-9a-f]{32}", metadata["artifact_id"]):
        raise RecordValidationError("invalid consultation artifact ID")
    if not re.fullmatch(r"con_[0-9a-f]{32}", metadata["consultation_id"]):
        raise RecordValidationError("invalid consultation artifact consultation ID")
    if not re.fullmatch(r"cus_[0-9a-f]{32}", metadata["customer_id"]):
        raise RecordValidationError("invalid consultation artifact customer ID")
    if metadata["provider_kind"] not in CONSULTATION_PROVIDER_KINDS:
        raise RecordValidationError("invalid consultation artifact provider kind")
    if not _CHANGE_ID_RE.fullmatch(metadata["ingest_change_id"]):
        raise RecordValidationError("invalid consultation artifact change ID")

    recorded_at = _parse_utc_time(metadata["recorded_at"])
    delete_after = _parse_utc_time(metadata["delete_after"])
    if delete_after != recorded_at + timedelta(days=30):
        raise RecordValidationError("consultation artifact retention must be 30 days")
    if metadata["payload_bytes"] != len(payload) or not 0 < len(payload) <= CONSULTATION_RESPONSE_MAX_UTF8_BYTES:
        raise RecordValidationError("consultation artifact byte length mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", metadata["payload_hash"]):
        raise RecordValidationError("invalid consultation artifact payload digest")
    if hashlib.sha256(payload).hexdigest() != metadata["payload_hash"]:
        raise RecordValidationError("consultation artifact payload digest mismatch")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecordValidationError("consultation artifact payload is not UTF-8") from exc
    if normalize_consultation_response_text(text) != text:
        raise RecordValidationError("consultation artifact payload is not normalized")

    if expected_customer_id is not None and metadata["customer_id"] != expected_customer_id:
        raise RecordValidationError("consultation artifact customer mismatch")
    if expected_consultation_id is not None and metadata["consultation_id"] != expected_consultation_id:
        raise RecordValidationError("consultation artifact consultation mismatch")
    if expected_confirmed is not None and metadata["canonical_confirmed"] is not expected_confirmed:
        raise RecordValidationError("consultation artifact confirmation mismatch")


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise RecordValidationError("invalid timestamp") from exc
    if parsed.tzinfo is None:
        raise RecordValidationError("timestamp must include timezone")
    return parsed


def _parse_utc_time(value: str) -> datetime:
    parsed = _parse_time(value)
    if parsed.utcoffset() != timedelta(0):
        raise RecordValidationError("timestamp must be UTC")
    return parsed


def _digest(metadata: dict[str, Any], body: str) -> str:
    payload = {key: value for key, value in metadata.items() if key != "record_digest"}
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical + b"\0" + body.encode("utf-8")).hexdigest()


def _is_exact_type(value: Any, expected: type | tuple[type, ...]) -> bool:
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    return any(type(value) is candidate for candidate in expected_types)


def _validate_source_refs(values: list[Any]) -> None:
    if any(type(value) is not str or not _SOURCE_ID_RE.fullmatch(value) for value in values):
        raise RecordValidationError("source_refs must contain opaque entity IDs")


def _serialize_proposal_payload(payload: dict[str, Any]) -> bytes:
    allowed = {"schema_version", "target_type", "operation", "content"}
    if type(payload) is not dict or set(payload) != allowed:
        raise RecordValidationError("proposal payload does not match JSON schema")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise RecordValidationError("unsupported proposal payload version")
    if type(payload["target_type"]) is not str or payload["target_type"] not in {
        "information",
        "menu",
        "meeting",
    }:
        raise RecordValidationError("invalid proposal target")
    if (
        type(payload["operation"]) is not str
        or payload["operation"] not in {"create", "update"}
        or type(payload["content"]) is not dict
    ):
        raise RecordValidationError("invalid proposal operation")
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def validate_record(metadata: dict[str, Any], body: str = "") -> None:
    if type(metadata) is not dict:
        raise RecordValidationError("frontmatter must be a mapping")
    record_type = metadata.get("record_type")
    if record_type not in _RECORD_FIELDS:
        raise RecordValidationError("unknown record type")
    expected = {**_COMMON_FIELDS, **_RECORD_FIELDS[record_type]}
    if set(metadata) != set(expected):
        raise RecordValidationError("record fields do not match schema")
    for key, expected_type in expected.items():
        if not _is_exact_type(metadata[key], expected_type):
            raise RecordValidationError(f"invalid field type: {key}")
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise RecordValidationError("unsupported schema version")
    if metadata["revision"] < 1:
        raise RecordValidationError("revision must be positive")
    if not _ID_RE.fullmatch(metadata["record_id"]):
        raise RecordValidationError("invalid record ID")
    if not re.fullmatch(r"cus_[0-9a-f]{32}", metadata["customer_id"]):
        raise RecordValidationError("invalid customer ID")
    if not metadata["record_id"].startswith(f"{_ID_PREFIX[record_type]}_"):
        raise RecordValidationError("record ID type mismatch")
    if record_type == "customer" and metadata["record_id"] != metadata["customer_id"]:
        raise RecordValidationError("customer IDs must match")
    _parse_time(metadata["created_at"])
    _parse_time(metadata["updated_at"])
    if record_type == "customer":
        if not normalize_customer_name(metadata["display_name"]):
            raise RecordValidationError("display name is empty")
        if metadata["status"] not in {"active", "archive"}:
            raise RecordValidationError("invalid customer status")
        if any(type(alias) is not str or not normalize_customer_name(alias) for alias in metadata["aliases"]):
            raise RecordValidationError("invalid alias")
    if "source_refs" in metadata:
        _validate_source_refs(metadata["source_refs"])
    if record_type == "information":
        if metadata["information_status"] != "current":
            raise RecordValidationError("information status is derived from supersession")
        if metadata["origin"] not in {"manual", "ai_assisted"}:
            raise RecordValidationError("invalid information origin")
        if metadata["supersedes"] is not None and not re.fullmatch(
            r"inf_[0-9a-f]{32}", metadata["supersedes"]
        ):
            raise RecordValidationError("invalid supersedes ID")
    if record_type == "menu" and metadata["menu_kind"] not in {"spot", "recurring"}:
        raise RecordValidationError("invalid menu kind")
    if record_type == "menu_event":
        if not re.fullmatch(r"mnu_[0-9a-f]{32}", metadata["menu_id"]):
            raise RecordValidationError("invalid menu ID")
        if metadata["event_sequence"] < 1:
            raise RecordValidationError("menu event sequence must be positive")
        if metadata["to_status"] not in {"proposed", "active", "paused", "completed", "cancelled"}:
            raise RecordValidationError("invalid menu status")
        if metadata["from_status"] not in {None, "proposed", "active", "paused", "completed", "cancelled"}:
            raise RecordValidationError("invalid previous menu status")
    if record_type == "meeting":
        if metadata["meeting_status"] not in {"draft", "confirmed"}:
            raise RecordValidationError("invalid meeting status")
        if any(type(item) is not str for item in metadata["decisions"] + metadata["next_actions"]):
            raise RecordValidationError("meeting lists must contain strings")
        if metadata["meeting_status"] == "confirmed" and (
            not metadata["confirmed_by"] or not metadata["confirmed_at"]
        ):
            raise RecordValidationError("confirmed meeting lacks confirmation metadata")
    if record_type == "proposal":
        if metadata["provider"] not in {"local_ollama", "local_lmstudio"}:
            raise RecordValidationError("proposal provider must be local")
        if metadata["proposal_status"] not in {"pending", "accepted", "rejected"}:
            raise RecordValidationError("invalid proposal status")
        for key in ("prompt_contract_hash", "input_digest", "payload_hash"):
            if not re.fullmatch(r"[0-9a-f]{64}", metadata[key]):
                raise RecordValidationError(f"invalid proposal digest: {key}")
        if not metadata["model"] or not metadata["model_identifier"]:
            raise RecordValidationError("proposal model identity is required")
        status = metadata["proposal_status"]
        reviewed = bool(metadata["reviewed_by"] and metadata["reviewed_at"])
        applied = metadata["applied_record_id"]
        if status == "pending" and (reviewed or applied is not None):
            raise RecordValidationError("pending proposal has terminal metadata")
        if status == "accepted" and (
            not reviewed
            or type(applied) is not str
            or not re.fullmatch(r"inf_[0-9a-f]{32}", applied)
        ):
            raise RecordValidationError("accepted proposal lacks applied record")
        if status == "rejected" and (not reviewed or applied is not None):
            raise RecordValidationError("rejected proposal metadata is invalid")
        if metadata["reviewed_at"] is not None:
            _parse_time(metadata["reviewed_at"])
    if record_type == "consultation":
        if body != "":
            raise RecordValidationError("consultation body must be empty")
        if metadata["revision"] != 1:
            raise RecordValidationError("consultation records are immutable")
        if metadata["provider_kind"] not in CONSULTATION_PROVIDER_KINDS:
            raise RecordValidationError("invalid consultation provider kind")
        _parse_utc_time(metadata["response_received_at"])
        _parse_utc_time(metadata["recorded_at"])
        if not re.fullmatch(r"art_[0-9a-f]{32}", metadata["inbound_artifact_ref"]):
            raise RecordValidationError("invalid consultation artifact reference")
        if not re.fullmatch(r"[0-9a-f]{64}", metadata["inbound_response_digest"]):
            raise RecordValidationError("invalid consultation response digest")
        if metadata["supersedes"] is not None and not re.fullmatch(
            r"con_[0-9a-f]{32}", metadata["supersedes"]
        ):
            raise RecordValidationError("invalid consultation supersedes ID")
        if not _CHANGE_ID_RE.fullmatch(metadata["ingest_change_id"]):
            raise RecordValidationError("invalid consultation ingest change ID")
        if not re.fullmatch(r"[0-9a-f]{32}", metadata["recorded_by_ref"]):
            raise RecordValidationError("invalid consultation recorder reference")
    expected_digest = _digest(metadata, body)
    if not re.fullmatch(r"[0-9a-f]{64}", metadata["record_digest"]):
        raise RecordValidationError("invalid record digest")
    if metadata["record_digest"] != expected_digest:
        raise RecordValidationError("record digest mismatch")


def serialize_record(metadata: dict[str, Any], body: str = "") -> str:
    candidate = dict(metadata)
    candidate["record_digest"] = _digest(candidate, body)
    validate_record(candidate, body)
    frontmatter = yaml.safe_dump(
        candidate,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    )
    return f"---\n{frontmatter}---\n{body}"


def parse_record(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        raise RecordValidationError("missing frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise RecordValidationError("unterminated frontmatter")
    raw_frontmatter = text[4:end]
    body = text[end + 5 :]
    try:
        metadata = yaml.safe_load(raw_frontmatter)
    except yaml.YAMLError as exc:
        raise RecordValidationError("invalid YAML frontmatter") from exc
    validate_record(metadata, body)
    return metadata, body


def _assert_no_symlink_components(path: Path, stop: Path) -> None:
    current = path
    while current != stop and current != current.parent:
        if current.exists() and current.is_symlink():
            raise CustomerStoreError("symlink path is not allowed")
        current = current.parent


def _safe_mkdir_tree(path: Path, boundary_parent: Path) -> None:
    try:
        relative = path.relative_to(boundary_parent)
    except ValueError as exc:
        raise CustomerStoreError("directory is outside the trusted boundary") from exc
    current = boundary_parent
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise CustomerStoreError("symlink or non-directory path is not allowed")
            _assert_private_path(current, directory=True)
            continue
        current.mkdir(mode=0o700)
        _assert_private_path(current, directory=True)


def _assert_private_path(path: Path, *, directory: bool) -> None:
    info = os.lstat(path)
    expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
    if os.name == "nt":
        attributes = getattr(info, "st_file_attributes", 0)
        if not expected_kind(info.st_mode) or attributes & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
        ):
            raise CustomerStoreError("customer data reparse points are not allowed")
        return
    if not expected_kind(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise CustomerStoreError("customer data permissions are not private")
    if sys.platform == "darwin":
        import ctypes

        libc = ctypes.CDLL("libc.dylib", use_errno=True)
        libc.acl_get_file.restype = ctypes.c_void_p
        acl = libc.acl_get_file(os.fsencode(path), 0x00000100)
        if not acl:
            if ctypes.get_errno() == 2:
                return
            raise CustomerStoreError("customer data ACL could not be verified")
        try:
            entry = ctypes.c_void_p()
            if libc.acl_get_entry(acl, 0, ctypes.byref(entry)) == 1:
                raise CustomerStoreError("customer data has an extended ACL")
        finally:
            libc.acl_free(ctypes.c_void_p(acl))


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, text: str, *, overwrite: bool) -> None:
    vault = next((parent for parent in path.parents if parent.name == "customer-vault"), None)
    if vault is None:
        raise CustomerStoreError("record write is outside customer-vault")
    _safe_mkdir_tree(path.parent, vault.parent)
    _assert_no_symlink_components(path.parent, vault.parent)
    if path.exists() and not overwrite:
        raise CustomerStoreError("immutable record already exists")
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temp_path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _assert_private_path(path, directory=False)
        _fsync_directory(path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink()


class CustomerStore:
    def __init__(self, root: Path, logger: AuditLogger | None = None):
        self.root = Path(root)
        if self.root.exists() and self.root.is_symlink():
            raise CustomerStoreError("data root must not be a symlink")
        self.vault = self.root / "customer-vault"
        self.customers_dir = self.vault / "customers"
        self.staging_dir = self.vault / "staging"
        self.transactions_dir = self.vault / "transactions"
        # Default audit setup is deferred so construction and root preflight are read-only.
        self.logger = logger

    def _root_state(self) -> tuple[str, tuple[int, int] | None]:
        if not self.root.exists():
            return "absent", None
        if self.root.is_symlink() or not self.root.is_dir():
            raise CustomerStoreError("data root must be a private directory")
        _assert_private_path(self.root, directory=True)
        root_info = os.lstat(self.root)
        identity = (root_info.st_dev, root_info.st_ino)
        if not any(self.root.iterdir()):
            return "empty", identity

        schema_path = self.vault / "schema.yaml"
        if not schema_path.is_file() or schema_path.is_symlink():
            raise RecordValidationError("non-empty data root is not an initialized store")
        try:
            data = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise RecordValidationError("invalid customer schema manifest") from exc
        if type(data) is not dict or set(data) != {"schema_version"}:
            raise RecordValidationError("unsupported customer schema manifest")
        version = data["schema_version"]
        if type(version) is not int or version != SCHEMA_VERSION:
            if version == 1:
                raise RecordValidationError("unsupported legacy synthetic customer store")
            raise RecordValidationError("unsupported customer schema manifest")
        return "current", identity

    @contextmanager
    def _initialization_lock(self) -> Iterator[None]:
        parent = self.root.parent
        if not parent.is_dir() or parent.is_symlink():
            raise CustomerStoreError("data root parent must be an existing directory")
        lock_path = parent / ".shokoban-customer-init.lock"
        if lock_path.is_symlink():
            raise CustomerStoreError("initialization lock must not be a symlink")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(lock_path, flags, 0o600)
            _assert_private_path(lock_path, directory=False)
            if os.name == "nt":
                import msvcrt

                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                    os.fsync(fd)
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
        except (OSError, IOError) as exc:
            if "fd" in locals():
                os.close(fd)
            raise CustomerStoreError("customer store initialization is locked") from exc
        try:
            yield
        finally:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def initialize(self, *, confirmation: str | None = None) -> None:
        state, identity = self._root_state()
        if state == "current":
            self._validate_initialized_layout()
            return
        if state == "empty" and confirmation != CLEAN_ROOT_CONFIRMATION:
            raise CustomerStoreError("existing empty data root requires literal confirmation")

        with self._initialization_lock():
            locked_state, locked_identity = self._root_state()
            if locked_state != state or locked_identity != identity:
                raise CustomerStoreError("data root changed during initialization")
            if locked_state == "absent":
                self.root.mkdir(mode=0o700)
                _assert_private_path(self.root, directory=True)
            elif any(self.root.iterdir()):
                raise CustomerStoreError("data root changed during initialization")

            _safe_mkdir_tree(self.customers_dir, self.root.parent)
            _safe_mkdir_tree(self.staging_dir / "artifacts", self.root.parent)
            _safe_mkdir_tree(self.staging_dir / "proposals", self.root.parent)
            _safe_mkdir_tree(self.transactions_dir, self.root.parent)
            _atomic_write(
                self.vault / "schema.yaml",
                yaml.safe_dump({"schema_version": SCHEMA_VERSION}, sort_keys=True),
                overwrite=False,
            )
            self._validate_initialized_layout()

    def _validate_initialized_layout(self) -> None:
        self.validate_schema_manifest()
        for path in (
            self.vault,
            self.customers_dir,
            self.staging_dir,
            self.staging_dir / "artifacts",
            self.staging_dir / "proposals",
            self.transactions_dir,
        ):
            if not path.is_dir() or path.is_symlink():
                raise RecordValidationError("partial customer store layout")
            _assert_private_path(path, directory=True)
        self.validate_all()

    def validate_schema_manifest(self) -> None:
        schema_path = self.vault / "schema.yaml"
        if not schema_path.is_file() or schema_path.is_symlink():
            raise RecordValidationError("missing customer schema manifest")
        data = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
        if type(data) is not dict or data != {"schema_version": SCHEMA_VERSION}:
            raise RecordValidationError("unsupported customer schema manifest")

    @contextmanager
    def _write_lock(self, *, recover: bool = True) -> Iterator[None]:
        self.validate_schema_manifest()
        _safe_mkdir_tree(self.vault, self.root.parent)
        lock_path = self.vault / ".write.lock"
        if lock_path.is_symlink():
            raise CustomerStoreError("customer store lock must not be a symlink")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(lock_path, flags, 0o600)
            if os.name == "nt":
                import msvcrt

                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                    os.fsync(fd)
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as exc:
            if "fd" in locals():
                os.close(fd)
            raise CustomerStoreError("customer store is locked") from exc
        try:
            self.validate_schema_manifest()
            if recover:
                self._recover_transactions_locked()
            yield
        finally:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _proposal_payload_locked(
        self, customer_id: str, proposal_id: str, metadata: dict[str, Any]
    ) -> tuple[dict[str, Any], bytes]:
        payload_path = self.staging_dir / "proposals" / f"{proposal_id}.json"
        if not payload_path.is_file() or payload_path.is_symlink():
            raise RecordValidationError("proposal payload is unavailable")
        _assert_private_path(payload_path, directory=False)
        payload_bytes = payload_path.read_bytes()
        if hashlib.sha256(payload_bytes).hexdigest() != metadata["payload_hash"]:
            raise RecordValidationError("proposal payload digest mismatch")
        try:
            payload = json.loads(payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecordValidationError("proposal payload is invalid") from exc
        if _serialize_proposal_payload(payload) != payload_bytes:
            raise RecordValidationError("proposal payload is not canonical JSON")
        if metadata["customer_id"] != customer_id:
            raise RecordValidationError("proposal customer mismatch")
        return payload, payload_bytes

    @staticmethod
    def _information_text(payload: dict[str, Any]) -> str:
        if (
            payload["target_type"] != "information"
            or payload["operation"] != "create"
            or set(payload["content"]) != {"text"}
            or type(payload["content"]["text"]) is not str
            or not payload["content"]["text"].strip()
        ):
            raise RecordValidationError("proposal is not strict information/create")
        return payload["content"]["text"]

    def _recover_transactions_locked(self) -> None:
        if not self.transactions_dir.exists():
            return
        for intent_path in sorted(self.transactions_dir.glob("*.json")):
            if intent_path.is_symlink() or not intent_path.is_file():
                raise CustomerStoreError("invalid proposal transaction intent")
            _assert_private_path(intent_path, directory=False)
            try:
                intent = json.loads(intent_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CustomerStoreError("invalid proposal transaction intent") from exc
            if type(intent) is not dict or intent.get("schema_version") != 1:
                raise CustomerStoreError("invalid proposal transaction intent")
            operation = intent.get("operation")
            if operation == "edit_payload":
                self._recover_proposal_edit_locked(intent_path, intent)
                continue
            if operation == "stage_proposal":
                self._recover_proposal_stage_locked(intent_path, intent)
                continue
            if operation == "stage_consultation_proposal":
                self._recover_consultation_proposal_stage_locked(intent_path, intent)
                continue
            if operation == "record_consultation":
                self._recover_consultation_locked(intent_path, intent)
                continue
            if operation == "delete_consultation_response":
                self._recover_consultation_response_delete_locked(intent_path, intent)
                continue
            if operation != "accept_apply":
                raise CustomerStoreError("unknown proposal transaction operation")
            expected = {
                "schema_version", "operation", "customer_id", "customer_revision", "proposal_id",
                "proposal_revision", "payload_hash", "information_id", "change_id",
                "reviewed_by", "reviewed_at",
            }
            if type(intent) is not dict or set(intent) != expected or intent["schema_version"] != 1:
                raise CustomerStoreError("invalid proposal transaction intent")
            customer_id = intent["customer_id"]
            proposal_id = intent["proposal_id"]
            information_id = intent["information_id"]
            if any(type(value) is not str for value in (customer_id, proposal_id, information_id)) or not re.fullmatch(r"cus_[0-9a-f]{32}", customer_id) or not re.fullmatch(
                r"prp_[0-9a-f]{32}", proposal_id
            ) or not re.fullmatch(r"inf_[0-9a-f]{32}", information_id) or type(
                intent["change_id"]
            ) is not str or not _CHANGE_ID_RE.fullmatch(intent["change_id"]):
                raise CustomerStoreError("invalid proposal transaction IDs")
            if (
                type(intent["customer_revision"]) is not int
                or type(intent["proposal_revision"]) is not int
                or type(intent["reviewed_by"]) is not str
                or not intent["reviewed_by"].strip()
                or type(intent["reviewed_at"]) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", intent["payload_hash"])
            ):
                raise CustomerStoreError("invalid proposal transaction values")
            _parse_time(intent["reviewed_at"])
            proposal_path = self._customer_dir(customer_id) / "proposals" / f"{proposal_id}.md"
            proposal, proposal_body = self._read(proposal_path)
            if proposal["payload_hash"] != intent["payload_hash"]:
                raise CustomerStoreError("proposal transaction payload mismatch")
            payload, _ = self._proposal_payload_locked(customer_id, proposal_id, proposal)
            text = self._information_text(payload)
            information_path = self._customer_dir(customer_id) / "information" / f"{information_id}.md"
            if not information_path.exists():
                if proposal["proposal_status"] != "pending" or proposal["revision"] != intent["proposal_revision"]:
                    raise CustomerStoreError("proposal transaction state mismatch")
                intent_path.unlink()
                _fsync_directory(self.transactions_dir)
                continue
            information, body = self._read(information_path)
            if body != text or information["origin"] != "ai_assisted" or information["source_refs"] != [proposal_id]:
                raise CustomerStoreError("proposal transaction information mismatch")
            if proposal["proposal_status"] == "pending":
                if proposal["revision"] != intent["proposal_revision"]:
                    raise CustomerStoreError("proposal transaction revision mismatch")
                updated = dict(proposal)
                updated.update(
                    proposal_status="accepted", reviewed_by=intent["reviewed_by"],
                    reviewed_at=intent["reviewed_at"], applied_record_id=information_id,
                    revision=proposal["revision"] + 1, updated_at=intent["reviewed_at"],
                )
                proposal = self._write(proposal_path, updated, proposal_body, overwrite=True)
            elif proposal["proposal_status"] != "accepted" or proposal["applied_record_id"] != information_id:
                raise CustomerStoreError("proposal transaction terminal state mismatch")
            logger = self.logger or AuditLogger(self.root)
            self.logger = logger
            if not logger.has_change_id(intent["change_id"]):
                self._audit(
                    "proposal-accept-apply", "proposal", proposal_id, proposal_path,
                    revision_before=intent["proposal_revision"], revision_after=proposal["revision"],
                    change_id=intent["change_id"],
                )
            intent_path.unlink()
            _fsync_directory(self.transactions_dir)

    def _recover_proposal_stage_locked(
        self, intent_path: Path, intent: dict[str, Any]
    ) -> None:
        expected = {
            "schema_version", "operation", "customer_id", "proposal_id",
            "payload_hash", "change_id",
        }
        if set(intent) != expected:
            raise CustomerStoreError("invalid proposal stage intent")
        customer_id = intent["customer_id"]
        proposal_id = intent["proposal_id"]
        change_id = intent["change_id"]
        if (
            type(customer_id) is not str
            or type(proposal_id) is not str
            or not re.fullmatch(r"cus_[0-9a-f]{32}", customer_id)
            or not re.fullmatch(r"prp_[0-9a-f]{32}", proposal_id)
            or type(intent["payload_hash"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", intent["payload_hash"])
        ):
            raise CustomerStoreError("invalid proposal stage transaction IDs")
        if type(change_id) is not str or not _CHANGE_ID_RE.fullmatch(change_id):
            raise CustomerStoreError("invalid proposal stage change ID")
        proposal_path = self._customer_dir(customer_id) / "proposals" / f"{proposal_id}.md"
        payload_path = self.staging_dir / "proposals" / f"{proposal_id}.json"
        pending_path = self.staging_dir / "proposals" / f".{proposal_id}.{change_id}.json"
        if not proposal_path.exists():
            if payload_path.exists():
                raise CustomerStoreError("orphaned proposal payload requires repair")
            if pending_path.exists():
                pending_path.unlink()
                _fsync_directory(pending_path.parent)
            intent_path.unlink()
            _fsync_directory(self.transactions_dir)
            return
        proposal, _ = self._read(proposal_path)
        if proposal["customer_id"] != customer_id or proposal["payload_hash"] != intent["payload_hash"]:
            raise CustomerStoreError("proposal stage metadata mismatch")
        if not payload_path.exists():
            if not pending_path.is_file() or pending_path.is_symlink():
                raise CustomerStoreError("proposal stage payload is unavailable")
            if hashlib.sha256(pending_path.read_bytes()).hexdigest() != intent["payload_hash"]:
                raise CustomerStoreError("proposal stage payload mismatch")
            os.replace(pending_path, payload_path)
            _fsync_directory(payload_path.parent)
        self._proposal_payload_locked(customer_id, proposal_id, proposal)
        if pending_path.exists():
            pending_path.unlink()
            _fsync_directory(pending_path.parent)
        logger = self.logger or AuditLogger(self.root)
        self.logger = logger
        if not logger.has_change_id(change_id):
            self._audit(
                "proposal-stage", "proposal", proposal_id, proposal_path,
                revision_after=proposal["revision"], change_id=change_id,
            )
        intent_path.unlink()
        _fsync_directory(self.transactions_dir)

    def _recover_consultation_proposal_stage_locked(
        self, intent_path: Path, intent: dict[str, Any]
    ) -> None:
        expected = {
            "schema_version", "operation", "customer_id", "consultation_id",
            "proposal_id", "payload_hash", "change_id",
        }
        if set(intent) != expected:
            raise CustomerStoreError("invalid consultation proposal stage intent")
        customer_id = intent["customer_id"]
        consultation_id = intent["consultation_id"]
        proposal_id = intent["proposal_id"]
        change_id = intent["change_id"]
        if (
            type(customer_id) is not str
            or not re.fullmatch(r"cus_[0-9a-f]{32}", customer_id)
            or type(consultation_id) is not str
            or not re.fullmatch(r"con_[0-9a-f]{32}", consultation_id)
            or type(proposal_id) is not str
            or not re.fullmatch(r"prp_[0-9a-f]{32}", proposal_id)
            or type(intent["payload_hash"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", intent["payload_hash"])
            or type(change_id) is not str
            or not _CHANGE_ID_RE.fullmatch(change_id)
        ):
            raise CustomerStoreError("invalid consultation proposal stage values")

        proposal_path = self._customer_dir(customer_id) / "proposals" / f"{proposal_id}.md"
        payload_path = self.staging_dir / "proposals" / f"{proposal_id}.json"
        pending_path = self.staging_dir / "proposals" / f".{proposal_id}.{change_id}.json"
        if not proposal_path.exists():
            if payload_path.exists():
                raise CustomerStoreError("orphaned consultation proposal payload requires repair")
            if pending_path.exists():
                pending_path.unlink()
                _fsync_directory(pending_path.parent)
            intent_path.unlink()
            _fsync_directory(self.transactions_dir)
            return

        proposal, _ = self._read(proposal_path)
        if (
            proposal["customer_id"] != customer_id
            or proposal["source_refs"] != [consultation_id]
            or proposal["payload_hash"] != intent["payload_hash"]
        ):
            raise CustomerStoreError("consultation proposal stage metadata mismatch")
        related = self._consultation_proposals_locked(customer_id, consultation_id)
        if len(related) != 1 or related[0][0]["record_id"] != proposal_id:
            raise CustomerStoreError("consultation proposal uniqueness violation")
        self._consultation_extraction_input_locked(
            customer_id, consultation_id, require_no_proposal=False
        )
        if not payload_path.exists():
            if not pending_path.is_file() or pending_path.is_symlink():
                raise CustomerStoreError("consultation proposal payload is unavailable")
            if hashlib.sha256(pending_path.read_bytes()).hexdigest() != intent["payload_hash"]:
                raise CustomerStoreError("consultation proposal payload mismatch")
            os.replace(pending_path, payload_path)
            _fsync_directory(payload_path.parent)
        self._proposal_payload_locked(customer_id, proposal_id, proposal)
        if pending_path.exists():
            pending_path.unlink()
            _fsync_directory(pending_path.parent)
        logger = self.logger or AuditLogger(self.root)
        self.logger = logger
        if not logger.has_change_id(change_id):
            self._audit(
                "consultation-extraction-stage", "proposal", proposal_id,
                proposal_path, revision_after=1, change_id=change_id,
            )
        intent_path.unlink()
        _fsync_directory(self.transactions_dir)

    def _recover_proposal_edit_locked(
        self, intent_path: Path, intent: dict[str, Any]
    ) -> None:
        expected = {
            "schema_version", "operation", "customer_id", "proposal_id",
            "proposal_revision", "previous_payload_hash", "new_payload_hash",
            "change_id",
        }
        if set(intent) != expected:
            raise CustomerStoreError("invalid proposal edit intent")
        customer_id = intent["customer_id"]
        proposal_id = intent["proposal_id"]
        change_id = intent["change_id"]
        if type(customer_id) is not str or type(proposal_id) is not str or not re.fullmatch(r"cus_[0-9a-f]{32}", customer_id) or not re.fullmatch(
            r"prp_[0-9a-f]{32}", proposal_id
        ) or type(change_id) is not str or not _CHANGE_ID_RE.fullmatch(change_id):
            raise CustomerStoreError("invalid proposal edit transaction IDs")
        if (
            type(intent["proposal_revision"]) is not int
            or not re.fullmatch(r"[0-9a-f]{64}", intent["previous_payload_hash"])
            or not re.fullmatch(r"[0-9a-f]{64}", intent["new_payload_hash"])
        ):
            raise CustomerStoreError("invalid proposal edit transaction values")
        proposal_path = self._customer_dir(customer_id) / "proposals" / f"{proposal_id}.md"
        payload_path = self.staging_dir / "proposals" / f"{proposal_id}.json"
        pending_path = self.staging_dir / "proposals" / f".{proposal_id}.{change_id}.json"
        proposal, _ = self._read(proposal_path)
        payload_bytes = payload_path.read_bytes()
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
        if proposal["revision"] == intent["proposal_revision"]:
            if proposal["payload_hash"] != intent["previous_payload_hash"] or payload_hash != intent["previous_payload_hash"]:
                raise CustomerStoreError("proposal edit rollback state mismatch")
            if pending_path.exists():
                pending_path.unlink()
                _fsync_directory(pending_path.parent)
            intent_path.unlink()
            _fsync_directory(self.transactions_dir)
            return
        if (
            proposal["revision"] != intent["proposal_revision"] + 1
            or proposal["payload_hash"] != intent["new_payload_hash"]
        ):
            raise CustomerStoreError("proposal edit commit state mismatch")
        if payload_hash == intent["previous_payload_hash"]:
            if not pending_path.is_file() or pending_path.is_symlink():
                raise CustomerStoreError("proposal edit payload is unavailable")
            pending_bytes = pending_path.read_bytes()
            if hashlib.sha256(pending_bytes).hexdigest() != intent["new_payload_hash"]:
                raise CustomerStoreError("proposal edit payload mismatch")
            os.replace(pending_path, payload_path)
            _fsync_directory(payload_path.parent)
            payload_bytes = pending_bytes
        elif payload_hash != intent["new_payload_hash"]:
            raise CustomerStoreError("proposal edit payload state mismatch")
        if pending_path.exists():
            pending_path.unlink()
            _fsync_directory(pending_path.parent)
        try:
            parsed = json.loads(payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CustomerStoreError("proposal edit payload is invalid") from exc
        if _serialize_proposal_payload(parsed) != payload_bytes:
            raise CustomerStoreError("proposal edit payload is not canonical JSON")
        logger = self.logger or AuditLogger(self.root)
        self.logger = logger
        if not logger.has_change_id(change_id):
            self._audit(
                "proposal-edit", "proposal", proposal_id, proposal_path,
                revision_before=intent["proposal_revision"],
                revision_after=proposal["revision"], change_id=change_id,
            )
        intent_path.unlink()
        _fsync_directory(self.transactions_dir)

    def _recover_consultation_locked(
        self, intent_path: Path, intent: dict[str, Any]
    ) -> None:
        expected = {
            "schema_version", "operation", "customer_id", "customer_revision",
            "provider_kind", "artifact_id", "consultation_id", "change_id",
            "response_digest", "payload_bytes", "recorded_by_ref", "recorded_at",
        }
        if set(intent) != expected:
            raise CustomerStoreError("invalid consultation transaction intent")
        customer_id = intent["customer_id"]
        artifact_id = intent["artifact_id"]
        consultation_id = intent["consultation_id"]
        change_id = intent["change_id"]
        if (
            type(customer_id) is not str
            or not re.fullmatch(r"cus_[0-9a-f]{32}", customer_id)
            or type(artifact_id) is not str
            or not re.fullmatch(r"art_[0-9a-f]{32}", artifact_id)
            or type(consultation_id) is not str
            or not re.fullmatch(r"con_[0-9a-f]{32}", consultation_id)
            or type(change_id) is not str
            or not _CHANGE_ID_RE.fullmatch(change_id)
        ):
            raise CustomerStoreError("invalid consultation transaction IDs")
        if (
            type(intent["customer_revision"]) is not int
            or type(intent["provider_kind"]) is not str
            or intent["provider_kind"] not in CONSULTATION_PROVIDER_KINDS
            or type(intent["response_digest"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", intent["response_digest"])
            or type(intent["payload_bytes"]) is not int
            or not (0 < intent["payload_bytes"] <= CONSULTATION_RESPONSE_MAX_UTF8_BYTES)
            or type(intent["recorded_by_ref"]) is not str
            or not re.fullmatch(r"[0-9a-f]{32}", intent["recorded_by_ref"])
            or type(intent["recorded_at"]) is not str
        ):
            raise CustomerStoreError("invalid consultation transaction values")
        _parse_utc_time(intent["recorded_at"])

        artifact_dir = self.staging_dir / "artifacts" / artifact_id
        pending_dir = self.staging_dir / "artifacts" / f".{artifact_id}.{change_id}.pending"
        consultation_path = (
            self._customer_dir(customer_id) / "consultations" / f"{consultation_id}.md"
        )
        customer, _ = self._read(self._customer_dir(customer_id) / "customer.md")
        if (
            customer["status"] != "active"
            or customer["revision"] != intent["customer_revision"]
        ):
            raise CustomerStoreError("consultation transaction customer mismatch")

        if not artifact_dir.exists() and not artifact_dir.is_symlink():
            if consultation_path.exists():
                raise CustomerStoreError(
                    "consultation record without artifact is not recoverable"
                )
            if pending_dir.exists() or pending_dir.is_symlink():
                if pending_dir.is_symlink() or not pending_dir.is_dir():
                    raise CustomerStoreError("invalid pending consultation artifact")
                _assert_private_path(pending_dir, directory=True)
                shutil.rmtree(pending_dir)
                _fsync_directory(pending_dir.parent)
            intent_path.unlink()
            _fsync_directory(self.transactions_dir)
            return

        if pending_dir.exists() or pending_dir.is_symlink():
            raise CustomerStoreError("conflicting pending consultation artifact")
        metadata, _, metadata_path = self._consultation_artifact_locked(
            artifact_id,
            expected_customer_id=customer_id,
            expected_consultation_id=consultation_id,
        )
        if metadata["ingest_change_id"] != change_id:
            raise CustomerStoreError("consultation artifact change ID mismatch")
        if metadata["provider_kind"] != intent["provider_kind"]:
            raise CustomerStoreError("consultation artifact provider mismatch")
        if metadata["payload_hash"] != intent["response_digest"]:
            raise CustomerStoreError("consultation artifact digest mismatch")
        if metadata["payload_bytes"] != intent["payload_bytes"]:
            raise CustomerStoreError("consultation artifact byte count mismatch")
        if metadata["recorded_at"] != intent["recorded_at"]:
            raise CustomerStoreError("consultation artifact timestamp mismatch")

        if consultation_path.exists():
            consultation, body = self._read(consultation_path)
            binding = {
                "record_id": consultation_id,
                "customer_id": customer_id,
                "provider_kind": intent["provider_kind"],
                "response_received_at": intent["recorded_at"],
                "inbound_artifact_ref": artifact_id,
                "inbound_response_digest": intent["response_digest"],
                "outbound_brief_ref": None,
                "outbound_brief_digest": None,
                "supersedes": None,
                "ingest_change_id": change_id,
                "recorded_by_ref": intent["recorded_by_ref"],
                "recorded_at": intent["recorded_at"],
                "created_at": intent["recorded_at"],
                "updated_at": intent["recorded_at"],
            }
            if body != "" or any(consultation[key] != value for key, value in binding.items()):
                raise CustomerStoreError("consultation record binding mismatch")
            if not metadata["canonical_confirmed"]:
                metadata["canonical_confirmed"] = True
                _atomic_write(
                    metadata_path,
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    overwrite=True,
                )
                self._consultation_artifact_locked(
                    artifact_id,
                    expected_customer_id=customer_id,
                    expected_consultation_id=consultation_id,
                    expected_confirmed=True,
                )
        else:
            if metadata["canonical_confirmed"]:
                raise CustomerStoreError("confirmed artifact lacks consultation record")
            shutil.rmtree(artifact_dir)
            _fsync_directory(artifact_dir.parent)
            intent_path.unlink()
            _fsync_directory(self.transactions_dir)
            return

        logger = self.logger or AuditLogger(self.root)
        self.logger = logger
        if not logger.has_change_id(change_id):
            self._audit(
                "consultation-record",
                "consultation",
                consultation_id,
                consultation_path,
                revision_after=1,
                change_id=change_id,
            )
        intent_path.unlink()
        _fsync_directory(self.transactions_dir)

    def _recover_consultation_response_delete_locked(
        self, intent_path: Path, intent: dict[str, Any]
    ) -> None:
        expected = {
            "schema_version", "operation", "customer_id", "consultation_id",
            "artifact_id", "delete_after", "change_id",
        }
        if set(intent) != expected:
            raise CustomerStoreError("invalid consultation response deletion intent")
        customer_id = intent["customer_id"]
        consultation_id = intent["consultation_id"]
        artifact_id = intent["artifact_id"]
        change_id = intent["change_id"]
        if (
            type(customer_id) is not str
            or not re.fullmatch(r"cus_[0-9a-f]{32}", customer_id)
            or type(consultation_id) is not str
            or not re.fullmatch(r"con_[0-9a-f]{32}", consultation_id)
            or type(artifact_id) is not str
            or not re.fullmatch(r"art_[0-9a-f]{32}", artifact_id)
            or type(change_id) is not str
            or not _CHANGE_ID_RE.fullmatch(change_id)
            or type(intent["delete_after"]) is not str
        ):
            raise CustomerStoreError("invalid consultation response deletion values")
        _parse_utc_time(intent["delete_after"])

        consultation_path = (
            self._customer_dir(customer_id) / "consultations" / f"{consultation_id}.md"
        )
        consultation, body = self._read(consultation_path)
        if (
            body != ""
            or consultation["customer_id"] != customer_id
            or consultation["inbound_artifact_ref"] != artifact_id
        ):
            raise CustomerStoreError("consultation response deletion binding mismatch")

        artifact_dir = self.staging_dir / "artifacts" / artifact_id
        if artifact_dir.exists() or artifact_dir.is_symlink():
            artifact, _, _ = self._consultation_artifact_locked(
                artifact_id,
                expected_customer_id=customer_id,
                expected_consultation_id=consultation_id,
                expected_confirmed=True,
            )
            if (
                artifact["delete_after"] != intent["delete_after"]
                or artifact["payload_hash"] != consultation["inbound_response_digest"]
                or artifact["provider_kind"] != consultation["provider_kind"]
                or artifact["ingest_change_id"] != consultation["ingest_change_id"]
                or artifact["recorded_at"] != consultation["recorded_at"]
            ):
                raise CustomerStoreError("consultation response deletion artifact mismatch")
            shutil.rmtree(artifact_dir)
            _fsync_directory(artifact_dir.parent)

        logger = self.logger or AuditLogger(self.root)
        self.logger = logger
        if not logger.has_change_id(change_id):
            self._audit(
                "consultation-response-delete",
                "consultation",
                consultation_id,
                artifact_dir,
                retention_action="expired",
                change_id=change_id,
            )
        intent_path.unlink()
        _fsync_directory(self.transactions_dir)

    def _consultation_artifact_locked(
        self,
        artifact_id: str,
        *,
        expected_customer_id: str,
        expected_consultation_id: str,
        expected_confirmed: bool | None = None,
    ) -> tuple[dict[str, Any], bytes, Path]:
        artifact_dir = self.staging_dir / "artifacts" / artifact_id
        if artifact_dir.is_symlink() or not artifact_dir.is_dir():
            raise CustomerStoreError("consultation artifact directory is invalid")
        _assert_private_path(artifact_dir, directory=True)
        entries = {path.name: path for path in artifact_dir.iterdir()}
        if set(entries) != {"metadata.json", "payload.txt"}:
            raise CustomerStoreError("consultation artifact directory has unexpected files")
        metadata_path = entries["metadata.json"]
        payload_path = entries["payload.txt"]
        for path in (metadata_path, payload_path):
            if path.is_symlink() or not path.is_file():
                raise CustomerStoreError("consultation artifact file is invalid")
            _assert_private_path(path, directory=False)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CustomerStoreError("consultation artifact metadata is malformed") from exc
        payload = payload_path.read_bytes()
        validate_consultation_artifact(
            metadata,
            payload,
            expected_customer_id=expected_customer_id,
            expected_consultation_id=expected_consultation_id,
            expected_confirmed=expected_confirmed,
        )
        return metadata, payload, metadata_path

    def _consultation_for_change_locked(
        self, change_id: str
    ) -> tuple[dict[str, Any], Path] | None:
        matches: list[tuple[dict[str, Any], Path]] = []
        for path in sorted(self.customers_dir.glob("cus_*/consultations/con_*.md")):
            metadata, body = self._read(path)
            if metadata["ingest_change_id"] == change_id:
                if body != "":
                    raise CustomerStoreError("consultation record body must be empty")
                matches.append((metadata, path))
        if len(matches) > 1:
            raise CustomerStoreError("duplicate consultation operation ID")
        return matches[0] if matches else None

    def _consultation_proposals_locked(
        self, customer_id: str, consultation_id: str
    ) -> list[tuple[dict[str, Any], Path]]:
        matches: list[tuple[dict[str, Any], Path]] = []
        for path in sorted(self.customers_dir.glob("cus_*/proposals/prp_*.md")):
            metadata, _ = self._read(path)
            if consultation_id not in metadata["source_refs"]:
                continue
            if metadata["customer_id"] != customer_id or metadata["source_refs"] != [consultation_id]:
                raise CustomerStoreError("consultation proposal source binding is invalid")
            matches.append((metadata, path))
        if len(matches) > 1:
            raise CustomerStoreError("consultation has duplicate extraction proposals")
        return matches

    def _proposal_information_locked(
        self, customer_id: str, proposal_id: str
    ) -> list[tuple[dict[str, Any], str, Path]]:
        matches: list[tuple[dict[str, Any], str, Path]] = []
        for path in sorted(self.customers_dir.glob("cus_*/information/inf_*.md")):
            metadata, body = self._read(path)
            if proposal_id not in metadata["source_refs"]:
                continue
            if (
                metadata["customer_id"] != customer_id
                or metadata["source_refs"] != [proposal_id]
                or metadata["origin"] != "ai_assisted"
            ):
                raise CustomerStoreError("proposal information binding is invalid")
            matches.append((metadata, body, path))
        if len(matches) > 1:
            raise CustomerStoreError("proposal has duplicate applied information")
        return matches

    def _consultation_extraction_input_locked(
        self,
        customer_id: str,
        consultation_id: str,
        *,
        require_no_proposal: bool,
    ) -> str:
        if not re.fullmatch(r"con_[0-9a-f]{32}", consultation_id):
            raise RecordValidationError("invalid consultation ID")
        customer, _ = self._read(self._customer_dir(customer_id) / "customer.md")
        if customer["status"] != "active":
            raise RecordValidationError("customer is not active")
        consultation_path = (
            self._customer_dir(customer_id) / "consultations" / f"{consultation_id}.md"
        )
        consultation, body = self._read(consultation_path)
        if consultation["customer_id"] != customer_id or body != "":
            raise RecordValidationError("consultation record binding is invalid")

        for path in sorted(consultation_path.parent.glob("con_*.md")):
            candidate, candidate_body = self._read(path)
            if candidate_body != "":
                raise RecordValidationError("consultation record body must be empty")
            if candidate["supersedes"] == consultation_id:
                raise RecordValidationError("consultation is not current")

        related = self._consultation_proposals_locked(customer_id, consultation_id)
        if require_no_proposal and related:
            raise RecordValidationError("consultation already has an extraction proposal")

        artifact_id = consultation["inbound_artifact_ref"]
        artifact, payload, _ = self._consultation_artifact_locked(
            artifact_id,
            expected_customer_id=customer_id,
            expected_consultation_id=consultation_id,
            expected_confirmed=True,
        )
        if (
            artifact["payload_hash"] != consultation["inbound_response_digest"]
            or artifact["provider_kind"] != consultation["provider_kind"]
            or artifact["ingest_change_id"] != consultation["ingest_change_id"]
            or artifact["recorded_at"] != consultation["recorded_at"]
        ):
            raise RecordValidationError("consultation artifact binding is invalid")
        if datetime.now(timezone.utc) >= _parse_utc_time(artifact["delete_after"]):
            raise RecordValidationError("consultation response has expired")
        return payload.decode("utf-8")

    def _customer_dir(self, customer_id: str) -> Path:
        if not re.fullmatch(r"cus_[0-9a-f]{32}", customer_id):
            raise RecordValidationError("invalid customer ID")
        return self.customers_dir / customer_id

    def _read(self, path: Path) -> tuple[dict[str, Any], str]:
        if not path.is_file() or path.is_symlink():
            raise CustomerStoreError("record not found")
        _assert_private_path(path, directory=False)
        return parse_record(path.read_text(encoding="utf-8"))

    def _write(
        self,
        path: Path,
        metadata: dict[str, Any],
        body: str = "",
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        text = serialize_record(metadata, body)
        _atomic_write(path, text, overwrite=overwrite)
        saved, saved_body = self._read(path)
        if saved_body != body:
            raise CustomerStoreError("record readback mismatch")
        return saved

    def _base_metadata(self, record_type: str, record_id: str, customer_id: str) -> dict[str, Any]:
        timestamp = _now()
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": record_type,
            "record_id": record_id,
            "customer_id": customer_id,
            "revision": 1,
            "created_at": timestamp,
            "updated_at": timestamp,
            "record_digest": "0" * 64,
        }

    def _audit(
        self,
        operation: str,
        record_type: str,
        record_id: str,
        path: Path,
        *,
        revision_before: int | None = None,
        revision_after: int | None = None,
        retention_action: str | None = None,
        change_id: str | None = None,
    ) -> None:
        logger = self.logger or AuditLogger(self.root)
        self.logger = logger
        logger.log(
            actor=_local_actor(),
            mode="customer",
            operation=operation,
            target_path=str(path),
            policy_result="allow",
            approval_status="confirmed",
            execution_result="success",
            entity_type=record_type,
            entity_ref=opaque_entity_ref(record_type, record_id),
            change_id=change_id or str(uuid.uuid4()),
            revision_before=revision_before,
            revision_after=revision_after,
            retention_action=retention_action,
        )

    def _active_names(self, *, exclude_customer_id: str | None = None) -> dict[str, str]:
        names: dict[str, str] = {}
        if not self.customers_dir.exists():
            return names
        for path in self.customers_dir.glob("cus_*/customer.md"):
            metadata, _ = self._read(path)
            if metadata["status"] != "active" or metadata["customer_id"] == exclude_customer_id:
                continue
            for value in [metadata["display_name"], *metadata["aliases"]]:
                normalized = normalize_customer_name(value)
                if normalized in names and names[normalized] != metadata["customer_id"]:
                    raise AliasConflictError("existing customer aliases are ambiguous")
                names[normalized] = metadata["customer_id"]
        return names

    def _check_name_conflicts(
        self,
        display_name: str,
        aliases: list[str],
        *,
        exclude_customer_id: str | None = None,
    ) -> None:
        normalized = [normalize_customer_name(value) for value in [display_name, *aliases]]
        if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
            raise AliasConflictError("customer names and aliases must be unique and non-empty")
        existing = self._active_names(exclude_customer_id=exclude_customer_id)
        if any(value in existing for value in normalized):
            raise AliasConflictError("customer name or alias already exists")

    def create_customer(
        self,
        *,
        display_name: str,
        aliases: list[str] | None = None,
        occupation: str = "",
        industry: str = "",
    ) -> dict[str, Any]:
        self.initialize()
        aliases = list(aliases or [])
        with self._write_lock():
            self._check_name_conflicts(display_name, aliases)
            customer_id = new_record_id("customer")
            metadata = self._base_metadata("customer", customer_id, customer_id)
            metadata.update(
                display_name=display_name,
                aliases=aliases,
                occupation=occupation,
                industry=industry,
                status="active",
            )
            path = self._customer_dir(customer_id) / "customer.md"
            saved = self._write(path, metadata)
            self._audit("customer-create", "customer", customer_id, path, revision_after=1)
            return saved

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        metadata, _ = self._read(self._customer_dir(customer_id) / "customer.md")
        return metadata

    def list_customers(self) -> list[dict[str, Any]]:
        if not self.customers_dir.exists():
            return []
        return [self._read(path)[0] for path in sorted(self.customers_dir.glob("cus_*/customer.md"))]

    def resolve_customer(self, name: str) -> str | None:
        normalized = normalize_customer_name(name)
        matches: list[str] = []
        for customer in self.list_customers():
            if customer["status"] != "active":
                continue
            values = [customer["display_name"], *customer["aliases"]]
            if normalized in {normalize_customer_name(value) for value in values}:
                matches.append(customer["customer_id"])
        if len(matches) > 1:
            raise AliasConflictError("customer name is ambiguous")
        return matches[0] if matches else None

    def update_customer(
        self,
        customer_id: str,
        *,
        expected_revision: int,
        display_name: str,
        aliases: list[str],
        occupation: str,
        industry: str,
        status: str,
    ) -> dict[str, Any]:
        path = self._customer_dir(customer_id) / "customer.md"
        with self._write_lock():
            current, body = self._read(path)
            if current["revision"] != expected_revision:
                raise RevisionConflictError("customer revision changed")
            if status == "active":
                self._check_name_conflicts(display_name, aliases, exclude_customer_id=customer_id)
            updated = dict(current)
            updated.update(
                display_name=display_name,
                aliases=list(aliases),
                occupation=occupation,
                industry=industry,
                status=status,
                revision=expected_revision + 1,
                updated_at=_now(),
            )
            saved = self._write(path, updated, body, overwrite=True)
            self._audit(
                "customer-update",
                "customer",
                customer_id,
                path,
                revision_before=expected_revision,
                revision_after=expected_revision + 1,
            )
            return saved

    def add_information(
        self,
        customer_id: str,
        *,
        text: str,
        kind: str,
        observed_at: str,
        source_refs: list[str] | None = None,
        supersedes: str | None = None,
        origin: str = "manual",
    ) -> dict[str, Any]:
        self.get_customer(customer_id)
        if origin == "ai_assisted":
            raise RecordValidationError("AI proposals require accept-and-apply")
        _parse_time(observed_at)
        if not text.strip():
            raise RecordValidationError("information body is empty")
        with self._write_lock():
            if supersedes is not None:
                old_path = self._customer_dir(customer_id) / "information" / f"{supersedes}.md"
                self._read(old_path)
            record_id = new_record_id("information")
            metadata = self._base_metadata("information", record_id, customer_id)
            metadata.update(
                kind=kind,
                observed_at=observed_at,
                source_refs=list(source_refs or []),
                information_status="current",
                supersedes=supersedes,
                origin=origin,
            )
            path = self._customer_dir(customer_id) / "information" / f"{record_id}.md"
            saved = self._write(path, metadata, text)
            self._audit("information-create", "information", record_id, path, revision_after=1)
            return saved

    def current_information(self, customer_id: str) -> list[tuple[dict[str, Any], str]]:
        info_dir = self._customer_dir(customer_id) / "information"
        records = [self._read(path) for path in sorted(info_dir.glob("inf_*.md"))] if info_dir.exists() else []
        superseded = {metadata["supersedes"] for metadata, _ in records if metadata["supersedes"]}
        return [(metadata, body) for metadata, body in records if metadata["record_id"] not in superseded]

    def list_meetings(self, customer_id: str) -> list[dict[str, Any]]:
        self.get_customer(customer_id)
        meetings_dir = self._customer_dir(customer_id) / "meetings"
        if not meetings_dir.exists():
            return []
        meetings = [
            self._read(path)[0] for path in sorted(meetings_dir.glob("mtg_*.md"))
        ]
        return sorted(
            meetings,
            key=lambda meeting: (
                meeting["occurred_at"],
                meeting["created_at"],
                meeting["record_id"],
            ),
            reverse=True,
        )

    def create_menu(
        self,
        customer_id: str,
        *,
        name: str,
        menu_kind: str,
        description: str,
        initial_status: str = "proposed",
        source_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        self.get_customer(customer_id)
        with self._write_lock():
            menu_id = new_record_id("menu")
            metadata = self._base_metadata("menu", menu_id, customer_id)
            metadata.update(menu_kind=menu_kind, name=name)
            path = self._customer_dir(customer_id) / "menus" / f"{menu_id}.md"
            saved = self._write(path, metadata, description)
            self._add_menu_event_locked(
                customer_id,
                menu_id,
                to_status=initial_status,
                effective_at=_now(),
                source_refs=list(source_refs or []),
            )
            self._audit("menu-create", "menu", menu_id, path, revision_after=1)
            return saved

    def _menu_events(self, customer_id: str, menu_id: str) -> list[dict[str, Any]]:
        events_dir = self._customer_dir(customer_id) / "menu-events"
        events = []
        sequences = set()
        if events_dir.exists():
            for path in events_dir.glob("mev_*.md"):
                metadata, _ = self._read(path)
                if metadata["menu_id"] == menu_id:
                    if metadata["event_sequence"] in sequences:
                        raise RecordValidationError("duplicate menu event sequence")
                    sequences.add(metadata["event_sequence"])
                    events.append(metadata)
        return sorted(
            events,
            key=lambda value: (value["effective_at"], value["event_sequence"]),
        )

    def menu_status(self, customer_id: str, menu_id: str) -> str | None:
        events = self._menu_events(customer_id, menu_id)
        return events[-1]["to_status"] if events else None

    def list_menus(self, customer_id: str) -> list[dict[str, Any]]:
        self.get_customer(customer_id)
        menus_dir = self._customer_dir(customer_id) / "menus"
        if not menus_dir.exists():
            return []
        menus = []
        for path in sorted(menus_dir.glob("mnu_*.md")):
            metadata, description = self._read(path)
            status = self.menu_status(customer_id, metadata["record_id"])
            if status is None:
                raise RecordValidationError("menu has no state event")
            metadata["description"] = description
            metadata["menu_status"] = status
            menus.append(metadata)
        return sorted(
            menus,
            key=lambda menu: (menu["created_at"], menu["record_id"]),
        )

    def _add_menu_event_locked(
        self,
        customer_id: str,
        menu_id: str,
        *,
        to_status: str,
        effective_at: str,
        source_refs: list[str],
    ) -> dict[str, Any]:
        self._read(self._customer_dir(customer_id) / "menus" / f"{menu_id}.md")
        _parse_time(effective_at)
        events = self._menu_events(customer_id, menu_id)
        current = events[-1]["to_status"] if events else None
        if to_status not in _MENU_TRANSITIONS.get(current, set()):
            raise RecordValidationError("invalid menu state transition")
        event_sequence = max(
            (event["event_sequence"] for event in events), default=0
        ) + 1
        event_id = new_record_id("menu_event")
        metadata = self._base_metadata("menu_event", event_id, customer_id)
        metadata.update(
            menu_id=menu_id,
            from_status=current,
            to_status=to_status,
            effective_at=effective_at,
            event_sequence=event_sequence,
            source_refs=source_refs,
        )
        path = self._customer_dir(customer_id) / "menu-events" / f"{event_id}.md"
        saved = self._write(path, metadata)
        self._audit("menu-event-create", "menu_event", event_id, path, revision_after=1)
        return saved

    def add_menu_event(
        self,
        customer_id: str,
        menu_id: str,
        *,
        to_status: str,
        effective_at: str | None = None,
        source_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._write_lock():
            return self._add_menu_event_locked(
                customer_id,
                menu_id,
                to_status=to_status,
                effective_at=effective_at or _now(),
                source_refs=list(source_refs or []),
            )

    def create_meeting_draft(
        self,
        customer_id: str,
        *,
        occurred_at: str,
        title: str,
        summary: str,
        details: str,
        decisions: list[str] | None = None,
        next_actions: list[str] | None = None,
        source_refs: list[str] | None = None,
    ) -> dict[str, Any]:
        self.get_customer(customer_id)
        _parse_time(occurred_at)
        with self._write_lock():
            meeting_id = new_record_id("meeting")
            metadata = self._base_metadata("meeting", meeting_id, customer_id)
            metadata.update(
                occurred_at=occurred_at,
                meeting_status="draft",
                title=title,
                summary=summary,
                details=details,
                decisions=list(decisions or []),
                next_actions=list(next_actions or []),
                source_refs=list(source_refs or []),
                confirmed_by=None,
                confirmed_at=None,
            )
            path = self._customer_dir(customer_id) / "meetings" / f"{meeting_id}.md"
            saved = self._write(path, metadata)
            self._audit("meeting-draft-create", "meeting", meeting_id, path, revision_after=1)
            return saved

    def confirm_meeting(
        self,
        customer_id: str,
        meeting_id: str,
        *,
        expected_revision: int,
        confirmed_by: str,
    ) -> dict[str, Any]:
        path = self._customer_dir(customer_id) / "meetings" / f"{meeting_id}.md"
        with self._write_lock():
            current, body = self._read(path)
            if current["revision"] != expected_revision:
                raise RevisionConflictError("meeting revision changed")
            if current["meeting_status"] != "draft":
                raise RecordValidationError("only a draft meeting can be confirmed")
            updated = dict(current)
            updated.update(
                meeting_status="confirmed",
                confirmed_by=confirmed_by,
                confirmed_at=_now(),
                revision=expected_revision + 1,
                updated_at=_now(),
            )
            saved = self._write(path, updated, body, overwrite=True)
            self._audit(
                "meeting-confirm",
                "meeting",
                meeting_id,
                path,
                revision_before=expected_revision,
                revision_after=expected_revision + 1,
            )
            return saved

    def reopen_meeting(
        self,
        customer_id: str,
        meeting_id: str,
        *,
        expected_revision: int,
    ) -> dict[str, Any]:
        """確定済みの打ち合わせを下書きに戻す（直してから、もう一度人が確定する）.

        版の番号を確かめ、確定した人・時刻を外し、監査の記録（meeting-reopen）を残す。本文は変えない。
        """
        path = self._customer_dir(customer_id) / "meetings" / f"{meeting_id}.md"
        with self._write_lock():
            current, body = self._read(path)
            if current["revision"] != expected_revision:
                raise RevisionConflictError("meeting revision changed")
            if current["meeting_status"] != "confirmed":
                raise RecordValidationError("only a confirmed meeting can be reopened")
            updated = dict(current)
            updated.update(
                meeting_status="draft",
                confirmed_by=None,
                confirmed_at=None,
                revision=expected_revision + 1,
                updated_at=_now(),
            )
            saved = self._write(path, updated, body, overwrite=True)
            self._audit(
                "meeting-reopen",
                "meeting",
                meeting_id,
                path,
                revision_before=expected_revision,
                revision_after=expected_revision + 1,
            )
            return saved

    def update_meeting_draft(
        self,
        customer_id: str,
        meeting_id: str,
        *,
        expected_revision: int,
        title: str,
        summary: str,
        details: str,
        decisions: list[str],
        next_actions: list[str],
    ) -> dict[str, Any]:
        path = self._customer_dir(customer_id) / "meetings" / f"{meeting_id}.md"
        with self._write_lock():
            current, body = self._read(path)
            if current["revision"] != expected_revision:
                raise RevisionConflictError("meeting revision changed")
            if current["meeting_status"] != "draft":
                raise RecordValidationError("confirmed meeting is immutable")
            updated = dict(current)
            updated.update(
                title=title,
                summary=summary,
                details=details,
                decisions=list(decisions),
                next_actions=list(next_actions),
                revision=expected_revision + 1,
                updated_at=_now(),
            )
            saved = self._write(path, updated, body, overwrite=True)
            self._audit(
                "meeting-draft-update",
                "meeting",
                meeting_id,
                path,
                revision_before=expected_revision,
                revision_after=expected_revision + 1,
            )
            return saved

    def stage_proposal(
        self,
        customer_id: str,
        *,
        payload: dict[str, Any],
        provider: str = "local_ollama",
        model: str,
        model_identifier: str,
        prompt_contract_version: str,
        prompt_contract_hash: str,
        source_refs: list[str],
        input_text: str,
    ) -> dict[str, Any]:
        self.get_customer(customer_id)
        if any(type(value) is str and value.startswith("con_") for value in source_refs):
            raise RecordValidationError(
                "consultation sources require the consultation extraction boundary"
            )
        payload_bytes = _serialize_proposal_payload(payload)
        proposal_id = new_record_id("proposal")
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
        with self._write_lock():
            payload_path = self.staging_dir / "proposals" / f"{proposal_id}.json"
            change_id = str(uuid.uuid4())
            pending_path = self.staging_dir / "proposals" / f".{proposal_id}.{change_id}.json"
            _atomic_write(pending_path, payload_bytes.decode("utf-8"), overwrite=False)
            intent = {
                "schema_version": 1,
                "operation": "stage_proposal",
                "customer_id": customer_id,
                "proposal_id": proposal_id,
                "payload_hash": payload_hash,
                "change_id": change_id,
            }
            intent_path = self.transactions_dir / f"{change_id}.json"
            _atomic_write(
                intent_path,
                json.dumps(intent, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                overwrite=False,
            )
            metadata = self._base_metadata("proposal", proposal_id, customer_id)
            metadata.update(
                provider=provider,
                model=model,
                model_identifier=model_identifier,
                prompt_contract_version=prompt_contract_version,
                prompt_contract_hash=prompt_contract_hash,
                input_digest=hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
                source_refs=list(source_refs),
                payload_hash=payload_hash,
                proposal_status="pending",
                reviewed_by=None,
                reviewed_at=None,
                human_edited=False,
                applied_record_id=None,
            )
            path = self._customer_dir(customer_id) / "proposals" / f"{proposal_id}.md"
            saved = self._write(path, metadata)
            os.replace(pending_path, payload_path)
            _fsync_directory(payload_path.parent)
            self._audit(
                "proposal-stage", "proposal", proposal_id, path,
                revision_after=1, change_id=change_id,
            )
            intent_path.unlink()
            _fsync_directory(self.transactions_dir)
            return saved

    def get_consultation_extraction_input(
        self, customer_id: str, consultation_id: str
    ) -> str:
        """Return a revalidated raw response only when extraction is available."""
        with self._write_lock():
            return self._consultation_extraction_input_locked(
                customer_id, consultation_id, require_no_proposal=True
            )

    def consultation_relation_state(
        self, customer_id: str, consultation_id: str
    ) -> str:
        """Validate and derive the consultation extraction lifecycle."""
        with self._write_lock():
            if not re.fullmatch(r"con_[0-9a-f]{32}", consultation_id):
                raise RecordValidationError("invalid consultation ID")
            consultation_path = (
                self._customer_dir(customer_id)
                / "consultations"
                / f"{consultation_id}.md"
            )
            consultation, body = self._read(consultation_path)
            if consultation["customer_id"] != customer_id or body != "":
                raise RecordValidationError("consultation record binding is invalid")
            for path in sorted(consultation_path.parent.glob("con_*.md")):
                candidate, _ = self._read(path)
                if candidate["supersedes"] == consultation_id:
                    raise RecordValidationError("consultation is not current")
            related = self._consultation_proposals_locked(customer_id, consultation_id)
            if not related:
                return "recorded"
            proposal, _ = related[0]
            information = self._proposal_information_locked(
                customer_id, proposal["record_id"]
            )
            if proposal["proposal_status"] == "pending":
                if information:
                    raise CustomerStoreError("pending proposal has applied information")
                return "proposal_pending"
            if proposal["proposal_status"] == "rejected":
                if information:
                    raise CustomerStoreError("rejected proposal has applied information")
                return "extraction_rejected"
            if (
                proposal["proposal_status"] != "accepted"
                or len(information) != 1
                or information[0][0]["record_id"] != proposal["applied_record_id"]
            ):
                raise CustomerStoreError("accepted consultation relation is incomplete")
            payload_path = (
                self.staging_dir / "proposals" / f"{proposal['record_id']}.json"
            )
            if payload_path.exists():
                payload, _ = self._proposal_payload_locked(
                    customer_id, proposal["record_id"], proposal
                )
                if information[0][1] != self._information_text(payload):
                    raise CustomerStoreError(
                        "accepted consultation payload relation is invalid"
                    )
            return "completed"

    def stage_consultation_extraction_proposal(
        self,
        customer_id: str,
        consultation_id: str,
        *,
        payload: dict[str, Any],
        provider: str,
        model: str,
        model_identifier: str,
        prompt_contract_version: str,
        prompt_contract_hash: str,
    ) -> dict[str, Any]:
        """Stage one proposal with a forced consultation source reference."""
        payload_bytes = _serialize_proposal_payload(payload)
        if self._information_text(payload) != payload["content"]["text"]:
            raise RecordValidationError("proposal is not strict information/create")
        with self._write_lock():
            input_text = self._consultation_extraction_input_locked(
                customer_id, consultation_id, require_no_proposal=True
            )
            proposal_id = new_record_id("proposal")
            payload_hash = hashlib.sha256(payload_bytes).hexdigest()
            change_id = str(uuid.uuid4())
            payload_path = self.staging_dir / "proposals" / f"{proposal_id}.json"
            pending_path = self.staging_dir / "proposals" / f".{proposal_id}.{change_id}.json"
            intent = {
                "schema_version": 1,
                "operation": "stage_consultation_proposal",
                "customer_id": customer_id,
                "consultation_id": consultation_id,
                "proposal_id": proposal_id,
                "payload_hash": payload_hash,
                "change_id": change_id,
            }
            intent_path = self.transactions_dir / f"{change_id}.json"
            _atomic_write(
                intent_path,
                json.dumps(intent, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                overwrite=False,
            )
            _atomic_write(pending_path, payload_bytes.decode("utf-8"), overwrite=False)
            metadata = self._base_metadata("proposal", proposal_id, customer_id)
            metadata.update(
                provider=provider,
                model=model,
                model_identifier=model_identifier,
                prompt_contract_version=prompt_contract_version,
                prompt_contract_hash=prompt_contract_hash,
                input_digest=hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
                source_refs=[consultation_id],
                payload_hash=payload_hash,
                proposal_status="pending",
                reviewed_by=None,
                reviewed_at=None,
                human_edited=False,
                applied_record_id=None,
            )
            path = self._customer_dir(customer_id) / "proposals" / f"{proposal_id}.md"
            saved = self._write(path, metadata)
            os.replace(pending_path, payload_path)
            _fsync_directory(payload_path.parent)
            self._audit(
                "consultation-extraction-stage", "proposal", proposal_id, path,
                revision_after=1, change_id=change_id,
            )
            intent_path.unlink()
            _fsync_directory(self.transactions_dir)
            return saved

    def get_proposal_payload(
        self, customer_id: str, proposal_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not re.fullmatch(r"prp_[0-9a-f]{32}", proposal_id):
            raise RecordValidationError("invalid proposal ID")
        path = self._customer_dir(customer_id) / "proposals" / f"{proposal_id}.md"
        metadata, _ = self._read(path)
        payload_path = self.staging_dir / "proposals" / f"{proposal_id}.json"
        if not payload_path.is_file() or payload_path.is_symlink():
            raise RecordValidationError("proposal payload is unavailable")
        _assert_private_path(payload_path, directory=False)
        payload_bytes = payload_path.read_bytes()
        if hashlib.sha256(payload_bytes).hexdigest() != metadata["payload_hash"]:
            raise RecordValidationError("proposal payload digest mismatch")
        try:
            payload = json.loads(payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecordValidationError("proposal payload is invalid") from exc
        if _serialize_proposal_payload(payload) != payload_bytes:
            raise RecordValidationError("proposal payload is not canonical JSON")
        return metadata, payload

    def get_proposal_metadata(
        self, customer_id: str, proposal_id: str
    ) -> dict[str, Any]:
        """Return validated canonical proposal metadata without staging data."""
        if not re.fullmatch(r"prp_[0-9a-f]{32}", proposal_id):
            raise RecordValidationError("invalid proposal ID")
        path = self._customer_dir(customer_id) / "proposals" / f"{proposal_id}.md"
        metadata, body = self._read(path)
        if body != "":
            raise RecordValidationError("proposal record body must be empty")
        return metadata

    def update_pending_proposal_payload(
        self,
        customer_id: str,
        proposal_id: str,
        *,
        expected_revision: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        payload_bytes = _serialize_proposal_payload(payload)
        if not re.fullmatch(r"prp_[0-9a-f]{32}", proposal_id):
            raise RecordValidationError("invalid proposal ID")
        path = self._customer_dir(customer_id) / "proposals" / f"{proposal_id}.md"
        with self._write_lock():
            current, body = self._read(path)
            if current["revision"] != expected_revision:
                raise RevisionConflictError("proposal revision changed")
            if current["proposal_status"] != "pending":
                raise RecordValidationError("only pending proposals can be edited")
            consultation_refs = [
                value for value in current["source_refs"] if value.startswith("con_")
            ]
            if consultation_refs:
                if current["source_refs"] != consultation_refs or len(consultation_refs) != 1:
                    raise RecordValidationError("consultation proposal source binding is invalid")
                related = self._consultation_proposals_locked(
                    customer_id, consultation_refs[0]
                )
                if len(related) != 1 or related[0][0]["record_id"] != proposal_id:
                    raise RecordValidationError("consultation proposal is not unique")
                self._information_text(payload)

            payload_path = self.staging_dir / "proposals" / f"{proposal_id}.json"
            if not payload_path.is_file() or payload_path.is_symlink():
                raise RecordValidationError("proposal payload is unavailable")
            previous_bytes = payload_path.read_bytes()
            if hashlib.sha256(previous_bytes).hexdigest() != current["payload_hash"]:
                raise RecordValidationError("proposal payload digest mismatch")
            try:
                previous_payload = json.loads(previous_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RecordValidationError("proposal payload is invalid") from exc
            if (
                payload["schema_version"] != previous_payload.get("schema_version")
                or payload["target_type"] != previous_payload.get("target_type")
                or payload["operation"] != previous_payload.get("operation")
            ):
                raise RecordValidationError("proposal edit cannot change its operation")

            timestamp = _now()
            change_id = str(uuid.uuid4())
            previous_hash = current["payload_hash"]
            new_hash = hashlib.sha256(payload_bytes).hexdigest()
            updated = dict(current)
            updated.update(
                payload_hash=new_hash,
                human_edited=True,
                revision=expected_revision + 1,
                updated_at=timestamp,
            )
            pending_path = self.staging_dir / "proposals" / f".{proposal_id}.{change_id}.json"
            _atomic_write(pending_path, payload_bytes.decode("utf-8"), overwrite=False)
            intent = {
                "schema_version": 1,
                "operation": "edit_payload",
                "customer_id": customer_id,
                "proposal_id": proposal_id,
                "proposal_revision": expected_revision,
                "previous_payload_hash": previous_hash,
                "new_payload_hash": new_hash,
                "change_id": change_id,
            }
            intent_path = self.transactions_dir / f"{change_id}.json"
            _atomic_write(
                intent_path,
                json.dumps(intent, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                overwrite=False,
            )
            saved = self._write(path, updated, body, overwrite=True)
            os.replace(pending_path, payload_path)
            _fsync_directory(payload_path.parent)
            self._audit(
                "proposal-edit",
                "proposal",
                proposal_id,
                path,
                revision_before=expected_revision,
                revision_after=expected_revision + 1,
                change_id=change_id,
            )
            intent_path.unlink()
            _fsync_directory(self.transactions_dir)
            return saved

    def stage_artifact(
        self,
        customer_id: str,
        *,
        kind: str,
        data: bytes,
        retention: str | None = None,
        created_at: str | None = None,
    ) -> str:
        self.get_customer(customer_id)
        if kind not in ARTIFACT_DEFAULT_RETENTION:
            raise RecordValidationError("invalid artifact kind")
        policy = retention or ARTIFACT_DEFAULT_RETENTION[kind]
        if policy not in RETENTION_DAYS:
            raise RecordValidationError("invalid retention policy")
        artifact_id = new_record_id("artifact")
        created = created_at or _now()
        created_dt = _parse_time(created)
        delete_after = (created_dt + timedelta(days=RETENTION_DAYS[policy])).isoformat()
        artifact_dir = self.staging_dir / "artifacts" / artifact_id
        with self._write_lock():
            _safe_mkdir_tree(artifact_dir, self.root.parent)
            data_path = artifact_dir / "payload.bin"
            fd = os.open(data_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                shutil.rmtree(artifact_dir, ignore_errors=True)
                raise
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "artifact_id": artifact_id,
                "customer_id": customer_id,
                "kind": kind,
                "retention": policy,
                "created_at": created,
                "delete_after": delete_after,
                "canonical_confirmed": False,
                "payload_hash": hashlib.sha256(data).hexdigest(),
            }
            _atomic_write(
                artifact_dir / "metadata.json",
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                overwrite=False,
            )
            return artifact_id

    def reject_proposal(
        self,
        customer_id: str,
        proposal_id: str,
        *,
        expected_revision: int,
        expected_customer_revision: int,
        confirmation: object,
        reviewed_at: str | None = None,
    ) -> dict[str, Any]:
        if confirmation is not True:
            raise RecordValidationError("proposal rejection requires literal confirmation")
        path = self._customer_dir(customer_id) / "proposals" / f"{proposal_id}.md"
        with self._write_lock():
            customer = self.get_customer(customer_id)
            if customer["status"] != "active" or customer["revision"] != expected_customer_revision:
                raise RevisionConflictError("customer revision changed")
            current, body = self._read(path)
            if current["revision"] != expected_revision:
                raise RevisionConflictError("proposal revision changed")
            if current["proposal_status"] != "pending":
                raise RecordValidationError("proposal was already reviewed")
            self._proposal_payload_locked(customer_id, proposal_id, current)
            timestamp = reviewed_at or _now()
            _parse_time(timestamp)
            updated = dict(current)
            updated.update(
                proposal_status="rejected",
                reviewed_by=_local_actor(),
                reviewed_at=timestamp,
                applied_record_id=None,
                revision=expected_revision + 1,
                updated_at=timestamp,
            )
            saved = self._write(path, updated, body, overwrite=True)
            self._audit(
                "proposal-reject",
                "proposal",
                proposal_id,
                path,
                revision_before=expected_revision,
                revision_after=expected_revision + 1,
            )
            return saved

    def accept_and_apply_information_proposal(
        self,
        customer_id: str,
        proposal_id: str,
        *,
        expected_revision: int,
        expected_customer_revision: int,
        confirmation: object,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if confirmation is not True:
            raise RecordValidationError("proposal acceptance requires literal confirmation")
        with self._write_lock():
            return self._accept_and_apply_information_proposal_locked(
                customer_id,
                proposal_id,
                expected_revision=expected_revision,
                expected_customer_revision=expected_customer_revision,
                consultation_id=None,
            )

    def accept_and_apply_consultation_proposal(
        self,
        customer_id: str,
        consultation_id: str,
        proposal_id: str,
        *,
        expected_revision: int,
        expected_customer_revision: int,
        confirmation: object,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if confirmation is not True:
            raise RecordValidationError("proposal acceptance requires literal confirmation")
        with self._write_lock():
            return self._accept_and_apply_information_proposal_locked(
                customer_id,
                proposal_id,
                expected_revision=expected_revision,
                expected_customer_revision=expected_customer_revision,
                consultation_id=consultation_id,
            )

    def _accept_and_apply_information_proposal_locked(
        self,
        customer_id: str,
        proposal_id: str,
        *,
        expected_revision: int,
        expected_customer_revision: int,
        consultation_id: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        proposal_path = self._customer_dir(customer_id) / "proposals" / f"{proposal_id}.md"
        customer = self.get_customer(customer_id)
        if customer["status"] != "active" or customer["revision"] != expected_customer_revision:
            raise RevisionConflictError("customer revision changed")
        proposal, proposal_body = self._read(proposal_path)
        if proposal["revision"] != expected_revision:
            raise RevisionConflictError("proposal revision changed")
        if proposal["proposal_status"] != "pending":
            raise RecordValidationError("proposal was already reviewed")
        if consultation_id is None:
            if any(value.startswith("con_") for value in proposal["source_refs"]):
                raise RecordValidationError(
                    "consultation proposals require the consultation acceptance boundary"
                )
        else:
            if proposal["source_refs"] != [consultation_id]:
                raise RecordValidationError("consultation proposal source binding is invalid")
            related = self._consultation_proposals_locked(customer_id, consultation_id)
            if len(related) != 1 or related[0][0]["record_id"] != proposal_id:
                raise RecordValidationError("consultation proposal is not unique")
            self._consultation_extraction_input_locked(
                customer_id, consultation_id, require_no_proposal=False
            )
        payload, _ = self._proposal_payload_locked(customer_id, proposal_id, proposal)
        text = self._information_text(payload)
        information_id = new_record_id("information")
        change_id = str(uuid.uuid4())
        reviewed_at = _now()
        reviewed_by = _local_actor()
        intent = {
            "schema_version": 1,
            "operation": "accept_apply",
            "customer_id": customer_id,
            "customer_revision": expected_customer_revision,
            "proposal_id": proposal_id,
            "proposal_revision": expected_revision,
            "payload_hash": proposal["payload_hash"],
            "information_id": information_id,
            "change_id": change_id,
            "reviewed_by": reviewed_by,
            "reviewed_at": reviewed_at,
        }
        intent_path = self.transactions_dir / f"{change_id}.json"
        _atomic_write(
            intent_path,
            json.dumps(intent, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            overwrite=False,
        )
        information = self._base_metadata("information", information_id, customer_id)
        information.update(
            kind="ai-note",
            observed_at=reviewed_at,
            source_refs=[proposal_id],
            information_status="current",
            supersedes=None,
            origin="ai_assisted",
        )
        information_path = self._customer_dir(customer_id) / "information" / f"{information_id}.md"
        saved_information = self._write(information_path, information, text)
        updated = dict(proposal)
        updated.update(
            proposal_status="accepted",
            reviewed_by=reviewed_by,
            reviewed_at=reviewed_at,
            applied_record_id=information_id,
            revision=expected_revision + 1,
            updated_at=reviewed_at,
        )
        saved_proposal = self._write(proposal_path, updated, proposal_body, overwrite=True)
        self._audit(
            "proposal-accept-apply",
            "proposal",
            proposal_id,
            proposal_path,
            revision_before=expected_revision,
            revision_after=expected_revision + 1,
            change_id=change_id,
        )
        intent_path.unlink()
        _fsync_directory(self.transactions_dir)
        return saved_proposal, saved_information

    def reap_expired_proposal_payloads(self, *, now: datetime | None = None) -> list[str]:
        current = now or datetime.now(timezone.utc)
        deleted: list[str] = []
        if not self.customers_dir.exists():
            return deleted
        with self._write_lock():
            for path in sorted(self.customers_dir.glob("cus_*/proposals/prp_*.md")):
                metadata, _ = self._read(path)
                status = metadata["proposal_status"]
                baseline = metadata["reviewed_at"] or metadata["created_at"]
                expires_at = _parse_time(baseline) + timedelta(
                    days=PROPOSAL_PAYLOAD_RETENTION[status]
                )
                payload_path = self.staging_dir / "proposals" / f"{metadata['record_id']}.json"
                if payload_path.exists() and current >= expires_at:
                    payload_path.unlink()
                    _fsync_directory(payload_path.parent)
                    self._audit(
                        "proposal-payload-delete",
                        "proposal",
                        metadata["record_id"],
                        payload_path,
                        retention_action=f"{status}-expired",
                    )
                    deleted.append(metadata["record_id"])
        return deleted

    def confirm_artifact(self, artifact_id: str, *, now: datetime | None = None) -> bool:
        artifact_dir = self.staging_dir / "artifacts" / artifact_id
        metadata_path = artifact_dir / "metadata.json"
        if not metadata_path.is_file() or metadata_path.is_symlink():
            raise CustomerStoreError("artifact not found")
        with self._write_lock():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["canonical_confirmed"] = True
            _atomic_write(
                metadata_path,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                overwrite=True,
            )
            if metadata["retention"] == "immediate":
                self._delete_artifact_locked(artifact_id, "expired-after-confirmation")
                return True
            return self._reap_artifact_locked(artifact_id, now=now)

    def _delete_artifact_locked(self, artifact_id: str, action: str) -> None:
        artifact_dir = self.staging_dir / "artifacts" / artifact_id
        shutil.rmtree(artifact_dir)
        self._audit(
            "artifact-delete",
            "artifact",
            artifact_id,
            artifact_dir,
            retention_action=action,
        )

    def _reap_artifact_locked(self, artifact_id: str, *, now: datetime | None = None) -> bool:
        artifact_dir = self.staging_dir / "artifacts" / artifact_id
        metadata_path = artifact_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("kind") == "consultation_response":
            return False
        current = now or datetime.now(timezone.utc)
        if metadata["canonical_confirmed"] and current >= _parse_time(metadata["delete_after"]):
            self._delete_artifact_locked(artifact_id, "expired")
            return True
        return False

    def reap_expired_artifacts(self, *, now: datetime | None = None) -> list[str]:
        deleted = self.reap_expired_consultation_responses(now=now)
        artifacts_dir = self.staging_dir / "artifacts"
        if not artifacts_dir.exists():
            return deleted
        with self._write_lock():
            for artifact_dir in sorted(artifacts_dir.glob("art_*")):
                if self._reap_artifact_locked(artifact_dir.name, now=now):
                    deleted.append(artifact_dir.name)
        return deleted

    def reap_expired_consultation_responses(
        self, *, now: datetime | None = None
    ) -> list[str]:
        current = now or datetime.now(timezone.utc)
        deleted: list[str] = []
        with self._write_lock(recover=False):
            intent_paths = sorted(self.transactions_dir.glob("*.json"))
            deletion_intents: list[tuple[Path, dict[str, Any]]] = []
            for intent_path in intent_paths:
                if intent_path.is_symlink() or not intent_path.is_file():
                    raise CustomerStoreError("invalid consultation response deletion intent")
                _assert_private_path(intent_path, directory=False)
                try:
                    intent = json.loads(intent_path.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise CustomerStoreError(
                        "invalid consultation response deletion intent"
                    ) from exc
                if (
                    type(intent) is not dict
                    or intent.get("operation") != "delete_consultation_response"
                ):
                    raise CustomerStoreError(
                        "consultation response deletion is blocked by another transaction"
                    )
                deletion_intents.append((intent_path, intent))

            artifact_ids = [intent.get("artifact_id") for _, intent in deletion_intents]
            if any(type(artifact_id) is not str for artifact_id in artifact_ids):
                raise CustomerStoreError("invalid consultation response deletion intent")
            if len(artifact_ids) != len(set(artifact_ids)):
                raise CustomerStoreError("duplicate consultation response deletion intent")
            for intent_path, intent in deletion_intents:
                self._recover_consultation_response_delete_locked(intent_path, intent)
                deleted.append(intent["artifact_id"])

            for consultation_path in sorted(
                self.customers_dir.glob("cus_*/consultations/con_*.md")
            ):
                consultation, body = self._read(consultation_path)
                if body != "":
                    raise CustomerStoreError("consultation record body must be empty")
                customer_id = consultation["customer_id"]
                consultation_id = consultation["record_id"]
                artifact_id = consultation["inbound_artifact_ref"]
                artifact_dir = self.staging_dir / "artifacts" / artifact_id
                if not artifact_dir.exists() and not artifact_dir.is_symlink():
                    continue
                artifact, _, _ = self._consultation_artifact_locked(
                    artifact_id,
                    expected_customer_id=customer_id,
                    expected_consultation_id=consultation_id,
                    expected_confirmed=True,
                )
                if (
                    artifact["payload_hash"] != consultation["inbound_response_digest"]
                    or artifact["provider_kind"] != consultation["provider_kind"]
                    or artifact["ingest_change_id"] != consultation["ingest_change_id"]
                    or artifact["recorded_at"] != consultation["recorded_at"]
                ):
                    raise CustomerStoreError("consultation artifact binding is invalid")
                if current < _parse_utc_time(artifact["delete_after"]):
                    continue

                change_id = str(uuid.uuid4())
                intent = {
                    "schema_version": 1,
                    "operation": "delete_consultation_response",
                    "customer_id": customer_id,
                    "consultation_id": consultation_id,
                    "artifact_id": artifact_id,
                    "delete_after": artifact["delete_after"],
                    "change_id": change_id,
                }
                intent_path = self.transactions_dir / f"{change_id}.json"
                _atomic_write(
                    intent_path,
                    json.dumps(
                        intent,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    overwrite=False,
                )
                self._recover_consultation_response_delete_locked(intent_path, intent)
                deleted.append(artifact_id)
        return deleted

    def record_consultation_response(
        self,
        customer_id: str,
        *,
        expected_customer_revision: int,
        provider_kind: str,
        normalized_text: str,
        operation_id: str,
        confirmation: object,
    ) -> dict[str, Any]:
        if confirmation is not True:
            raise RecordValidationError(
                "consultation recording requires literal confirmation"
            )
        if type(provider_kind) is not str or provider_kind not in CONSULTATION_PROVIDER_KINDS:
            raise RecordValidationError("invalid consultation provider kind")
        if type(operation_id) is not str or not _CHANGE_ID_RE.fullmatch(operation_id):
            raise RecordValidationError("operation ID must be a canonical UUID4")
        if type(expected_customer_revision) is not int or expected_customer_revision < 1:
            raise RecordValidationError("invalid expected customer revision")
        text = normalize_consultation_response_text(normalized_text)
        if text != normalized_text:
            raise RecordValidationError("consultation response must already be normalized")
        payload_bytes = text.encode("utf-8")
        response_digest = hashlib.sha256(payload_bytes).hexdigest()
        with self._write_lock():
            completed = self._consultation_for_change_locked(operation_id)
            if completed is not None:
                consultation, _ = completed
                if (
                    consultation["customer_id"] != customer_id
                    or consultation["provider_kind"] != provider_kind
                    or consultation["inbound_response_digest"] != response_digest
                ):
                    raise CustomerStoreError("operation ID conflict with different binding")
                artifact_dir = (
                    self.staging_dir
                    / "artifacts"
                    / consultation["inbound_artifact_ref"]
                )
                if artifact_dir.exists() or artifact_dir.is_symlink():
                    metadata, _, _ = self._consultation_artifact_locked(
                        consultation["inbound_artifact_ref"],
                        expected_customer_id=customer_id,
                        expected_consultation_id=consultation["record_id"],
                        expected_confirmed=True,
                    )
                    if (
                        metadata["ingest_change_id"] != operation_id
                        or metadata["provider_kind"] != provider_kind
                        or metadata["payload_hash"] != response_digest
                    ):
                        raise CustomerStoreError("completed consultation artifact mismatch")
                return consultation

            customer = self.get_customer(customer_id)
            if customer["status"] != "active":
                raise RecordValidationError("customer is not active")
            if customer["revision"] != expected_customer_revision:
                raise RevisionConflictError("customer revision changed")

            recorded_by_ref = opaque_entity_ref("actor", _local_actor())
            recorded_at = _now()
            change_id = operation_id
            artifact_id = new_record_id("artifact")
            consultation_id = new_record_id("consultation")
            intent = {
                "schema_version": 1,
                "operation": "record_consultation",
                "customer_id": customer_id,
                "customer_revision": expected_customer_revision,
                "provider_kind": provider_kind,
                "artifact_id": artifact_id,
                "consultation_id": consultation_id,
                "change_id": change_id,
                "response_digest": response_digest,
                "payload_bytes": len(payload_bytes),
                "recorded_by_ref": recorded_by_ref,
                "recorded_at": recorded_at,
            }
            intent_path = self.transactions_dir / f"{change_id}.json"
            _atomic_write(
                intent_path,
                json.dumps(intent, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                overwrite=False,
            )
            intent_created = True
            artifact_dir = self.staging_dir / "artifacts" / artifact_id
            pending_dir = (
                self.staging_dir / "artifacts" / f".{artifact_id}.{change_id}.pending"
            )
            try:
                _safe_mkdir_tree(pending_dir, self.root.parent)
                _atomic_write(pending_dir / "payload.txt", text, overwrite=False)
                delete_after = (
                    _parse_utc_time(recorded_at) + timedelta(days=30)
                ).isoformat()
                artifact_metadata = {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "consultation_response",
                    "artifact_id": artifact_id,
                    "consultation_id": consultation_id,
                    "customer_id": customer_id,
                    "provider_kind": provider_kind,
                    "ingest_change_id": change_id,
                    "recorded_at": recorded_at,
                    "delete_after": delete_after,
                    "payload_bytes": len(payload_bytes),
                    "payload_hash": response_digest,
                    "canonical_confirmed": False,
                }
                _atomic_write(
                    pending_dir / "metadata.json",
                    json.dumps(artifact_metadata, ensure_ascii=False, sort_keys=True),
                    overwrite=False,
                )
                validate_consultation_artifact(
                    artifact_metadata,
                    payload_bytes,
                    expected_customer_id=customer_id,
                    expected_consultation_id=consultation_id,
                    expected_confirmed=False,
                )
                os.replace(pending_dir, artifact_dir)
                _fsync_directory(artifact_dir.parent)
                self._consultation_artifact_locked(
                    artifact_id,
                    expected_customer_id=customer_id,
                    expected_consultation_id=consultation_id,
                    expected_confirmed=False,
                )

                consultation_metadata = self._base_metadata(
                    "consultation", consultation_id, customer_id
                )
                consultation_metadata.update(
                    created_at=recorded_at,
                    updated_at=recorded_at,
                    provider_kind=provider_kind,
                    response_received_at=recorded_at,
                    inbound_artifact_ref=artifact_id,
                    inbound_response_digest=response_digest,
                    outbound_brief_ref=None,
                    outbound_brief_digest=None,
                    supersedes=None,
                    ingest_change_id=change_id,
                    recorded_by_ref=recorded_by_ref,
                    recorded_at=recorded_at,
                )
                consultation_path = (
                    self._customer_dir(customer_id)
                    / "consultations"
                    / f"{consultation_id}.md"
                )
                saved = self._write(consultation_path, consultation_metadata, "")

                artifact_metadata["canonical_confirmed"] = True
                metadata_path = artifact_dir / "metadata.json"
                _atomic_write(
                    metadata_path,
                    json.dumps(artifact_metadata, ensure_ascii=False, sort_keys=True),
                    overwrite=True,
                )
                self._consultation_artifact_locked(
                    artifact_id,
                    expected_customer_id=customer_id,
                    expected_consultation_id=consultation_id,
                    expected_confirmed=True,
                )
                self._audit(
                    "consultation-record",
                    "consultation",
                    consultation_id,
                    consultation_path,
                    revision_after=1,
                    change_id=change_id,
                )
                intent_path.unlink()
                _fsync_directory(self.transactions_dir)
                return saved
            except Exception as exc:
                if intent_created:
                    raise CustomerStoreError(
                        "consultation recording is indeterminate; reload to recover"
                    ) from exc
                raise

    def validate_all(self) -> int:
        self.validate_schema_manifest()
        count = 0
        for path in self.customers_dir.glob("cus_*/*.md"):
            self._read(path)
            count += 1
        for path in self.customers_dir.glob("cus_*/*/*.md"):
            self._read(path)
            count += 1
        return count

    def migration_plan(self, target_version: int = SCHEMA_VERSION) -> dict[str, Any]:
        self.validate_schema_manifest()
        if target_version != SCHEMA_VERSION:
            raise RecordValidationError("no registered migration for target schema")
        return {
            "from_version": SCHEMA_VERSION,
            "to_version": target_version,
            "action": "validate-only",
            "record_count": self.validate_all(),
        }

    def migrate(self, *, target_version: int = SCHEMA_VERSION, dry_run: bool = True) -> dict[str, Any]:
        plan = self.migration_plan(target_version)
        if not dry_run:
            plan = {**plan, "action": "no-op-current-schema"}
        return plan
