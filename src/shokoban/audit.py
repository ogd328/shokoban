"""監査ログ出力."""

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .utils import sanitize_audit_path


@dataclass(frozen=True)
class AuditEvent:
    timestamp: str
    actor: str
    mode: str
    operation: str
    target_path: str
    output_path: str | None
    policy_result: str
    approval_status: str
    execution_result: str
    error_reason: str | None
    session_id: str
    brief_hash: str | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    source_kind: str | None = None
    event_id: str | None = None
    entity_type: str | None = None
    entity_ref: str | None = None
    change_id: str | None = None
    revision_before: int | None = None
    revision_after: int | None = None
    retention_action: str | None = None
    error_code: str | None = None

    def to_dict(self) -> dict:
        data = {
            "timestamp": self.timestamp,
            "actor": self.actor,
            "mode": self.mode,
            "operation": self.operation,
            "target_path": self.target_path,
            "output_path": self.output_path,
            "policy_result": self.policy_result,
            "approval_status": self.approval_status,
            "execution_result": self.execution_result,
            "error_reason": self.error_reason,
            "session_id": self.session_id,
        }
        if self.brief_hash is not None:
            data["brief_hash"] = self.brief_hash
        if self.approved_by is not None:
            data["approved_by"] = self.approved_by
        if self.approved_at is not None:
            data["approved_at"] = self.approved_at
        if self.source_kind is not None:
            data["source_kind"] = self.source_kind
        for key in (
            "event_id",
            "entity_type",
            "entity_ref",
            "change_id",
            "revision_before",
            "revision_after",
            "retention_action",
            "error_code",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


def opaque_entity_ref(entity_type: str, entity_id: str) -> str:
    """Return a stable opaque reference without exposing a customer path."""
    value = f"{entity_type}:{entity_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:32]


class AuditLogger:
    """JSONL形式で監査ログを記録."""

    def __init__(self, root: Path):
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.log_dir = root / "audit-log"
        self.log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(self.log_dir, 0o700)
        self.log_path = self.log_dir / "audit.jsonl"

    def _now(self) -> str:
        return datetime.now(timezone.utc).astimezone().isoformat()

    def _atomic_replace_log(self, text: str) -> None:
        temporary = self.log_path.with_name(
            f".{self.log_path.name}.{uuid.uuid4().hex}.tmp"
        )
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.log_path)
            directory_fd = os.open(self.log_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def log(
        self,
        *,
        actor: str,
        mode: str,
        operation: str,
        target_path: str,
        output_path: str | None = None,
        policy_result: str,
        approval_status: str,
        execution_result: str,
        error_reason: str | None = None,
        brief_hash: str | None = None,
        approved_by: str | None = None,
        approved_at: str | None = None,
        source_kind: str | None = None,
        event_id: str | None = None,
        entity_type: str | None = None,
        entity_ref: str | None = None,
        change_id: str | None = None,
        revision_before: int | None = None,
        revision_after: int | None = None,
        retention_action: str | None = None,
        error_code: str | None = None,
    ) -> AuditEvent:
        safe_target_path = sanitize_audit_path(target_path) or "<unknown>"
        safe_output_path = sanitize_audit_path(output_path)
        event = AuditEvent(
            timestamp=self._now(),
            actor=actor,
            mode=mode,
            operation=operation,
            target_path=safe_target_path,
            output_path=safe_output_path,
            policy_result=policy_result,
            approval_status=approval_status,
            execution_result=execution_result,
            error_reason=error_reason,
            session_id=str(uuid.uuid4())[:8],
            brief_hash=brief_hash,
            approved_by=approved_by,
            approved_at=approved_at,
            source_kind=source_kind,
            event_id=event_id or str(uuid.uuid4()),
            entity_type=entity_type,
            entity_ref=entity_ref,
            change_id=change_id,
            revision_before=revision_before,
            revision_after=revision_after,
            retention_action=retention_action,
            error_code=error_code,
        )
        fd = os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return event

    def has_change_id(self, change_id: str) -> bool:
        """Check a durable event, tolerating only a torn final JSONL line."""
        if not self.log_path.exists():
            return False
        text = self.log_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        found = False
        for index, line in enumerate(lines):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    valid = "\n".join(lines[:index])
                    self._atomic_replace_log(valid + "\n" if valid else "")
                    return found
                raise
            if data.get("change_id") == change_id:
                found = True
        if text and not text.endswith("\n"):
            self._atomic_replace_log(text + "\n")
        return found

    def read(self, tail: int | None = None) -> list[AuditEvent]:
        if not self.log_path.exists():
            return []
        events: list[AuditEvent] = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                events.append(AuditEvent(**data))
        if tail is not None:
            events = events[-tail:]
        return events
