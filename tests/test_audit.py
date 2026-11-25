"""監査ログのテスト."""

import json
import unittest
from pathlib import Path

from shokoban.audit import AuditLogger


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__import__("tempfile").mkdtemp())
        self.logger = AuditLogger(self.tmp)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_log_contains_required_fields(self):
        ev = self.logger.log(
            actor="tester",
            mode="share",
            operation="make-brief",
            target_path="sanitized-briefs/a.md",
            output_path="sanitized-briefs/a.md",
            policy_result="allow",
            approval_status="pending",
            execution_result="success",
        )
        self.assertIsNotNone(ev.timestamp)
        self.assertEqual(ev.mode, "share")
        self.assertEqual(ev.operation, "make-brief")

    def test_log_does_not_contain_body(self):
        self.logger.log(
            actor="tester",
            mode="customer",
            operation="read",
            target_path="customer-vault/acme/confidential.md",
            output_path=None,
            policy_result="allow",
            approval_status="approved",
            execution_result="success",
        )
        log_text = (self.tmp / "audit-log" / "audit.jsonl").read_text(encoding="utf-8")
        data = json.loads(log_text.strip())
        # 監査ログに本文や顧客名、金額、契約条件が含まれないことを確認
        self.assertNotIn("生データ本文", log_text)
        self.assertNotIn("本文", data)
        self.assertNotIn("金額", data)
        self.assertNotIn("契約", data)
        self.assertEqual(data["target_path"], "customer-vault/<redacted>")
        self.assertNotIn("acme", log_text)
        self.assertNotIn("confidential", log_text)

    def test_logger_applies_common_path_redaction(self):
        self.logger.log(
            actor="tester",
            mode="share",
            operation="export",
            target_path="sanitized-briefs/顧客名入り.md",
            output_path="exports/顧客名_五万円.md",
            policy_result="allow",
            approval_status="approved",
            execution_result="success",
        )
        log_text = (self.tmp / "audit-log" / "audit.jsonl").read_text(encoding="utf-8")
        data = json.loads(log_text.strip())
        self.assertEqual(data["target_path"], "sanitized-briefs/<redacted>")
        self.assertEqual(data["output_path"], "exports/<redacted>")
        self.assertNotIn("顧客名", log_text)
        self.assertNotIn("五万円", log_text)

    def test_read_logs(self):
        self.logger.log(
            actor="tester",
            mode="share",
            operation="export",
            target_path="sanitized-briefs/a.md",
            output_path="exports/a.md",
            policy_result="allow",
            approval_status="approved",
            execution_result="success",
        )
        events = self.logger.read()
        self.assertEqual(len(events), 1)

    def test_change_id_lookup_atomically_discards_only_torn_final_line(self):
        self.logger.log(
            actor="synthetic-user",
            mode="customer",
            operation="proposal-edit",
            target_path="customer-vault/redacted",
            policy_result="allow",
            approval_status="confirmed",
            execution_result="success",
            change_id="00000000-0000-4000-8000-000000000001",
        )
        with self.logger.log_path.open("ab") as handle:
            handle.write(b'{"change_id":"torn"')
        self.assertTrue(
            self.logger.has_change_id("00000000-0000-4000-8000-000000000001")
        )
        self.assertTrue(self.logger.log_path.read_bytes().endswith(b"\n"))
        self.assertFalse(self.logger.has_change_id("missing"))

    def test_change_id_lookup_rejects_malformed_nonfinal_line(self):
        self.logger.log_path.write_text('{broken}\n{"change_id":"ok"}\n', encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            self.logger.has_change_id("ok")

    def test_change_id_lookup_restores_missing_final_newline(self):
        self.logger.log_path.write_text('{"change_id":"complete"}', encoding="utf-8")

        self.assertTrue(self.logger.has_change_id("complete"))
        self.assertEqual(
            self.logger.log_path.read_text(encoding="utf-8"),
            '{"change_id":"complete"}\n',
        )
