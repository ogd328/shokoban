"""監査ログ用のパス匿名化（``sanitize_audit_path``）の試験."""

import unittest

from shokoban import utils


class UtilsTest(unittest.TestCase):
    def test_sanitize_audit_path_customer_vault_deep(self):
        self.assertEqual(
            utils.sanitize_audit_path("customer-vault/acme/sub/顧客名入りファイル.md"),
            "customer-vault/<redacted>",
        )

    def test_sanitize_audit_path_customer_vault_top(self):
        self.assertEqual(
            utils.sanitize_audit_path("customer-vault/acme/memo.md"),
            "customer-vault/<redacted>",
        )

    def test_sanitize_audit_path_customer_vault_dir_only(self):
        self.assertEqual(
            utils.sanitize_audit_path("customer-vault"),
            "customer-vault/<redacted>",
        )

    def test_sanitize_audit_path_non_customer(self):
        self.assertEqual(
            utils.sanitize_audit_path("sanitized-briefs/brief.md"),
            "sanitized-briefs/<redacted>",
        )

    def test_sanitize_audit_path_memory(self):
        self.assertEqual(
            utils.sanitize_audit_path("memory/memory_abc123.md"),
            "memory/<redacted>",
        )

    def test_sanitize_audit_path_exports(self):
        self.assertEqual(
            utils.sanitize_audit_path("exports/顧客名_五万円.md"),
            "exports/<redacted>",
        )


