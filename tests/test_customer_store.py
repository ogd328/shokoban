"""Strict customer record and retention tests."""

import json
import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

from shokoban.customer_store import (
    AliasConflictError,
    CLEAN_ROOT_CONFIRMATION,
    CustomerStore,
    CustomerStoreError,
    RecordValidationError,
    RevisionConflictError,
    parse_record,
    serialize_record,
)


class CustomerStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="shokoban_phase5a_"))
        self.store = CustomerStore(self.tmp)
        self.store.initialize(confirmation=CLEAN_ROOT_CONFIRMATION)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def create_customer(self, name="完全架空顧客A", aliases=None):
        return self.store.create_customer(
            display_name=name,
            aliases=aliases or ["架空A"],
            occupation="架空担当",
            industry="架空業",
        )

    def test_schema_v2_initialization_accepts_only_absent_or_confirmed_empty_root(self):
        self.assertEqual(
            (self.store.vault / "schema.yaml").read_text(encoding="utf-8"),
            "schema_version: 2\n",
        )

        absent = self.tmp.parent / f"{self.tmp.name}-absent"
        absent_store = CustomerStore(absent)
        absent_store.initialize()
        self.assertEqual(
            (absent_store.vault / "schema.yaml").read_text(encoding="utf-8"),
            "schema_version: 2\n",
        )
        shutil.rmtree(absent, ignore_errors=True)

        empty = self.tmp.parent / f"{self.tmp.name}-empty"
        empty.mkdir(mode=0o700)
        try:
            with self.assertRaises(CustomerStoreError):
                CustomerStore(empty).initialize()
            self.assertEqual(list(empty.iterdir()), [])
        finally:
            shutil.rmtree(empty, ignore_errors=True)

    def test_schema_v2_initialization_rejects_existing_roots_without_mutation(self):
        manifests = (
            "schema_version: 1\n",
            "schema_version: 2\n",
            "schema_version: 99\n",
            "broken: true\n",
        )
        for index, manifest in enumerate(manifests):
            root = self.tmp.parent / f"{self.tmp.name}-rejected-{index}"
            root.mkdir(mode=0o700)
            vault = root / "customer-vault"
            vault.mkdir(mode=0o700)
            schema = vault / "schema.yaml"
            schema.write_text(manifest, encoding="utf-8")
            schema.chmod(0o600)
            before = {
                str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            try:
                with self.assertRaises(RecordValidationError):
                    CustomerStore(root).initialize(confirmation=CLEAN_ROOT_CONFIRMATION)
                after = {
                    str(path.relative_to(root)): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file()
                }
                self.assertEqual(after, before)
            finally:
                shutil.rmtree(root, ignore_errors=True)

        nonempty = self.tmp.parent / f"{self.tmp.name}-uninitialized"
        nonempty.mkdir(mode=0o700)
        marker = nonempty / "synthetic-marker"
        marker.write_bytes(b"unchanged")
        marker.chmod(0o600)
        try:
            with self.assertRaises(RecordValidationError):
                CustomerStore(nonempty).initialize(confirmation=CLEAN_ROOT_CONFIRMATION)
            self.assertEqual(marker.read_bytes(), b"unchanged")
            self.assertEqual([path.name for path in nonempty.iterdir()], ["synthetic-marker"])
        finally:
            shutil.rmtree(nonempty, ignore_errors=True)

    def test_schema_v2_initialization_rechecks_empty_root_under_lock(self):
        root = self.tmp.parent / f"{self.tmp.name}-changed"
        root.mkdir(mode=0o700)
        marker = root / "concurrent-marker"

        @contextmanager
        def mutate_before_locked_recheck():
            marker.write_bytes(b"concurrent")
            marker.chmod(0o600)
            yield

        store = CustomerStore(root)
        try:
            with mock.patch.object(
                store, "_initialization_lock", side_effect=mutate_before_locked_recheck
            ):
                with self.assertRaises(RecordValidationError):
                    store.initialize(confirmation=CLEAN_ROOT_CONFIRMATION)
            self.assertEqual(marker.read_bytes(), b"concurrent")
            self.assertFalse((root / "customer-vault").exists())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_consultation_schema_accepts_only_exact_empty_body_record(self):
        timestamp = "2026-07-20T08:00:00+00:00"
        metadata = {
            "schema_version": 2,
            "record_type": "consultation",
            "record_id": "con_" + "1" * 32,
            "customer_id": "cus_" + "2" * 32,
            "revision": 1,
            "created_at": timestamp,
            "updated_at": timestamp,
            "record_digest": "0" * 64,
            "provider_kind": "chatgpt_manual",
            "response_received_at": "2026-07-20T07:59:00Z",
            "inbound_artifact_ref": "art_" + "3" * 32,
            "inbound_response_digest": "4" * 64,
            "outbound_brief_ref": None,
            "outbound_brief_digest": None,
            "supersedes": None,
            "ingest_change_id": "12345678-1234-4234-9234-123456789abc",
            "recorded_by_ref": "5" * 32,
            "recorded_at": timestamp,
        }

        for provider in (
            "chatgpt_manual",
            "claude_manual",
            "gemini_manual",
            "copilot_manual",
            "other_external_ai_manual",
        ):
            with self.subTest(provider=provider):
                candidate = {**metadata, "provider_kind": provider}
                parsed, body = parse_record(serialize_record(candidate))
                self.assertEqual(parsed["record_type"], "consultation")
                self.assertEqual(parsed["provider_kind"], provider)
                self.assertEqual(body, "")

        invalid_fields = {
            "provider_kind": "unknown_manual",
            "response_received_at": "2026-07-20T16:59:00+09:00",
            "inbound_artifact_ref": "art_invalid",
            "inbound_response_digest": "A" * 64,
            "outbound_brief_ref": "brf_" + "6" * 32,
            "outbound_brief_digest": "7" * 64,
            "supersedes": "inf_" + "8" * 32,
            "ingest_change_id": "12345678-1234-1234-9234-123456789abc",
            "recorded_by_ref": "g" * 32,
            "recorded_at": "2026-07-20T17:00:00+09:00",
            "revision": 2,
        }
        for field, value in invalid_fields.items():
            with self.subTest(field=field, value=value):
                with self.assertRaises(RecordValidationError):
                    serialize_record({**metadata, field: value})

        with self.assertRaises(RecordValidationError):
            serialize_record(metadata, "hostile response body")
        with self.assertRaises(RecordValidationError):
            serialize_record({**metadata, "unexpected": "value"})

    def test_strict_markdown_treats_frontmatter_like_content_as_data(self):
        customer = self.create_customer(name="---\nstatus: archive", aliases=["No"])
        path = self.store.customers_dir / customer["customer_id"] / "customer.md"
        parsed, body = parse_record(path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["display_name"], "---\nstatus: archive")
        self.assertEqual(parsed["aliases"], ["No"])
        self.assertEqual(parsed["status"], "active")
        self.assertEqual(body, "")

        text = path.read_text(encoding="utf-8").replace(
            "record_type: customer", "unknown: injected\nrecord_type: customer"
        )
        with self.assertRaises(RecordValidationError):
            parse_record(text)

    def test_customer_crud_alias_resolution_and_revision_fail_closed(self):
        customer = self.create_customer()
        customer_id = customer["customer_id"]
        self.assertEqual(self.store.resolve_customer("  架空Ａ "), customer_id)
        with self.assertRaises(AliasConflictError):
            self.create_customer(name="別の架空顧客", aliases=["架空A"])

        updated = self.store.update_customer(
            customer_id,
            expected_revision=1,
            display_name="完全架空顧客A",
            aliases=["架空A", "架空別名"],
            occupation="架空責任者",
            industry="架空業",
            status="active",
        )
        self.assertEqual(updated["revision"], 2)
        with self.assertRaises(RevisionConflictError):
            self.store.update_customer(
                customer_id,
                expected_revision=1,
                display_name="完全架空顧客A",
                aliases=[],
                occupation="",
                industry="",
                status="active",
            )

    def test_information_is_append_only_and_supersession_is_derived(self):
        customer = self.create_customer()
        observed = datetime.now(timezone.utc).isoformat()
        first = self.store.add_information(
            customer["customer_id"],
            text="完全架空の初期情報",
            kind="note",
            observed_at=observed,
        )
        second = self.store.add_information(
            customer["customer_id"],
            text="完全架空の更新情報",
            kind="note",
            observed_at=observed,
            supersedes=first["record_id"],
        )
        current = self.store.current_information(customer["customer_id"])
        self.assertEqual([item[0]["record_id"] for item in current], [second["record_id"]])
        old_path = (
            self.store.customers_dir
            / customer["customer_id"]
            / "information"
            / f"{first['record_id']}.md"
        )
        self.assertTrue(old_path.is_file())

    def test_menu_status_is_derived_from_immutable_events(self):
        customer = self.create_customer()
        menu = self.store.create_menu(
            customer["customer_id"],
            name="完全架空の継続支援",
            menu_kind="recurring",
            description="完全架空の説明",
            initial_status="active",
        )
        self.assertEqual(self.store.menu_status(customer["customer_id"], menu["record_id"]), "active")
        self.store.add_menu_event(
            customer["customer_id"], menu["record_id"], to_status="paused"
        )
        self.assertEqual(self.store.menu_status(customer["customer_id"], menu["record_id"]), "paused")
        listed = self.store.list_menus(customer["customer_id"])
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["description"], "完全架空の説明")
        self.assertEqual(listed[0]["menu_status"], "paused")
        with self.assertRaises(RecordValidationError):
            self.store.add_menu_event(
                customer["customer_id"], menu["record_id"], to_status="proposed"
            )

    def test_same_time_menu_events_follow_persisted_sequence_not_random_id(self):
        customer = self.create_customer()
        timestamp = "2026-07-20T10:00:00+00:00"
        ids = [
            "mnu_" + "1" * 32,
            "mev_" + "f" * 32,
            "mev_" + "0" * 32,
        ]

        with mock.patch("shokoban.customer_store._now", return_value=timestamp), mock.patch(
            "shokoban.customer_store.new_record_id", side_effect=ids
        ):
            menu = self.store.create_menu(
                customer["customer_id"],
                name="完全架空の決定性確認",
                menu_kind="recurring",
                description="完全架空の説明",
                initial_status="active",
            )
            self.store.add_menu_event(
                customer["customer_id"],
                menu["record_id"],
                to_status="paused",
                effective_at=timestamp,
            )

        events = self.store._menu_events(customer["customer_id"], menu["record_id"])
        self.assertEqual([event["event_sequence"] for event in events], [1, 2])
        self.assertEqual(self.store.menu_status(customer["customer_id"], menu["record_id"]), "paused")

    def test_list_menus_rejects_menu_without_state_event(self):
        customer = self.create_customer()
        menu = self.store.create_menu(
            customer["customer_id"],
            name="完全架空の単発支援",
            menu_kind="spot",
            description="完全架空の説明",
        )
        events_dir = self.store.customers_dir / customer["customer_id"] / "menu-events"
        for path in events_dir.glob("mev_*.md"):
            path.unlink()

        with self.assertRaises(RecordValidationError):
            self.store.list_menus(customer["customer_id"])

    def test_confirmed_meeting_can_be_reopened_with_revision_check(self):
        customer = self.create_customer()
        meeting = self.store.create_meeting_draft(
            customer["customer_id"],
            occurred_at=datetime.now(timezone.utc).isoformat(),
            title="完全架空打ち合わせ",
            summary="",
            details="完全架空詳細",
        )
        cid, mid = customer["customer_id"], meeting["record_id"]
        with self.assertRaises(RecordValidationError):  # 下書きは戻せない
            self.store.reopen_meeting(cid, mid, expected_revision=1)
        confirmed = self.store.confirm_meeting(cid, mid, expected_revision=1, confirmed_by="synthetic-human")
        with self.assertRaises(RevisionConflictError):
            self.store.reopen_meeting(cid, mid, expected_revision=1)
        reopened = self.store.reopen_meeting(cid, mid, expected_revision=confirmed["revision"])
        self.assertEqual(reopened["meeting_status"], "draft")
        self.assertIsNone(reopened["confirmed_by"])
        self.assertIsNone(reopened["confirmed_at"])
        self.assertEqual(reopened["details"], "完全架空詳細")
        self.assertEqual(reopened["revision"], confirmed["revision"] + 1)
        updated = self.store.update_meeting_draft(
            cid, mid, expected_revision=reopened["revision"], title="完全架空打ち合わせ", summary="",
            details="直した詳細", decisions=[], next_actions=[],
        )
        again = self.store.confirm_meeting(cid, mid, expected_revision=updated["revision"], confirmed_by="synthetic-human")
        self.assertEqual(again["meeting_status"], "confirmed")
        self.assertEqual(again["details"], "直した詳細")
        audit = (self.store.root / "audit-log" / "audit.jsonl").read_text(encoding="utf-8")
        self.assertIn('"operation": "meeting-reopen"', audit)

    def test_meeting_requires_revision_and_human_confirmation(self):
        customer = self.create_customer()
        meeting = self.store.create_meeting_draft(
            customer["customer_id"],
            occurred_at=datetime.now(timezone.utc).isoformat(),
            title="完全架空打ち合わせ",
            summary="完全架空要約",
            details="完全架空詳細",
        )
        updated = self.store.update_meeting_draft(
            customer["customer_id"],
            meeting["record_id"],
            expected_revision=1,
            title="完全架空打ち合わせ改訂",
            summary="完全架空要約改訂",
            details="完全架空詳細改訂",
            decisions=["完全架空の決定"],
            next_actions=["完全架空の次作業"],
        )
        confirmed = self.store.confirm_meeting(
            customer["customer_id"],
            meeting["record_id"],
            expected_revision=updated["revision"],
            confirmed_by="synthetic-human",
        )
        self.assertEqual(confirmed["meeting_status"], "confirmed")
        with self.assertRaises(RecordValidationError):
            self.store.update_meeting_draft(
                customer["customer_id"],
                meeting["record_id"],
                expected_revision=confirmed["revision"],
                title="変更不可",
                summary="",
                details="",
                decisions=[],
                next_actions=[],
            )

    def test_proposal_payload_is_staged_and_never_updates_canonical_record(self):
        customer = self.create_customer()
        original = self.store.get_customer(customer["customer_id"])
        proposal = self.store.stage_proposal(
            customer["customer_id"],
            payload={
                "schema_version": 1,
                "target_type": "information",
                "operation": "create",
                "content": {"text": "完全架空のAI案"},
            },
            model="gemma4:12b",
            model_identifier="local-test-model",
            prompt_contract_version="phase5a-test-v1",
            prompt_contract_hash="a" * 64,
            source_refs=[],
            input_text="完全架空入力",
        )
        payload_path = self.store.staging_dir / "proposals" / f"{proposal['record_id']}.json"
        self.assertTrue(payload_path.is_file())
        reviewed_at = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        self.store.reject_proposal(
            customer["customer_id"],
            proposal["record_id"],
            expected_revision=1,
            expected_customer_revision=customer["revision"],
            confirmation=True,
            reviewed_at=reviewed_at,
        )
        deleted = self.store.reap_expired_proposal_payloads(now=datetime.now(timezone.utc))
        self.assertEqual(deleted, [proposal["record_id"]])
        self.assertFalse(payload_path.exists())
        self.assertEqual(self.store.get_customer(customer["customer_id"]), original)

    def test_pending_proposal_payload_edit_is_revision_checked_and_noncanonical(self):
        customer = self.create_customer()
        original = self.store.get_customer(customer["customer_id"])
        proposal = self.store.stage_proposal(
            customer["customer_id"],
            payload={
                "schema_version": 1,
                "target_type": "information",
                "operation": "create",
                "content": {"text": "完全架空のAI案"},
            },
            model="gemma4:12b",
            model_identifier="local-test-model",
            prompt_contract_version="phase5d-test-v1",
            prompt_contract_hash="a" * 64,
            source_refs=[],
            input_text="完全架空入力",
        )

        updated = self.store.update_pending_proposal_payload(
            customer["customer_id"],
            proposal["record_id"],
            expected_revision=proposal["revision"],
            payload={
                "schema_version": 1,
                "target_type": "information",
                "operation": "create",
                "content": {"text": "完全架空の本人編集案"},
            },
        )

        metadata, payload = self.store.get_proposal_payload(
            customer["customer_id"], proposal["record_id"]
        )
        self.assertEqual(metadata, updated)
        self.assertEqual(updated["revision"], 2)
        self.assertTrue(updated["human_edited"])
        self.assertEqual(payload["content"]["text"], "完全架空の本人編集案")
        self.assertEqual(self.store.get_customer(customer["customer_id"]), original)

        with self.assertRaises(RevisionConflictError):
            self.store.update_pending_proposal_payload(
                customer["customer_id"],
                proposal["record_id"],
                expected_revision=1,
                payload=payload,
            )

    def test_pending_proposal_edit_cannot_change_operation(self):
        customer = self.create_customer()
        proposal = self.store.stage_proposal(
            customer["customer_id"],
            payload={
                "schema_version": 1,
                "target_type": "information",
                "operation": "create",
                "content": {"text": "完全架空のAI案"},
            },
            model="gemma4:12b",
            model_identifier="local-test-model",
            prompt_contract_version="phase5d-test-v1",
            prompt_contract_hash="a" * 64,
            source_refs=[],
            input_text="完全架空入力",
        )

        with self.assertRaises(RecordValidationError):
            self.store.update_pending_proposal_payload(
                customer["customer_id"],
                proposal["record_id"],
                expected_revision=1,
                payload={
                    "schema_version": 1,
                    "target_type": "menu",
                    "operation": "create",
                    "content": {"text": "完全架空の変更"},
                },
            )

        metadata, payload = self.store.get_proposal_payload(
            customer["customer_id"], proposal["record_id"]
        )
        self.assertEqual(metadata["revision"], 1)
        self.assertFalse(metadata["human_edited"])
        self.assertEqual(payload["target_type"], "information")

    def test_pending_proposal_edit_recovers_metadata_before_payload_replace(self):
        customer = self.create_customer()
        proposal = self.store.stage_proposal(
            customer["customer_id"],
            payload={
                "schema_version": 1,
                "target_type": "information",
                "operation": "create",
                "content": {"text": "完全架空の編集前"},
            },
            model="gemma4:12b",
            model_identifier="local-test-model",
            prompt_contract_version="phase5d-test-v1",
            prompt_contract_hash="a" * 64,
            source_refs=[],
            input_text="完全架空入力",
        )
        payload_path = self.store.staging_dir / "proposals" / f"{proposal['record_id']}.json"
        original_replace = os.replace

        def fail_final_payload_replace(source, destination):
            if Path(destination) == payload_path and Path(source).name.startswith(".prp_"):
                raise OSError("synthetic injected crash")
            return original_replace(source, destination)

        with mock.patch("shokoban.customer_store.os.replace", side_effect=fail_final_payload_replace):
            with self.assertRaises(OSError):
                self.store.update_pending_proposal_payload(
                    customer["customer_id"], proposal["record_id"],
                    expected_revision=proposal["revision"],
                    payload={
                        "schema_version": 1,
                        "target_type": "information",
                        "operation": "create",
                        "content": {"text": "完全架空の編集後"},
                    },
                )

        recovered_store = CustomerStore(self.tmp)
        recovered_store.create_customer(display_name="架空顧客編集復旧後")
        metadata, payload = recovered_store.get_proposal_payload(
            customer["customer_id"], proposal["record_id"]
        )
        self.assertEqual(metadata["revision"], proposal["revision"] + 1)
        self.assertEqual(payload["content"]["text"], "完全架空の編集後")
        self.assertEqual(list(recovered_store.transactions_dir.glob("*.json")), [])

    def test_proposal_stage_recovers_metadata_before_payload_replace(self):
        customer = self.create_customer()
        proposals_dir = self.store.staging_dir / "proposals"
        original_replace = os.replace

        def fail_final_payload_replace(source, destination):
            destination = Path(destination)
            if destination.parent == proposals_dir and destination.name.startswith("prp_"):
                raise OSError("synthetic injected crash")
            return original_replace(source, destination)

        with mock.patch("shokoban.customer_store.os.replace", side_effect=fail_final_payload_replace):
            with self.assertRaises(OSError):
                self.store.stage_proposal(
                    customer["customer_id"],
                    payload={
                        "schema_version": 1,
                        "target_type": "information",
                        "operation": "create",
                        "content": {"text": "完全架空のstage復旧案"},
                    },
                    model="gemma4:12b",
                    model_identifier="local-test-model",
                    prompt_contract_version="phase5d-test-v1",
                    prompt_contract_hash="a" * 64,
                    source_refs=[],
                    input_text="完全架空入力",
                )

        proposal_path = next(
            (self.store.customers_dir / customer["customer_id"] / "proposals").glob("prp_*.md")
        )
        recovered_store = CustomerStore(self.tmp)
        recovered_store.create_customer(display_name="架空顧客stage復旧後")
        metadata, payload = recovered_store.get_proposal_payload(
            customer["customer_id"], proposal_path.stem
        )
        self.assertEqual(metadata["proposal_status"], "pending")
        self.assertEqual(payload["content"]["text"], "完全架空のstage復旧案")
        self.assertEqual(list(recovered_store.transactions_dir.glob("*.json")), [])

    def test_ai_assisted_information_requires_atomic_proposal_application(self):
        customer = self.create_customer()
        proposal = self.store.stage_proposal(
            customer["customer_id"],
            payload={
                "schema_version": 1,
                "target_type": "information",
                "operation": "create",
                "content": {"text": "完全架空のAI追記案"},
            },
            model="gemma4:12b",
            model_identifier="local-test-model",
            prompt_contract_version="phase5a-test-v1",
            prompt_contract_hash="a" * 64,
            source_refs=[],
            input_text="完全架空入力",
        )
        with self.assertRaises(RecordValidationError):
            self.store.add_information(
                customer["customer_id"],
                text="完全架空のAI追記案",
                kind="note",
                observed_at=datetime.now(timezone.utc).isoformat(),
                source_refs=[proposal["record_id"]],
                origin="ai_assisted",
            )
        accepted, saved = self.store.accept_and_apply_information_proposal(
            customer["customer_id"],
            proposal["record_id"],
            expected_revision=1,
            expected_customer_revision=customer["revision"],
            confirmation=True,
        )
        self.assertEqual(accepted["applied_record_id"], saved["record_id"])
        self.assertEqual(saved["origin"], "ai_assisted")
        self.assertEqual(saved["source_refs"], [proposal["record_id"]])
        self.assertEqual(self.store.current_information(customer["customer_id"])[0][1], "完全架空のAI追記案")

        with self.assertRaises(RecordValidationError):
            self.store.add_information(
                customer["customer_id"],
                text="完全架空の差し替え本文",
                kind="note",
                observed_at=datetime.now(timezone.utc).isoformat(),
                source_refs=[proposal["record_id"]],
                origin="ai_assisted",
            )
        with self.assertRaises(RecordValidationError):
            self.store.accept_and_apply_information_proposal(
                customer["customer_id"],
                proposal["record_id"],
                expected_revision=accepted["revision"],
                expected_customer_revision=customer["revision"],
                confirmation=True,
            )
        self.assertEqual(len(self.store.current_information(customer["customer_id"])), 1)

    def test_proposal_terminal_actions_require_literal_confirmation_and_fresh_customer(self):
        customer = self.create_customer()
        proposal = self.store.stage_proposal(
            customer["customer_id"],
            payload={
                "schema_version": 1,
                "target_type": "information",
                "operation": "create",
                "content": {"text": "完全架空の確認案"},
            },
            model="gemma4:12b",
            model_identifier="local-test-model",
            prompt_contract_version="phase5d-test-v1",
            prompt_contract_hash="a" * 64,
            source_refs=[],
            input_text="完全架空入力",
        )
        for value in (False, 1, "yes", None):
            with self.assertRaises(RecordValidationError):
                self.store.reject_proposal(
                    customer["customer_id"], proposal["record_id"],
                    expected_revision=proposal["revision"],
                    expected_customer_revision=customer["revision"],
                    confirmation=value,
                )
        with self.assertRaises(RevisionConflictError):
            self.store.reject_proposal(
                customer["customer_id"], proposal["record_id"],
                expected_revision=proposal["revision"],
                expected_customer_revision=customer["revision"] + 1,
                confirmation=True,
            )
        rejected = self.store.reject_proposal(
            customer["customer_id"], proposal["record_id"],
            expected_revision=proposal["revision"],
            expected_customer_revision=customer["revision"],
            confirmation=True,
        )
        self.assertEqual(rejected["proposal_status"], "rejected")
        self.assertIsNone(rejected["applied_record_id"])
        self.assertEqual(self.store.current_information(customer["customer_id"]), [])

    def test_accept_recovers_info_written_before_proposal_transition(self):
        customer = self.create_customer()
        proposal = self.store.stage_proposal(
            customer["customer_id"],
            payload={
                "schema_version": 1,
                "target_type": "information",
                "operation": "create",
                "content": {"text": "完全架空の復旧案"},
            },
            model="gemma4:12b",
            model_identifier="local-test-model",
            prompt_contract_version="phase5d-test-v1",
            prompt_contract_hash="a" * 64,
            source_refs=[],
            input_text="完全架空入力",
        )
        original_write = self.store._write

        def fail_proposal_transition(path, metadata, body="", *, overwrite=False):
            if metadata.get("record_type") == "proposal" and metadata.get("proposal_status") == "accepted":
                raise OSError("synthetic injected crash")
            return original_write(path, metadata, body, overwrite=overwrite)

        with mock.patch.object(self.store, "_write", side_effect=fail_proposal_transition):
            with self.assertRaises(OSError):
                self.store.accept_and_apply_information_proposal(
                    customer["customer_id"], proposal["record_id"],
                    expected_revision=proposal["revision"],
                    expected_customer_revision=customer["revision"],
                    confirmation=True,
                )

        self.assertEqual(len(list(self.store.transactions_dir.glob("*.json"))), 1)
        recovered_store = CustomerStore(self.tmp)
        recovered_store.create_customer(display_name="架空顧客復旧後")
        recovered, _ = recovered_store.get_proposal_payload(
            customer["customer_id"], proposal["record_id"]
        )
        self.assertEqual(recovered["proposal_status"], "accepted")
        self.assertEqual(len(recovered_store.current_information(customer["customer_id"])), 1)
        self.assertEqual(list(recovered_store.transactions_dir.glob("*.json")), [])

    def test_source_refs_reject_paths(self):
        customer = self.create_customer()
        with self.assertRaises(RecordValidationError):
            self.store.add_information(
                customer["customer_id"],
                text="完全架空情報",
                kind="note",
                observed_at=datetime.now(timezone.utc).isoformat(),
                source_refs=["customer-vault/実名.md"],
            )

    def test_customer_write_rejects_symlinked_storage_component(self):
        shutil.rmtree(self.store.customers_dir)
        outside = self.tmp / "outside"
        outside.mkdir()
        self.store.customers_dir.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(CustomerStoreError):
            self.create_customer()
        self.assertEqual(list(outside.iterdir()), [])

    @unittest.skipIf(os.name == "nt", "POSIX permission contract")
    def test_customer_write_rejects_broad_data_directory_permissions(self):
        os.chmod(self.store.vault, 0o755)
        with self.assertRaises(CustomerStoreError):
            self.create_customer()

    def test_artifact_retention_waits_for_confirmation(self):
        customer = self.create_customer()
        old = datetime.now(timezone.utc) - timedelta(days=31)
        transcript_id = self.store.stage_artifact(
            customer["customer_id"],
            kind="transcript",
            data="完全架空文字起こし".encode(),
            created_at=old.isoformat(),
        )
        self.assertEqual(self.store.reap_expired_artifacts(), [])
        self.assertTrue(self.store.confirm_artifact(transcript_id))

        audio_id = self.store.stage_artifact(
            customer["customer_id"], kind="audio", data=b"synthetic-audio"
        )
        self.assertTrue(self.store.confirm_artifact(audio_id))
        self.assertFalse((self.store.staging_dir / "artifacts" / audio_id).exists())

    def test_customer_audit_contains_no_customer_content(self):
        customer = self.create_customer(name="監査に出してはいけない架空名", aliases=["秘密別名"])
        self.store.add_information(
            customer["customer_id"],
            text="監査に出してはいけない架空本文",
            kind="note",
            observed_at=datetime.now(timezone.utc).isoformat(),
        )
        text = (self.tmp / "audit-log" / "audit.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("監査に出してはいけない架空名", text)
        self.assertNotIn("秘密別名", text)
        self.assertNotIn("監査に出してはいけない架空本文", text)
        self.assertNotIn(customer["customer_id"], text)
        for line in text.splitlines():
            event = json.loads(line)
            self.assertEqual(event["target_path"], "customer-vault/<redacted>")
            self.assertRegex(event["entity_ref"], r"^[0-9a-f]{32}$")


if __name__ == "__main__":
    unittest.main()
