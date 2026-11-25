"""受け取りフォルダ取り込みの試験（すべて tmp_path 上で動く）."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from shokoban import inbox_import
from shokoban.customer_store import CLEAN_ROOT_CONFIRMATION, CustomerStore
from shokoban.inbox_import import (
    ALREADY_IMPORTED_REASON,
    ARCHIVE_DIR_NAME,
    MAX_FILE_BYTES,
    STATUS_FAILED,
    STATUS_IMPORTED,
    STATUS_SKIPPED,
    InboxImportError,
    import_inbox,
)

JST = timezone(timedelta(hours=9))
NOW = datetime(2026, 11, 9, 12, 0, tzinfo=JST)


def nfc(value: str) -> str:
    """macOS のファイル名正規化差を吸収して比べる."""
    return unicodedata.normalize("NFC", value)


@pytest.fixture()
def store(tmp_path: Path) -> CustomerStore:
    target = CustomerStore(tmp_path / "store")
    target.initialize(confirmation=CLEAN_ROOT_CONFIRMATION)
    return target


@pytest.fixture()
def inbox(tmp_path: Path) -> Path:
    path = tmp_path / "inbox"
    path.mkdir(mode=0o700)
    return path


def write_note(inbox: Path, name: str, text: str) -> Path:
    path = inbox / name
    path.write_text(text, encoding="utf-8")
    return path


def test_imports_only_regular_notes_with_matching_names(store: CustomerStore, inbox: Path, tmp_path: Path):
    note = write_note(inbox, "2026-11-09 架空商事.txt", "打ち合わせメモ\n次回は11月20日。\n")
    markdown = write_note(inbox, "2026-11-08 架空工業.md", "# 見積の相談\n")
    part = write_note(inbox, "2026-11-09 架空商事.txt.part", "書きかけ")
    hidden = write_note(inbox, ".2026-11-09 架空商事.txt", "隠し")
    write_note(inbox, "2026-11-09メモ.txt", "空白なし")
    write_note(inbox, "メモ.txt", "日付なし")
    outside = tmp_path / "linked.txt"
    outside.write_text("リンク先", encoding="utf-8")
    (inbox / "2026-11-09 架空リンク.txt").symlink_to(outside)
    oversized = inbox / "2026-11-10 巨大.txt"
    with oversized.open("wb") as handle:
        handle.truncate(MAX_FILE_BYTES + 1)

    results = import_inbox(store, inbox, now=NOW)

    imported = [result for result in results if result.status == STATUS_IMPORTED]
    assert [(nfc(result.source_name), result.status) for result in imported] == [
        ("2026-11-08 架空工業.md", STATUS_IMPORTED),
        ("2026-11-09 架空商事.txt", STATUS_IMPORTED),
    ]
    skipped = {nfc(result.source_name): result.reason for result in results if result.status == STATUS_SKIPPED}
    assert set(skipped) == {
        ".2026-11-09 架空商事.txt",
        "2026-11-09 架空商事.txt.part",
        "2026-11-09メモ.txt",
        "2026-11-09 架空リンク.txt",
        "2026-11-10 巨大.txt",
        "メモ.txt",
    }
    assert all(reason for reason in skipped.values())
    assert "シンボリックリンク" in skipped["2026-11-09 架空リンク.txt"]
    assert "5MB" in skipped["2026-11-10 巨大.txt"]
    assert ".part" in skipped["2026-11-09 架空商事.txt.part"]
    assert "「.」で始まる" in skipped[".2026-11-09 架空商事.txt"]

    assert not note.exists()
    assert not markdown.exists()
    assert part.is_file()
    assert hidden.is_file()
    assert oversized.is_file()
    assert outside.is_file()


def test_creates_new_customer_and_reuses_existing_one(store: CustomerStore, inbox: Path):
    existing = store.create_customer(display_name="架空商事株式会社", aliases=["架空商事"])
    write_note(inbox, "2026-11-09 架空商事.txt", "既存の別名で届いたメモ\n")
    write_note(inbox, "2026-11-09 架空新規工業.txt", "新しい顧客のメモ\n")

    results = import_inbox(store, inbox, now=NOW)

    by_name = {nfc(result.source_name): result for result in results}
    assert by_name["2026-11-09 架空商事.txt"].customer_id == existing["customer_id"]
    new_id = by_name["2026-11-09 架空新規工業.txt"].customer_id
    assert new_id is not None and new_id != existing["customer_id"]
    display_names = {customer["customer_id"]: nfc(customer["display_name"]) for customer in store.list_customers()}
    assert display_names[new_id] == "架空新規工業"
    assert len(store.list_customers()) == 2


def test_two_notes_for_same_new_customer_share_one_customer(store: CustomerStore, inbox: Path):
    write_note(inbox, "2026-11-08 架空商事.txt", "1件目\n")
    write_note(inbox, "2026-11-09 架空商事.txt", "2件目\n")

    results = import_inbox(store, inbox, now=NOW)

    assert {result.customer_id for result in results} == {store.list_customers()[0]["customer_id"]}
    assert len(store.list_customers()) == 1
    assert len(store.list_meetings(results[0].customer_id)) == 2


def test_imports_as_unconfirmed_draft_with_filename_date(store: CustomerStore, inbox: Path):
    body = "1行目: 初回訪問\n2行目: 見積は来週\n" + "長" * 200
    write_note(inbox, "2026-11-07 架空商事.txt", body)

    [result] = import_inbox(store, inbox, now=NOW)

    assert result.status == STATUS_IMPORTED
    assert result.meeting_id is not None
    [meeting] = store.list_meetings(result.customer_id)
    assert meeting["record_id"] == result.meeting_id
    assert meeting["meeting_status"] == "draft"
    assert meeting["confirmed_by"] is None
    assert meeting["confirmed_at"] is None
    assert nfc(meeting["title"]) == "2026-11-07 取り込み"
    assert meeting["occurred_at"] == "2026-11-07T00:00:00+09:00"
    assert meeting["details"] == body
    assert meeting["summary"] == body.strip()[:120]
    assert len(meeting["summary"]) == 120
    assert result.checked_at == "2026-11-09T12:00:00+09:00"


def test_markdown_note_is_imported_too(store: CustomerStore, inbox: Path):
    write_note(inbox, "2026-11-09 架空商事.md", "# 議事録\n\n- 決定: 次回訪問\n")

    [result] = import_inbox(store, inbox, now=NOW)

    assert result.status == STATUS_IMPORTED
    [meeting] = store.list_meetings(result.customer_id)
    assert meeting["details"] == "# 議事録\n\n- 決定: 次回訪問\n"


def test_moves_note_to_archive_without_overwriting_and_is_idempotent(store: CustomerStore, inbox: Path):
    note = write_note(inbox, "2026-11-09 架空商事.txt", "1回目\n")
    archive = inbox / ARCHIVE_DIR_NAME
    archive.mkdir(mode=0o700)
    previous = archive / "2026-11-09 架空商事.txt"
    previous.write_text("以前の取り込み", encoding="utf-8")

    [first] = import_inbox(store, inbox, now=NOW)

    assert first.status == STATUS_IMPORTED
    assert not note.exists()
    assert first.archived_to == archive / "2026-11-09 架空商事 (2).txt"
    assert first.archived_to.read_text(encoding="utf-8") == "1回目\n"
    assert previous.read_text(encoding="utf-8") == "以前の取り込み"
    assert oct(first.archived_to.stat().st_mode & 0o777) == "0o600"
    assert oct(archive.stat().st_mode & 0o777) == "0o700"

    assert import_inbox(store, inbox, now=NOW) == []
    assert len(store.list_meetings(first.customer_id)) == 1


def test_draft_is_created_and_reported_when_archive_move_fails(store: CustomerStore, inbox: Path):
    note = write_note(inbox, "2026-11-09 架空商事.txt", "移動できない本文\n")
    (inbox / ARCHIVE_DIR_NAME).write_text("取り込み済みをファイルで塞ぐ", encoding="utf-8")

    [result] = import_inbox(store, inbox, now=NOW)

    assert result.status == STATUS_FAILED
    assert result.meeting_id is not None
    assert "移動できませんでした" in result.reason
    assert note.is_file()
    [meeting] = store.list_meetings(result.customer_id)
    assert meeting["record_id"] == result.meeting_id
    assert meeting["meeting_status"] == "draft"


def test_invalid_date_and_directory_are_reported(store: CustomerStore, inbox: Path):
    write_note(inbox, "2026-02-30 架空商事.txt", "存在しない日付\n")
    (inbox / "作業中").mkdir()

    results = import_inbox(store, inbox, now=NOW)

    assert [result.status for result in results] == [STATUS_SKIPPED, STATUS_SKIPPED]
    assert "日付" in results[0].reason
    assert "フォルダ" in results[1].reason
    assert (inbox / "2026-02-30 架空商事.txt").is_file()
    assert store.list_customers() == []


def test_non_utf8_note_fails_and_stays_in_inbox(store: CustomerStore, inbox: Path):
    path = inbox / "2026-11-09 架空商事.txt"
    path.write_bytes(b"\xff\xfe\x00broken")

    [result] = import_inbox(store, inbox, now=NOW)

    assert result.status == STATUS_FAILED
    assert "UTF-8" in result.reason
    assert path.is_file()
    assert store.list_customers() == []
    assert not (inbox / ARCHIVE_DIR_NAME).exists()


def test_unreadable_note_fails_and_stays_in_inbox(store: CustomerStore, inbox: Path):
    if os.geteuid() == 0:
        pytest.skip("root では読み取り権限を落とせない")
    path = inbox / "2026-11-09 架空商事.txt"
    path.write_text("読めないはずの本文\n", encoding="utf-8")
    path.chmod(0o000)
    try:
        [result] = import_inbox(store, inbox, now=NOW)
    finally:
        path.chmod(0o600)

    assert result.status == STATUS_FAILED
    assert result.reason
    assert path.is_file()
    assert store.list_customers() == []


def test_inbox_origin_is_reported_when_store_rejects_it(store: CustomerStore, inbox: Path):
    write_note(inbox, "2026-11-09 架空商事.txt", "取り込み元の記録テスト\n")

    [result] = import_inbox(store, inbox, now=NOW)

    [meeting] = store.list_meetings(result.customer_id)
    # CustomerStore は source_refs に不透明ID以外を拒否するため、取り込み元は
    # 案件ファイルに残らず、結果の notes で知らせる。
    assert meeting["source_refs"] == []
    assert result.notes and "source_refs" in result.notes[0]


def test_missing_inbox_is_rejected(store: CustomerStore, tmp_path: Path):
    with pytest.raises(InboxImportError):
        import_inbox(store, tmp_path / "no-such-inbox", now=NOW)


def test_uninitialized_store_is_rejected(tmp_path: Path):
    store = CustomerStore(tmp_path / "store")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    write_note(inbox, "2026-11-09 架空商事.txt", "初期化前\n")

    with pytest.raises(InboxImportError):
        import_inbox(store, inbox, now=NOW)

    assert not (tmp_path / "store").exists()
    assert (inbox / "2026-11-09 架空商事.txt").is_file()


def test_accept_name_skips_without_moving_or_reading(store: CustomerStore, inbox: Path):
    rejected = write_note(inbox, "2026-11-09 実在っぽい名前.txt", "取り込まない本文\n")
    accepted = write_note(inbox, "2026-11-09 架空顧客Z.txt", "取り込む本文\n")
    asked: list[str] = []

    def accept(name: str) -> bool:
        asked.append(name)
        return name.startswith("架空顧客")

    results = import_inbox(store, inbox, now=NOW, accept_name=accept)

    # 名前部分（日付の後ろ・拡張子の前）だけが関数へ渡る。
    assert set(asked) == {"実在っぽい名前", "架空顧客Z"}
    by_name = {result.source_name: result for result in results}
    rejected_result = by_name["2026-11-09 実在っぽい名前.txt"]
    assert rejected_result.status == STATUS_SKIPPED
    assert rejected_result.reason == "架空データの期間は「架空顧客」で始まる名前だけ取り込みます"
    assert rejected_result.customer_id is None
    assert rejected_result.meeting_id is None
    assert rejected.is_file()
    assert not accepted.exists()
    # 見送ったファイルからは顧客も下書きも作られない。
    assert [customer["display_name"] for customer in store.list_customers()] == [
        "架空顧客Z"
    ]
    assert len(store.list_meetings(by_name["2026-11-09 架空顧客Z.txt"].customer_id)) == 1


def test_accept_name_is_not_called_for_invalid_entries(store: CustomerStore, inbox: Path):
    write_note(inbox, "メモ.txt", "日付なし\n")
    write_note(inbox, "2026-11-09 架空顧客A.txt.part", "書きかけ")
    asked: list[str] = []

    results = import_inbox(
        store, inbox, now=NOW, accept_name=lambda name: asked.append(name) or True
    )

    assert asked == []
    assert {result.status for result in results} == {STATUS_SKIPPED}
    assert store.list_customers() == []


def test_source_refs_carry_inbox_name_and_sha256() -> None:
    name = "2026-11-09 架空商事.txt"
    digest = hashlib.sha256("本文\n".encode("utf-8")).hexdigest()

    assert inbox_import._source_refs(name, digest) == [
        f"inbox:{name}",
        f"inbox-sha256:{digest}",
    ]


def test_archive_move_failure_does_not_duplicate_draft_on_retry(
    store: CustomerStore, inbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """移動に失敗して2回取り込んでも、下書きは1件のまま."""
    note = write_note(inbox, "2026-11-09 架空商事.txt", "再試行の本文\n")
    real_move = inbox_import._move_to_archive

    def fail_move(*args, **kwargs):
        raise OSError("archive unavailable")

    monkeypatch.setattr(inbox_import, "_move_to_archive", fail_move)

    first = import_inbox(store, inbox, now=NOW)
    second = import_inbox(store, inbox, now=NOW)

    assert [result.status for result in first] == [STATUS_FAILED]
    assert [result.status for result in second] == [STATUS_FAILED]
    assert note.is_file()
    meetings = store.list_meetings(first[0].customer_id)
    assert len(meetings) == 1
    assert meetings[0]["meeting_status"] == "draft"
    assert meetings[0]["record_id"] == first[0].meeting_id
    assert second[0].meeting_id == first[0].meeting_id
    assert "前回取り込み済み" in second[0].reason

    # 移動が直った3回目: 取り込み済みへ移り、下書きは1件のまま。
    monkeypatch.setattr(inbox_import, "_move_to_archive", real_move)
    [third] = import_inbox(store, inbox, now=NOW)

    assert third.status == STATUS_IMPORTED
    assert third.reason == ALREADY_IMPORTED_REASON
    assert third.meeting_id == first[0].meeting_id
    assert not note.exists()
    assert third.archived_to is not None
    assert third.archived_to.read_text(encoding="utf-8") == "再試行の本文\n"
    assert len(store.list_meetings(first[0].customer_id)) == 1


def test_same_content_under_new_name_is_not_drafted_twice(
    store: CustomerStore, inbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    note = write_note(inbox, "2026-11-09 架空商事.txt", "同じ本文\n")
    real_move = inbox_import._move_to_archive

    def fail_move(*args, **kwargs):
        raise OSError("archive unavailable")

    monkeypatch.setattr(inbox_import, "_move_to_archive", fail_move)
    [first] = import_inbox(store, inbox, now=NOW)
    assert first.status == STATUS_FAILED
    note.unlink()

    # 同じ本文が別の名前で再送された（中身の sha256 で同じと分かる）。
    renamed = write_note(inbox, "2026-11-10 架空商事.txt", "同じ本文\n")
    monkeypatch.setattr(inbox_import, "_move_to_archive", real_move)
    [second] = import_inbox(store, inbox, now=NOW)

    assert second.status == STATUS_IMPORTED
    assert second.reason == ALREADY_IMPORTED_REASON
    assert second.meeting_id == first.meeting_id
    assert not renamed.exists()
    assert len(store.list_meetings(first.customer_id)) == 1


def test_reimport_of_same_content_after_success_creates_no_draft(
    store: CustomerStore, inbox: Path
) -> None:
    write_note(inbox, "2026-11-09 架空商事.txt", "同じ本文\n")
    [first] = import_inbox(store, inbox, now=NOW)
    assert first.status == STATUS_IMPORTED

    write_note(inbox, "2026-11-12 架空商事.txt", "同じ本文\n")
    [second] = import_inbox(store, inbox, now=NOW)

    assert second.status == STATUS_IMPORTED
    assert second.reason == ALREADY_IMPORTED_REASON
    assert second.meeting_id == first.meeting_id
    assert len(store.list_meetings(first.customer_id)) == 1


def _fail_move(*args: object, **kwargs: object) -> None:
    raise OSError("archive unavailable")


def test_import_journal_records_the_identity_separately_from_the_body(
    store: CustomerStore, inbox: Path
) -> None:
    """本文とは別の記録帳に、入力の sha256 と下書きの id を残す."""
    write_note(inbox, "2026-11-09 架空商事.txt", "記録帳の本文\n")

    [result] = import_inbox(store, inbox, now=NOW)

    journal = inbox / inbox_import.JOURNAL_NAME
    assert journal.is_file()
    assert oct(journal.stat().st_mode & 0o777) == "0o600"
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert set(entry) == {
        "sha256",
        "source_name",
        "customer_id",
        "meeting_id",
        "at",
    }
    assert entry["sha256"] == hashlib.sha256("記録帳の本文\n".encode("utf-8")).hexdigest()
    assert entry["source_name"] == "2026-11-09 架空商事.txt"
    assert entry["customer_id"] == result.customer_id
    assert entry["meeting_id"] == result.meeting_id
    assert entry["at"] == "2026-11-09T12:00:00+09:00"

    # 記録帳は取り込み対象にならない（結果にも出ない）。
    assert sorted(path.name for path in inbox.iterdir()) == sorted(
        [inbox_import.JOURNAL_NAME, ARCHIVE_DIR_NAME]
    )


def test_edited_draft_after_failed_move_is_not_imported_twice(
    store: CustomerStore, inbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """移動に失敗→下書きの本文を直す→再取り込みしても、下書きは1件のまま."""
    note = write_note(inbox, "2026-11-09 架空商事.txt", "原文の誤字あり\n")
    real_move = inbox_import._move_to_archive
    monkeypatch.setattr(inbox_import, "_move_to_archive", _fail_move)

    [first] = import_inbox(store, inbox, now=NOW)

    assert first.status == STATUS_FAILED
    customer_id = first.customer_id
    [meeting] = store.list_meetings(customer_id)
    store.update_meeting_draft(
        customer_id,
        meeting["record_id"],
        expected_revision=meeting["revision"],
        title=meeting["title"],
        summary=meeting["summary"],
        details="原文の誤字を直した\n",
        decisions=meeting["decisions"],
        next_actions=meeting["next_actions"],
    )

    monkeypatch.setattr(inbox_import, "_move_to_archive", real_move)
    [second] = import_inbox(store, inbox, now=NOW)

    assert second.status == STATUS_IMPORTED
    assert second.reason == ALREADY_IMPORTED_REASON
    assert second.meeting_id == first.meeting_id
    assert not note.exists()
    meetings = store.list_meetings(customer_id)
    assert len(meetings) == 1
    assert meetings[0]["details"] == "原文の誤字を直した\n"


def test_edited_confirmed_meeting_is_not_imported_twice(
    store: CustomerStore, inbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """確定した後でも、同じ入力は新しい下書きにしない."""
    note = write_note(inbox, "2026-11-09 架空商事.txt", "確定前の本文\n")
    real_move = inbox_import._move_to_archive
    monkeypatch.setattr(inbox_import, "_move_to_archive", _fail_move)
    [first] = import_inbox(store, inbox, now=NOW)

    customer_id = first.customer_id
    [meeting] = store.list_meetings(customer_id)
    updated = store.update_meeting_draft(
        customer_id,
        meeting["record_id"],
        expected_revision=meeting["revision"],
        title=meeting["title"],
        summary=meeting["summary"],
        details="確定した本文\n",
        decisions=meeting["decisions"],
        next_actions=meeting["next_actions"],
    )
    store.confirm_meeting(
        customer_id,
        meeting["record_id"],
        expected_revision=updated["revision"],
        confirmed_by="test",
    )

    monkeypatch.setattr(inbox_import, "_move_to_archive", real_move)
    [second] = import_inbox(store, inbox, now=NOW)

    assert second.status == STATUS_IMPORTED
    assert second.reason == ALREADY_IMPORTED_REASON
    assert second.meeting_id == first.meeting_id
    assert not note.exists()
    [confirmed] = store.list_meetings(customer_id)
    assert confirmed["meeting_status"] == "confirmed"
    assert confirmed["details"] == "確定した本文\n"


def test_broken_journal_lines_are_skipped_everywhere(
    store: CustomerStore, inbox: Path
) -> None:
    """記録帳が壊れていても落ちない（読めない行は飛ばす）."""
    journal = inbox / inbox_import.JOURNAL_NAME
    journal.write_text(
        "これは JSON ではない\n"
        '{"sha256": 3, "customer_id": null}\n'
        "[1, 2, 3]\n"
        "\n",
        encoding="utf-8",
    )
    write_note(inbox, "2026-11-09 架空商事.txt", "壊れた記録帳でも取り込む\n")

    [result] = import_inbox(store, inbox, now=NOW)

    assert result.status == STATUS_IMPORTED
    assert len(store.list_meetings(result.customer_id)) == 1
    text = journal.read_text(encoding="utf-8")
    assert text.startswith("これは JSON ではない\n")  # 壊れた行はそのまま残す
    assert len(text.splitlines()) == 5  # 壊れた4行＋追記した1行


def test_a_journal_that_cannot_be_written_does_not_stop_the_import(
    store: CustomerStore, inbox: Path
) -> None:
    """記録帳を書けない時は、今までどおり本文の照合に任せて取り込みは続ける."""
    (inbox / inbox_import.JOURNAL_NAME).mkdir()
    source = write_note(inbox, "2026-11-09 架空商事.txt", "記録帳が書けない本文\n")

    [result] = import_inbox(store, inbox, now=NOW)

    assert result.status == STATUS_IMPORTED
    assert not source.exists()
    assert any("記録帳" in message for message in result.notes)


def test_an_import_from_before_the_journal_is_added_to_it(
    store: CustomerStore, inbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """記録帳より前の取り込みでも、本文の照合で拾ったら記録帳に残す（編集後の重複防止）."""
    body = "記録帳より前の取り込み\n"
    customer = store.create_customer(display_name="架空商事")
    customer_id = customer["customer_id"]
    draft = store.create_meeting_draft(
        customer_id,
        occurred_at="2026-11-09T00:00:00+09:00",
        title="2026-11-09 取り込み",
        summary=body.strip(),
        details=body,
    )
    write_note(inbox, "2026-11-09 架空商事.txt", body)
    monkeypatch.setattr(inbox_import, "_move_to_archive", _fail_move)

    [first] = import_inbox(store, inbox, now=NOW)

    assert first.status == STATUS_FAILED
    assert first.meeting_id == draft["record_id"]
    journal = inbox / inbox_import.JOURNAL_NAME
    entries = [
        json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()
    ]
    assert any(entry["meeting_id"] == draft["record_id"] for entry in entries)

    store.update_meeting_draft(
        customer_id,
        draft["record_id"],
        expected_revision=draft["revision"],
        title=draft["title"],
        summary=draft["summary"],
        details="記録帳より前の取り込みを直した\n",
        decisions=[],
        next_actions=[],
    )
    [second] = import_inbox(store, inbox, now=NOW)

    assert second.meeting_id == draft["record_id"]
    assert len(store.list_meetings(customer_id)) == 1


