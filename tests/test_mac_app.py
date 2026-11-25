"""実用版の保存先・手元AI・下書き一覧（mac_app）の試験（すべて tmp_path）."""

from __future__ import annotations

import json
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from shokoban import llm_runtime
from shokoban.customer_store import CLEAN_ROOT_CONFIRMATION, CustomerStore
from shokoban.llm_runtime import LLMRuntimeError, RuntimeConfig
from shokoban.local_llm import LocalLLMSettings
from shokoban.mac_app import (
    SYNTHETIC_NAME_PREFIX,
    MacAppError,
    ensure_inbox,
    list_draft_meetings,
    load_share_labels,
    load_status_record,
    mac_app_paths_from_env,
    prepare_local_llm,
    runtime_status,
    save_share_label,
    save_status_record,
    share_dir_for_mode,
    status_local_dir_for_mode,
    status_record_path,
    store_root_for_mode,
    synthetic_inbox_filter,
)
from shokoban.progress_status import (
    ProgressStatus,
    ProgressStatusError,
)

NOW = datetime(2026, 11, 9, 10, 0, 0, tzinfo=timezone(timedelta(hours=9)))
STATUS = ProgressStatus("提案", "見積送付", "2026-11-20", "先方")


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def make_runtime_config(tmp_path: Path) -> RuntimeConfig:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model = models_dir / "fake-model.gguf"
    model.write_bytes(b"GGUF-fake")
    return RuntimeConfig(
        port=18765,
        ctx=4096,
        idle_minutes=10,
        min_free_percent=20,
        default_model="fake-model",
        models={"fake-model": str(model), "fake-small": str(model)},
        state_dir=tmp_path / "runtime-state",
    )


@pytest.fixture()
def store(tmp_path: Path) -> CustomerStore:
    target = CustomerStore(tmp_path / "store")
    target.initialize(confirmation=CLEAN_ROOT_CONFIRMATION)
    return target


def test_without_env_returns_none() -> None:
    assert mac_app_paths_from_env({}) is None


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_env_is_rejected(value: str) -> None:
    with pytest.raises(MacAppError):
        mac_app_paths_from_env({"SHOKOBAN_WORK_DIR": value})


def test_relative_env_is_rejected() -> None:
    with pytest.raises(MacAppError):
        mac_app_paths_from_env({"SHOKOBAN_WORK_DIR": "work"})


def test_missing_env_dir_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MacAppError):
        mac_app_paths_from_env({"SHOKOBAN_WORK_DIR": str(tmp_path / "no-such-dir")})


def test_file_is_rejected(tmp_path: Path) -> None:
    file_path = tmp_path / "work"
    file_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(MacAppError):
        mac_app_paths_from_env({"SHOKOBAN_WORK_DIR": str(file_path)})


def test_symlink_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(MacAppError):
        mac_app_paths_from_env({"SHOKOBAN_WORK_DIR": str(link)})


def test_valid_env_yields_every_path(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()

    paths = mac_app_paths_from_env({"SHOKOBAN_WORK_DIR": str(work)})

    assert paths is not None
    assert paths.work_dir == work
    assert paths.store_root == work / "store-synthetic"
    assert paths.inbox == work / "inbox"
    assert paths.outbox == work / "outbox"
    assert paths.held == work / "outbox-held"
    assert paths.runtime_config == work / "runtime.json"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("架空顧客A", True),
        ("架空顧客Z", True),
        ("架空顧客", True),
        ("架空商事", False),
        ("顧客架空", False),
        ("実在っぽい名前", False),
        ("", False),
    ],
)
def test_synthetic_inbox_filter(name: str, expected: bool) -> None:
    assert SYNTHETIC_NAME_PREFIX == "架空顧客"
    assert synthetic_inbox_filter(name) is expected


def test_prepare_local_llm_returns_lmstudio_settings_with_ensured_alias(
    tmp_path: Path,
) -> None:
    cfg = make_runtime_config(tmp_path)
    calls: list[tuple[RuntimeConfig, str | None]] = []

    def fake_ensure(config: RuntimeConfig, alias: str | None) -> str:
        calls.append((config, alias))
        return "fake-small"

    settings = prepare_local_llm(cfg, "fake-small", ensure=fake_ensure)

    assert settings == LocalLLMSettings(provider="lmstudio", model="fake-small")
    assert calls == [(cfg, "fake-small")]


def test_prepare_local_llm_passes_none_alias_through(tmp_path: Path) -> None:
    cfg = make_runtime_config(tmp_path)
    seen: list[str | None] = []

    def fake_ensure(config: RuntimeConfig, alias: str | None) -> str:
        seen.append(alias)
        return "fake-model"

    settings = prepare_local_llm(cfg, None, ensure=fake_ensure)

    assert settings.model == "fake-model"
    assert seen == [None]


def test_prepare_local_llm_reraises_runtime_error(tmp_path: Path) -> None:
    cfg = make_runtime_config(tmp_path)

    def boom(config: RuntimeConfig, alias: str | None) -> str:
        raise LLMRuntimeError("メモリが足りないので手元AIを起動しません（空き20%）")

    with pytest.raises(LLMRuntimeError) as caught:
        prepare_local_llm(cfg, None, ensure=boom)
    assert "空き20%" in str(caught.value)


def test_runtime_status_reports_state_without_starting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_runtime_config(tmp_path)
    cfg.state_dir.mkdir(parents=True)

    class FakeLedger:
        alias = "fake-model"

    # 台帳方式: 稼働中と別名は「自分のものと確認できた台帳」から読む
    monkeypatch.setattr(llm_runtime, "current_ledger", lambda config: FakeLedger())
    monkeypatch.setattr(llm_runtime, "idle_seconds", lambda config: 123.5)

    assert runtime_status(cfg) == {
        "running": True,
        "alias": "fake-model",
        "idle_seconds": 123.5,
    }


def test_runtime_status_when_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = make_runtime_config(tmp_path)
    monkeypatch.setattr(llm_runtime, "is_running", lambda config: False)
    monkeypatch.setattr(llm_runtime, "idle_seconds", lambda config: None)

    assert runtime_status(cfg) == {
        "running": False,
        "alias": None,
        "idle_seconds": None,
    }


def test_list_draft_meetings_returns_only_drafts_newest_first(
    store: CustomerStore,
) -> None:
    customer_a = store.create_customer(display_name="架空顧客A")
    customer_b = store.create_customer(display_name="架空顧客B")
    old = store.create_meeting_draft(
        customer_a["customer_id"],
        occurred_at="2026-11-01T00:00:00+09:00",
        title="古い下書き",
        summary="要約A",
        details="本文A",
    )
    new = store.create_meeting_draft(
        customer_b["customer_id"],
        occurred_at="2026-11-05T00:00:00+09:00",
        title="新しい下書き",
        summary="要約B",
        details="本文B",
    )
    confirmed = store.create_meeting_draft(
        customer_a["customer_id"],
        occurred_at="2026-11-09T00:00:00+09:00",
        title="確定済み",
        summary="要約C",
        details="本文C",
    )
    store.confirm_meeting(
        customer_a["customer_id"],
        confirmed["record_id"],
        expected_revision=confirmed["revision"],
        confirmed_by="synthetic-human",
    )

    drafts = list_draft_meetings(store)

    assert [draft["record_id"] for draft in drafts] == [
        new["record_id"],
        old["record_id"],
    ]
    assert drafts[0] == {
        "customer_id": customer_b["customer_id"],
        "display_name": "架空顧客B",
        "record_id": new["record_id"],
        "revision": new["revision"],
        "occurred_at": "2026-11-05T00:00:00+09:00",
        "title": "新しい下書き",
        "summary": "要約B",
        "details": "本文B",
        "meeting_status": "draft",
    }
    # include_confirmed=True なら確定済みも入れる
    with_confirmed = list_draft_meetings(store, include_confirmed=True)
    assert confirmed["record_id"] in [draft["record_id"] for draft in with_confirmed]
    assert len(with_confirmed) == len(drafts) + 1


def test_list_draft_meetings_is_empty_without_meetings(store: CustomerStore) -> None:
    assert list_draft_meetings(store) == []


def test_store_root_for_mode_switches_real_and_synthetic(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    paths = mac_app_paths_from_env({"SHOKOBAN_WORK_DIR": str(work)})
    assert paths is not None

    assert store_root_for_mode(paths, "real") == work / "store"
    assert store_root_for_mode(paths, "synthetic") == work / "store-synthetic"
    assert store_root_for_mode(paths, "synthetic") == paths.store_root


def test_store_root_for_mode_rejects_unknown_mode(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    paths = mac_app_paths_from_env({"SHOKOBAN_WORK_DIR": str(work)})
    assert paths is not None

    with pytest.raises(MacAppError):
        store_root_for_mode(paths, "production")


def make_paths(tmp_path: Path) -> tuple:
    work = tmp_path / "work"
    work.mkdir()
    paths = mac_app_paths_from_env({"SHOKOBAN_WORK_DIR": str(work)})
    assert paths is not None
    return paths, work


def test_share_dir_for_mode_switches_real_and_synthetic(tmp_path: Path) -> None:
    paths, work = make_paths(tmp_path)

    assert share_dir_for_mode(paths, "real") == work / "share"
    assert share_dir_for_mode(paths, "synthetic") == work / "share-synthetic"
    with pytest.raises(MacAppError):
        share_dir_for_mode(paths, "production")


def test_status_local_dir_for_mode_switches_real_and_synthetic(
    tmp_path: Path,
) -> None:
    paths, work = make_paths(tmp_path)

    assert status_local_dir_for_mode(paths, "real") == work / "status-local"
    assert (
        status_local_dir_for_mode(paths, "synthetic")
        == work / "status-local-synthetic"
    )
    with pytest.raises(MacAppError):
        status_local_dir_for_mode(paths, "production")


def test_status_record_path_uses_the_customer_id(tmp_path: Path) -> None:
    paths, work = make_paths(tmp_path)

    assert status_record_path(paths, "real", "cus_abc123") == (
        work / "status-local" / "cus_abc123.json"
    )
    for bad in ("", "../evil", "a/b", "a.b", None):
        with pytest.raises(MacAppError):
            status_record_path(paths, "real", bad)  # type: ignore[arg-type]


def test_save_and_load_share_labels(tmp_path: Path) -> None:
    paths, work = make_paths(tmp_path)
    label_path = status_local_dir_for_mode(paths, "real") / "share-labels.json"

    assert load_share_labels(label_path) == {}
    save_share_label(label_path, "cus_a", "  やまだ-01  ")
    save_share_label(label_path, "cus_b", "甲&乙")

    assert load_share_labels(label_path) == {
        "cus_a": "やまだ-01",
        "cus_b": "甲&乙",
    }
    assert json.loads(label_path.read_text(encoding="utf-8")) == {
        "cus_a": "やまだ-01",
        "cus_b": "甲&乙",
    }
    assert mode_of(label_path) == 0o600
    assert mode_of(label_path.parent) == 0o700

    save_share_label(label_path, "cus_a", "やまだ-02")
    assert load_share_labels(label_path)["cus_a"] == "やまだ-02"
    assert load_share_labels(label_path)["cus_b"] == "甲&乙"


def test_save_share_label_rejects_bad_values(tmp_path: Path) -> None:
    paths, work = make_paths(tmp_path)
    label_path = work / "status-local" / "share-labels.json"

    with pytest.raises(ProgressStatusError):
        save_share_label(label_path, "cus_a", "やまだ/01")
    with pytest.raises(MacAppError):
        save_share_label(label_path, "../cus_a", "やまだ")
    assert not label_path.exists()


def test_save_share_label_refuses_a_label_used_by_another_customer(
    tmp_path: Path,
) -> None:
    """別のお客様に同じ呼び名を登録できない."""
    paths, work = make_paths(tmp_path)
    label_path = work / "status-local" / "share-labels.json"
    save_share_label(label_path, "cus_a", "案件A")

    with pytest.raises(MacAppError) as error:
        save_share_label(label_path, "cus_b", "案件A")

    assert str(error.value) == "その呼び名は別のお客様が使っています。"
    assert load_share_labels(label_path) == {"cus_a": "案件A"}
    assert json.loads(label_path.read_text(encoding="utf-8")) == {"cus_a": "案件A"}


def test_save_share_label_treats_case_variants_as_the_same_label(
    tmp_path: Path,
) -> None:
    """大文字小文字だけ違う呼び名も取り違えの元なので拒否する（同じ顧客の付け直しは可）."""
    paths, work = make_paths(tmp_path)
    label_path = work / "status-local" / "share-labels.json"
    save_share_label(label_path, "cus_a", "Project-A")

    for variant in ("project-a", "PROJECT-A"):
        with pytest.raises(MacAppError):
            save_share_label(label_path, "cus_b", variant)

    # 同じお客様の付け直しは通る。
    save_share_label(label_path, "cus_a", "project-a")
    assert load_share_labels(label_path) == {"cus_a": "project-a"}


def test_load_share_labels_ignores_broken_and_unknown_files(
    tmp_path: Path,
) -> None:
    paths, work = make_paths(tmp_path)
    directory = status_local_dir_for_mode(paths, "real")
    directory.mkdir()
    label_path = directory / "share-labels.json"

    label_path.write_text("{broken", encoding="utf-8")
    assert load_share_labels(label_path) == {}

    label_path.write_text(
        json.dumps(
            {"cus_a": "やまだ", "cus_b": "a/b", "cus_c": 3, "cus_d": "1"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert load_share_labels(label_path) == {"cus_a": "やまだ", "cus_d": "1"}

    target = tmp_path / "target.json"
    target.write_text('{"cus_a": "やまだ"}', encoding="utf-8")
    link = directory / "linked.json"
    link.symlink_to(target)
    assert load_share_labels(link) == {}


def test_save_and_load_status_record(tmp_path: Path) -> None:
    paths, work = make_paths(tmp_path)
    record_path = status_record_path(paths, "real", "cus_a")
    evidence = {
        "今どこまで": "見積もりを作った。",
        "次にやること": "見積もりを送る。",
        "期限": "来週金曜まで。",
        "次に動く側": "先方の返事。",
    }

    saved = save_status_record(
        record_path,
        STATUS,
        evidence,
        raw_ok=True,
        updated=NOW,
        source="手元AI",
        inferred=STATUS,
    )

    assert saved == record_path
    assert mode_of(record_path) == 0o600
    assert mode_of(record_path.parent) == 0o700
    loaded = load_status_record(record_path)
    assert loaded is not None
    assert loaded["status"] == STATUS
    assert loaded["inferred"] == STATUS
    assert loaded["evidence"] == evidence
    assert loaded["raw_ok"] is True
    assert loaded["source"] == "手元AI"
    assert loaded["updated"] == "2026-11-09T10:00:00+09:00"

    edited = ProgressStatus("契約", "制作", "なし", "こちら")
    save_status_record(
        record_path,
        edited,
        {"今どこまで": "x" * 81},
        raw_ok=False,
        updated=NOW,
        source="人が修正",
        inferred=STATUS,
    )
    loaded = load_status_record(record_path)
    assert loaded is not None
    assert loaded["status"] == edited
    assert loaded["inferred"] == STATUS  # 修正前の手元AIの値は手元に残る
    assert loaded["evidence"]["今どこまで"] == ""
    assert loaded["raw_ok"] is False
    assert loaded["source"] == "人が修正"


def test_save_status_record_rejects_bad_values(tmp_path: Path) -> None:
    paths, work = make_paths(tmp_path)
    record_path = status_record_path(paths, "real", "cus_a")

    with pytest.raises(ProgressStatusError):
        save_status_record(
            record_path,
            ProgressStatus("提案中", "見積送付", "2026-11-20", "先方"),
            {},
            raw_ok=True,
            updated=NOW,
            source="手元AI",
        )
    with pytest.raises(MacAppError):
        save_status_record(
            record_path, STATUS, {}, raw_ok=True, updated=NOW, source="自由文"
        )
    with pytest.raises(MacAppError):
        save_status_record(
            record_path, STATUS, {}, raw_ok=True, updated="now", source="手元AI"  # type: ignore[arg-type]
        )
    assert not record_path.exists()


def test_load_status_record_rejects_missing_or_invalid_files(
    tmp_path: Path,
) -> None:
    paths, work = make_paths(tmp_path)
    record_path = status_record_path(paths, "real", "cus_a")

    assert load_status_record(record_path) is None

    record_path.parent.mkdir()
    record_path.write_text("{broken", encoding="utf-8")
    assert load_status_record(record_path) is None

    record_path.write_text(
        json.dumps(
            {
                "status": {
                    "stage": "提案中",
                    "next_task": "見積送付",
                    "due": "2026-11-20",
                    "waiting": "先方",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert load_status_record(record_path) is None


def test_ensure_inbox_creates_a_private_folder_once(tmp_path: Path) -> None:
    paths = mac_app_paths_from_env({"SHOKOBAN_WORK_DIR": str(tmp_path)})
    assert paths is not None and not paths.inbox.exists()
    assert ensure_inbox(paths) == paths.inbox
    assert paths.inbox.is_dir()
    assert stat.S_IMODE(paths.inbox.stat().st_mode) == 0o700
    (paths.inbox / "2030-01-15 架空顧客サンプル.txt").write_text("本文", encoding="utf-8")
    ensure_inbox(paths)  # 二度目は中身に触れない
    assert (paths.inbox / "2030-01-15 架空顧客サンプル.txt").read_text(encoding="utf-8") == "本文"


def test_ensure_inbox_refuses_a_symlink_or_a_file(tmp_path: Path) -> None:
    paths = mac_app_paths_from_env({"SHOKOBAN_WORK_DIR": str(tmp_path)})
    assert paths is not None
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    paths.inbox.symlink_to(elsewhere)
    with pytest.raises(MacAppError):
        ensure_inbox(paths)
    paths.inbox.unlink()
    paths.inbox.write_text("ファイル", encoding="utf-8")
    with pytest.raises(MacAppError):
        ensure_inbox(paths)
