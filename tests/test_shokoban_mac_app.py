"""書庫番の実用画面（gui/shokoban_mac.py）の試験.

``SHOKOBAN_WORK_DIR`` を tmp_path に向け、実プロセス（llama-server・streamlit）と
ネット接続は一切使わない。手元AIは prepare_local_llm / generate_local_text を
偽物にして確かめる。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from shokoban.customer_store import CLEAN_ROOT_CONFIRMATION, CustomerStore
from shokoban.llm_runtime import LLMRuntimeError
from shokoban.local_llm import LocalLLMError, LocalLLMSettings
from shokoban.progress_status import MOVER, STAGES, TASKS

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - GUI dependencies are optional.
    AppTest = None

APP_PATH = Path(__file__).resolve().parents[1] / "gui" / "shokoban_mac.py"
STATUS_OK = json.dumps(
    {
        "今どこまで": "見積",
        "次にやること": "見積送付",
        "期限": "2026-11-20",
        "次に動く側": "先方",
        "根拠": {
            "今どこまで": "見積もりを作った。",
            "次にやること": "見積もりを送る。",
            "期限": "来週金曜まで。",
            "次に動く側": "先方の返事。",
        },
    },
    ensure_ascii=False,
)
STATUS_BAD = json.dumps(
    {
        "今どこまで": "たぶん提案",
        "次にやること": "自由に書きます",
        "期限": "2027-12-31",
        "次に動く側": "山田さん",
        "根拠": {"今どこまで": "根拠の自由文"},
    },
    ensure_ascii=False,
)
FAKE_SETTINGS = LocalLLMSettings(provider="lmstudio", model="fake-model")

pytestmark = pytest.mark.skipif(
    AppTest is None, reason="install the gui extra to run Streamlit tests"
)


def make_work_dir(tmp_path: Path) -> Path:
    """runtime.json（偽の .gguf）と inbox/outbox/outbox-held を持つ作業場所."""
    work = tmp_path / "work"
    work.mkdir()
    for name in ("inbox", "outbox", "outbox-held"):
        (work / name).mkdir()
    models_dir = work / "models"
    models_dir.mkdir()
    model = models_dir / "fake-model.gguf"
    model.write_bytes(b"GGUF-fake")
    (work / "runtime.json").write_text(
        json.dumps(
            {
                "port": 1234,
                "ctx": 8192,
                "idle_minutes": 10,
                "min_free_percent": 20,
                "default_model": "fake-model",
                "models": {
                    "fake-small": str(model),
                    "fake-model": str(model),
                },
                "state_dir": str(work / "runtime-state"),
            }
        ),
        encoding="utf-8",
    )
    return work


def use_env(
    monkeypatch: pytest.MonkeyPatch, work: Path, mode: str | None
) -> None:
    monkeypatch.setenv("SHOKOBAN_WORK_DIR", str(work))
    if mode is None:
        monkeypatch.delenv("SHOKOBAN_DATA_MODE", raising=False)
    else:
        monkeypatch.setenv("SHOKOBAN_DATA_MODE", mode)


def run_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert list(app.exception) == []
    return app


def click(app: AppTest, key: str) -> AppTest:
    app.button(key=key).click().run()
    assert list(app.exception) == []
    return app


def open_customer(app: AppTest, customer_id: str) -> AppTest:
    return click(app, f"open_customer_{customer_id}")


def seed_store(work: Path, name: str = "山田商事") -> tuple[CustomerStore, dict]:
    """real モードの保存先に、お客様1人だけを手で作る（種まきではない）."""
    store = CustomerStore(work / "store")
    store.initialize(confirmation=CLEAN_ROOT_CONFIRMATION)
    return store, store.create_customer(display_name=name)


def add_draft(
    store: CustomerStore,
    customer_id: str,
    *,
    details: str = "取り込んだ下書きの本文です。\n",
    day: str = "2026-11-09",
) -> dict:
    return store.create_meeting_draft(
        customer_id,
        occurred_at=f"{day}T00:00:00+09:00",
        title=f"{day} 取り込み",
        summary=details.strip()[:120],
        details=details,
    )


def add_confirmed(
    store: CustomerStore,
    customer_id: str,
    *,
    details: str = "確定済みの打ち合わせ本文です。",
    day: str = "2026-11-05",
    title: str = "打合せ",
) -> dict:
    meeting = add_draft(store, customer_id, details=details, day=day)
    store.update_meeting_draft(
        customer_id,
        meeting["record_id"],
        expected_revision=meeting["revision"],
        title=title,
        summary="",
        details=details,
        decisions=[],
        next_actions=[],
    )
    current = store.list_meetings(customer_id)[0]
    return store.confirm_meeting(
        customer_id,
        current["record_id"],
        expected_revision=current["revision"],
        confirmed_by="test",
    )


def snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def fake_llm(
    *,
    settings: LocalLLMSettings = FAKE_SETTINGS,
    text: str = "答えです。",
):
    """prepare_local_llm と generate_local_text をまとめて差し替える."""
    return (
        mock.patch("shokoban.mac_app.prepare_local_llm", return_value=settings),
        mock.patch("shokoban.local_llm.generate_local_text", return_value=text),
    )


# --- 起動の守り -----------------------------------------------------------


def test_data_mode_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, None)

    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()

    assert list(app.exception) == []
    assert any("SHOKOBAN_DATA_MODE" in message.value for message in app.error)
    assert list(app.header) == []
    assert not (work / "store").exists()
    assert not (work / "store-synthetic").exists()


def test_unknown_data_mode_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "production")

    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()

    assert list(app.exception) == []
    assert any(
        "real か synthetic" in message.value for message in app.error
    )
    assert not (work / "store").exists()
    assert not (work / "store-synthetic").exists()


def test_no_customers_are_seeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")

    run_app()

    assert CustomerStore(work / "store").list_customers() == []
    assert not (work / "store-synthetic").exists()


# --- 案件一覧 -------------------------------------------------------------


def test_real_mode_imports_names_that_are_not_synthetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    source = work / "inbox" / "2026-11-09 山田商事.txt"
    source.write_text("実際の打ち合わせ本文です。\n", encoding="utf-8")

    app = run_app()

    assert any("本番データ" in message.value for message in app.markdown)
    assert any(
        message.value == "対象ファイル: 1件" for message in app.markdown
    )
    click(app, "run_inbox_import")

    rows = app.dataframe[0].value
    by_name = dict(zip(list(rows["ファイル名"]), list(rows["結果"])))
    assert by_name["2026-11-09 山田商事.txt"] == "取り込み"
    assert not source.exists()

    store = CustomerStore(work / "store")
    customers = store.list_customers()
    assert [customer["display_name"] for customer in customers] == ["山田商事"]
    [meeting] = store.list_meetings(customers[0]["customer_id"])
    assert meeting["meeting_status"] == "draft"
    assert meeting["details"] == "実際の打ち合わせ本文です。\n"
    assert not (work / "store-synthetic").exists()


def test_synthetic_mode_takes_only_synthetic_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "synthetic")
    accepted = work / "inbox" / "2026-11-09 架空顧客Z.txt"
    accepted.write_text("架空の取り込み本文です。\n", encoding="utf-8")
    rejected = work / "inbox" / "2026-11-09 実在っぽい名前.txt"
    rejected.write_text("取り込まない本文です。\n", encoding="utf-8")

    app = run_app()

    assert any("架空データ" in message.value for message in app.markdown)
    assert any(
        message.value == "対象ファイル: 1件" for message in app.markdown
    )
    click(app, "run_inbox_import")

    rows = app.dataframe[0].value
    by_name = dict(zip(list(rows["ファイル名"]), list(rows["結果"])))
    assert by_name["2026-11-09 架空顧客Z.txt"] == "取り込み"
    assert by_name["2026-11-09 実在っぽい名前.txt"] == "見送り"
    assert rejected.is_file()

    store = CustomerStore(work / "store-synthetic")
    assert [customer["display_name"] for customer in store.list_customers()] == [
        "架空顧客Z"
    ]
    assert not (work / "store").exists()


def test_create_customer_opens_the_existing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    app = run_app()

    app.text_input(key="new_customer_name").input("新規商事").run()
    click(app, "create_customer")

    store = CustomerStore(work / "store")
    assert [customer["display_name"] for customer in store.list_customers()] == [
        "新規商事"
    ]
    assert app.session_state.shokoban_screen == "customer"

    click(app, "nav_cases")
    app.text_input(key="new_customer_name").input("新規商事").run()
    click(app, "create_customer")

    assert len(store.list_customers()) == 1
    assert app.session_state.shokoban_screen == "customer"


# --- 案件ファイル ---------------------------------------------------------


def test_draft_can_be_edited_then_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work)
    draft = add_draft(store, customer["customer_id"])
    app = run_app()
    open_customer(app, customer["customer_id"])

    confirm_key = f"confirm_draft_{draft['record_id']}"
    assert app.button(key=confirm_key).disabled
    app.text_area(key=f"draft_details_{draft['record_id']}").input(
        "直した本文です。\n"
    ).run()
    assert list(app.exception) == []
    app.checkbox(key=f"draft_checked_{draft['record_id']}").check().run()
    assert not app.button(key=confirm_key).disabled

    click(app, confirm_key)

    [meeting] = store.list_meetings(customer["customer_id"])
    assert meeting["meeting_status"] == "confirmed"
    assert meeting["details"] == "直した本文です。\n"
    assert meeting["confirmed_by"] == "gui"
    assert any(
        message.value == "下書きを確定しました。" for message in app.success
    )


def test_confirmed_meeting_can_be_reopened_edited_and_confirmed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work)
    draft = add_draft(store, customer["customer_id"])
    confirmed = store.confirm_meeting(
        customer["customer_id"], draft["record_id"], expected_revision=draft["revision"], confirmed_by="gui"
    )
    app = run_app()
    open_customer(app, customer["customer_id"])

    reopen_key = f"reopen_meeting_{draft['record_id']}"
    assert app.button(key=reopen_key).disabled  # チェックを入れるまで押せない
    app.checkbox(key=f"reopen_checked_{draft['record_id']}").check().run()
    click(app, reopen_key)

    [meeting] = store.list_meetings(customer["customer_id"])
    assert meeting["meeting_status"] == "draft"
    assert meeting["confirmed_by"] is None and meeting["confirmed_at"] is None
    assert meeting["revision"] == confirmed["revision"] + 1
    assert any("下書きに戻しました" in message.value for message in app.success)

    # 下書きの欄で直して、もう一度確定する（確認のチェックは付け直す）
    confirm_key = f"confirm_draft_{draft['record_id']}"
    assert app.button(key=confirm_key).disabled
    app.text_area(key=f"draft_details_{draft['record_id']}").input("確定のあとで直した本文です。\n").run()
    app.checkbox(key=f"draft_checked_{draft['record_id']}").check().run()
    click(app, confirm_key)
    [meeting] = store.list_meetings(customer["customer_id"])
    assert meeting["meeting_status"] == "confirmed"
    assert meeting["details"] == "確定のあとで直した本文です。\n"
    assert list(app.exception) == []


def test_note_requires_the_checkbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work)
    app = run_app()
    open_customer(app, customer["customer_id"])

    assert app.button(key="add_manual_note").disabled
    app.text_area(key="manual_note_text").input("電話で確認した。").run()
    assert app.button(key="add_manual_note").disabled
    app.checkbox(key="manual_note_checked").check().run()
    assert not app.button(key="add_manual_note").disabled

    click(app, "add_manual_note")

    [item] = store.current_information(customer["customer_id"])
    metadata, body = item
    assert body == "電話で確認した。"
    assert metadata["kind"] == "note"
    assert metadata["origin"] == "manual"
    assert metadata["observed_at"].endswith("+09:00")


def test_ask_shows_answer_and_saves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work)
    add_confirmed(store, customer["customer_id"])
    before = snapshot(work / "store")
    app = run_app()
    open_customer(app, customer["customer_id"])

    prepare, generate = fake_llm(text="見積もりの返事待ちです。")
    with prepare, generate:
        app.text_input(key="case_question").input("次は何を待つ?").run()
        app.button(key="ask_case").click().run()

    assert list(app.exception) == []
    # 答えは Markdown ではなく、素のテキストとして出す。
    assert any(
        message.value == "答え: 見積もりの返事待ちです。" for message in app.text
    )
    assert any(
        message.value == "使った記録: 2026-11-05 打合せ" for message in app.text
    )
    assert not any(
        "見積もりの返事待ちです。" in message.value for message in app.markdown
    )
    assert snapshot(work / "store") == before


def test_answer_can_be_added_as_a_note_only_after_checking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work)
    app = run_app()
    open_customer(app, customer["customer_id"])

    prepare, generate = fake_llm(text="記録にありません")
    with prepare, generate:
        app.text_input(key="case_question").input("何かありますか").run()
        app.button(key="ask_case").click().run()
    assert list(app.exception) == []
    assert app.button(key="add_answer_note").disabled

    app.checkbox(key="answer_note_checked").check().run()
    assert not app.button(key="add_answer_note").disabled
    click(app, "add_answer_note")

    [item] = store.current_information(customer["customer_id"])
    metadata, body = item
    assert "答え: 記録にありません" in body
    assert metadata["origin"] == "manual"
    assert metadata["kind"] == "note"


def test_llm_errors_are_shown_with_their_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work)
    app = run_app()
    open_customer(app, customer["customer_id"])

    app.text_input(key="case_question").input("何かありますか").run()
    with mock.patch(
        "shokoban.mac_app.prepare_local_llm",
        side_effect=LLMRuntimeError(
            "メモリが足りないので手元AIを起動しません（空き20%）"
        ),
    ):
        app.button(key="ask_case").click().run()
    assert any(
        message.value == "メモリが足りないので手元AIを起動しません（空き20%）"
        for message in app.error
    )

    with mock.patch(
        "shokoban.mac_app.prepare_local_llm", return_value=FAKE_SETTINGS
    ), mock.patch(
        "shokoban.local_llm.generate_local_text",
        side_effect=LocalLLMError("connection refused"),
    ):
        app.button(key="ask_case").click().run()
    assert any(
        message.value == "手元AIが応答しませんでした。" for message in app.error
    )


# --- 進行の状態 -----------------------------------------------------------


def save_shared_label(app: Any, customer_id: str, label: str) -> Any:
    app.text_input(key=f"share_label_input_{customer_id}").input(label).run()
    return click(app, f"save_share_label_{customer_id}")


def build_status(app: Any, customer_id: str) -> Any:
    return click(app, f"build_progress_status_{customer_id}")


def share_path(work: Path, mode: str, label: str) -> Path:
    directory = work / ("share" if mode == "real" else "share-synthetic")
    return directory / f"{label}.md"


def parse_share_file(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- "):
            key, _, value = line[2:].partition(": ")
            fields[key] = value
    return fields


def all_screen_text(app: Any) -> str:
    parts: list[str] = []
    for name in (
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "code",
        "text",
        "info",
        "warning",
        "error",
        "success",
        "button",
        "selectbox",
        "text_input",
        "date_input",
        "checkbox",
    ):
        for element in getattr(app, name, []):
            for attribute in ("value", "label"):
                value = getattr(element, attribute, None)
                if value is not None:
                    parts.append(str(value))
    return "\n".join(parts)


# 「ブラウザが勝手に取りに行く」文字。
PAYLOAD_IMAGE = "![x](http://127.0.0.1:9/p?v=1)"
PAYLOAD_HTML = "<img src=http://127.0.0.1:9/q>"
PAYLOAD_LINK = "[link](http://127.0.0.1:9/r)"
LOOPBACK_FRAGMENTS = ("![", "](http", "<img")
STATUS_WITH_PAYLOAD_EVIDENCE = json.dumps(
    {
        "今どこまで": "見積",
        "次にやること": "見積送付",
        "期限": "2026-11-20",
        "次に動く側": "先方",
        "根拠": {
            "今どこまで": f"根拠{PAYLOAD_IMAGE}",
            "次にやること": f"根拠{PAYLOAD_HTML}",
            "期限": f"根拠{PAYLOAD_LINK}",
            "次に動く側": f"根拠{PAYLOAD_IMAGE}{PAYLOAD_HTML}{PAYLOAD_LINK}",
        },
    },
    ensure_ascii=False,
)


def markdown_and_label_text(app: Any) -> list[str]:
    """Markdown として解釈されうる場所（見出し・本文・ラベル）の文字列.

    ここに ``![``・``](http``・``<img`` が生の形で残っていると、ブラウザが
    外部（この場合は loopback）へ勝手に要求を出す。
    """
    parts: list[str] = []
    for name in (
        "title",
        "header",
        "subheader",
        "caption",
        "markdown",
        "info",
        "warning",
        "error",
        "success",
    ):
        for element in getattr(app, name, []):
            value = getattr(element, "value", None)
            if value is not None:
                parts.append(str(value))
    for name in (
        "button",
        "checkbox",
        "expander",
        "selectbox",
        "text_input",
        "text_area",
        "date_input",
    ):
        for element in getattr(app, name, []):
            label = getattr(element, "label", None)
            if label is not None:
                parts.append(str(label))
    return parts


def _has_unescaped(text: str, fragment: str) -> bool:
    """Markdown の意味を持つ形（バックスラッシュで無害化されていない形）で残っているか."""
    start = 0
    while True:
        index = text.find(fragment, start)
        if index < 0:
            return False
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return True
        start = index + 1


def test_external_text_never_reaches_markdown_or_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """答え・お客様名・題・根拠に画像/HTML/リンクを入れても、
    Markdown のラベルに生の形で残らない（画面から勝手に通信しない）."""
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work, name=f"山田{PAYLOAD_IMAGE}商事")
    customer_id = customer["customer_id"]
    add_confirmed(
        store,
        customer_id,
        title=f"打合せ{PAYLOAD_HTML}",
        details=f"確定本文{PAYLOAD_LINK}",
    )
    store.create_meeting_draft(
        customer_id,
        occurred_at="2026-11-08T00:00:00+09:00",
        title=f"下書き{PAYLOAD_LINK}",
        summary="",
        details=f"下書き本文{PAYLOAD_IMAGE}",
    )
    app = run_app()
    open_customer(app, customer_id)
    save_shared_label(app, customer_id, "やまだ-01")

    prepare, generate = fake_llm(text=STATUS_WITH_PAYLOAD_EVIDENCE)
    with prepare, generate:
        app.text_input(key="case_question").input("次は?").run()
        app.button(key="ask_case").click().run()
        build_status(app, customer_id)
    assert list(app.exception) == []

    # 生の文字はプレーンテキスト（st.text）としてはそのまま出る（消していない）。
    body = "\n".join(str(message.value) for message in app.text)
    assert PAYLOAD_IMAGE in body and PAYLOAD_HTML in body and PAYLOAD_LINK in body

    # 検査そのものが空振りしないこと（無害化していない文字なら必ず見つかる）。
    assert _has_unescaped(f"答え: {PAYLOAD_IMAGE}", "![")
    assert _has_unescaped(f"答え: {PAYLOAD_HTML}", "<img")
    assert not _has_unescaped(r"答え: \!\[x\]", "![")

    # しかし Markdown のラベルには生の形（無害化していない形）で残らない。
    for text in markdown_and_label_text(app):
        for fragment in LOOPBACK_FRAGMENTS:
            assert not _has_unescaped(str(text), fragment), (fragment, text)


def test_progress_status_is_not_shared_without_a_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work)
    add_confirmed(store, customer["customer_id"])
    customer_id = customer["customer_id"]
    app = run_app()
    open_customer(app, customer_id)

    assert any(
        "ほかのエージェントに見えるのはこの呼び名だけです。" in message.value
        for message in app.caption
    )

    prepare, generate = fake_llm(text=STATUS_OK)
    with prepare, generate:
        build_status(app, customer_id)

    assert any(
        message.value == "呼び名が決まっていないので共有していません。"
        for message in app.warning
    )
    assert not (work / "share").exists()

    saved = json.loads(
        (work / "status-local" / f"{customer_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["status"] == {
        "stage": "見積",
        "next_task": "見積送付",
        "due": "2026-11-20",
        "waiting": "先方",
    }
    assert saved["source"] == "手元AI"

    # 作った結果は、その場で下の4欄にも反映される（初期値は保存済みの結果）。
    assert app.selectbox(key=f"status_stage_{customer_id}").value == "見積"
    assert (
        app.selectbox(key=f"status_next_task_{customer_id}").value == "見積送付"
    )
    assert app.selectbox(key=f"status_due_kind_{customer_id}").value == "日付"
    due_input = app.date_input(key=f"status_due_date_{customer_id}")
    assert due_input.value.isoformat() == "2026-11-20"
    assert app.selectbox(key=f"status_mover_{customer_id}").value == "先方"

    # 修正して共有、の側でも呼び名が無ければ共有しない。
    click(app, f"share_edited_status_{customer_id}")
    assert not (work / "share").exists()
    assert any(
        message.value == "呼び名が決まっていないので共有していません。"
        for message in app.warning
    )


def test_share_label_is_validated_then_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work)
    customer_id = customer["customer_id"]
    app = run_app()
    open_customer(app, customer_id)
    labels_path = work / "status-local" / "share-labels.json"

    save_shared_label(app, customer_id, "2026")
    assert any(message.value for message in app.error)
    assert not labels_path.exists()

    save_shared_label(app, customer_id, " やまだ-01 ")
    assert any(
        message.value == "共有用の呼び名を決めました。" for message in app.success
    )
    assert any(
        message.value == "共有用の呼び名: やまだ-01" for message in app.text
    )
    assert json.loads(labels_path.read_text(encoding="utf-8")) == {
        customer_id: "やまだ-01"
    }
    app.run()
    assert list(app.exception) == []
    assert any(
        message.value == "共有用の呼び名: やまだ-01" for message in app.text
    )


def test_share_label_cannot_be_reused_by_another_customer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """画面から別のお客様に同じ呼び名を付けられない（大文字小文字違いも）."""
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work, name="山田商事")
    other = store.create_customer(display_name="佐藤工業")
    labels_path = work / "status-local" / "share-labels.json"
    app = run_app()
    open_customer(app, customer["customer_id"])
    save_shared_label(app, customer["customer_id"], "案件A")

    click(app, "nav_cases")
    open_customer(app, other["customer_id"])
    save_shared_label(app, other["customer_id"], "案件a")

    assert any("別のお客様" in message.value for message in app.error)
    assert json.loads(labels_path.read_text(encoding="utf-8")) == {
        customer["customer_id"]: "案件A"
    }


def test_progress_status_without_confirmed_meetings_does_not_call_the_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work)
    add_draft(store, customer["customer_id"])
    customer_id = customer["customer_id"]
    app = run_app()
    open_customer(app, customer_id)
    save_shared_label(app, customer_id, "やまだ-01")

    with mock.patch(
        "shokoban.mac_app.prepare_local_llm"
    ) as prepare, mock.patch(
        "shokoban.local_llm.generate_local_text"
    ) as generate:
        build_status(app, customer_id)

    assert any(
        message.value == "確定した打ち合わせがまだありません。"
        for message in app.info
    )
    prepare.assert_not_called()
    generate.assert_not_called()
    assert not (work / "share").exists()
    assert not (work / "status-local" / f"{customer_id}.json").exists()


def test_share_file_has_only_choices_and_no_evidence_or_record_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work)
    add_confirmed(
        store, customer["customer_id"], details="社外秘の打ち合わせ本文です。山田商事の話。"
    )
    customer_id = customer["customer_id"]
    app = run_app()
    open_customer(app, customer_id)
    save_shared_label(app, customer_id, "やまだ-01")

    prepare, generate = fake_llm(text=STATUS_BAD)
    with prepare, generate:
        build_status(app, customer_id)

    assert any(
        message.value == "進行の状態を共有しました（ほかのエージェントが読む場所）"
        for message in app.success
    )
    path = share_path(work, "real", "やまだ-01")
    assert path.is_file()
    fields = parse_share_file(path)
    assert fields["今どこまで"] in STAGES
    assert fields["次にやること"] in TASKS
    assert fields["次に動く側"] in MOVER
    assert (
        fields["期限"] in ("なし", "不明")
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", fields["期限"]) is not None
    )
    assert fields["出どころ"] == "手元AI"

    body = path.read_text(encoding="utf-8")
    for forbidden in (
        "根拠",
        "山田商事",
        "社外秘の打ち合わせ本文です。",
        "根拠の自由文",
        "見積もりを作った。",
    ):
        assert forbidden not in body

    saved = json.loads(
        (work / "status-local" / f"{customer_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["raw_ok"] is False
    assert saved["evidence"]["今どこまで"] == "根拠の自由文"


def test_edited_status_is_shared_with_the_manual_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work)
    add_confirmed(store, customer["customer_id"])
    customer_id = customer["customer_id"]
    app = run_app()
    open_customer(app, customer_id)
    save_shared_label(app, customer_id, "やまだ-01")

    prepare, generate = fake_llm(text=STATUS_OK)
    with prepare, generate:
        build_status(app, customer_id)

    app.selectbox(key=f"status_stage_{customer_id}").select("納品済み").run()
    assert list(app.exception) == []
    app.selectbox(key=f"status_next_task_{customer_id}").select("請求").run()
    assert list(app.exception) == []
    app.selectbox(key=f"status_mover_{customer_id}").select("こちら").run()
    assert list(app.exception) == []
    app.selectbox(key=f"status_due_kind_{customer_id}").select("なし").run()
    assert list(app.exception) == []
    click(app, f"share_edited_status_{customer_id}")

    fields = parse_share_file(share_path(work, "real", "やまだ-01"))
    assert fields["案件"] == "やまだ-01"
    assert fields["案件番号"] == customer_id
    assert fields["今どこまで"] == "納品済み"
    assert fields["次にやること"] == "請求"
    assert fields["期限"] == "なし"
    assert fields["次に動く側"] == "こちら"
    assert fields["出どころ"] == "人が修正"

    saved = json.loads(
        (work / "status-local" / f"{customer_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["status"]["stage"] == "納品済み"
    assert saved["source"] == "人が修正"
    assert saved["inferred"]["stage"] == "見積"  # 修正前の手元AIの値


def test_share_folder_is_split_by_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "synthetic")
    store = CustomerStore(work / "store-synthetic")
    store.initialize(confirmation=CLEAN_ROOT_CONFIRMATION)
    customer = store.create_customer(display_name="架空顧客Z")
    add_confirmed(store, customer["customer_id"])
    customer_id = customer["customer_id"]
    app = run_app()
    open_customer(app, customer_id)
    save_shared_label(app, customer_id, "架空顧客ゼット")

    prepare, generate = fake_llm(text=STATUS_OK)
    with prepare, generate:
        build_status(app, customer_id)

    assert (work / "share-synthetic" / "架空顧客ゼット.md").is_file()
    assert not (work / "share").exists()
    assert (
        work / "status-local-synthetic" / f"{customer_id}.json"
    ).is_file()
    assert not (work / "status-local").exists()


def test_progress_status_local_llm_error_is_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work)
    add_confirmed(store, customer["customer_id"])
    customer_id = customer["customer_id"]
    app = run_app()
    open_customer(app, customer_id)
    save_shared_label(app, customer_id, "やまだ-01")

    with mock.patch(
        "shokoban.mac_app.prepare_local_llm", return_value=FAKE_SETTINGS
    ), mock.patch(
        "shokoban.local_llm.generate_local_text",
        side_effect=LocalLLMError("connection refused"),
    ):
        build_status(app, customer_id)

    assert any(
        message.value == "手元AIが応答しませんでした。" for message in app.error
    )
    assert not (work / "share").exists()
    assert not (work / "status-local" / f"{customer_id}.json").exists()


def test_screens_never_show_ollama_or_lm_studio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    store, customer = seed_store(work)
    add_confirmed(store, customer["customer_id"])
    customer_id = customer["customer_id"]
    app = run_app()
    open_customer(app, customer_id)

    for screen_app in (app, click(app, "nav_llm")):
        text = all_screen_text(screen_app)
        assert "Ollama" not in text
        assert "LM Studio" not in text


# --- 手元AI ---------------------------------------------------------------


def test_llm_screen_shows_status_alias_and_stop_button(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = make_work_dir(tmp_path)
    use_env(monkeypatch, work, "real")
    app = run_app()

    click(app, "nav_llm")

    assert any(
        message.value == "手元AIは止まっています。" for message in app.info
    )
    assert any(
        message.value == "手元AIは使う時だけ自動で起動し、10分使わなければ止まります。"
        for message in app.info
    )
    stop_button = app.button(key="stop_local_llm")
    assert stop_button.label == "手元AIを今すぐ止める"
    alias_select = app.selectbox(key="mac_llm_alias_select")
    assert alias_select.value == "fake-model"
    assert set(alias_select.options) == {"fake-small", "fake-model"}
    assert app.session_state.mac_llm_alias == "fake-model"
