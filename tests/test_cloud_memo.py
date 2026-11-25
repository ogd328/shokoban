"""クラウドの読む役で5欄を作る（``shokoban.cloud_memo``）の試験.

本物の ``claude``・ネット・本番データは使わない。読む役はスパイか偽物。文はすべて架空。
"""

from __future__ import annotations

import json
import os
import stat
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from shokoban.assemble_r3 import F_PATH, SHARED_FIELDS
from shokoban.cloud_memo import (
    CloudMemoError,
    cloud_reader_from_env,
    fake_cloud_reader,
    load_memo,
    make_memo,
    memo_path,
    save_edited,
    save_memo,
    split_memo_sections,
)
from shokoban.mac_app import MacAppPaths
from shokoban.name_dict import empty_dict
from shokoban.pick_bench import FIELDS
from shokoban.reader_stage5 import ReaderReply
from shokoban.send_review import (
    SendReviewError,
    approval_path,
    final_masked,
    prepare_send_review,
    save_approval,
)

SENTENCES = [
    "こんにちは、今日はよろしくお願いします。",
    "チラシは来週までにお送りします。",
    "ばあちゃんのショートステイの迎えがあって、火曜は動けません。",
    "水曜の午後でお願いします。",
]
NOW = datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc)


def paths_in(root: Path) -> MacAppPaths:
    return MacAppPaths(
        work_dir=root,
        store_root=root / "store",
        inbox=root / "inbox",
        outbox=root / "outbox",
        held=root / "held",
        runtime_config=root / "runtime.json",
    )


def make_review():
    return prepare_send_review(SENTENCES, name_dict=empty_dict(), candidates=[])


class Spy:
    """呼ばれた回数と受け取った指示文を覚える読む役."""

    def __init__(self, text: str = '{"speakers":[],"items":[]}', model: str = "spy-model") -> None:
        self.text = text
        self.model = model
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> ReaderReply:
        self.prompts.append(prompt)
        return ReaderReply(text=self.text, model=self.model)


@pytest.fixture
def approved(tmp_path: Path):
    """確認済み（全文を読んだ）の記録つきの、(review, 人が伏せた番号, 記録のパス)."""
    review = make_review()
    path = approval_path(paths_in(tmp_path), "synthetic", "c001", "m001")
    save_approval(path, review, [3], reviewed_at=NOW, read_all=True)
    return review, [3], path


def test_make_memo_refuses_without_approval_and_never_calls_reader(tmp_path: Path) -> None:
    review = make_review()
    spy = Spy()
    with pytest.raises(CloudMemoError):
        make_memo(final_masked(review, [3]), approval_file=tmp_path / "none.json", reader=spy)
    assert spy.prompts == []


def test_make_memo_refuses_when_payload_changed_after_approval(approved) -> None:
    review, hide, path = approved
    spy = Spy()
    with pytest.raises(CloudMemoError):
        make_memo(final_masked(review, []), approval_file=path, reader=spy)  # 伏せ方を変えた
    with pytest.raises(CloudMemoError):
        make_memo(final_masked(review, [2, 3]), approval_file=path, reader=spy)
    assert spy.prompts == []


def test_make_memo_refuses_an_old_approval_without_payload_fingerprint(tmp_path: Path) -> None:
    review = make_review()
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"version": 1, "sha256": "x", "reviewed_at": "2026-11-22"}), encoding="utf-8")
    spy = Spy()
    with pytest.raises(CloudMemoError):
        make_memo(final_masked(review, []), approval_file=path, reader=spy)
    assert spy.prompts == []


def test_make_memo_returns_five_columns_and_sends_only_the_approved_payload(approved) -> None:
    review, hide, path = approved
    spy = Spy()
    masked = final_masked(review, hide)
    memo = make_memo(masked, approval_file=path, reader=fake_and_spy(spy), meeting_date=date(2026, 11, 9), created_at=NOW)
    assert len(spy.prompts) == 1
    assert SENTENCES[2] not in spy.prompts[0]  # 人が伏せた文の元の文は、送った指示文のどこにも無い
    assert memo["reader_model"] == "fake"
    assert memo["created_at"] == NOW.isoformat()
    assert memo["reason"] is None
    assert memo["payload_sha256"] == json.loads(path.read_text(encoding="utf-8"))["payload_sha256"]
    assert tuple(memo["columns"]) == SHARED_FIELDS  # 道筋の欄が先頭、続けて5欄
    assert memo["shared"].splitlines()[0] == f"【{F_PATH}】"
    for field in SHARED_FIELDS:
        assert f"【{field}】" in memo["shared"]
    assert set(memo["field_counts"]) == set(SHARED_FIELDS)
    # 偽の読む役は道筋を1件返す（こちらが話したことにしてある）
    assert memo["field_counts"][F_PATH] == 1
    assert memo["columns"][F_PATH][0]["speaker"] == "支援側" and memo["columns"][F_PATH][0]["step"] == 1
    assert "[こちら] ステップ1: " in memo["shared"]
    # 5欄は道筋を含まない（欄の名前の並びは FIELDS のまま）
    assert FIELDS == tuple(SHARED_FIELDS[1:])


def fake_and_spy(spy: Spy):
    """偽の読む役の返事を返しつつ、指示文を覚える."""

    def reader(prompt: str) -> ReaderReply:
        spy.prompts.append(prompt)
        return fake_cloud_reader(prompt)

    return reader


def test_make_memo_shape_error_gives_empty_columns_with_reason(approved) -> None:
    review, hide, path = approved
    memo = make_memo(final_masked(review, hide), approval_file=path, reader=Spy(text="これは JSON ではありません"), created_at=NOW)
    assert memo["reason"] == "形崩れ"
    assert memo["shared"].count("なし") == 6  # 道筋の欄＋5欄
    assert memo["counts"] == {"形崩れ": 1}
    assert memo["field_counts"] == {field: 0 for field in SHARED_FIELDS}


def test_fake_reader_returns_one_path_item_said_by_our_side() -> None:
    from shokoban.reader_r3 import build_reader_prompt_r3

    masked = final_masked(make_review(), [3])
    reply = json.loads(fake_cloud_reader(build_reader_prompt_r3(masked)).text)
    kinds = [item["kind"] for item in reply["items"]]
    assert kinds == ["未決", "困りごと", "道筋"]
    path_item = reply["items"][-1]
    assert path_item["step"] == 1 and path_item["final"] is False
    assert reply["speakers"] == [{"from": path_item["sent"], "to": path_item["sent"], "who": "支援側"}]


def test_split_memo_sections_counts_items_and_ignores_none() -> None:
    text = "\n".join(
        [
            f"【{F_PATH}】",
            "- [こちら] ステップ1: 入力をそろえる",
            "- [こちら] 最後: 振り返りを楽にする",
            "【困りごと】",
            "なし",
            "【まだ決まっていないこと】",
            "- [不明] ひとつめ",
            "・[お客様] ふたつめ",
            "",
        ]
    )
    preface, sections = split_memo_sections(text)
    assert preface == []
    assert [(name, len(lines)) for name, lines in sections] == [(F_PATH, 2), ("困りごと", 0), ("まだ決まっていないこと", 2)]
    assert sections[0][1][0] == "[こちら] ステップ1: 入力をそろえる"
    assert sections[2][1] == ["[不明] ひとつめ", "[お客様] ふたつめ"]


def test_split_memo_sections_reads_old_five_column_memos_and_free_text() -> None:
    old = "\n".join(f"【{field}】\n- 行" for field in FIELDS)
    preface, sections = split_memo_sections(old)
    assert preface == [] and [name for name, _ in sections] == list(FIELDS)
    preface, sections = split_memo_sections("見出しの無い直し\n二行目")
    assert preface == ["見出しの無い直し", "二行目"] and sections == []
    assert split_memo_sections("") == ([], [])


def test_make_memo_reader_exception_keeps_only_type_and_message(approved) -> None:
    review, hide, path = approved

    def broken(prompt: str) -> Any:
        raise TimeoutError("時間切れ")

    with pytest.raises(CloudMemoError) as info:
        make_memo(final_masked(review, hide), approval_file=path, reader=broken)
    assert "TimeoutError" in str(info.value) and "時間切れ" in str(info.value)
    for sentence in SENTENCES:
        assert sentence not in str(info.value)


def test_memo_path_and_permissions_and_edited_keeps_other_fields(approved, tmp_path: Path) -> None:
    review, hide, approval = approved
    paths = paths_in(tmp_path)
    path = memo_path(paths, "synthetic", "c001", "m001")
    assert str(path).endswith("store/cloud-memos/c001/m001.json")
    assert path.parents[2] == approval.parents[2]
    memo = make_memo(final_masked(review, hide), approval_file=approval, reader=Spy(), created_at=NOW)
    save_memo(path, memo)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(path.parent.parent).st_mode) == 0o700
    assert load_memo(path) == memo
    save_edited(path, "直した版")
    save_edited(path, "もう一度直した版")
    after = load_memo(path)
    assert after["edited"] == "もう一度直した版"
    assert {k: v for k, v in after.items() if k != "edited"} == memo
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_save_edited_without_memo_and_load_missing(tmp_path: Path) -> None:
    assert load_memo(tmp_path / "x.json") is None
    with pytest.raises(CloudMemoError):
        save_edited(tmp_path / "x.json", "a")


def test_memo_path_rejects_real_mode_and_bad_ids(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)
    with pytest.raises(SendReviewError):
        memo_path(paths, "real", "c001", "m001")
    for bad in ("../x", ".hidden", ""):
        with pytest.raises(SendReviewError):
            memo_path(paths, "synthetic", "c001", bad)
    with pytest.raises(SendReviewError):
        memo_path(paths, "synthetic", "c/01", "m001")


def test_reader_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHOKOBAN_CLOUD_READER", raising=False)
    assert cloud_reader_from_env() == (None, "止めてあります")
    monkeypatch.setenv("SHOKOBAN_CLOUD_READER", "  ")
    assert cloud_reader_from_env()[0] is None

    monkeypatch.setenv("SHOKOBAN_CLOUD_READER", "fake")
    reader, name = cloud_reader_from_env()
    assert callable(reader) and name == "fake"
    reply = reader("")
    assert reply.model == "fake" and json.loads(reply.text) == {"speakers": [], "items": []}

    monkeypatch.setenv("SHOKOBAN_CLOUD_READER", "gpt")
    with pytest.raises(CloudMemoError):
        cloud_reader_from_env()


def test_reader_from_env_claude_is_only_constructed(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("本物を起動してはいけません")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.delenv("SHOKOBAN_CLAUDE_MODEL", raising=False)
    for value in ("claude", "opus"):  # opus は claude の古い呼び名
        monkeypatch.setenv("SHOKOBAN_CLOUD_READER", value)
        reader, name = cloud_reader_from_env()
        assert callable(reader) and name == "claude (opus)"
    monkeypatch.setenv("SHOKOBAN_CLAUDE_MODEL", "sonnet")
    reader, name = cloud_reader_from_env()
    assert callable(reader) and name == "claude (sonnet)"


def test_reader_from_env_claude_model_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_make(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return lambda prompt: ReaderReply(text="{}", model=kwargs["model"])

    monkeypatch.setattr("shokoban.cloud_memo.make_claude_reader", fake_make)
    monkeypatch.setenv("SHOKOBAN_CLOUD_READER", "claude")
    monkeypatch.setenv("SHOKOBAN_CLAUDE_MODEL", "haiku")
    cloud_reader_from_env()
    assert seen["model"] == "haiku"


def test_reader_from_env_local(monkeypatch: pytest.MonkeyPatch) -> None:
    from shokoban.cloud_memo import reader_is_local

    monkeypatch.setenv("SHOKOBAN_CLOUD_READER", "local")
    monkeypatch.delenv("SHOKOBAN_READER_URL", raising=False)
    monkeypatch.delenv("SHOKOBAN_READER_MODEL", raising=False)
    with pytest.raises(CloudMemoError):  # モデル名が無ければ断る
        cloud_reader_from_env()

    calls: list[Any] = []

    class Response:
        status = 200

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode("utf-8")

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    def opener(request: Any, timeout: Any = None) -> Any:
        calls.append(request.full_url)
        return Response()

    monkeypatch.setattr("shokoban.reader_local._default_opener", opener)
    monkeypatch.setenv("SHOKOBAN_READER_MODEL", "my-model")
    reader, name = cloud_reader_from_env()
    assert name == "local (my-model)" and reader_is_local(name)
    assert not reader_is_local("claude (opus)") and not reader_is_local("fake")
    assert calls == []  # 作るだけでは送らない
    assert reader("指示").text == "{}"
    assert calls == ["http://127.0.0.1:1234/v1/chat/completions"]

    monkeypatch.setenv("SHOKOBAN_READER_URL", "http://localhost:11434/v1")
    cloud_reader_from_env()[0]("指示")
    assert calls[-1] == "http://localhost:11434/v1/chat/completions"


def test_reader_from_env_local_refuses_external_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOKOBAN_CLOUD_READER", "local")
    monkeypatch.setenv("SHOKOBAN_READER_MODEL", "my-model")
    for url in ("https://api.example.com/v1", "http://10.0.0.5:1234/v1"):
        monkeypatch.setenv("SHOKOBAN_READER_URL", url)
        with pytest.raises(CloudMemoError):
            cloud_reader_from_env()


def test_reserve_reader_call_limit_message_is_generic(tmp_path: Path) -> None:
    from shokoban.cloud_memo import reserve_reader_call

    path = tmp_path / "usage.json"
    reserve_reader_call(path, "2026-11", 1)
    with pytest.raises(CloudMemoError) as caught:
        reserve_reader_call(path, "2026-11", 1)
    assert "SHOKOBAN_READER_MONTHLY_LIMIT" in str(caught.value)


# --- 共有フォルダに渡す（share/進行メモ/）----------------------------------------------

def test_share_memo_writes_private_file_with_edited_text_and_readme(tmp_path: Path) -> None:
    from datetime import date, datetime, timezone

    from shokoban.cloud_memo import SHARE_MEMO_DIR, share_memo, share_memo_path

    share = tmp_path / "share"
    memo = {"version": 1, "shared": "【困りごと】\n- [お客様] 元の版", "edited": "【困りごと】\n- [お客様] 直した版",
            "reader_model": "fake", "created_at": "2026-11-24T09:00:00+00:00"}
    when = datetime(2026, 11, 24, 19, 0, tzinfo=timezone.utc)
    path = share_memo(share, "架空A", memo, record_id="mtg_0123456789abcdef", meeting_date=date(2026, 11, 9), shared_at=when)
    assert path == share / SHARE_MEMO_DIR / "架空A" / "2026-11-09_89abcdef.md"
    assert path == share_memo_path(share, "架空A", "mtg_0123456789abcdef", date(2026, 11, 9))
    text = path.read_text(encoding="utf-8")
    assert "直した版" in text and "元の版" not in text and "人が直した版: はい" in text
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    for directory in (share, share / SHARE_MEMO_DIR, path.parent):
        assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
    readme = share / SHARE_MEMO_DIR / "README.md"
    assert "札" in readme.read_text(encoding="utf-8") and stat.S_IMODE(os.stat(readme).st_mode) == 0o600

    # 直した版が無ければ元の版。渡し直すと上書き
    memo.pop("edited")
    share_memo(share, "架空A", memo, record_id="mtg_0123456789abcdef", meeting_date=date(2026, 11, 9), shared_at=when)
    text = path.read_text(encoding="utf-8")
    assert "元の版" in text and "人が直した版: いいえ" in text


@pytest.mark.parametrize("label, record_id", [("../外", "mtg_0123456789abcdef"), ("架空A", "../../store"), ("", "mtg_0123456789abcdef")])
def test_share_memo_refuses_bad_label_or_record_id(tmp_path: Path, label: str, record_id: str) -> None:
    from datetime import datetime, timezone

    from shokoban.cloud_memo import share_memo

    with pytest.raises(CloudMemoError):
        share_memo(tmp_path / "share", label, {"version": 1, "shared": "x"}, record_id=record_id,
                   meeting_date=None, shared_at=datetime.now(timezone.utc))
    assert not (tmp_path / "share").exists()


# --- 読む役の月の回数（普段使いの上限 200）---------------------------------------------

def test_reserve_reader_call_counts_per_month_and_stops_at_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from shokoban.cloud_memo import DEFAULT_MONTHLY_LIMIT, load_usage, monthly_limit, reserve_reader_call

    monkeypatch.delenv("SHOKOBAN_READER_MONTHLY_LIMIT", raising=False)
    assert monthly_limit() == DEFAULT_MONTHLY_LIMIT == 200
    for bad in ("0", "-3", "abc", ""):
        monkeypatch.setenv("SHOKOBAN_READER_MONTHLY_LIMIT", bad)
        assert monthly_limit() == 200
    path = tmp_path / "cloud-memos" / "reader-usage.json"
    assert reserve_reader_call(path, "2026-11", 2) == 1
    assert reserve_reader_call(path, "2026-11", 2) == 2
    with pytest.raises(CloudMemoError, match="上限"):
        reserve_reader_call(path, "2026-11", 2)
    assert reserve_reader_call(path, "2026-12", 2) == 1  # 月が変われば数え直し
    assert load_usage(path) == {"2026-11": 2, "2026-12": 1}
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    path.write_text("壊れた", encoding="utf-8")
    with pytest.raises(CloudMemoError):
        reserve_reader_call(path, "2026-11", 2)
