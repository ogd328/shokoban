"""名前の確認（``shokoban.name_review`` と画面の「名前の確認」）の試験.

実プロセス（llama-server・GiNZA の別環境）・ネット・実際の顧客データは使わない。12B は偽の
候補出し役、GiNZA は偽の項目か偽の subprocess 実行役にする。文はすべて架空。
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from shokoban.customer_store import CLEAN_ROOT_CONFIRMATION, CustomerStore
from shokoban.llm_runtime import LLMRuntimeError
from shokoban.local_llm import LocalLLMSettings
from shokoban.mac_app import MacAppPaths, mac_app_paths_from_env
from shokoban.name_candidates import Candidate, terms_to_mask
from shokoban.name_dict import KIND_ORG, KIND_PERSON, NameDict, load_name_dict, save_name_dict
from shokoban.name_review import (
    DECISION_NOT_NAME,
    NameReviewError,
    ReviewResult,
    ReviewRow,
    apply_review,
    build_review,
    choice_to_decision,
    collect_decisions,
    count_choices,
    ginza_paths,
    highlight_html,
    masked_preview,
    meeting_sentences,
    name_dict_path,
    pick_name_model,
    prepare_review,
    remaining_rows,
    run_ginza,
    save_decisions,
)

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover - GUI dependencies are optional.
    AppTest = None

APP_PATH = Path(__file__).resolve().parents[1] / "gui" / "shokoban_mac.py"
CUSTOMER_ID = "cus_" + "0" * 32

SENTENCES = [
    "今日は山本さんと打ち合わせをしました。",
    "青空商店の仕入れの件です。",
    "商工会議所に相談しました。",
    "中村ゆかりと言います。",
    "すみれ通りの店です。",
]
GINZA_ITEMS = [
    {"word": "中村", "kind": KIND_PERSON, "label": "名詞-固有名詞-人名-姓", "sentence": 4},
    {"word": "青空商店", "kind": KIND_ORG, "label": "Company", "sentence": 2},
]


def fake_extractor(text: str) -> list[dict[str, str]]:
    return [
        {"word": "青空商店", "kind": KIND_ORG, "judgment": "確か"},
        {"word": "すみれ通り", "kind": KIND_ORG, "judgment": "迷う"},
        {"word": "商工会議所", "kind": "それ以外", "judgment": "確か"},
    ]


def make_paths(tmp_path: Path) -> MacAppPaths:
    work = tmp_path / "work"
    work.mkdir()
    paths = mac_app_paths_from_env({"SHOKOBAN_WORK_DIR": str(work)})
    assert paths is not None
    return paths


def rows_by_term(result: ReviewResult) -> dict[str, ReviewRow]:
    return {row.term: row for row in result.rows}


# --- 辞書の場所 ------------------------------------------------------------


def test_dict_path_is_per_customer_inside_the_store_root(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    path = name_dict_path(paths, "synthetic", CUSTOMER_ID)
    assert path == paths.work_dir / "store-synthetic" / "name-dicts" / f"{CUSTOMER_ID}.json"
    # 呼んだだけでは何も作らない
    assert not (paths.work_dir / "store-synthetic").exists()


@pytest.mark.parametrize("bad", ["", "../x", "a/b", "cus 1", "x" * 81, None, 5])
def test_dict_path_rejects_bad_customer_id(tmp_path: Path, bad: Any) -> None:
    with pytest.raises(NameReviewError):
        name_dict_path(make_paths(tmp_path), "synthetic", bad)


def test_real_mode_is_refused_everywhere_without_touching_the_disk(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    calls: list[Any] = []
    with pytest.raises(NameReviewError, match="本番データ"):
        name_dict_path(paths, "real", CUSTOMER_ID)
    with pytest.raises(NameReviewError, match="本番データ"):
        prepare_review(
            SENTENCES,
            dict_path=tmp_path / "d.json",
            mode="real",
            extractor=lambda t: calls.append(t) or [],
            ginza_runner=lambda s: calls.append(s) or [],
        )
    with pytest.raises(NameReviewError, match="本番データ"):
        build_review(paths, "real", CUSTOMER_ID, SENTENCES, alias=None)
    with pytest.raises(NameReviewError):
        name_dict_path(paths, "other", CUSTOMER_ID)
    assert calls == []
    assert not (paths.work_dir / "store").exists()


def test_ginza_paths_use_the_work_venv_and_the_repo_script(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    python_path, script_path = ginza_paths(paths)
    assert python_path == paths.work_dir / "venv-ginza" / "bin" / "python"
    assert script_path.name == "ginza_names.py" and script_path.is_file()


# --- apply_review ----------------------------------------------------------


def test_apply_review_writes_mask_and_pass_with_0600(tmp_path: Path) -> None:
    target = tmp_path / "store" / "name-dicts" / "d.json"
    (tmp_path / "store").mkdir()
    applied = apply_review(
        target,
        {"山本": KIND_PERSON, "青空商店": KIND_ORG, "商工会議所": DECISION_NOT_NAME},
    )
    assert (applied.masked, applied.passed, applied.total) == (2, 1, 3)
    loaded = load_name_dict(target)
    assert dict(loaded.mask_terms()) == {"山本": KIND_PERSON, "青空商店": KIND_ORG}
    assert loaded.pass_terms == ("商工会議所",)
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(target.parent).st_mode) == 0o700
    assert [p.name for p in target.parent.iterdir()] == ["d.json"]  # 一時ファイルが残らない


def test_apply_review_keeps_existing_terms_and_can_flip_a_decision(tmp_path: Path) -> None:
    target = tmp_path / "d.json"
    nd = NameDict()
    nd.add_mask("山本", KIND_PERSON)
    save_name_dict(target, nd)
    apply_review(target, {"青空商店": KIND_ORG})
    apply_review(target, {"山本": DECISION_NOT_NAME})
    loaded = load_name_dict(target)
    assert dict(loaded.mask_terms()) == {"青空商店": KIND_ORG}
    assert loaded.pass_terms == ("山本",)


def test_apply_review_with_nothing_decided_writes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "d.json"
    assert apply_review(target, {}).total == 0
    assert not target.exists()


@pytest.mark.parametrize(
    "decisions",
    [
        {"山本": "あとで"},
        {"山本": "それ以外"},
        {"山本": None},
        {"": KIND_PERSON},
        {"a\nb": KIND_PERSON},
        {"x" * 21: KIND_PERSON},
        {5: KIND_PERSON},
        [("山本", KIND_PERSON)],
    ],
)
def test_apply_review_rejects_bad_input_and_writes_nothing(tmp_path: Path, decisions: Any) -> None:
    target = tmp_path / "d.json"
    with pytest.raises(NameReviewError):
        apply_review(target, decisions)
    assert not target.exists()


def test_apply_review_is_all_or_nothing(tmp_path: Path) -> None:
    target = tmp_path / "d.json"
    nd = NameDict()
    nd.add_mask("山本", KIND_PERSON)
    save_name_dict(target, nd)
    before = target.read_bytes()
    with pytest.raises(NameReviewError):
        apply_review(target, {"青空商店": KIND_ORG, "中村": "知らない種類"})
    assert target.read_bytes() == before


def test_apply_review_rejects_terms_outside_the_allowed_candidates(tmp_path: Path) -> None:
    target = tmp_path / "d.json"
    with pytest.raises(NameReviewError, match="候補に無い"):
        apply_review(target, {"山本": KIND_PERSON, "勝手な語": KIND_ORG}, allowed_terms=["山本"])
    assert not target.exists()


def test_apply_review_fails_on_broken_dictionary_without_overwriting(tmp_path: Path) -> None:
    target = tmp_path / "d.json"
    target.write_text("{壊れた", encoding="utf-8")
    with pytest.raises(NameReviewError):
        apply_review(target, {"山本": KIND_PERSON})
    assert target.read_text(encoding="utf-8") == "{壊れた"


def test_apply_review_refuses_a_symlinked_dictionary(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    save_name_dict(real, NameDict())
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(NameReviewError, match="シンボリックリンク"):
        apply_review(link, {"山本": KIND_PERSON})


def test_apply_review_replaces_atomically(tmp_path: Path) -> None:
    target = tmp_path / "d.json"
    save_name_dict(target, NameDict())
    with mock.patch("shokoban.name_dict.os.replace", side_effect=OSError("boom")):
        with pytest.raises(NameReviewError):
            apply_review(target, {"山本": KIND_PERSON})
    assert load_name_dict(target).mask_terms() == ()
    assert [p.name for p in tmp_path.iterdir()] == ["d.json"]


def test_choice_to_decision() -> None:
    assert choice_to_decision("名前") == KIND_PERSON
    assert choice_to_decision("社名・店名・団体") == KIND_ORG
    assert choice_to_decision("名前じゃない") == DECISION_NOT_NAME
    assert choice_to_decision("あとで") is None
    with pytest.raises(NameReviewError):
        choice_to_decision("人名")


# --- prepare_review --------------------------------------------------------


def test_prepare_review_sorts_sure_first_and_sets_defaults(tmp_path: Path) -> None:
    result = prepare_review(
        SENTENCES,
        dict_path=tmp_path / "none.json",
        mode="synthetic",
        extractor=fake_extractor,
        ginza_runner=lambda s: GINZA_ITEMS,
    )
    judgments = [row.judgment for row in result.rows]
    assert judgments == sorted(judgments, key=lambda j: 0 if j == "確か" else 1)
    assert result.rows[-1].term == "すみれ通り" and result.rows[-1].judgment == "迷う"
    rows = rows_by_term(result)
    assert set(rows) == {"山本", "青空商店", "商工会議所", "中村", "すみれ通り"}
    assert rows["青空商店"].sources == ("ginza", "12b")
    assert rows["山本"].default_choice == "名前"
    assert rows["青空商店"].default_choice == "社名・店名・団体"
    assert rows["すみれ通り"].default_choice == "あとで"  # 迷うは決めない
    assert rows["商工会議所"].default_choice == "あとで"  # それ以外の見立ては決めない
    assert rows["青空商店"].examples == ("青空商店の仕入れの件です。",)
    assert result.ginza_error is None and result.ginza_used
    # 候補はすべて「伏せたまま」の元になる
    assert {c.term for c in result.candidates} == set(rows)


def test_prepare_review_gives_at_most_two_examples(tmp_path: Path) -> None:
    sentences = [f"{i}回目は山本さんが来ました。" for i in range(5)]
    result = prepare_review(sentences, dict_path=tmp_path / "n.json", mode="synthetic")
    (row,) = [r for r in result.rows if r.term == "山本"]
    assert row.examples == tuple(sentences[:2])


def test_prepare_review_leaves_out_terms_already_in_the_dictionary(tmp_path: Path) -> None:
    target = tmp_path / "d.json"
    nd = NameDict()
    nd.add_mask("山本", KIND_PERSON)
    nd.add_pass("商工会議所")
    save_name_dict(target, nd)
    result = prepare_review(
        SENTENCES,
        dict_path=target,
        mode="synthetic",
        extractor=fake_extractor,
        ginza_runner=lambda s: GINZA_ITEMS,
    )
    assert "山本" not in rows_by_term(result)
    assert "商工会議所" not in rows_by_term(result)
    assert "青空商店" in rows_by_term(result)


def test_prepare_review_without_extractor_or_ginza_uses_the_rule_net_only(tmp_path: Path) -> None:
    result = prepare_review(SENTENCES, dict_path=tmp_path / "n.json", mode="synthetic")
    assert "山本" in rows_by_term(result)
    assert "青空商店" not in rows_by_term(result)
    assert not result.ginza_used and result.ginza_error is None


def test_ginza_failure_is_reported_and_other_sources_still_give_candidates(tmp_path: Path) -> None:
    def broken(sentences: Any) -> Any:
        raise NameReviewError("GiNZA が途中で止まりました。")

    result = prepare_review(
        SENTENCES,
        dict_path=tmp_path / "n.json",
        mode="synthetic",
        extractor=fake_extractor,
        ginza_runner=broken,
    )
    assert result.ginza_error == "GiNZA が途中で止まりました。"
    assert not result.ginza_used
    rows = rows_by_term(result)
    assert "青空商店" in rows and "山本" in rows  # 12B と網の候補は残る
    assert "中村" not in rows  # GiNZA だけが見つけるはずだった語
    # 見つけた候補は、伏せる語として使われる（伏せたまま）
    preview = masked_preview(SENTENCES, None, result.candidates)
    assert "山本" not in preview[0] and "青空商店" not in preview[1]


def test_ginza_unexpected_exception_does_not_leak_its_message(tmp_path: Path) -> None:
    def broken(sentences: Any) -> Any:
        raise RuntimeError("中村さんの文が入ったエラー")

    result = prepare_review(
        SENTENCES, dict_path=tmp_path / "n.json", mode="synthetic", ginza_runner=broken
    )
    assert result.ginza_error == "GiNZA を使えませんでした。"
    assert "中村" not in result.ginza_error


def test_prepare_review_reports_failed_12b_chunks(tmp_path: Path) -> None:
    def broken(text: str) -> Any:
        raise RuntimeError("down")

    result = prepare_review(
        SENTENCES, dict_path=tmp_path / "n.json", mode="synthetic", extractor=broken
    )
    assert result.llm_failed_chunks == 1
    assert "山本" in rows_by_term(result)


def test_prepare_review_fails_on_a_broken_dictionary(tmp_path: Path) -> None:
    target = tmp_path / "d.json"
    target.write_text("[]", encoding="utf-8")
    with pytest.raises(NameReviewError):
        prepare_review(SENTENCES, dict_path=target, mode="synthetic")


def test_prepare_review_rejects_a_string_instead_of_sentences(tmp_path: Path) -> None:
    with pytest.raises(NameReviewError):
        prepare_review("山本です。", dict_path=tmp_path / "n.json", mode="synthetic")  # type: ignore[arg-type]


# --- 決めたことの集め方・保存・残り ---------------------------------------------


def make_result(tmp_path: Path) -> ReviewResult:
    return prepare_review(
        SENTENCES,
        dict_path=tmp_path / "d.json",
        mode="synthetic",
        extractor=fake_extractor,
        ginza_runner=lambda s: GINZA_ITEMS,
    )


def test_collect_decisions_skips_later_and_can_limit_to_sure(tmp_path: Path) -> None:
    result = make_result(tmp_path)
    choices = {row.term: row.default_choice for row in result.rows}
    choices["すみれ通り"] = "社名・店名・団体"  # 迷うを自分で決めた
    choices["商工会議所"] = "名前じゃない"
    every = collect_decisions(result, choices, sure_only=False)
    assert every == {
        "山本": KIND_PERSON,
        "青空商店": KIND_ORG,
        "中村": KIND_PERSON,
        "すみれ通り": KIND_ORG,
        "商工会議所": DECISION_NOT_NAME,
    }
    sure = collect_decisions(result, choices, sure_only=True)
    assert "すみれ通り" not in sure and sure["山本"] == KIND_PERSON
    assert collect_decisions(result, {"山本": "あとで"}, sure_only=False) == {}


def test_count_choices() -> None:
    assert count_choices({"a": "名前", "b": "あとで", "c": "名前じゃない"}) == (3, 2, 1)
    assert count_choices({}) == (0, 0, 0)


def test_save_decisions_removes_saved_rows_and_keeps_the_rest(tmp_path: Path) -> None:
    result = make_result(tmp_path)
    target = tmp_path / "d.json"
    applied, rest = save_decisions(target, result, {"山本": KIND_PERSON, "青空商店": KIND_ORG})
    assert applied.masked == 2
    left = {row.term for row in rest.rows}
    assert "山本" not in left and "青空商店" not in left and "すみれ通り" in left
    assert {c.term for c in rest.candidates} == left
    # 次に候補を出しても、保存した語は出てこない。決めなかった語は、また候補（＝伏せたまま）
    again = prepare_review(
        SENTENCES, dict_path=target, mode="synthetic", extractor=fake_extractor,
        ginza_runner=lambda s: GINZA_ITEMS,
    )
    assert "山本" not in rows_by_term(again) and "すみれ通り" in rows_by_term(again)
    assert [r.term for r in remaining_rows(result.rows, target)] == [r.term for r in rest.rows]


def test_save_decisions_rejects_terms_that_are_not_candidates(tmp_path: Path) -> None:
    result = make_result(tmp_path)
    with pytest.raises(NameReviewError):
        save_decisions(tmp_path / "d.json", result, {"勝手な語": KIND_ORG})
    assert not (tmp_path / "d.json").exists()


# --- masked_preview ---------------------------------------------------------


def test_masked_preview_masks_dictionary_and_unconfirmed_but_not_pass_terms(tmp_path: Path) -> None:
    result = make_result(tmp_path)
    nd = NameDict()
    nd.add_mask("青空商店", KIND_ORG)
    nd.add_pass("商工会議所")
    candidates = [c for c in result.candidates if c.term != "青空商店"]
    preview = masked_preview(SENTENCES, nd, candidates)
    assert preview[0] == "今日は〈名前〉と打ち合わせをしました。"  # 未確認の候補も伏せる
    assert preview[1] == "〈社名〉の仕入れの件です。"  # 辞書の伏せる語
    assert preview[2] == "商工会議所に相談しました。"  # 伏せない語・それ以外は伏せない
    assert preview[4] == "〈社名〉の店です。"
    assert masked_preview(SENTENCES, None, []) is not None


def test_masked_preview_does_not_mask_pass_terms_even_if_candidate() -> None:
    nd = NameDict()
    nd.add_pass("すみれ通り")
    candidate = Candidate("すみれ通り", KIND_ORG, "迷う", ("12b",), (5,))
    assert masked_preview(SENTENCES, nd, [candidate])[4] == SENTENCES[4]
    assert masked_preview(SENTENCES, None, [candidate])[4] == "〈社名〉の店です。"


def test_masked_preview_rejects_a_string() -> None:
    with pytest.raises(NameReviewError):
        masked_preview("山本", None, [])  # type: ignore[arg-type]


# --- run_ginza（偽の subprocess）--------------------------------------------


def make_fake_ginza(tmp_path: Path) -> tuple[Path, Path]:
    python = tmp_path / "venv-ginza" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    script = tmp_path / "ginza_names.py"
    script.write_text("# fake\n", encoding="utf-8")
    return python, script


def completed(stdout: Any = b"", returncode: int = 0) -> Any:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=b"")


def test_run_ginza_passes_sentences_on_stdin_and_reads_items(tmp_path: Path) -> None:
    python, script = make_fake_ginza(tmp_path)
    seen: dict[str, Any] = {}

    def runner(argv: list[str], **kwargs: Any) -> Any:
        seen["argv"], seen["kwargs"] = argv, kwargs
        return completed(json.dumps({"items": GINZA_ITEMS}, ensure_ascii=False).encode("utf-8"))

    items = run_ginza(SENTENCES, python_path=python, script_path=script, timeout=7, runner=runner)
    assert items == GINZA_ITEMS
    assert seen["argv"] == [str(python), "-I", str(script), "--stdin"]
    assert json.loads(seen["kwargs"]["input"].decode("utf-8")) == {"sentences": SENTENCES}
    assert seen["kwargs"]["timeout"] == 7
    assert seen["kwargs"]["shell"] is False
    env = seen["kwargs"]["env"]
    assert env["HF_HUB_OFFLINE"] == "1" and "HOME" not in env  # 手元の環境変数を渡さない


@pytest.mark.parametrize(
    "outcome",
    [
        completed(b"", returncode=2),
        completed(b"not json"),
        completed(b'{"items": "x"}'),
        completed(b"[]"),
        completed(b'{"items": [1]}'),
        completed(b"\xff\xfe"),
        completed(None),
        subprocess.TimeoutExpired("x", 1),
        OSError("exec failed"),
    ],
)
def test_run_ginza_fails_closed_without_leaking_sentences(tmp_path: Path, outcome: Any) -> None:
    python, script = make_fake_ginza(tmp_path)

    def runner(argv: list[str], **kwargs: Any) -> Any:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    with pytest.raises(NameReviewError) as info:
        run_ginza(SENTENCES, python_path=python, script_path=script, timeout=1, runner=runner)
    assert "山本" not in str(info.value)


def test_run_ginza_refuses_missing_environment_or_script(tmp_path: Path) -> None:
    python, script = make_fake_ginza(tmp_path)

    def runner(*args: Any, **kwargs: Any) -> Any:  # 呼ばれてはいけない
        raise AssertionError("must not run")

    with pytest.raises(NameReviewError, match="環境"):
        run_ginza(SENTENCES, python_path=tmp_path / "none", script_path=script, timeout=1, runner=runner)
    with pytest.raises(NameReviewError, match="道具"):
        run_ginza(SENTENCES, python_path=python, script_path=tmp_path / "none.py", timeout=1, runner=runner)
    python.chmod(0o644)
    with pytest.raises(NameReviewError, match="環境"):
        run_ginza(SENTENCES, python_path=python, script_path=script, timeout=1, runner=runner)


# --- ginza_names.py --stdin（spaCy なし）----------------------------------------


def test_ginza_script_stdin_helpers() -> None:
    from scripts.ginza_names import parse_stdin_sentences, run_stdin

    assert parse_stdin_sentences('{"sentences": ["a", "b"]}'.encode()) == ["a", "b"]
    for bad in (b"x", b"[]", b'{"sentences": "a"}', b'{"sentences": [1]}', b'{"sentences": [], "x": 1}', b"\xff"):
        with pytest.raises(ValueError):
            parse_stdin_sentences(bad)

    class FakeToken:
        def __init__(self, text: str, tag: str) -> None:
            self.text, self.tag_ = text, tag

    class FakeEnt:
        def __init__(self, text: str, label: str) -> None:
            self.text, self.label_ = text, label

    class FakeDoc(list):
        ents: list[Any] = []

    class FakeNlp:
        def pipe(self, texts: list[str], batch_size: int = 32) -> Any:
            for text in texts:
                doc = FakeDoc([FakeToken("山本", "名詞-固有名詞-人名-姓")] if "山本" in text else [])
                doc.ents = [FakeEnt("青空商店さん", "Company")] if "青空" in text else []
                yield doc

    out = run_stdin(
        json.dumps({"sentences": ["山本が来た。", "青空商店さんに行く。"]}, ensure_ascii=False).encode(),
        FakeNlp(),
    )
    items = json.loads(out.decode("utf-8"))["items"]
    assert {(i["word"], i["kind"], i["sentence"]) for i in items} == {
        ("山本", KIND_PERSON, 1),
        ("青空商店", KIND_ORG, 2),
    }
    assert json.loads(run_stdin(b'{"sentences": []}', FakeNlp()).decode()) == {"items": []}


# --- 12B の選び方・文 ----------------------------------------------------------


def test_pick_name_model_prefers_the_measured_12b() -> None:
    both = SimpleNamespace(models={"gemma-4-12b-ja": "x", "other": "y"})
    only_other = SimpleNamespace(models={"other": "y"})
    assert pick_name_model(both, "other") == "gemma-4-12b-ja"
    with pytest.raises(NameReviewError):
        pick_name_model(only_other, "other")


def test_pick_name_model_follows_env_and_names_the_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOKOBAN_TAGGER_MODEL", "my-tagger")
    cfg = SimpleNamespace(models={"my-tagger": "x", "gemma-4-12b-ja": "y"})
    assert pick_name_model(cfg, None) == "my-tagger"
    # 環境変数の名前が設定に無いときは、既定名があっても止まり、直し方を示す
    other = SimpleNamespace(models={"gemma-4-12b-ja": "y"})
    with pytest.raises(NameReviewError) as caught:
        pick_name_model(other, None)
    message = str(caught.value)
    assert "my-tagger" in message
    assert "SHOKOBAN_TAGGER_MODEL か runtime.json の models に、その名前を入れてください" in message
    # 空の指定は既定に戻る
    monkeypatch.setenv("SHOKOBAN_TAGGER_MODEL", "  ")
    assert pick_name_model(other, None) == "gemma-4-12b-ja"
    monkeypatch.delenv("SHOKOBAN_TAGGER_MODEL")
    assert pick_name_model(other, None) == "gemma-4-12b-ja"


def test_meeting_sentences_collects_all_text_once() -> None:
    meeting = {
        "title": "打合せ",
        "summary": "要約です。",
        "details": "山本さんが来ました。要約です。",
        "decisions": ["青空商店に連絡する"],
        "next_actions": ["中村さんへ電話"],
    }
    sentences = meeting_sentences(meeting)
    assert sentences.count("山本さんが来ました。") == 1
    assert "青空商店に連絡する" in sentences and "中村さんへ電話" in sentences
    assert meeting_sentences({}) == []


def test_highlight_html_escapes_everything_and_marks_the_term() -> None:
    out = highlight_html("<img src=x onerror=alert(1)>山本さん&来た", "山本")
    assert "<img" not in out and "&lt;img" in out and "&amp;" in out
    assert "<mark>山本</mark>" in out
    long = "あ" * 200 + "山本" + "い" * 200
    snippet = highlight_html(long, "山本")
    assert snippet.startswith("…") and snippet.endswith("…") and len(snippet) < 200
    assert highlight_html("山本", "存在しない") == "山本"


# --- build_review（12B と GiNZA は偽物）-----------------------------------------


def runtime_json(work: Path) -> None:
    model = work / "m.gguf"
    model.write_bytes(b"GGUF")
    (work / "runtime.json").write_text(
        json.dumps(
            {
                "port": 1234,
                "ctx": 8192,
                "idle_minutes": 10,
                "min_free_percent": 20,
                "default_model": "fake-model",
                "models": {"fake-model": str(model)},
                "state_dir": str(work / "runtime-state"),
            }
        ),
        encoding="utf-8",
    )


def test_build_review_combines_sources_with_fakes(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    runtime_json(paths.work_dir)
    with mock.patch("shokoban.name_review.local_name_extractor", return_value=fake_extractor), mock.patch(
        "shokoban.name_review.run_ginza", return_value=GINZA_ITEMS
    ) as ginza:
        result = build_review(paths, "synthetic", CUSTOMER_ID, SENTENCES, alias="fake-model")
    assert {"山本", "青空商店", "中村", "すみれ通り"} <= set(rows_by_term(result))
    assert ginza.call_args.kwargs["python_path"] == paths.work_dir / "venv-ginza" / "bin" / "python"
    assert result.llm_error is None and result.ginza_error is None


def test_build_review_continues_when_12b_cannot_start(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    runtime_json(paths.work_dir)
    with mock.patch(
        "shokoban.name_review.local_name_extractor", side_effect=LLMRuntimeError("起動に失敗しました")
    ), mock.patch("shokoban.name_review.run_ginza", return_value=GINZA_ITEMS):
        result = build_review(paths, "synthetic", CUSTOMER_ID, SENTENCES, alias="fake-model")
    assert result.llm_error == "起動に失敗しました"
    rows = rows_by_term(result)
    assert "中村" in rows and "山本" in rows and "すみれ通り" not in rows


def test_build_review_reports_missing_runtime_config_and_ginza(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)  # runtime.json も venv-ginza も無い
    result = build_review(paths, "synthetic", CUSTOMER_ID, SENTENCES, alias=None)
    assert result.llm_error and result.ginza_error == "GiNZA の環境が見つかりません。"
    assert "山本" in rows_by_term(result)  # 網の候補は出る


# --- 画面（Streamlit AppTest）-----------------------------------------------------

gui = pytest.mark.skipif(AppTest is None, reason="install the gui extra to run Streamlit tests")
FAKE_SETTINGS = LocalLLMSettings(provider="lmstudio", model="fake-model")


def make_gui_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> tuple[Path, str]:
    work = tmp_path / "work"
    work.mkdir()
    for name in ("inbox", "outbox", "outbox-held"):
        (work / name).mkdir()
    runtime_json(work)
    monkeypatch.setenv("SHOKOBAN_WORK_DIR", str(work))
    monkeypatch.setenv("SHOKOBAN_DATA_MODE", mode)
    store = CustomerStore(work / ("store-synthetic" if mode == "synthetic" else "store"))
    store.initialize(confirmation=CLEAN_ROOT_CONFIRMATION)
    customer = store.create_customer(display_name="架空顧客A")
    store.create_meeting_draft(
        customer["customer_id"],
        occurred_at="2026-11-09T00:00:00+09:00",
        title="架空の打合せ",
        summary="",
        details="".join(SENTENCES),
    )
    return work, customer["customer_id"]


def open_names(app: Any) -> Any:
    app.button(key="nav_names").click().run()
    assert list(app.exception) == []
    return app


@gui
def test_names_screen_is_not_enabled_in_real_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work, _ = make_gui_work(tmp_path, monkeypatch, "real")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with mock.patch("shokoban.name_review.build_review") as build, mock.patch(
        "shokoban.mac_app.list_draft_meetings"
    ) as listing:
        open_names(app)
    assert any("本番データではまだ使えません" in m.value for m in app.info)
    assert not [b for b in app.button if b.key == "name_review_run"]
    build.assert_not_called()
    listing.assert_not_called()
    assert not (work / "store" / "name-dicts").exists()


@gui
def test_names_screen_flow_in_synthetic_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work, customer_id = make_gui_work(tmp_path, monkeypatch, "synthetic")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert list(app.exception) == []
    with mock.patch("shokoban.name_review.local_name_extractor", return_value=fake_extractor), mock.patch(
        "shokoban.name_review.run_ginza", return_value=GINZA_ITEMS
    ):
        open_names(app)
        assert any("すべて伏せたまま" in m.value for m in app.info)
        app.button(key="name_review_run").click().run()
        assert list(app.exception) == []

        key = lambda term: f"name_choice_{customer_id}_{term}"  # noqa: E731
        # 確かは見立てが選ばれ、迷うは「あとで」
        assert app.radio(key=key("山本")).value == "名前"
        assert app.radio(key=key("青空商店")).value == "社名・店名・団体"
        assert app.radio(key=key("すみれ通り")).value == "あとで"
        text = " ".join(t.value for t in app.text)
        assert "候補5・確定3・あとで2" in text  # 山本・青空商店・中村が確定、すみれ通り・商工会議所があとで
        assert not app.warning
        # 例文は HTML で、語が mark で囲まれる
        assert any("<mark>山本</mark>" in str(el.value) for el in app.get("html"))

        # 迷うを自分で決める → 数が変わる
        app.radio(key=key("すみれ通り")).set_value("社名・店名・団体").run()
        app.radio(key=key("商工会議所")).set_value("名前じゃない").run()
        assert "候補5・確定5・あとで0" in " ".join(t.value for t in app.text)

        app.radio(key=key("中村")).set_value("あとで").run()
        app.button(key="name_review_save").click().run()
        assert list(app.exception) == []

    path = work / "store-synthetic" / "name-dicts" / f"{customer_id}.json"
    saved = load_name_dict(path)
    assert dict(saved.mask_terms()) == {"山本": KIND_PERSON, "青空商店": KIND_ORG, "すみれ通り": KIND_ORG}
    assert saved.pass_terms == ("商工会議所",)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert any("辞書に保存しました" in m.value for m in app.success)
    # 保存した語は一覧から消え、あとでの語（中村）だけ残る
    assert [r.key for r in app.radio] == [f"name_choice_{customer_id}_中村"]
    assert not (work / "store").exists()


@gui
def test_names_screen_bulk_confirm_saves_only_sure_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work, customer_id = make_gui_work(tmp_path, monkeypatch, "synthetic")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with mock.patch("shokoban.name_review.local_name_extractor", return_value=fake_extractor), mock.patch(
        "shokoban.name_review.run_ginza", return_value=GINZA_ITEMS
    ):
        open_names(app)
        app.button(key="name_review_run").click().run()
        app.radio(key=f"name_choice_{customer_id}_すみれ通り").set_value("社名・店名・団体").run()
        app.button(key="name_review_sure").click().run()
    saved = load_name_dict(work / "store-synthetic" / "name-dicts" / f"{customer_id}.json")
    assert dict(saved.mask_terms()) == {"山本": KIND_PERSON, "青空商店": KIND_ORG, "中村": KIND_PERSON}
    assert any(r.key.endswith("_すみれ通り") for r in app.radio)  # 迷うは保存されていない


@gui
def test_names_screen_shows_ginza_error_and_keeps_going(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_gui_work(tmp_path, monkeypatch, "synthetic")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with mock.patch("shokoban.name_review.local_name_extractor", return_value=fake_extractor), mock.patch(
        "shokoban.name_review.run_ginza", side_effect=NameReviewError("GiNZA が途中で止まりました。")
    ):
        open_names(app)
        app.button(key="name_review_run").click().run()
    assert list(app.exception) == []
    assert any("GiNZA の候補は出せませんでした" in w.value for w in app.warning)
    assert any(r.key.endswith("_青空商店") for r in app.radio)  # ほかの出どころの候補は出ている


@gui
def test_names_screen_shows_llm_error_and_keeps_going(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_gui_work(tmp_path, monkeypatch, "synthetic")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with mock.patch(
        "shokoban.name_review.local_name_extractor", side_effect=LLMRuntimeError("起動に失敗しました")
    ), mock.patch("shokoban.name_review.run_ginza", return_value=GINZA_ITEMS):
        open_names(app)
        app.button(key="name_review_run").click().run()
    assert any("手元AIの候補は出せませんでした" in w.value for w in app.warning)
    assert any(r.key.endswith("_中村") for r in app.radio)


# --- 送る前の確認（③）-------------------------------------------------------


@gui
def test_send_screen_requires_names_first_and_approves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work, customer_id = make_gui_work(tmp_path, monkeypatch, "synthetic")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with mock.patch("shokoban.name_review.local_name_extractor", return_value=fake_extractor), mock.patch(
        "shokoban.name_review.run_ginza", return_value=GINZA_ITEMS
    ), mock.patch("shokoban.mac_app.prepare_local_llm", side_effect=LLMRuntimeError("止めてあります")):
        app.button(key="nav_send").click().run()
        assert list(app.exception) == []
        assert any("先に「名前の確認」" in w.value for w in app.warning)
        assert not [b for b in app.button if b.key == "send_review_run"]

        open_names(app)
        app.button(key="name_review_run").click().run()
        app.button(key="nav_send").click().run()
        app.button(key="send_review_run").click().run()
        assert list(app.exception) == []
        assert any("区間はすべて伏せました" in w.value for w in app.warning)
        shown = " ".join(str(el.value) for el in app.get("html"))
        assert "山本" not in shown and "青空商店" not in shown  # 候補は伏せたまま見せる
        assert any(t.value.startswith("全") and "目立たせ" in t.value and "短い相づち" in t.value for t in app.text)

        app.checkbox(key="send_review_read_all").check().run()
        app.button(key="send_review_approve").click().run()
        assert list(app.exception) == []
        assert any("確認済みです" in m.value for m in app.success)

        record_id = next(iter((work / "store-synthetic" / "send-approvals" / customer_id).glob("*.json")))
        assert stat.S_IMODE(os.stat(record_id).st_mode) == 0o600
        saved = record_id.read_text(encoding="utf-8")
        assert "山本" not in saved and "架空の打合せ" not in saved

        # 確認したあとで「さらに伏せる」を変えると、確認済みではなくなる
        first = next(c for c in app.checkbox if c.key.startswith("send_hide_"))
        first.check().run()
        assert any("中身が変わっています" in w.value for w in app.warning)
    assert not (work / "store").exists()


@gui
def test_send_screen_is_not_enabled_in_real_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work, _ = make_gui_work(tmp_path, monkeypatch, "real")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with mock.patch("shokoban.mac_app.list_draft_meetings") as listing:
        app.button(key="nav_send").click().run()
    assert any("本番データではまだ使えません" in m.value for m in app.info)
    listing.assert_not_called()
    assert not (work / "store" / "send-approvals").exists()


# --- 全体共通の辞書（2段）--------------------------------------------------


def test_merge_keeps_masks_over_passes() -> None:
    from shokoban.name_dict import merge_name_dicts

    common, customer = NameDict(), NameDict()
    common.add_mask("すみれ", KIND_ORG)
    common.add_pass("チラシ")
    customer.add_pass("すみれ")  # お客様ごとの伏せないより、共通の伏せるが勝つ
    customer.add_mask("チラシ", KIND_ORG)  # 共通の伏せないより、お客様ごとの伏せるが勝つ
    merged = merge_name_dicts(common, customer)
    assert dict(merged.mask_terms()) == {"すみれ": KIND_ORG, "チラシ": KIND_ORG}
    assert merged.pass_terms == ()
    assert merge_name_dicts(common, NameDict(), common_pass=False).pass_terms == ()


def test_common_dictionary_masks_for_every_customer_and_passes_only_unsure(tmp_path: Path) -> None:
    common = NameDict()
    common.add_mask("山本", KIND_PERSON)  # 共通で伏せる → 候補に出ず、伏せる語に入る
    common.add_pass("すみれ通り")  # 迷うの候補 → 外れる
    common.add_pass("青空商店")  # 確かの候補 → 画面に残り、伏せたまま
    save_name_dict(tmp_path / "common.json", common)
    result = prepare_review(
        SENTENCES,
        dict_path=tmp_path / "c001.json",
        mode="synthetic",
        extractor=fake_extractor,
        ginza_runner=lambda s: GINZA_ITEMS,
        common_path=tmp_path / "common.json",
    )
    rows = rows_by_term(result)
    assert "山本" not in rows
    assert "すみれ通り" not in rows
    assert "青空商店" in rows
    from shokoban.name_review import load_review_dicts

    name_dict, common_pass = load_review_dicts(tmp_path / "c001.json", tmp_path / "common.json")
    extras = dict(terms_to_mask(name_dict, result.candidates))
    assert extras["山本"] == KIND_PERSON and "青空商店" in extras and "すみれ通り" not in extras
    assert common_pass == frozenset({"すみれ通り", "青空商店"})


def test_save_decisions_splits_common_and_customer(tmp_path: Path) -> None:
    result = prepare_review(
        SENTENCES,
        dict_path=tmp_path / "c001.json",
        mode="synthetic",
        extractor=fake_extractor,
        ginza_runner=lambda s: GINZA_ITEMS,
    )
    applied, updated = save_decisions(
        tmp_path / "c001.json",
        result,
        {"山本": KIND_PERSON, "商工会議所": "名前じゃない"},
        common_path=tmp_path / "common" / "common.json",
        common_terms={"商工会議所"},
    )
    assert (applied.masked, applied.passed) == (1, 1)
    assert load_name_dict(tmp_path / "common" / "common.json").pass_terms == ("商工会議所",)
    assert dict(load_name_dict(tmp_path / "c001.json").mask_terms()) == {"山本": KIND_PERSON}
    assert {"山本", "商工会議所"}.isdisjoint(rows_by_term(updated))
    with pytest.raises(NameReviewError):
        save_decisions(tmp_path / "c001.json", updated, {"中村": KIND_PERSON}, common_terms={"中村"})


@gui
def test_names_screen_saves_common_terms_to_the_common_dictionary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, customer_id = make_gui_work(tmp_path, monkeypatch, "synthetic")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with mock.patch("shokoban.name_review.local_name_extractor", return_value=fake_extractor), mock.patch(
        "shokoban.name_review.run_ginza", return_value=GINZA_ITEMS
    ):
        open_names(app)
        app.button(key="name_review_run").click().run()
        app.radio(key=f"name_choice_{customer_id}_商工会議所").set_value("名前じゃない").run()
        app.checkbox(key=f"name_common_{customer_id}_商工会議所").check().run()
        app.button(key="name_review_save").click().run()
        assert list(app.exception) == []
    common = load_name_dict(work / "store-synthetic" / "name-dicts-common" / "common.json")
    assert common.pass_terms == ("商工会議所",)
    own = load_name_dict(work / "store-synthetic" / "name-dicts" / f"{customer_id}.json")
    assert "商工会議所" not in own.pass_terms and "山本" in dict(own.mask_terms())
    assert stat.S_IMODE(os.stat(work / "store-synthetic" / "name-dicts-common" / "common.json").st_mode) == 0o600


@gui
def test_send_screen_hidden_sentence_word_goes_to_the_dictionary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, customer_id = make_gui_work(tmp_path, monkeypatch, "synthetic")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with mock.patch("shokoban.name_review.local_name_extractor", return_value=fake_extractor), mock.patch(
        "shokoban.name_review.run_ginza", return_value=GINZA_ITEMS
    ), mock.patch("shokoban.mac_app.prepare_local_llm", side_effect=LLMRuntimeError("止めてあります")):
        open_names(app)
        app.button(key="name_review_run").click().run()
        app.button(key="nav_send").click().run()
        app.button(key="send_review_run").click().run()
        hide = next(c for c in app.checkbox if c.key.startswith("send_hide_") and c.key.endswith("_4"))
        record_id = hide.key[len("send_hide_"):-len("_4")]
        hide.check().run()
        app.text_input(key=f"send_word_{record_id}_4").input("商工会議所").run()
        app.radio(key=f"send_word_kind_{record_id}_4").set_value("社名・店名・団体").run()
        app.checkbox(key="send_review_read_all").check().run()
        app.button(key="send_review_approve").click().run()
        assert list(app.exception) == []
        assert not app.error
    saved = load_name_dict(work / "store-synthetic" / "name-dicts" / f"{customer_id}.json")
    assert dict(saved.mask_terms()).get("商工会議所") == KIND_ORG


# --- 全文を読んだ確認とクラウドの5欄-----------------------------------------


def open_send_review(app: Any) -> Any:
    """名前の候補を出し、送る前の確認の「伏せて確かめる」まで進める（外には何も送らない）."""
    open_names(app)
    app.button(key="name_review_run").click().run()
    app.button(key="nav_send").click().run()
    app.button(key="send_review_run").click().run()
    assert list(app.exception) == []
    return app


def _gui_patches() -> list[Any]:
    return [
        mock.patch("shokoban.name_review.local_name_extractor", return_value=fake_extractor),
        mock.patch("shokoban.name_review.run_ginza", return_value=GINZA_ITEMS),
        mock.patch("shokoban.mac_app.prepare_local_llm", side_effect=LLMRuntimeError("止めてあります")),
    ]


@gui
def test_send_screen_approve_is_blocked_without_the_full_read_checkbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, customer_id = make_gui_work(tmp_path, monkeypatch, "synthetic")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with _gui_patches()[0], _gui_patches()[1], _gui_patches()[2]:
        open_send_review(app)
        approvals = work / "store-synthetic" / "send-approvals" / customer_id

        # 付けずに押す → 断って、何も書かない
        app.button(key="send_review_approve").click().run()
        assert list(app.exception) == []
        assert any("全文を読んだ確認" in e.value for e in app.error)
        assert not approvals.exists()
        assert not any("確認済みです" in m.value for m in app.success)

        # 付けても、目立たせた文だけ表示している間は付けていないのと同じ（チェックは押せない）
        app.checkbox(key="send_review_read_all").check().run()
        app.checkbox(key="send_review_only_flagged").check().run()
        assert app.checkbox(key="send_review_read_all").disabled
        # 付いたまま灰色に見えない（外して見せる）
        assert app.checkbox(key="send_review_read_all").value is False
        app.button(key="send_review_approve").click().run()
        assert any("全文を読んだ確認" in e.value for e in app.error)
        assert not approvals.exists()

        # 全文表示に戻すと外れたまま。付け直せば通る
        app.checkbox(key="send_review_only_flagged").uncheck().run()
        assert app.checkbox(key="send_review_read_all").value is False
        app.checkbox(key="send_review_read_all").check().run()
        app.button(key="send_review_approve").click().run()
        assert any("確認済みです" in m.value for m in app.success)
        saved = json.loads(next(approvals.glob("*.json")).read_text(encoding="utf-8"))
        assert saved["read_all"] is True and len(saved["payload_sha256"]) == 64


@gui
def test_send_screen_cloud_memo_is_off_when_env_is_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work, customer_id = make_gui_work(tmp_path, monkeypatch, "synthetic")
    monkeypatch.delenv("SHOKOBAN_CLOUD_READER", raising=False)
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with _gui_patches()[0], _gui_patches()[1], _gui_patches()[2]:
        open_send_review(app)
        assert any("クラウドの読む役は止めてあります" in i.value for i in app.info)
        app.checkbox(key="send_review_read_all").check().run()
        app.button(key="send_review_approve").click().run()
        assert any("クラウドの読む役は止めてあります" in i.value for i in app.info)
        assert not [b for b in app.button if b.key == "cloud_memo_make"]
    assert not (work / "store-synthetic" / "cloud-memos").exists()


@gui
def test_send_screen_cloud_memo_with_fake_reader_make_and_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work, customer_id = make_gui_work(tmp_path, monkeypatch, "synthetic")
    monkeypatch.setenv("SHOKOBAN_CLOUD_READER", "fake")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with _gui_patches()[0], _gui_patches()[1], _gui_patches()[2]:
        open_send_review(app)
        # 確認済みの前は、作るボタンを出さず、先に確認済みにするよう案内する
        assert not [b for b in app.button if b.key == "cloud_memo_make"]
        assert any("確認済みにする" in c.value for c in app.caption)

        app.checkbox(key="send_review_read_all").check().run()
        app.button(key="send_review_approve").click().run()
        assert [b for b in app.button if b.key == "cloud_memo_make"]
        assert not app.text_area

        app.button(key="cloud_memo_make").click().run()
        assert list(app.exception) == []
        assert not app.error
        assert len(app.text_area) == 1
        area = app.text_area[0]
        assert "【困りごと】" in area.value
        assert any("fake" in c.value for c in app.caption)
        memos = list((work / "store-synthetic" / "cloud-memos" / customer_id).glob("*.json"))
        assert len(memos) == 1 and stat.S_IMODE(os.stat(memos[0]).st_mode) == 0o600

        area.input("人が直した5欄").run()
        app.button(key="cloud_memo_save_edited").click().run()
        assert list(app.exception) == []
        assert json.loads(memos[0].read_text(encoding="utf-8"))["edited"] == "人が直した5欄"

        # 同じ中身で作り直すには、はっきり選ぶ（人が直した版を黙って上書きしない）
        make = next(b for b in app.button if b.key == "cloud_memo_make")
        assert make.disabled
        redo = next(c for c in app.checkbox if c.key.startswith("cloud_memo_redo_"))
        assert "人が直した版は消えます" in redo.label
        redo.check().run()
        next(b for b in app.button if b.key == "cloud_memo_make").click().run()
        assert list(app.exception) == []
        remade = json.loads(memos[0].read_text(encoding="utf-8"))
        assert "edited" not in remade
        # 作り直したあとは、選んだままにならない
        assert next(b for b in app.button if b.key == "cloud_memo_make").disabled
        assert not any(c.value for c in app.checkbox if c.key.startswith("cloud_memo_redo_"))
        assert app.text_area[0].value == remade["shared"]
        area = app.text_area[0]
        area.input("人が直した5欄").run()
        app.button(key="cloud_memo_save_edited").click().run()

        # 伏せ方を変えると確認済みではなくなり、5欄は隠れる（別の中身には使えない）
        assert app.checkbox(key="send_review_read_all").value is True
        first = next(c for c in app.checkbox if c.key.startswith("send_hide_"))
        first.check().run()
        assert not app.text_area
        # 全文を読んだ確認も外れる（中身が変わったので付け直す）
        assert app.checkbox(key="send_review_read_all").value is False
        assert not [b for b in app.button if b.key == "cloud_memo_make"]


@gui
def test_send_screen_cloud_memo_shows_columns_with_headings_edited_and_old_memos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, customer_id = make_gui_work(tmp_path, monkeypatch, "synthetic")
    monkeypatch.setenv("SHOKOBAN_CLOUD_READER", "fake")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with _gui_patches()[0], _gui_patches()[1], _gui_patches()[2]:
        open_send_review(app)
        app.checkbox(key="send_review_read_all").check().run()
        app.button(key="send_review_approve").click().run()
        app.button(key="cloud_memo_make").click().run()
        assert list(app.exception) == [] and not app.error
        memos = list((work / "store-synthetic" / "cloud-memos" / customer_id).glob("*.json"))
        assert len(memos) == 1

        def texts() -> list[str]:
            return [t.value for t in app.text]

        # 欄ごとに見出し（件数つき）。道筋がいちばん上で、番号つき・行頭の [こちら] は省く
        headings = [t for t in texts() if t.startswith("■ ")]
        assert headings[0] == "■ こちらから話した道筋・提案（1）"
        assert headings[1:] == ["■ 困りごと（1）", "■ 頼まれていること（0）", "■ こちらがやること（0）", "■ 決まったこと（0）"]
        body = texts()[texts().index(headings[0]) + 1]
        assert body.startswith("1. ステップ1: ") and "[こちら]" not in body
        assert any(t.startswith("・[不明] ") for t in texts())  # 困りごとの行には話し手の印
        # 読む役が作った元の版を出しているので、直した版の案内は無い
        assert not any("人が直した版を表示" in c.value for c in app.caption)
        # まだ決まっていないことは畳む。全体を直す（text_area と 直して保存）も畳みの中
        labels = [e.label for e in app.expander]
        assert "■ まだ決まっていないこと（1）" in labels and "全体を直す" in labels
        undecided = next(e for e in app.expander if e.label.startswith("■ まだ決まっていないこと"))
        assert undecided.proto.expanded is False
        assert len(app.text_area) == 1

        # 直した版を保存すると、直した版を出していることが分かる。本文は Markdown にしない
        edited = "【困りごと】\n- [お客様] ![x](http://example.invalid/a.png)\n【まだ決まっていないこと】\n- [不明] 直した未決"
        app.text_area[0].input(edited).run()
        app.button(key="cloud_memo_save_edited").click().run()
        assert list(app.exception) == [] and not app.error
        assert any("人が直した版を表示" in c.value for c in app.caption)
        assert "■ 困りごと（1）" in texts()
        assert "・[お客様] ![x](http://example.invalid/a.png)" in texts()
        assert not any("example.invalid" in m.value for m in app.markdown)
        assert "■ こちらから話した道筋・提案（1）" not in texts()  # 直した版には道筋の欄が無い

        # 前に保存した道筋の無い5欄（旧い形）もそのまま出せる
        old = json.loads(memos[0].read_text(encoding="utf-8"))
        old.pop("edited")
        old.pop("field_counts")
        old["shared"] = "\n".join(
            f"【{field}】\n- 旧い行" if field != "決まったこと" else f"【{field}】\nなし"
            for field in ("困りごと", "頼まれていること", "こちらがやること", "決まったこと", "まだ決まっていないこと")
        )
        memos[0].write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
        app.run()
        assert list(app.exception) == [] and not app.error
        assert [t for t in texts() if t.startswith("■ ")] == [
            "■ 困りごと（1）",
            "■ 頼まれていること（1）",
            "■ こちらがやること（1）",
            "■ 決まったこと（0）",
        ]
        assert any(e.label == "■ まだ決まっていないこと（1）" for e in app.expander)
        assert not any("人が直した版を表示" in c.value for c in app.caption)


@gui
def test_send_screen_cloud_memo_bad_env_value_shows_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work, _ = make_gui_work(tmp_path, monkeypatch, "synthetic")
    monkeypatch.setenv("SHOKOBAN_CLOUD_READER", "unknown")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with _gui_patches()[0], _gui_patches()[1], _gui_patches()[2]:
        open_send_review(app)
        assert any("値が正しくありません" in e.value for e in app.error)
        assert not [b for b in app.button if b.key == "cloud_memo_make"]


# --- 実データを使うモード（SHOKOBAN_REVIEW_REAL=1 で立てた画面）-----------------


def test_require_review_mode_real_needs_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from shokoban.name_review import require_review_mode

    monkeypatch.delenv("SHOKOBAN_REVIEW_REAL", raising=False)
    with pytest.raises(NameReviewError):
        require_review_mode("real")
    for value in ("", "0", "true", "yes"):
        monkeypatch.setenv("SHOKOBAN_REVIEW_REAL", value)
        with pytest.raises(NameReviewError):
            require_review_mode("real")
    monkeypatch.setenv("SHOKOBAN_REVIEW_REAL", "1")
    require_review_mode("real")
    require_review_mode("synthetic")
    with pytest.raises(NameReviewError):
        require_review_mode("other")


@gui
def test_send_screen_opens_in_real_mode_only_with_trial_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work, customer_id = make_gui_work(tmp_path, monkeypatch, "real")
    monkeypatch.setenv("SHOKOBAN_REVIEW_REAL", "1")
    monkeypatch.setenv("SHOKOBAN_CLOUD_READER", "fake")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with _gui_patches()[0], _gui_patches()[1], _gui_patches()[2]:
        open_send_review(app)
        assert not any("本番データではまだ使えません" in m.value for m in app.info)
        app.checkbox(key="send_review_read_all").check().run()
        app.button(key="send_review_approve").click().run()
        app.button(key="cloud_memo_make").click().run()
        assert list(app.exception) == []
    # 実データ用の保存先（store/）に書き、架空の保存先には書かない
    assert list((work / "store" / "send-approvals" / customer_id).glob("*.json"))
    assert list((work / "store" / "cloud-memos" / customer_id).glob("*.json"))
    assert not (work / "store-synthetic").exists()


def _confirm_only_meeting(work: Path, mode: str, customer_id: str) -> None:
    """下書きを確定済みにする（確定済みの打ち合わせを選べるかを見るため）."""
    store = CustomerStore(work / ("store-synthetic" if mode == "synthetic" else "store"))
    for meeting in store.list_meetings(customer_id):
        store.confirm_meeting(
            customer_id, meeting["record_id"], expected_revision=meeting["revision"], confirmed_by="test"
        )


@gui
def test_trial_screens_list_confirmed_meetings_only_with_trial_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, customer_id = make_gui_work(tmp_path, monkeypatch, "real")
    _confirm_only_meeting(work, "real", customer_id)
    store = CustomerStore(work / "store")
    before = store.list_meetings(customer_id)
    monkeypatch.setenv("SHOKOBAN_REVIEW_REAL", "1")
    monkeypatch.setenv("SHOKOBAN_CLOUD_READER", "fake")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with _gui_patches()[0], _gui_patches()[1], _gui_patches()[2]:
        open_names(app)
        box = app.selectbox(key="name_review_meeting")
        assert len(box.options) == 1 and "（確定済み）" in box.options[0]
        open_send_review(app)
        app.checkbox(key="send_review_read_all").check().run()
        app.button(key="send_review_approve").click().run()
        app.button(key="cloud_memo_make").click().run()
        assert list(app.exception) == []
    assert list((work / "store" / "cloud-memos" / customer_id).glob("*.json"))
    # 確定済みの記録は変えない
    assert store.list_meetings(customer_id) == before


@gui
def test_synthetic_screens_do_not_list_confirmed_meetings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work, customer_id = make_gui_work(tmp_path, monkeypatch, "synthetic")
    _confirm_only_meeting(work, "synthetic", customer_id)
    monkeypatch.setenv("SHOKOBAN_REVIEW_REAL", "1")  # 架空データでは印があっても確定済みは出さない
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    open_names(app)
    assert any("下書きの打ち合わせがありません" in m.value for m in app.info)


@gui
def test_cloud_memo_share_needs_label_and_writes_share(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from shokoban.mac_app import mac_app_paths_from_env, save_share_label, status_local_dir_for_mode

    work, customer_id = make_gui_work(tmp_path, monkeypatch, "synthetic")
    monkeypatch.setenv("SHOKOBAN_CLOUD_READER", "fake")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    with _gui_patches()[0], _gui_patches()[1], _gui_patches()[2]:
        open_send_review(app)
        app.checkbox(key="send_review_read_all").check().run()
        app.button(key="send_review_approve").click().run()
        app.button(key="cloud_memo_make").click().run()
        # 呼び名が無いうちは渡せない
        assert any("共有用の呼び名" in i.value for i in app.info)
        assert not [b for b in app.button if b.key == "cloud_memo_share"]

        labels = status_local_dir_for_mode(mac_app_paths_from_env(), "synthetic") / "share-labels.json"
        save_share_label(labels, customer_id, "架空A")
        app.run()
        app.button(key="cloud_memo_share").click().run()
        assert list(app.exception) == []
        assert any("渡しました" in s.value for s in app.success)
    files = list((work / "share-synthetic" / "進行メモ" / "架空A").glob("*.md"))
    assert len(files) == 1 and "【こちらから話した道筋・提案】" in files[0].read_text(encoding="utf-8")
    assert (work / "share-synthetic" / "進行メモ" / "README.md").is_file()
    assert not (work / "share").exists()


@gui
def test_cloud_memo_counts_real_reader_calls_and_stops_at_monthly_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shokoban.cloud_memo import fake_cloud_reader

    work, customer_id = make_gui_work(tmp_path, monkeypatch, "synthetic")
    monkeypatch.setenv("SHOKOBAN_CLOUD_READER", "fake")
    monkeypatch.setenv("SHOKOBAN_READER_MONTHLY_LIMIT", "1")
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    # 本物（opus）の名前で、中身は偽物の読む役（外へは出さない）
    with _gui_patches()[0], _gui_patches()[1], _gui_patches()[2], mock.patch(
        "shokoban.cloud_memo.cloud_reader_from_env", return_value=(fake_cloud_reader, "opus")
    ):
        open_send_review(app)
        app.checkbox(key="send_review_read_all").check().run()
        app.button(key="send_review_approve").click().run()
        assert any("今月の読む役: 0/1回" in c.value for c in app.caption)
        app.button(key="cloud_memo_make").click().run()
        assert list(app.exception) == []
        assert any("今月の読む役: 1/1回" in c.value for c in app.caption)
        usage = json.loads((work / "store-synthetic" / "cloud-memos" / "reader-usage.json").read_text(encoding="utf-8"))
        assert sum(usage.values()) == 1
        # 上限に届いたら、作り直しを選んでも呼ばない
        next(c for c in app.checkbox if c.key.startswith("cloud_memo_redo_")).check().run()
        with mock.patch("shokoban.cloud_memo.make_memo") as made:
            app.button(key="cloud_memo_make").click().run()
        assert any("上限" in e.value for e in app.error)
    usage = json.loads((work / "store-synthetic" / "cloud-memos" / "reader-usage.json").read_text(encoding="utf-8"))
    assert sum(usage.values()) == 1
