"""名前の確認の画面が使う、候補出し・辞書への保存・伏せた文の見直し.

方針:

- 名前・社名は、網（ルールによる検出）・手元の12B・GiNZA の3つの出どころで候補を出す。
- **確認されていない候補は、すべて伏せたまま送る**（確か・迷うとも）。人が画面で
  決めた語だけが、お客様ごとの辞書（伏せる語・伏せない語）に入る。

このモジュールは Streamlit に依存しない（画面はここで作った値を表示するだけ）。

辞書の場所: ``<保存先>/name-dicts/<customer_id>.json``（0600）。保存先はデータモードで決まる
（``mac_app.store_root_for_mode``。架空データは ``store-synthetic``）。お客様の記録の置き場
（``customer-vault``）の外に置く。辞書は手元の保存先だけにあり、git・クラウドに入れない。

GiNZA は別の venv（``<WORK>/venv-ginza``）の Python で ``scripts/ginza_names.py --stdin`` を
動かして使う。本体の venv には spaCy を入れない（このモジュールは spaCy を import しない）。
GiNZA が動かなかったときは、エラーを画面に出し、ほかの出どころの候補だけで続ける
（見つかった候補は伏せたまま）。

架空データ（``synthetic``）では常に使える。本番データ（``real``）では ``SHOKOBAN_REVIEW_REAL=1``
のときだけ開く（それ以外は何も読まず、使えないと答える）。
"""

from __future__ import annotations

import html
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import llm_runtime
from .llm_runtime import LLMRuntimeError, RuntimeConfig, load_runtime_config
from .mac_app import (
    MacAppError,
    MacAppPaths,
    _check_customer_id,
    _ensure_private_dir,
    prepare_local_llm,
    store_root_for_mode,
)
from .mask_r3 import MaskR3Error, dry_tagger, mask_r3, tagger_model
from .name_candidates import (
    JUDGMENT_SURE,
    JUDGMENT_UNSURE,
    Candidate,
    CandidateResult,
    find_candidates,
    dict_terms_to_mask,
    make_llama_name_extractor,
    terms_to_mask,
)
from .name_dict import (
    merge_name_dicts,
    KIND_ORG,
    KIND_PERSON,
    MASK_KINDS,
    NameDict,
    NameDictError,
    check_term,
    load_name_dict,
    save_name_dict,
)
from .pick_bench import split_sentences

__all__ = [
    "CHOICE_LABELS",
    "CHOICE_LATER",
    "DECISION_KINDS",
    "DECISION_NOT_NAME",
    "GINZA_TIMEOUT_SECONDS",
    "NAME_DICT_DIR",
    "NAME_DICT_SUFFIX",
    "REVIEW_MODES",
    "REVIEW_REAL_ENV",
    "JUDGMENT_ORDER",
    "ApplyResult",
    "NameReviewError",
    "ReviewResult",
    "ReviewRow",
    "apply_review",
    "build_review",
    "choice_to_decision",
    "collect_decisions",
    "count_choices",
    "ginza_paths",
    "highlight_html",
    "local_name_extractor",
    "masked_preview",
    "meeting_sentences",
    "name_dict_path",
    "common_dict_path",
    "load_review_dicts",
    "COMMON_DICT_DIR",
    "pick_name_model",
    "prepare_review",
    "remaining_rows",
    "require_review_mode",
    "review_real_enabled",
    "run_ginza",
    "save_decisions",
]

# 名前の確認を使えるデータモード（本番データは、明示的に有効にするまで使わない）。
REVIEW_MODES: tuple[str, ...] = ("synthetic",)
# 本番データ（real）でこの画面（名前の確認・送る前の確認）を使うときに "1" にする環境変数。
REVIEW_REAL_ENV = "SHOKOBAN_REVIEW_REAL"

NAME_DICT_DIR = "name-dicts"
NAME_DICT_SUFFIX = ".json"
# 全体共通の辞書（お客様ごとの辞書と合わせる2段）。お客様の識別子と重ならないよう、別のフォルダに置く。
COMMON_DICT_DIR = "name-dicts-common"
COMMON_DICT_NAME = "common.json"

DECISION_NOT_NAME = "名前じゃない"
# apply_review が受ける決め方。人名・社名は辞書の「伏せる語」、名前じゃないは「伏せない語」に入る。
DECISION_KINDS: tuple[str, ...] = (KIND_PERSON, KIND_ORG, DECISION_NOT_NAME)

# 画面の選択肢（見える文字）と、辞書への決め方。「あとで」は決めない（辞書に入れない）。
CHOICE_LATER = "あとで"
CHOICE_LABELS: dict[str, str | None] = {
    "名前": KIND_PERSON,
    KIND_ORG: KIND_ORG,
    DECISION_NOT_NAME: DECISION_NOT_NAME,
    CHOICE_LATER: None,
}

GINZA_TIMEOUT_SECONDS = 300
GINZA_VENV_DIR = "venv-ginza"
GINZA_SCRIPT_NAME = "ginza_names.py"
EXAMPLES_PER_TERM = 2
# 画面に出す例文の、語の前後の長さ（文字数）。
SNIPPET_CONTEXT = 50


class NameReviewError(RuntimeError):
    """名前の確認が続けられないとき（メッセージは画面に出してよい固定の文言）."""


# ---------------------------------------------------------------- 場所・モード

def review_real_enabled(mode: str) -> bool:
    """本番データで、``SHOKOBAN_REVIEW_REAL=1`` が付いているか."""
    return mode == "real" and os.environ.get(REVIEW_REAL_ENV, "").strip() == "1"


def require_review_mode(mode: str) -> None:
    """この画面を使えるデータモードか（使えなければ ``NameReviewError``）."""
    if mode == "real":
        if os.environ.get(REVIEW_REAL_ENV, "").strip() == "1":
            return
        raise NameReviewError("名前の確認は、本番データではまだ使えません。")
    if mode not in REVIEW_MODES:
        raise NameReviewError("名前の確認は、架空データのときだけ使えます。")


def name_dict_path(paths: MacAppPaths, mode: str, customer_id: str) -> Path:
    """``<保存先>/name-dicts/<customer_id>.json`` の場所を返す（フォルダ・ファイルは作らない）."""
    require_review_mode(mode)
    try:
        root = store_root_for_mode(paths, mode)
    except MacAppError as exc:
        raise NameReviewError("保存先を決められません。") from exc
    if not _check_customer_id(customer_id):
        raise NameReviewError("お客様の識別子が不正です。")
    return root / NAME_DICT_DIR / f"{customer_id}{NAME_DICT_SUFFIX}"


def common_dict_path(paths: MacAppPaths, mode: str) -> Path:
    """``<保存先>/name-dicts-common/common.json``（全体共通の辞書。フォルダ・ファイルは作らない）."""
    require_review_mode(mode)
    try:
        root = store_root_for_mode(paths, mode)
    except MacAppError as exc:
        raise NameReviewError("保存先を決められません。") from exc
    return root / COMMON_DICT_DIR / COMMON_DICT_NAME


def _load(path: Any) -> NameDict:
    if path is None:
        return NameDict()
    try:
        return load_name_dict(path, missing_ok=True)
    except (NameDictError, OSError) as exc:
        raise NameReviewError("辞書を読めません。") from exc


def load_review_dicts(dict_path: Any, common_path: Any = None) -> tuple[NameDict, frozenset[str]]:
    """候補出し・伏せ方に使う辞書と、全体共通の「伏せない語」を返す.

    返す辞書は、お客様ごとの辞書に共通の「伏せる語」を足したもの（共通の伏せない語は入れない）。
    共通の伏せない語は、自信の低い候補（迷う）を外すときだけに使う（``apply_common_pass``）。
    """
    customer = _load(dict_path)
    common = _load(common_path)
    return merge_name_dicts(common, customer, common_pass=False), frozenset(common.pass_terms)


def apply_common_pass(candidates: Iterable[Candidate], common_pass: Iterable[str]) -> tuple[Candidate, ...]:
    """全体共通の「伏せない語」に入った候補のうち、迷うの候補だけを外す（確かは画面に出し、伏せたまま）."""
    passed = set(common_pass)
    return tuple(c for c in candidates if not (c.term in passed and c.judgment == JUDGMENT_UNSURE))


def ginza_paths(paths: MacAppPaths) -> tuple[Path, Path]:
    """GiNZA を動かす ``(venv-ginza の python, scripts/ginza_names.py)``（あるかは確かめない）."""
    python_path = paths.work_dir / GINZA_VENV_DIR / "bin" / "python"
    script_path = Path(__file__).resolve().parents[2] / "scripts" / GINZA_SCRIPT_NAME
    return python_path, script_path


# ---------------------------------------------------------------- GiNZA（別の環境）

def run_ginza(
    sentences: Sequence[str],
    *,
    python_path: Any,
    script_path: Any,
    timeout: float = GINZA_TIMEOUT_SECONDS,
    runner: Callable[..., Any] = subprocess.run,
) -> list[dict[str, Any]]:
    """``scripts/ginza_names.py --stdin`` を GiNZA の環境の Python で動かし、項目の並びを返す.

    文は標準入力の JSON（``{"sentences": [...]}``）で渡し、標準出力の JSON（``{"items": [...]}``）を
    読む。ファイルには書かない。Python は ``-I``（環境変数・カレントの影響を受けない）で動かし、
    環境変数は最小限だけ渡す（ネットには出ない設定）。失敗（環境が無い・時間切れ・異常終了・
    返事が壊れている）は ``NameReviewError``（メッセージに文・語は入れない）。
    """
    if type(sentences) is str:
        raise NameReviewError("文は並びで渡してください。")
    python = Path(python_path)
    script = Path(script_path)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise NameReviewError("GiNZA の環境が見つかりません。")
    if not script.is_file():
        raise NameReviewError("GiNZA の道具（ginza_names.py）が見つかりません。")
    payload = json.dumps({"sentences": [str(item) for item in sentences]}, ensure_ascii=False)
    env = {
        "PATH": "/usr/bin:/bin",
        "LANG": "ja_JP.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    try:
        completed = runner(
            [str(python), "-I", str(script), "--stdin"],
            input=payload.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
            env=env,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NameReviewError("GiNZA が時間内に終わりませんでした。") from exc
    except OSError as exc:
        raise NameReviewError("GiNZA を動かせませんでした。") from exc
    if getattr(completed, "returncode", 1) != 0:
        raise NameReviewError("GiNZA が途中で止まりました。")
    stdout = getattr(completed, "stdout", None)
    try:
        data = json.loads(stdout.decode("utf-8") if isinstance(stdout, bytes) else stdout)
    except (AttributeError, UnicodeError, ValueError, TypeError) as exc:
        raise NameReviewError("GiNZA の返事を読めませんでした。") from exc
    if type(data) is not dict or type(data.get("items")) is not list:
        raise NameReviewError("GiNZA の返事の形が正しくありません。")
    if any(type(item) is not dict for item in data["items"]):
        raise NameReviewError("GiNZA の返事の形が正しくありません。")
    return list(data["items"])


# ---------------------------------------------------------------- 12B（手元AI）

def pick_name_model(cfg: RuntimeConfig, chosen_alias: str | None = None) -> str:
    """候補出しに使う手元AIの別名. 札付けと同じ ``tagger_model()`` だけを使う.

    名前は環境変数 ``SHOKOBAN_TAGGER_MODEL``（既定 ``gemma-4-12b-ja``）。設定に無ければ
    止める（画面で選んだ別のモデルへは切り替えない。精度を確かめていないモデルで名前を拾うと、
    漏れの数が分からないため）。``chosen_alias`` は使わない。
    """
    models = getattr(cfg, "models", {}) or {}
    name = tagger_model()
    if name in models:
        return name
    raise NameReviewError(
        f"候補出しに使う手元AI（{name}）が設定にありません。"
        "SHOKOBAN_TAGGER_MODEL か runtime.json の models に、その名前を入れてください。"
    )


def local_name_extractor(
    cfg: RuntimeConfig,
    alias: str | None,
    *,
    ensure: Callable[[RuntimeConfig, str | None], str] = llm_runtime.ensure_running,
) -> Any:
    """手元AIを（必要なら起こして）使い、固有名詞を拾う関数を返す（画面のほかの手元AI利用と同じ起動・停止のしかた）.

    起動に失敗したら ``LLMRuntimeError`` をそのまま上げる（別のモデル・クラウドには切り替えない）。
    使い終わった後は、手元AIの「設定の時間使わなければ止まる」に任せる。
    """
    settings = prepare_local_llm(cfg, pick_name_model(cfg, alias), ensure=ensure)
    return make_llama_name_extractor(settings.model)


# ---------------------------------------------------------------- 文

def meeting_sentences(meeting: Mapping[str, Any]) -> list[str]:
    """打ち合わせ1件（``list_draft_meetings`` の1件）の文字を、文の並びにする（空の文は除く）."""
    details = str(meeting.get("details") or "")
    summary = str(meeting.get("summary") or "")
    parts = [str(meeting.get("title") or "")]
    if summary.strip() and summary.strip() not in details:
        parts.append(summary)
    parts.append(details)
    parts += [str(value) for value in meeting.get("decisions") or []]
    parts += [str(value) for value in meeting.get("next_actions") or []]
    return split_sentences("\n".join(part for part in parts if part.strip()))


def highlight_html(sentence: str, term: str) -> str:
    """例文の中の語を ``<mark>`` で囲んだ HTML（文字はすべてエスケープ。長い文は語の前後だけ）."""
    if type(sentence) is not str or type(term) is not str:
        raise TypeError("sentence と term は文字列で渡してください")
    start = sentence.find(term) if term else -1
    if start < 0:
        text = sentence[: SNIPPET_CONTEXT * 2]
        return html.escape(text) + ("…" if len(sentence) > len(text) else "")
    stop = start + len(term)
    head = sentence[max(0, start - SNIPPET_CONTEXT) : start]
    tail = sentence[stop : stop + SNIPPET_CONTEXT]
    return (
        ("…" if start > SNIPPET_CONTEXT else "")
        + html.escape(head)
        + "<mark>"
        + html.escape(term)
        + "</mark>"
        + html.escape(tail)
        + ("…" if len(sentence) - stop > SNIPPET_CONTEXT else "")
    )


# ---------------------------------------------------------------- 候補

@dataclass(frozen=True)
class ReviewRow:
    """確認の画面の1行（辞書にまだ無い候補）."""

    term: str
    kind: str  # 人名 / 社名・店名・団体 / それ以外（システムの見立て）
    judgment: str  # 確か / 迷う
    sources: tuple[str, ...]
    examples: tuple[str, ...]  # 出てきた文（元の文。手元のこの画面にだけ出す。最大2つ）

    @property
    def default_decision(self) -> str | None:
        """初めに選んでおく決め方。確かは見立ての種類、迷う・それ以外の見立ては決めない（None）."""
        if self.judgment == JUDGMENT_SURE and self.kind in MASK_KINDS:
            return self.kind
        return None

    @property
    def default_choice(self) -> str:
        """``default_decision`` に当たる、画面の選択肢の文字."""
        decision = self.default_decision
        for label, value in CHOICE_LABELS.items():
            if value == decision:
                return label
        return CHOICE_LATER


@dataclass(frozen=True)
class ReviewResult:
    rows: tuple[ReviewRow, ...]  # 確か → 迷う の順
    candidates: tuple[Candidate, ...]  # 辞書に無い候補すべて（伏せる語の元）
    ginza_error: str | None = None  # GiNZA が使えなかったときの文言
    llm_error: str | None = None  # 12B が使えなかったときの文言
    llm_failed_chunks: int = 0  # 12B が返事を返せなかった区切りの数
    ginza_used: bool = False


# 画面で並べる順（確かを先に）。
JUDGMENT_ORDER: tuple[str, ...] = (JUDGMENT_SURE, JUDGMENT_UNSURE)


def choice_to_decision(choice: str) -> str | None:
    """画面の選択肢を、辞書への決め方にする（あとで → None。知らない選択肢は ``NameReviewError``）."""
    if type(choice) is not str or choice not in CHOICE_LABELS:
        raise NameReviewError("選択肢が正しくありません。")
    return CHOICE_LABELS[choice]


def _examples(candidate: Candidate, sentences: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for number in candidate.sentences[:EXAMPLES_PER_TERM]:
        if 1 <= number <= len(sentences):
            out.append(sentences[number - 1])
    return tuple(out)


def _row_order(candidate: Candidate) -> tuple[int, int, str]:
    first = candidate.sentences[0] if candidate.sentences else 10**9
    return (0 if candidate.judgment == JUDGMENT_SURE else 1, first, candidate.term)


def prepare_review(
    sentences: Sequence[str],
    *,
    dict_path: Any,
    mode: str,
    extractor: Callable[[str], Sequence[Any]] | None = None,
    ginza_runner: Callable[[Sequence[str]], Sequence[Any]] | None = None,
    company_name: str = "",
    aliases: Sequence[str] = (),
    second_pass: bool = False,
    common_path: Any = None,
) -> ReviewResult:
    """辞書にまだ無い候補を、確認の画面の行にして返す（確か → 迷う の順）.

    ``common_path`` は 全体共通の辞書。共通の伏せる語は候補に出ず伏せる。共通の伏せない語は、
    迷うの候補だけを外す。

    ``extractor`` は12Bの候補出し（省略すると12Bは使わない）、``ginza_runner`` は
    ``sentences`` を受けて GiNZA の項目を返す関数（省略すると GiNZA は使わない）。GiNZA が
    ``NameReviewError`` などで失敗したときは ``ginza_error`` に文言を入れ、ほかの出どころの候補で
    続ける（見つけた候補は伏せたまま）。辞書にある語（伏せる・伏せない）は候補に出ない。
    """
    require_review_mode(mode)
    if type(sentences) is str:
        raise NameReviewError("文は並びで渡してください。")
    texts = [str(item) for item in sentences]
    name_dict, common_pass = load_review_dicts(dict_path, common_path)

    ginza_items: list[Any] | None = None
    ginza_error: str | None = None
    if ginza_runner is not None:
        try:
            ginza_items = list(ginza_runner(texts))
        except NameReviewError as exc:
            ginza_error = str(exc)
        except Exception:  # noqa: BLE001 - GiNZA は別の環境。どんな失敗でも、ほかの出どころで続ける
            ginza_error = "GiNZA を使えませんでした。"

    found: CandidateResult = find_candidates(
        texts,
        extractor=extractor,
        name_dict=name_dict,
        company_name=company_name,
        aliases=aliases,
        ginza_items=ginza_items,
        second_pass=second_pass,
    )
    kept = apply_common_pass(found.candidates, common_pass)
    ordered = sorted(kept, key=_row_order)
    rows = tuple(
        ReviewRow(
            term=c.term,
            kind=c.kind,
            judgment=c.judgment,
            sources=c.sources,
            examples=_examples(c, texts),
        )
        for c in ordered
    )
    return ReviewResult(
        rows=rows,
        candidates=kept,
        ginza_error=ginza_error,
        llm_failed_chunks=found.llm.chunks_fail + found.llm2.chunks_fail,
        ginza_used=ginza_runner is not None and ginza_error is None,
    )


def build_review(
    paths: MacAppPaths,
    mode: str,
    customer_id: str,
    sentences: Sequence[str],
    *,
    alias: str | None,
    company_name: str = "",
    aliases: Sequence[str] = (),
) -> ReviewResult:
    """画面の「候補を出す」: 網＋手元の12B＋GiNZA で候補を出す.

    12B（手元AI）が起動しない・設定が読めないときは ``llm_error`` に文言を入れ、網と GiNZA で続ける。
    本番データでは何も読まずに ``NameReviewError``。
    """
    require_review_mode(mode)
    dict_path = name_dict_path(paths, mode, customer_id)

    extractor = None
    llm_error: str | None = None
    try:
        cfg = load_runtime_config(paths.runtime_config)
        extractor = local_name_extractor(cfg, alias)
    except (LLMRuntimeError, MaskR3Error, NameReviewError) as exc:
        llm_error = str(exc) or "手元AIを使えませんでした。"
    python_path, script_path = ginza_paths(paths)

    def ginza(texts: Sequence[str]) -> list[dict[str, Any]]:
        return run_ginza(
            texts,
            python_path=python_path,
            script_path=script_path,
            timeout=GINZA_TIMEOUT_SECONDS,
        )

    result = prepare_review(
        sentences,
        dict_path=dict_path,
        mode=mode,
        extractor=extractor,
        ginza_runner=ginza,
        company_name=company_name,
        aliases=aliases,
        common_path=common_dict_path(paths, mode),
    )
    if llm_error is None:
        return result
    return ReviewResult(
        rows=result.rows,
        candidates=result.candidates,
        ginza_error=result.ginza_error,
        llm_error=llm_error,
        llm_failed_chunks=result.llm_failed_chunks,
        ginza_used=result.ginza_used,
    )


def remaining_rows(rows: Iterable[ReviewRow], dict_path: Any, common_path: Any = None) -> tuple[ReviewRow, ...]:
    """保存のあとに、どちらの辞書にもまだ入っていない行だけを返す."""
    customer = _load(dict_path)
    common = _load(common_path)
    return tuple(row for row in rows if not customer.knows(row.term) and not common.knows(row.term))


# ---------------------------------------------------------------- 辞書への保存

@dataclass(frozen=True)
class ApplyResult:
    masked: int  # 伏せる語に入れた数
    passed: int  # 伏せない語に入れた数

    @property
    def total(self) -> int:
        return self.masked + self.passed


def apply_review(
    dict_path: Any,
    decisions: Mapping[str, str],
    *,
    allowed_terms: Iterable[str] | None = None,
) -> ApplyResult:
    """人が決めたことを辞書に書く（人名・社名・店名・団体 → 伏せる語、名前じゃない → 伏せない語）.

    ``decisions`` は ``{語: 決め方}``。決めなかった語（あとで）は渡さない（辞書に入れない＝
    次も候補に出て、伏せたまま）。知らない決め方・正しくない語は ``NameReviewError``、
    ``allowed_terms`` を渡したときは、その外の語も ``NameReviewError``。**全部を確かめてから
    書く**（途中までは書かない）。書くときは同じフォルダの一時ファイル→``os.replace``（0600）。
    """
    if not isinstance(decisions, Mapping):
        raise NameReviewError("決めたことは {語: 決め方} で渡してください。")
    allowed = set(allowed_terms) if allowed_terms is not None else None
    checked: list[tuple[str, str]] = []
    for term, kind in decisions.items():
        try:
            value = check_term(term)
        except NameDictError as exc:
            raise NameReviewError("語が正しくありません。") from exc
        if type(kind) is not str or kind not in DECISION_KINDS:
            raise NameReviewError("決め方は 人名・社名・店名・団体・名前じゃない のどれかにしてください。")
        if allowed is not None and value not in allowed:
            raise NameReviewError("候補に無い語は保存できません。")
        checked.append((value, kind))
    if not checked:
        return ApplyResult(0, 0)

    target = Path(dict_path)
    if target.is_symlink():
        raise NameReviewError("辞書の場所がシンボリックリンクです。")
    try:
        name_dict = load_name_dict(target, missing_ok=True)
    except NameDictError as exc:
        raise NameReviewError("辞書を読めません。") from exc
    masked = passed = 0
    try:
        for term, kind in checked:
            if kind == DECISION_NOT_NAME:
                name_dict.add_pass(term)
                passed += 1
            else:
                name_dict.add_mask(term, kind)
                masked += 1
        _ensure_private_dir(target.parent)
        save_name_dict(target, name_dict)
    except (NameDictError, MacAppError, OSError) as exc:
        raise NameReviewError("辞書を保存できませんでした。") from exc
    return ApplyResult(masked, passed)


def count_choices(choices: Mapping[str, str]) -> tuple[int, int, int]:
    """``(候補の数, 確定の数, あとでの数)``（画面の「候補N・確定M・あとでK」）."""
    total = len(choices)
    later = sum(1 for choice in choices.values() if choice == CHOICE_LATER)
    return total, total - later, later


def collect_decisions(
    result: ReviewResult, choices: Mapping[str, str], *, sure_only: bool
) -> dict[str, str]:
    """画面の選択肢から、辞書に書く決め方を集める（あとでの語は入れない）.

    ``sure_only`` が真なら「確か」の行だけ（「確かの分をまとめて確定」）。
    """
    rows = {
        row.term: row
        for row in result.rows
        if not sure_only or row.judgment == JUDGMENT_SURE
    }
    decisions: dict[str, str] = {}
    for term, choice in choices.items():
        decision = choice_to_decision(choice)
        if decision is not None and term in rows:
            decisions[term] = decision
    return decisions


def save_decisions(
    dict_path: Any,
    result: ReviewResult,
    decisions: Mapping[str, str],
    *,
    common_path: Any = None,
    common_terms: Iterable[str] = (),
) -> tuple[ApplyResult, ReviewResult]:
    """決めた分を辞書に保存し、保存した語を外した ``ReviewResult`` を返す.

    保存できる語は、今の候補の中だけ（``apply_review(allowed_terms=...)``）。``common_terms`` に
    入れた語は 全体共通の辞書（``common_path``）へ、ほかはお客様ごとの辞書へ。
    """
    shared = set(common_terms)
    if shared and common_path is None:
        raise NameReviewError("共通の辞書の場所がありません。")
    allowed = [row.term for row in result.rows]
    own = {term: kind for term, kind in decisions.items() if term not in shared}
    common = {term: kind for term, kind in decisions.items() if term in shared}
    applied_own = apply_review(dict_path, own, allowed_terms=allowed)
    applied_common = apply_review(common_path, common, allowed_terms=allowed) if common else ApplyResult(0, 0)
    applied = ApplyResult(
        applied_own.masked + applied_common.masked, applied_own.passed + applied_common.passed
    )
    rest = remaining_rows(result.rows, dict_path, common_path)
    left = {row.term for row in rest}
    updated = ReviewResult(
        rows=rest,
        candidates=tuple(c for c in result.candidates if c.term in left),
        ginza_error=result.ginza_error,
        llm_error=result.llm_error,
        llm_failed_chunks=result.llm_failed_chunks,
        ginza_used=result.ginza_used,
    )
    return applied, updated


# ---------------------------------------------------------------- 伏せた文の見直し

def masked_preview(
    sentences: Sequence[str],
    name_dict: NameDict | None,
    candidates: Sequence[Candidate],
    *,
    company_name: str = "",
    aliases: Sequence[str] = (),
) -> list[str]:
    """送るときと同じ伏せ方をした文の並び（あとの「見直し」の画面用）.

    ``mask_r3(extra_terms=terms_to_mask(辞書, 候補))``。辞書の伏せる語に、まだ確認されていない
    候補（確か・迷うとも）を足して伏せる。伏せない語は伏せない。区間の札付けは使わない
    （``dry_tagger``。名前の見直しだけを見せる）。
    """
    if type(sentences) is str:
        raise NameReviewError("文は並びで渡してください。")
    extras = terms_to_mask(name_dict, candidates)
    result = mask_r3(
        [str(item) for item in sentences],
        tagger=dry_tagger,
        tagger_model="dry",
        company_name=company_name,
        aliases=aliases,
        extra_terms=extras,
        dict_terms=dict_terms_to_mask(name_dict),
    )
    return [record.text for record in result.records]
