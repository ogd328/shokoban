"""読む役（大きいAI）を ``claude`` コマンド（Claude Code CLI）で呼ぶ部品.

- ``claude_command``: 呼び出しの引数を組む。
- ``run_claude``: ``claude -p`` を1回呼び、返事の本文と返ってきたモデル名を
  ``ReaderReply`` で返す。
- ``make_claude_reader``: 上を ``reader(prompt)`` の形の関数にする。テストでは
  偽の読む役（``cloud_memo.fake_cloud_reader`` など）を代わりに差し込む。

呼び方の決まり:

- ``claude -p`` を、道具を全部切る指定（``--tools`` に空文字）で呼ぶ。
- ``--model <名前>`` で使うモデルを固定する（1回の run の途中で替えない）。
- ``--bare`` は使わない（ログインの設定を読ませるため）。
- 権限を飛ばす指定（``--dangerously-skip-permissions`` など）は使わない。
- 作業フォルダは空の一時フォルダ。入力は標準入力で渡す。
- 返ったモデル名を ``ReaderReply.model`` に入れる（``cloud_memo`` が送ったメモの
  ``reader_model`` に書く）。

指示文の組み立てと返事の読み取りは ``reader_r3``、ここを使うのは ``cloud_memo`` と
``reader_local``。名前に ``stage5`` が付いているのは歴史的な名前。

この module は本物の ``claude`` を import 時に起動しない。呼んだときだけ動く。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_CLAUDE_BIN = "claude"
DEFAULT_TIMEOUT = 600
# 道具を全部切る指定は空文字で渡す。--bare と権限を飛ばす指定は使わない。
TOOLS_OPTION = "--tools"
DISABLED_TOOLS = ""
OUTPUT_FORMAT_OPTION = "--output-format"
OUTPUT_FORMAT_JSON = "json"
# 使うモデルを固定する（1回の run の途中で替えない）。
MODEL_OPTION = "--model"
FORBIDDEN_OPTIONS: tuple[str, ...] = ("--bare", "--dangerously-skip-permissions")


class ReaderStage5Error(RuntimeError):
    """読む役を続けられないときに送出する."""


@dataclass(frozen=True)
class ReaderReply:
    """読む役の返事（本文と、返ってきたモデル名）."""

    text: str
    model: str = ""


def _checked_model(model: Any) -> str:
    """使うモデル名を検査する（空なら ``ReaderStage5Error``）."""
    if type(model) is not str or not model.strip():
        raise ReaderStage5Error(
            "読む役（claude）のモデル名が空です。1回の run の途中でモデルを替えないため、"
            "使うモデル名を指定してください"
        )
    return model.strip()


def claude_command(binary: str = DEFAULT_CLAUDE_BIN, *, model: str) -> list[str]:
    """本物の読む役の引数（モデルを固定・道具を切る・--bare を使わない・権限を飛ばさない）."""
    name = _checked_model(model)
    command = [
        binary,
        "-p",
        MODEL_OPTION,
        name,
        TOOLS_OPTION,
        DISABLED_TOOLS,
        OUTPUT_FORMAT_OPTION,
        OUTPUT_FORMAT_JSON,
    ]
    for option in FORBIDDEN_OPTIONS:
        if option in command:  # pragma: no cover - 定数を変えたときに気づくため
            raise ReaderStage5Error(f"使わない指定が入っています: {option}")
    return command


def _reply_from_payload(raw: str, returncode: int) -> ReaderReply:
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise ReaderStage5Error(
            f"読む役（claude）の返答を JSON として読めません（終了コード {returncode}）"
        ) from exc
    if type(payload) is not dict:
        raise ReaderStage5Error("読む役（claude）の返答がオブジェクトではありません")
    text = payload.get("result")
    if type(text) is not str:
        raise ReaderStage5Error("読む役（claude）の返答に result がありません")
    model = payload.get("model")
    if type(model) is not str or not model.strip():
        usage = payload.get("modelUsage")
        if type(usage) is dict:
            names = [name for name in usage if type(name) is str and name.strip()]
            model = names[0] if names else ""
    return ReaderReply(text=text, model=str(model or "").strip())


def run_claude(
    prompt: str,
    *,
    model: str,
    binary: str = DEFAULT_CLAUDE_BIN,
    timeout: int = DEFAULT_TIMEOUT,
    runner: Callable[..., Any] | None = None,
) -> ReaderReply:
    """``claude -p`` を1回呼び、返事とモデル名を返す（テストでは起動しない）.

    作業フォルダは空の一時フォルダ。入力は標準入力。``--model`` で使うモデルを
    固定し、``--output-format json`` で受けた ``result`` を本文に、``model``
    （無ければ ``modelUsage`` の名前）を ``reader_model`` にする。
    """
    if type(prompt) is not str or not prompt.strip():
        raise ReaderStage5Error("読む役に渡す指示文が空です")
    workdir = tempfile.mkdtemp(prefix="shokoban-s5-reader-")
    command = claude_command(binary, model=model)
    run = runner or subprocess.run
    try:
        completed = run(
            command,
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workdir,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReaderStage5Error(f"読む役（claude）を起動できません: {exc}") from exc
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    stdout = getattr(completed, "stdout", b"") or b""
    raw = stdout.decode("utf-8", "replace").strip() if isinstance(stdout, bytes) else str(stdout)
    if completed.returncode != 0:
        raise ReaderStage5Error(
            f"読む役（claude）が失敗しました（終了コード {completed.returncode}）"
        )
    if not raw:
        raise ReaderStage5Error("読む役（claude）の返答が空です")
    return _reply_from_payload(raw, completed.returncode)


def make_claude_reader(
    *,
    model: str,
    binary: str = DEFAULT_CLAUDE_BIN,
    timeout: int = DEFAULT_TIMEOUT,
    runner: Callable[..., Any] | None = None,
) -> Callable[[str], ReaderReply]:
    """本物の読む役（Claude Code CLI）を、``reader`` として渡せる関数にする.

    ``model`` は必須。1回の run の途中でモデルを替えないため、呼ぶ前に固定する。
    """

    def reader(prompt: str) -> ReaderReply:
        return run_claude(
            prompt, model=model, binary=binary, timeout=timeout, runner=runner
        )

    return reader
