"""手元AI（llama-server）を必要な時だけ起こし、使わなければ止める.

設定は ``runtime.json`` から読むだけ。待ち受けは 127.0.0.1 に固定する。

自分が起こしたプロセスの同一性は起動台帳 ``state_dir/llama-server.json``
（0600）だけで判断する。台帳には pid だけでなく、起動時刻
（``ps -o lstart=``）・起動した引数列・port・model・alias・起動ごとの乱数
token を残す。停止の直前には毎回「台帳の pid が今も同じ起動時刻で、同じ
引数（llama-server・127.0.0.1・port・model・alias）で動いているか」を確かめ、
1つでも合わなければシグナルを送らない。旧形式の pid/alias ファイルは読まずに
消す（止めもしない）。

確認と kill の間の競合は残る（macOS に pidfd が無いため）。そのため、
SIGKILL の直前にもう一度だけ同一性を確認し、確認できなければ送らない。

起動・切替・停止は ``state_dir/llama-server.lock`` の排他ロック
（``fcntl.flock(LOCK_EX)``）を取ってから行い、同じ Mac の別プロセス
（画面や自動停止のプロセスなど）の間で直列にする。取れるまで最大30秒待ち、
取れなければ ``LLMRuntimeError``。

停止は開始時に読んだ台帳の世代（token・pid・start_time）を固定し、以後の
再確認はその世代と完全一致する時だけ「まだ同じもの」とする。台帳が別の
世代に替わったら、新しい世代には何も送らず・台帳も消さずに戻る。

新しく起動した子の後始末（起動時刻が取れない・台帳を書けない・health が
来ない・同一性を確認できない）は、``popen`` が返した手綱
（``terminate`` / ``kill`` / ``wait``）だけで行い、数値の pid に
``os.kill`` を送らない。止められた確証が無いときは
``state_dir/llama-server.orphan.json`` に ``{pid, argv, at}`` を残し、
エラー文言に「止めきれなかった可能性」を含める。

テスト用の差し込み口（すべて既定は本物の実装）:
``ensure_running`` の ``popen`` / ``http_get`` / ``free_percent`` /
``port_in_use`` / ``process_info`` / ``kill`` / ``sleep`` / ``monotonic``、
``stop`` / ``is_running`` / ``current_ledger`` の ``process_info`` / ``kill`` /
``sleep``。``process_info(pid, field)`` は ``field``（``"lstart"`` か
``"command"``）の文字列を返し、その pid が居なければ None を返す。
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

LOOPBACK_HOST = "127.0.0.1"
LLAMA_SERVER_ENV = "SHOKOBAN_LLAMA_SERVER"
LLAMA_SERVER_FALLBACK = "/opt/homebrew/bin/llama-server"
RUNTIME_CONFIG_ENV = "SHOKOBAN_RUNTIME_CONFIG"
WORK_DIR_ENV = "SHOKOBAN_WORK_DIR"
REQUIRED_PORT = 1234

LEDGER_FILENAME = "llama-server.json"
LOCK_FILENAME = "llama-server.lock"
ORPHAN_FILENAME = "llama-server.orphan.json"
LEGACY_PID_FILENAME = "llama-server.pid"
LEGACY_ALIAS_FILENAME = "llama-server.alias"

_MIN_PORT = 1024
_MAX_PORT = 65535
_START_TIMEOUT_SECONDS = 120.0
_STOP_GRACE_SECONDS = 10.0
_STOP_POLL_SECONDS = 0.2
_HEALTH_POLL_SECONDS = 1.0
_PORT_RELEASE_SECONDS = 3.0
_HEALTH_TIMEOUT_SECONDS = 2.0
_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_POLL_SECONDS = 0.1

_CONFIG_KEYS = {
    "port",
    "ctx",
    "idle_minutes",
    "min_free_percent",
    "default_model",
    "models",
    "state_dir",
}

_MEMORY_PATTERN = re.compile(
    r"System-wide memory free percentage:\s*(\d+(?:\.\d+)?)\s*%"
)
_ALIAS_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")
_TOKEN_PATTERN = re.compile(r"[0-9a-f]{16}")

ProcessInfo = Callable[[int, str], "str | None"]


def _plain_ascii_path(text: str) -> bool:
    """空白を含まない、半角の英数字と記号だけのパスか.

    起動した llama-server が自分のものかは ``ps`` の出力（引数を空白でつないだ1行。
    半角以外の文字は別の記号列に置き換わる）で見分ける。パスに空白や日本語があると
    見分けられず、起動直後に「確認できない」で止まるので、設定を読む時点で断る。
    """
    return bool(text) and all(33 <= ord(ch) <= 126 for ch in text)


def llama_server_binary() -> str:
    """使う llama-server の場所を決める（呼ぶたびに環境から読む）.

    ``SHOKOBAN_LLAMA_SERVER`` があればそれ、無ければ PATH 上の ``llama-server``、
    それも無ければ ``/opt/homebrew/bin/llama-server``。場所に空白や半角以外の文字が
    あれば ``LLMRuntimeError``（起動したプロセスを見分けられないため）。
    """
    configured = os.environ.get(LLAMA_SERVER_ENV, "").strip()
    binary = configured or shutil.which("llama-server") or LLAMA_SERVER_FALLBACK
    if not _plain_ascii_path(binary):
        raise LLMRuntimeError(
            "llama-server の場所に空白や日本語などが含まれています。"
            "半角の英数字と記号だけの場所に置くか、SHOKOBAN_LLAMA_SERVER で指定してください"
        )
    return binary


def default_config_path() -> Path:
    """``runtime.json`` の場所を決める（呼ぶたびに環境から読む）.

    ``SHOKOBAN_RUNTIME_CONFIG`` があればそれ、無ければ ``SHOKOBAN_WORK_DIR/runtime.json``、
    それも無ければ ``~/.shokoban/runtime.json``。
    """
    configured = os.environ.get(RUNTIME_CONFIG_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    work_dir = os.environ.get(WORK_DIR_ENV, "").strip()
    if work_dir:
        return Path(work_dir).expanduser() / "runtime.json"
    return Path.home() / ".shokoban" / "runtime.json"


class LLMRuntimeError(RuntimeError):
    """手元AIを安全に扱えないときに投げる."""


class LLMRuntimeConfigError(LLMRuntimeError):
    """``runtime.json`` の内容が不正なときに投げる."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """``runtime.json`` の内容（host は 127.0.0.1 固定）."""

    port: int
    ctx: int
    idle_minutes: float
    min_free_percent: float
    default_model: str
    models: dict[str, Path]
    state_dir: Path
    host: str = LOOPBACK_HOST

    def __post_init__(self) -> None:
        if self.host != LOOPBACK_HOST:
            raise LLMRuntimeConfigError(
                "手元AIの待ち受けは 127.0.0.1 以外にできません"
            )
        if type(self.port) is not int or not (_MIN_PORT <= self.port <= _MAX_PORT):
            raise LLMRuntimeConfigError(
                f"port は {_MIN_PORT}〜{_MAX_PORT} の整数で指定してください"
            )
        if type(self.ctx) is not int or self.ctx <= 0:
            raise LLMRuntimeConfigError("ctx は正の整数で指定してください")
        if (
            type(self.idle_minutes) not in (int, float)
            or isinstance(self.idle_minutes, bool)
            or not self.idle_minutes > 0
        ):
            raise LLMRuntimeConfigError("idle_minutes は正の数で指定してください")
        if (
            type(self.min_free_percent) not in (int, float)
            or isinstance(self.min_free_percent, bool)
            or not 0 <= self.min_free_percent <= 100
        ):
            raise LLMRuntimeConfigError(
                "min_free_percent は 0〜100 の数で指定してください"
            )
        if type(self.default_model) is not str or not self.default_model:
            raise LLMRuntimeConfigError("default_model が空です")

        if type(self.models) is not dict or not self.models:
            raise LLMRuntimeConfigError("models が空です")
        normalized_models: dict[str, Path] = {}
        for alias, raw_path in self.models.items():
            if (
                type(alias) is not str
                or not _ALIAS_PATTERN.fullmatch(alias)
            ):
                raise LLMRuntimeConfigError(
                    f"models の別名が不正です: {alias!r}"
                )
            if not isinstance(raw_path, (str, Path)):
                raise LLMRuntimeConfigError(f"models[{alias}] がパスではありません")
            path = Path(raw_path)
            if not path.is_absolute():
                raise LLMRuntimeConfigError(
                    f"models[{alias}] は絶対パスで指定してください"
                )
            if path.suffix.lower() != ".gguf":
                raise LLMRuntimeConfigError(
                    f"models[{alias}] は .gguf を指定してください"
                )
            if not _plain_ascii_path(str(path)):
                raise LLMRuntimeConfigError(
                    f"models[{alias}] のパスに空白や日本語などが含まれています。"
                    "半角の英数字と記号だけの場所に置いてください"
                    "（起動した手元AIを見分けるために必要です）"
                )
            if not path.is_file():
                raise LLMRuntimeConfigError(
                    f"models[{alias}] のファイルがありません: {path}"
                )
            normalized_models[alias] = path
        if self.default_model not in normalized_models:
            raise LLMRuntimeConfigError(
                f"default_model「{self.default_model}」が models にありません"
            )

        if not isinstance(self.state_dir, (str, Path)):
            raise LLMRuntimeConfigError("state_dir がパスではありません")
        state_dir = Path(self.state_dir)
        if not state_dir.is_absolute():
            raise LLMRuntimeConfigError("state_dir は絶対パスで指定してください")

        object.__setattr__(self, "models", normalized_models)
        object.__setattr__(self, "state_dir", state_dir)

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    @property
    def ledger_path(self) -> Path:
        return self.state_dir / LEDGER_FILENAME

    @property
    def lock_path(self) -> Path:
        """起動・切替・停止を直列にする排他ロック（消さない）."""
        return self.state_dir / LOCK_FILENAME

    @property
    def orphan_path(self) -> Path:
        """手綱で止めきれなかった可能性があるプロセスの記録."""
        return self.state_dir / ORPHAN_FILENAME

    @property
    def pid_path(self) -> Path:
        """旧形式（読まない・使わない。互換のためだけに残す）."""
        return self.state_dir / LEGACY_PID_FILENAME

    @property
    def alias_path(self) -> Path:
        """旧形式（読まない・使わない。互換のためだけに残す）."""
        return self.state_dir / LEGACY_ALIAS_FILENAME

    @property
    def last_used_path(self) -> Path:
        return self.state_dir / "last_used"


@dataclass(frozen=True, slots=True)
class Ledger:
    """起動台帳の1件（自分が起こしたプロセスの個体識別情報）."""

    pid: int
    pgid: int
    start_time: str
    argv: tuple[str, ...]
    port: int
    model_path: str
    alias: str
    token: str


_LEDGER_KEYS = {
    "pid",
    "pgid",
    "start_time",
    "argv",
    "port",
    "model_path",
    "alias",
    "token",
}


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise LLMRuntimeConfigError(f"{label} は数で指定してください")
    return float(value)


def load_runtime_config(path: Path) -> RuntimeConfig:
    """``runtime.json`` を読む（作らない・直さない）.

    接続先（``prepare_local_llm``）が 1234 固定なので、port は 1234 以外を
    拒否する。設定だけ変えて別の番号で待ち受け、別の相手に繋ぐ事故を防ぐ。
    """
    selected = Path(path)
    if selected.is_symlink():
        raise LLMRuntimeConfigError("設定ファイルにシンボリックリンクは使えません")
    try:
        info = os.lstat(selected)
    except FileNotFoundError as exc:
        raise LLMRuntimeConfigError(f"設定ファイルがありません: {selected}") from exc
    except OSError as exc:
        raise LLMRuntimeConfigError("設定ファイルを読めません") from exc
    if not stat.S_ISREG(info.st_mode):
        raise LLMRuntimeConfigError("設定ファイルが通常ファイルではありません")

    try:
        parsed = json.loads(selected.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LLMRuntimeConfigError("設定ファイルが JSON として読めません") from exc
    if type(parsed) is not dict:
        raise LLMRuntimeConfigError("設定ファイルのルートがオブジェクトではありません")

    if "host" in parsed:
        raise LLMRuntimeConfigError(
            "host は設定から変えられません（常に 127.0.0.1）"
        )
    unknown = sorted(set(parsed) - _CONFIG_KEYS)
    if unknown:
        raise LLMRuntimeConfigError(
            "設定ファイルに知らない項目があります: " + ", ".join(unknown)
        )
    missing = sorted(_CONFIG_KEYS - set(parsed))
    if missing:
        raise LLMRuntimeConfigError(
            "設定ファイルに足りない項目があります: " + ", ".join(missing)
        )

    raw_models = parsed["models"]
    if type(raw_models) is not dict:
        raise LLMRuntimeConfigError("models がオブジェクトではありません")

    config = RuntimeConfig(
        port=parsed["port"],
        ctx=parsed["ctx"],
        idle_minutes=_number(parsed["idle_minutes"], "idle_minutes"),
        min_free_percent=_number(parsed["min_free_percent"], "min_free_percent"),
        default_model=parsed["default_model"],
        models=dict(raw_models),
        state_dir=parsed["state_dir"],
    )
    if config.port != REQUIRED_PORT:
        raise LLMRuntimeConfigError(
            f"port は {REQUIRED_PORT} 固定です（接続先が {REQUIRED_PORT} のため）"
        )
    return config


def _ensure_state_dir(directory: Path) -> None:
    """状態フォルダを 0700 で用意する（シンボリックリンクは拒否）."""
    try:
        if directory.is_symlink():
            raise LLMRuntimeError("状態フォルダにシンボリックリンクは使えません")
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
    except LLMRuntimeError:
        raise
    except OSError as exc:
        raise LLMRuntimeError("状態フォルダを用意できませんでした") from exc


def _write_state_text(path: Path, text: str) -> None:
    """状態ファイルを 0600 で置き換える（ディレクトリは 0700）."""
    parent = path.parent
    _ensure_state_dir(parent)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            os.chmod(temporary_path, 0o600)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise LLMRuntimeError("手元AIの状態を保存できませんでした") from exc


@contextlib.contextmanager
def _state_lock(
    cfg: RuntimeConfig,
    *,
    sleep: Callable[[float], Any] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
):
    """``state_dir/llama-server.lock`` の排他ロックを取る（最大30秒待ち）.

    同じ Mac の別プロセス（画面や自動停止のプロセスなど）の間で、起動・
    切替・停止を直列にする。取れなければ ``LLMRuntimeError``。
    """
    path = cfg.lock_path
    _ensure_state_dir(path.parent)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise LLMRuntimeError("手元AIの操作の準備ができませんでした") from exc
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except OSError:
            pass
        deadline = monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if monotonic() >= deadline:
                    raise LLMRuntimeError(
                        "手元AIの操作が混み合っています。少し待ってからもう一度。"
                    ) from None
                sleep(_LOCK_POLL_SECONDS)
            except OSError as exc:
                raise LLMRuntimeError(
                    "手元AIの操作の準備ができませんでした"
                ) from exc
        try:
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(descriptor)


def _read_state_text(path: Path) -> str | None:
    """状態ファイルを読む（通常ファイル以外・読めない場合は None）."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _remove_state_file(path: Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _state_file_exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except OSError:
        return False
    return True


def _discard_legacy_state(cfg: RuntimeConfig) -> None:
    """旧形式の pid/alias ファイルは読まずに消す（止めもしない）."""
    for path in (cfg.pid_path, cfg.alias_path):
        if _state_file_exists(path):
            _remove_state_file(path)


def _ledger_from_payload(payload: Any) -> Ledger:
    if type(payload) is not dict or set(payload) != _LEDGER_KEYS:
        raise LLMRuntimeError("手元AIの起動台帳の形が不正なので中止します")
    pid = payload["pid"]
    pgid = payload["pgid"]
    start_time = payload["start_time"]
    argv = payload["argv"]
    port = payload["port"]
    model_path = payload["model_path"]
    alias = payload["alias"]
    token = payload["token"]
    if type(pid) is not int or not (0 < pid <= 2**31 - 1):
        raise LLMRuntimeError("手元AIの起動台帳の pid が不正なので中止します")
    if type(pgid) is not int or not (0 < pgid <= 2**31 - 1):
        raise LLMRuntimeError("手元AIの起動台帳の pgid が不正なので中止します")
    if type(start_time) is not str or not start_time:
        raise LLMRuntimeError("手元AIの起動台帳の起動時刻が不正なので中止します")
    if (
        type(argv) is not list
        or not argv
        or any(type(value) is not str or not value for value in argv)
    ):
        raise LLMRuntimeError("手元AIの起動台帳の引数が不正なので中止します")
    if type(port) is not int or not (_MIN_PORT <= port <= _MAX_PORT):
        raise LLMRuntimeError("手元AIの起動台帳の port が不正なので中止します")
    if type(model_path) is not str or not model_path:
        raise LLMRuntimeError("手元AIの起動台帳の model が不正なので中止します")
    if type(alias) is not str or not _ALIAS_PATTERN.fullmatch(alias):
        raise LLMRuntimeError("手元AIの起動台帳の alias が不正なので中止します")
    if type(token) is not str or not _TOKEN_PATTERN.fullmatch(token):
        raise LLMRuntimeError("手元AIの起動台帳の token が不正なので中止します")
    return Ledger(
        pid=pid,
        pgid=pgid,
        start_time=start_time,
        argv=tuple(argv),
        port=port,
        model_path=model_path,
        alias=alias,
        token=token,
    )


def _read_ledger(cfg: RuntimeConfig) -> Ledger | None:
    """起動台帳を読む。無ければ None、壊れていれば LLMRuntimeError（止めない）."""
    path = cfg.ledger_path
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise LLMRuntimeError("手元AIの起動台帳を確認できないので中止します") from exc
    if not stat.S_ISREG(info.st_mode):
        raise LLMRuntimeError("手元AIの起動台帳が通常ファイルではないので中止します")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LLMRuntimeError("手元AIの起動台帳を読めないので中止します") from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise LLMRuntimeError("手元AIの起動台帳が壊れているので中止します") from exc
    return _ledger_from_payload(payload)


def _write_ledger(cfg: RuntimeConfig, ledger: Ledger) -> None:
    payload = {
        "pid": ledger.pid,
        "pgid": ledger.pgid,
        "start_time": ledger.start_time,
        "argv": list(ledger.argv),
        "port": ledger.port,
        "model_path": ledger.model_path,
        "alias": ledger.alias,
        "token": ledger.token,
    }
    _write_state_text(
        cfg.ledger_path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _clear_ledger(cfg: RuntimeConfig) -> None:
    _remove_state_file(cfg.ledger_path)
    _remove_state_file(cfg.last_used_path)


def touch_used(cfg: RuntimeConfig, now: Callable[[], float] = time.time) -> None:
    """最終使用時刻を記録する."""
    _write_state_text(cfg.last_used_path, f"{float(now()):.6f}\n")


def idle_seconds(
    cfg: RuntimeConfig, now: Callable[[], float] = time.time
) -> float | None:
    """最終使用からの経過秒（記録が無ければ None）."""
    text = _read_state_text(cfg.last_used_path)
    if text is None:
        return None
    try:
        recorded = float(text.strip())
    except ValueError:
        return None
    return max(0.0, float(now()) - recorded)


def _default_http_get(url: str) -> bool:
    """``GET /health`` が ok なら True。つながらなければ False."""
    try:
        with urllib.request.urlopen(url, timeout=_HEALTH_TIMEOUT_SECONDS) as response:
            body = response.read(4096)
    except Exception:
        return False
    try:
        payload = json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        return body.strip().lower() == b"ok"
    return type(payload) is dict and payload.get("status") == "ok"


def _health_ok(
    cfg: RuntimeConfig, http_get: Callable[[str], Any]
) -> bool:
    try:
        result = http_get(cfg.health_url)
    except Exception:
        return False
    if type(result) is bool:
        return result
    if type(result) is dict:
        return result.get("status") == "ok"
    if type(result) is str:
        return result.strip().lower() == "ok"
    return bool(result)


def _default_free_percent() -> float:
    """``memory_pressure -Q`` から空きメモリ％を読む."""
    binary = shutil.which("memory_pressure") or "/usr/bin/memory_pressure"
    try:
        completed = subprocess.run(
            [binary, "-Q"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        raise LLMRuntimeError(
            "空きメモリを確認できないので手元AIを起動しません"
        ) from exc
    haystack = (completed.stdout or "") + "\n" + (completed.stderr or "")
    match = _MEMORY_PATTERN.search(haystack)
    if match is None:
        raise LLMRuntimeError(
            "空きメモリを確認できないので手元AIを起動しません"
        )
    return float(match.group(1))


def _default_port_in_use(port: int) -> bool:
    """127.0.0.1:port に誰かが待ち受けているか."""
    try:
        with socket.create_connection((LOOPBACK_HOST, port), timeout=0.5):
            return True
    except OSError:
        return False


def _default_process_field(pid: int, field: str) -> str | None:
    """``ps`` で pid の起動時刻（lstart）か実行コマンド（command）を引く.

    その pid が居なければ None。``ps`` 自体を実行できないときは
    ``LLMRuntimeError``（判別できないまま kill しないため）。
    """
    if field not in ("lstart", "command"):
        raise LLMRuntimeError("手元AIのプロセス確認の種類が不正です")
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            ["/bin/ps", "-ww", "-o", f"{field}=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env=environment,
        )
    except Exception as exc:
        raise LLMRuntimeError("手元AIのプロセスを確認できないので中止します") from exc
    if completed.returncode != 0:
        return None
    text = (completed.stdout or "").strip()
    return text or None


def _process_field(
    process_info: ProcessInfo, pid: int, field: str
) -> str | None:
    try:
        value = process_info(pid, field)
    except LLMRuntimeError:
        raise
    except Exception as exc:
        raise LLMRuntimeError("手元AIのプロセスを確認できないので中止します") from exc
    if value is None:
        return None
    if type(value) is not str:
        raise LLMRuntimeError("手元AIのプロセス確認の応答が不正なので中止します")
    return value


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _has_pair(argv: list[str], flag: str, value: str) -> bool:
    return any(
        argv[index] == flag and argv[index + 1] == value
        for index in range(len(argv) - 1)
    )


def _command_matches(argv: list[str], ledger: Ledger) -> bool:
    """引数列が台帳の起動内容と一致するか（1つでも欠ければ False）."""
    if not argv:
        return False
    if llama_server_binary() not in argv and Path(argv[0]).name != "llama-server":
        return False
    expected = (
        ("--host", LOOPBACK_HOST),
        ("--port", str(ledger.port)),
        ("-m", ledger.model_path),
        ("--alias", ledger.alias),
    )
    return all(_has_pair(argv, flag, value) for flag, value in expected)


def _process_state(
    ledger: Ledger, process_info: ProcessInfo | None = None
) -> str:
    """台帳の1世代が今も同じ個体か調べる（ps の再照会だけ）.

    ``gone`` プロセスが居ない / ``foreign`` 起動時刻か引数が違う /
    ``unverifiable`` プロセスは居るが引数を確認できない / ``owned`` 一致。
    """
    lookup = _default_process_field if process_info is None else process_info
    start_time = _process_field(lookup, ledger.pid, "lstart")
    if start_time is None:
        return "gone"
    if start_time != ledger.start_time:
        return "foreign"
    command = _process_field(lookup, ledger.pid, "command")
    if command is None:
        # 起動時刻を引いた直後にプロセスが終わった（止める途中によく起きる）。
        # もう一度起動時刻を引いて、居なければ「居ない」とみなす。
        if _process_field(lookup, ledger.pid, "lstart") is None:
            return "gone"
        return "unverifiable"
    if not _command_matches(_split_command(command), ledger):
        return "foreign"
    return "owned"


def _inspect_process(
    cfg: RuntimeConfig, *, process_info: ProcessInfo | None = None
) -> tuple[str, Ledger | None]:
    """台帳の pid が今も自分の llama-server かを調べる.

    戻り値の状態:
    ``none`` 台帳が無い / ``gone`` プロセスが居ない /
    ``foreign`` 別のプロセス（起動時刻か引数が違う） /
    ``unverifiable`` プロセスは居るが引数を確認できない /
    ``owned`` 台帳と一致する自分のプロセス。
    """
    ledger = _read_ledger(cfg)
    if ledger is None:
        return "none", None
    return _process_state(ledger, process_info), ledger


def _generation(ledger: Ledger) -> tuple[str, int, str]:
    """台帳の世代（停止中に乗り換えてはいけない識別子）."""
    return (ledger.token, ledger.pid, ledger.start_time)


def _read_generation(
    cfg: RuntimeConfig, generation: tuple[str, int, str]
) -> tuple[str, Ledger | None]:
    """台帳が固定した世代のままか読む.

    ``same`` 同じ世代 / ``missing`` 台帳が無い / ``replaced`` 別の世代。
    """
    ledger = _read_ledger(cfg)
    if ledger is None:
        return "missing", None
    if _generation(ledger) != generation:
        return "replaced", None
    return "same", ledger


def _clear_ledger_if_generation(
    cfg: RuntimeConfig, generation: tuple[str, int, str]
) -> bool:
    """台帳がまだ指定の世代の時だけ消す（新しい世代は消さない）."""
    try:
        ledger = _read_ledger(cfg)
    except LLMRuntimeError:
        return False
    if ledger is None or _generation(ledger) != generation:
        return False
    _clear_ledger(cfg)
    return True


def _owned_process(
    cfg: RuntimeConfig, *, process_info: ProcessInfo | None = None
) -> Ledger | None:
    """自分が起こしたプロセスの台帳。1つでも合わなければ None."""
    state, ledger = _inspect_process(cfg, process_info=process_info)
    return ledger if state == "owned" else None


def current_ledger(
    cfg: RuntimeConfig, *, process_info: ProcessInfo | None = None
) -> Ledger | None:
    """表示用: 自分の llama-server の台帳（合わなければ None）."""
    if not isinstance(cfg, RuntimeConfig):
        raise LLMRuntimeError("手元AIの設定が不正です")
    _discard_legacy_state(cfg)
    return _owned_process(cfg, process_info=process_info)


def _format_percent(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def stop(
    cfg: RuntimeConfig,
    *,
    kill: Callable[[int, int], Any] = os.kill,
    process_info: ProcessInfo | None = None,
    sleep: Callable[[float], Any] = time.sleep,
) -> bool:
    """台帳で自分のものと確認できた llama-server だけを止める.

    排他ロックを取ってから、開始時に読んだ台帳の世代（token・pid・
    start_time）に固定して止める。台帳が別の世代に替わったら、新しい世代
    には何も送らず・台帳も消さずに False を返す。確認できないときは
    シグナルを送らず ``LLMRuntimeError``。
    """
    if not isinstance(cfg, RuntimeConfig):
        raise LLMRuntimeError("手元AIの設定が不正です")
    with _state_lock(cfg, sleep=sleep):
        return _stop_locked(
            cfg, kill=kill, process_info=process_info, sleep=sleep
        )


def _stop_locked(
    cfg: RuntimeConfig,
    *,
    kill: Callable[[int, int], Any],
    process_info: ProcessInfo | None,
    sleep: Callable[[float], Any],
) -> bool:
    """排他ロックを持った状態で止める（``stop`` と ``ensure_running`` 用）."""
    _discard_legacy_state(cfg)

    state, ledger = _inspect_process(cfg, process_info=process_info)
    if state == "none":
        _remove_state_file(cfg.last_used_path)
        return False
    if state == "gone":
        if ledger is not None:
            _clear_ledger_if_generation(cfg, _generation(ledger))
        return False
    if state != "owned" or ledger is None:
        raise LLMRuntimeError("手元AIの停止対象を確認できないので止めません")

    generation = _generation(ledger)
    try:
        kill(ledger.pid, signal.SIGTERM)
    except ProcessLookupError:
        _clear_ledger_if_generation(cfg, generation)
        return False
    except OSError as exc:
        raise LLMRuntimeError("手元AIを止められませんでした") from exc

    polls = max(1, int(_STOP_GRACE_SECONDS / _STOP_POLL_SECONDS))
    for _ in range(polls):
        status, current = _read_generation(cfg, generation)
        if status != "same" or current is None:
            # 台帳が消えた・別の世代に替わった。新しい世代には何も送らず、
            # 台帳も消さずに戻る（戻り値 False）。
            return False
        state = _process_state(current, process_info)
        if state in ("gone", "foreign"):
            # 消えた、または同じ pid が別プロセスに替わった。どちらも
            # 自分が起こしたプロセスはもう居ないので、それ以上は送らない。
            _clear_ledger_if_generation(cfg, generation)
            return True
        if state == "unverifiable":
            raise LLMRuntimeError(
                "手元AIの停止を確認できないので、それ以上は止めません"
            )
        sleep(_STOP_POLL_SECONDS)

    # SIGKILL の直前にもう一度だけ、同じ世代かどうかを確認する。
    status, current = _read_generation(cfg, generation)
    if status != "same" or current is None:
        return False
    state = _process_state(current, process_info)
    if state == "owned":
        try:
            kill(current.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise LLMRuntimeError("手元AIを止められませんでした") from exc
        _clear_ledger_if_generation(cfg, generation)
        return True
    if state in ("gone", "foreign"):
        _clear_ledger_if_generation(cfg, generation)
        return True
    raise LLMRuntimeError("手元AIの停止対象を確認できないので止めません")


def is_running(
    cfg: RuntimeConfig,
    *,
    process_info: ProcessInfo | None = None,
) -> bool:
    """自分が起こした llama-server が台帳どおりに動いているか.

    台帳が壊れている・確認できないときは False（止めない側に倒す）。
    """
    if not isinstance(cfg, RuntimeConfig):
        raise LLMRuntimeError("手元AIの設定が不正です")
    _discard_legacy_state(cfg)
    try:
        return _owned_process(cfg, process_info=process_info) is not None
    except LLMRuntimeError:
        return False


def _wait_port_free(
    port: int,
    port_in_use: Callable[[int], bool],
    sleep: Callable[[float], Any],
    monotonic: Callable[[], float],
) -> bool:
    if not port_in_use(port):
        return True
    deadline = monotonic() + _PORT_RELEASE_SECONDS
    while monotonic() < deadline:
        sleep(0.1)
        if not port_in_use(port):
            return True
    return False


def _build_command(cfg: RuntimeConfig, alias: str) -> list[str]:
    return [
        llama_server_binary(),
        "--host",
        LOOPBACK_HOST,
        "--port",
        str(cfg.port),
        "--alias",
        alias,
        "-m",
        str(cfg.models[alias]),
        "-c",
        str(cfg.ctx),
        "--no-webui",
        "--reasoning",
        "off",
    ]


def ensure_running(
    cfg: RuntimeConfig,
    model_alias: str | None = None,
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    http_get: Callable[[str], Any] = _default_http_get,
    free_percent: Callable[[], float] = _default_free_percent,
    now: Callable[[], float] = time.time,
    port_in_use: Callable[[int], bool] = _default_port_in_use,
    process_info: ProcessInfo | None = None,
    kill: Callable[[int, int], Any] = os.kill,
    sleep: Callable[[float], Any] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> str:
    """手元AIを使える状態にして別名を返す（必要なら起こす）.

    起動・切替・停止は排他ロックの中で行う。health が ok でも、子が生きて
    いて起動台帳がちょうど今書いた世代を指していなければ成功にしない。
    """
    if not isinstance(cfg, RuntimeConfig):
        raise LLMRuntimeError("手元AIの設定が不正です")
    wanted = cfg.default_model if model_alias is None else model_alias
    if type(wanted) is not str or not wanted:
        raise LLMRuntimeError("手元AIのモデル名が不正です")
    if wanted not in cfg.models:
        raise LLMRuntimeError(f"手元AIのモデル「{wanted}」は設定にありません")

    with _state_lock(cfg, sleep=sleep, monotonic=monotonic):
        return _ensure_running_locked(
            cfg,
            wanted,
            popen=popen,
            http_get=http_get,
            free_percent=free_percent,
            now=now,
            port_in_use=port_in_use,
            process_info=process_info,
            kill=kill,
            sleep=sleep,
            monotonic=monotonic,
        )


def _ensure_running_locked(
    cfg: RuntimeConfig,
    wanted: str,
    *,
    popen: Callable[..., Any],
    http_get: Callable[[str], Any],
    free_percent: Callable[[], float],
    now: Callable[[], float],
    port_in_use: Callable[[int], bool],
    process_info: ProcessInfo | None,
    kill: Callable[[int, int], Any],
    sleep: Callable[[float], Any],
    monotonic: Callable[[], float],
) -> str:
    """排他ロックを持った状態で ``ensure_running`` の本体を行う."""
    _discard_legacy_state(cfg)
    state, owned = _inspect_process(cfg, process_info=process_info)
    healthy = _health_ok(cfg, http_get)

    if state == "owned" and owned is not None:
        if owned.port == cfg.port and owned.alias == wanted:
            if healthy:
                touch_used(cfg, now=now)
                return wanted
        _stop_locked(
            cfg, kill=kill, process_info=process_info, sleep=sleep
        )
    else:
        if state == "unverifiable":
            raise LLMRuntimeError("手元AIの状態を確認できないので中止します")
        if healthy or port_in_use(cfg.port):
            raise LLMRuntimeError(
                "ほかのプログラムが手元AIの番号を使っています。止めずに中止します。"
            )
        # 台帳が無い・プロセスが消えた・別人に替わった。どれも kill しない。
        if owned is not None:
            _clear_ledger_if_generation(cfg, _generation(owned))
        else:
            _remove_state_file(cfg.last_used_path)

    if not _wait_port_free(cfg.port, port_in_use, sleep, monotonic):
        raise LLMRuntimeError(
            f"ポート{cfg.port}は別のプロセスが使っているので手元AIを起動しません"
        )

    free = float(free_percent())
    if free < cfg.min_free_percent:
        raise LLMRuntimeError(
            "メモリが足りないので手元AIを起動しません"
            f"（空き{_format_percent(free)}%）"
        )

    command = _build_command(cfg, wanted)
    try:
        process = popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise LLMRuntimeError(
            f"llama-server が見つかりません（{command[0]}）。"
            f"llama-server を入れて PATH に通すか、環境変数 {LLAMA_SERVER_ENV} に場所を指定してください"
        ) from exc
    except Exception as exc:
        raise LLMRuntimeError("手元AIを起動できませんでした") from exc

    new_pid = getattr(process, "pid", None)
    if type(new_pid) is not int or new_pid <= 0:
        confirmed = _abort_started_process(cfg, process, command, now=now)
        raise LLMRuntimeError(
            _startup_failure("手元AIのプロセスIDを取得できませんでした", confirmed)
        )

    lookup = _default_process_field if process_info is None else process_info
    try:
        start_time = _process_field(lookup, new_pid, "lstart")
    except LLMRuntimeError as exc:
        confirmed = _abort_started_process(cfg, process, command, now=now)
        raise LLMRuntimeError(
            _startup_failure(
                "起動した手元AIの起動時刻を確認できないので中止します", confirmed
            )
        ) from exc
    if start_time is None:
        confirmed = _abort_started_process(cfg, process, command, now=now)
        raise LLMRuntimeError(
            _startup_failure(
                "起動した手元AIの起動時刻を確認できないので中止します", confirmed
            )
        )

    # start_new_session=True なので、子は自分自身がプロセスグループ長。
    ledger = Ledger(
        pid=new_pid,
        pgid=new_pid,
        start_time=start_time,
        argv=tuple(command),
        port=cfg.port,
        model_path=str(cfg.models[wanted]),
        alias=wanted,
        token=secrets.token_hex(8),
    )
    try:
        _write_ledger(cfg, ledger)
    except LLMRuntimeError as exc:
        # 台帳に残せないプロセスは管理できない。起動したばかりの自分だけを、
        # popen の手綱だけで止める（数値の pid には送らない）。
        confirmed = _abort_started_process(cfg, process, command, now=now)
        raise LLMRuntimeError(_startup_failure(str(exc), confirmed)) from exc

    healthy = False
    deadline = monotonic() + _START_TIMEOUT_SECONDS
    while monotonic() < deadline:
        if _health_ok(cfg, http_get):
            healthy = True
            break
        sleep(_HEALTH_POLL_SECONDS)
    if not healthy:
        confirmed = _abort_started_process(cfg, process, command, now=now)
        if confirmed:
            _clear_ledger_if_generation(cfg, _generation(ledger))
        raise LLMRuntimeError(
            _startup_failure("手元AIが120秒以内に応答しませんでした", confirmed)
        )

    # health が ok でも、子が生きていて、台帳がちょうど今書いた世代を
    # 指している時だけ成功にする。
    owned: Ledger | None = None
    if _child_is_alive(process):
        try:
            owned = _owned_process(cfg, process_info=process_info)
        except LLMRuntimeError:
            owned = None
    if owned is not None and _generation(owned) == _generation(ledger):
        try:
            touch_used(cfg, now=now)
        except LLMRuntimeError as exc:
            confirmed = _abort_started_process(cfg, process, command, now=now)
            if confirmed:
                _clear_ledger_if_generation(cfg, _generation(ledger))
            raise LLMRuntimeError(_startup_failure(str(exc), confirmed)) from exc
        return wanted

    confirmed = _abort_started_process(cfg, process, command, now=now)
    if confirmed:
        _clear_ledger_if_generation(cfg, _generation(ledger))
    raise LLMRuntimeError(
        _startup_failure("起動した手元AIを確認できないので中止します", confirmed)
    )


def _child_is_alive(process: Any) -> bool:
    """``popen`` が返した子がまだ生きているか（手綱の ``poll`` だけ）."""
    poll = getattr(process, "poll", None)
    if not callable(poll):
        return False
    try:
        return poll() is None
    except Exception:
        return False


def _wait_for_exit(
    process: Any, timeout: float = _STOP_GRACE_SECONDS
) -> bool:
    """手綱の ``wait`` で終了を確認する（確認できなければ False）."""
    wait = getattr(process, "wait", None)
    if not callable(wait):
        return False
    try:
        wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    return True


def _abort_started_process(
    cfg: RuntimeConfig,
    process: Any,
    command: list[str],
    *,
    now: Callable[[], float] = time.time,
) -> bool:
    """起動直後の子を ``popen`` の手綱だけで止める.

    数値の pid に ``os.kill`` は送らない。``terminate`` → ``wait``、
    だめなら ``kill`` → ``wait``。止められた確証が無ければ
    ``state_dir/llama-server.orphan.json`` に残して False を返す。
    """
    try:
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            terminate()
    except Exception:
        pass
    if _wait_for_exit(process):
        return True

    try:
        killer = getattr(process, "kill", None)
        if callable(killer):
            killer()
    except Exception:
        pass
    if _wait_for_exit(process):
        return True

    _record_orphan(cfg, getattr(process, "pid", None), command, now=now)
    return False


def _record_orphan(
    cfg: RuntimeConfig,
    pid: Any,
    command: list[str],
    *,
    now: Callable[[], float] = time.time,
) -> None:
    """手綱で止めきれなかった可能性を残す（best effort）."""
    try:
        at = datetime.fromtimestamp(float(now())).isoformat(timespec="seconds")
    except Exception:
        at = ""
    payload = {"pid": pid, "argv": list(command), "at": at}
    try:
        _write_state_text(
            cfg.orphan_path,
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        )
    except LLMRuntimeError:
        pass


def _startup_failure(message: str, confirmed: bool) -> str:
    """起動失敗の文言。止められた確証が無ければ可能性を添える."""
    if confirmed:
        return message
    return (
        message
        + "（起動した手元AIを止めきれなかった可能性があります。"
        "状態フォルダの llama-server.orphan.json を確認してください）"
    )
