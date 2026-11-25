"""手元AIの起動・停止（llm_runtime）の試験.

本物の llama-server は一切起動しない。popen・http・kill・ps は全部偽物。
起動台帳（state_dir/llama-server.json）に残した個体識別情報と、
「1つでも合わなければ止めない」ことを中心に試す。
"""

import contextlib
import fcntl
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shokoban import llm_runtime
from shokoban.llm_runtime import (
    llama_server_binary,
    LLMRuntimeConfigError,
    LLMRuntimeError,
    RuntimeConfig,
    ensure_running,
    idle_seconds,
    is_running,
    load_runtime_config,
    stop,
    touch_used,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "llm_idle_stop.py"

LSTART = "Mon Nov  9 12:00:00 2026"
NEW_LSTART = "Mon Nov  9 12:05:00 2026"
LEDGER_NAME = "llama-server.json"


def load_idle_stop_script():
    spec = importlib.util.spec_from_file_location("llm_idle_stop_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


idle_stop = load_idle_stop_script()


class FakeProcess:
    """偽のプロセス（起動時刻と実行コマンド）."""

    def __init__(self, command, lstart=LSTART):
        self.command = command
        self.lstart = lstart


class FakeProcessTable:
    """pid -> FakeProcess の偽物（ps と kill の代わり）."""

    def __init__(self, alive=None, on_kill=None):
        self.alive = dict(alive or {})
        self.on_kill = on_kill
        self.kill_calls = []

    def __call__(self, pid, field):
        process = self.alive.get(pid)
        if process is None:
            return None
        if field == "lstart":
            return process.lstart
        if field == "command":
            return process.command
        raise AssertionError(f"unexpected ps field: {field}")

    def kill(self, pid, sig):
        self.kill_calls.append((pid, sig))
        if self.on_kill is not None:
            self.on_kill(self, pid, sig)
        else:
            self.alive.pop(pid, None)


class FakeStartedProcess:
    """``popen`` が返す手綱（terminate/kill/wait/poll）の偽物."""

    def __init__(self, pid, table=None, *, unkillable=False):
        self.pid = pid
        self.table = table
        self.unkillable = unkillable
        self.returncode = None
        self.terminate_calls = []
        self.kill_calls = []
        self.wait_calls = []
        self.poll_calls = 0

    def _alive(self):
        if self.returncode is not None:
            return False
        if self.table is None:
            return True
        return self.pid in self.table.alive

    def poll(self):
        self.poll_calls += 1
        if self._alive():
            return None
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminate_calls.append(self.pid)
        if self.unkillable:
            return
        if self.table is not None:
            self.table.alive.pop(self.pid, None)
        else:
            self.returncode = -signal.SIGTERM

    def kill(self):
        self.kill_calls.append(self.pid)
        if self.unkillable:
            return
        if self.table is not None:
            self.table.alive.pop(self.pid, None)
        else:
            self.returncode = -signal.SIGKILL

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self._alive():
            raise subprocess.TimeoutExpired(
                "llama-server", 0 if timeout is None else timeout
            )
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class FakePopen:
    def __init__(
        self, calls, pid=4321, on_call=None, table=None, lstart=NEW_LSTART,
        unkillable=False,
    ):
        self.calls = calls
        self.pid = pid
        self.on_call = on_call
        self.table = table
        self.lstart = lstart
        self.unkillable = unkillable
        self.handle = None

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), dict(kwargs)))
        if self.table is not None:
            self.table.alive[self.pid] = FakeProcess(" ".join(command), self.lstart)
        self.handle = FakeStartedProcess(
            self.pid, self.table, unkillable=self.unkillable
        )
        if self.on_call is not None:
            self.on_call()
        return self.handle


class SwitchHealth:
    """最初は不健康、起こした後だけ健康になる偽の health."""

    def __init__(self, healthy=False, urls=None):
        self.healthy = healthy
        self.urls = urls if urls is not None else []

    def __call__(self, url):
        self.urls.append(url)
        return self.healthy


class RuntimeTestCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"SHOKOBAN_LLAMA_SERVER": "/opt/test/bin/llama-server"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = Path(tempfile.mkdtemp(prefix="llm_runtime_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        models_dir = self.tmp / "models"
        models_dir.mkdir()
        models = {}
        for alias, filename in (
            ("model-f", "model-f-q4_0.gguf"),
            ("model-e-1b", "MODEL-E-1B-Q4_K_M.gguf"),
        ):
            path = models_dir / filename
            path.write_bytes(b"GGUF-fake")
            models[alias] = str(path)
        self.models = models
        self.state_dir = self.tmp / "runtime-state"
        self.cfg = RuntimeConfig(
            port=1234,
            ctx=8192,
            idle_minutes=10,
            min_free_percent=35,
            default_model="model-f",
            models=models,
            state_dir=self.state_dir,
        )

    @property
    def ledger_path(self) -> Path:
        return self.state_dir / LEDGER_NAME

    def write_legacy_state(self, pid=4242, alias="model-f"):
        """旧形式の pid/alias ファイル（読まれず、消されるべきもの）."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "llama-server.pid").write_text(f"{pid}\n", encoding="utf-8")
        (self.state_dir / "llama-server.alias").write_text(alias + "\n", encoding="utf-8")

    def hold_state_lock(self):
        """別プロセスが持っている排他ロックの代わり（同じファイルの別 fd）."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.cfg.lock_path, os.O_RDWR | os.O_CREAT, 0o600
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        self.addCleanup(os.close, descriptor)
        return descriptor

    def ledger_argv(self, *, alias=None, port=1234, model_path=None):
        alias = self.cfg.default_model if alias is None else alias
        model_path = self.models.get(alias, model_path) if model_path is None else model_path
        return [
            llama_server_binary(),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--alias",
            alias,
            "-m",
            model_path,
            "-c",
            "8192",
            "--no-webui",
            "--reasoning",
            "off",
        ]

    def write_ledger(
        self,
        *,
        pid=4242,
        alias=None,
        port=1234,
        start_time=LSTART,
        model_path=None,
        pgid=None,
        token="0123456789abcdef",
        argv=None,
    ):
        alias = self.cfg.default_model if alias is None else alias
        model_path = (
            self.models.get(alias, str(self.tmp / "models" / f"{alias}.gguf"))
            if model_path is None
            else model_path
        )
        argv = (
            self.ledger_argv(alias=alias, port=port, model_path=model_path)
            if argv is None
            else list(argv)
        )
        payload = {
            "pid": pid,
            "pgid": pid if pgid is None else pgid,
            "start_time": start_time,
            "argv": argv,
            "port": port,
            "model_path": model_path,
            "alias": alias,
            "token": token,
        }
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(self.ledger_path, 0o600)
        return payload

    def read_ledger(self):
        return json.loads(self.ledger_path.read_text(encoding="utf-8"))

    def table_for(self, payload, *, command=None, lstart=None, on_kill=None):
        return FakeProcessTable(
            {
                payload["pid"]: FakeProcess(
                    command if command is not None else " ".join(payload["argv"]),
                    payload["start_time"] if lstart is None else lstart,
                )
            },
            on_kill=on_kill,
        )

    def ensure(self, *args, **kwargs):
        ticks = {"value": 0.0}

        def monotonic():
            ticks["value"] += 1.0
            return ticks["value"]

        kwargs.setdefault("popen", FakePopen([]))
        kwargs.setdefault("http_get", lambda url: False)
        kwargs.setdefault("free_percent", lambda: 50.0)
        kwargs.setdefault("port_in_use", lambda port: False)
        kwargs.setdefault("process_info", FakeProcessTable())
        kwargs.setdefault("kill", FakeProcessTable().kill)
        kwargs.setdefault("sleep", lambda seconds: None)
        kwargs.setdefault("monotonic", monotonic)
        kwargs.setdefault("now", lambda: 1000.0)
        return ensure_running(self.cfg, *args, **kwargs)


class RuntimeConfigTest(RuntimeTestCase):
    def write_config(self, **overrides):
        payload = {
            "port": 1234,
            "ctx": 8192,
            "idle_minutes": 10,
            "min_free_percent": 35,
            "default_model": "model-f",
            "models": self.models,
            "state_dir": str(self.state_dir),
        }
        payload.update(overrides)
        path = self.tmp / "runtime.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_loads_valid_config_with_loopback_host(self):
        cfg = load_runtime_config(self.write_config())
        self.assertEqual(cfg.port, 1234)
        self.assertEqual(cfg.ctx, 8192)
        self.assertEqual(cfg.idle_minutes, 10)
        self.assertEqual(cfg.min_free_percent, 35)
        self.assertEqual(cfg.default_model, "model-f")
        self.assertEqual(cfg.host, "127.0.0.1")
        self.assertEqual(
            cfg.models["model-e-1b"], Path(self.models["model-e-1b"])
        )
        self.assertEqual(cfg.state_dir, self.state_dir)
        self.assertEqual(cfg.health_url, "http://127.0.0.1:1234/health")
        self.assertEqual(cfg.ledger_path, self.state_dir / LEDGER_NAME)

    def test_host_cannot_be_changed(self):
        with self.assertRaises(LLMRuntimeConfigError) as caught:
            load_runtime_config(self.write_config(host="0.0.0.0"))
        self.assertIn("127.0.0.1", str(caught.exception))
        with self.assertRaises(LLMRuntimeConfigError):
            RuntimeConfig(
                port=1234,
                ctx=8192,
                idle_minutes=10,
                min_free_percent=35,
                default_model="model-f",
                models=self.models,
                state_dir=self.state_dir,
                host="0.0.0.0",
            )

    def test_rejects_bad_port(self):
        for port in (80, 1023, 65536, 0, -1, True, "1234", 1234.0):
            with self.subTest(port=port), self.assertRaises(LLMRuntimeConfigError):
                load_runtime_config(self.write_config(port=port))

    def test_rejects_port_other_than_1234(self):
        # 接続先（prepare_local_llm）が 1234 固定なので、別の番号は受け付けない。
        for port in (1235, 4321, 65535):
            with self.subTest(port=port), self.assertRaises(LLMRuntimeConfigError) as caught:
                load_runtime_config(self.write_config(port=port))
            self.assertIn("1234", str(caught.exception))

    def test_rejects_bad_numbers(self):
        for key, value in (
            ("ctx", 0),
            ("ctx", "8192"),
            ("idle_minutes", 0),
            ("idle_minutes", -1),
            ("min_free_percent", 101),
            ("min_free_percent", -1),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(
                LLMRuntimeConfigError
            ):
                load_runtime_config(self.write_config(**{key: value}))

    def test_rejects_missing_model_file(self):
        payload = dict(self.models)
        payload["ghost"] = str(self.tmp / "models" / "ghost.gguf")
        with self.assertRaises(LLMRuntimeConfigError) as caught:
            load_runtime_config(self.write_config(models=payload))
        self.assertIn("ghost", str(caught.exception))

    def test_rejects_relative_or_non_gguf_model(self):
        relative = dict(self.models)
        relative["model-e-1b"] = "models/MODEL-E-1B-Q4_K_M.gguf"
        with self.assertRaises(LLMRuntimeConfigError):
            load_runtime_config(self.write_config(models=relative))

        wrong_suffix = dict(self.models)
        wrong_suffix["model-e-1b"] = str(self.tmp / "models" / "note.txt")
        (self.tmp / "models" / "note.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(LLMRuntimeConfigError):
            load_runtime_config(self.write_config(models=wrong_suffix))

    def test_rejects_unknown_default_model(self):
        with self.assertRaises(LLMRuntimeConfigError):
            load_runtime_config(self.write_config(default_model="model-g"))

    def test_rejects_missing_extra_or_broken_config(self):
        path = self.write_config()
        missing = json.loads(path.read_text(encoding="utf-8"))
        del missing["ctx"]
        path.write_text(json.dumps(missing), encoding="utf-8")
        with self.assertRaises(LLMRuntimeConfigError):
            load_runtime_config(path)

        path = self.write_config(extra_key="x")
        with self.assertRaises(LLMRuntimeConfigError):
            load_runtime_config(path)

        with self.assertRaises(LLMRuntimeConfigError):
            load_runtime_config(self.tmp / "no-such-file.json")

        broken = self.tmp / "broken.json"
        broken.write_text("{", encoding="utf-8")
        with self.assertRaises(LLMRuntimeConfigError):
            load_runtime_config(broken)

        array = self.tmp / "array.json"
        array.write_text("[]", encoding="utf-8")
        with self.assertRaises(LLMRuntimeConfigError):
            load_runtime_config(array)

    def test_rejects_symlink_and_directory_config(self):
        path = self.write_config()
        link = self.tmp / "link.json"
        os.symlink(path, link)
        with self.assertRaises(LLMRuntimeConfigError):
            load_runtime_config(link)
        with self.assertRaises(LLMRuntimeConfigError):
            load_runtime_config(self.tmp)


class EnsureRunningTest(RuntimeTestCase):
    def test_reuses_healthy_server_without_starting(self):
        payload = self.write_ledger()
        table = self.table_for(payload)
        popen = FakePopen([], table=table)
        urls = []

        def http_get(url):
            urls.append(url)
            return True

        alias = self.ensure(
            popen=popen,
            http_get=http_get,
            process_info=table,
            kill=table.kill,
            now=lambda: 2000.0,
        )
        self.assertEqual(alias, "model-f")
        self.assertEqual(popen.calls, [])
        self.assertEqual(table.kill_calls, [])
        self.assertEqual(urls, ["http://127.0.0.1:1234/health"])
        self.assertTrue(self.ledger_path.is_file())
        self.assertEqual(self.read_ledger()["pid"], 4242)
        self.assertEqual(idle_seconds(self.cfg, now=lambda: 2010.0), 10.0)

    def test_different_alias_stops_own_server_then_starts(self):
        payload = self.write_ledger()
        table = self.table_for(payload)
        popen = FakePopen([], pid=5000, table=table)
        alias = self.ensure(
            "model-e-1b",
            popen=popen,
            http_get=lambda url: True,
            process_info=table,
            kill=table.kill,
            free_percent=lambda: 62.0,
            now=lambda: 3000.0,
        )
        self.assertEqual(alias, "model-e-1b")
        self.assertEqual(table.kill_calls, [(4242, signal.SIGTERM)])
        self.assertEqual(len(popen.calls), 1)
        command = popen.calls[0][0]
        self.assertEqual(command[0], llama_server_binary())
        self.assertEqual(command[1:3], ["--host", "127.0.0.1"])
        self.assertEqual(command[3:5], ["--port", "1234"])
        self.assertEqual(command[5:7], ["--alias", "model-e-1b"])
        self.assertEqual(command[7], "-m")
        self.assertEqual(command[8], self.models["model-e-1b"])
        self.assertEqual(command[9:11], ["-c", "8192"])
        self.assertEqual(command[11:], ["--no-webui", "--reasoning", "off"])
        kwargs = popen.calls[0][1]
        self.assertTrue(kwargs["start_new_session"])
        written = self.read_ledger()
        self.assertEqual(written["pid"], 5000)
        self.assertEqual(written["pgid"], 5000)
        self.assertEqual(written["alias"], "model-e-1b")
        self.assertEqual(written["port"], 1234)
        self.assertEqual(written["model_path"], self.models["model-e-1b"])
        self.assertEqual(written["start_time"], NEW_LSTART)
        self.assertEqual(written["argv"], command)
        self.assertEqual(idle_seconds(self.cfg, now=lambda: 3007.0), 7.0)

    def test_low_memory_refuses_to_start(self):
        popen = FakePopen([])
        with self.assertRaises(LLMRuntimeError) as caught:
            self.ensure(popen=popen, free_percent=lambda: 20.0)
        message = str(caught.exception)
        self.assertIn("メモリが足りない", message)
        self.assertIn("空き20%", message)
        self.assertEqual(popen.calls, [])
        self.assertFalse(self.ledger_path.exists())

    def test_port_used_by_other_process_is_not_started_or_killed(self):
        popen = FakePopen([])
        calls = []
        with self.assertRaises(LLMRuntimeError) as caught:
            self.ensure(
                popen=popen,
                port_in_use=lambda port: True,
                kill=lambda pid, sig: calls.append((pid, sig)),
            )
        self.assertIn("ほかのプログラムが手元AIの番号を使っています", str(caught.exception))
        self.assertEqual(popen.calls, [])
        self.assertEqual(calls, [])

    def test_healthy_unknown_process_is_never_killed(self):
        popen = FakePopen([])
        calls = []
        with self.assertRaises(LLMRuntimeError) as caught:
            self.ensure(
                popen=popen,
                http_get=lambda url: True,
                kill=lambda pid, sig: calls.append((pid, sig)),
            )
        self.assertIn("ほかのプログラムが手元AIの番号を使っています", str(caught.exception))
        self.assertEqual(popen.calls, [])
        self.assertEqual(calls, [])
        self.assertFalse(self.ledger_path.exists())

    def test_health_never_ok_stops_and_raises(self):
        # 起動に失敗した時の後始末は popen の手綱だけで行い、
        # 数値の pid にシグナルを送らない（table.kill に SIGTERM を送らない）。
        table = FakeProcessTable()
        popen = FakePopen([], pid=7777, table=table)
        clock = [0.0]

        def monotonic():
            clock[0] += 30.0
            return clock[0]

        with self.assertRaises(LLMRuntimeError) as caught:
            self.ensure(
                popen=popen,
                http_get=lambda url: False,
                free_percent=lambda: 50.0,
                process_info=table,
                kill=table.kill,
                monotonic=monotonic,
            )
        self.assertIn("120秒", str(caught.exception))
        self.assertEqual(table.kill_calls, [])
        assert popen.handle is not None
        self.assertEqual(popen.handle.terminate_calls, [7777])
        self.assertFalse(self.ledger_path.exists())

    def test_ledger_pid_of_other_program_is_not_killed(self):
        payload = self.write_ledger()
        table = FakeProcessTable({4242: FakeProcess("/usr/bin/python3", LSTART)})
        calls = []
        health = SwitchHealth()
        popen = FakePopen(
            [], pid=6000, table=table, on_call=lambda: setattr(health, "healthy", True)
        )
        alias = self.ensure(
            popen=popen,
            http_get=health,
            process_info=table,
            kill=lambda pid, sig: (calls.append((pid, sig)), table.kill(pid, sig)),
            free_percent=lambda: 61.0,
        )
        self.assertEqual(alias, "model-f")
        self.assertEqual(calls, [])
        self.assertEqual(table.alive[4242].command, "/usr/bin/python3")
        self.assertEqual(self.read_ledger()["pid"], 6000)
        self.assertNotEqual(payload["pid"], self.read_ledger()["pid"])

    def test_unknown_alias_is_rejected_before_any_action(self):
        popen = FakePopen([])
        with self.assertRaises(LLMRuntimeError):
            self.ensure("no-such-model", popen=popen)
        self.assertEqual(popen.calls, [])

    def test_state_files_are_private(self):
        table = FakeProcessTable()
        health = SwitchHealth()
        self.ensure(
            popen=FakePopen(
                [], table=table, on_call=lambda: setattr(health, "healthy", True)
            ),
            http_get=health,
            process_info=table,
            kill=table.kill,
            now=lambda: 1000.0,
        )
        self.assertEqual(stat.S_IMODE(self.state_dir.stat().st_mode), 0o700)
        for name in (LEDGER_NAME, "last_used", llm_runtime.LOCK_FILENAME):
            with self.subTest(name=name):
                self.assertEqual(
                    stat.S_IMODE((self.state_dir / name).stat().st_mode), 0o600
                )
        self.assertFalse((self.state_dir / "llama-server.pid").exists())
        self.assertFalse((self.state_dir / "llama-server.alias").exists())

    def test_ledger_records_identity_fields(self):
        table = FakeProcessTable()
        health = SwitchHealth()
        self.ensure(
            popen=FakePopen(
                [], pid=4321, table=table, on_call=lambda: setattr(health, "healthy", True)
            ),
            http_get=health,
            process_info=table,
            kill=table.kill,
            now=lambda: 1000.0,
        )
        payload = self.read_ledger()
        self.assertEqual(
            set(payload),
            {"pid", "pgid", "start_time", "argv", "port", "model_path", "alias", "token"},
        )
        self.assertEqual(payload["pid"], 4321)
        self.assertEqual(payload["pgid"], 4321)
        self.assertEqual(payload["start_time"], NEW_LSTART)
        self.assertEqual(payload["port"], 1234)
        self.assertEqual(payload["alias"], "model-f")
        self.assertEqual(payload["model_path"], self.models["model-f"])
        self.assertEqual(payload["argv"][0], llama_server_binary())
        self.assertRegex(payload["token"], re.compile(r"^[0-9a-f]{16}$"))

    def test_ensure_running_starts_when_ledger_process_is_gone(self):
        self.write_ledger()
        table = FakeProcessTable()
        health = SwitchHealth()
        popen = FakePopen(
            [], pid=5000, table=table, on_call=lambda: setattr(health, "healthy", True)
        )
        alias = self.ensure(
            popen=popen,
            process_info=table,
            kill=table.kill,
            http_get=health,
            free_percent=lambda: 60.0,
        )
        self.assertEqual(alias, "model-f")
        self.assertEqual(table.kill_calls, [])
        self.assertEqual(len(popen.calls), 1)
        self.assertEqual(self.read_ledger()["pid"], 5000)

    def test_ensure_running_rejects_healthy_server_not_in_ledger(self):
        # 旧形式の pid/alias があっても、台帳が無ければ
        # health が ok でも自分のものとして使わない（ps も kill も呼ばない）。
        self.write_legacy_state(pid=4242, alias="model-f")
        popen = FakePopen([])

        def forbidden(pid, field):
            raise AssertionError("台帳が無いのに ps を呼んではいけない")

        with self.assertRaises(LLMRuntimeError) as caught:
            self.ensure(
                popen=popen,
                http_get=lambda url: True,
                process_info=forbidden,
                kill=lambda pid, sig: self.fail("kill を呼んではいけない"),
            )
        self.assertIn("ほかのプログラムが手元AIの番号を使っています", str(caught.exception))
        self.assertEqual(popen.calls, [])
        self.assertFalse((self.state_dir / "llama-server.pid").exists())
        self.assertFalse((self.state_dir / "llama-server.alias").exists())

    def test_ensure_running_rejects_llama_server_on_another_port(self):
        # 同じ機械の別ポートの llama-server は自分ではない。
        payload = self.write_ledger()
        other = self.ledger_argv(
            alias=payload["alias"], port=9999, model_path=payload["model_path"]
        )
        table = FakeProcessTable({4242: FakeProcess(" ".join(other), LSTART)})
        popen = FakePopen([], table=table)
        with self.assertRaises(LLMRuntimeError) as caught:
            self.ensure(
                popen=popen,
                http_get=lambda url: True,
                process_info=table,
                kill=table.kill,
            )
        self.assertIn("ほかのプログラムが手元AIの番号を使っています", str(caught.exception))
        self.assertEqual(popen.calls, [])
        self.assertEqual(table.kill_calls, [])
        self.assertTrue(self.ledger_path.is_file())

    def test_ensure_running_does_not_trust_health_when_start_time_differs(self):
        payload = self.write_ledger()
        table = self.table_for(payload, lstart=NEW_LSTART)
        popen = FakePopen([], table=table)
        with self.assertRaises(LLMRuntimeError) as caught:
            self.ensure(
                popen=popen,
                http_get=lambda url: True,
                process_info=table,
                kill=table.kill,
            )
        self.assertIn("ほかのプログラムが手元AIの番号を使っています", str(caught.exception))
        self.assertEqual(popen.calls, [])
        self.assertEqual(table.kill_calls, [])

    def test_broken_ledger_blocks_ensure_running(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text("{not json", encoding="utf-8")
        popen = FakePopen([])
        with self.assertRaises(LLMRuntimeError) as caught:
            self.ensure(popen=popen, kill=lambda pid, sig: self.fail("kill しない"))
        self.assertIn("台帳", str(caught.exception))
        self.assertEqual(popen.calls, [])

    # --- 起動失敗の後始末は popen の手綱だけで行う ---

    def test_startup_missing_start_time_never_signals_numeric_pid(self):
        # lstart が取れない時に (8888, SIGTERM) を
        # 数値 pid へ送らない。手綱の terminate だけで後始末する。
        table = FakeProcessTable()
        popen = FakePopen([], pid=8888)  # 台帳（ps）からは見えない子
        with self.assertRaises(LLMRuntimeError) as caught:
            self.ensure(
                popen=popen,
                process_info=table,
                kill=table.kill,
            )
        self.assertIn("起動時刻を確認できない", str(caught.exception))
        self.assertNotIn("止めきれなかった可能性", str(caught.exception))
        self.assertEqual(table.kill_calls, [])
        assert popen.handle is not None
        self.assertEqual(popen.handle.terminate_calls, [8888])
        self.assertEqual(popen.handle.kill_calls, [])
        self.assertFalse(self.ledger_path.exists())
        self.assertFalse(self.cfg.orphan_path.exists())

    def test_startup_ledger_write_failure_uses_only_process_handle(self):
        # 台帳を書けない時も、数値 pid ではなく手綱で止める。
        table = FakeProcessTable()
        popen = FakePopen([], pid=7777, table=table)
        with mock.patch.object(
            llm_runtime,
            "_write_ledger",
            side_effect=LLMRuntimeError("手元AIの状態を保存できませんでした"),
        ):
            with self.assertRaises(LLMRuntimeError) as caught:
                self.ensure(
                    popen=popen, process_info=table, kill=table.kill
                )
        self.assertIn("状態を保存できませんでした", str(caught.exception))
        self.assertNotIn("止めきれなかった可能性", str(caught.exception))
        self.assertEqual(table.kill_calls, [])
        assert popen.handle is not None
        self.assertEqual(popen.handle.terminate_calls, [7777])
        self.assertFalse(self.ledger_path.exists())
        self.assertFalse(self.cfg.orphan_path.exists())

    def test_startup_cleanup_records_orphan_when_handle_does_not_confirm(self):
        # 手綱で止められたか分からない時は、数値 pid へ送らずに
        # llama-server.orphan.json へ {pid, argv, at} を残す。
        table = FakeProcessTable()
        popen = FakePopen([], pid=8888, unkillable=True)
        with self.assertRaises(LLMRuntimeError) as caught:
            self.ensure(popen=popen, process_info=table, kill=table.kill)
        message = str(caught.exception)
        self.assertIn("起動時刻を確認できない", message)
        self.assertIn("止めきれなかった可能性", message)
        self.assertEqual(table.kill_calls, [])
        assert popen.handle is not None
        self.assertEqual(popen.handle.terminate_calls, [8888])
        self.assertEqual(popen.handle.kill_calls, [8888])
        orphan = json.loads(self.cfg.orphan_path.read_text(encoding="utf-8"))
        self.assertEqual(orphan["pid"], 8888)
        self.assertEqual(orphan["argv"][0], llama_server_binary())
        self.assertIn("at", orphan)
        self.assertFalse(self.ledger_path.exists())

    # --- health が ok でも、生きている子と今書いた世代の両方を確認 ---

    def test_startup_health_ok_but_process_replaced_is_not_success(self):
        # health 中に同じ pid が別プロセスへ
        # 替わったら、成功（モデル名）を返さない。
        table = FakeProcessTable()
        urls = []

        def health(url):
            urls.append(url)
            if len(urls) == 1:
                return False
            table.alive[4321] = FakeProcess("/usr/bin/python3", NEW_LSTART)
            return True

        popen = FakePopen([], table=table)
        with self.assertRaises(LLMRuntimeError) as caught:
            self.ensure(
                popen=popen,
                http_get=health,
                process_info=table,
                kill=table.kill,
            )
        self.assertIn("確認できない", str(caught.exception))
        self.assertNotIn("止めきれなかった可能性", str(caught.exception))
        self.assertEqual(table.kill_calls, [])
        assert popen.handle is not None
        self.assertEqual(popen.handle.terminate_calls, [4321])
        self.assertFalse(self.ledger_path.exists())

    def test_startup_health_ok_but_child_exited_is_not_success(self):
        table = FakeProcessTable()
        urls = []

        def health(url):
            urls.append(url)
            if len(urls) == 1:
                return False
            table.alive.pop(4321, None)
            return True

        popen = FakePopen([], table=table)
        with self.assertRaises(LLMRuntimeError) as caught:
            self.ensure(
                popen=popen,
                http_get=health,
                process_info=table,
                kill=table.kill,
            )
        self.assertIn("確認できない", str(caught.exception))
        self.assertEqual(table.kill_calls, [])
        self.assertFalse(self.ledger_path.exists())

    def test_startup_health_ok_but_new_ledger_generation_is_not_success(self):
        # health 中に別の起動が台帳を新しい世代へ替えたら、その世代を
        # 「今書いた世代」と取り違えず、成功を返さない。
        table = FakeProcessTable()
        replacement = {}
        urls = []

        def health(url):
            urls.append(url)
            if len(urls) == 1:
                return False
            replacement.update(
                self.write_ledger(
                    pid=5252, start_time=NEW_LSTART, token="fedcba9876543210"
                )
            )
            table.alive[5252] = FakeProcess(
                " ".join(replacement["argv"]), NEW_LSTART
            )
            return True

        popen = FakePopen([], table=table)
        with self.assertRaises(LLMRuntimeError) as caught:
            self.ensure(
                popen=popen,
                http_get=health,
                process_info=table,
                kill=table.kill,
            )
        self.assertIn("確認できない", str(caught.exception))
        self.assertEqual(table.kill_calls, [])
        assert popen.handle is not None
        self.assertEqual(popen.handle.terminate_calls, [4321])
        # 新しい世代には何も送らず、台帳も消さない。
        written = self.read_ledger()
        self.assertEqual(written["pid"], 5252)
        self.assertEqual(written["token"], "fedcba9876543210")
        self.assertIn(5252, table.alive)


class StopTest(RuntimeTestCase):
    def test_stops_only_ledger_process(self):
        payload = self.write_ledger()
        table = self.table_for(payload)
        touch_used(self.cfg, now=lambda: 900.0)

        self.assertTrue(stop(self.cfg, kill=table.kill, process_info=table))
        self.assertEqual(table.kill_calls, [(4242, signal.SIGTERM)])
        self.assertFalse(self.ledger_path.exists())
        self.assertFalse((self.state_dir / "last_used").exists())

    def test_stop_treats_exit_between_ps_queries_as_gone(self):
        # 実際の環境で起きた: SIGTERM の後、起動時刻を引いた直後に
        # プロセスが終わり、実行コマンドが引けなかった。止まったのに
        # 「確認できない」で例外になっていた。
        payload = self.write_ledger()
        table = self.table_for(payload, on_kill=lambda t, pid, sig: None)
        original = table.__call__
        state = {"exiting": False}

        def process_info(pid, field):
            if state["exiting"]:
                if field == "lstart" and not state.get("lstart_seen"):
                    state["lstart_seen"] = True
                    return payload["start_time"]
                return None
            return original(pid, field)

        def kill(pid, sig):
            table.kill_calls.append((pid, sig))
            state["exiting"] = True

        self.assertTrue(stop(self.cfg, kill=kill, process_info=process_info, sleep=lambda s: None))
        self.assertEqual(table.kill_calls, [(4242, signal.SIGTERM)])
        self.assertFalse(self.ledger_path.exists())

    def test_does_not_kill_when_pid_is_not_llama_server(self):
        payload = self.write_ledger()
        table = FakeProcessTable({4242: FakeProcess("/usr/bin/python3", LSTART)})
        with self.assertRaises(LLMRuntimeError) as caught:
            stop(self.cfg, kill=table.kill, process_info=table)
        self.assertIn("確認できない", str(caught.exception))
        self.assertEqual(table.kill_calls, [])
        self.assertEqual(table.alive[4242].command, "/usr/bin/python3")
        # 台帳は消さない（何かが動いている可能性を人間が確かめられるように）。
        self.assertTrue(self.ledger_path.is_file())

    def test_ledger_stop_without_pid_file_returns_false(self):
        table = FakeProcessTable()
        self.assertFalse(stop(self.cfg, kill=table.kill, process_info=table))
        self.assertEqual(table.kill_calls, [])

    def test_legacy_pid_file_is_removed_without_killing(self):
        self.write_legacy_state(pid=4242)
        table = FakeProcessTable({4242: FakeProcess("/opt/homebrew/bin/llama-server")})
        self.assertFalse(stop(self.cfg, kill=table.kill, process_info=table))
        self.assertEqual(table.kill_calls, [])
        self.assertFalse((self.state_dir / "llama-server.pid").exists())
        self.assertFalse((self.state_dir / "llama-server.alias").exists())
        self.assertFalse(self.ledger_path.exists())

    def test_stop_without_ledger_returns_false(self):
        calls = []
        self.assertFalse(
            stop(self.cfg, kill=lambda pid, sig: calls.append((pid, sig)))
        )
        self.assertEqual(calls, [])

    def test_escalates_to_sigkill_after_ten_seconds(self):
        payload = self.write_ledger()
        table = self.table_for(
            payload,
            on_kill=lambda table, pid, sig: (
                table.alive.pop(pid, None) if sig == signal.SIGKILL else None
            ),
        )
        self.assertTrue(stop(self.cfg, kill=table.kill, process_info=table, sleep=lambda seconds: None))
        self.assertEqual(
            table.kill_calls, [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]
        )
        self.assertFalse(self.ledger_path.exists())

    def test_stop_clears_ledger_when_process_already_gone(self):
        self.write_ledger()
        table = FakeProcessTable()
        self.assertFalse(stop(self.cfg, kill=table.kill, process_info=table))
        self.assertEqual(table.kill_calls, [])
        self.assertFalse(self.ledger_path.exists())

    def test_stop_does_not_kill_when_start_time_differs(self):
        payload = self.write_ledger()
        table = self.table_for(payload, lstart=NEW_LSTART)
        with self.assertRaises(LLMRuntimeError):
            stop(self.cfg, kill=table.kill, process_info=table)
        self.assertEqual(table.kill_calls, [])
        self.assertTrue(self.ledger_path.is_file())

    def test_stop_does_not_kill_llama_server_on_another_port(self):
        # 別ポートの llama-server は SIGTERM しない。
        payload = self.write_ledger()
        other = self.ledger_argv(
            alias=payload["alias"], port=9999, model_path=payload["model_path"]
        )
        table = FakeProcessTable({4242: FakeProcess(" ".join(other), LSTART)})
        with self.assertRaises(LLMRuntimeError):
            stop(self.cfg, kill=table.kill, process_info=table)
        self.assertEqual(table.kill_calls, [])

    def test_stop_does_not_kill_when_argv_differs(self):
        payload = self.write_ledger()
        broken = list(payload["argv"])
        broken[broken.index("-m") + 1] = str(self.tmp / "models" / "other.gguf")
        table = self.table_for(payload, command=" ".join(broken))
        with self.assertRaises(LLMRuntimeError):
            stop(self.cfg, kill=table.kill, process_info=table)
        self.assertEqual(table.kill_calls, [])

    def test_stop_does_not_sigkill_when_pid_becomes_another_process(self):
        # SIGTERM 後に同じ pid が別プロセスへ替わったら
        # SIGKILL を送らない。
        payload = self.write_ledger()

        def replace(table, pid, sig):
            if sig == signal.SIGTERM:
                table.alive[pid] = FakeProcess("/usr/bin/python3", LSTART)

        table = self.table_for(payload, on_kill=replace)
        self.assertTrue(stop(self.cfg, kill=table.kill, process_info=table, sleep=lambda seconds: None))
        self.assertEqual(table.kill_calls, [(4242, signal.SIGTERM)])
        self.assertEqual(table.alive[4242].command, "/usr/bin/python3")
        self.assertFalse(self.ledger_path.exists())

    def test_broken_ledger_is_not_killed(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text("garbage", encoding="utf-8")
        calls = []
        with self.assertRaises(LLMRuntimeError):
            stop(self.cfg, kill=lambda pid, sig: calls.append((pid, sig)))
        self.assertEqual(calls, [])

    def test_is_running_requires_matching_ledger(self):
        self.assertFalse(is_running(self.cfg, process_info=FakeProcessTable()))
        payload = self.write_ledger()
        self.assertTrue(is_running(self.cfg, process_info=self.table_for(payload)))

    def test_is_running_is_false_when_start_time_differs(self):
        payload = self.write_ledger()
        table = self.table_for(payload, lstart=NEW_LSTART)
        self.assertFalse(is_running(self.cfg, process_info=table))

    def test_is_running_ignores_legacy_pid_file(self):
        self.write_legacy_state(pid=4242)
        self.assertFalse(is_running(self.cfg, process_info=FakeProcessTable()))
        self.assertFalse((self.state_dir / "llama-server.pid").exists())

    # --- 停止は開始時の台帳の世代に固定する ---

    def test_stop_does_not_follow_replacement_ledger_generation(self):
        # 4242 へ SIGTERM 後、台帳が別の世代
        # （5252）に替わったら、5252 には何も送らず台帳も消さない。
        initial = self.write_ledger(pid=4242)
        table = self.table_for(initial)
        replacement = {}

        def kill(pid, sig):
            table.kill_calls.append((pid, sig))
            if pid == 4242:
                table.alive.pop(pid, None)
                replacement.update(
                    self.write_ledger(
                        pid=5252, start_time=NEW_LSTART, token="fedcba9876543210"
                    )
                )
                table.alive[5252] = FakeProcess(
                    " ".join(replacement["argv"]), NEW_LSTART
                )

        self.assertFalse(
            stop(self.cfg, kill=kill, process_info=table, sleep=lambda _: None)
        )
        self.assertEqual(table.kill_calls, [(4242, signal.SIGTERM)])
        written = self.read_ledger()
        self.assertEqual(written["pid"], 5252)
        self.assertEqual(written["token"], "fedcba9876543210")
        self.assertIn(5252, table.alive)

    def test_stop_checks_generation_before_sigkill(self):
        # SIGKILL の直前に台帳が別世代へ替わったら SIGKILL を送らない。
        initial = self.write_ledger(pid=4242)
        table = self.table_for(
            initial, on_kill=lambda table, pid, sig: None
        )  # SIGTERM では死なない
        polls = max(
            1,
            int(
                llm_runtime._STOP_GRACE_SECONDS
                / llm_runtime._STOP_POLL_SECONDS
            ),
        )
        sleeps = {"count": 0}

        def sleep(seconds):
            sleeps["count"] += 1
            if sleeps["count"] == polls:
                replacement = self.write_ledger(
                    pid=5252, start_time=NEW_LSTART, token="fedcba9876543210"
                )
                table.alive[5252] = FakeProcess(
                    " ".join(replacement["argv"]), NEW_LSTART
                )

        self.assertFalse(
            stop(self.cfg, kill=table.kill, process_info=table, sleep=sleep)
        )
        self.assertEqual(table.kill_calls, [(4242, signal.SIGTERM)])
        self.assertEqual(self.read_ledger()["pid"], 5252)
        self.assertIn(5252, table.alive)
        self.assertIn(4242, table.alive)

    def test_stop_keeps_new_generation_when_ledger_replaces_during_stop(self):
        # 台帳の入れ替わりで戻る時は、新しい世代の last_used も消さない。
        initial = self.write_ledger(pid=4242)
        table = self.table_for(initial)
        touch_used(self.cfg, now=lambda: 900.0)

        def kill(pid, sig):
            table.kill_calls.append((pid, sig))
            table.alive.pop(pid, None)
            self.write_ledger(
                pid=5252, start_time=NEW_LSTART, token="fedcba9876543210"
            )
            table.alive[5252] = FakeProcess(
                " ".join(self.read_ledger()["argv"]), NEW_LSTART
            )

        self.assertFalse(
            stop(self.cfg, kill=kill, process_info=table, sleep=lambda _: None)
        )
        self.assertTrue((self.state_dir / "last_used").is_file())
        self.assertEqual(self.read_ledger()["pid"], 5252)


class StateLockTest(RuntimeTestCase):
    """起動・切替・停止の排他ロック（llama-server.lock）."""

    def test_ensure_running_reports_busy_when_lock_is_held(self):
        self.hold_state_lock()
        popen = FakePopen([])
        with mock.patch.object(llm_runtime, "_LOCK_TIMEOUT_SECONDS", 0.0):
            with self.assertRaises(LLMRuntimeError) as caught:
                self.ensure(popen=popen, sleep=lambda seconds: None)
        self.assertEqual(
            str(caught.exception),
            "手元AIの操作が混み合っています。少し待ってからもう一度。",
        )
        self.assertEqual(popen.calls, [])
        self.assertFalse(self.ledger_path.exists())

    def test_stop_reports_busy_when_lock_is_held(self):
        self.write_ledger()
        self.hold_state_lock()
        table = FakeProcessTable({4242: FakeProcess("/usr/bin/python3", LSTART)})
        with mock.patch.object(llm_runtime, "_LOCK_TIMEOUT_SECONDS", 0.0):
            with self.assertRaises(LLMRuntimeError) as caught:
                stop(
                    self.cfg,
                    kill=table.kill,
                    process_info=table,
                    sleep=lambda seconds: None,
                )
        self.assertEqual(
            str(caught.exception),
            "手元AIの操作が混み合っています。少し待ってからもう一度。",
        )
        self.assertEqual(table.kill_calls, [])

    def test_lock_is_released_after_stop(self):
        self.assertFalse(
            stop(self.cfg, kill=FakeProcessTable().kill, process_info=FakeProcessTable())
        )
        descriptor = os.open(self.cfg.lock_path, os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)

    def test_lock_file_is_private(self):
        self.hold_state_lock()
        self.assertEqual(
            stat.S_IMODE(self.cfg.lock_path.stat().st_mode), 0o600
        )


class IdleSecondsTest(RuntimeTestCase):
    def test_missing_record_returns_none(self):
        self.assertIsNone(idle_seconds(self.cfg))

    def test_touch_and_read(self):
        touch_used(self.cfg, now=lambda: 1000.0)
        self.assertEqual(stat.S_IMODE(self.state_dir.stat().st_mode), 0o700)
        self.assertEqual(
            stat.S_IMODE((self.state_dir / "last_used").stat().st_mode), 0o600
        )
        idle = idle_seconds(self.cfg, now=lambda: 1000.0)
        self.assertIsNotNone(idle)
        self.assertEqual(idle, 0.0)
        idle = idle_seconds(self.cfg, now=lambda: 1060.0)
        self.assertIsNotNone(idle)
        self.assertEqual(idle, 60.0)
        self.assertEqual(idle_seconds(self.cfg, now=lambda: 900.0), 0.0)

    def test_broken_record_returns_none(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "last_used").write_text("garbage\n", encoding="utf-8")
        self.assertIsNone(idle_seconds(self.cfg))


class PathResolutionTest(unittest.TestCase):
    """llama-server と runtime.json の場所の決め方（環境変数 → 既定）."""

    def env(self, **values):
        base = {
            key: ""
            for key in ("SHOKOBAN_LLAMA_SERVER", "SHOKOBAN_RUNTIME_CONFIG", "SHOKOBAN_WORK_DIR")
        }
        base.update(values)
        patcher = mock.patch.dict(os.environ, base)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_binary_env_wins(self):
        self.env(SHOKOBAN_LLAMA_SERVER="/x/llama-server")
        self.assertEqual(llm_runtime.llama_server_binary(), "/x/llama-server")

    def test_binary_from_path_then_fallback(self):
        self.env()
        with mock.patch("shutil.which", return_value="/usr/local/bin/llama-server"):
            self.assertEqual(llm_runtime.llama_server_binary(), "/usr/local/bin/llama-server")
        with mock.patch("shutil.which", return_value=None):
            self.assertEqual(
                llm_runtime.llama_server_binary(), "/opt/homebrew/bin/llama-server"
            )

    def test_config_path_order(self):
        self.env(SHOKOBAN_RUNTIME_CONFIG="/c/rt.json", SHOKOBAN_WORK_DIR="/w")
        self.assertEqual(llm_runtime.default_config_path(), Path("/c/rt.json"))
        self.env(SHOKOBAN_WORK_DIR="/w")
        self.assertEqual(llm_runtime.default_config_path(), Path("/w/runtime.json"))
        self.env()
        self.assertEqual(
            llm_runtime.default_config_path(), Path.home() / ".shokoban" / "runtime.json"
        )

    def test_missing_binary_gives_japanese_error(self):
        tmp = Path(tempfile.mkdtemp(prefix="llm_runtime_test_"))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        model = tmp / "m.gguf"
        model.write_bytes(b"GGUF-fake")
        cfg = RuntimeConfig(
            port=1234,
            ctx=8192,
            idle_minutes=10,
            min_free_percent=35,
            default_model="m",
            models={"m": str(model)},
            state_dir=tmp / "state",
        )
        self.env(SHOKOBAN_LLAMA_SERVER="/nonexistent/llama-server")

        def popen(*args, **kwargs):
            raise FileNotFoundError(2, "No such file", args[0][0])

        with self.assertRaises(LLMRuntimeError) as caught:
            ensure_running(
                cfg,
                popen=popen,
                http_get=lambda url: False,
                free_percent=lambda: 50.0,
                port_in_use=lambda port: False,
                process_info=FakeProcessTable(),
                sleep=lambda seconds: None,
            )
        message = str(caught.exception)
        self.assertIn("llama-server が見つかりません", message)
        self.assertIn("SHOKOBAN_LLAMA_SERVER", message)


class PlainPathTest(RuntimeTestCase):
    """起動した手元AIを ps の出力で見分けるので、空白や半角以外の文字を含むパスは設定の時点で断る."""

    def config_with_model(self, directory_name):
        folder = self.tmp / directory_name
        folder.mkdir()
        model = folder / "model.gguf"
        model.write_bytes(b"GGUF-fake")
        return dict(
            port=1234,
            ctx=8192,
            idle_minutes=10,
            min_free_percent=35,
            default_model="model-x",
            models={"model-x": str(model)},
            state_dir=self.state_dir,
        )

    def test_model_path_with_space_is_rejected_with_a_clear_message(self):
        with self.assertRaises(LLMRuntimeConfigError) as raised:
            RuntimeConfig(**self.config_with_model("My Models"))
        self.assertIn("空白や日本語", str(raised.exception))

    def test_model_path_with_non_ascii_is_rejected(self):
        with self.assertRaises(LLMRuntimeConfigError):
            RuntimeConfig(**self.config_with_model("モデル置き場"))

    def test_plain_model_path_is_accepted(self):
        cfg = RuntimeConfig(**self.config_with_model("models-plain"))
        self.assertEqual(cfg.default_model, "model-x")

    def test_llama_server_location_with_space_is_rejected(self):
        with mock.patch.dict(os.environ, {"SHOKOBAN_LLAMA_SERVER": "/Applications/My Apps/llama-server"}):
            with self.assertRaises(LLMRuntimeError) as raised:
                llama_server_binary()
        self.assertIn("SHOKOBAN_LLAMA_SERVER", str(raised.exception))


class IdleStopScriptTest(RuntimeTestCase):
    """``scripts/llm_idle_stop.py``（使っていない手元AIを止める。定期的に呼ぶ）."""

    def write_config(self):
        path = self.tmp / "runtime.json"
        path.write_text(
            json.dumps(
                {
                    "port": 1234,
                    "ctx": 8192,
                    "idle_minutes": 10,
                    "min_free_percent": 35,
                    "default_model": "model-f",
                    "models": self.models,
                    "state_dir": str(self.state_dir),
                }
            ),
            encoding="utf-8",
        )
        return path

    def run_script(self, *args):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = idle_stop.main([str(arg) for arg in args])
        return code, stdout.getvalue()

    def test_reports_not_running_in_one_line(self):
        code, output = self.run_script("--config", self.write_config())
        self.assertEqual(code, 0)
        self.assertEqual(output, "手元AIは動いていません\n")
        self.assertEqual(output.count("\n"), 1)

    def test_stops_when_idle_exceeded(self):
        config = self.write_config()
        stopped = []
        with mock.patch.object(idle_stop, "is_running", return_value=True), \
                mock.patch.object(idle_stop, "idle_seconds", return_value=700.0), \
                mock.patch.object(
                    idle_stop, "stop", side_effect=lambda cfg: stopped.append(cfg) or True
                ):
            code, output = self.run_script("--config", config)
        self.assertEqual(code, 0)
        self.assertEqual(output, "手元AIを止めました（アイドル11.7分）\n")
        self.assertEqual(len(stopped), 1)
        self.assertEqual(stopped[0].port, 1234)

    def test_keeps_server_when_idle_is_under_limit(self):
        config = self.write_config()
        with mock.patch.object(idle_stop, "is_running", return_value=True), \
                mock.patch.object(idle_stop, "idle_seconds", return_value=60.0), \
                mock.patch.object(idle_stop, "stop") as stop_mock:
            code, output = self.run_script("--config", config)
        self.assertEqual(code, 0)
        self.assertIn("アイドル1.0分", output)
        self.assertEqual(output.count("\n"), 1)
        stop_mock.assert_not_called()

    def test_reports_missing_record_without_stopping(self):
        config = self.write_config()
        with mock.patch.object(idle_stop, "is_running", return_value=True), \
                mock.patch.object(idle_stop, "idle_seconds", return_value=None), \
                mock.patch.object(idle_stop, "stop") as stop_mock:
            code, output = self.run_script("--config", config)
        self.assertEqual(code, 0)
        self.assertEqual(output.count("\n"), 1)
        stop_mock.assert_not_called()

    def test_reports_broken_config_in_one_line(self):
        code, output = self.run_script("--config", self.tmp / "missing.json")
        self.assertEqual(code, 2)
        self.assertEqual(output.count("\n"), 1)
        self.assertIn("設定を読めない", output)

    def test_default_config_comes_from_the_work_dir(self):
        config = self.write_config()
        with mock.patch.dict(os.environ, {"SHOKOBAN_WORK_DIR": str(self.tmp)}, clear=False):
            os.environ.pop("SHOKOBAN_RUNTIME_CONFIG", None)
            code, output = self.run_script()
        self.assertTrue(config.is_file())
        self.assertEqual(code, 0)
        self.assertEqual(output, "手元AIは動いていません\n")


if __name__ == "__main__":
    unittest.main()
