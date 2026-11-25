"""Local LLM provider and settings tests."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from shokoban.local_llm import (
    LocalLLMConfigError,
    LocalLLMConnectionError,
    LocalLLMResponseError,
    LocalLLMSettings,
    LocalLLMSettingsError,
    generate_local_text,
    list_local_models,
    load_local_llm_settings,
    save_local_llm_settings,
)


class LocalLLMTest(unittest.TestCase):
    def test_defaults_and_config_rejections(self) -> None:
        self.assertEqual(LocalLLMSettings(), LocalLLMSettings("ollama", "gemma4:12b"))
        for settings in (
            {"provider": "openai", "model": "model-x"},
            {"provider": "ollama", "model": "model-a:cloud"},
            {"provider": "lmstudio", "model": ""},
            {"provider": "ollama", "model": "bad model"},
        ):
            with self.subTest(settings=settings), self.assertRaises(LocalLLMConfigError):
                LocalLLMSettings(**settings)

    def test_ollama_model_list_uses_fixed_loopback_and_filters_cloud(self) -> None:
        calls = []

        def caller(method, url, payload, timeout):
            calls.append((method, url, payload, timeout))
            return {
                "models": [
                    {"name": "gemma4:12b"},
                    {"name": "gemma4:12b"},
                    {"name": "model-b:cloud"},
                    {"name": "model-c:4b"},
                ]
            }

        self.assertEqual(
            list_local_models("ollama", http_caller=caller),
            ("gemma4:12b", "model-c:4b"),
        )
        self.assertEqual(calls[0][1], "http://127.0.0.1:11434/api/tags")

    def test_lmstudio_model_list_uses_fixed_loopback(self) -> None:
        def caller(method, url, payload, timeout):
            self.assertEqual(url, "http://127.0.0.1:1234/v1/models")
            return {"data": [{"id": "vendor/model-d"}]}

        self.assertEqual(
            list_local_models("lmstudio", http_caller=caller),
            ("vendor/model-d",),
        )

    def test_model_list_failure_is_fixed_and_non_sensitive(self) -> None:
        with self.assertRaises(LocalLLMConnectionError) as caught:
            list_local_models("ollama", http_caller=lambda *args: {"bad": "secret"})
        self.assertNotIn("secret", str(caught.exception))

    def test_ollama_generation_contract(self) -> None:
        captured = {}

        def caller(method, url, payload, timeout):
            captured.update(method=method, url=url, payload=payload)
            return {"response": "完全架空の応答"}

        result = generate_local_text(
            LocalLLMSettings(), "完全架空入力", http_caller=caller
        )
        self.assertEqual(result, "完全架空の応答")
        self.assertEqual(captured["url"], "http://127.0.0.1:11434/api/generate")
        self.assertEqual(captured["payload"]["model"], "gemma4:12b")
        self.assertFalse(captured["payload"]["stream"])
        self.assertFalse(captured["payload"]["think"])

    def test_lmstudio_generation_contract(self) -> None:
        captured = {}

        def caller(method, url, payload, timeout):
            captured.update(method=method, url=url, payload=payload)
            return {"choices": [{"message": {"content": "完全架空の応答"}}]}

        result = generate_local_text(
            LocalLLMSettings("lmstudio", "vendor/model-d"),
            "完全架空入力",
            http_caller=caller,
        )
        self.assertEqual(result, "完全架空の応答")
        self.assertEqual(
            captured["url"], "http://127.0.0.1:1234/v1/chat/completions"
        )
        self.assertEqual(captured["payload"]["messages"][0]["role"], "user")

    def test_connection_failure_does_not_leak_prompt(self) -> None:
        prompt = "完全架空の秘密文字列"
        caller = lambda *args: (_ for _ in ()).throw(RuntimeError(prompt))
        with self.assertRaises(LocalLLMConnectionError) as caught:
            generate_local_text(LocalLLMSettings(), prompt, http_caller=caller)
        self.assertNotIn(prompt, str(caught.exception))

    def test_invalid_provider_responses_are_rejected(self) -> None:
        cases = (
            (LocalLLMSettings(), {"response": ""}),
            (LocalLLMSettings("lmstudio", "local/model"), {"choices": []}),
        )
        for settings, response in cases:
            with self.subTest(provider=settings.provider), self.assertRaises(
                LocalLLMResponseError
            ):
                generate_local_text(
                    settings,
                    "完全架空入力",
                    http_caller=lambda *args, response=response: response,
                )

    def test_settings_roundtrip_and_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "config" / "settings.json"
            expected = LocalLLMSettings("lmstudio", "vendor/model-d")
            self.assertEqual(load_local_llm_settings(path), LocalLLMSettings())
            save_local_llm_settings(expected, path)
            self.assertEqual(load_local_llm_settings(path), expected)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            self.assertEqual(
                set(json.loads(path.read_text(encoding="utf-8"))),
                {"schema_version", "provider", "model"},
            )

    def test_invalid_and_unknown_settings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "settings.json"
            for value in (
                "not json",
                '{"schema_version":1,"provider":"ollama","model":"gemma4:12b","extra":1}',
                '{"schema_version":true,"provider":"ollama","model":"gemma4:12b"}',
            ):
                path.write_text(value, encoding="utf-8")
                with self.assertRaises(LocalLLMSettingsError):
                    load_local_llm_settings(path)

    def test_symlink_settings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "settings.json"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaises(LocalLLMSettingsError):
                load_local_llm_settings(link)
            with self.assertRaises(LocalLLMSettingsError):
                save_local_llm_settings(LocalLLMSettings(), link)

    def test_dangling_symlink_settings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory).resolve() / "settings.json"
            try:
                link.symlink_to(link.parent / "missing.json")
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaises(LocalLLMSettingsError):
                load_local_llm_settings(link)


if __name__ == "__main__":
    unittest.main()
