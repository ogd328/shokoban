"""Local-only LLM providers and cross-platform user settings."""

from __future__ import annotations

import json
import os
import re
import stat
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


SETTINGS_SCHEMA_VERSION = 1
RECOMMENDED_PROVIDER = "ollama"
RECOMMENDED_MODEL = "gemma4:12b"
RECOMMENDED_MODEL_LABEL = "Gemma 4 12B（推奨・動作確認済み）"
PROVIDER_LABELS = {"ollama": "Ollama", "lmstudio": "LM Studio"}
_BASE_URLS = {
    "ollama": "http://127.0.0.1:11434",
    "lmstudio": "http://127.0.0.1:1234",
}
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9._:/-]{1,200}")
_CLOUD_TOKEN_RE = re.compile(r"(?:^|[:/_-])cloud(?:$|[:/_-])", re.IGNORECASE)

HTTPCaller = Callable[[str, str, dict[str, Any] | None, int], dict[str, Any]]


class LocalLLMError(RuntimeError):
    """Raised when a local LLM operation cannot be completed safely."""


class LocalLLMConfigError(LocalLLMError):
    pass


class LocalLLMConnectionError(LocalLLMError):
    pass


class LocalLLMResponseError(LocalLLMError):
    pass


class LocalLLMSettingsError(LocalLLMError):
    pass


def _validate_provider(provider: str) -> None:
    if type(provider) is not str or provider not in _BASE_URLS:
        raise LocalLLMConfigError("unsupported local LLM provider")


def _validate_model(model: str) -> None:
    if type(model) is not str or not _MODEL_ID_RE.fullmatch(model):
        raise LocalLLMConfigError("invalid local LLM model identifier")
    if _CLOUD_TOKEN_RE.search(model):
        raise LocalLLMConfigError("Cloud LLM models are not allowed")


@dataclass(frozen=True, slots=True)
class LocalLLMSettings:
    provider: str = RECOMMENDED_PROVIDER
    model: str = RECOMMENDED_MODEL

    def __post_init__(self) -> None:
        _validate_provider(self.provider)
        _validate_model(self.model)


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: int,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if type(parsed) is not dict:
        raise ValueError("response root is not an object")
    return parsed


def list_local_models(
    provider: str,
    *,
    timeout: int = 5,
    http_caller: HTTPCaller | None = None,
) -> tuple[str, ...]:
    """List safe model IDs reported by one fixed loopback provider."""
    _validate_provider(provider)
    caller = http_caller or _http_json
    if provider == "ollama":
        url = f"{_BASE_URLS[provider]}/api/tags"
    else:
        url = f"{_BASE_URLS[provider]}/v1/models"
    try:
        response = caller("GET", url, None, timeout)
        entries = response["models"] if provider == "ollama" else response["data"]
        if type(entries) is not list:
            raise TypeError
        key = "name" if provider == "ollama" else "id"
        candidates = [entry[key] for entry in entries if type(entry) is dict]
    except Exception as exc:
        raise LocalLLMConnectionError("local LLM model list is unavailable") from exc

    models: set[str] = set()
    for candidate in candidates:
        try:
            _validate_model(candidate)
        except LocalLLMConfigError:
            continue
        models.add(candidate)
    return tuple(sorted(models))


def generate_local_text(
    settings: LocalLLMSettings,
    prompt: str,
    *,
    num_predict: int = 512,
    timeout: int = 120,
    http_caller: HTTPCaller | None = None,
) -> str:
    """Generate text through the selected fixed loopback provider."""
    if type(settings) is not LocalLLMSettings:
        raise LocalLLMConfigError("invalid local LLM settings")
    if type(prompt) is not str or not prompt.strip():
        raise LocalLLMConfigError("local LLM prompt is empty or invalid")
    caller = http_caller or _http_json
    if settings.provider == "ollama":
        url = f"{_BASE_URLS['ollama']}/api/generate"
        payload = {
            "model": settings.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1, "num_predict": num_predict},
        }
    else:
        url = f"{_BASE_URLS['lmstudio']}/v1/chat/completions"
        payload = {
            "model": settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": num_predict,
            "stream": False,
        }
    try:
        response = caller("POST", url, payload, timeout)
    except Exception as exc:
        raise LocalLLMConnectionError("local LLM request failed") from exc
    try:
        if settings.provider == "ollama":
            text = response["response"]
        else:
            text = response["choices"][0]["message"]["content"]
        if type(text) is not str or not text.strip():
            raise TypeError
    except (KeyError, IndexError, TypeError) as exc:
        raise LocalLLMResponseError("local LLM response is invalid") from exc
    return text


def default_settings_path() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "ShokobanLLM" / "settings.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ShokobanLLM" / "settings.json"
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "shokoban-llm" / "settings.json"


def _reject_symlink_components(path: Path) -> None:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    while True:
        if current.is_symlink():
            raise LocalLLMSettingsError("settings path contains a symlink")
        if current.parent == current:
            return
        current = current.parent


def load_local_llm_settings(path: Path | None = None) -> LocalLLMSettings:
    selected_path = path or default_settings_path()
    if selected_path.is_symlink():
        raise LocalLLMSettingsError("settings path contains a symlink")
    if not selected_path.exists():
        return LocalLLMSettings()
    _reject_symlink_components(selected_path)
    try:
        info = os.lstat(selected_path)
        if not stat.S_ISREG(info.st_mode):
            raise LocalLLMSettingsError("settings path is not a regular file")
        parsed = json.loads(selected_path.read_text(encoding="utf-8"))
        if type(parsed) is not dict or set(parsed) != {
            "schema_version",
            "provider",
            "model",
        }:
            raise LocalLLMSettingsError("settings file does not match schema")
        if type(parsed["schema_version"]) is not int or parsed["schema_version"] != 1:
            raise LocalLLMSettingsError("unsupported settings schema")
        return LocalLLMSettings(provider=parsed["provider"], model=parsed["model"])
    except LocalLLMSettingsError:
        raise
    except Exception as exc:
        raise LocalLLMSettingsError("settings file is invalid") from exc


def save_local_llm_settings(
    settings: LocalLLMSettings,
    path: Path | None = None,
) -> Path:
    if type(settings) is not LocalLLMSettings:
        raise LocalLLMSettingsError("invalid settings value")
    selected_path = path or default_settings_path()
    _reject_symlink_components(selected_path.parent)
    if selected_path.is_symlink():
        raise LocalLLMSettingsError("settings path contains a symlink")
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "schema_version": SETTINGS_SCHEMA_VERSION,
            "provider": settings.provider,
            "model": settings.model,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=selected_path.parent,
            prefix=f".{selected_path.name}.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            os.chmod(temporary_path, 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, selected_path)
        os.chmod(selected_path, 0o600)
    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise LocalLLMSettingsError("settings file could not be saved") from exc
    return selected_path
