"""手元の読む役: OpenAI 互換の ``/v1/chat/completions`` に1回送る.

llama-server・LM Studio・Ollama などの互換口を想定する。**手元だけで完結する経路**なので、
宛先のホストが ``127.0.0.1`` / ``localhost`` / ``::1`` 以外なら送らずに断る（外のサーバーへ
送らないため）。リダイレクトは追わず、環境のプロキシも使わない。

HTTP の呼び出しは ``opener`` で差し替えられる（テストでは偽物。既定は標準ライブラリの urllib）。
この module は import 時に何も送らない。呼んだときだけ動く。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from .reader_stage5 import ReaderReply, ReaderStage5Error

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
DEFAULT_LOCAL_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LOCAL_TIMEOUT = 1800  # 秒（クラウドの読む役と同じ）
CHAT_PATH = "/chat/completions"


def check_local_url(base_url: Any) -> str:
    """宛先が手元（loopback）の http(s) であることを確かめ、末尾の ``/`` を除いた URL を返す."""
    if type(base_url) is not str or not base_url.strip():
        raise ReaderStage5Error("手元の読む役の URL が空です")
    url = base_url.strip()
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
        parts.port  # 番号が不正なら ValueError
    except ValueError as exc:
        raise ReaderStage5Error("手元の読む役の URL を読めません") from exc
    if parts.scheme not in ("http", "https"):
        raise ReaderStage5Error("手元の読む役の URL は http:// か https:// で始めてください")
    if host not in LOCAL_HOSTS:
        raise ReaderStage5Error(
            "手元の読む役は、この PC の中（127.0.0.1・localhost・::1）のサーバーにだけ送れます。"
            "外のサーバーには送りません"
        )
    if parts.username is not None or parts.password is not None:
        raise ReaderStage5Error("手元の読む役の URL に認証情報は書けません")
    return url.rstrip("/")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        return None


def _default_opener(request: Any, timeout: float) -> Any:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
    return opener.open(request, timeout=timeout)


def _choice_text(payload: Any) -> str:
    if type(payload) is not dict:
        raise ReaderStage5Error("手元の読む役の返答がオブジェクトではありません")
    choices = payload.get("choices")
    if type(choices) is not list or not choices or type(choices[0]) is not dict:
        raise ReaderStage5Error("手元の読む役の返答に choices がありません")
    message = choices[0].get("message")
    content = message.get("content") if type(message) is dict else None
    if type(content) is not str:
        raise ReaderStage5Error("手元の読む役の返答に本文（message.content）がありません")
    return content


def run_local(
    prompt: str,
    *,
    base_url: str,
    model: str,
    timeout: float = DEFAULT_LOCAL_TIMEOUT,
    opener: Callable[..., Any] | None = None,
) -> ReaderReply:
    """手元の互換サーバーに ``prompt`` を1回送り、返事とモデル名を返す."""
    if type(prompt) is not str or not prompt.strip():
        raise ReaderStage5Error("読む役に渡す指示文が空です")
    if type(model) is not str or not model.strip():
        raise ReaderStage5Error("手元の読む役のモデル名が空です")
    endpoint = check_local_url(base_url) + CHAT_PATH
    body = json.dumps(
        {
            "model": model.strip(),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "stream": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    call = opener or _default_opener
    try:
        with call(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise ReaderStage5Error(f"手元の読む役が HTTP {exc.code} を返しました") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ReaderStage5Error(f"手元の読む役に届きません: {type(exc).__name__}") from exc
    if type(status) is int and status >= 400:
        raise ReaderStage5Error(f"手元の読む役が HTTP {status} を返しました")
    text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise ReaderStage5Error("手元の読む役の返答を JSON として読めません") from exc
    content = _choice_text(payload)
    returned = payload.get("model")
    name = returned.strip() if type(returned) is str and returned.strip() else model.strip()
    return ReaderReply(text=content, model=name)


def make_local_reader(
    *,
    base_url: str,
    model: str,
    timeout: float = DEFAULT_LOCAL_TIMEOUT,
    opener: Callable[..., Any] | None = None,
) -> Callable[[str], ReaderReply]:
    """手元の読む役を ``reader`` として渡せる関数にする（宛先と model は作る時に検査する）."""
    check_local_url(base_url)
    if type(model) is not str or not model.strip():
        raise ReaderStage5Error("手元の読む役のモデル名が空です")

    def reader(prompt: str) -> ReaderReply:
        return run_local(prompt, base_url=base_url, model=model, timeout=timeout, opener=opener)

    return reader
