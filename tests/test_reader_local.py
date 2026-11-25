"""手元の読む役（``shokoban.reader_local``）の試験.

本物の HTTP は使わない。送り先は偽の opener。文はすべて架空。
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from shokoban.reader_local import DEFAULT_LOCAL_URL, check_local_url, make_local_reader, run_local
from shokoban.reader_stage5 import ReaderReply, ReaderStage5Error


class FakeResponse:
    def __init__(self, body: Any, status: int = 200) -> None:
        self._body = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class Opener:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[Any, Any]] = []

    def __call__(self, request: Any, timeout: Any = None) -> Any:
        self.calls.append((request, timeout))
        if self.error is not None:
            raise self.error
        return self.result


OK = {"model": "served-model", "choices": [{"message": {"role": "assistant", "content": "{\"items\":[]}"}}]}


def test_success_posts_chat_completions_and_returns_reply() -> None:
    opener = Opener(FakeResponse(OK))
    reply = run_local("指示文", base_url=DEFAULT_LOCAL_URL, model="m1", timeout=7, opener=opener)
    assert reply == ReaderReply(text='{"items":[]}', model="served-model")
    request, timeout = opener.calls[0]
    assert request.full_url == "http://127.0.0.1:1234/v1/chat/completions"
    assert request.get_method() == "POST"
    assert timeout == 7
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == "m1" and body["stream"] is False
    assert body["messages"] == [{"role": "user", "content": "指示文"}]


def test_model_name_falls_back_to_configured() -> None:
    payload = {"choices": [{"message": {"content": "x"}}]}
    reply = make_local_reader(base_url="http://localhost:11434/v1/", model="m1", opener=Opener(FakeResponse(payload)))("p")
    assert reply == ReaderReply(text="x", model="m1")


def test_http_error_is_reported_without_body() -> None:
    err = urllib.error.HTTPError("http://127.0.0.1:1234/v1/chat/completions", 500, "boom", {}, io.BytesIO(b"secret"))  # type: ignore[arg-type]
    with pytest.raises(ReaderStage5Error) as caught:
        run_local("p", base_url=DEFAULT_LOCAL_URL, model="m1", opener=Opener(error=err))
    assert "500" in str(caught.value) and "secret" not in str(caught.value)


def test_status_error_and_connection_error() -> None:
    with pytest.raises(ReaderStage5Error):
        run_local("p", base_url=DEFAULT_LOCAL_URL, model="m1", opener=Opener(FakeResponse(OK, status=503)))
    with pytest.raises(ReaderStage5Error):
        run_local("p", base_url=DEFAULT_LOCAL_URL, model="m1", opener=Opener(error=urllib.error.URLError("refused")))


@pytest.mark.parametrize(
    "body",
    [b"not json", b"[]", {"choices": []}, {"choices": [{"message": {}}]}, {"choices": [{"message": {"content": 5}}]}],
)
def test_broken_json_or_shape_is_rejected(body: Any) -> None:
    with pytest.raises(ReaderStage5Error):
        run_local("p", base_url=DEFAULT_LOCAL_URL, model="m1", opener=Opener(FakeResponse(body)))


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1",
        "https://api.openai.com/v1",
        "http://192.168.0.5:1234/v1",
        "http://127.0.0.1.evil.example/v1",
        "http://localhost@evil.example/v1",
        "http://user:pw@127.0.0.1:1234/v1",
        "ftp://127.0.0.1/v1",
        "127.0.0.1:1234/v1",
        "",
    ],
)
def test_non_local_host_is_refused_and_nothing_is_sent(url: str) -> None:
    opener = Opener(FakeResponse(OK))
    with pytest.raises(ReaderStage5Error):
        run_local("p", base_url=url, model="m1", opener=opener)
    with pytest.raises(ReaderStage5Error):
        make_local_reader(base_url=url, model="m1", opener=opener)
    assert opener.calls == []


@pytest.mark.parametrize("url", ["http://127.0.0.1:1234/v1", "http://localhost:8080/v1", "http://[::1]:1234/v1"])
def test_local_hosts_are_accepted(url: str) -> None:
    assert check_local_url(url + "/") == url


def test_empty_prompt_or_model_is_rejected() -> None:
    opener = Opener(FakeResponse(OK))
    with pytest.raises(ReaderStage5Error):
        run_local("  ", base_url=DEFAULT_LOCAL_URL, model="m1", opener=opener)
    with pytest.raises(ReaderStage5Error):
        make_local_reader(base_url=DEFAULT_LOCAL_URL, model=" ")
    assert opener.calls == []
