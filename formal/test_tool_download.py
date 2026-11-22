"""Pinned tool downloads tolerate transient release-host failures."""

import hashlib
import io
from urllib.error import HTTPError, URLError

import pytest

from formal.tla import check


def _lock(payload=b"tool"):
    return {"url": "https://example.com/tool?version=1",
            "sha256": hashlib.sha256(payload).hexdigest()}


@pytest.mark.parametrize("cached", [False, True])
def test_download_checks_content_and_reuses_cache(tmp_path, monkeypatch, cached):
    calls = []
    if cached:
        (tmp_path / "tool.jar").write_bytes(b"tool")

    def open_url(url, timeout):
        calls.append(url)
        return io.BytesIO(b"tool")

    monkeypatch.setattr(check.urllib.request, "urlopen", open_url)
    assert check.download(tmp_path, "tool.jar", _lock()).read_bytes() == b"tool"
    assert len(calls) == (0 if cached else 1)


@pytest.mark.parametrize("error", [HTTPError("url", code, "failure", {}, None)
                                   for code in (403, 408, 429, 500, 502, 503, 504)]
                         + [URLError("connection reset")])
def test_download_retries_transient_errors(tmp_path, monkeypatch, error):
    calls, delays = [], []

    def open_url(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise error
        return io.BytesIO(b"tool")

    monkeypatch.setattr(check.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(check.time, "sleep", delays.append)
    assert check.download(tmp_path, "tool.jar", _lock()).read_bytes() == b"tool"
    assert calls[0] == _lock()["url"]
    assert calls[1].startswith(_lock()["url"] + "&download_retry=")
    assert delays == [1]


@pytest.mark.parametrize("code, attempts", [(403, 4), (404, 1)])
def test_download_propagates_exhausted_or_permanent_errors(tmp_path, monkeypatch, code, attempts):
    calls = []

    def open_url(url, timeout):
        calls.append(url)
        raise HTTPError(url, code, "failure", {}, None)

    monkeypatch.setattr(check.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(check.time, "sleep", lambda delay: None)
    with pytest.raises(HTTPError):
        check.download(tmp_path, "tool.jar", _lock())
    assert len(calls) == attempts
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("payload", [b"", b"wrong tool"])
def test_download_rejects_invalid_content(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(check.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        check.download(tmp_path, "tool.jar", _lock())
