"""bootstrap 指纹缓存：只为省掉每轮重复的首页预热，绝不能影响正确性。

上游首页预热（GET /）只用来拿 PoW 的脚本指纹，实测该指纹在账号间完全一致。
这里锁住三条行为：命中缓存不再发请求、缓存过期会重新取、取不到指纹时退回默认脚本。
"""
import time

import pytest

from services import openai_backend_api as backend_module
from services.openai_backend_api import DEFAULT_POW_SCRIPT, OpenAIBackendAPI


class _FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code
        self.headers = {}
        self.content = text.encode()

    def json(self):
        return {}


class _FakeSession:
    def __init__(self, text):
        self.text = text
        self.calls = 0
        self.headers = {
            "Sec-Ch-Ua": '"Chromium";v="131"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
        }

    def get(self, *args, **kwargs):
        self.calls += 1
        return _FakeResponse(self.text)

    def post(self, *args, **kwargs):
        raise AssertionError("bootstrap should not POST")


PAGE_WITH_SCRIPTS = (
    '<html data-build="prod-deadbeefcafe">'
    '<script src="/cdn/assets/c/abc123/_next/static/chunks/main.js"></script>'
    "</html>"
)


@pytest.fixture(autouse=True)
def _clear_cache():
    backend_module._pow_fingerprint_cache.clear()
    yield
    backend_module._pow_fingerprint_cache.clear()


def _backend(session):
    backend = OpenAIBackendAPI.__new__(OpenAIBackendAPI)
    backend.base_url = "https://chatgpt.com"
    backend.session = session
    backend.user_agent = "test-agent"
    backend.pow_script_sources = []
    backend.pow_data_build = ""
    return backend


def test_first_bootstrap_fetches_and_second_uses_cache():
    session = _FakeSession(PAGE_WITH_SCRIPTS)
    first = _backend(session)
    first._bootstrap()
    assert session.calls == 1
    assert first.pow_data_build

    second = _backend(session)
    second._bootstrap()
    assert session.calls == 1, "缓存命中后不应再请求首页"
    assert second.pow_data_build == first.pow_data_build
    assert second.pow_script_sources == first.pow_script_sources


def test_expired_cache_refetches():
    session = _FakeSession(PAGE_WITH_SCRIPTS)
    _backend(session)._bootstrap()
    assert session.calls == 1

    backend_module._pow_fingerprint_cache["at"] = (
        time.monotonic() - backend_module._POW_FINGERPRINT_TTL_SECS - 1
    )
    _backend(session)._bootstrap()
    assert session.calls == 2, "缓存过期后应重新取指纹"


def test_missing_scripts_falls_back_to_default():
    session = _FakeSession("<html><body>no scripts here</body></html>")
    backend = _backend(session)
    backend._bootstrap()
    assert backend.pow_script_sources == [DEFAULT_POW_SCRIPT]


def test_cached_fingerprint_is_copied_not_shared():
    session = _FakeSession(PAGE_WITH_SCRIPTS)
    first = _backend(session)
    first._bootstrap()
    first.pow_script_sources.append("https://example.com/mutated.js")

    second = _backend(session)
    second._bootstrap()
    assert "https://example.com/mutated.js" not in second.pow_script_sources
