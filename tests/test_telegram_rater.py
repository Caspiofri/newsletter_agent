"""Offline tests for the Telegram rating bot. Run from the repo root: python -m tests.test_telegram_rater
No network and no real token: the Bot API is a callable passed in and faked."""
import requests

from evals import telegram_api

TOKEN = "123456:SECRET-test-token"


class _Resp:
    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


def test_api_returns_result():
    calls = []

    def post(url, json=None, timeout=None):
        calls.append((url, json, timeout))
        return _Resp({"ok": True, "result": {"message_id": 7}})

    api = telegram_api.make_api(TOKEN, post=post, timeout=5)
    assert api("sendMessage", {"chat_id": "1", "text": "hi"}) == {"message_id": 7}
    assert calls == [(f"https://api.telegram.org/bot{TOKEN}/sendMessage", {"chat_id": "1", "text": "hi"}, 5)]


def test_api_raises_on_not_ok():
    api = telegram_api.make_api(TOKEN, post=lambda url, json=None, timeout=None:
                                _Resp({"ok": False, "description": "Bad Request: message is too long"}))
    try:
        api("sendMessage", {})
    except telegram_api.TelegramError as e:
        assert "message is too long" in str(e)
    else:
        raise AssertionError("no TelegramError")


def test_api_error_hides_token():
    def post(url, json=None, timeout=None):
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}")

    api = telegram_api.make_api(TOKEN, post=post)
    try:
        api("getUpdates", {})
    except telegram_api.TelegramError as e:
        assert TOKEN not in str(e) and "SECRET" not in str(e)
        assert "getUpdates" in str(e)
    else:
        raise AssertionError("no TelegramError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
