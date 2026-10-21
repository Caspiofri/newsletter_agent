"""Minimal Telegram Bot API client: one function per bot, api(method, params) -> result.

Errors are raised as TelegramError with the method name and Telegram's description; the token (which is part of
every request URL) never appears in the message, so errors can be printed and logged as they are.
"""
from typing import Callable

import requests

API_URL = "https://api.telegram.org/bot{token}/{method}"


class TelegramError(Exception):
    pass


def make_api(token: str, post=requests.post, timeout: int = 60) -> Callable[[str, dict], dict]:
    def api(method: str, params: dict) -> dict:
        try:
            body = post(API_URL.format(token=token, method=method), json=params, timeout=timeout).json()
        except (requests.RequestException, ValueError) as e:
            # requests puts the full URL (with the token) in its messages; the cause is dropped for the same reason
            raise TelegramError(f"{method}: {type(e).__name__}: {str(e).replace(token, '<token>')}") from None
        if not body.get("ok"):
            raise TelegramError(f"{method}: {body.get('description', 'unknown error')}")
        return body["result"]
    return api
