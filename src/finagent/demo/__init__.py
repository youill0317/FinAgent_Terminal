"""데모 모드 공통 상수 및 오프라인 클라이언트."""

from __future__ import annotations

from typing import Final

import httpx

VALUE: Final = 999.0
SHARE: Final = VALUE / 100
COUNT: Final = int(VALUE)


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError(f"데모 모드: 네트워크 요청 없음 ({request.url.host})")


def offline() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(_refuse))
