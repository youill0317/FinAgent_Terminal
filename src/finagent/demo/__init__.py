"""``--demo``: every page's layout on fixed values, for checking the UI only.

Each page asks its usual entry point for data and, in the demo, gets a
module here instead: ``dashboard`` for the dashboard and the AI analysis,
``screen`` for the screen, ``portfolio`` for the portfolio's prices and
every calculation on them, ``training`` for the generator training jobs and
``workspace`` for the settings and the trade book. Nothing is fetched or
computed, and every number is ``VALUE``, so a demo value can never pass for
a real one. ``offline`` fails any request the demo would make.
"""

from __future__ import annotations

from typing import Final

import httpx

VALUE: Final = 999.0
SHARE: Final = VALUE / 100  # a share, return or probability: shows as 999%
COUNT: Final = int(VALUE)  # a count of days, paths or exceptions


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError(f"데모 모드: 네트워크 요청 없음 ({request.url.host})")


def offline() -> httpx.AsyncClient:
    """A client whose every request fails: the demo fetches nothing."""
    return httpx.AsyncClient(transport=httpx.MockTransport(_refuse))
