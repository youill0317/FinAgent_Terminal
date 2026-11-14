from __future__ import annotations

import json
from datetime import timedelta
from typing import TYPE_CHECKING, Final

import httpx

if TYPE_CHECKING:
    from datetime import date

SEP_RELEASE_ID: Final = 326
SEP_DAYS_AHEAD: Final = (37, 128, 219, 310)


def handle(request: httpx.Request, today: date) -> httpx.Response | None:
    path = request.url.path
    if request.url.host == "api.stlouisfed.org":
        if path == "/fred/series/release":
            return _json(
                {
                    "releases": [
                        {
                            "id": SEP_RELEASE_ID,
                            "name": "Summary of Economic Projections",
                        }
                    ]
                }
            )
        if path == "/fred/release/dates":
            return _json(
                {
                    "release_dates": [
                        {
                            "release_id": SEP_RELEASE_ID,
                            "date": (today + timedelta(days=days)).isoformat(),
                        }
                        for days in SEP_DAYS_AHEAD
                    ]
                }
            )
    return None


def _json(payload: object) -> httpx.Response:
    return httpx.Response(200, content=json.dumps(payload).encode())
