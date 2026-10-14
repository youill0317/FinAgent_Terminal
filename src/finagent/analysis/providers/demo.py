"""A scripted provider for ``--demo`` and the tests. No model is called.

It reads values out of the run's evidence, calls one tool the way a model
would, and writes an answer that exercises every verifier outcome.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Final, cast

from finagent.analysis.providers.base import ProviderStatus, TextDelta, Usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from finagent.analysis.providers.base import (
        Kind,
        ProviderEvent,
        ProviderId,
        RunRequest,
    )

PICKS: Final = ("us_cpi", "us_core_pce", "us_unrate", "us_10y2y", "kr_base")
CHUNK: Final = 14
DELAY_SECONDS: Final = 0.01

INTRO: Final = "데모 공급자의 예시 답입니다. 모델 없이 규칙대로 문서 값을 씁니다."
SUMMARY: Final = "**결론**: 핵심 지표를 표로 옮기고 도구로 긴 이력을 받았습니다."


class DemoProvider:
    id: ProviderId = "demo"
    title: str = "데모 (모델 없음)"
    kind: Kind = "demo"
    default_model: str = "demo"

    def __init__(self, delay: float = DELAY_SECONDS) -> None:
        """``delay`` spaces out text chunks so the page shows streaming."""
        self._delay: float = delay

    async def status(self) -> ProviderStatus:
        return ProviderStatus("ready", "예시 답을 만드는 데모 공급자")

    async def stream(self, request: RunRequest) -> AsyncIterator[ProviderEvent]:
        evidence = request.evidence
        result = await request.run_tool(
            "get_series", {"id": "us_unrate", "start": "2016-01"}
        )
        long_run = ""
        if result.ok:
            body = cast("dict[str, object]", json.loads(result.text))
            points = cast("list[list[float]]", body["points"])
            values = [float(point[1]) for point in points]
            low, high = _num(min(values)), _num(max(values))
            since = points[0][0]
            long_run = f"도구로 받은 실업률은 {since}부터 {low}%~{high}% 범위였다."
            long_run += " [[us_unrate]]"
        rows = [
            _row(item.label, item.points[-1], item.points[-2], item.cite)
            for cite in PICKS
            if (item := evidence.get(cite)) is not None and len(item.points) >= 2  # noqa: PLR2004
        ]
        lines = [
            INTRO,
            "",
            SUMMARY,
            "",
            "| 지표 | 최신 | 직전 | 근거 |",
            "|---|---|---|---|",
            *rows,
            "",
            long_run,
            "연내 2회 인하 여부는 이 문서만으로 판단할 수 없다.",
            "(검증 예시) 일부러 틀린 값: 실업률 9.9%. [[us_unrate]]",
        ]
        text = "\n".join(lines)
        for start in range(0, len(text), CHUNK):
            if self._delay:
                await asyncio.sleep(self._delay)
            yield TextDelta(text[start : start + CHUNK])
        yield Usage(note="데모 · 모델 호출 없음")


def _num(value: float) -> str:
    return f"{value:,.1f}" if abs(value) < 1000 else f"{value:,.0f}"  # noqa: PLR2004


def _row(label: str, last: float, prev: float, cite: str) -> str:
    return f"| {label} | {_num(last)} | {_num(prev)} | [[{cite}]]"
