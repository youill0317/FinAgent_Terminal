from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

import httpx

from finagent import catalog
from finagent.analysis.context import build_document, fmt_value
from finagent.analysis.evidence import Evidence
from finagent.analysis.presets import PRESET_BY_ID, PRESETS
from finagent.analysis.prompt import Conversation, ToolRecord, Turn, system_prompt
from finagent.analysis.tools import MAX_POINTS, MAX_TOOL_CALLS, TOOL_NAMES, Toolbox
from finagent.analysis.verify import verify
from finagent.demo import DEMO_KEYS, demo_closes, demo_transport
from finagent.models import Fmt
from finagent.service import DashboardService

if TYPE_CHECKING:
    from collections.abc import Mapping

    from finagent.analysis.context import Document
    from finagent.analysis.tools import ToolResult

NOW = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def service(env: dict[str, str] | None = None) -> DashboardService:
    transport = demo_transport(NOW.date())
    return DashboardService(
        DEMO_KEYS if env is None else env,
        client_factory=lambda: httpx.AsyncClient(transport=transport),
        clock=lambda: NOW,
        retry_delay=0,
        yahoo=demo_closes(NOW.date()),
    )


def document(svc: DashboardService) -> Document:
    return build_document(asyncio.run(svc.get()), view="핵심")


def call(name: str, args: Mapping[str, object], svc: DashboardService) -> ToolResult:
    async def run() -> ToolResult:
        async with svc.fetch_context() as ctx:
            box = Toolbox(ctx, svc.series_cache, svc.today())
            return await box.run(name, args)

    return asyncio.run(run())


def body(result: ToolResult) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(result.text))


# ---------------------------------------------------------------- document


def test_document_has_every_shown_row_formatted_like_the_page() -> None:
    svc = service()
    dashboard = asyncio.run(svc.get())
    doc = build_document(dashboard, view="핵심")

    rows = [row for panel in dashboard.panels for row in panel.rows]
    assert len(rows) == len(catalog.CORE)
    for row in rows:
        assert f'<indicator id="{row.id}"' in doc.text
        assert row.id in doc.evidence
    unrate = next(row for row in rows if row.id == "us_unrate")
    assert unrate.value is not None
    assert f"최신 {unrate.date} {fmt_value(unrate.value, unrate.fmt)}" in doc.text
    assert {"regime_inflation", "regime_labor", "regime_curve", "regime_fincon"} <= set(
        doc.evidence
    )
    assert "sep_fedfunds" in doc.evidence
    assert '<indicator id="curve"' not in doc.text  # curve view is off by default
    assert doc.text == build_document(dashboard, view="핵심").text  # stable for caching


def test_rows_without_data_carry_status_and_reason_only() -> None:
    doc = document(service({"FRED_API_KEY": "demo"}))

    assert 'id="kr_base"' in doc.text
    assert 'status="missing_key"' in doc.text
    assert "ECOS_API_KEY" in doc.text
    assert "kr_base" not in doc.evidence


def test_format_matches_the_page() -> None:
    assert fmt_value(-0.52, Fmt(dec=2, signed=True)) == "−0.52"
    assert fmt_value(0.57, Fmt(suf="%p", dec=2, signed=True)) == "+0.57%p"
    assert fmt_value(1234567.0, Fmt(pre="$", suf="B", dec=0)) == "$1,234,567B"


# ---------------------------------------------------------------- verify


EVIDENCE: Final[dict[str, Evidence]] = {
    "us_unrate": Evidence("us_unrate", "실업률", (4.1, 4.2, 4.3)),
    "us_10y2y": Evidence("us_10y2y", "10년 − 2년", (0.54, 0.57)),
}


def counts(text: str) -> dict[str, int]:
    return {str(key): value for key, value in verify(text, EVIDENCE).counts.items()}


def test_cited_numbers_are_checked_at_the_precision_written() -> None:
    report = verify("실업률은 9월 4.3%로 직전 4.2%에서 올랐다. [[us_unrate]]", EVIDENCE)

    assert report.counts["ok"] == 2
    assert report.counts["mismatch"] == 0
    assert report.cited == {"us_unrate"}


def test_differences_basis_points_and_minus_signs_match() -> None:
    assert counts("0.1%p 올랐다(4.3 − 4.2 = 0.1). [[us_unrate]]")["mismatch"] == 0
    assert counts("금리차는 57bp다. [[us_10y2y]]") == {
        "ok": 1,
        "mismatch": 0,
        "unsupported": 0,
        "unknown_id": 0,
    }


def test_wrong_numbers_report_the_closest_real_value() -> None:
    report = verify("실업률은 4.5%까지 올랐다. [[us_unrate]]", EVIDENCE)

    lines = cast("list[list[dict[str, object]]]", report.to_json()["lines"])
    numbers = [piece for line in lines for piece in line if "n" in piece]
    assert numbers == [{"n": "4.5%", "s": "mismatch", "a": "4.3"}]


def test_uncited_and_unknown_ids_are_flagged() -> None:
    assert counts("연내 2회 인하가 예상된다.")["unsupported"] == 1
    assert counts("PCE는 2.8%다. [[us_pce_x]]")["unknown_id"] == 1


def test_dates_tenors_and_list_marks_are_not_values() -> None:
    text = "1. 2026-09 기준 10년물과 2년물, 3개월 전 2026Q2 값. [[us_unrate]]"

    assert sum(counts(text).values()) == 0


def test_a_mark_cites_only_its_own_sentence_but_a_whole_table_row() -> None:
    sentences = counts("CPI는 9.9%다. 실업률은 4.3%다. [[us_unrate]]")
    row = counts("| 실업률 | 4.3% | 4.2% | [[us_unrate]]")

    assert sentences == {"ok": 1, "mismatch": 0, "unsupported": 1, "unknown_id": 0}
    assert row["ok"] == 2


# ---------------------------------------------------------------- prompt and presets


def test_system_prompt_fills_snapshot_and_tool_limit() -> None:
    text = system_prompt("2026-10-12T09:00:00+00:00")

    assert "2026-10-12T09:00:00+00:00 시점" in text
    assert f"{MAX_TOOL_CALLS}회를 넘기지 않는다" in text
    assert "{snapshot}" not in text
    assert "{max_tools}" not in text


def test_conversation_puts_the_document_first_and_keeps_earlier_tool_data() -> None:
    record = ToolRecord(
        "get_series", '{"id":"us_unrate"}', '{"cite_as":"us_unrate"}', ok=True
    )
    talk = Conversation(
        system="s",
        document="<indicator_document/>",
        turns=(Turn("처음 질문", "첫 답 [[us_unrate]]", (record,)),),
        request="후속 질문",
    )

    messages = talk.messages()
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assert messages[0].parts[0] == "<indicator_document/>"
    assert "<request>\n처음 질문\n</request>" in messages[0].parts[1]
    assert '<tool_result name="get_series"' in messages[1].text
    assert messages[2].text == "<request>\n후속 질문\n</request>"


def test_presets_load_their_text_and_name_real_panels() -> None:
    panel_ids = {spec.id for spec in catalog.PANELS}

    for preset in PRESETS:
        assert preset.text.startswith("<task>")
        assert set(preset.panels) <= panel_ids
    assert [preset.id for preset in PRESETS] == ["macro"]
    macro = PRESET_BY_ID["macro"]
    assert {"markets", "commodities", "credit", "housing"} <= set(macro.panels)
    for anchor in ("매우 불리", "불리·주의", "혼재·보통", "유리", "매우 유리"):
        assert anchor in macro.text


# ---------------------------------------------------------------- tools


def test_tool_names_are_stable() -> None:
    assert TOOL_NAMES == ("search_series", "get_series", "source_guide")


def test_catalog_search_finds_indicators_outside_the_view() -> None:
    result = call("search_series", {"source": "catalog", "query": "JOLTS"}, service())

    hits = cast("list[dict[str, object]]", body(result)["results"])
    assert result.ok
    assert {hit["id"] for hit in hits} >= {"us_jolts", "us_quits"}


def test_get_series_by_id_reaches_back_further_than_the_page() -> None:
    result = call("get_series", {"id": "us_unrate", "start": "2012-01"}, service())

    data = body(result)
    points = cast("list[list[object]]", data["points"])
    assert result.ok
    assert data["cite_as"] == "us_unrate"
    assert points[0][0] == "2012-01"
    assert len(points) > 24
    assert result.evidence is not None
    assert len(result.evidence.points) == len(points)


def test_get_series_for_any_source_code() -> None:
    svc = service()
    fred = call(
        "get_series", {"source": "FRED", "code": "T10Y2Y", "start": "2018"}, svc
    )
    ecos = call(
        "get_series",
        {"source": "ECOS", "code": "817Y002/D/010200000", "start": "2024"},
        svc,
    )
    ecb = call(
        "get_series",
        {"source": "ECB", "code": "EXR/D.USD.EUR.SP00.A", "transform": "yoy"},
        svc,
    )

    assert body(fred)["cite_as"] == "FRED:T10Y2Y"
    assert body(ecos)["cite_as"] == "ECOS:817Y002/D/010200000"
    assert body(ecb)["transform"] == "yoy"
    assert all(result.ok for result in (fred, ecos, ecb))


def test_long_daily_series_are_averaged_down_to_the_point_cap() -> None:
    result = call(
        "get_series", {"source": "FRED", "code": "T10Y2Y", "start": "2016"}, service()
    )

    data = body(result)
    assert data["frequency"] in {"M", "Q"}
    assert cast("int", data["count"]) <= MAX_POINTS
    assert "note" in data


def test_search_and_structure_for_ecos_and_sdmx() -> None:
    svc = service()
    items = call(
        "search_series", {"source": "ECOS", "query": "", "dataset": "817Y002"}, svc
    )
    dims = call("search_series", {"source": "ECB", "query": "", "dataset": "EXR"}, svc)

    codes = [
        hit["code"] for hit in cast("list[dict[str, object]]", body(items)["results"])
    ]
    assert "817Y002/D/010200000" in codes
    assert len(cast("list[object]", body(dims)["results"])) >= 4


def test_tool_errors_are_results_not_exceptions() -> None:
    svc = service()

    unknown = call("get_series", {"id": "nope"}, svc)
    bad_code = call("get_series", {"source": "ECOS", "code": "bad"}, svc)
    no_key = call("get_series", {"id": "kr_base"}, service({"FRED_API_KEY": "demo"}))

    assert not unknown.ok
    assert "search_series" in cast("str", body(unknown)["error"])
    assert not bad_code.ok
    assert "ECOS_API_KEY" in cast("str", body(no_key)["error"])


def test_tool_calls_stop_at_the_cap() -> None:
    async def run() -> list[bool]:
        svc = service()
        async with svc.fetch_context() as ctx:
            box = Toolbox(ctx, svc.series_cache, svc.today())
            return [
                (await box.run("source_guide", {"source": "FRED"})).ok
                for _ in range(MAX_TOOL_CALLS + 1)
            ]

    assert asyncio.run(run()) == [True] * MAX_TOOL_CALLS + [False]


def test_source_guide_lists_dashboard_codes_and_docs_section() -> None:
    data = body(call("source_guide", {"source": "ECOS"}, service()))

    assert "ECOS:722Y001/0101000" in " ".join(
        cast("list[str]", data["dashboard_codes"])
    )
    assert cast("str", data["data_sources_md"]).startswith("## 4. 한국은행 ECOS")
