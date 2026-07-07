from __future__ import annotations

from enum import StrEnum, unique
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

REPORT_DISCLAIMER: Final = "Research assistance only; not investment advice."


@unique
class DepthMode(StrEnum):
    QUICK = "Quick"
    STANDARD = "Standard"
    DEEP = "Deep"


class ReportContext(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1, max_length=32)
    market_summary: str = Field(min_length=1)
    indicator_summary: str = ""
    warnings: tuple[str, ...] = ()


class ReportSectionSpec(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    requires_indicators: bool = False


class ReportSectionPrompt(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    context: ReportContext
    depth: DepthMode
    section: ReportSectionSpec

    def render(self) -> str:
        warnings = "\n".join(f"- {warning}" for warning in self.context.warnings)
        warning_text = warnings or "- none"
        return (
            "Create one structured equity research report section.\n"
            f"Depth: {self.depth.value}\n"
            f"Symbol: {self.context.symbol}\n"
            f"Section: {self.section.title}\n"
            f"Section instructions: {self.section.prompt}\n\n"
            f"Market context:\n{self.context.market_summary}\n\n"
            "Indicator context:\n"
            f"{self.context.indicator_summary or 'not supplied'}\n\n"
            f"Warnings:\n{warning_text}\n\n"
            "Return JSON with: section_id, title, summary, key_points, risks, "
            "disclaimer. The disclaimer must state this is research assistance "
            "only and not investment advice."
        )


class ReportSectionResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    key_points: tuple[str, ...] = Field(min_length=1)
    risks: tuple[str, ...] = ()
    disclaimer: str = REPORT_DISCLAIMER

    @field_validator("disclaimer")
    @classmethod
    def require_non_advice_disclaimer(cls, value: str) -> str:
        normalized = value.lower()
        has_research_label = "research assistance" in normalized
        has_advice_warning = "not investment advice" in normalized
        if not has_research_label or not has_advice_warning:
            msg = (
                "report disclaimer must say research assistance only and "
                "not investment advice"
            )
            raise ValueError(msg)
        return value


QUICK_SECTIONS: Final = (
    ReportSectionSpec(
        id="overview",
        title="Overview",
        prompt="Summarize the current setup using only the supplied context.",
    ),
    ReportSectionSpec(
        id="risk_notes",
        title="Risk Notes",
        prompt="List the most relevant research risks and missing-data caveats.",
    ),
)

STANDARD_EXTRA_SECTIONS: Final = (
    ReportSectionSpec(
        id="technical_indicators",
        title="Technical Indicators",
        prompt="Explain the supplied indicator context without inventing values.",
        requires_indicators=True,
    ),
    ReportSectionSpec(
        id="bull_case",
        title="Bull Case",
        prompt="Describe plausible upside arguments from the supplied context.",
    ),
    ReportSectionSpec(
        id="bear_case",
        title="Bear Case",
        prompt="Describe plausible downside arguments from the supplied context.",
    ),
)

DEEP_EXTRA_SECTIONS: Final = (
    ReportSectionSpec(
        id="scenario_watch",
        title="Scenario Watch",
        prompt=(
            "List scenarios to monitor and what context would confirm or "
            "weaken each."
        ),
    ),
    ReportSectionSpec(
        id="open_questions",
        title="Open Questions",
        prompt="Identify research questions that need fresh external data later.",
    ),
)


def section_specs_for_depth(depth: DepthMode) -> tuple[ReportSectionSpec, ...]:
    section_map: Final = {
        DepthMode.QUICK: QUICK_SECTIONS,
        DepthMode.STANDARD: QUICK_SECTIONS + STANDARD_EXTRA_SECTIONS,
        DepthMode.DEEP: QUICK_SECTIONS + STANDARD_EXTRA_SECTIONS + DEEP_EXTRA_SECTIONS,
    }
    return section_map[depth]
