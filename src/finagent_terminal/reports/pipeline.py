from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from finagent_terminal.llm.base import ProviderError, StructuredLLMProvider
from finagent_terminal.reports.models import (
    DepthMode,
    ReportContext,
    ReportSectionPrompt,
    ReportSectionResult,
    section_specs_for_depth,
)
from finagent_terminal.reports.progress import ReportProgress, new_report_progress


class ReportRunResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    sections: tuple[ReportSectionResult, ...]
    progress: ReportProgress
    timeline: tuple[ReportProgress, ...]


type ReportUpdate = Callable[[ReportRunResult], None]


@dataclass(frozen=True, slots=True)
class ReportPipeline:
    client: StructuredLLMProvider
    provider: str = "fake"
    model: str = "fake-model"
    on_update: ReportUpdate | None = None

    def run(
        self,
        context: ReportContext,
        depth: DepthMode,
        started_at: float = 0.0,
    ) -> ReportRunResult:
        progress = new_report_progress(
            provider=self.provider,
            model=self.model,
            depth=depth,
            started_at=started_at,
        )
        timeline: list[ReportProgress] = [progress]
        sections: list[ReportSectionResult] = []
        self._publish(sections, progress, timeline)

        for index, section in enumerate(section_specs_for_depth(depth), start=1):
            if section.requires_indicators and not context.indicator_summary.strip():
                progress = progress.mark_skipped(
                    section.id,
                    "missing indicator context",
                )
                timeline.append(progress)
                self._publish(sections, progress, timeline)
                continue

            now = started_at + float(index)
            progress = progress.mark_running(section.id, now=now)
            timeline.append(progress)
            self._publish(sections, progress, timeline)
            prompt = ReportSectionPrompt(context=context, depth=depth, section=section)
            try:
                result = self.client.generate_structured_report_section(prompt)
            except ProviderError as exc:
                progress = progress.mark_failed(section.id, str(exc), now=now + 0.5)
                timeline.append(progress)
                self._publish(sections, progress, timeline)
                continue
            sections.append(result)
            progress = progress.mark_complete(section.id, now=now + 0.5)
            timeline.append(progress)
            self._publish(sections, progress, timeline)

        return _result(sections, progress, timeline)

    def _publish(
        self,
        sections: list[ReportSectionResult],
        progress: ReportProgress,
        timeline: list[ReportProgress],
    ) -> None:
        if self.on_update is not None:
            self.on_update(_result(sections, progress, timeline))


def _result(
    sections: list[ReportSectionResult],
    progress: ReportProgress,
    timeline: list[ReportProgress],
) -> ReportRunResult:
    return ReportRunResult(
        sections=tuple(sections),
        progress=progress,
        timeline=tuple(timeline),
    )


def run_report_pipeline(
    *,
    client: StructuredLLMProvider,
    context: ReportContext,
    depth: DepthMode,
) -> ReportRunResult:
    return ReportPipeline(client=client).run(context=context, depth=depth)
