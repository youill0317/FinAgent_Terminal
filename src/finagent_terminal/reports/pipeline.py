from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from finagent_terminal.llm.base import ProviderError, StructuredLLMProvider
from finagent_terminal.reports.models import (
    DepthMode,
    ReportContext,
    ReportFinalEditorOutput,
    ReportRoleArtifact,
    ReportRoleId,
    ReportRoleStatus,
    ReportSectionPrompt,
    ReportSectionResult,
    ReportSectionSpec,
    role_specs_for_depth,
)
from finagent_terminal.reports.progress import ReportProgress, new_report_progress


class ReportRunResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    sections: tuple[ReportSectionResult, ...]
    progress: ReportProgress
    timeline: tuple[ReportProgress, ...]
    final_report: ReportFinalEditorOutput | None = None


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
        artifacts: list[ReportRoleArtifact] = []
        final_report: ReportFinalEditorOutput | None = None
        self._publish(sections, progress, timeline, final_report)

        for index, role in enumerate(role_specs_for_depth(depth), start=1):
            section = ReportSectionSpec(
                id=role.id,
                title=role.title,
                prompt=role.prompt,
            )
            now = started_at + float(index)
            progress = progress.mark_running(section.id, now=now)
            timeline.append(progress)
            self._publish(sections, progress, timeline, final_report)
            prompt = ReportSectionPrompt(
                context=context,
                depth=depth,
                section=section,
                prior_artifacts=tuple(artifacts),
            )
            try:
                result = self.client.generate_structured_report_section(prompt)
            except ProviderError as exc:
                progress = progress.mark_failed(section.id, str(exc), now=now + 0.5)
                timeline.append(progress)
                self._publish(sections, progress, timeline, final_report)
                continue
            sections.append(result)
            artifact = ReportRoleArtifact(
                role_id=role.id,
                title=role.title,
                status=ReportRoleStatus.COMPLETE,
                body=result.summary,
                summary=result.summary,
                key_points=result.key_points,
                risks=result.risks,
            )
            artifacts.append(artifact)
            if role.id is ReportRoleId.REPORTS_EDITOR:
                final_report = ReportFinalEditorOutput(
                    title=result.title,
                    body=result.summary,
                    disclaimer=result.disclaimer,
                )
            progress = progress.mark_complete(
                section.id,
                now=now + 0.5,
                artifact=artifact,
            )
            timeline.append(progress)
            self._publish(sections, progress, timeline, final_report)

        return _result(sections, progress, timeline, final_report)

    def _publish(
        self,
        sections: list[ReportSectionResult],
        progress: ReportProgress,
        timeline: list[ReportProgress],
        final_report: ReportFinalEditorOutput | None,
    ) -> None:
        if self.on_update is not None:
            self.on_update(_result(sections, progress, timeline, final_report))


def _result(
    sections: list[ReportSectionResult],
    progress: ReportProgress,
    timeline: list[ReportProgress],
    final_report: ReportFinalEditorOutput | None,
) -> ReportRunResult:
    return ReportRunResult(
        sections=tuple(sections),
        progress=progress,
        timeline=tuple(timeline),
        final_report=final_report,
    )


def run_report_pipeline(
    *,
    client: StructuredLLMProvider,
    context: ReportContext,
    depth: DepthMode,
) -> ReportRunResult:
    return ReportPipeline(client=client).run(context=context, depth=depth)
