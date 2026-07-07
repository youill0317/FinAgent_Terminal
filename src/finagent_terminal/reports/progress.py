from __future__ import annotations

from enum import StrEnum, unique
from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict

from finagent_terminal.reports.models import DepthMode, section_specs_for_depth


@unique
class SectionState(StrEnum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class UnknownSectionError(Exception):
    section_id: str

    def __init__(self, *, section_id: str) -> None:
        """Create an unknown-section error."""
        self.section_id = section_id
        super().__init__(f"unknown report section: {section_id}")


class SectionProgress(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    id: str
    title: str
    state: SectionState = SectionState.NOT_STARTED
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None


class ReportProgress(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    provider: str
    model: str
    depth: DepthMode
    started_at: float
    finished_at: float | None = None
    sections: tuple[SectionProgress, ...]
    latest_error: str | None = None

    @property
    def active_section(self) -> str | None:
        for section in self.sections:
            if section.state is SectionState.RUNNING:
                return section.id
        return None

    @property
    def completed_count(self) -> int:
        return sum(
            1 for section in self.sections if section.state is SectionState.COMPLETE
        )

    @property
    def total_count(self) -> int:
        return len(self.sections)

    def elapsed_seconds(self, now: float) -> float:
        end = self.finished_at if self.finished_at is not None else now
        return max(0.0, end - self.started_at)

    def section(self, section_id: str) -> SectionProgress:
        for section in self.sections:
            if section.id == section_id:
                return section
        raise UnknownSectionError(section_id=section_id)

    def mark_running(self, section_id: str, now: float) -> ReportProgress:
        return self._replace(section_id, SectionState.RUNNING, now=now)

    def mark_complete(self, section_id: str, now: float) -> ReportProgress:
        return self._replace(section_id, SectionState.COMPLETE, now=now)

    def mark_failed(self, section_id: str, error: str, now: float) -> ReportProgress:
        return self._replace(section_id, SectionState.FAILED, error=error, now=now)

    def mark_skipped(self, section_id: str, error: str) -> ReportProgress:
        return self._replace(section_id, SectionState.SKIPPED, error=error, now=None)

    def _replace(
        self,
        section_id: str,
        state: SectionState,
        *,
        error: str | None = None,
        now: float | None,
    ) -> ReportProgress:
        found = False
        sections: list[SectionProgress] = []
        for section in self.sections:
            if section.id != section_id:
                sections.append(section)
                continue
            found = True
            sections.append(_updated_section(section, state, error=error, now=now))
        if not found:
            raise UnknownSectionError(section_id=section_id)
        return self.model_copy(
            update={
                "sections": tuple(sections),
                "finished_at": _finished_at(tuple(sections), now),
                "latest_error": (
                    error if state is SectionState.FAILED else self.latest_error
                ),
            },
        )


def _updated_section(
    section: SectionProgress,
    state: SectionState,
    *,
    error: str | None,
    now: float | None,
) -> SectionProgress:
    updates: Final = {
        SectionState.RUNNING: {"state": state, "started_at": now, "error": None},
        SectionState.COMPLETE: {"state": state, "completed_at": now, "error": None},
        SectionState.FAILED: {"state": state, "completed_at": now, "error": error},
        SectionState.SKIPPED: {"state": state, "completed_at": now, "error": error},
        SectionState.NOT_STARTED: {"state": state, "error": None},
    }
    return section.model_copy(update=updates[state])


def _finished_at(
    sections: tuple[SectionProgress, ...],
    now: float | None,
) -> float | None:
    if now is None:
        return None
    terminal_states: Final = {
        SectionState.COMPLETE,
        SectionState.FAILED,
        SectionState.SKIPPED,
    }
    if all(section.state in terminal_states for section in sections):
        return now
    return None


def new_report_progress(
    *,
    provider: str,
    model: str,
    depth: DepthMode,
    started_at: float = 0.0,
) -> ReportProgress:
    return ReportProgress(
        provider=provider,
        model=model,
        depth=depth,
        started_at=started_at,
        sections=tuple(
            SectionProgress(id=section.id, title=section.title)
            for section in section_specs_for_depth(depth)
        ),
    )
