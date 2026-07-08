from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final, assert_never

from pydantic import BaseModel, ConfigDict

from finagent_terminal.reports.models import (
    DepthMode,
    ReportRoleArtifact,
    ReportRoleId,
    ReportRoleStatus,
    role_specs_for_depth,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

SectionState = ReportRoleStatus


_SECTION_ALIASES: Final[Mapping[str, tuple[ReportRoleId, str]]] = MappingProxyType(
    {role_id.value: (role_id, "") for role_id in ReportRoleId}
    | {
        "overview": (ReportRoleId.REPORTS_EDITOR, "Overview"),
        "risk_notes": (ReportRoleId.REPORTS_EDITOR, "Risk Notes"),
        "technical_indicators": (
            ReportRoleId.TECHNICAL_ANALYST,
            "Technical Indicators",
        ),
        "bull_case": (ReportRoleId.BULLISH_RESEARCHER, "Bull Case"),
        "bear_case": (ReportRoleId.BEARISH_RESEARCHER, "Bear Case"),
        "scenario_watch": (ReportRoleId.DEBATE_REFINEMENT, "Scenario Watch"),
        "open_questions": (ReportRoleId.REPORTS_EDITOR, "Open Questions"),
    },
)


class UnknownSectionError(Exception):
    section_id: str

    def __init__(self, *, section_id: str) -> None:
        """Create an unknown-section error."""
        self.section_id = section_id
        super().__init__(f"unknown report role: {section_id}")


class RoleProgress(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    id: ReportRoleId
    title: str
    status: ReportRoleStatus = ReportRoleStatus.NOT_STARTED
    artifact: ReportRoleArtifact | None = None
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None

    @property
    def state(self) -> ReportRoleStatus:
        return self.status


SectionProgress = RoleProgress


class ReportProgress(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    provider: str
    model: str
    depth: DepthMode
    started_at: float
    finished_at: float | None = None
    roles: tuple[RoleProgress, ...]
    latest_error: str | None = None

    @property
    def sections(self) -> tuple[RoleProgress, ...]:
        return self.roles

    @property
    def active_section(self) -> str | None:
        for role in self.roles:
            if role.status is ReportRoleStatus.RUNNING:
                return role.id
        return None

    @property
    def completed_count(self) -> int:
        return sum(1 for role in self.roles if role.status is ReportRoleStatus.COMPLETE)

    @property
    def total_count(self) -> int:
        return len(self.roles)

    def elapsed_seconds(self, now: float) -> float:
        end = self.finished_at if self.finished_at is not None else now
        return max(0.0, end - self.started_at)

    def section(self, section_id: str) -> RoleProgress:
        alias = _SECTION_ALIASES.get(section_id)
        if alias is None:
            raise UnknownSectionError(section_id=section_id)
        role_id, legacy_title = alias
        for role in self.roles:
            if role.id == role_id:
                if legacy_title:
                    return role.model_copy(update={"title": legacy_title})
                return role
        raise UnknownSectionError(section_id=section_id)

    def mark_running(self, section_id: str, now: float) -> ReportProgress:
        return self._replace(section_id, ReportRoleStatus.RUNNING, now=now)

    def mark_complete(
        self,
        section_id: str,
        now: float,
        artifact: ReportRoleArtifact | None = None,
    ) -> ReportProgress:
        return self._replace(
            section_id,
            ReportRoleStatus.COMPLETE,
            now=now,
            artifact=artifact,
        )

    def mark_failed(self, section_id: str, error: str, now: float) -> ReportProgress:
        return self._replace(section_id, ReportRoleStatus.FAILED, error=error, now=now)

    def mark_skipped(self, section_id: str, error: str) -> ReportProgress:
        return self._replace(
            section_id,
            ReportRoleStatus.SKIPPED,
            error=error,
            now=None,
        )

    def _replace(
        self,
        section_id: str,
        status: ReportRoleStatus,
        *,
        error: str | None = None,
        now: float | None,
        artifact: ReportRoleArtifact | None = None,
    ) -> ReportProgress:
        alias = _SECTION_ALIASES.get(section_id)
        if alias is None:
            raise UnknownSectionError(section_id=section_id)
        role_id, legacy_title = alias
        found = False
        roles: list[RoleProgress] = []
        for role in self.roles:
            if role.id != role_id:
                roles.append(role)
                continue
            found = True
            roles.append(
                _updated_role(
                    role,
                    status,
                    error=error,
                    now=now,
                    artifact=artifact,
                ),
            )
            if legacy_title:
                roles[-1] = roles[-1].model_copy(update={"title": legacy_title})
        if not found:
            raise UnknownSectionError(section_id=section_id)
        return self.model_copy(
            update={
                "roles": tuple(roles),
                "finished_at": _finished_at(tuple(roles), now),
                "latest_error": (
                    error if status is ReportRoleStatus.FAILED else self.latest_error
                ),
            },
        )


def _updated_role(
    role: RoleProgress,
    status: ReportRoleStatus,
    *,
    error: str | None,
    now: float | None,
    artifact: ReportRoleArtifact | None,
) -> RoleProgress:
    match status:
        case ReportRoleStatus.RUNNING:
            return role.model_copy(
                update={"status": status, "started_at": now, "error": None},
            )
        case ReportRoleStatus.COMPLETE:
            return role.model_copy(
                update={
                    "status": status,
                    "completed_at": now,
                    "error": None,
                    "artifact": artifact,
                },
            )
        case ReportRoleStatus.FAILED:
            return role.model_copy(
                update={"status": status, "completed_at": now, "error": error},
            )
        case ReportRoleStatus.SKIPPED:
            return role.model_copy(
                update={"status": status, "completed_at": now, "error": error},
            )
        case _:
            if status is ReportRoleStatus.NOT_STARTED:
                return role.model_copy(update={"status": status, "error": None})
            assert_never(status)


def _finished_at(
    roles: tuple[RoleProgress, ...],
    now: float | None,
) -> float | None:
    if now is None:
        return None
    terminal_states: Final = {
        ReportRoleStatus.COMPLETE,
        ReportRoleStatus.FAILED,
        ReportRoleStatus.SKIPPED,
    }
    if all(role.status in terminal_states for role in roles):
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
        roles=tuple(
            RoleProgress(id=role.id, title=role.title)
            for role in role_specs_for_depth(depth)
        ),
    )
