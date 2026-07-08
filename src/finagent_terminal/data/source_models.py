from __future__ import annotations

from datetime import datetime  # noqa: TC003
from enum import StrEnum, unique
from typing import ClassVar

from pydantic import BaseModel, ConfigDict


@unique
class SourceStatusValue(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    WARNING = "warning"


class SourceStatus(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source: str
    status: SourceStatusValue
    message: str | None = None
    throttle: str | None = None
    retry_after: str | None = None


class SourceAvailability(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source: str
    env_var: str
    configured: bool
    required: bool

    @property
    def status(self) -> SourceStatusValue:
        if self.configured:
            return SourceStatusValue.AVAILABLE
        return SourceStatusValue.UNAVAILABLE


class SourceItem(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source: str
    title: str
    summary: str | None = None
    url: str | None = None
    published_at: datetime | None = None


class SourceRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source: str
    method: str
    url: str
    status_code: int | None = None
    throttle: str | None = None
    retry_after: str | None = None


class SourceBundle(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    source: str
    items: tuple[SourceItem, ...] = ()
    statuses: tuple[SourceStatus, ...] = ()
    warnings: tuple[str, ...] = ()
    requests: tuple[SourceRequest, ...] = ()
