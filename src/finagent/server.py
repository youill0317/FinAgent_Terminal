from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from finagent.analysis.providers.base import ProviderError
from finagent.analysis.service import RunInput

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from finagent.analysis.service import AnalysisService
    from finagent.models import Dashboard
    from finagent.service import DashboardService

WEB = files("finagent") / "web"
STATIC_TYPES = {
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "analysis.js": "text/javascript; charset=utf-8",
}
NO_STORE = {"Cache-Control": "no-store"}


class ViewBody(BaseModel):
    deep: list[str]


class ChoiceBody(BaseModel):
    provider: str
    model: str = ""


class AnalysisBody(BaseModel):
    request: str
    provider: str = ""
    model: str = ""
    conversation: str = ""
    preset: str = ""
    scope: Literal["view", "all"] = "view"


def create_app(
    service: DashboardService,
    *,
    warm: bool = True,
    analysis: AnalysisService | None = None,
) -> FastAPI:
    """Build the local web app. ``warm`` starts the first fetch at startup."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        task = asyncio.ensure_future(service.get()) if warm else None
        yield
        if task is not None and not task.done():
            _ = task.cancel()

    app = FastAPI(title="FinAgent", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:  # pyright: ignore[reportUnusedFunction]
        return HTMLResponse((WEB / "index.html").read_text(encoding="utf-8"))

    @app.get("/static/{name}")
    async def static(name: str) -> Response:  # pyright: ignore[reportUnusedFunction]
        media_type = STATIC_TYPES.get(name)
        if media_type is None:
            return Response(status_code=404)
        return Response((WEB / name).read_text(encoding="utf-8"), media_type=media_type)

    @app.get("/api/dashboard")
    async def dashboard(  # pyright: ignore[reportUnusedFunction]
        refresh: Annotated[bool, Query()] = False,
    ) -> Response:
        return _json(await service.get(refresh=refresh))

    @app.put("/api/settings")
    async def settings(body: ViewBody) -> Response:  # pyright: ignore[reportUnusedFunction]
        return _json(await service.set_view(frozenset(body.deep)))

    if analysis is not None:
        _analysis_routes(app, analysis)
    return app


def _analysis_routes(app: FastAPI, analysis: AnalysisService) -> None:
    @app.get("/api/analysis/meta")
    async def meta() -> Response:  # pyright: ignore[reportUnusedFunction]
        return JSONResponse(await analysis.meta(), headers=NO_STORE)

    @app.put("/api/analysis/choice")
    async def choice(body: ChoiceBody) -> Response:  # pyright: ignore[reportUnusedFunction]
        try:
            picked = analysis.choose(body.provider, body.model)
        except ProviderError as error:
            return JSONResponse({"error": str(error)}, status_code=400)
        except OSError as error:
            return JSONResponse({"error": f"설정을 저장하지 못했습니다: {error}"}, 500)
        return JSONResponse({"provider": picked.provider, "models": picked.models})

    @app.post("/api/analysis")
    async def run(body: AnalysisBody) -> Response:  # pyright: ignore[reportUnusedFunction]
        given = RunInput(
            request=body.request,
            provider=body.provider,
            model=body.model,
            conversation=body.conversation,
            preset=body.preset,
            scope=body.scope,
        )

        async def events() -> AsyncGenerator[str]:
            async for event in analysis.run(given):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            events(), media_type="text/event-stream", headers=NO_STORE
        )


def _json(dashboard: Dashboard) -> Response:
    return Response(
        dashboard.model_dump_json(),
        media_type="application/json",
        headers=NO_STORE,
    )
