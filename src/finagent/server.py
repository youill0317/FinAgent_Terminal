from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import TYPE_CHECKING, Annotated

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response

from finagent import catalog

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from finagent.service import DashboardService

WEB = files("finagent") / "web"
STATIC_TYPES = {
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
}


def create_app(service: DashboardService, *, warm: bool = True) -> FastAPI:
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
        deep: Annotated[str, Query()] = "",
    ) -> Response:
        views = frozenset(deep.split(",")) & catalog.ALL_VIEWS
        dashboard = await service.get(refresh=refresh, deep=views)
        return Response(
            dashboard.model_dump_json(),
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )

    return app
