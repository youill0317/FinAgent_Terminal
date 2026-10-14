"""A minimal MCP server (streamable HTTP, JSON responses) for the Codex run.

Only what a client needs to list and call FinAgent's data tools: initialize,
ping, tools/list and tools/call. Requests must carry the bearer token of the
run in progress, so only the Codex process FinAgent started can call in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from finagent import __version__
from finagent.analysis.providers.base import ProviderError
from finagent.analysis.tools import TOOL_SPECS

if TYPE_CHECKING:
    from finagent.analysis.service import AnalysisService

type Json = dict[str, object]

DEFAULT_PROTOCOL: Final = "2025-06-18"
METHOD_NOT_FOUND: Final = -32601
INVALID_REQUEST: Final = -32600


async def handle(analysis: AnalysisService, message: object) -> Json | None:
    """Answer one JSON-RPC message; ``None`` for notifications."""
    if not isinstance(message, dict):
        return _error(None, INVALID_REQUEST, "JSON-RPC 객체가 아닙니다")
    body = cast("dict[str, object]", message)
    if "id" not in body:
        return None
    ident = body.get("id")
    method = body.get("method")
    params = body.get("params")
    args = cast("dict[str, object]", params) if isinstance(params, dict) else {}
    if method == "tools/call":
        return _result(ident, await _call(analysis, args))
    answer = {"initialize": _initialize, "ping": _ping, "tools/list": _list}.get(
        method if isinstance(method, str) else ""
    )
    if answer is None:
        return _error(ident, METHOD_NOT_FOUND, f"지원하지 않는 메서드: {method}")
    return _result(ident, answer(args))


def _initialize(args: dict[str, object]) -> Json:
    version = args.get("protocolVersion")
    return {
        "protocolVersion": version if isinstance(version, str) else DEFAULT_PROTOCOL,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": "finagent", "version": __version__},
    }


def _ping(_args: dict[str, object]) -> Json:
    return {}


def _list(_args: dict[str, object]) -> Json:
    return {
        "tools": [
            {
                "name": name,
                "description": description,
                "inputSchema": schema,
                "annotations": {"readOnlyHint": True},
            }
            for name, description, schema in TOOL_SPECS
        ]
    }


async def _call(analysis: AnalysisService, args: dict[str, object]) -> Json:
    name = args.get("name")
    arguments = args.get("arguments")
    try:
        result = await analysis.mcp_call(
            name if isinstance(name, str) else "",
            cast("dict[str, object]", arguments) if isinstance(arguments, dict) else {},
        )
    except ProviderError as error:
        return {"content": [{"type": "text", "text": str(error)}], "isError": True}
    return {
        "content": [{"type": "text", "text": result.text}],
        "isError": not result.ok,
    }


def _result(ident: object, result: Json) -> Json:
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def _error(ident: object, code: int, message: str) -> Json:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}
