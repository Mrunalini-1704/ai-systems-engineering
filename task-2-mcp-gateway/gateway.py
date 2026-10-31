"""MCP Security Gateway.

Sits between an AI agent client and a downstream MCP server (JSON-RPC 2.0
over HTTP, MCP "Streamable HTTP" style). Responsibilities:

  1. Authentication: every request must carry `Authorization: Bearer
     <token>`; the token is resolved to a role (admin / viewer). Missing or
     unknown tokens are rejected with HTTP 401 before anything else.
  2. Envelope validation: the body must be a single JSON-RPC 2.0 request,
     notification, or response object (responses answer server-initiated
     requests such as sampling) of at most 1 MiB. Anything else is answered
     by the gateway itself (-32700 parse error, -32600 invalid request) and
     never forwarded.
  3. Method-level authorization:
       - tools/list  -> forwarded transparently; for non-admins the
         admin_ tools are filtered out of the response.
       - tools/call  -> if params.name starts with "admin_", the caller's
         role must be admin; otherwise the gateway returns JSON-RPC
         -32001 "Unauthorized Tool Call" itself and the downstream server
         is never contacted.
       - anything else (initialize, notifications, ...) -> forwarded.
  4. Forwarding: one shared, pooled HTTP client; the downstream response is
     streamed back as-is (status, content-type, Mcp-Session-Id), so both
     plain JSON responses and SSE streams pass through. Downstream failures
     (unreachable, timeout, 5xx) become a clean JSON-RPC -32003 error, never
     a raw 500 or a leaked upstream stack trace.

Security note on what gets forwarded: the gateway forwards its own
re-serialization of the *parsed* body, not the client's raw bytes. If it
forwarded raw bytes, a payload like {"params": {"name": "get_weather",
"name": "admin_reset_key"}} could be authorized against one value while a
downstream JSON parser with different duplicate-key semantics executes the
other (a parser-differential bypass). Re-serializing guarantees the
downstream sees exactly the request that was authorized.
"""

import asyncio
import hmac
import json
import logging
import math
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("mcp-gateway")

DOWNSTREAM_URL = os.environ.get("DOWNSTREAM_MCP_URL", "http://127.0.0.1:8100/mcp")
DOWNSTREAM_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
# The Streamable HTTP GET stream is a long-lived SSE channel that can be idle
# for minutes, so it gets no per-read timeout.
SSE_STREAM_TIMEOUT = httpx.Timeout(30.0, connect=5.0, read=None)
MAX_BODY_BYTES = 1024 * 1024

# Demo-only static token -> role map. In a real deployment this would be
# JWT verification / token introspection against an identity provider.
TOKEN_ROLES = {
    "admin-secret-123": "admin",
    "viewer-secret-456": "viewer",
}

# JSON-RPC error codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
UNAUTHORIZED_TOOL_CALL = -32001  # also used for unauthenticated callers (HTTP 401)
DOWNSTREAM_UNAVAILABLE = -32003

ADMIN_TOOL_PREFIX = "admin_"

# Client headers that are part of the MCP transport and must reach the
# downstream server. Everything else -- notably the client's Authorization
# header -- is deliberately not forwarded.
FORWARDED_REQUEST_HEADERS = ("accept", "content-type", "mcp-session-id", "mcp-protocol-version", "last-event-id")
FORWARDED_RESPONSE_HEADERS = ("content-type", "mcp-session-id", "cache-control")


def _ignore_peer_reset(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    # On Windows the Proactor event loop logs a full traceback when a client
    # resets a connection that is already being closed (e.g. an MCP client
    # tearing down its SSE stream). That's a CPython quirk, not an error in
    # this service; everything else goes to the default handler.
    if isinstance(context.get("exception"), ConnectionResetError) and "_call_connection_lost" in str(context.get("handle")):
        return
    loop.default_exception_handler(context)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.get_running_loop().set_exception_handler(_ignore_peer_reset)
    # One pooled client for the gateway's lifetime instead of a new
    # connection (TCP handshake, and TLS in production) per proxied call.
    async with httpx.AsyncClient(timeout=DOWNSTREAM_TIMEOUT) as client:
        app.state.http_client = client
        yield


app = FastAPI(title="MCP Security Gateway", lifespan=lifespan)


def _jsonrpc_error(req_id, code: int, message: str, status_code: int = 200, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(
        {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}},
        status_code=status_code,
        headers=headers,
    )


def _reject_json_constant(name: str):
    raise ValueError(f"non-standard JSON constant: {name}")


def _parse_finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):  # e.g. 1e999
        raise ValueError(f"number out of range: {text}")
    return value


def _loads_strict(raw: bytes):
    # Python's json accepts NaN/Infinity, and turns 1e999 into inf; none of
    # these is valid JSON (RFC 8259). Left in, they can't be re-serialized for
    # the downstream or echoed back as an id (Starlette renders with
    # allow_nan=False), so they are a parse error here.
    return json.loads(raw, parse_constant=_reject_json_constant, parse_float=_parse_finite_float)


async def _read_body(request: Request) -> bytes | None:
    """The request body, or None if it exceeds MAX_BODY_BYTES (read in
    chunks, so an oversized body is never held in memory whole)."""
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _extract_role(request: Request) -> str | None:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer":
        return None
    # Constant-time comparison against every known token, so response timing
    # doesn't reveal how much of a guessed token was correct.
    presented = token.strip().encode("utf-8")
    role = None
    for known, known_role in TOKEN_ROLES.items():
        if hmac.compare_digest(presented, known.encode("utf-8")):
            role = known_role
    return role


def _unauthenticated(req_id=None) -> JSONResponse:
    return _jsonrpc_error(
        req_id,
        UNAUTHORIZED_TOOL_CALL,
        "Unauthorized: missing or invalid bearer token",
        status_code=401,
        headers={"WWW-Authenticate": 'Bearer realm="mcp-gateway"'},
    )


def _is_admin_tool(tool_name: str) -> bool:
    # casefold() + strip(): a downstream that normalizes tool names (case-
    # insensitive registry, trimmed input) must not turn "ADMIN_reset_key"
    # or " admin_reset_key" into an unguarded admin call. Erring toward
    # *more* restrictive is the safe direction for an authz check.
    return tool_name.strip().casefold().startswith(ADMIN_TOOL_PREFIX)


def _without_admin_tools(message):
    """A tools/list response with the admin_ tools removed (anything else is
    returned unchanged)."""
    result = message.get("result") if isinstance(message, dict) else None
    tools = result.get("tools") if isinstance(result, dict) else None
    if not isinstance(tools, list):
        return message
    kept = [t for t in tools if not (isinstance(t, dict) and isinstance(t.get("name"), str) and _is_admin_tool(t["name"]))]
    return {**message, "result": {**result, "tools": kept}}


async def _filtered_sse(upstream: httpx.Response):
    """Re-emit an SSE stream event by event, filtering any tools/list
    response it carries. Non-data fields (event:, id:, retry:) are kept."""
    event: list[str] = []
    async for line in upstream.aiter_lines():
        if line:
            event.append(line)
            continue
        data = [ln[5:].removeprefix(" ") for ln in event if ln.startswith("data:")]
        if data:
            try:
                message = json.loads("\n".join(data))
            except ValueError:
                pass
            else:
                event = [ln for ln in event if not ln.startswith("data:")]
                event.append("data: " + json.dumps(_without_admin_tools(message)))
        yield ("\n".join(event) + "\n\n").encode("utf-8")
        event = []
    if event:
        yield ("\n".join(event) + "\n\n").encode("utf-8")


async def _forward(request: Request, method: str, content: bytes | None, req_id=None,
                   timeout: httpx.Timeout = DOWNSTREAM_TIMEOUT, filter_admin_tools: bool = False) -> Response:
    client: httpx.AsyncClient = request.app.state.http_client
    headers = {k: v for k in FORWARDED_REQUEST_HEADERS if (v := request.headers.get(k)) is not None}
    if content is not None:
        headers["content-type"] = "application/json"
    # The body is relayed byte-for-byte (aiter_raw), so ask for it
    # uncompressed; otherwise httpx's default Accept-Encoding: gzip would let
    # the downstream send gzip that reaches the client without its header.
    headers["accept-encoding"] = "identity"

    try:
        upstream_req = client.build_request(method, DOWNSTREAM_URL, content=content, headers=headers, timeout=timeout)
        upstream = await client.send(upstream_req, stream=True)
    except httpx.HTTPError as exc:
        logger.warning("downstream request failed: %s", type(exc).__name__)
        return _jsonrpc_error(req_id, DOWNSTREAM_UNAVAILABLE, "Downstream MCP server is unavailable")

    if upstream.status_code >= 500:
        # Don't relay a downstream 5xx body: it's where stack traces and
        # internal details leak from.
        await upstream.aclose()
        logger.warning("downstream returned HTTP %d", upstream.status_code)
        return _jsonrpc_error(req_id, DOWNSTREAM_UNAVAILABLE, "Downstream MCP server error")

    response_headers = {k: v for k in FORWARDED_RESPONSE_HEADERS if (v := upstream.headers.get(k)) is not None}
    content_type = upstream.headers.get("content-type", "")

    if filter_admin_tools and upstream.status_code == 200 and content_type.startswith("application/json"):
        # A tools/list result is small: read it whole and filter it.
        try:
            raw = await upstream.aread()
        except httpx.HTTPError as exc:
            logger.warning("downstream read failed: %s", type(exc).__name__)
            return _jsonrpc_error(req_id, DOWNSTREAM_UNAVAILABLE, "Downstream MCP server is unavailable")
        finally:
            await upstream.aclose()
        try:
            message = json.loads(raw)
        except ValueError:
            return Response(raw, status_code=200, headers=response_headers)
        return Response(json.dumps(_without_admin_tools(message)).encode("utf-8"), status_code=200,
                        headers=response_headers)

    filter_sse = filter_admin_tools and upstream.status_code == 200 and content_type.startswith("text/event-stream")

    async def body():
        try:
            async for chunk in (_filtered_sse(upstream) if filter_sse else upstream.aiter_raw()):
                yield chunk
        except httpx.HTTPError as exc:
            # Mid-stream downstream failure: headers are already sent, so
            # the only option is to end the stream.
            logger.warning("downstream stream aborted: %s", type(exc).__name__)

    # Streamed, not buffered: works for plain JSON and for SSE responses
    # alike. The BackgroundTask releases the upstream connection even if
    # the client disconnects mid-stream.
    return StreamingResponse(
        body(),
        status_code=upstream.status_code,
        headers=response_headers,
        background=BackgroundTask(upstream.aclose),
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/mcp")
async def gateway_endpoint(request: Request):
    role = _extract_role(request)
    if role is None:
        logger.warning("rejected unauthenticated request from %s", request.client.host if request.client else "?")
        return _unauthenticated()

    raw = await _read_body(request)
    if raw is None:
        return _jsonrpc_error(None, INVALID_REQUEST, "Invalid Request: body too large", status_code=413)
    try:
        body = _loads_strict(raw)
    except (ValueError, UnicodeDecodeError, RecursionError):  # RecursionError: absurdly deep nesting
        return _jsonrpc_error(None, PARSE_ERROR, "Parse error: body is not valid JSON")

    # MCP (2025-06-18+) removed JSON-RPC batching, and the authorization
    # logic below reasons about exactly one call -- so a batch (list) or any
    # other non-object body is rejected rather than half-inspected.
    if not isinstance(body, dict):
        return _jsonrpc_error(None, INVALID_REQUEST, "Invalid Request: body must be a single JSON-RPC object")

    req_id = body.get("id")
    if not isinstance(req_id, (str, int, float, type(None))) or isinstance(req_id, bool):
        return _jsonrpc_error(None, INVALID_REQUEST, "Invalid Request: id must be a string, number, or null")
    if body.get("jsonrpc") != "2.0":
        return _jsonrpc_error(req_id, INVALID_REQUEST, "Invalid Request: expected jsonrpc '2.0'")
    method = body.get("method")
    if method is None and "id" in body and (("result" in body) != ("error" in body)):
        # A JSON-RPC *response*: the client answering a server-initiated
        # request (sampling/createMessage, elicitation/create, roots/list,
        # ping). It carries no tool call, so it is forwarded as-is.
        logger.info("forwarding client response id=%r role=%s", req_id, role)
        return await _forward(request, "POST", json.dumps(body).encode("utf-8"), req_id)
    if not isinstance(method, str):
        return _jsonrpc_error(req_id, INVALID_REQUEST, "Invalid Request: expected a string method")

    # JSON-RPC notifications (no "id") get no JSON-RPC reply. Streamable HTTP
    # says a refused notification gets an HTTP error status instead (the body
    # may carry an id-less error), so these map to 400/403 rather than 200.
    is_notification = "id" not in body

    if method == "tools/call":
        params = body.get("params")
        if not isinstance(params, dict):
            return _jsonrpc_error(req_id, INVALID_PARAMS, "Invalid params: params must be an object",
                                  status_code=400 if is_notification else 200)
        tool_name = params.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            return _jsonrpc_error(req_id, INVALID_PARAMS, "Invalid params: 'name' must be a non-empty string",
                                  status_code=400 if is_notification else 200)
        if _is_admin_tool(tool_name) and role != "admin":
            logger.warning("BLOCKED tools/call tool=%r role=%s id=%r", tool_name, role, req_id)
            return _jsonrpc_error(req_id, UNAUTHORIZED_TOOL_CALL, "Unauthorized Tool Call",
                                  status_code=403 if is_notification else 200)
        logger.info("ALLOWED tools/call tool=%r role=%s id=%r", tool_name, role, req_id)
    else:
        # tools/list and every other method: the request is forwarded
        # transparently.
        logger.info("forwarding method=%s role=%s id=%r", method, role, req_id)

    # Tool filtering: a non-admin's tools/list *response* omits the admin_
    # tools, so an agent never plans around tools it isn't allowed to call.
    # The tools/call check above stays the actual enforcement.
    return await _forward(request, "POST", json.dumps(body).encode("utf-8"), req_id,
                          filter_admin_tools=method == "tools/list" and role != "admin")


@app.api_route("/mcp", methods=["GET", "DELETE"])
async def gateway_passthrough(request: Request):
    # Streamable HTTP also uses GET (server->client SSE stream) and DELETE
    # (session termination). Neither carries a tool call, so they only need
    # authentication, not payload inspection.
    role = _extract_role(request)
    if role is None:
        return _unauthenticated()
    # A GET with Last-Event-ID resumes an interrupted POST stream, so it can
    # replay a tools/list response: filter it like the original would be.
    return await _forward(request, request.method, None,
                          timeout=SSE_STREAM_TIMEOUT if request.method == "GET" else DOWNSTREAM_TIMEOUT,
                          filter_admin_tools=request.method == "GET" and role != "admin")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("GATEWAY_PORT", "8200")))
