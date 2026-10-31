"""LLM Gateway streaming endpoint with a real-time PII redaction guardrail.

Routes OpenAI-compatible chat-completion requests to an upstream provider
(UPSTREAM_URL; the bundled mock_provider.py by default), parses the
provider's Server-Sent-Events stream delta-by-delta, redacts PII in flight
via redaction.StreamRedactor, and re-emits the same chunk format to the
client.

Latency/memory design:
  - Nothing waits for the full response. Each upstream delta is parsed and
    fed to the redactor as soon as its SSE event is complete; whatever the
    redactor releases is written to the client immediately.
  - Per stream we hold one small SSE line buffer plus one bounded redactor
    carry per choice index (see redaction.py) -- memory is O(1) in
    response length.
  - One pooled upstream HTTP client for the whole process.
  - If the client disconnects, the generator's `finally` closes the
    upstream response so the provider stops generating for nobody.

Endpoints:
  POST /v1/chat/completions   OpenAI-compatible; "stream": true (default)
                              returns SSE, "stream": false returns one JSON
                              completion (redacted with the same code path).
  POST /v1/chat/stream        Convenience: {"prompt": "..."} -> redacted
                              text/plain stream.
"""

import json
import logging
import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator

from redaction import StreamRedactor, redact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("llm-guardrail-gateway")
logging.getLogger("httpx").setLevel(logging.WARNING)

UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8301/v1/chat/completions")
# Sent as "Authorization: Bearer ..." when set -- needed for a real provider.
UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY")
UPSTREAM_TIMEOUT = httpx.Timeout(60.0, connect=5.0)
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}  # no proxy buffering


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        app.state.http_client = client
        yield


app = FastAPI(title="LLM Gateway Streaming Guardrail", lifespan=lifespan)


def _reject_json_constant(name: str):
    raise ValueError(f"non-standard JSON constant: {name}")


def _parse_finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):  # e.g. 1e999
        raise ValueError(f"number out of range: {text}")
    return value


def _loads_strict(raw: bytes):
    # Python's json accepts NaN/Infinity, and turns 1e999 into inf; none of
    # these is valid JSON (RFC 8259), and httpx refuses to re-encode them for
    # the upstream call.
    return json.loads(raw, parse_constant=_reject_json_constant, parse_float=_parse_finite_float)


def _error(status: int, message: str, err_type: str = "upstream_error") -> JSONResponse:
    return JSONResponse({"error": {"type": err_type, "message": message}}, status_code=status)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Same error shape as the rest of the gateway, not FastAPI's {"detail": ...}.
    return _error(400, "Request body failed validation.", "invalid_request_error")


async def _open_upstream(request: Request, payload: dict) -> httpx.Response | JSONResponse:
    """Start the upstream stream. Returns the open response, or a sanitized
    error response if the provider couldn't be reached / refused."""
    client: httpx.AsyncClient = request.app.state.http_client
    headers = {"Authorization": f"Bearer {UPSTREAM_API_KEY}"} if UPSTREAM_API_KEY else None
    try:
        upstream = await client.send(client.build_request("POST", UPSTREAM_URL, json=payload, headers=headers), stream=True)
    except httpx.HTTPError as exc:
        logger.warning("upstream unreachable: %s", type(exc).__name__)
        return _error(502, "The model provider is unavailable.")
    if upstream.status_code != 200:
        # Don't relay the provider's error body -- it can carry internals.
        await upstream.aclose()
        logger.warning("upstream returned HTTP %d", upstream.status_code)
        return _error(502, "The model provider returned an error.")
    return upstream


async def _sse_events(upstream: httpx.Response) -> AsyncIterator[str]:
    """Minimal incremental SSE parser: yields each event's data payload as
    soon as the event's terminating blank line arrives."""
    data_lines: list[str] = []
    async for line in upstream.aiter_lines():
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
        elif line.startswith("data:"):
            data_lines.append(line[5:].removeprefix(" "))
        # other SSE fields (event:, id:, retry:, ":" comments) are ignored
    if data_lines:
        yield "\n".join(data_lines)


def _stream_error() -> dict:
    return {"error": {"type": "upstream_error", "message": "The model provider stream was interrupted."}}


def _redact_strings(value):
    """Whole-value redaction of every string inside a JSON value, for parts
    of the stream that aren't incrementally streamed text."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redact_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: _redact_strings(v) for k, v in value.items()}
    return value


def _choices(chunk: dict) -> list:
    choices = chunk.get("choices")
    return choices if isinstance(choices, list) else []


def _fix_surrogates(text: str) -> str:
    # A provider may emit a surrogate pair as two JSON escapes, even split
    # across deltas; joined they are one character, and any lone half becomes
    # U+FFFD instead of crashing the UTF-8 encode of the response.
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def _merge_tool_call(calls: dict[int, dict], fragment) -> None:
    """Fold one streamed tool-call fragment into the assembled call (for the
    stream=false response): id/type/name arrive once, arguments in pieces."""
    if not isinstance(fragment, dict) or not isinstance(fragment.get("index", 0), int):
        return
    call = calls.setdefault(fragment.get("index", 0), {"id": None, "type": "function",
                                                       "function": {"name": None, "arguments": ""}})
    for key in ("id", "type"):
        if isinstance(fragment.get(key), str):
            call[key] = fragment[key]
    function = fragment.get("function")
    if isinstance(function, dict):
        if isinstance(function.get("name"), str):
            call["function"]["name"] = function["name"]
        if isinstance(function.get("arguments"), str):
            call["function"]["arguments"] += function["arguments"]


async def redacted_chunks(upstream: httpx.Response) -> AsyncIterator[dict]:
    """Upstream chat.completion.chunk stream -> the same stream with every
    streamed text field of each choice's delta (`content`, and e.g.
    `refusal` / `reasoning_content`) passed through its own StreamRedactor.

    Each tool call's `function.arguments` fragments get their own
    StreamRedactor too, so PII split across fragments is caught.

    Yields chunk dicts; on an upstream failure (transport error, provider
    error event, or a stream cut off before its "[DONE]"), yields one final
    {"error": ...} dict instead. Other delta fields (role, finish_reason)
    pass through, re-ordered only so that a choice's redactors are flushed
    *before* its finish_reason is forwarded.
    """
    # (choice index, delta field, tool-call index or None) -> redactor
    redactors: dict[tuple[int, str, int | None], StreamRedactor] = {}
    meta: dict = {}

    def text_chunk(index: int, field: str, text: str, tool_index: int | None = None) -> dict:
        if tool_index is None:
            delta = {field: text}
        else:
            delta = {"tool_calls": [{"index": tool_index, "function": {"arguments": text}}]}
        return {**meta, "choices": [{"index": index, "delta": delta, "finish_reason": None}]}

    def flush(index: int | None = None):
        for (i, field, tool_index), redactor in redactors.items():
            if index is None or i == index:
                tail = redactor.flush()
                if tail:
                    yield text_chunk(i, field, tail, tool_index)

    def redact_tool_calls(index: int, tool_calls: list) -> list:
        out = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call = dict(call)
            tool_index = call.get("index", 0)
            function = call.get("function")
            if (isinstance(function, dict) and isinstance(function.get("arguments"), str)
                    and isinstance(tool_index, int) and not isinstance(tool_index, bool)):
                redactor = redactors.setdefault((index, "tool_calls", tool_index), StreamRedactor())
                call["function"] = {**function, "arguments": redactor.feed(function["arguments"])}
            out.append(call)
        return out

    # Deliberately NOT flushing the carries on any failure below: on an
    # aborted stream they may hold the first half of a PII token
    # ("john.doe@exa") that never completed into something the patterns
    # recognise.
    done = False
    try:
        async for data in _sse_events(upstream):
            if data == "[DONE]":
                done = True
                break
            try:
                obj = _loads_strict(data)
            except (ValueError, RecursionError):
                obj = None
            if not isinstance(obj, dict):
                logger.warning("skipping malformed upstream SSE event")
                continue
            if "error" in obj:
                # A provider error event mid-stream: its message is
                # unredacted and may carry internals, so it isn't relayed.
                logger.warning("upstream sent an error event mid-stream")
                yield _stream_error()
                return
            meta = {k: obj[k] for k in ("id", "object", "created", "model") if k in obj}
            choices = _choices(obj)
            if not choices:
                yield _redact_strings(obj)  # e.g. a trailing usage-only chunk
                continue
            for choice in choices:
                index = choice.get("index", 0) if isinstance(choice, dict) else None
                if not isinstance(index, int) or isinstance(index, bool):
                    logger.warning("skipping malformed upstream choice")
                    continue
                delta = choice.get("delta")
                delta = dict(delta) if isinstance(delta, dict) else {}
                for field in [k for k, v in delta.items() if k != "role" and isinstance(v, str)]:
                    text = delta.pop(field)
                    if text:
                        out = redactors.setdefault((index, field, None), StreamRedactor()).feed(text)
                        if out:
                            yield text_chunk(index, field, out)
                tool_calls = delta.pop("tool_calls", None)
                # Whole-value redaction for whatever else the delta carries;
                # tool-call arguments were just stream-redacted instead.
                delta = _redact_strings(delta)
                if isinstance(tool_calls, list):
                    delta["tool_calls"] = redact_tool_calls(index, tool_calls)
                finish = choice.get("finish_reason")
                if finish is not None:
                    for chunk in flush(index):
                        yield chunk
                if delta or finish is not None:
                    yield {**meta, "choices": [{"index": index, "delta": delta, "finish_reason": finish}]}
    except httpx.HTTPError as exc:
        logger.warning("upstream stream aborted mid-response: %s", type(exc).__name__)
        yield _stream_error()
        return
    except Exception:
        logger.exception("unexpected error while relaying the upstream stream")
        yield _stream_error()
        return

    if not done:
        # The connection closed cleanly but without the "[DONE]" terminator
        # every OpenAI-compatible stream ends with: a cut-off stream, not a
        # finished one.
        logger.warning("upstream stream ended without [DONE]")
        yield _stream_error()
        return

    for chunk in flush():
        yield chunk


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = _loads_strict(await request.body())
    except (ValueError, RecursionError):  # RecursionError: absurdly deep nesting
        return _error(400, "Request body must be JSON.", "invalid_request_error")
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return _error(400, "'messages' must be a list.", "invalid_request_error")
    try:
        json.dumps(body, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate escape such as "\ud800"
        return _error(400, "Request body contains invalid Unicode.", "invalid_request_error")

    want_stream = body.get("stream", True) is not False
    upstream = await _open_upstream(request, {**body, "stream": True})
    if isinstance(upstream, JSONResponse):
        return upstream

    if not want_stream:
        # Client asked for a single JSON response: same redaction path,
        # collected at the end.
        try:
            parts: dict[int, dict[str, list[str]]] = {}  # index -> text field -> pieces
            calls: dict[int, dict[int, dict]] = {}  # index -> tool-call index -> assembled call
            finish: dict[int, str] = {}
            meta: dict = {}
            async for chunk in redacted_chunks(upstream):
                if "error" in chunk:
                    return _error(502, chunk["error"]["message"])
                meta = {k: chunk[k] for k in ("id", "created", "model") if k in chunk} or meta
                for c in _choices(chunk):
                    fields = parts.setdefault(c["index"], {})
                    for field, text in c["delta"].items():
                        if field != "role" and isinstance(text, str):
                            fields.setdefault(field, []).append(text)
                    for fragment in c["delta"].get("tool_calls") or []:
                        _merge_tool_call(calls.setdefault(c["index"], {}), fragment)
                    if c.get("finish_reason"):
                        finish[c["index"]] = c["finish_reason"]
        finally:
            await upstream.aclose()
        choices = []
        for i in sorted(parts.keys() | calls.keys()):
            message = {"role": "assistant", "content": "",
                       **{f: _fix_surrogates("".join(t)) for f, t in parts.get(i, {}).items()}}
            if calls.get(i):
                for call in calls[i].values():
                    call["function"]["arguments"] = _fix_surrogates(call["function"]["arguments"])
                message["tool_calls"] = [calls[i][t] for t in sorted(calls[i])]
            choices.append({"index": i, "message": message, "finish_reason": finish.get(i)})
        return JSONResponse({**meta, "object": "chat.completion", "choices": choices})

    async def sse() -> AsyncIterator[bytes]:
        try:
            async for chunk in redacted_chunks(upstream):
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        finally:
            await upstream.aclose()

    return StreamingResponse(sse(), media_type="text/event-stream", headers=SSE_HEADERS)


class PromptRequest(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def valid_unicode(cls, v: str) -> str:
        v.encode("utf-8")  # a lone surrogate escape ("\ud800") -> 422, not a 500 later
        return v


@app.post("/v1/chat/stream")
async def chat_stream(req: PromptRequest, request: Request):
    logger.info("streaming completion requested, prompt_len=%d", len(req.prompt))
    upstream = await _open_upstream(request, {"messages": [{"role": "user", "content": req.prompt}], "stream": True})
    if isinstance(upstream, JSONResponse):
        return upstream

    async def text() -> AsyncIterator[bytes]:
        try:
            async for chunk in redacted_chunks(upstream):
                if "error" in chunk:
                    yield b"\n[stream interrupted]\n"
                    return
                for c in _choices(chunk):
                    content = c["delta"].get("content")
                    if isinstance(content, str) and content:
                        yield _fix_surrogates(content).encode("utf-8")
        finally:
            await upstream.aclose()

    return StreamingResponse(text(), media_type="text/plain; charset=utf-8", headers=SSE_HEADERS)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("GATEWAY_PORT", "8300")))
