"""Rate-limiting & fallback router for an LLM gateway.

Flow per request:
  1. Token estimate, computed server-side: prompt tokens (~4 UTF-8 bytes
     per token) + the completion budget (max_tokens). A client-supplied
     `estimated_tokens` can raise the reservation but never lower it below
     the server's own estimate -- a gateway can't let callers under-declare
     their cost.
  2. Reserve that many tokens in the tenant's sliding window
     (rate_limiter.py). Over budget -> standardized 429 + Retry-After,
     without contacting any provider.
  3. Call the primary provider under a hard 3000ms deadline covering the
     whole exchange (connect + send + full response body). On a 429, a
     timeout, a 5xx, a connection error, or a malformed response, fail
     over to the secondary provider (5000ms deadline).
  4. Settle the reservation to the provider-reported usage (refunding any
     unused estimate), or release it entirely if neither provider served
     the request.
  5. Every error -- validation, rate limit, upstream failure, or a bug in
     this file -- leaves as {"error": {"code", "message"}}. Upstream bodies,
     exception text and stack traces are logged server-side under a
     request id (returned to the client as X-Request-ID), never returned.

Why asyncio.timeout() and not just httpx's timeout=3.0: httpx timeouts are
per-phase (connect, write, pool) and per-*read*, not end-to-end. A primary
that trickles one byte every 2s never trips a 3s read timeout and could
hold the request open indefinitely. asyncio.timeout() bounds the total, and
on expiry cancels the in-flight call, which closes its connection.
"""

import asyncio
import json
import logging
import math
import os
import uuid
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

import rate_limiter
from rate_limiter import TOKENS_PER_MINUTE, RateLimitExceeded

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("llm-router")
# httpx logs one INFO line per outbound request by default -- under the
# concurrency probes (100+ requests) that drowns out the router's own lines.
logging.getLogger("httpx").setLevel(logging.WARNING)

PRIMARY_URL = os.environ.get("PRIMARY_URL", "http://127.0.0.1:8500/primary/complete")
SECONDARY_URL = os.environ.get("SECONDARY_URL", "http://127.0.0.1:8500/secondary/complete")
PRIMARY_TIMEOUT_SECONDS = 3.0
SECONDARY_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_TOKENS = 256
EVICTION_INTERVAL_SECONDS = 30
# In-flight calls per provider. Kept at httpx's default pool size: past ~100
# concurrent keep-alive connections, httpx/httpcore was measured to start
# requests in a slow trickle (on Windows), making a healthy provider look slow.
PROVIDER_CONCURRENCY = 100


async def _evict_periodically() -> None:
    while True:
        await asyncio.sleep(EVICTION_INTERVAL_SECONDS)
        try:
            deleted = await rate_limiter.evict_expired()
            if deleted:
                logger.info("evicted %d expired usage rows", deleted)
        except Exception:
            logger.exception("periodic eviction failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A shared, connection-pooling client instead of one per request. Its
    # own timeouts are only a backstop; the real deadlines are the
    # asyncio.timeout() calls around each provider call. The pool has a
    # connection for every provider slot (see _call_provider), so a request
    # holding a slot never queues for a connection.
    sweeper = asyncio.create_task(_evict_periodically())
    slots = PROVIDER_CONCURRENCY * 2
    limits = httpx.Limits(max_connections=slots, max_keepalive_connections=slots)
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=2.0), limits=limits) as client:
        app.state.http_client = client
        app.state.provider_slots = {name: asyncio.Semaphore(PROVIDER_CONCURRENCY) for name in ("primary", "secondary")}
        yield
    sweeper.cancel()
    with suppress(asyncio.CancelledError):
        await sweeper


app = FastAPI(title="Rate-Limiting & Fallback Router", lifespan=lifespan)


class CompletionRequest(BaseModel):
    tenant_key: str = Field(..., min_length=1, max_length=256)
    prompt: str
    max_tokens: int | None = Field(default=None, gt=0, le=TOKENS_PER_MINUTE)
    estimated_tokens: int | None = Field(default=None, gt=0)
    # Test-only hooks to force provider failure modes on demand; a real
    # deployment would not expose these.
    simulate_primary: str | None = None
    simulate_secondary: str | None = None

    @field_validator("tenant_key", "prompt")
    @classmethod
    def valid_unicode(cls, v: str) -> str:
        v.encode("utf-8")  # a lone surrogate escape ("\ud800") -> 422, not a 500 later
        return v


class ProviderFailure(Exception):
    """A provider attempt failed in a way that should trigger failover.
    `reason` is a short, internal-only label for logs."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _gateway_error(code: str, message: str, status_code: int, request_id: str | None = None,
                   headers: dict | None = None) -> JSONResponse:
    headers = dict(headers or {})
    if request_id:
        headers["X-Request-ID"] = request_id
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}}, headers=headers)


def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    if rid is None:
        rid = request.state.request_id = uuid.uuid4().hex
    return rid


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Same envelope for validation failures as for everything else, and no
    # echo of pydantic's internals (field paths/input values stay in logs).
    logger.info("request %s failed validation: %s", _request_id(request), exc.errors())
    return _gateway_error("INVALID_REQUEST", "The request body failed validation.", 422, _request_id(request))


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    codes = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}
    return _gateway_error(codes.get(exc.status_code, "HTTP_ERROR"), "The request could not be served.",
                          exc.status_code, _request_id(request))


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Last line of defense: a bug anywhere in this service must not leak a
    # stack trace to the caller.
    logger.exception("request %s: unhandled error", _request_id(request))
    return _gateway_error("INTERNAL_ERROR", "An internal error occurred.", 500, _request_id(request))


def _reject_json_constant(name: str):
    raise ValueError(f"non-standard JSON constant: {name}")


def _parse_finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):  # e.g. 1e999
        raise ValueError(f"number out of range: {text}")
    return value


def _loads_strict(raw: bytes):
    # NaN/Infinity/1e999 parse in Python but aren't JSON, and the response
    # renderer (allow_nan=False) would fail on them after the tenant is charged.
    return json.loads(raw, parse_constant=_reject_json_constant, parse_float=_parse_finite_float)


def estimate_prompt_tokens(prompt: str) -> int:
    # ~4 bytes/token for English BPE vocabularies; counting UTF-8 bytes
    # rather than characters keeps CJK/emoji-heavy prompts from being
    # under-estimated. A production gateway would use the model's tokenizer.
    return max(1, -(-len(prompt.encode("utf-8")) // 4))


async def _call_provider(client: httpx.AsyncClient, slots: asyncio.Semaphore, name: str, url: str,
                         payload: dict, headers: dict, deadline: float) -> dict:
    try:
        # The slot is taken *before* the deadline starts: under a burst, time
        # spent waiting for capacity inside this gateway must not count as
        # the provider timing out (and fail over a healthy primary).
        async with slots, asyncio.timeout(deadline):
            resp = await client.post(url, json=payload, headers=headers)
    except TimeoutError:
        raise ProviderFailure(f"{name} exceeded {deadline * 1000:.0f}ms deadline")
    except httpx.HTTPError as exc:
        raise ProviderFailure(f"{name} transport error: {type(exc).__name__}")

    if resp.status_code == 429:
        raise ProviderFailure(f"{name} returned 429")
    if resp.status_code != 200:
        raise ProviderFailure(f"{name} returned HTTP {resp.status_code}")
    try:
        data = _loads_strict(resp.content)
        # Must re-render as the client's response; a lone surrogate escape
        # ("\ud83d") parses fine but can't be UTF-8 encoded.
        json.dumps(data, ensure_ascii=False).encode("utf-8")
    except (ValueError, RecursionError):  # incl. UnicodeEncodeError
        raise ProviderFailure(f"{name} returned a malformed JSON body")
    # A 200 is only a success if it is one: an {"error": ...} body (with
    # whatever stack trace it carries) is a failure, not something to relay.
    if not isinstance(data, dict) or "error" in data or not isinstance(data.get("completion"), str):
        raise ProviderFailure(f"{name} returned a 200 without a completion")
    return data


def _reported_usage(data: dict) -> int | None:
    usage = data.get("usage")
    total = usage.get("total_tokens") if isinstance(usage, dict) else None
    return total if isinstance(total, int) and not isinstance(total, bool) and total >= 0 else None


@app.post("/v1/complete")
async def complete(req: CompletionRequest, request: Request):
    request_id = _request_id(request)
    prompt_tokens = estimate_prompt_tokens(req.prompt)
    reserved = max(prompt_tokens + (req.max_tokens or DEFAULT_MAX_TOKENS), req.estimated_tokens or 0)

    try:
        reservation = await rate_limiter.reserve(req.tenant_key, reserved)
    except RateLimitExceeded as exc:
        # INFO, not WARNING: an over-budget client is routine traffic
        # shaping, not an operational problem.
        logger.info("request %s rate limited: used=%d requested=%d", request_id, exc.used_tokens, exc.requested)
        if exc.retry_after is None:
            return _gateway_error(
                "RATE_LIMIT_EXCEEDED",
                f"This request's estimated size ({reserved} tokens) exceeds the per-tenant limit "
                f"of {TOKENS_PER_MINUTE} tokens/min.",
                429, request_id,
            )
        return _gateway_error(
            "RATE_LIMIT_EXCEEDED",
            f"Token rate limit exceeded for this tenant ({TOKENS_PER_MINUTE} tokens/min). Try again shortly.",
            429, request_id,
            headers={"Retry-After": str(max(1, math.ceil(exc.retry_after)))},
        )

    # The client's own completion cap. `estimated_tokens` only ever raises the
    # reservation, never this cap, and reserved >= prompt + max_tokens, so
    # actual usage can't exceed the reservation unless the provider ignores it.
    payload = {"prompt": req.prompt, "max_tokens": req.max_tokens or DEFAULT_MAX_TOKENS}
    client: httpx.AsyncClient = request.app.state.http_client
    attempts = [
        ("primary", PRIMARY_URL, PRIMARY_TIMEOUT_SECONDS, req.simulate_primary),
        ("secondary", SECONDARY_URL, SECONDARY_TIMEOUT_SECONDS, req.simulate_secondary),
    ]

    served = False
    try:
        for name, url, deadline, simulate in attempts:
            headers = {"X-Request-ID": request_id}
            if simulate:
                headers["X-Simulate"] = simulate
            try:
                data = await _call_provider(client, request.app.state.provider_slots[name], name, url,
                                            payload, headers, deadline)
            except ProviderFailure as failure:
                logger.warning("request %s: %s", request_id, failure.reason)
                continue

            served = True
            actual = _reported_usage(data)
            if actual is not None:
                try:
                    await rate_limiter.settle(reservation, actual)
                except Exception:
                    # The provider already served (and billed) this request; don't
                    # throw the answer away. The reservation stays at the estimate.
                    logger.exception("request %s: settling usage failed; keeping the reservation", request_id)
            data["_routed_via"] = name
            return JSONResponse(content=data, headers={"X-Request-ID": request_id})
    finally:
        if not served:
            # Shielded so the refund still happens if this request is being
            # cancelled (e.g. the client disconnected mid-failover).
            await asyncio.shield(rate_limiter.release(reservation))

    logger.warning("request %s: all providers failed; returning sanitized gateway error", request_id)
    return _gateway_error(
        "UPSTREAM_UNAVAILABLE",
        "The completion service is temporarily unavailable. Please try again.",
        502, request_id,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8600, log_level="warning")
