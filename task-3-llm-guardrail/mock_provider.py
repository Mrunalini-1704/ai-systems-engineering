"""Deterministic mock LLM provider (no API key required).

Serves an OpenAI-compatible streaming endpoint, POST /v1/chat/completions
with "stream": true, emitting Server-Sent Events:

    data: {"id": ..., "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "..."}}]}
    ...
    data: [DONE]

The text deliberately contains PII split across chunk boundaries (an email
split mid-domain, a card number split mid-number) to prove the gateway's
redaction survives PII split mid-pattern, not just PII that happens to land
inside a single delta.

Test-only knobs (request body fields a real provider wouldn't have):
  - "mock_delay": seconds between deltas (default 0.05; "long" script: per 100 deltas)
  - "mock_script": "default" | "long" (many deltas, for TTFT/memory tests)
                   | "fail_midstream" (drops the connection partway)
                   | "error_event" (sends an OpenAI-style {"error": ...} event partway)
                   | "usage_null" (ends with a usage-only chunk whose "choices" is null)
                   | "truncated" (closes cleanly partway, without [DONE])
                   | "odd_chunks" (malformed choices, PII in delta.refusal and in a choice-less event)
                   | "tool_call" (a tool call whose arguments split an email and a card across fragments)
  - "mock_status": return this HTTP status instead of streaming
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

FULL_TEXT_CHUNKS = [
    "Hello! Here is the customer summary you requested.\n",
    "Primary contact email: john.doe@exa",  # email split mid-domain
    "mple.com, reachable during business hours.\n",
    "Backup contact: jane.smith@example.org.\n",
    "SSN on file for identity verification: 123-45-6789.\n",
    "Card used for the refund: 4111 1111 ",  # credit card split mid-number
    "1111 1111, transaction approved.\n",
    "Let us know if you need anything else!\n",
]

app = FastAPI(title="Mock LLM Provider")


async def stream_completion(prompt: str, delay: float = 0.05) -> AsyncIterator[str]:
    for chunk in FULL_TEXT_CHUNKS:
        await asyncio.sleep(delay)
        yield chunk


def _chunk(completion_id: str, created: int, delta: dict, finish_reason=None) -> bytes:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "mock-llm-1",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if body.get("mock_status"):
        return JSONResponse({"error": {"message": "upstream exploded at /srv/llm/worker.py:88"}},
                            status_code=int(body["mock_status"]))

    delay = float(body.get("mock_delay", 0.05))
    script = body.get("mock_script", "default")
    if script == "long":
        pieces = [f"token{i} " for i in range(5000)] + ["contact: ceo@corp", "-secret.com done\n"]
    elif script == "tool_call":
        pieces = []  # only the tool call below
    else:
        pieces = FULL_TEXT_CHUNKS

    completion_id, created = "chatcmpl-mock", int(time.time())

    async def sse() -> AsyncIterator[bytes]:
        yield _chunk(completion_id, created, {"role": "assistant", "content": ""})
        for i, piece in enumerate(pieces):
            # "long" sleeps once per 100 deltas: on Windows asyncio.sleep()
            # has ~15ms granularity, so a per-delta sleep would dominate.
            if delay and (script != "long" or i % 100 == 0):
                await asyncio.sleep(delay)
            if script == "fail_midstream" and i == 3:
                raise RuntimeError("simulated provider crash mid-stream")
            if script == "truncated" and i == 2:
                return  # closes cleanly mid-email, with no finish_reason or [DONE]
            if script == "odd_chunks" and i == 0:
                for odd in ({"choices": [{"index": 0, "delta": "not-an-object"}]},
                            {"choices": [{"index": [0], "delta": {"content": "bad index"}}]},
                            {"choices": [], "note": "mail bob@corp.com"}):
                    yield f"data: {json.dumps(odd)}\n\n".encode()
                yield _chunk(completion_id, created, {"refusal": "SSN 123-45-6789 "})
            if script == "error_event" and i == 3:
                error = {"error": {"message": "quota exceeded for ops@internal.example at /srv/llm/billing.py:12"}}
                yield f"data: {json.dumps(error)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return
            yield _chunk(completion_id, created, {"content": piece})
        if script == "tool_call":
            # An email split across argument fragments, like a real model streams it.
            fragments = ['{"to": "john.do', 'e@exam', 'ple.com", "card": "4111 1111 ', '1111 1111"}']
            yield _chunk(completion_id, created, {"tool_calls": [
                {"index": 0, "id": "call_1", "type": "function", "function": {"name": "send_email", "arguments": ""}}]})
            for fragment in fragments:
                await asyncio.sleep(delay)
                yield _chunk(completion_id, created, {"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]})
            yield _chunk(completion_id, created, {}, finish_reason="tool_calls")
            yield b"data: [DONE]\n\n"
            return
        yield _chunk(completion_id, created, {}, finish_reason="stop")
        if script == "usage_null":
            usage = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": "mock-llm-1",
                     "choices": None, "usage": {"prompt_tokens": 5, "completion_tokens": 60, "total_tokens": 65}}
            yield f"data: {json.dumps(usage)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8301, log_level="warning")
