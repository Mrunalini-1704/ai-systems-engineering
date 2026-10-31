"""Mock primary/secondary model providers.

Each accepts an X-Simulate header so tests can force specific failure
modes on demand:
  - primary: "429" -> returns HTTP 429; "timeout" -> sleeps past the
    router's 3000ms budget; "trickle" -> starts responding immediately but
    dribbles the body out one byte per second (never idle for >1s, so a
    per-read timeout never fires -- only a total deadline catches it);
    "usage:<N>" -> succeeds, reporting N total tokens used; "error200" ->
    HTTP 200 whose body is an error with a stack trace; "nan" -> HTTP 200
    whose body has a non-JSON NaN; anything else (or absent) -> succeeds
    normally.
  - secondary: "fail" -> returns HTTP 500; "usage:<N>" as above; anything
    else -> succeeds.

Successful responses report OpenAI-style `usage`. By default the mock
behaves like a model that spends its whole completion budget
(completion_tokens == max_tokens), the worst case the router reserves for.
"""

import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI(title="Mock Model Providers")


def _success(source: str, body: dict, simulate: str) -> dict:
    prompt = body.get("prompt", "")
    prompt_tokens = max(1, -(-len(prompt.encode("utf-8")) // 4))
    completion_tokens = int(body.get("max_tokens") or 16)
    if simulate.startswith("usage:"):
        total = int(simulate.split(":", 1)[1])
        completion_tokens = max(0, total - prompt_tokens)
    return {
        "source": source,
        "completion": f"[{source}] response to: {prompt}",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/primary/complete")
async def primary_complete(request: Request):
    body = await request.json()
    simulate = request.headers.get("x-simulate", "ok")

    if simulate == "429":
        return JSONResponse(status_code=429, content={"error": "rate limited upstream (simulated)"})
    if simulate == "error200":
        return JSONResponse(content={"error": "Traceback (most recent call last): File /opt/provider/internal.py KeyError: secret"})
    if simulate == "nan":
        return Response(content=b'{"completion": "x", "score": NaN, "usage": {"total_tokens": 10}}',
                        media_type="application/json")
    if simulate == "timeout":
        await asyncio.sleep(5.0)
    if simulate == "trickle":
        payload = json.dumps(_success("primary", body, simulate)).encode()

        async def dribble():
            for i in range(6):
                yield b" "  # JSON allows leading whitespace
                await asyncio.sleep(1.0)
            yield payload

        return StreamingResponse(dribble(), media_type="application/json")

    return JSONResponse(content=_success("primary", body, simulate))


@app.post("/secondary/complete")
async def secondary_complete(request: Request):
    body = await request.json()
    simulate = request.headers.get("x-simulate", "ok")

    if simulate == "fail":
        return JSONResponse(
            status_code=500,
            content={"error": "Traceback (most recent call last): File /opt/provider/internal.py KeyError: secret"},
        )

    return JSONResponse(content=_success("secondary", body, simulate))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8500, log_level="warning")
