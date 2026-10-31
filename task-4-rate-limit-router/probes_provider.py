"""Configurable mock provider for adversarial probes.

Router only ever forwards one custom header, X-Simulate (built from the
CompletionRequest.simulate_primary / simulate_secondary fields), so this
mock parses structured values out of that single header:

  - "sleep:<seconds>"  -> sleep for an exact float number of seconds before
                          responding 200 (for precise timeout-boundary tests).
  - "status:<code>"    -> return that HTTP status code immediately.
  - anything else / absent -> normal 200 response.

Also exposes:
  - GET  /count : {"primary": N, "secondary": N} request counters, to prove
    whether a call ever reached a provider at all.
  - POST /reset : zero the counters.
"""

import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Probe Providers")

counters = {"primary": 0, "secondary": 0}


async def _handle(request: Request, which: str):
    counters[which] += 1
    body = await request.json()
    simulate = request.headers.get("x-simulate", "")

    if simulate.startswith("sleep:"):
        await asyncio.sleep(float(simulate.split(":", 1)[1]))
    elif simulate.startswith("status:"):
        return JSONResponse(status_code=int(simulate.split(":", 1)[1]), content={"error": "simulated"})

    prompt = body.get("prompt", "")
    prompt_tokens = max(1, -(-len(prompt.encode("utf-8")) // 4))
    completion_tokens = int(body.get("max_tokens") or 16)  # spends its whole budget
    return JSONResponse(content={
        "source": which,
        "completion": f"[{which}] {prompt}",
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    })


@app.post("/primary/complete")
async def primary_complete(request: Request):
    return await _handle(request, "primary")


@app.post("/secondary/complete")
async def secondary_complete(request: Request):
    return await _handle(request, "secondary")


@app.get("/count")
async def count():
    return counters


@app.post("/reset")
async def reset():
    counters["primary"] = 0
    counters["secondary"] = 0
    return counters


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="warning")
