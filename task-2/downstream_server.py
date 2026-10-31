"""Mock downstream MCP server.

Speaks JSON-RPC 2.0 over a single HTTP POST endpoint (/mcp). Trusts that
whoever is calling it (the gateway) has already done authorization -- it
just executes whatever tool it's asked to run.

Test-only extras:
  - Every request it receives is recorded and exposed at GET /_debug/received,
    so tests can prove a blocked call never reached this server.
  - Tool "slow_stream" answers as an SSE stream (Streamable HTTP style), to
    prove the gateway passes streamed responses through.
  - Tool "crash" returns an HTTP 500 with a fake stack trace, to prove the
    gateway doesn't relay internal error details.
"""

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

app = FastAPI(title="Downstream Mock MCP Server")

RECEIVED: list[dict] = []

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "admin_reset_key",
        "description": "Reset an API key for a customer (admin-only tool).",
        "inputSchema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    },
]


@app.get("/_debug/received")
async def debug_received():
    return RECEIVED


@app.delete("/_debug/received")
async def debug_reset():
    RECEIVED.clear()
    return {"cleared": True}


@app.get("/mcp")
async def mcp_resume(request: Request):
    # Streamable HTTP resumption: replays a stored tools/list response over SSE.
    async def sse():
        message = {"jsonrpc": "2.0", "id": 12, "result": {"tools": TOOLS}}
        yield b"event: message\nid: ev-2\ndata: " + json.dumps(message).encode() + b"\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    body = await request.json()
    RECEIVED.append({"body": body, "authorization": request.headers.get("authorization"),
                     "accept_encoding": request.headers.get("accept-encoding")})
    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}

    if "id" not in body or "method" not in body:
        # A notification, or a response to a server-initiated request: no
        # response body, per Streamable HTTP.
        return Response(status_code=202)

    if method == "tools/list":
        if params.get("_mock_sse"):  # answer as an SSE stream, Streamable HTTP style
            async def sse_list():
                message = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
                yield b"event: message\nid: ev-1\ndata: " + json.dumps(message).encode() + b"\n\n"

            return StreamingResponse(sse_list(), media_type="text/event-stream")
        return JSONResponse(
            {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}},
            headers={"Mcp-Session-Id": "mock-session-1"},
        )

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}

        if name == "crash":
            return PlainTextResponse(
                'Traceback (most recent call last):\n  File "/srv/internal/secret_module.py", line 42\nKeyError: db_password',
                status_code=500,
            )

        if name == "slow_stream":
            async def sse():
                yield b"event: message\ndata: " + json.dumps(
                    {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 50}}
                ).encode() + b"\n\n"
                yield b"event: message\ndata: " + json.dumps(
                    {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": "done"}], "isError": False}}
                ).encode() + b"\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"[downstream] executed '{name}' with arguments={arguments}",
                        }
                    ],
                    "isError": False,
                },
            }
        )

    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8100)