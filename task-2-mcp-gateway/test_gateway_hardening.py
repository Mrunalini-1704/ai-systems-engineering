"""Hardening tests for the MCP gateway (beyond the core happy/blocked paths
in test_gateway.py).

Uses the downstream mock's /_debug/received log to prove *where* a request
was stopped -- a blocked call must not merely return -32001, it must never
reach the downstream server at all.

Run with: python test_gateway_hardening.py
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
PYTHON = sys.executable

DOWNSTREAM = "http://127.0.0.1:8100"
GATEWAY_URL = "http://127.0.0.1:8200/mcp"
ADMIN = {"Authorization": "Bearer admin-secret-123"}
VIEWER = {"Authorization": "Bearer viewer-secret-456"}

failures = []


def check(condition: bool, description: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}" + (f"  ({detail})" if detail and not condition else ""))
    if not condition:
        failures.append(description)


def wait_for(url: str, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=1.0)
            return True
        except httpx.HTTPError:
            time.sleep(0.2)
    return False


def received() -> list[dict]:
    return httpx.get(f"{DOWNSTREAM}/_debug/received", timeout=5).json()


def reset_received() -> None:
    httpx.delete(f"{DOWNSTREAM}/_debug/received", timeout=5)


def rpc(method: str, params=None, req_id=1, headers=VIEWER) -> httpx.Response:
    body = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        body["params"] = params
    return httpx.post(GATEWAY_URL, json=body, headers=headers, timeout=5)


def raw(content: bytes, headers=VIEWER) -> httpx.Response:
    return httpx.post(GATEWAY_URL, content=content, headers={**headers, "Content-Type": "application/json"}, timeout=5)


def err_code(resp: httpx.Response):
    try:
        return resp.json().get("error", {}).get("code")
    except ValueError:
        return None


def main() -> None:
    downstream = subprocess.Popen([PYTHON, str(HERE / "downstream_server.py")])
    gateway = subprocess.Popen([PYTHON, str(HERE / "gateway.py")])
    try:
        if not (wait_for(f"{DOWNSTREAM}/_debug/received") and wait_for("http://127.0.0.1:8200/healthz")):
            print("servers did not start in time")
            sys.exit(1)

        # --- Authentication ---
        reset_received()
        for label, hdrs in [
            ("no Authorization header", {}),
            ("unknown token", {"Authorization": "Bearer not-a-real-token"}),
            ("wrong scheme (Basic)", {"Authorization": "Basic admin-secret-123"}),
            ("bearer scheme with no token", {"Authorization": "Bearer"}),
        ]:
            r = rpc("tools/list", {}, headers=hdrs)
            check(r.status_code == 401 and err_code(r) == -32001 and "bearer" in r.headers.get("www-authenticate", "").lower(),
                  f"{label}: tools/list rejected with HTTP 401 + WWW-Authenticate", f"{r.status_code} {r.text}")
        check(received() == [], "unauthenticated requests never reach the downstream server")

        r = rpc("tools/list", {}, headers={"Authorization": "bearer viewer-secret-456"})
        check(r.status_code == 200 and "result" in r.json(), "Bearer scheme is matched case-insensitively (RFC 7235)")

        # --- Authorization: blocked calls never reach downstream ---
        reset_received()
        for name in ["admin_reset_key", "ADMIN_reset_key", "Admin_Reset_Key", " admin_reset_key"]:
            r = rpc("tools/call", {"name": name, "arguments": {}}, req_id=f"blk-{name}")
            check(err_code(r) == -32001 and r.json().get("id") == f"blk-{name}"
                  and r.json()["error"]["message"] == "Unauthorized Tool Call",
                  f"viewer calling {name!r} -> -32001 'Unauthorized Tool Call' echoing request id", r.text)
        check(received() == [], "none of the blocked admin_ calls reached the downstream server", str(received()))

        reset_received()
        r = rpc("tools/call", {"name": "admin_reset_key", "arguments": {"customer_id": "CUST-1"}}, headers=ADMIN)
        got = received()
        check("result" in r.json() and len(got) == 1 and got[0]["body"]["params"]["name"] == "admin_reset_key",
              "admin calling admin_reset_key is forwarded exactly once")
        check(got and got[0]["authorization"] is None,
              "the client's bearer token is not leaked to the downstream server")

        # --- Duplicate-key smuggling (parser differential) ---
        reset_received()
        r = raw(b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_weather","name":"admin_reset_key"}}')
        check(err_code(r) == -32001 and received() == [],
              "duplicate 'name' keys resolving to admin_ are blocked and not forwarded")
        reset_received()
        r = raw(b'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"admin_reset_key","name":"get_weather","arguments":{"city":"x"}}}')
        got = received()
        check(len(got) == 1 and got[0]["body"]["params"]["name"] == "get_weather",
              "downstream receives exactly the (re-serialized) request that was authorized, not the raw ambiguous bytes",
              str(got))

        # --- JSON-RPC envelope validation ---
        reset_received()
        r = raw(b"{not json")
        check(err_code(r) == -32700, "malformed JSON -> -32700 Parse error", r.text)
        r = raw(b'[{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key"}}]')
        check(err_code(r) == -32600, "JSON-RPC batch (array) -> -32600, not partially inspected", r.text)
        r = raw(b'"just a string"')
        check(err_code(r) == -32600, "non-object JSON body -> -32600", r.text)
        r = raw(b'{"id":1,"method":"tools/list"}')
        check(err_code(r) == -32600, "missing jsonrpc:'2.0' -> -32600", r.text)
        r = raw(b'{"jsonrpc":"2.0","id":1,"method":42}')
        check(err_code(r) == -32600, "non-string method -> -32600", r.text)
        r = raw(b'{"jsonrpc":"2.0","id":{"a":1},"method":"tools/list"}')
        check(err_code(r) == -32600, "object-valued id -> -32600", r.text)
        r = raw(b'{"jsonrpc":"2.0","id":NaN,"method":"tools/call","params":{"name":"admin_reset_key"}}')
        check(r.status_code == 200 and err_code(r) == -32700, "non-standard NaN literal -> -32700, not a 500", r.text)
        r = raw(b'{"jsonrpc":"2.0","id":1e999,"method":"tools/call","params":{"name":"admin_reset_key"}}')
        check(r.status_code == 200 and err_code(r) == -32700, "overflowing number (1e999 -> inf) -> -32700, not a 500", r.text)
        r = raw(b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":' + b"[" * 50000 + b"]" * 50000 + b"}")
        check(r.status_code == 200 and err_code(r) == -32700, "pathologically deep nesting -> -32700, not a 500", r.text)
        r = raw(b'{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"pad":"' + b"x" * (1024 * 1024) + b'"}}')
        check(r.status_code == 413 and err_code(r) == -32600, "body over 1 MiB -> HTTP 413, not read whole", r.text[:200])
        r = raw(b'{"jsonrpc":"2.0","method":"tools/call","params":{"name":"admin_reset_key"}}')
        check(r.status_code == 403 and err_code(r) == -32001,
              "blocked admin_ call sent as a notification -> HTTP 403 (no 200 reply to a notification)", r.text)
        r = rpc("tools/call", {"name": "", "arguments": {}})
        check(err_code(r) == -32602, "empty tool name -> -32602", r.text)
        r = rpc("tools/call", None)
        check(err_code(r) == -32602, "tools/call with no params -> -32602", r.text)
        check(received() == [], "no invalid envelope was forwarded downstream", str(received()))

        # --- Transparent forwarding details ---
        r = rpc("tools/list", {})
        check(r.headers.get("mcp-session-id") == "mock-session-1", "Mcp-Session-Id response header is passed through")
        names = [t["name"] for t in r.json()["result"]["tools"]]
        check(names == ["get_weather"], "viewer's JSON tools/list response has admin_ tools filtered out", str(names))
        with httpx.stream("POST", GATEWAY_URL, headers={**VIEWER, "Accept": "application/json, text/event-stream"},
                          json={"jsonrpc": "2.0", "id": 11, "method": "tools/list", "params": {"_mock_sse": True}},
                          timeout=5) as s:
            raw_sse = s.read().decode()
        data = [ln[len("data: "):] for ln in raw_sse.splitlines() if ln.startswith("data: ")]
        names = [t["name"] for t in json.loads(data[0])["result"]["tools"]] if len(data) == 1 else None
        check(names == ["get_weather"] and "event: message" in raw_sse and "id: ev-1" in raw_sse,
              "viewer's SSE tools/list response is filtered too, other SSE fields intact", raw_sse)
        r = rpc("tools/list", {"_mock_sse": True}, headers=ADMIN)
        check("admin_reset_key" in r.text, "admin's tools/list is not filtered", r.text[:200])
        r = httpx.get(GATEWAY_URL, headers={**VIEWER, "Accept": "text/event-stream", "Last-Event-ID": "ev-1"}, timeout=5)
        check(r.status_code == 200 and "get_weather" in r.text and "admin_reset_key" not in r.text,
              "a tools/list response replayed on a resumed GET stream is filtered for a viewer", r.text)
        r = httpx.post(GATEWAY_URL, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=VIEWER, timeout=5)
        check(r.status_code == 202 and r.content == b"", "notification forwarded; downstream 202 + empty body passed through",
              f"{r.status_code} {r.content!r}")
        reset_received()
        r = httpx.post(GATEWAY_URL, json={"jsonrpc": "2.0", "id": "srv-1", "result": {"roots": []}}, headers=VIEWER, timeout=5)
        got = received()
        check(r.status_code == 202 and len(got) == 1 and got[0]["body"].get("result") == {"roots": []},
              "client JSON-RPC response (answer to a server request) is forwarded, not rejected", f"{r.status_code} {got}")
        check(got and got[0]["accept_encoding"] == "identity",
              "downstream is asked for an uncompressed body (raw bytes are relayed)", str(got))

        with httpx.stream("POST", GATEWAY_URL, headers={**VIEWER, "Accept": "application/json, text/event-stream"},
                          json={"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": "slow_stream"}},
                          timeout=5) as s:
            ctype = s.headers.get("content-type", "")
            body = s.read().decode()
        check(ctype.startswith("text/event-stream") and "notifications/progress" in body and '"id": 9' in body,
              "SSE (Streamable HTTP) responses are streamed through intact", f"{ctype} {body!r}")

        r = rpc("tools/call", {"name": "crash", "arguments": {}}, req_id=10)
        check(err_code(r) == -32003 and "Traceback" not in r.text and "db_password" not in r.text,
              "downstream HTTP 500 -> sanitized -32003, internal stack trace not relayed", r.text)

        # --- Concurrency: authz decisions stay correct per request ---
        async def burst():
            async with httpx.AsyncClient(timeout=10) as c:
                async def one(i):
                    hdrs = ADMIN if i % 2 == 0 else VIEWER
                    tool = "admin_reset_key" if i % 3 == 0 else "get_weather"
                    resp = await c.post(GATEWAY_URL, headers=hdrs, json={
                        "jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": tool, "arguments": {}}})
                    data = resp.json()
                    should_block = tool == "admin_reset_key" and hdrs is VIEWER
                    blocked = data.get("error", {}).get("code") == -32001
                    return data.get("id") == i and blocked == should_block
                return await asyncio.gather(*(one(i) for i in range(120)))

        reset_received()
        results = asyncio.run(burst())
        forwarded_admin_calls = [x for x in received() if x["body"]["params"]["name"] == "admin_reset_key"]
        check(all(results), f"120 concurrent mixed-role calls all got the correct allow/block decision and id ({sum(results)}/120)")
        check(len(forwarded_admin_calls) == 20,
              f"exactly the 20 admin-token admin_ calls reached downstream under concurrency (got {len(forwarded_admin_calls)})")
    finally:
        gateway.terminate()
        downstream.terminate()
        gateway.wait(timeout=5)
        downstream.wait(timeout=5)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
