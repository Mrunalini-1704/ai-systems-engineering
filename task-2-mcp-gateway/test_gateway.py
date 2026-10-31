"""End-to-end test for the MCP gateway.

Starts the downstream mock server and the gateway as subprocesses, then
fires scripted JSON-RPC requests through the gateway to prove:
  - tools/list is forwarded transparently.
  - a viewer token calling an admin_ tool is blocked with -32001, and the
    downstream server never sees that call.
  - a viewer token calling a normal tool is forwarded and succeeds.
  - an admin token calling an admin_ tool is forwarded and succeeds.

Run with: python test_gateway.py
"""

import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
PYTHON = sys.executable

DOWNSTREAM_URL = "http://127.0.0.1:8100/mcp"
GATEWAY_URL = "http://127.0.0.1:8200/mcp"

ADMIN_TOKEN = "admin-secret-123"
VIEWER_TOKEN = "viewer-secret-456"

failures = []


def check(condition: bool, description: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    if not condition:
        failures.append(description)


def wait_for(url: str, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.post(url, json={"jsonrpc": "2.0", "id": 0, "method": "tools/list"}, timeout=1.0)
            return True
        except httpx.HTTPError:
            time.sleep(0.2)
    return False


def call(url: str, method: str, params: dict, token: str | None = None, req_id: int = 1) -> dict:
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    resp = httpx.post(
        url,
        json={"jsonrpc": "2.0", "id": req_id, "method": method, "params": params},
        headers=headers,
        timeout=5.0,
    )
    return resp.json()


def main() -> None:
    downstream_proc = subprocess.Popen([PYTHON, str(HERE / "downstream_server.py")])
    gateway_proc = subprocess.Popen([PYTHON, str(HERE / "gateway.py")])

    try:
        if not wait_for(DOWNSTREAM_URL):
            print("downstream server did not start in time")
            sys.exit(1)
        if not wait_for(GATEWAY_URL):
            print("gateway did not start in time")
            sys.exit(1)

        # tools/list is forwarded; a viewer's response omits the admin_ tools
        resp = call(GATEWAY_URL, "tools/list", {}, token=ADMIN_TOKEN, req_id=1)
        tool_names = {t["name"] for t in resp.get("result", {}).get("tools", [])}
        check(
            {"get_weather", "admin_reset_key"} <= tool_names,
            "tools/list is forwarded through the gateway (admin sees every tool)",
        )
        resp = call(GATEWAY_URL, "tools/list", {}, token=VIEWER_TOKEN, req_id=1)
        tool_names = {t["name"] for t in resp.get("result", {}).get("tools", [])}
        check(
            tool_names == {"get_weather"},
            "a viewer's tools/list has the admin_ tools filtered out",
        )

        # viewer calling an admin_ tool -> blocked
        resp = call(
            GATEWAY_URL,
            "tools/call",
            {"name": "admin_reset_key", "arguments": {"customer_id": "CUST-00001"}},
            token=VIEWER_TOKEN,
            req_id=2,
        )
        check(
            resp.get("error", {}).get("code") == -32001,
            "viewer token calling admin_reset_key is blocked with -32001",
        )

        # viewer calling a normal tool -> forwarded and succeeds
        resp = call(
            GATEWAY_URL,
            "tools/call",
            {"name": "get_weather", "arguments": {"city": "Austin"}},
            token=VIEWER_TOKEN,
            req_id=3,
        )
        check(
            "result" in resp and resp["result"].get("isError") is False,
            "viewer token calling get_weather is forwarded and succeeds",
        )

        # admin calling an admin_ tool -> forwarded and succeeds
        resp = call(
            GATEWAY_URL,
            "tools/call",
            {"name": "admin_reset_key", "arguments": {"customer_id": "CUST-00001"}},
            token=ADMIN_TOKEN,
            req_id=4,
        )
        check(
            "result" in resp and resp["result"].get("isError") is False,
            "admin token calling admin_reset_key is forwarded and succeeds",
        )

        # no token / unknown token calling an admin_ tool -> blocked
        resp = call(
            GATEWAY_URL,
            "tools/call",
            {"name": "admin_reset_key", "arguments": {"customer_id": "CUST-00001"}},
            token=None,
            req_id=5,
        )
        check(
            resp.get("error", {}).get("code") == -32001,
            "missing/unknown token calling admin_reset_key is blocked with -32001",
        )

        # malformed request: params.name explicitly null -> clean JSON-RPC
        # error, not an unhandled 500 (a real bug caught during review: a
        # naive `params.get("name", "")` returns None here, not "", and
        # crashes on `.startswith`)
        resp_raw = httpx.post(
            GATEWAY_URL,
            json={"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": None, "arguments": {}}},
            headers={"Authorization": f"Bearer {VIEWER_TOKEN}"},
            timeout=5.0,
        )
        check(
            resp_raw.status_code == 200 and resp_raw.json().get("error", {}).get("code") == -32602,
            "params.name = null is rejected cleanly with -32602, not an unhandled 500",
        )

        # malformed request: params is not an object at all
        resp_raw = httpx.post(
            GATEWAY_URL,
            json={"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": "not-an-object"},
            headers={"Authorization": f"Bearer {VIEWER_TOKEN}"},
            timeout=5.0,
        )
        check(
            resp_raw.status_code == 200 and resp_raw.json().get("error", {}).get("code") == -32602,
            "non-object params is rejected cleanly with -32602, not an unhandled 500",
        )

        # downstream unreachable -> clean JSON-RPC error, not a raw 500 with
        # a stack trace (a real bug caught during review: the forwarding
        # call wasn't wrapped in try/except)
        downstream_proc.terminate()
        downstream_proc.wait(timeout=5)
        time.sleep(0.3)
        resp_raw = httpx.post(
            GATEWAY_URL,
            json={"jsonrpc": "2.0", "id": 8, "method": "tools/list", "params": {}},
            headers={"Authorization": f"Bearer {VIEWER_TOKEN}"},
            timeout=5.0,
        )
        body_text = resp_raw.text.lower()
        check(
            resp_raw.status_code == 200
            and resp_raw.json().get("error", {}).get("code") == -32003
            and "traceback" not in body_text
            and "site-packages" not in body_text,
            "downstream unreachable returns a clean JSON-RPC -32003 error, not a raw 500/stack trace",
        )

    finally:
        gateway_proc.terminate()
        downstream_proc.terminate()
        gateway_proc.wait(timeout=5)
        downstream_proc.wait(timeout=5)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
