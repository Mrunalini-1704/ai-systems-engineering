r"""Regression tests for validation/isolation gaps found in a later review.

  1. customer_id "CUST-00001\n" was accepted: `^...$` with re.match() lets
     "$" match just before a trailing newline.
  2. customer_id with non-ASCII Unicode digits (Arabic-Indic U+0660..) was
     accepted: `\d` matches any Unicode decimal digit.
  3. reason = 12 spaces passed min_length=10.
  4. Well-formed call for a customer that doesn't exist must be a tool-level
     execution error (isError: true), not a "successful" result.
  5. fd-level stdout isolation: after _reserve_stdout_for_protocol(), a
     stray print() *and* a raw os.write(1, ...) both land on stderr, and
     only the transport's own writes reach the real stdout.

Run with: python test_hardening.py
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent
SERVER_PATH = HERE / "server.py"
PYTHON = sys.executable

failures = []


def check(condition: bool, description: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}" + (f"  ({detail})" if detail and not condition else ""))
    if not condition:
        failures.append(description)


def protocol_checks() -> None:
    proc = subprocess.Popen(
        [PYTHON, str(SERVER_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    stdout_lines: list[bytes] = []

    def read_stdout():
        for line in proc.stdout:
            stdout_lines.append(line)

    threading.Thread(target=read_stdout, daemon=True).start()
    threading.Thread(target=lambda: [None for _ in proc.stderr], daemon=True).start()

    def send(msg: dict) -> None:
        # ensure_ascii=False so non-ASCII digits go over the wire as real UTF-8.
        proc.stdin.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
        proc.stdin.flush()

    def response(req_id, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            for line in list(stdout_lines):
                obj = json.loads(line)
                if obj.get("id") == req_id:
                    return obj
            time.sleep(0.05)
        return None

    def call(req_id: int, name: str, arguments: dict):
        send({"jsonrpc": "2.0", "id": req_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}})
        return response(req_id)

    try:
        send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
        })
        check(response(1) is not None, "initialize handshake succeeds")
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        r = call(2, "get_customer_record", {"customer_id": "CUST-00001\n"})
        check(r is not None and r.get("error", {}).get("code") == -32602,
              "customer_id with trailing newline is rejected with -32602", str(r))

        r = call(3, "get_customer_record", {"customer_id": "CUST-٠٠٠٠١"})
        check(r is not None and r.get("error", {}).get("code") == -32602,
              "customer_id with non-ASCII Unicode digits is rejected with -32602", str(r))

        r = call(4, "trigger_refund", {"customer_id": "CUST-00001", "amount": 5.0, "reason": " " * 12})
        check(r is not None and r.get("error", {}).get("code") == -32602,
              "whitespace-only reason (12 spaces) is rejected with -32602", str(r))

        r = call(5, "trigger_refund", {"customer_id": "CUST-00001", "amount": 5.0, "reason": "  short  "})
        check(r is not None and r.get("error", {}).get("code") == -32602,
              "reason that is under 10 chars once padding is stripped is rejected with -32602", str(r))

        r = call(6, "get_customer_record", {"customer_id": "CUST-99999"})
        check(r is not None and r.get("result", {}).get("isError") is True,
              "well-formed but unknown customer returns a tool-level isError result, not success", str(r))

        r = call(7, "get_customer_record", {"customer_id": "CUST-00001"})
        ok = r is not None and r.get("result", {}).get("isError") is False
        record = json.loads(r["result"]["content"][0]["text"]) if ok else {}
        check(ok and record.get("customer_id") == "CUST-00001" and record.get("name") == "Alice Johnson",
              "valid lookup returns the record as parseable JSON text", str(r))

        r = call(8, "trigger_refund", {"customer_id": "CUST-00002", "amount": 19.999, "reason": "Item arrived damaged"})
        ok = r is not None and r.get("result", {}).get("isError") is False
        refund = json.loads(r["result"]["content"][0]["text"]) if ok else {}
        check(ok and refund.get("status") == "refund_triggered" and refund.get("amount") == 20.0,
              "valid refund succeeds and returns structured JSON", str(r))

        r = call(9, "no_such_tool", {})
        check(r is not None and r.get("error", {}).get("code") == -32602
              and "Unknown tool" in r["error"].get("message", ""),
              "unknown tool -> -32602 'Unknown tool' (MCP spec protocol error)", str(r))

        time.sleep(0.2)
        bad = [l for l in stdout_lines if not l.strip()]
        for l in stdout_lines:
            try:
                json.loads(l)
            except ValueError:
                bad.append(l)
        check(not bad, "every stdout line is valid JSON-RPC", repr(bad[:3]))
        check(all(not l.endswith(b"\r\n") for l in stdout_lines),
              "stdout messages are '\\n'-delimited with no '\\r' (no Windows newline translation)")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


# Run in a child process: imports server, takes over stdout exactly as
# main() does, then misbehaves in every way that would normally corrupt
# stdout -- Python print(), a raw fd-1 write (what C extensions do), and a
# child process inheriting fd 1.
ROGUE_SCRIPT = r"""
import os, subprocess, sys
sys.path.insert(0, sys.argv[1])
import server
proto = server._reserve_stdout_for_protocol()
print("STRAY-PRINT")
sys.stdout.flush()
os.write(1, b"STRAY-FD1-WRITE\n")
subprocess.run([sys.executable, "-c", "print('STRAY-CHILD-PROCESS')"])
proto.wrapped.write('{"jsonrpc":"2.0","id":1,"result":{}}\n')
proto.wrapped.flush()
"""


def fd_isolation_checks() -> None:
    res = subprocess.run(
        [PYTHON, "-c", ROGUE_SCRIPT, str(HERE)], capture_output=True, timeout=30
    )
    out = res.stdout.decode("utf-8", "replace")
    err = res.stderr.decode("utf-8", "replace")
    check(res.returncode == 0, "fd-isolation probe ran cleanly", err[-500:])
    check(out == '{"jsonrpc":"2.0","id":1,"result":{}}\n',
          "only the transport's message reaches real stdout", repr(out))
    check(all(s in err for s in ("STRAY-PRINT", "STRAY-FD1-WRITE", "STRAY-CHILD-PROCESS")),
          "stray print(), raw fd-1 write, and child-process output are all diverted to stderr", repr(err[-300:]))


def malformed_input_checks() -> None:
    """Input the SDK's stdio transport would only log about (no reply), and
    a client that closes stdin right after its last request."""
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        '{"jsonrpc":"2.0","id":2,"method":"tools/call", BROKEN',
        '[{"jsonrpc":"2.0","id":3,"method":"tools/list"}]',
        json.dumps({"jsonrpc": "1.0", "id": 4, "method": "tools/list"}),
        json.dumps({"jsonrpc": "2.0", "id": 5, "method": "foo/bar"}),
        json.dumps({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "trigger_refund", "arguments": {
            "customer_id": "CUST-00001", "amount": 0.001, "reason": "sub-cent refund test"}}}),
        json.dumps({"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"arguments": {}}}),
        json.dumps({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": ["get_customer_record"]}),
        json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/list"}),  # last line before EOF
    ]
    res = subprocess.run([PYTHON, str(SERVER_PATH)], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, encoding="utf-8", timeout=30)
    replies = [json.loads(line) for line in res.stdout.splitlines()]
    codes = [(r.get("id"), r.get("error", {}).get("code")) for r in replies if "error" in r]
    check(codes[:2] == [(None, -32700), (None, -32600)],
          "invalid JSON -> -32700 and a batch array -> -32600, both with id null (not just a log notification)", str(codes))
    check((4, -32600) in codes and (5, -32601) in codes,
          "'jsonrpc': '1.0' -> -32600 and an unknown method -> -32601 Method not found, with the request's id", str(codes))
    check((6, -32602) in codes, "a sub-cent amount (0.001, which would refund 0.00) -> -32602", str(codes))
    messages = {r.get("id"): r["error"]["message"] for r in replies if "error" in r}
    check(messages.get(8) == "params.name: Field required" and messages.get(9) == "params must be an object",
          "malformed tools/call params -> -32602 naming the actual problem (not the SDK's generic message)",
          str(messages))
    check(any(r.get("id") == 7 and "result" in r for r in replies) and res.returncode == 0,
          "the request sent right before stdin closes is still answered, then the server exits cleanly",
          f"rc={res.returncode} ids={[r.get('id') for r in replies]}")


def main() -> None:
    protocol_checks()
    fd_isolation_checks()
    malformed_input_checks()
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()