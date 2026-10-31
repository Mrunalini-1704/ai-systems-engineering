"""Manual JSON-RPC-over-stdio test harness for server.py.

Spawns the server as a subprocess and talks to it using raw newline-delimited
JSON-RPC (no MCP client library) so we can independently verify:
  1. Every single line written to stdout parses as valid JSON (protocol
     purity -- no stray print()/logging leaking onto stdout).
  2. Invalid tool arguments come back as a JSON-RPC error with code -32602
     (INVALID_PARAMS), not a crash or a silently "successful" response.
  3. Valid calls succeed normally.

Run with: python test_stdio_isolation.py
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

SERVER_PATH = Path(__file__).parent / "server.py"
PYTHON = sys.executable

failures = []


def check(condition: bool, description: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    if not condition:
        failures.append(description)


def main() -> None:
    proc = subprocess.Popen(
        [PYTHON, str(SERVER_PATH)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stdout():
        for line in proc.stdout:
            stdout_lines.append(line)

    def read_stderr():
        for line in proc.stderr:
            stderr_lines.append(line)

    t_out = threading.Thread(target=read_stdout, daemon=True)
    t_err = threading.Thread(target=read_stderr, daemon=True)
    t_out.start()
    t_err.start()

    def send(msg: dict) -> None:
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    def responses_with_id(req_id, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            for line in list(stdout_lines):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("id") == req_id:
                    return obj
            time.sleep(0.05)
        return None

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "0.1"},
                },
            }
        )
        init_resp = responses_with_id(1)
        check(init_resp is not None and "result" in init_resp, "initialize handshake succeeds")

        send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        list_resp = responses_with_id(2)
        tool_names = (
            {t["name"] for t in list_resp["result"]["tools"]}
            if list_resp and "result" in list_resp
            else set()
        )
        check(
            {"get_customer_record", "trigger_refund"} <= tool_names,
            "tools/list returns both expected tools",
        )

        send(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "get_customer_record", "arguments": {"customer_id": "CUST-00001"}},
            }
        )
        valid_resp = responses_with_id(3)
        check(
            valid_resp is not None and "result" in valid_resp,
            "valid get_customer_record call succeeds",
        )

        send(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "get_customer_record", "arguments": {"customer_id": "not-a-valid-id"}},
            }
        )
        invalid_resp = responses_with_id(4)
        check(
            invalid_resp is not None
            and "error" in invalid_resp
            and invalid_resp["error"]["code"] == -32602,
            "malformed customer_id is rejected with JSON-RPC -32602 (INVALID_PARAMS)",
        )

        send(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "trigger_refund",
                    "arguments": {"customer_id": "CUST-00001", "amount": -5.0, "reason": "too short"},
                },
            }
        )
        bad_refund_resp = responses_with_id(5)
        check(
            bad_refund_resp is not None
            and "error" in bad_refund_resp
            and bad_refund_resp["error"]["code"] == -32602,
            "negative amount + short reason both rejected with -32602",
        )

        send(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "trigger_refund",
                    "arguments": {
                        "customer_id": "CUST-00002",
                        "amount": 19.99,
                        "reason": "Customer requested a refund via support ticket #481",
                    },
                },
            }
        )
        good_refund_resp = responses_with_id(6)
        check(
            good_refund_resp is not None and "result" in good_refund_resp,
            "valid trigger_refund call succeeds",
        )

        send(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "not_a_real_tool", "arguments": {}},
            }
        )
        unknown_resp = responses_with_id(7)
        check(
            unknown_resp is not None
            and "error" in unknown_resp
            and unknown_resp["error"]["code"] == -32602,
            "unknown tool name rejected with -32602 (INVALID_PARAMS, per MCP spec)",
        )

        # edge case: unknown/extra property should be rejected (schema says
        # additionalProperties: false), not silently ignored
        send(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "get_customer_record",
                    "arguments": {"customer_id": "CUST-00001", "sneaky_extra_field": "x"},
                },
            }
        )
        extra_field_resp = responses_with_id(8)
        check(
            extra_field_resp is not None
            and "error" in extra_field_resp
            and extra_field_resp["error"]["code"] == -32602,
            "unexpected extra argument is rejected with -32602, not silently ignored",
        )

        # edge case: a numeric string for `amount` must be rejected, not
        # silently coerced to a float -- "amount" is specified as a float
        send(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {
                    "name": "trigger_refund",
                    "arguments": {
                        "customer_id": "CUST-00001",
                        "amount": "19.99",
                        "reason": "Refund for a duplicate charge",
                    },
                },
            }
        )
        string_amount_resp = responses_with_id(9)
        check(
            string_amount_resp is not None
            and "error" in string_amount_resp
            and string_amount_resp["error"]["code"] == -32602,
            "numeric-string amount ('19.99') is rejected rather than silently coerced",
        )

        # edge case: reason at exactly the 10-character boundary must succeed
        send(
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "trigger_refund",
                    "arguments": {"customer_id": "CUST-00001", "amount": 5, "reason": "1234567890"},
                },
            }
        )
        boundary_ok_resp = responses_with_id(10)
        check(
            boundary_ok_resp is not None and "result" in boundary_ok_resp,
            "reason at exactly the 10-character minimum boundary is accepted",
        )

        time.sleep(0.2)

        stdout_parse_failures = []
        for line in stdout_lines:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                stdout_parse_failures.append(line)
        check(
            len(stdout_parse_failures) == 0,
            f"every stdout line is valid JSON (found {len(stdout_parse_failures)} non-JSON lines)",
        )

        check(len(stderr_lines) > 0, "log output was written to stderr")

    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

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