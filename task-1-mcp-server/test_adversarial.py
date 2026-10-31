"""Adversarial probe suite for server.py -- independent verification pass.

Spawns the server fresh and fires edge-case JSON-RPC requests not covered by
test_stdio_isolation.py. Captures raw stdout and asserts every non-empty line
is valid JSON, and checks specific error codes / behaviors.

Run with: python test_adversarial.py
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


def check(condition: bool, description: str, extra: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}" + (f" -- {extra}" if extra and not condition else ""))
    if not condition:
        failures.append(description + (f" ({extra})" if extra else ""))


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
                    "clientInfo": {"name": "adversarial-test-client", "version": "0.1"},
                },
            }
        )
        init_resp = responses_with_id(1)
        check(init_resp is not None and "result" in init_resp, "initialize handshake succeeds")
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        next_id = 2

        # --- 1. amount = 1e300 (very large float) ---
        req_id = next_id; next_id += 1
        send({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": "trigger_refund", "arguments": {
                "customer_id": "CUST-00001", "amount": 1e300, "reason": "Large refund test case"}},
        })
        resp = responses_with_id(req_id)
        check(resp is not None and "result" in resp,
              "amount=1e300 (very large float) is accepted (positive, finite)",
              str(resp))

        # --- 2. amount = Infinity (not valid JSON, but test if float('inf') via literal is even parseable) ---
        # Standard json module can't encode inf as valid JSON; but Python's json.dumps CAN emit
        # non-standard `Infinity` token which some parsers accept. Send it raw to probe robustness.
        req_id = next_id; next_id += 1
        raw = json.dumps({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": "trigger_refund", "arguments": {
                "customer_id": "CUST-00001", "amount": None, "reason": "placeholder reason"}},
        })
        # Manually replace the amount value token to inject literal Infinity (invalid strict JSON,
        # but Python's json.loads accepts it by default -- probing parser leniency).
        raw = raw.replace('"amount": null', '"amount": Infinity')
        proc.stdin.write(raw + "\n")
        proc.stdin.flush()
        resp = responses_with_id(req_id)
        check(resp is not None and "error" in resp,
              "amount=Infinity (non-finite) is rejected with an error, not accepted/crash",
              str(resp))

        # --- 3. amount = -Infinity ---
        req_id = next_id; next_id += 1
        raw = json.dumps({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": "trigger_refund", "arguments": {
                "customer_id": "CUST-00001", "amount": None, "reason": "placeholder reason"}},
        })
        raw = raw.replace('"amount": null', '"amount": -Infinity')
        proc.stdin.write(raw + "\n")
        proc.stdin.flush()
        resp = responses_with_id(req_id)
        check(resp is not None and "error" in resp and resp["error"].get("code") == -32602,
              "amount=-Infinity is rejected with -32602",
              str(resp))

        # --- 4. amount = NaN ---
        req_id = next_id; next_id += 1
        raw = json.dumps({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": "trigger_refund", "arguments": {
                "customer_id": "CUST-00001", "amount": None, "reason": "placeholder reason"}},
        })
        raw = raw.replace('"amount": null', '"amount": NaN')
        proc.stdin.write(raw + "\n")
        proc.stdin.flush()
        resp = responses_with_id(req_id)
        check(resp is not None and "error" in resp,
              "amount=NaN is rejected with an error (NaN > 0 is False in Python, but Pydantic may special-case)",
              str(resp))

        # --- 5. customer_id with leading/trailing whitespace ---
        req_id = next_id; next_id += 1
        send({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": "get_customer_record", "arguments": {"customer_id": " CUST-00001 "}},
        })
        resp = responses_with_id(req_id)
        check(resp is not None and "error" in resp and resp["error"].get("code") == -32602,
              "customer_id with leading/trailing whitespace is rejected with -32602",
              str(resp))

        # --- 6. customer_id with 6 digits instead of 5 ---
        req_id = next_id; next_id += 1
        send({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": "get_customer_record", "arguments": {"customer_id": "CUST-123456"}},
        })
        resp = responses_with_id(req_id)
        check(resp is not None and "error" in resp and resp["error"].get("code") == -32602,
              "customer_id with 6 digits (CUST-123456) is rejected with -32602",
              str(resp))

        # --- 7. reason exactly length 10, all whitespace ---
        req_id = next_id; next_id += 1
        send({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": "trigger_refund", "arguments": {
                "customer_id": "CUST-00001", "amount": 10.0, "reason": "          "}},
        })
        resp = responses_with_id(req_id)
        # Not covered by original spec whether whitespace-only should be rejected; report actual behavior.
        print(f"    -> reason=10 spaces result: {resp}")
        check(resp is not None, "reason of 10 whitespace chars gets *some* response (not a hang)", str(resp))

        # --- 8. reason as non-string (a number) ---
        req_id = next_id; next_id += 1
        send({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": "trigger_refund", "arguments": {
                "customer_id": "CUST-00001", "amount": 10.0, "reason": 1234567890}},
        })
        resp = responses_with_id(req_id)
        check(resp is not None and "error" in resp and resp["error"].get("code") == -32602,
              "reason as a non-string number is rejected with -32602",
              str(resp))

        # --- 9. missing arguments entirely (omit "arguments" key) ---
        req_id = next_id; next_id += 1
        send({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": "get_customer_record"},
        })
        resp = responses_with_id(req_id)
        check(resp is not None and "error" in resp,
              "tools/call with 'arguments' key entirely omitted returns an error (not a crash)",
              str(resp))

        # --- 10. arguments explicitly null ---
        req_id = next_id; next_id += 1
        send({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": "get_customer_record", "arguments": None},
        })
        resp = responses_with_id(req_id)
        check(resp is not None and "error" in resp,
              "tools/call with arguments=null returns an error (not a crash)",
              str(resp))

        # --- 11. tools/call request with no "params" key at all ---
        req_id = next_id; next_id += 1
        send({"jsonrpc": "2.0", "id": req_id, "method": "tools/call"})
        resp = responses_with_id(req_id)
        check(resp is not None and "error" in resp,
              "tools/call with no 'params' key at all returns an error (not a crash/hang)",
              str(resp))

        # --- 12. tool name with trailing space ---
        req_id = next_id; next_id += 1
        send({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": "get_customer_record ", "arguments": {"customer_id": "CUST-00001"}},
        })
        resp = responses_with_id(req_id)
        check(resp is not None and "error" in resp and resp["error"].get("code") == -32602,
              "tool name with trailing space ('get_customer_record ') is rejected as unknown tool (-32602)",
              str(resp))

        # --- 13. concurrent/pipelined requests: fire multiple before reading responses ---
        pipelined_ids = list(range(next_id, next_id + 5))
        next_id += 5
        for i, pid in enumerate(pipelined_ids):
            send({
                "jsonrpc": "2.0", "id": pid, "method": "tools/call",
                "params": {"name": "get_customer_record", "arguments": {"customer_id": f"CUST-0000{i % 2 + 1}"}},
            })
        all_ok = True
        results = {}
        for pid in pipelined_ids:
            r = responses_with_id(pid, timeout=8.0)
            results[pid] = r
            if r is None or r.get("id") != pid:
                all_ok = False
        check(all_ok, "pipelined/concurrent requests all get responses with correctly correlated IDs",
              str(results))

        # --- 14. amount as boolean (True/False are int subclass in Python -- classic gotcha) ---
        req_id = next_id; next_id += 1
        raw = json.dumps({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": "trigger_refund", "arguments": {
                "customer_id": "CUST-00001", "amount": True, "reason": "Boolean amount test case"}},
        })
        proc.stdin.write(raw + "\n")
        proc.stdin.flush()
        resp = responses_with_id(req_id)
        check(resp is not None and "error" in resp and resp["error"].get("code") == -32602,
              "amount=true (bool) is rejected with -32602, not silently accepted as 1.0",
              str(resp))

        # --- 15. customer_id lowercase ---
        req_id = next_id; next_id += 1
        send({
            "jsonrpc": "2.0", "id": req_id, "method": "tools/call",
            "params": {"name": "get_customer_record", "arguments": {"customer_id": "cust-00001"}},
        })
        resp = responses_with_id(req_id)
        check(resp is not None and "error" in resp and resp["error"].get("code") == -32602,
              "lowercase customer_id ('cust-00001') is rejected with -32602",
              str(resp))

        time.sleep(0.3)

        # --- Final: re-verify stdout purity across ALL adversarial traffic ---
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
            f"every stdout line under adversarial load is valid JSON (found {len(stdout_parse_failures)} bad lines)",
            "\n".join(stdout_parse_failures[:5]),
        )

        check(proc.poll() is None, "server process is still alive after all adversarial probes (no crash)")

    finally:
        proc.stdin.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print()
    print(f"--- stderr had {len(stderr_lines)} lines ---")
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All adversarial checks passed.")


if __name__ == "__main__":
    main()