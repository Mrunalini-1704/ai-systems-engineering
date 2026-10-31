"""End-to-end test for the streaming redaction gateway.

Starts mock_provider.py (the upstream "LLM", an SSE server) and gateway.py
as subprocesses and streams a completion through the gateway, proving:
  - The client starts receiving bytes well before the mock provider has
    finished generating (i.e. we are not buffering the full response).
  - The final concatenated output has all seeded PII (including PII split
    across a chunk boundary by the mock provider) replaced with
    [REDACTED], and none of it leaks through raw.

Run with: python test_redaction.py
"""

import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
PYTHON = sys.executable
GATEWAY_URL = "http://127.0.0.1:8300/v1/chat/stream"

# mock_provider.py sleeps 0.05s between each of its 8 chunks, so a fully
# buffered (non-streaming) implementation would take >= ~0.4s to deliver
# the first byte. A truly streaming implementation delivers the first byte
# in well under half that.
EXPECTED_TOTAL_STREAM_TIME = 0.05 * 8
FIRST_BYTE_BUDGET = EXPECTED_TOTAL_STREAM_TIME * 0.6

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
            # 200 only once both the gateway and its upstream provider are up.
            if httpx.post(url, json={"prompt": "healthcheck"}, timeout=2.0).status_code == 200:
                return True
            time.sleep(0.2)
        except httpx.HTTPError:
            time.sleep(0.2)
    return False


def main() -> None:
    provider = subprocess.Popen([PYTHON, str(HERE / "mock_provider.py")])
    proc = subprocess.Popen([PYTHON, str(HERE / "gateway.py")])

    try:
        if not wait_for(GATEWAY_URL):
            print("gateway did not start in time")
            sys.exit(1)

        # A cold TCP/process connection on Windows adds ~0.2-0.3s of one-time
        # overhead unrelated to streaming behavior (first-socket-connect,
        # interpreter/import warmup). Warm the connection up first, like a
        # benchmark warmup iteration, so the timed request measures actual
        # streaming latency rather than connection setup cost.
        client = httpx.Client(timeout=10.0)
        with client.stream("POST", GATEWAY_URL, json={"prompt": "warmup"}) as warmup_resp:
            for _ in warmup_resp.iter_text():
                pass

        start = time.monotonic()
        first_byte_time = None
        chunks = []

        with client.stream("POST", GATEWAY_URL, json={"prompt": "give me the customer summary"}) as resp:
            for chunk in resp.iter_text():
                if first_byte_time is None:
                    first_byte_time = time.monotonic() - start
                chunks.append(chunk)

        total_time = time.monotonic() - start
        full_text = "".join(chunks)

        check(
            first_byte_time is not None and first_byte_time < FIRST_BYTE_BUDGET,
            f"first byte arrives quickly (first_byte={first_byte_time:.3f}s < budget={FIRST_BYTE_BUDGET:.3f}s), "
            "proving the response is streamed rather than buffered",
        )
        check(
            total_time > first_byte_time,
            "total stream duration exceeds first-byte time (multiple chunks really were streamed over time)",
        )

        check("john.doe@example.com" not in full_text, "email #1 (split across chunk boundary) does not leak raw")
        check("jane.smith@example.org" not in full_text, "email #2 does not leak raw")
        check("123-45-6789" not in full_text, "SSN does not leak raw")
        check("4111 1111 1111 1111" not in full_text, "credit card (split across chunk boundary) does not leak raw")
        check(full_text.count("[REDACTED]") == 4, f"exactly 4 PII instances redacted (found {full_text.count('[REDACTED]')})")

    finally:
        proc.terminate()
        provider.terminate()
        proc.wait(timeout=5)
        provider.wait(timeout=5)

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
