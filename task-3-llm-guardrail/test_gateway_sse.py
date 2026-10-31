"""End-to-end tests for the OpenAI-compatible SSE path of the guardrail
gateway (POST /v1/chat/completions), against mock_provider.py over real HTTP.

Proves:
  - The output is a valid chat.completion.chunk SSE stream ending in
    [DONE], with role / finish_reason deltas preserved and PII redacted.
  - Time-to-first-content-token through the gateway is within a few ms of
    the provider's own TTFT (the guardrail adds negligible latency), and the
    first content arrives long before the stream ends.
  - A 5,000-delta stream with PII split across the very last two deltas is
    redacted correctly (the redactor state survives a long stream).
  - Provider HTTP errors -> sanitized 502 (no provider internals relayed).
  - Provider crash mid-stream -> a clean error event, and no partial,
    unredacted tail is flushed.
  - "stream": false returns one redacted chat.completion JSON.

Run with: python test_gateway_sse.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
PYTHON = sys.executable
GATEWAY = "http://127.0.0.1:8300/v1/chat/completions"
PROVIDER = "http://127.0.0.1:8301/v1/chat/completions"
MSGS = [{"role": "user", "content": "give me the customer summary"}]

failures = []


def check(condition: bool, description: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}" + (f"  ({detail})" if detail and not condition else ""))
    if not condition:
        failures.append(description)


def wait_ready(timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.post(GATEWAY, json={"messages": MSGS, "stream": False, "mock_delay": 0}, timeout=2)
            if r.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    return False


def read_sse(client: httpx.Client, url: str, body: dict) -> tuple[list, float | None, float]:
    """Returns (events, time_to_first_content, total_time)."""
    events, ttfc = [], None
    start = time.perf_counter()
    with client.stream("POST", url, json=body) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                events.append("[DONE]")
                continue
            obj = json.loads(data)
            events.append(obj)
            if ttfc is None and any((c.get("delta") or {}).get("content") for c in obj.get("choices", [])):
                ttfc = time.perf_counter() - start
    return events, ttfc, time.perf_counter() - start


def content_of(events: list) -> str:
    return "".join(
        (c.get("delta") or {}).get("content") or ""
        for e in events if isinstance(e, dict) for c in e.get("choices", [])
    )


def main() -> None:
    provider = subprocess.Popen([PYTHON, str(HERE / "mock_provider.py")])
    gateway = subprocess.Popen([PYTHON, str(HERE / "gateway.py")], stderr=subprocess.DEVNULL)
    try:
        if not wait_ready():
            print("gateway/provider did not start in time")
            sys.exit(1)
        client = httpx.Client(timeout=30)
        # Warm both connections (cold loopback connects on Windows add a
        # one-off ~0.2s unrelated to streaming).
        read_sse(client, GATEWAY, {"messages": MSGS})
        read_sse(client, PROVIDER, {"messages": MSGS, "stream": True})

        events, ttfc, total = read_sse(client, GATEWAY, {"messages": MSGS})
        text = content_of(events)
        chunks = [e for e in events if isinstance(e, dict)]
        check(events and events[-1] == "[DONE]", "stream terminates with data: [DONE]")
        check(all(e.get("object") == "chat.completion.chunk" and e.get("id") == "chatcmpl-mock" for e in chunks),
              "every event is a chat.completion.chunk carrying the upstream id")
        check(chunks and chunks[0]["choices"][0]["delta"].get("role") == "assistant",
              "the upstream role delta is passed through first")
        finish = [c for e in chunks for c in e["choices"] if c.get("finish_reason")]
        check(len(finish) == 1 and finish[0]["finish_reason"] == "stop" and chunks[-1]["choices"][0]["finish_reason"] == "stop",
              "finish_reason 'stop' is forwarded once, after all content")
        check(text.count("[REDACTED]") == 4 and "john.doe" not in text and "123-45-6789" not in text
              and "4111" not in text and "jane.smith" not in text,
              "all 4 PII instances redacted in the SSE content", text)
        check("Hello! Here is the customer summary" in text and "Let us know if you need anything else!" in text,
              "non-PII text passes through unchanged")

        # TTFT: gateway vs. provider directly.
        g_ttfc = min(read_sse(client, GATEWAY, {"messages": MSGS})[1] for _ in range(3))
        p_ttfc = min(read_sse(client, PROVIDER, {"messages": MSGS, "stream": True})[1] for _ in range(3))
        check(g_ttfc - p_ttfc < 0.05,
              f"guardrail adds negligible TTFT (gateway {g_ttfc*1000:.0f}ms vs provider {p_ttfc*1000:.0f}ms)")
        check(ttfc is not None and ttfc < total / 3,
              f"first content token ({ttfc:.3f}s) arrives long before the stream ends ({total:.3f}s): streamed, not buffered")

        # Long stream, PII split across the final two deltas.
        events, ttfc_long, total_long = read_sse(client, GATEWAY, {"messages": MSGS, "mock_script": "long", "mock_delay": 0.02})
        text = content_of(events)
        check(text.startswith("token0 token1") and text.count("token") == 5000 and "ceo@corp" not in text
              and text.endswith("contact: [REDACTED] done\n"),
              "5,000-delta stream: all text intact, email split across last two deltas redacted", text[-60:])
        check(ttfc_long < total_long / 5,
              f"long stream: first content at {ttfc_long*1000:.0f}ms of {total_long*1000:.0f}ms total")

        # Provider error before streaming starts -> sanitized 502.
        r = client.post(GATEWAY, json={"messages": MSGS, "mock_status": 500})
        check(r.status_code == 502 and r.json()["error"]["type"] == "upstream_error" and "worker.py" not in r.text,
              "provider HTTP 500 -> sanitized 502, provider internals not relayed", r.text)

        # Provider crash mid-stream.
        events, _, _ = read_sse(client, GATEWAY, {"messages": MSGS, "mock_script": "fail_midstream", "mock_delay": 0.01})
        errors = [e for e in events if isinstance(e, dict) and "error" in e]
        text = content_of(events)
        check(len(errors) == 1 and events[-1] == "[DONE]" and "Traceback" not in json.dumps(errors),
              "provider crash mid-stream -> one clean error event, then [DONE]", str(events[-3:]))
        check("john.doe@exa" not in text and "@exa" not in text,
              "aborted stream does not flush the half-received email ('john.doe@exa') held in the carry", text)

        # Provider sends an error event mid-stream (unredacted PII + internals in its message).
        events, _, _ = read_sse(client, GATEWAY, {"messages": MSGS, "mock_script": "error_event", "mock_delay": 0.01})
        errors = [e for e in events if isinstance(e, dict) and "error" in e]
        check(len(errors) == 1 and events[-1] == "[DONE]" and "ops@internal" not in json.dumps(events)
              and "billing.py" not in json.dumps(events),
              "provider error event mid-stream -> one sanitized error event, its message not relayed", str(events[-3:]))
        events, _, _ = read_sse(client, GATEWAY, {"messages": MSGS, "mock_script": "truncated", "mock_delay": 0.01})
        errors = [e for e in events if isinstance(e, dict) and "error" in e]
        check(len(errors) == 1 and events[-1] == "[DONE]" and "@exa" not in content_of(events),
              "stream cut off without [DONE] -> error event, half-received email not flushed", str(events[-3:]))

        events, _, _ = read_sse(client, GATEWAY, {"messages": MSGS, "mock_script": "odd_chunks", "mock_delay": 0.01})
        dumped = json.dumps(events)
        check(events and events[-1] == "[DONE]" and not any(isinstance(e, dict) and "error" in e for e in events)
              and content_of(events).count("[REDACTED]") == 4,
              "malformed upstream choices are skipped; the stream still completes", dumped[-300:])
        check("123-45-6789" not in dumped and "bob@corp.com" not in dumped,
              "PII in delta.refusal and in a choice-less event is redacted too", dumped)

        events, _, _ = read_sse(client, GATEWAY, {"messages": MSGS, "mock_script": "tool_call", "mock_delay": 0.01})
        args = "".join(
            ((t.get("function") or {}).get("arguments") or "")
            for e in events if isinstance(e, dict) for c in e.get("choices", []) for t in c["delta"].get("tool_calls", [])
        )
        check(args == '{"to": "[REDACTED]", "card": "[REDACTED]"}' and events[-1] == "[DONE]",
              "PII split across tool-call argument fragments is redacted in the stream", args)
        r = client.post(GATEWAY, json={"messages": MSGS, "stream": False, "mock_script": "tool_call", "mock_delay": 0})
        call = (r.json()["choices"][0]["message"].get("tool_calls") or [{}])[0]
        check(r.status_code == 200 and call.get("id") == "call_1" and call["function"]["name"] == "send_email"
              and call["function"]["arguments"] == '{"to": "[REDACTED]", "card": "[REDACTED]"}'
              and r.json()["choices"][0]["finish_reason"] == "tool_calls",
              "stream=false assembles the (redacted) tool call instead of dropping it", r.text[:300])

        r = client.post(GATEWAY, json={"messages": MSGS, "stream": False, "mock_script": "error_event", "mock_delay": 0})
        check(r.status_code == 502 and "ops@internal" not in r.text,
              "provider error event with stream=false -> sanitized 502", r.text)

        # Non-streaming mode.
        r = client.post(GATEWAY, json={"messages": MSGS, "stream": False, "mock_delay": 0})
        body = r.json()
        msg = body["choices"][0]["message"]["content"]
        check(r.status_code == 200 and body["object"] == "chat.completion" and msg.count("[REDACTED]") == 4
              and body["choices"][0]["finish_reason"] == "stop",
              "stream=false returns one redacted chat.completion", str(body)[:200])
        r = client.post(GATEWAY, json={"messages": MSGS, "stream": False, "mock_delay": 0, "mock_script": "usage_null"})
        check(r.status_code == 200 and r.json()["choices"][0]["message"]["content"].count("[REDACTED]") == 4,
              "trailing usage chunk with \"choices\": null doesn't break stream=false", r.text[:200])

        r = client.post(GATEWAY, json={"prompt": "no messages"})
        check(r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error",
              "request without a messages list -> 400 invalid_request_error")
        r = client.post(GATEWAY, content=b'{"messages": [], "temperature": NaN}', headers={"content-type": "application/json"})
        check(r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error",
              "non-standard NaN literal in body -> 400, not a 500", r.text)
        r = client.post(GATEWAY.replace("/completions", "/stream"), json={"no_prompt": True})
        check(r.status_code == 400 and r.json().get("error", {}).get("type") == "invalid_request_error",
              "/v1/chat/stream validation errors use the gateway's error shape, not FastAPI's {detail}", r.text)
        for bad in (b'{"messages": [], "temperature": 1e400}', b'{"messages": [{"role": "user", "content": "\\ud800"}]}'):
            r = client.post(GATEWAY, content=bad, headers={"content-type": "application/json"})
            check(r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error",
                  f"invalid body {bad[-30:]!r} -> 400, not a 500", r.text)
    finally:
        gateway.terminate()
        provider.terminate()
        gateway.wait(timeout=5)
        provider.wait(timeout=5)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
