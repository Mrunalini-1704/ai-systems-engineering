"""End-to-end test for the rate-limiting & fallback router.

Starts providers_mock.py and router.py as subprocesses and proves:
  - A tenant that exceeds 50,000 tokens/min is rejected with a
    standardized 429, without ever reaching a provider.
  - A simulated primary 429 fails over to the secondary provider.
  - A simulated primary timeout (>3000ms) fails over to the secondary
    provider.
  - If both primary and secondary fail, the client gets a standardized,
    sanitized error body with no stack trace or internal detail leakage.

Run with: python test_router.py
"""

import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

HERE = Path(__file__).parent
PYTHON = sys.executable

PROVIDERS_URL_PROBE = "http://127.0.0.1:8500/primary/complete"
ROUTER_URL = "http://127.0.0.1:8600/v1/complete"

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
            httpx.post(url, json={"prompt": "healthcheck"}, timeout=1.0)
            return True
        except httpx.HTTPError:
            time.sleep(0.2)
    return False


def complete(**kwargs) -> httpx.Response:
    return httpx.post(ROUTER_URL, json=kwargs, timeout=10.0)


def main() -> None:
    # Fresh usage.db per run so rate-limit state doesn't leak across test runs.
    db_path = HERE / "usage.db"
    if db_path.exists():
        db_path.unlink()

    providers_proc = subprocess.Popen([PYTHON, str(HERE / "providers_mock.py")])
    router_proc = subprocess.Popen([PYTHON, str(HERE / "router.py")])

    try:
        if not wait_for(PROVIDERS_URL_PROBE):
            print("providers_mock did not start in time")
            sys.exit(1)

        # router needs its own readiness probe (different payload shape)
        deadline = time.time() + 10.0
        router_ready = False
        while time.time() < deadline:
            try:
                httpx.post(ROUTER_URL, json={"tenant_key": "healthcheck", "prompt": "hi", "estimated_tokens": 1}, timeout=1.0)
                router_ready = True
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        if not router_ready:
            print("router did not start in time")
            sys.exit(1)

        # --- 1. normal request succeeds via primary ---
        tenant = f"tenant-{uuid.uuid4()}"
        resp = complete(tenant_key=tenant, prompt="hello", estimated_tokens=100)
        body = resp.json()
        check(
            resp.status_code == 200 and body.get("_routed_via") == "primary",
            "a normal request is routed to primary and succeeds",
        )

        # --- 2. rate limit: two requests that together exceed 50,000 tokens/min ---
        rl_tenant = f"tenant-{uuid.uuid4()}"
        # max_tokens: the mock spends its whole completion budget, so the
        # first request still holds ~30,000 tokens after settling.
        first = complete(tenant_key=rl_tenant, prompt="big request 1", max_tokens=30_000)
        second = complete(tenant_key=rl_tenant, prompt="big request 2", max_tokens=30_000)
        check(first.status_code == 200, "first 30,000-token request is admitted")
        check(
            second.status_code == 429 and second.json().get("error", {}).get("code") == "RATE_LIMIT_EXCEEDED",
            "second request pushing tenant over 50,000 tokens/min is rejected with standardized 429",
        )

        # a different tenant is unaffected by rl_tenant's usage
        other_tenant = f"tenant-{uuid.uuid4()}"
        other_resp = complete(tenant_key=other_tenant, prompt="hello", estimated_tokens=30_000)
        check(other_resp.status_code == 200, "a different tenant's budget is independent (not shared)")

        # --- 3. primary returns 429 -> failover to secondary ---
        tenant = f"tenant-{uuid.uuid4()}"
        resp = complete(
            tenant_key=tenant, prompt="hello", estimated_tokens=100, simulate_primary="429"
        )
        body = resp.json()
        check(
            resp.status_code == 200 and body.get("_routed_via") == "secondary" and body.get("source") == "secondary",
            "primary 429 triggers automatic failover to secondary",
        )

        # --- 4. primary times out (>3000ms) -> failover to secondary ---
        tenant = f"tenant-{uuid.uuid4()}"
        start = time.monotonic()
        resp = complete(
            tenant_key=tenant, prompt="hello", estimated_tokens=100, simulate_primary="timeout"
        )
        elapsed = time.monotonic() - start
        body = resp.json()
        check(
            resp.status_code == 200 and body.get("_routed_via") == "secondary",
            "primary timeout (mock sleeps 5s against a 3s budget) triggers failover to secondary",
        )
        check(elapsed < 4.5, f"failover happens close to the 3s timeout budget, not after the full 5s primary sleep (took {elapsed:.2f}s)")

        # --- 5. both primary and secondary fail -> sanitized standardized error ---
        tenant = f"tenant-{uuid.uuid4()}"
        resp = complete(
            tenant_key=tenant,
            prompt="hello",
            estimated_tokens=100,
            simulate_primary="429",
            simulate_secondary="fail",
        )
        body = resp.json()
        check(resp.status_code == 502, "both providers failing returns HTTP 502")
        check(
            "error" in body and set(body["error"].keys()) == {"code", "message"},
            "error body has the standardized {code, message} shape",
        )
        error_text = str(body).lower()
        check(
            "traceback" not in error_text and "exception" not in error_text and "site-packages" not in error_text,
            "error body contains no stack trace / internal implementation details",
        )

        # --- 6. request validation errors use the SAME standardized error
        # shape as upstream/provider errors, not FastAPI's default {"detail": [...]}
        missing_key_resp = httpx.post(ROUTER_URL, json={"prompt": "hello", "estimated_tokens": 100}, timeout=5.0)
        check(
            missing_key_resp.status_code == 422
            and set(missing_key_resp.json().get("error", {}).keys()) == {"code", "message"},
            "missing tenant_key returns the standardized {error: {code, message}} shape, not FastAPI's default",
        )

        empty_key_resp = complete(tenant_key="", prompt="hello", estimated_tokens=100)
        check(
            empty_key_resp.status_code == 422,
            "empty-string tenant_key is rejected rather than accepted as a valid tenant",
        )

        zero_tokens_resp = complete(tenant_key=f"tenant-{uuid.uuid4()}", prompt="hello", estimated_tokens=0)
        check(zero_tokens_resp.status_code == 422, "estimated_tokens=0 is rejected, not silently admitted")

    finally:
        router_proc.terminate()
        providers_proc.terminate()
        router_proc.wait(timeout=5)
        providers_proc.wait(timeout=5)

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
