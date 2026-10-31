"""Hardening tests for the router (beyond test_router.py / test_extended_probes.py).

  1. Total-deadline failover: a primary that trickles its body one byte per
     second never trips httpx's per-read 3s timeout (shown by a control
     request), but the router still fails over at ~3s.
  2. Reserve -> settle: provider-reported usage replaces the estimate, so
     unused budget is refunded.
  3. Release: a request no provider served costs the tenant nothing.
  4. A client can't under-declare: a tiny `estimated_tokens` on a huge
     prompt still reserves the server-side estimate.
  5. 429s carry a sensible Retry-After; a request larger than the whole
     budget gets none (it can never succeed).
  6. Error sanitization: upstream internals never reach the client, every
     error has exactly {code, message} plus an X-Request-ID header, and
     framework-level errors (404, malformed JSON) use the same envelope.
  7. Global eviction removes idle tenants' expired rows.
  8. Cancellation safety: cancelling many in-flight limiter calls never
     corrupts the shared connection's transaction state.

Run with: python test_hardening.py
"""

import asyncio
import random
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

HERE = Path(__file__).parent
PYTHON = sys.executable
sys.path.insert(0, str(HERE))

ROUTER_URL = "http://127.0.0.1:8600/v1/complete"
PRIMARY_URL = "http://127.0.0.1:8500/primary/complete"
DB_PATH = HERE / "usage.db"

failures = []


def check(condition: bool, description: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}" + (f"  ({detail})" if detail and not condition else ""))
    if not condition:
        failures.append(description)


def complete(**kwargs) -> httpx.Response:
    return httpx.post(ROUTER_URL, json=kwargs, timeout=20.0)


def used(tenant: str) -> int:
    import rate_limiter

    conn = sqlite3.connect(DB_PATH)
    try:
        return conn.execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant_id = ? AND ts > ?",
            (rate_limiter.tenant_id(tenant), time.time() - rate_limiter.WINDOW_SECONDS),
        ).fetchone()[0]
    finally:
        conn.close()


def wait_ready(timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if complete(tenant_key="hc", prompt="hi", max_tokens=1).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    return False


def http_checks() -> None:
    # 1. Trickling primary: control first, then the router.
    start = time.monotonic()
    r = httpx.post(PRIMARY_URL, json={"prompt": "x"}, headers={"X-Simulate": "trickle"}, timeout=3.0)
    control = time.monotonic() - start
    check(r.status_code == 200 and control > 5.0,
          f"control: plain httpx timeout=3.0 does NOT catch a trickling body (took {control:.1f}s)")

    start = time.monotonic()
    r = complete(tenant_key=f"t-{uuid.uuid4()}", prompt="hi", simulate_primary="trickle")
    elapsed = time.monotonic() - start
    check(r.status_code == 200 and r.json().get("_routed_via") == "secondary" and 2.9 < elapsed < 4.0,
          f"router enforces a TOTAL 3000ms deadline: trickling primary fails over to secondary at {elapsed:.2f}s",
          r.text[:200])

    # 2. Settle to actual usage.
    t = f"settle-{uuid.uuid4()}"
    r = complete(tenant_key=t, prompt="hi", estimated_tokens=40_000, simulate_primary="usage:50")
    check(r.status_code == 200 and used(t) == 50,
          f"reservation of 40,000 settles to provider-reported usage of 50 tokens (recorded {used(t)})")
    r = complete(tenant_key=t, prompt="hi", estimated_tokens=40_000)
    check(r.status_code == 200, "refunded budget is usable: a second 40,000-token request is admitted")

    t = f"settle-fo-{uuid.uuid4()}"
    complete(tenant_key=t, prompt="hi", estimated_tokens=30_000, simulate_primary="429", simulate_secondary="usage:75")
    check(used(t) == 75, f"settlement also applies when the secondary serves the request (recorded {used(t)})")

    # 3. Release when nothing served the request.
    t = f"release-{uuid.uuid4()}"
    r = complete(tenant_key=t, prompt="hi", estimated_tokens=45_000, simulate_primary="429", simulate_secondary="fail")
    check(r.status_code == 502 and used(t) == 0,
          f"both providers failed -> 502 and the 45,000-token reservation is released (recorded {used(t)})")

    # 4. Server-side estimate floor.
    t = f"underdeclare-{uuid.uuid4()}"
    big_prompt = "word " * 8_000  # 40,000 bytes ~ 10,000 tokens
    r = complete(tenant_key=t, prompt=big_prompt, estimated_tokens=1)
    check(r.status_code == 200 and used(t) >= 10_000,
          f"estimated_tokens=1 on a ~10k-token prompt still charges the server estimate (recorded {used(t)})")
    cjk_t = f"cjk-{uuid.uuid4()}"
    r = complete(tenant_key=cjk_t, prompt="漢" * 4000, max_tokens=1)  # 4000 chars, 12000 UTF-8 bytes
    check(r.status_code == 200 and used(cjk_t) >= 3000,
          f"CJK prompts are estimated by UTF-8 bytes, not characters (recorded {used(cjk_t)} for 4,000 chars)")

    # 5. Retry-After.
    t = f"retry-{uuid.uuid4()}"
    complete(tenant_key=t, prompt="hi", estimated_tokens=45_000, simulate_primary="usage:45000")
    r = complete(tenant_key=t, prompt="hi", estimated_tokens=10_000)
    ra = r.headers.get("retry-after")
    check(r.status_code == 429 and ra is not None and 1 <= int(ra) <= 60,
          f"429 includes Retry-After derived from when the window frees up (Retry-After: {ra})")
    r = complete(tenant_key=f"huge-{uuid.uuid4()}", prompt="hi", estimated_tokens=60_000)
    check(r.status_code == 429 and "retry-after" not in r.headers,
          "a request larger than the entire budget gets 429 without a (misleading) Retry-After")

    # 6. Sanitization and envelope consistency.
    r = complete(tenant_key=f"t-{uuid.uuid4()}", prompt="hi", simulate_primary="429", simulate_secondary="fail")
    body = r.json()
    check(r.status_code == 502 and set(body) == {"error"} and set(body["error"]) == {"code", "message"}
          and "Traceback" not in r.text and "secret" not in r.text and "internal.py" not in r.text,
          "secondary's 500 body (with a fake stack trace) is never relayed to the client", r.text)
    check(len(r.headers.get("x-request-id", "")) == 32, "errors carry an X-Request-ID for log correlation")

    r = httpx.post(ROUTER_URL, content=b"{not json", headers={"Content-Type": "application/json"}, timeout=5)
    check(r.status_code == 422 and set(r.json().get("error", {})) == {"code", "message"} and "detail" not in r.json(),
          "malformed JSON body -> standardized 422 (not FastAPI's default {detail: ...})", r.text)
    r = httpx.get("http://127.0.0.1:8600/does-not-exist", timeout=5)
    check(r.status_code == 404 and r.json().get("error", {}).get("code") == "NOT_FOUND",
          "unknown route -> standardized 404 envelope", r.text)
    r = complete(tenant_key="t", prompt="hi", max_tokens=0)
    check(r.status_code == 422 and "max_tokens" not in r.text,
          "validation errors don't echo field names/values back (no internal schema leakage)", r.text)
    r = httpx.post(ROUTER_URL, content=b'{"tenant_key": "t", "prompt": "\\ud800"}',
                   headers={"Content-Type": "application/json"}, timeout=5)
    check(r.status_code == 422, "a lone-surrogate prompt is a 422 validation error, not a 500", r.text)

    for mode in ("error200", "nan"):
        t = f"bad200-{uuid.uuid4()}"
        r = complete(tenant_key=t, prompt="hi", simulate_primary=mode, simulate_secondary="usage:20")
        check(r.status_code == 200 and r.json().get("_routed_via") == "secondary" and "Traceback" not in r.text
              and used(t) == 20,
              f"primary 200 with a malformed body ({mode}) -> failover, not relayed or a 500", r.text[:200])

    r = complete(tenant_key=f"cap-{uuid.uuid4()}", prompt="hi", max_tokens=100, estimated_tokens=40_000)
    check(r.status_code == 200 and r.json()["usage"]["completion_tokens"] == 100,
          "estimated_tokens raises only the reservation; the provider still gets the client's max_tokens", r.text[:200])


async def limiter_checks() -> None:
    import rate_limiter

    # 7. Global eviction of idle tenants.
    idle = [f"idle-{uuid.uuid4()}" for _ in range(5)]
    conn = sqlite3.connect(DB_PATH)
    for t in idle:
        conn.execute("INSERT INTO token_usage (tenant_id, ts, tokens) VALUES (?, ?, ?)",
                     (rate_limiter.tenant_id(t), time.time() - 3600, 1000))
    conn.commit()
    deleted = await rate_limiter.evict_expired()
    remaining = conn.execute(
        "SELECT COUNT(*) FROM token_usage WHERE tenant_id IN (%s)" % ",".join("?" * len(idle)),
        [rate_limiter.tenant_id(t) for t in idle],
    ).fetchone()[0]
    stored_plain = conn.execute("SELECT COUNT(*) FROM token_usage WHERE tenant_id LIKE 'idle-%'").fetchone()[0]
    conn.close()
    check(deleted >= 5 and remaining == 0, f"evict_expired() removes idle tenants' expired rows (deleted {deleted})")
    check(stored_plain == 0, "tenant API keys are stored hashed, never in plaintext")

    # 8. Cancellation safety.
    tenant = f"cancel-{uuid.uuid4()}"
    errors = []

    async def one():
        try:
            r = await rate_limiter.reserve(tenant, 10)
            await rate_limiter.settle(r, 5)
        except asyncio.CancelledError:
            raise
        except rate_limiter.RateLimitExceeded:
            pass
        except Exception as exc:  # a corrupted connection shows up here
            errors.append(repr(exc))

    tasks = [asyncio.create_task(one()) for _ in range(400)]
    rng = random.Random(7)
    for _ in range(200):
        await asyncio.sleep(0)
        rng.choice(tasks).cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    # The connection must still be healthy afterwards.
    try:
        await rate_limiter.reserve(f"after-{uuid.uuid4()}", 1)
        healthy = True
    except Exception as exc:
        healthy = False
        errors.append(repr(exc))
    check(not errors and healthy,
          "400 concurrent reserve/settle calls with 200 random cancellations: no transaction errors, connection healthy",
          str(errors[:3]))

    # A reserve() cancelled while queued behind other DB work must not leave
    # an orphaned reservation charging the tenant for the rest of the window.
    orphan = f"orphan-{uuid.uuid4()}"
    rate_limiter._db_thread.submit(time.sleep, 0.3)  # keep the DB thread busy
    task = asyncio.create_task(rate_limiter.reserve(orphan, 40_000))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0.6)  # the queued insert lands, then its release runs
    check(await rate_limiter.used_tokens(orphan) == 0,
          "a reservation whose caller was cancelled mid-reserve is released, not charged for 60s")


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    providers = subprocess.Popen([PYTHON, str(HERE / "providers_mock.py")])
    router = subprocess.Popen([PYTHON, str(HERE / "router.py")])
    try:
        if not wait_ready():
            print("router/providers did not start in time")
            sys.exit(1)
        http_checks()
        asyncio.run(limiter_checks())
    finally:
        router.terminate()
        providers.terminate()
        router.wait(timeout=5)
        providers.wait(timeout=5)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
