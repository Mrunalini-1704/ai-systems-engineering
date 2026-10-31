"""Adversarial verification probes for task-4-rate-limit-router.

Independent of test_router.py / test_concurrency.py. Starts probes_provider.py
(a controllable mock) and router.py pointed at it via env vars, then exercises
scenarios the existing test suite does not cover.

Run with: python test_extended_probes.py
"""

import asyncio
import os
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

PROVIDER_URL = "http://127.0.0.1:8501"
ROUTER_URL = "http://127.0.0.1:8600/v1/complete"
DB_PATH = HERE / "usage.db"

failures = []


def check(condition: bool, description: str, extra: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}" + (f"  ({extra})" if extra else ""))
    if not condition:
        failures.append(description + (f" -- {extra}" if extra else ""))


def wait_for(url: str, payload=None, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.post(url, json=payload or {}, timeout=1.0)
            return True
        except httpx.HTTPError:
            time.sleep(0.2)
    return False


def complete(**kwargs) -> httpx.Response:
    return httpx.post(ROUTER_URL, json=kwargs, timeout=15.0)


def raw_post(json_body: dict) -> httpx.Response:
    return httpx.post(ROUTER_URL, json=json_body, timeout=10.0)


def reset_counters():
    httpx.post(f"{PROVIDER_URL}/reset", timeout=5.0)


def get_counters() -> dict:
    return httpx.get(f"{PROVIDER_URL}/count", timeout=5.0).json()


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()

    providers_proc = subprocess.Popen([PYTHON, str(HERE / "probes_provider.py")])
    env = os.environ.copy()
    env["PRIMARY_URL"] = f"{PROVIDER_URL}/primary/complete"
    env["SECONDARY_URL"] = f"{PROVIDER_URL}/secondary/complete"
    router_proc = subprocess.Popen([PYTHON, str(HERE / "router.py")], env=env)

    try:
        if not wait_for(f"{PROVIDER_URL}/primary/complete", {"prompt": "hi"}):
            print("probes_provider did not start")
            sys.exit(1)
        deadline = time.time() + 10.0
        ready = False
        while time.time() < deadline:
            try:
                httpx.post(ROUTER_URL, json={"tenant_key": "hc", "prompt": "hi", "estimated_tokens": 1}, timeout=1.0)
                ready = True
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        if not ready:
            print("router did not start")
            sys.exit(1)

        # ================================================================
        # Probe A: sliding-window EVICTION actually works (not just admission
        # math). Directly insert a stale row via sqlite3, bypassing the API,
        # then prove a fresh check treats it as expired and also physically
        # deletes it.
        # ================================================================
        import rate_limiter  # local import so it shares DB_PATH with the running router (same file)

        evict_tenant = f"evict-{uuid.uuid4()}"
        stale_ts = time.time() - 1000  # far outside the 60s window
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS token_usage (id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, "
            "ts REAL NOT NULL, tokens INTEGER NOT NULL)"
        )
        conn.execute(
            "INSERT INTO token_usage (tenant_id, ts, tokens) VALUES (?, ?, ?)",
            (rate_limiter.tenant_id(evict_tenant), stale_ts, 49_999),
        )
        conn.commit()
        conn.close()

        # If eviction were broken (stale row counted), 49_999 + 40_000 > 50_000 -> would raise.
        try:
            asyncio.run(rate_limiter.check_and_record(evict_tenant, 40_000))
            evicted_ok = True
        except rate_limiter.RateLimitExceeded:
            evicted_ok = False
        check(evicted_ok, "stale (>60s old) usage row is excluded from the sum, not just from future budget math")

        conn = sqlite3.connect(DB_PATH)
        remaining_stale = conn.execute(
            "SELECT COUNT(*) FROM token_usage WHERE tenant_id = ? AND ts = ?",
            (rate_limiter.tenant_id(evict_tenant), stale_ts),
        ).fetchone()[0]
        conn.close()
        check(remaining_stale == 0, "the stale row is physically DELETEd from the table, not just ignored")

        # ================================================================
        # Probe B: primary AND secondary both slow -- must not hang; total
        # wait must be bounded (primary 3s budget + secondary 5s budget).
        # ================================================================
        tenant = f"bothslow-{uuid.uuid4()}"
        start = time.monotonic()
        try:
            resp = complete(
                tenant_key=tenant,
                prompt="hi",
                estimated_tokens=100,
                simulate_primary="sleep:10",
                simulate_secondary="sleep:10",
            )
            elapsed = time.monotonic() - start
            hung = False
        except httpx.ReadTimeout:
            elapsed = time.monotonic() - start
            hung = True
            resp = None
        check(not hung, "router does not hang forever when both primary and secondary are simultaneously slow",
              f"elapsed={elapsed:.2f}s (client-side 15s safety timeout)")
        if resp is not None:
            check(
                elapsed < 9.5,
                "total wait time is bounded (~3s primary + ~5s secondary), not left unbounded",
                f"elapsed={elapsed:.2f}s, status={resp.status_code}",
            )
            check(resp.status_code == 502, "both-slow ultimately surfaces as a clean 502, not a hang/crash",
                  f"status={resp.status_code}")

        # ================================================================
        # Probe C: malformed input -- missing tenant_key, empty tenant_key,
        # zero/negative estimated_tokens. Must be clean validation errors,
        # never a 500 / stack trace.
        # ================================================================
        r1 = raw_post({"prompt": "hi", "estimated_tokens": 100})  # missing tenant_key
        check(r1.status_code in (400, 422), "missing tenant_key is rejected with a client error (not 500)",
              f"status={r1.status_code} body={r1.text[:200]}")
        check("traceback" not in r1.text.lower() and "site-packages" not in r1.text.lower(),
              "missing-tenant_key error body has no stack trace / internal path leakage")

        r2 = raw_post({"tenant_key": "", "prompt": "hi", "estimated_tokens": 100})  # empty tenant_key
        check(r2.status_code in (200, 400, 422),
              "empty-string tenant_key does not crash the server (200/400/422 all acceptable, 500 is not)",
              f"status={r2.status_code} body={r2.text[:200]}")

        r3 = raw_post({"tenant_key": "t", "prompt": "hi", "estimated_tokens": 0})  # zero tokens
        check(r3.status_code in (400, 422), "estimated_tokens=0 is rejected with a validation error, not admitted or crashed",
              f"status={r3.status_code} body={r3.text[:200]}")

        r4 = raw_post({"tenant_key": "t", "prompt": "hi", "estimated_tokens": -500})  # negative tokens
        check(r4.status_code in (400, 422), "negative estimated_tokens is rejected with a validation error",
              f"status={r4.status_code} body={r4.text[:200]}")

        # ================================================================
        # Probe D (F in the task list): when the rate limit itself rejects a
        # request, NO HTTP call reaches either provider.
        # ================================================================
        reset_counters()
        rl_tenant = f"noleak-{uuid.uuid4()}"
        over_resp = complete(tenant_key=rl_tenant, prompt="hi", estimated_tokens=60_000)  # exceeds budget outright
        counters_after = get_counters()
        check(over_resp.status_code == 429, "single request exceeding the whole budget is rejected", f"status={over_resp.status_code}")
        check(
            counters_after["primary"] == 0 and counters_after["secondary"] == 0,
            "rate-limited requests never reach primary or secondary providers at all",
            f"counters={counters_after}",
        )

        # ================================================================
        # Probe E: token accounting still happens correctly when a request is
        # ultimately served by the SECONDARY (failover) provider -- not just
        # on primary success.
        # ================================================================
        fo_tenant = f"failover-acct-{uuid.uuid4()}"
        # max_tokens, not estimated_tokens: the probe provider spends its
        # whole completion budget, so usage = 1 prompt token + 20,000.
        fo_resp = complete(
            tenant_key=fo_tenant, prompt="hi", max_tokens=20_000, simulate_primary="status:429"
        )
        check(
            fo_resp.status_code == 200 and fo_resp.json().get("_routed_via") == "secondary",
            "failover request succeeds via secondary", f"body={fo_resp.json() if fo_resp.status_code==200 else fo_resp.text}",
        )
        conn = sqlite3.connect(DB_PATH)
        used = conn.execute(
            "SELECT COALESCE(SUM(tokens),0) FROM token_usage WHERE tenant_id = ?", (rate_limiter.tenant_id(fo_tenant),)
        ).fetchone()[0]
        conn.close()
        check(used == 20_001, "tokens for a failover-routed request are recorded against the tenant's budget",
              f"recorded={used}, expected=20001")
        # Push the same tenant over budget purely via failover-routed calls to prove it's enforced end-to-end.
        fo_resp2 = complete(
            tenant_key=fo_tenant, prompt="hi", max_tokens=31_000, simulate_primary="status:429"
        )
        check(fo_resp2.status_code == 429,
              "a further failover-routed request that would exceed budget (20k+31k>50k) is correctly rejected",
              f"status={fo_resp2.status_code}")

        # ================================================================
        # Probe F: precise timeout boundary -- 2.9s must succeed on primary,
        # 3.1s must fail over. (3.0s exactly is a coin flip / scheduler noise,
        # deliberately not tested.)
        # ================================================================
        t_fast = f"boundary-fast-{uuid.uuid4()}"
        r_fast = complete(tenant_key=t_fast, prompt="hi", estimated_tokens=10, simulate_primary="sleep:2.9")
        check(
            r_fast.status_code == 200 and r_fast.json().get("_routed_via") == "primary",
            "a primary responding in 2.9s (under the 3.0s budget) is NOT failed over",
            f"status={r_fast.status_code} body={r_fast.json() if r_fast.status_code==200 else r_fast.text}",
        )

        t_slow = f"boundary-slow-{uuid.uuid4()}"
        r_slow = complete(tenant_key=t_slow, prompt="hi", estimated_tokens=10, simulate_primary="sleep:3.1")
        check(
            r_slow.status_code == 200 and r_slow.json().get("_routed_via") == "secondary",
            "a primary responding in 3.1s (just over the 3.0s budget) IS failed over",
            f"status={r_slow.status_code} body={r_slow.json() if r_slow.status_code==200 else r_slow.text}",
        )

        # ================================================================
        # Probe G: cross-tenant isolation under load -- tenant A hammered
        # with 100 concurrent near-budget requests must not meaningfully slow
        # down or incorrectly reject tenant B's independent, modest request
        # stream running at the same time.
        # ================================================================
        async def hammer_tenant_a():
            tenant_a = f"crossload-A-{uuid.uuid4()}"
            async with httpx.AsyncClient(timeout=15.0) as client:
                tasks = [
                    client.post(ROUTER_URL, json={"tenant_key": tenant_a, "prompt": "x", "estimated_tokens": 1000})
                    for _ in range(100)
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

        async def tenant_b_latency():
            tenant_b = f"crossload-B-{uuid.uuid4()}"
            latencies = []
            async with httpx.AsyncClient(timeout=15.0) as client:
                for _ in range(10):
                    start = time.monotonic()
                    await client.post(
                        ROUTER_URL, json={"tenant_key": tenant_b, "prompt": "x", "estimated_tokens": 1000}
                    )
                    latencies.append(time.monotonic() - start)
                    await asyncio.sleep(0.02)
            return latencies

        async def run_cross_load():
            b_latencies_baseline = await tenant_b_latency()  # baseline, no contention
            a_task = asyncio.create_task(hammer_tenant_a())
            await asyncio.sleep(0.05)  # let A's load ramp up
            b_latencies_loaded = await tenant_b_latency()
            await a_task
            return b_latencies_baseline, b_latencies_loaded

        baseline, loaded = asyncio.run(run_cross_load())
        avg_baseline = sum(baseline) / len(baseline)
        avg_loaded = sum(loaded) / len(loaded)
        check(
            all(l < 3.0 for l in loaded),
            "tenant B's requests all complete promptly (<3s) even while tenant A is under heavy concurrent load",
            f"avg_baseline={avg_baseline*1000:.1f}ms avg_loaded={avg_loaded*1000:.1f}ms max_loaded={max(loaded)*1000:.1f}ms",
        )

        # ================================================================
        # Probe H: time spent waiting for gateway capacity must not count
        # against the primary's 3000ms. 10 concurrent calls through 5
        # provider slots, primary answering in 2.0s: the second five wait
        # ~2s for a slot, then take 2s. If the deadline covered the wait
        # (~4s total) they would fail over; it must only cover the call.
        # ================================================================
        import router

        async def slot_probe() -> list[bool]:
            async with router.lifespan(router.app):
                client, slots = router.app.state.http_client, asyncio.Semaphore(5)

                async def call() -> bool:
                    try:
                        await router._call_provider(client, slots, "primary", f"{PROVIDER_URL}/primary/complete",
                                                    {"prompt": "x", "max_tokens": 1}, {"X-Simulate": "sleep:2.0"},
                                                    router.PRIMARY_TIMEOUT_SECONDS)
                        return True
                    except router.ProviderFailure:
                        return False

                return await asyncio.gather(*[call() for _ in range(10)])

        start = time.monotonic()
        served = asyncio.run(slot_probe())
        elapsed = time.monotonic() - start
        check(all(served) and elapsed > 3.5,
              "requests queued for a provider slot don't time out: the 3000ms deadline covers only the provider call",
              f"served={sum(served)}/10 elapsed={elapsed:.1f}s")

    finally:
        router_proc.terminate()
        providers_proc.terminate()
        try:
            router_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            router_proc.kill()
        try:
            providers_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            providers_proc.kill()
        # router.py's subprocess may still be releasing its file handle on
        # the persistent WAL connection right after terminate(); this
        # cleanup is best-effort, not load-bearing (main() unlinks it fresh
        # on the next run anyway).
        try:
            if DB_PATH.exists():
                DB_PATH.unlink()
        except OSError:
            pass

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All probes passed.")


if __name__ == "__main__":
    main()
