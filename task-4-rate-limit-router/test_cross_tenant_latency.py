"""Cross-tenant latency regression test.

An earlier implementation opened a fresh SQLite connection per
check-and-record call and relied on BEGIN IMMEDIATE for serialization.
Under review, 100 concurrent requests for tenant A pushed a completely
unrelated tenant B's request latency from ~0.25s baseline up to 10+
seconds -- correct admit/reject decisions throughout, but a severe
head-of-line blocking problem for an LLM gateway's hot path.

This test reproduces that exact scenario against rate_limiter.py directly
(no HTTP layer) and asserts tenant B's latency stays bounded even while
tenant A hammers the limiter concurrently.

Run with: python test_cross_tenant_latency.py
"""

import asyncio
import sys
import time
import uuid

from rate_limiter import DB_PATH, check_and_record

# Generous but meaningful: previously observed up to 10.4s under this exact
# load; a healthy implementation should stay within a couple hundred ms.
MAX_ACCEPTABLE_TENANT_B_LATENCY = 2.0

failures = []


def check(condition: bool, description: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    if not condition:
        failures.append(description)


async def hammer(tenant: str, n: int) -> None:
    async def one():
        try:
            await check_and_record(tenant, 100)
        except Exception:
            pass

    await asyncio.gather(*(one() for _ in range(n)))


async def main() -> None:
    if DB_PATH.exists():
        try:
            DB_PATH.unlink()
        except OSError:
            pass

    tenant_a = f"hammer-{uuid.uuid4()}"
    tenant_b = f"quiet-{uuid.uuid4()}"

    # baseline latency for tenant B with no contention
    start = time.monotonic()
    await check_and_record(tenant_b, 100)
    baseline = time.monotonic() - start
    print(f"    tenant B baseline latency: {baseline:.3f}s")

    # fire tenant A's hammer concurrently with several tenant B requests
    async def tenant_b_requests():
        latencies = []
        for _ in range(5):
            start = time.monotonic()
            await check_and_record(tenant_b, 100)
            latencies.append(time.monotonic() - start)
            await asyncio.sleep(0.05)
        return latencies

    _, b_latencies = await asyncio.gather(hammer(tenant_a, 100), tenant_b_requests())

    max_b_latency = max(b_latencies)
    avg_b_latency = sum(b_latencies) / len(b_latencies)
    print(f"    tenant B latencies while tenant A hammers 100 concurrent requests: {[f'{x:.3f}' for x in b_latencies]}")
    print(f"    max={max_b_latency:.3f}s avg={avg_b_latency:.3f}s")

    check(
        max_b_latency < MAX_ACCEPTABLE_TENANT_B_LATENCY,
        f"tenant B's worst-case latency ({max_b_latency:.3f}s) stays well under "
        f"{MAX_ACCEPTABLE_TENANT_B_LATENCY}s even while an unrelated tenant hammers the limiter "
        f"(previously observed up to ~10.4s here)",
    )

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
