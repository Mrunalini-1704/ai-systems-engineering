"""Concurrency stress test for the rate limiter itself.

Talks to rate_limiter.py directly (no HTTP layer) to isolate the limiter's
own concurrency correctness. Fires 100 concurrent token-check requests for
a single tenant, each for 1,000 tokens (100,000 tokens requested total
against a 50,000/min budget). If check-then-record had a TOCTOU race, we'd
see more than 50 admitted (over-admission past budget). If locking dropped
or double-counted requests, we'd see a count that doesn't land exactly at
the budget boundary.

Run with: python test_concurrency.py
"""

import asyncio
import sys
import uuid

from rate_limiter import DB_PATH, TOKENS_PER_MINUTE, RateLimitExceeded, check_and_record

TOKENS_EACH = 1_000
N_REQUESTS = 100  # 100,000 tokens requested against a 50,000 budget

failures = []


def check(condition: bool, description: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    if not condition:
        failures.append(description)


async def attempt(tenant: str) -> bool:
    try:
        await check_and_record(tenant, TOKENS_EACH)
        return True
    except RateLimitExceeded:
        return False


async def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()

    tenant = f"stress-{uuid.uuid4()}"
    results = await asyncio.gather(*[attempt(tenant) for _ in range(N_REQUESTS)])
    admitted = sum(results)
    total_tokens_admitted = admitted * TOKENS_EACH

    check(
        total_tokens_admitted <= TOKENS_PER_MINUTE,
        f"never over-admits past the {TOKENS_PER_MINUTE}-token budget under concurrency "
        f"(admitted {total_tokens_admitted})",
    )
    check(
        total_tokens_admitted > TOKENS_PER_MINUTE - TOKENS_EACH,
        f"fully utilizes the budget rather than dropping valid requests (admitted {total_tokens_admitted}, "
        f"expected > {TOKENS_PER_MINUTE - TOKENS_EACH})",
    )
    check(
        admitted * TOKENS_EACH == TOKENS_PER_MINUTE,
        f"admits exactly up to the budget boundary with no off-by-one under concurrent load "
        f"(admitted exactly {admitted * TOKENS_EACH} of {TOKENS_PER_MINUTE})",
    )

    # rate_limiter now keeps one persistent connection open for the process
    # lifetime (see rate_limiter.py docstring); on Windows that keeps the
    # file locked, so a post-test cleanup unlink would fail here. The
    # pre-test cleanup above is what matters for test isolation.
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
        print("All checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
