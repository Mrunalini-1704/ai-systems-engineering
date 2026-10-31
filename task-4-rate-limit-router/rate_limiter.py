"""Token-aware sliding-window rate limiter, backed by on-disk SQLite.

Model: reserve -> settle.
  1. reserve(tenant, estimate): before calling any provider, atomically
     evict this tenant's rows older than the 60s window, sum what's left,
     and either reject (RateLimitExceeded, with a computed Retry-After) or
     insert a reservation row for the estimated tokens.
  2. settle(reservation, actual): once the provider reports real usage,
     the reservation row is corrected to the actual token count -- unused
     budget is returned to the tenant, overage is charged.
  3. release(reservation): if no provider served the request, the
     reservation is deleted -- a failed request costs the tenant nothing.

Reserving up front (rather than recording only after the call) is what
makes this safe under concurrency: 100 concurrent requests can't all pass
the check before any of them is recorded.

Each row is (tenant_id, ts, tokens), so the window is exact: a row stops
counting exactly 60s after it was written (unlike a fixed-bucket counter,
which over-admits up to 2x across a bucket edge). Eviction is two-level:
every reserve() deletes the requesting tenant's expired rows, and
evict_expired() (run periodically by the router) deletes every tenant's
expired rows, so idle tenants don't leave rows behind forever.

Tenant API keys are stored as SHA-256 digests, never in plaintext: the
usage DB is an operational artifact (backups, debugging copies) and should
not double as a credential store.

Concurrency: one persistent WAL-mode connection, owned by one dedicated
DB thread (a single-worker executor). Every operation is queued to that
thread, so database access is serialized by construction, in FIFO order.
History:
  - An early version opened a connection per call and relied on SQLite's
    file lock (BEGIN IMMEDIATE). Correct, but under 100 concurrent requests
    for one tenant it pushed an *unrelated* tenant's latency from ~0.25s
    to 10+s via busy-wait/backoff on the file lock.
  - The next version used one connection behind an asyncio.Lock held
    across asyncio.to_thread(). Fast, but not cancellation-safe: if the
    awaiting request is cancelled (client disconnect, timeout), `async
    with` releases the lock while the worker thread is still mid-
    transaction, and the next caller can use the same connection
    concurrently. With a single owning thread there is no lock to release
    early -- a cancelled caller just stops waiting for its result.
"""

import asyncio
import hashlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).parent / "usage.db"
WINDOW_SECONDS = 60
TOKENS_PER_MINUTE = 50_000

_connection: sqlite3.Connection | None = None
# The one thread that ever touches _connection.
_db_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ratelimit-db")


async def _on_db_thread(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_db_thread, fn, *args)


class RateLimitExceeded(Exception):
    def __init__(self, used_tokens: int, requested: int, limit: int, retry_after: float | None):
        self.used_tokens = used_tokens
        self.requested = requested
        self.limit = limit
        # Seconds until enough of the window expires for this request to
        # fit; None if it can never fit (requested > limit).
        self.retry_after = retry_after
        super().__init__(f"would exceed {limit} tokens/min (used {used_tokens}, requested {requested})")


@dataclass(frozen=True)
class Reservation:
    row_id: int
    tenant_id: str
    tokens: int


def tenant_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is not None:
        return _connection

    # Created on, and only used from, the DB thread: the default
    # check_same_thread=True holds, and enforces that invariant.
    conn = sqlite3.connect(DB_PATH, timeout=5, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            ts REAL NOT NULL,
            tokens INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_tenant_ts ON token_usage(tenant_id, ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_token_usage_ts ON token_usage(ts)")
    _connection = conn
    return conn


def _retry_after(conn: sqlite3.Connection, tid: str, used: int, requested: int, now: float) -> float | None:
    if requested > TOKENS_PER_MINUTE:
        return None
    must_free = used + requested - TOKENS_PER_MINUTE
    freed = 0
    for ts, tokens in conn.execute(
        "SELECT ts, tokens FROM token_usage WHERE tenant_id = ? ORDER BY ts", (tid,)
    ):
        freed += tokens
        if freed >= must_free:
            return max(0.0, ts + WINDOW_SECONDS - now)
    return float(WINDOW_SECONDS)


def _reserve_sync(api_key: str, tokens: int) -> Reservation:
    tid = tenant_id(api_key)
    now = time.time()
    conn = _get_connection()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM token_usage WHERE tenant_id = ? AND ts <= ?", (tid, now - WINDOW_SECONDS))
        used = conn.execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant_id = ?", (tid,)
        ).fetchone()[0]
        if used + tokens > TOKENS_PER_MINUTE:
            retry_after = _retry_after(conn, tid, used, tokens, now)
            conn.execute("COMMIT")  # keep the eviction
            raise RateLimitExceeded(used, tokens, TOKENS_PER_MINUTE, retry_after)
        cur = conn.execute("INSERT INTO token_usage (tenant_id, ts, tokens) VALUES (?, ?, ?)", (tid, now, tokens))
        conn.execute("COMMIT")
        return Reservation(cur.lastrowid, tid, tokens)
    except RateLimitExceeded:
        raise
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _execute_sync(sql: str, params: tuple) -> int:
    return _get_connection().execute(sql, params).rowcount


def _release_if_reserved(future: asyncio.Future) -> None:
    if not future.cancelled() and future.exception() is None:
        _db_thread.submit(_execute_sync, "DELETE FROM token_usage WHERE id = ?", (future.result().row_id,))


async def reserve(api_key: str, tokens: int) -> Reservation:
    """Reserve `tokens` against the tenant's trailing-60s budget, or raise
    RateLimitExceeded without recording anything."""
    future = asyncio.get_running_loop().run_in_executor(_db_thread, _reserve_sync, api_key, tokens)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        # The caller gave up (e.g. client disconnect) but the DB thread may
        # still insert the reservation. Nobody would ever settle or release
        # it, so it would count against the tenant for the full window --
        # give it back as soon as it lands.
        future.add_done_callback(_release_if_reserved)
        raise


async def settle(reservation: Reservation, actual_tokens: int) -> None:
    """Correct a reservation to the provider-reported actual usage."""
    await _on_db_thread(
        _execute_sync, "UPDATE token_usage SET tokens = ? WHERE id = ?", (max(0, actual_tokens), reservation.row_id)
    )


async def release(reservation: Reservation) -> None:
    """Give a reservation back (the request was never served)."""
    await _on_db_thread(_execute_sync, "DELETE FROM token_usage WHERE id = ?", (reservation.row_id,))


async def evict_expired() -> int:
    """Delete every tenant's rows that have left the window. Returns rows deleted."""
    return await _on_db_thread(_execute_sync, "DELETE FROM token_usage WHERE ts <= ?", (time.time() - WINDOW_SECONDS,))


async def used_tokens(api_key: str) -> int:
    """Tokens currently counted against the tenant's window (for tests/metrics)."""
    def query() -> int:
        return _get_connection().execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant_id = ? AND ts > ?",
            (tenant_id(api_key), time.time() - WINDOW_SECONDS),
        ).fetchone()[0]

    return await _on_db_thread(query)


# Back-compat name: reserve-and-keep, as the original single-step API did.
check_and_record = reserve
