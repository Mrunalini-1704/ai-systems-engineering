# AI Field Developer / Solutions Engineer Assessment — Reference Implementations

Four runnable, independently testable implementations: an MCP server, an
MCP security gateway, an LLM streaming PII guardrail, and a rate-limiting /
fallback router. All Python. Each task folder has its own `requirements.txt`
and `test_*.py` suites that start the real processes and drive them over
real stdio/HTTP (no mocked internals).

Tested with Python 3.14 (3.11+ required: Task 4 uses `asyncio.timeout`).

```
pip install -r task-1-mcp-server/requirements.txt -r task-2-mcp-gateway/requirements.txt \
            -r task-3-llm-guardrail/requirements.txt -r task-4-rate-limit-router/requirements.txt
python run_all_tests.py      # all 13 suites, ~2-3 minutes; suites use fixed local ports, so run one at a time
```

| Task | Ports | Suites |
|---|---|---|
| 1 MCP server | stdio | `test_stdio_isolation.py`, `test_adversarial.py`, `test_hardening.py` |
| 2 MCP gateway | 8100 downstream, 8200 gateway | `test_gateway.py`, `test_gateway_hardening.py` |
| 3 LLM guardrail | 8301 provider, 8300 gateway | `test_redaction_unit.py`, `test_redaction.py`, `test_gateway_sse.py` |
| 4 Router | 8500/8501 providers, 8600 router | `test_router.py`, `test_concurrency.py`, `test_cross_tenant_latency.py`, `test_extended_probes.py`, `test_hardening.py` |

---

## Task 1 — `task-1-mcp-server/`

An MCP server over stdio exposing `get_customer_record` and
`trigger_refund`, built on the official `mcp` SDK's low-level `Server` API.

**Validation (Pydantic, `extra="forbid"`):**
- `customer_id`: `fullmatch(r"CUST-[0-9]{5}")`. Not `^...\d$` + `match()`:
  `$` matches before a trailing newline (`"CUST-00001\n"` would pass) and
  `\d` matches any Unicode digit (Arabic-Indic `٠٠٠٠١` would pass).
- `amount`: strict float (no `"10"` strings or `true` booleans coerced),
  finite (`allow_inf_nan=False`), and at least 0.01: refunds are paid in
  whole cents, so `0.001` would otherwise "succeed" as a refund of 0.00.
- `reason`: at least 10 characters *after* trimming, so 12 spaces is rejected.

**Protocol errors vs. tool errors:**
- Invalid arguments and unknown tools → top-level JSON-RPC error
  `-32602 INVALID_PARAMS` (the MCP spec's classification for both).
- Well-formed call, but the customer doesn't exist → a tool *execution*
  error (`result.isError: true`) the model can see and react to.
- The handler is registered directly on `server.request_handlers[CallToolRequest]`
  instead of via `@server.call_tool()`, because the decorator turns every
  exception (including `McpError`) into an `isError` result, so a genuine
  JSON-RPC error could never be returned.
- Each stdin line is pre-checked before the SDK sees it. On its own, the
  SDK's stdio transport answers a malformed line with only a log
  notification, and an unknown method with `-32602`. The server instead
  returns `-32700` for invalid JSON, `-32600` for valid JSON that isn't a
  JSON-RPC message (a batch array, `"jsonrpc": "1.0"`), and `-32601` for an
  unknown method. Params are checked against the method's own SDK model,
  so `-32602` names the actual problem (`params.name: Field required`). The
  SDK alone sends a generic "Invalid request parameters" and logs about
  6 KB to stderr for each bad request.
- When stdin closes, the server first answers the requests still in flight
  (bounded wait), so `cat requests.jsonl | python server.py` gets every
  response. The SDK alone drops the last one.

**stdout isolation, at the file-descriptor level:** logging goes to stderr,
but that only covers code that uses `logging`. At startup the server `dup`s
fd 1 to a private fd that only the JSON-RPC transport writes to, then points
fd 1 and `sys.stdout` at stderr. A stray `print()`, a C extension writing to
fd 1, or a child process inheriting stdout all land on stderr and can't
corrupt the protocol stream. `test_hardening.py` proves all three. Messages
are written `\n`-delimited (no Windows `\r\n` translation).

`mcp` is pinned to `1.30.0`: the `2.x` line changed exception names and
error-propagation semantics. Don't bump the pin without re-running the suites.

Run directly with `python server.py` (stdio). Tests: 12 + 18 + 19 checks.

## Task 2 — `task-2-mcp-gateway/`

A reverse proxy between an agent and a downstream MCP server (JSON-RPC 2.0
over HTTP). Demo tokens: `admin-secret-123` → admin, `viewer-secret-456` → viewer.

- **Authentication:** missing or unknown bearer token → HTTP 401 +
  `WWW-Authenticate` + JSON-RPC `-32001`, before anything is parsed or
  forwarded. Tokens are compared in constant time.
- **Envelope validation:** bodies over 1 MiB → HTTP 413. Invalid JSON →
  `-32700`. That includes what Python's parser would otherwise accept or
  choke on: `NaN`/`Infinity`, numbers that overflow to infinity (`1e999`),
  and pathologically deep nesting. Batch arrays, non-object bodies, missing
  `jsonrpc:"2.0"`, a non-string `method`, or an object `id` → `-32600`.
  `tools/call` without a non-empty string `params.name` → `-32602`.
- **Tool filtering:** `tools/list` requests are forwarded transparently.
  For a non-admin, the gateway removes the `admin_` tools from the
  *response*, JSON or SSE alike, so an agent never plans around tools it
  can't call. This includes a response replayed on a resumed `GET` stream
  (`Last-Event-ID`). The `tools/call` check below is still the actual enforcement.
- **Authorization:** other methods are forwarded transparently. So are
  JSON-RPC *responses* from the client, which answer the server's own
  requests (sampling, elicitation). A `tools/call` whose
  name starts with `admin_` (checked case-insensitively and after trimming)
  from a non-admin → `-32001 Unauthorized Tool Call`, returned by the
  gateway. When the blocked call is a notification (no `id`), that is HTTP
  403 rather than a 200, because notifications get no JSON-RPC reply. The
  downstream mock records every request it receives, and the tests use that
  record to prove blocked calls never reach it.
- **Parser-differential defense:** the gateway forwards its re-serialization
  of the parsed body, not the raw bytes. Otherwise a body like
  `{"name":"get_weather","name":"admin_reset_key"}` could be authorized as
  one tool and executed as the other by a downstream parser with different
  duplicate-key rules.
- **Forwarding:** one pooled `httpx` client. Responses are streamed through
  (JSON and SSE alike) with `Mcp-Session-Id` preserved, and `GET`/`DELETE`
  are proxied for Streamable HTTP sessions. The long-lived `GET` SSE stream
  has no idle timeout. The client's bearer token is not forwarded
  downstream. The downstream is asked for an uncompressed body, since the
  bytes are relayed as-is. A downstream that is unreachable, times out, or
  returns 5xx → sanitized `-32003`, with no relayed stack trace.

Tests: 9 + 41 checks, including 120 concurrent mixed-role calls. The gateway
was also exercised end to end with the official SDK client in front of a
real SDK-built MCP server.

## Task 3 — `task-3-llm-guardrail/`

An OpenAI-compatible gateway (`POST /v1/chat/completions`, plus a plain-text
`POST /v1/chat/stream`) in front of a real HTTP SSE provider
(`mock_provider.py`, no API key needed; to use a real one, set `UPSTREAM_URL`
and `UPSTREAM_API_KEY`, which is sent as `Authorization: Bearer`).
The gateway parses upstream SSE events incrementally and re-emits the same
chunk format. Every streamed text field of each choice's delta (`content`,
and also `refusal` or `reasoning_content`) runs through its own
`StreamRedactor`. So does each tool call's `function.arguments`, keyed by
tool-call index, so PII split across argument fragments is caught. Events
that carry no choices are redacted whole. With `"stream": false`, the
redacted tool calls are assembled into `message.tool_calls`.

What becomes `[REDACTED]`:
- **Emails:** ASCII and Latin accented letters (`josé@exämple.com`).
- **SSNs:** `123-45-6789` and `123 45 6789`.
- **Luhn-valid card numbers:**
  - 13–19 digits unseparated
  - grouped 4-4-4-4, 4-6-5, 4-4-4-3 or 4-4-4-4-3, with space, dash or dot
    separators
  - full-width digits as well

Digit patterns are bounded by "not a digit" rather than `\b`, so
`card4111111111111111` is still caught. Known limits, chosen for precision:
- An unseparated 9-digit run isn't treated as an SSN; too many IDs look
  like one.
- Emails in non-Latin scripts aren't matched. Allowing them would hold back
  unspaced CJK text and delay it.

**Correctness contract (property-tested):** for any text and any way of
chunking it, the streamed output equals whole-text redaction. That means
no leaks and no over-redaction. `test_redaction_unit.py` checks this over
3,000 random texts × random chunkings, plus every fixed chunk size on a
PII-dense text. The property test found one real bug during development:
`finditer` skipped a valid card number that overlapped a Luhn-invalid
candidate. The fix and a regression check are in.

**How the carry boundary works (`redaction.py`):**
- Digit patterns (SSN, cards ≤ 23 chars) → hold back the last 24 chars.
- Emails are variable-length, so an anchored regex holds back only the
  email-shaped run still growing at the very end of the stream. The email
  regex uses RFC 5321 bounds (local part ≤ 64), which bounds that run. With
  an unbounded `+`, no hold-back is ever sufficient and memory is unbounded.
- A complete match straddling the boundary pulls it back, and one character
  of left context keeps the digit-boundary lookbehind correct across
  flushes.
- Memory is O(1) in response length. The patterns themselves bound the
  carry: a 100k-character token with no whitespace holds back 64
  characters, and back-to-back card numbers hold back 82. A hard cap of
  `MAX_CARRY = 1024` is kept as a safety net; no input found so far
  reaches it.

**Latency:** time to first token through the gateway matches the provider's
own (about 60 ms vs 60 ms in the test). With no artificial delay, 5,000 deltas
stream in about 0.33 s through the gateway versus 0.12 s direct, which is
roughly 40 µs of added overhead per delta. SSE responses send
`X-Accel-Buffering: no`. When a client disconnects, the upstream stream is closed.

**Failure handling:** a provider HTTP error → sanitized 502. Three
mid-stream failures → one error event and `[DONE]`:
- a provider crash
- a provider `{"error": ...}` event, whose message is unredacted and may
  carry internals
- a stream that closes without `[DONE]`

In all three, the unflushed tail is *dropped*, not flushed, because it may
hold half an email (`john.doe@exa`) that never completed into a
recognizable pattern.

Malformed upstream chunks (a non-object `delta`, a non-integer `index`) are
skipped. Surrogate pairs split across deltas don't break the text
endpoints. Request bodies with `NaN`, `1e999` or lone-surrogate escapes →
400.

Validation errors on either endpoint use the gateway's own error shape.

Tests: 26 + 7 + 27 checks.

## Task 4 — `task-4-rate-limit-router/`

`POST /v1/complete` with body `{tenant_key, prompt, max_tokens?, estimated_tokens?}`.

**Token-aware sliding window (on-disk SQLite, `rate_limiter.py`):** reserve,
then settle.
1. The router estimates the request's cost server-side: prompt tokens
   (UTF-8 bytes / 4, so CJK isn't under-counted) + `max_tokens`
   (default 256). A client's `estimated_tokens` can raise the reservation
   but never lower it, and it never changes the `max_tokens` sent to the
   provider.
2. `reserve()` atomically evicts the tenant's rows older than 60 s, sums
   the rest, and either rejects (429 + `Retry-After`, computed from when
   enough of the window expires) or inserts a reservation row. Reserving
   *before* the call is what prevents concurrent over-admission
   (`test_concurrency.py`: 100 concurrent requests admit exactly 50,000).
3. `settle()` corrects the row to the provider-reported `usage.total_tokens`,
   refunding unused budget. `release()` deletes it if no provider served the
   request. If the caller is cancelled (e.g. the client disconnects) while
   its reservation is still being written, the reservation is released as
   soon as it lands, rather than charging the tenant for the full window.
4. A background sweep (every 30 s) evicts *all* tenants' expired rows, so
   idle tenants don't accumulate rows. API keys are stored as SHA-256
   digests, never in plaintext.

One WAL-mode connection is owned by a single DB thread, which serializes
all access in FIFO order. History: per-call connections with `BEGIN IMMEDIATE`
caused cross-tenant head-of-line blocking of 10+ s
(`test_cross_tenant_latency.py`). An `asyncio.Lock` held across
`to_thread` wasn't cancellation-safe, because a cancelled caller released
the lock while its thread was still mid-transaction.

**Failover:** the primary gets a *total* 3000 ms deadline
(`asyncio.timeout` around connect + send + full body). A plain `httpx`
`timeout=3.0` is per-read, so a primary that trickles a byte a second never
trips it. `test_hardening.py` shows that control case taking 6.3 s, while
the router fails over at about 3.2 s. Any of these from the primary →
secondary (5000 ms deadline):
- a 429
- a timeout
- a 5xx
- a transport error
- a 200 that isn't a usable completion: non-JSON, `NaN`, an
  `{"error": ...}` body, or no `completion`

If both fail → 502.

Each provider has 100 call slots, and the connection pool has a connection
for every slot. A request takes its slot *before* its deadline starts, so
under a burst, time spent waiting for capacity inside the gateway doesn't
count as the primary timing out. With 10 calls through 5 slots against a
2.0 s primary, the old ordering failed over half of them; the probe in
`test_extended_probes.py` shows all 10 served. The cap stays at 100 because
past about 100 concurrent keep-alive connections, httpx/httpcore was
measured to start requests in a slow trickle on Windows. A larger pool made
a healthy primary look slow. If recording the provider's
usage fails after a request was served, the response is still returned and
the tenant stays charged the reserved estimate.

**Error sanitization:** every error, including validation errors, 404s,
and unhandled exceptions, is exactly `{"error": {"code", "message"}}`.
Upstream bodies, exception text, and validation details are logged
server-side under a request ID that is returned to the caller as
`X-Request-ID`.

Mock knobs (test-only): `simulate_primary` / `simulate_secondary` →
`X-Simulate: 429 | timeout | trickle | fail | error200 | nan | usage:<N>`
(`providers_mock.py`), or `sleep:<s> | status:<code>` (`probes_provider.py`).
By default the mocks report usage of the prompt plus the full `max_tokens`
budget, the worst case.

`usage.db` is created on first run. The test suites delete it for a clean slate.
