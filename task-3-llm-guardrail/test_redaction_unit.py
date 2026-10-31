"""Unit-level and property-based tests for StreamRedactor (no HTTP layer).

The central check is a randomized property test of the redactor's contract:
for any text and any chunking, streamed output == whole-text redact(text).
Plus targeted regressions found during review:
  - A long (64-char local-part) email streamed in tiny chunks must not leak
    its first characters before the "@"/domain arrive (an earlier version
    used a fixed hold-back, which no unbounded email regex can satisfy).
  - A token that never terminates must still be flushed in full at the end.
  - Memory is hard-bounded: neither ordinary prose nor adversarial input
    (one giant whitespace-free token, back-to-back card numbers) can grow
    the carry buffer beyond MAX_CARRY.
  - "\\b" context across a flush boundary ("x" + "4111 1111 1111 1111").

Run with: python test_redaction_unit.py
"""

import random
import sys
import time

from mock_provider import FULL_TEXT_CHUNKS
from redaction import MAX_CARRY, REDACTED, StreamRedactor, redact

failures = []


def check(condition: bool, description: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}" + (f"  ({detail})" if detail and not condition else ""))
    if not condition:
        failures.append(description)


def stream(chunks: list[str], track: list[int] | None = None) -> str:
    r = StreamRedactor()
    out = []
    for c in chunks:
        out.append(r.feed(c))
        if track is not None:
            track.append(r.carry_len)
    out.append(r.flush())
    return "".join(out)


def split_every(text: str, n: int) -> list[str]:
    return [text[i : i + n] for i in range(0, len(text), n)]


FRAGMENTS = [
    "john.doe@example.com", "a@b.co", "x" * 70 + "@long-domain.example.org", "user+tag@sub.domain.io",
    "123-45-6789", "123-45-67890", "4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111",
    "4111 1111 1111 1112", "3782 822463 10005", "378282246310005", "1234 5678 9012 3456 7",
    "hello", "world", "the", "order", "#12345", "foo@", "@bar.com", "trailing.", "-", ".", "@", "1", "12",
    "1234", "user@", "@", "x", "a.b.c", "555-12-3456", "9" * 30, "z" * 90,
    "4111.1111.1111.1111", "3782 8224 6310 005", "4222222222222", "6011 0009 9013 9424 3", "123 45 6789",
    "card4111111111111111", "josé@exämple.com", "５５５５" * 3 + "４４４４",
]
SEPARATORS = [" ", " ", " ", "\n", ", ", ".", "", "-", "@", "(", ")"]


def random_text(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(1, 40)):
        parts.append(rng.choice(FRAGMENTS))
        parts.append(rng.choice(SEPARATORS))
    return "".join(parts)


def random_chunks(rng: random.Random, text: str) -> list[str]:
    chunks, i = [], 0
    while i < len(text):
        n = rng.choice([1, 1, 2, 3, 4, 5, 7, 11, 16, 40, 200])
        chunks.append(text[i : i + n])
        i += n
    return chunks


def main() -> None:
    # --- Property test: streaming == whole-text redaction, for any chunking ---
    rng = random.Random(1234)
    mismatches = []
    trials = 3000
    for _ in range(trials):
        text = random_text(rng)
        chunks = random_chunks(rng, text)
        got, want = stream(chunks), redact(text)
        if got != want:
            mismatches.append((text, chunks, got, want))
    check(not mismatches, f"property: streamed output == redact(full text) for {trials} random texts x random chunkings",
          repr(mismatches[0]) if mismatches else "")

    # Every chunking of one PII-dense text, including 1-char-at-a-time.
    dense = "Mail john.doe@example.com, SSN 123-45-6789, card 4111 1111 1111 1111; amex 378282246310005."
    bad = [n for n in range(1, len(dense) + 1) if stream(split_every(dense, n)) != redact(dense)]
    check(not bad, "every fixed chunk size 1..len gives identical output on a PII-dense text", str(bad))
    check(redact(dense).count(REDACTED) == 4 and "@" not in redact(dense),
          "PII-dense text: email, SSN, Visa and Amex all redacted", redact(dense))

    # --- Targeted regressions ---
    local, domain = "a" * 64, "b" * 60 + ".com"
    text = f"contact {local}@{domain} now"
    for n in (1, 3):
        result = stream(split_every(text, n))
        check(result == f"contact {REDACTED} now",
              f"64-char local-part email streamed {n} char(s) at a time is fully redacted, no prefix leak", result[:80])

    never_ending = "x " + "a" * 200
    check(stream(split_every(never_ending, 5)) == never_ending,
          "a non-PII token that never hits a word boundary is flushed in full at end-of-stream")
    digits = "id " + "1" * 500
    check(stream(split_every(digits, 7)) == digits, "a non-PII digit run that never completes a pattern is flushed in full")

    check(stream(["1", "4111 1111 1111 1111"]) == "14111 1111 1111 1111" == redact("14111 1111 1111 1111"),
          "digit context survives a flush boundary ('1' + card digits is a longer number, not a card)")
    check(redact("order 4111 1111 1111 1112 shipped") == "order 4111 1111 1111 1112 shipped",
          "Luhn-invalid 16-digit number (e.g. an order id) is not redacted")
    # Found by the property test: a Luhn-invalid candidate overlapping a
    # valid card must not hide it (finditer resumes after the bogus match).
    check(redact("1234-4111111111111111)") == f"1234-{REDACTED})",
          "a valid card overlapping a Luhn-invalid candidate is still redacted")
    check(redact("1123-45-6789") == "1123-45-6789" and redact("123-45-67890") == "123-45-67890"
          and redact("ssn 123-45-6789") == f"ssn {REDACTED}",
          "SSN pattern doesn't match inside a longer number")

    # Formats beyond the basic ones: each must be redacted on its own.
    for pii in ["4111.1111.1111.1111", "4111 1111-1111 1111", "3782 8224 6310 005", "378282246310005",
                "4222222222222", "123 45 6789", "josé@exämple.com", "５５５５" * 3 + "４４４４"]:
        # ascii(): a piped Windows console is cp1252 and can't print full-width digits.
        check(redact(f"x {pii} y") == f"x {REDACTED} y", f"redacted: {ascii(pii)}", ascii(redact(f"x {pii} y")))
    check(redact("card4111111111111111 ssn123-45-6789") == f"card{REDACTED} ssn{REDACTED}",
          "digits glued to a word are still redacted", redact("card4111111111111111 ssn123-45-6789"))
    check(redact("ts 1700000000001 ok") == "ts 1700000000001 ok", "a Luhn-invalid 13-digit number is not redacted")

    # --- Memory bounds ---
    sizes: list[int] = []
    stream([f"word{i} " for i in range(10_000)], sizes)
    check(max(sizes) <= 30, f"ordinary prose: carry stays tiny across 10,000 chunks (max {max(sizes)} chars)")

    sizes = []
    stream(["A" * 50] * 2_000, sizes)  # 100k chars, no whitespace ever
    check(max(sizes) <= MAX_CARRY, f"one 100k-char whitespace-free token: carry hard-bounded (max {max(sizes)} <= {MAX_CARRY})")

    sizes = []
    adversarial = ".".join(["4111-1111-1111-1111"] * 3000)
    out = stream(split_every(adversarial, 13), sizes)
    check(max(sizes) <= MAX_CARRY and "4111-1111-1111-1111" not in out,
          f"back-to-back card numbers with no whitespace: carry bounded (max {max(sizes)}) and nothing leaks")

    # --- Throughput / per-chunk latency ---
    prose = "The quick brown fox emailed jane@example.org about card 4111 1111 1111 1111. " * 2000
    chunks = split_every(prose, 4)
    t0 = time.perf_counter()
    stream(chunks)
    elapsed = time.perf_counter() - t0
    per_chunk_us = elapsed / len(chunks) * 1e6
    check(per_chunk_us < 200, f"redaction overhead is small: {per_chunk_us:.1f}us per 4-char delta over {len(chunks)} deltas")

    # --- The mock provider's own script ---
    full_mock = stream(FULL_TEXT_CHUNKS)
    check(full_mock == redact("".join(FULL_TEXT_CHUNKS)) and full_mock.count(REDACTED) == 4,
          f"mock provider script: all 4 PII instances redacted (found {full_mock.count(REDACTED)})")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
