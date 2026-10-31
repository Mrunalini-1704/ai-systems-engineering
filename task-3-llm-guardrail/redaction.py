"""Streaming PII redaction.

Provider-agnostic: `StreamRedactor` consumes text deltas one at a time and
emits redacted text, without ever buffering the full response. Only a
small, hard-bounded "carry" tail is held in memory at any time.

Correctness contract (enforced by test_redaction_unit.py's randomized
property test): for ANY text and ANY way of splitting it into chunks,

    "".join(feed(c) for c in chunks) + flush()  ==  redact(text)

i.e. streaming never leaks something whole-text redaction would catch, and
never over-redacts something whole-text redaction would leave alone.

How the carry boundary ("cut") is chosen on every feed():

  1. Digit patterns (SSN, card numbers) are at most MAX_FIXED_MATCH_LEN
     chars. Any match that could still *complete* once
     more text arrives must start within the last MAX_FIXED_MATCH_LEN
     chars, so we never flush closer than HOLD_BACK to the end.

  2. Emails have variable length, so a fixed hold-back isn't enough. But a
     not-yet-complete email can only be a run of email characters touching
     the very end of the text received so far (local-part, optionally "@"
     and a partial domain). TAIL_EMAIL_RE finds exactly that run, anchored
     at the end, and the cut is pulled back to its start. Ordinary prose
     hits a space every few characters, so this usually costs ~1 word.
     The run is bounded because EMAIL_RE itself uses RFC 5321 limits
     (local-part <= 64, domain <= 253 + TLD) -- a "+" quantifier here
     would make the hold-back, and therefore memory, unbounded.

  3. A match that's already complete but straddles the cut pulls the cut
     back to its start, so a match is redacted (and flushed) as one unit.

Matches are computed on the carry *plus one character of already-flushed
left context*, so the "(?<!\\d)" lookbehind at the carry's start sees the
real preceding character -- without it, "14111 1111 1111 1111" streamed as
"1" + "4111..." would look like a card number that starts a number.

Card numbers are Luhn-checked, so arbitrary 15/16-digit IDs (order numbers,
timestamps) are not redacted -- precision matters for a guardrail that sits
in every response.
"""

import re
import unicodedata

REDACTED = "[REDACTED]"

# Letters allowed in emails: ASCII plus Latin-1 / Latin Extended-A/B (josé@,
# müller@), minus the U+00D7/U+00F7 math signs. Not every script: that would
# also hold back unspaced CJK text (see TAIL_EMAIL_RE) and delay it.
_LETTERS = "A-Za-zÀ-ÖØ-öø-ɏ"
# RFC 5321 bounds: local-part <= 64, domain <= 255 incl. TLD.
EMAIL_RE = re.compile(rf"[{_LETTERS}0-9._%+-]{{1,64}}@[{_LETTERS}0-9.-]{{1,252}}\.[{_LETTERS}]{{2,63}}")

# Digit patterns are bounded by "not a digit" rather than "\b", so a number
# glued to a word ("card4111111111111111", "ssn123-45-6789") is still caught.
# "\d" (any Unicode digit) is deliberate: full-width digits are PII too, and
# _luhn_ok() reads digit values with unicodedata.
SSN_RE = re.compile(r"(?<!\d)\d{3}([- ])\d{2}\1\d{4}(?!\d)")  # 123-45-6789, 123 45 6789


def _grouped(*sizes: int) -> re.Pattern:
    # Digit groups joined by an optional single space, dash or dot.
    return re.compile(r"(?<!\d)" + "[ .-]?".join(rf"\d{{{n}}}" for n in sizes) + r"(?!\d)")


# Every card pattern has one possible match per start position, which
# _spans() relies on. Candidates are Luhn-checked.
CARD_RES = [
    re.compile(r"(?<!\d)\d{13,19}(?!\d)"),  # unseparated, 13-19 digits
    _grouped(4, 4, 4, 4),  # Visa / MC / Discover
    _grouped(4, 6, 5),  # Amex
    _grouped(4, 4, 4, 3),  # Amex, 4-grouped
    _grouped(4, 4, 4, 4, 3),  # 19-digit
]

MAX_FIXED_MATCH_LEN = 23  # longest SSN/card match: 19 digits + 4 separators
HOLD_BACK = MAX_FIXED_MATCH_LEN + 1  # +1: the "(?!\d)" after a match needs the next char

# A still-growing potential email touching the end of the text: up to 64
# local-part chars, optionally "@" and a partial domain/TLD so far.
TAIL_EMAIL_RE = re.compile(rf"[{_LETTERS}0-9._%+-]{{1,64}}(?:@[{_LETTERS}0-9.-]{{0,316}})?\Z")
MAX_TAIL_EMAIL_LEN = 64 + 1 + 316

# Hard ceiling on the carry, whatever the input. Normal operation never
# gets near it (the longest legitimately held span is a max-length email,
# ~380 chars); it exists so adversarial input can't make the redactor hold
# an unbounded amount of text.
MAX_CARRY = 1024


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = unicodedata.digit(ch)
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _spans(text: str, start: int = 0) -> list[tuple[int, int]]:
    """Sorted, merged [start, end) spans of every PII match in text[start:]
    (text[:start] is left context only, for the "(?<!\\d)" lookbehinds)."""
    found = []
    for m in EMAIL_RE.finditer(text, start):
        found.append(m.span())
    for m in SSN_RE.finditer(text, start):
        found.append(m.span())
    # Not finditer(): it resumes *after* each match, so a Luhn-invalid
    # candidate would hide a valid card overlapping it (e.g. in
    # "1234 4111 1111 1111 1111", the bogus "1234 4111 1111 1111" swallows
    # the start of the real card). On a Luhn failure, resume one char later.
    for card_re in CARD_RES:
        pos = start
        while (m := card_re.search(text, pos)) is not None:
            if _luhn_ok(re.sub(r"[ .-]", "", m.group())):
                found.append(m.span())
                pos = m.end()
            else:
                pos = m.start() + 1
    found.sort()
    merged: list[tuple[int, int]] = []
    for s, e in found:
        if merged and s < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _apply(text: str, spans: list[tuple[int, int]], start: int, end: int) -> str:
    out, pos = [], start
    for s, e in spans:
        out.append(text[pos:s])
        out.append(REDACTED)
        pos = e
    out.append(text[pos:end])
    return "".join(out)


def redact(text: str) -> str:
    """Whole-text redaction -- the reference the streaming path must match."""
    return _apply(text, _spans(text), 0, len(text))


class StreamRedactor:
    def __init__(self) -> None:
        self._carry = ""
        self._left = ""  # last already-emitted char, for "\b" context

    def feed(self, chunk: str) -> str:
        """Feed a new delta in; returns the portion now safe to emit."""
        if not chunk:
            return ""
        text = self._left + self._carry + chunk
        base = len(self._left)
        spans = _spans(text, base)

        cut = max(base, len(text) - HOLD_BACK)
        tail = TAIL_EMAIL_RE.search(text, max(base, len(text) - MAX_TAIL_EMAIL_LEN))
        if tail is not None:
            cut = min(cut, tail.start())
        # Merged spans never overlap each other, so one pass is enough.
        for s, e in spans:
            if s < cut < e:
                cut = s
                break

        if len(text) - cut > MAX_CARRY:
            # Only reachable with adversarial input. Never split a redaction
            # across the forced cut: if a span straddles it, flush through
            # the span's end instead.
            cut = len(text) - MAX_CARRY
            for s, e in spans:
                if s < cut < e:
                    cut = e
                    break

        emitted = [(s, e) for s, e in spans if e <= cut]
        out = _apply(text, emitted, base, cut)
        self._carry = text[cut:]
        if cut > base:
            self._left = text[cut - 1]
        return out

    def flush(self) -> str:
        """Call at end-of-stream to emit whatever remains in the carry buffer."""
        text = self._left + self._carry
        base = len(self._left)
        out = _apply(text, _spans(text, base), base, len(text))
        self._carry = ""
        return out

    @property
    def carry_len(self) -> int:
        return len(self._carry)
