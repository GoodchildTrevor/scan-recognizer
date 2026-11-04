"""Token probabilities: raw storage, uncertain spans and statistics (section 6.6)."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

from ocr_archive.models import Span, TokenProbs, TokenStats
from ocr_archive.settings import MAX_TOKEN_P, ChecksConfig

logger = logging.getLogger(__name__)

LETTER = "letter"
DIGIT = "digit"
OTHER = "other"


def token_kind(token: str) -> str:
    """Classify a token: any digit → digit; a letter without digits → letter; otherwise other."""
    if any(ch.isdigit() for ch in token):
        return DIGIT
    if any(ch.isalpha() for ch in token):
        return LETTER
    return OTHER


def parse_logprobs(content: str, entries: Sequence[dict[str, Any]] | None) -> TokenProbs | None:
    """Build raw token probabilities from ``choices[0].logprobs.content``.

    Offsets are accumulated token lengths. When the concatenated tokens do not equal the
    message text, nothing is stored.

    :param content: ``message.content``.
    :param entries: logprob entries with ``token`` and ``logprob``.
    :return: probabilities, or None when unavailable or inconsistent.
    """
    if not entries:
        return None
    offset = 0
    letters = digits = 0
    tokens: list[tuple[int, int, float, str]] = []
    pieces: list[str] = []
    for entry in entries:
        token = entry.get("token")
        logprob = entry.get("logprob")
        if not isinstance(token, str) or not isinstance(logprob, int | float):
            logger.warning("logprob entry without token or logprob, probabilities dropped")
            return None
        pieces.append(token)
        start, end = offset, offset + len(token)
        offset = end
        kind = token_kind(token)
        if kind == LETTER:
            letters += 1
        elif kind == DIGIT:
            digits += 1
        else:
            continue
        p = math.exp(min(0.0, float(logprob)))
        if p < MAX_TOKEN_P:
            tokens.append((start, end, round(p, 6), kind))
    if "".join(pieces) != content:
        logger.warning(
            "token concatenation does not match the text (%d vs %d chars), probabilities dropped",
            offset,
            len(content),
        )
        return None
    return TokenProbs(letter_tokens=letters, digit_tokens=digits, tokens=tokens)


def _is_low(kind: str, p: float, cfg: ChecksConfig) -> bool:
    if kind == DIGIT:
        return p < cfg.digit_token_p
    if kind == LETTER:
        return p < cfg.low_conf_token_p
    return False


def low_confidence_spans(probs: TokenProbs, cfg: ChecksConfig) -> list[Span]:
    """Merge consecutive uncertain tokens into spans ``(start, end, min_p)``.

    Tokens are consecutive when no other stored letter/digit token lies between them and the
    text between them is only whitespace-free glue, i.e. their offsets touch.
    """
    spans: list[Span] = []
    for start, end, p, kind in probs.tokens:
        if not _is_low(kind, p, cfg):
            continue
        if spans and spans[-1].end == start:
            last = spans[-1]
            spans[-1] = Span(last.start, end, min(last.min_p, p))
        else:
            spans.append(Span(start, end, p))
    return spans


def token_stats(probs: TokenProbs, cfg: ChecksConfig) -> TokenStats:
    """Statistics of section 6.6 for the configured thresholds."""
    letter_ps = [p for _, _, p, kind in probs.tokens if kind == LETTER]
    digit_ps = [p for _, _, p, kind in probs.tokens if kind == DIGIT]
    return TokenStats(
        letter_min_p=min(letter_ps, default=1.0),
        letter_low=sum(1 for p in letter_ps if p < cfg.low_conf_token_p),
        digit_min_p=min(digit_ps, default=1.0),
        digit_low=sum(1 for p in digit_ps if p < cfg.digit_token_p),
        digit_tokens=probs.digit_tokens,
    )


def merge_token_probs(parts: Sequence[tuple[TokenProbs | None, int]]) -> TokenProbs | None:
    """Join probabilities of split parts; offsets of each part are shifted by its text offset.

    :param parts: (probabilities, offset of the part text in the combined text) per part.
    :return: None if any part has no probabilities.
    """
    letters = digits = 0
    tokens: list[tuple[int, int, float, str]] = []
    for probs, shift in parts:
        if probs is None:
            return None
        letters += probs.letter_tokens
        digits += probs.digit_tokens
        tokens.extend((s + shift, e + shift, p, k) for s, e, p, k in probs.tokens)
    return TokenProbs(letter_tokens=letters, digit_tokens=digits, tokens=tokens)
