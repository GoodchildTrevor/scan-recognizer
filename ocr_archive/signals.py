"""Page signals: counters that explain why a page may contain an error (section 6.10).

They are not probabilities of an error and are never calibrated.
"""

from __future__ import annotations

from ocr_archive.checks import line_gap
from ocr_archive.models import LineDetection, PassComparison, VlmResult
from ocr_archive.settings import ChecksConfig
from ocr_archive.vlm.logprobs import token_stats


def page_signals(
    candidate: VlmResult,
    comparison: PassComparison | None,
    lines: LineDetection,
    *,
    checks: ChecksConfig,
) -> dict[str, float | bool]:
    """Compute the signal counters of a candidate.

    :param candidate: chosen attempt.
    :param comparison: comparison of the two passes, None when only one pass works.
    :param lines: line detection of the page.
    :param checks: thresholds of uncertain tokens.
    """
    signals: dict[str, float | bool] = {"layout_unreliable": not lines.reliable}
    if comparison is not None:
        signals["diff_words"] = comparison.diff_words
        signals["digit_mismatch"] = comparison.digit_mismatch
        signals["similarity"] = round(comparison.similarity, 4)
    if candidate.token_probs is None:
        signals["no_logprobs"] = True
    else:
        stats = token_stats(candidate.token_probs, checks)
        signals["low_digits"] = stats.digit_low
        signals["low_letters"] = stats.letter_low
        signals["min_digit_p"] = round(stats.digit_min_p, 4)
        signals["min_letter_p"] = round(stats.letter_min_p, 4)
    gap = line_gap(candidate.text, lines)
    if gap is not None:
        signals["line_gap"] = gap
    return signals
