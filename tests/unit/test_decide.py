"""Page decision, signals, configuration of flags, versions (6.9–6.11)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from ocr_archive import versions
from ocr_archive.exceptions import ConfigError
from ocr_archive.models import (
    FlagCode,
    LineBox,
    LineDetection,
    PageAttempts,
    PageStatus,
    PageTrack,
    VlmResult,
)
from ocr_archive.pipeline import decide
from ocr_archive.settings import Settings, load_settings
from ocr_archive.signals import page_signals
from ocr_archive.vlm.logprobs import parse_logprobs

pytestmark = pytest.mark.p2

TEXT = "\n".join(f"Строка номер {i} текста страницы без ошибок" for i in range(1, 11))


def lines_of(n: int, reliable: bool = True) -> LineDetection:
    boxes = [LineBox(50, 50 + 20 * i, 500, 62 + 20 * i) for i in range(n)]
    return LineDetection(boxes=boxes, reliable=reliable, columns_detected=False, skew_deg=0.0)


def res(
    text: str, finish: str = "stop", low: dict[str, float] | None = None, logprobs: bool = True
) -> VlmResult:
    probs = None
    if logprobs:
        import re

        tokens = re.findall(r"\s+|\S+", text)
        entries = [{"token": t, "logprob": math.log((low or {}).get(t, 1.0))} for t in tokens]
        probs = parse_logprobs(text, entries)
    return VlmResult(text, finish, 1, 1, 1, probs)


def page(a1: VlmResult | None, a2: VlmResult | None) -> PageAttempts:
    return PageAttempts(
        page_id=1,
        page_index=0,
        track=PageTrack.VLM,
        first=a1,
        second=a2,
        first_attempt_id=11 if a1 else None,
        second_attempt_id=22 if a2 else None,
    )


S = Settings()


def test_clean_page_ok() -> None:
    d = decide(page(res(TEXT), res(TEXT)), lines_of(10), S, "rv")
    assert d.status == PageStatus.OK and not d.flags
    assert d.candidate_attempt_id == 22
    assert d.rules_version == "rv"


def test_both_defective_broken() -> None:
    d = decide(page(res(TEXT, "length"), res(TEXT, "length")), lines_of(10), S, "rv")
    assert d.flags == {FlagCode.BROKEN}
    assert d.candidate_attempt_id == 22


def test_one_defective_unverified() -> None:
    d = decide(page(res(TEXT), res(TEXT, "length")), lines_of(10), S, "rv")
    assert d.flags == {FlagCode.UNVERIFIED} and d.candidate_attempt_id == 11
    d = decide(page(res(TEXT, "length"), res(TEXT)), lines_of(10), S, "rv")
    assert d.flags == {FlagCode.UNVERIFIED} and d.candidate_attempt_id == 22


def test_digits_and_tables() -> None:
    d = decide(page(res(TEXT), res(TEXT.replace("номер 5", "номер 6"))), lines_of(10), S, "rv")
    assert FlagCode.DIGITS_DISAGREEMENT in d.flags
    t1 = TEXT + "\n| a | b |\n|---|---|\n| 1 | 2 |"
    t2 = TEXT + "\n| a | b | c |\n|---|---|---|\n| 1 | 2 | 3 |"
    d = decide(page(res(t1), res(t2)), lines_of(10), S, "rv")
    assert FlagCode.TABLE_MISMATCH in d.flags


def test_pass_diff_threshold_is_inclusive() -> None:
    other = TEXT.replace("текста", "тексто", 2)
    d = decide(page(res(TEXT), res(other)), lines_of(10), S, "rv")
    assert d.signals["diff_words"] == 4
    assert FlagCode.PASS_DIFF in d.flags
    one = TEXT.replace("текста", "тексто", 1)
    assert FlagCode.PASS_DIFF not in decide(page(res(TEXT), res(one)), lines_of(10), S, "rv").flags


def test_low_conf_digit_gives_flag() -> None:
    d = decide(page(res(TEXT), res(TEXT, low={"5": 0.8})), lines_of(10), S, "rv")
    assert FlagCode.LOW_CONF_DIGIT in d.flags
    assert d.signals["low_digits"] == 1


def test_low_conf_text_threshold() -> None:
    low = {w: 0.5 for w in ("Строка", "номер", "текста", "страницы", "без")}
    d = decide(page(res(TEXT), res(TEXT, low=low)), lines_of(10), S, "rv")
    assert FlagCode.LOW_CONF_TEXT in d.flags


def test_no_logprobs_no_flag() -> None:
    d = decide(page(res(TEXT, logprobs=False), res(TEXT, logprobs=False)), lines_of(10), S, "rv")
    assert d.status == PageStatus.OK
    assert d.signals["no_logprobs"] is True
    assert "low_digits" not in d.signals


def test_line_count_and_markers() -> None:
    d = decide(page(res(TEXT), res(TEXT)), lines_of(14), S, "rv")
    assert FlagCode.LINE_COUNT_MISMATCH in d.flags
    marked = TEXT + "\nконец [неразб.: слово?]"
    d = decide(page(res(marked), res(marked)), lines_of(11), S, "rv")
    assert FlagCode.MARKERS in d.flags
    read_note = TEXT + "\n[рук.: принято]"
    assert (
        decide(page(res(read_note), res(read_note)), lines_of(10), S, "rv").status == PageStatus.OK
    )


def test_switched_off_flag_is_suppressed() -> None:
    settings = Settings(decide={"flags": ["broken", "unverified", "markers"]})
    d = decide(
        page(res(TEXT), res(TEXT.replace("номер 5", "номер 6"))), lines_of(10), settings, "rv"
    )
    assert d.status == PageStatus.OK
    assert d.suppressed_flags == {FlagCode.DIGITS_DISAGREEMENT}


def test_mandatory_flags_cannot_be_removed(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("decide:\n  flags: [broken, markers]\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(config)


def test_threshold_above_099_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("checks:\n  digit_token_p: 0.995\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(config)


def test_page_signals_line_gap_only_when_reliable() -> None:
    sig = page_signals(res(TEXT), None, lines_of(12), checks=S.checks)
    assert sig["line_gap"] == 2 and sig["layout_unreliable"] is False
    sig = page_signals(res(TEXT), None, lines_of(12, reliable=False), checks=S.checks)
    assert "line_gap" not in sig and sig["layout_unreliable"] is True


def test_rules_version_stable_and_sensitive() -> None:
    a = versions.rules_version(Settings(signals={"min_diff_words": 4, "min_low_digits": 1}))
    b = versions.rules_version(Settings(signals={"min_low_digits": 1, "min_diff_words": 4}))
    assert a == b
    assert a != versions.rules_version(Settings(signals={"min_diff_words": 5}))
    assert a == versions.rules_version(Settings(review={"stale_draft_days": 7}))


def test_profile_id_sensitivity(monkeypatch: pytest.MonkeyPatch) -> None:
    base = versions.profile_id("pass2", S)
    weights = Settings(vlm={"weights": {"gguf_file": "other.gguf"}})
    assert versions.profile_id("pass2", weights) != base
    assert versions.profile_id("pass2", Settings(checks={"split_min_lines": 30})) != base
    assert versions.profile_id(
        "pass1", Settings(checks={"split_min_lines": 30})
    ) == versions.profile_id("pass1", S)
    monkeypatch.setattr(versions, "OCR_IMPL_VERSION", "999")
    assert versions.profile_id("pass2", S) != base


def test_fragment_prompt_changes_pass2(monkeypatch: pytest.MonkeyPatch) -> None:
    base2, base1 = versions.profile_id("pass2", S), versions.profile_id("pass1", S)
    original = versions.load_prompt

    def patched(name: str) -> str:
        return original(name) + (" изменено" if name == "fragment_v1" else "")

    monkeypatch.setattr(versions, "load_prompt", patched)
    assert versions.profile_id("pass2", S) != base2
    assert versions.profile_id("pass1", S) == base1
