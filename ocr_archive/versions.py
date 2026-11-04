"""OCR version (profile id) and rules version (section 6.11)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pymupdf

from ocr_archive.lines import lines_version
from ocr_archive.settings import Settings
from ocr_archive.vlm import FRAGMENT_PROMPT, load_prompt

#: Bump on any change of rendering or splitting code: it requires a new recognition.
OCR_IMPL_VERSION = "1"
#: Bump on any change of comparison, signal or hint code: it requires only re-deciding.
RULES_IMPL_VERSION = "1"

PDF_TEXT_PROFILE = "pdf_text"


def canonical_json(value: Any) -> str:
    """Canonical JSON used for all version hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def profile_config(name: str, settings: Settings) -> dict[str, Any]:
    """Return everything the OCR version of a profile depends on."""
    if name == PDF_TEXT_PROFILE:
        return {
            "name": name,
            "fast_track": settings.fast_track.model_dump(mode="json"),
            "pymupdf": pymupdf.VersionBind,
            "impl": OCR_IMPL_VERSION,
        }
    profile = settings.profiles[name]
    prompts = [profile.prompt]
    if name == "pass2":
        prompts.append(FRAGMENT_PROMPT)
    config: dict[str, Any] = {
        "name": name,
        "profile": profile.model_dump(mode="json"),
        "model": settings.vlm.model,
        "weights": settings.vlm.weights.model_dump(mode="json"),
        "prompts": {prompt: load_prompt(prompt) for prompt in sorted(set(prompts))},
        "impl": OCR_IMPL_VERSION,
    }
    if name == "pass2":
        config["split"] = {
            "split_min_lines": settings.checks.split_min_lines,
            "split_gap_max_ink": settings.checks.split_gap_max_ink,
            "lines": lines_version(),
        }
    return config


def profile_id(name: str, settings: Settings) -> str:
    """OCR version of a profile: sha256 of its canonical configuration."""
    return _sha256(profile_config(name, settings))


def rules_version(settings: Settings) -> str:
    """Rules version: sha256 of decide, signals, hints, checks, code and line detector versions."""
    return _sha256(
        {
            "decide": settings.decide.model_dump(mode="json"),
            "signals": settings.signals.model_dump(mode="json"),
            "hints": settings.hints.model_dump(mode="json"),
            "checks": settings.checks.model_dump(mode="json"),
            "impl": RULES_IMPL_VERSION,
            "lines": lines_version(),
        }
    )
