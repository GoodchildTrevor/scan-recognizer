"""Vision-language model access: prompts, llama-server client, token probabilities."""

from __future__ import annotations

from functools import cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).with_name("prompts")

TRANSCRIBE_PROMPT = "transcribe_v1"
#: Prompt for split parts and regions; part of the pass2 OCR version (section 6.11).
FRAGMENT_PROMPT = "fragment_v1"


@cache
def load_prompt(name: str) -> str:
    """Return the text of ``prompts/<name>.md``; the file name is the prompt version."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
