"""Marker vocabulary of the prompts (section 7): parser contract of checks, review and build."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

MARKER_RE = re.compile(
    r"\[(неразб\.|рук\.|печать|подпись|рис\.|таблица: проверить|формула: проверить|обрезано)"
    r"(:[^\]]*)?\]"
)

HANDWRITING_KIND = "рук."


@dataclass(frozen=True)
class Marker:
    start: int
    end: int
    kind: str  # first group, e.g. "неразб."
    inner: str  # text between the brackets, e.g. "неразб.: вариант?"


def find_markers(text: str) -> list[Marker]:
    """Return all markers in order of appearance."""
    return [
        Marker(m.start(), m.end(), m.group(1), m.group(1) + (m.group(2) or ""))
        for m in MARKER_RE.finditer(text)
    ]


def is_hard_marker(inner: str, hard_markers: Iterable[str]) -> bool:
    """Tell whether a marker means "not read": equal to a list item or starting with item + colon.

    :param inner: text between the brackets.
    :param hard_markers: ``decide.hard_markers``.
    """
    return any(inner == item or inner.startswith(item + ":") for item in hard_markers)


def hard_markers_in(text: str, hard_markers: Iterable[str]) -> list[Marker]:
    """Return markers of the text that mean "not read"."""
    items = list(hard_markers)
    return [m for m in find_markers(text) if is_hard_marker(m.inner, items)]


def handwriting_markers(text: str, hard_markers: Iterable[str]) -> list[Marker]:
    """Return read handwritten notes ``[рук.: текст]``, excluding unreadable ones."""
    items = list(hard_markers)
    return [
        m
        for m in find_markers(text)
        if m.kind == HANDWRITING_KIND and m.inner != m.kind and not is_hard_marker(m.inner, items)
    ]


def is_marker_only(line: str) -> bool:
    """Tell whether a line consists of markers and whitespace only, with at least one marker."""
    if not MARKER_RE.search(line):
        return False
    return MARKER_RE.sub("", line).strip() == ""
