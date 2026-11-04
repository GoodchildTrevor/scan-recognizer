"""Text line detector on the page image (section 6.3), split gap (6.7) and line plausibility.

This is not OCR: it finds horizontal bands of ink with numpy. ``reliable`` is a heuristic
"one column, even lines, no large blocks"; every consumer has a safe fallback without it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pymupdf
from PIL import Image

from ocr_archive.models import LineBox, LineDetection
from ocr_archive.settings import ChecksConfig

logger = logging.getLogger(__name__)

#: Raise when the detector code changes: stored boxes are then recomputed (section 6.11).
LINES_IMPL_VERSION = "1"

#: All detector parameters; they are part of the rules version and the OCR version of pass2.
LINES_PARAMS: dict[str, float | int] = {
    "render_long_side": 2000,
    "skew_long_side": 1000,
    "skew_max_deg": 3.0,
    "skew_step_deg": 0.25,
    "min_contrast": 40,
    "frame_column_ink": 0.5,
    "smooth_px": 3,
    "row_ink_min": 0.005,
    "merge_gap_ratio": 0.25,
    "min_band_ratio": 0.3,
    "box_percentile_lo": 1,
    "box_percentile_hi": 99,
    "column_empty_band_share": 0.8,
    "column_gap_min_width": 0.02,
    "column_gap_zone_lo": 0.2,
    "column_gap_zone_hi": 0.8,
    "min_bands": 3,
    "max_height_cv": 0.6,
    "max_band_height_ratio": 3.0,
    "plausible_min_chars": 10,
    "plausible_tolerance": 0.35,
}

BoolArray = npt.NDArray[np.bool_]
UInt8Array = npt.NDArray[np.uint8]


def lines_version() -> str:
    """Return the detector version: implementation version plus a hash of the parameters.

    :return: string stored in ``pages.line_boxes_version``.
    """
    payload = json.dumps(
        {"impl": LINES_IMPL_VERSION, "params": LINES_PARAMS}, sort_keys=True, separators=(",", ":")
    )
    return f"{LINES_IMPL_VERSION}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _render_gray(page: pymupdf.Page, long_side: int) -> tuple[UInt8Array, float]:
    """Render a page in grayscale so that its long side equals ``long_side`` pixels."""
    rect = page.rect
    zoom = long_side / max(rect.width, rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY, alpha=False)
    data = np.frombuffer(pix.samples, dtype=np.uint8)
    gray = data.reshape(pix.height, pix.stride)[:, : pix.width].copy()
    return gray, zoom


def _otsu(gray: UInt8Array) -> int | None:
    """Return the Otsu threshold, or None when the image has no ink/paper contrast."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    levels = np.arange(256, dtype=np.float64)
    mu = np.cumsum(prob * levels)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_total * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = 0.0
    threshold = int(np.argmax(sigma_b))
    low_weight = omega[threshold]
    if low_weight <= 0.0 or low_weight >= 1.0:
        return None
    mean_low = mu[threshold] / low_weight
    mean_high = (mu_total - mu[threshold]) / (1.0 - low_weight)
    if mean_high - mean_low < float(LINES_PARAMS["min_contrast"]):
        return None
    return threshold


def _ink_mask(gray: UInt8Array, threshold: int | None) -> BoolArray:
    """Dark pixels as ink; columns that are mostly ink (frames, black margins) are ignored."""
    if threshold is None:
        return np.zeros(gray.shape, dtype=np.bool_)
    ink = gray <= threshold
    column_share = ink.mean(axis=0)
    ink[:, column_share > float(LINES_PARAMS["frame_column_ink"])] = False
    return ink


def _page_ink(page: pymupdf.Page) -> tuple[UInt8Array, BoolArray, int | None, float]:
    gray, zoom = _render_gray(page, int(LINES_PARAMS["render_long_side"]))
    threshold = _otsu(gray)
    return gray, _ink_mask(gray, threshold), threshold, zoom


def _estimate_skew(gray: UInt8Array, threshold: int | None) -> float:
    """Return the straightening angle: the largest variance of the horizontal projection.

    The angle follows PIL: counter-clockwise on screen.
    """
    if threshold is None:
        return 0.0
    height, width = gray.shape
    scale = float(LINES_PARAMS["skew_long_side"]) / max(height, width)
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = np.asarray(Image.fromarray(gray).resize(small_size, Image.Resampling.BOX))
    mask = _ink_mask(small, threshold)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    max_deg = float(LINES_PARAMS["skew_max_deg"])
    step = float(LINES_PARAMS["skew_step_deg"])
    count = round(max_deg / step)
    # Smaller angles first, so ties resolve to the least rotation.
    angles = sorted((k * step for k in range(-count, count + 1)), key=abs)
    best_angle = 0.0
    best_score = -1.0
    for angle in angles:
        rotated = image.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
        projection = np.asarray(rotated, dtype=np.float64).sum(axis=1)
        score = float(projection.var())
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def _rotate_mask(mask: BoolArray, angle: float) -> BoolArray:
    if angle == 0.0:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    rotated = image.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    return np.asarray(rotated) > 127


def _runs(flags: BoolArray) -> list[tuple[int, int]]:
    """Return half-open ``[start, end)`` runs of True values."""
    padded = np.concatenate(([0], flags.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.nonzero(diff == 1)[0]
    ends = np.nonzero(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def _find_bands(mask: BoolArray) -> list[tuple[int, int]]:
    width = mask.shape[1]
    row_share = mask.sum(axis=1) / float(width)
    window = int(LINES_PARAMS["smooth_px"])
    smoothed = np.convolve(row_share, np.ones(window) / window, mode="same")
    bands = _runs(smoothed > float(LINES_PARAMS["row_ink_min"]))
    if not bands:
        return []
    median_height = float(np.median([end - start for start, end in bands]))
    merged: list[tuple[int, int]] = [bands[0]]
    for start, end in bands[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < float(LINES_PARAMS["merge_gap_ratio"]) * median_height:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    median_height = float(np.median([end - start for start, end in merged]))
    min_height = float(LINES_PARAMS["min_band_ratio"]) * median_height
    return [(start, end) for start, end in merged if end - start >= min_height]


def _unrotate_point(
    x: float, y: float, angle_deg: float, cx: float, cy: float
) -> tuple[float, float]:
    """Map a point of the rotated mask back to the original image.

    PIL rotates counter-clockwise on screen (y axis down) around the image center.
    """
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx, dy = x - cx, y - cy
    return cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t


def _band_box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    angle: float,
    shape: tuple[int, int],
    zoom: float,
    origin: pymupdf.Point,
) -> LineBox:
    height, width = shape
    cx, cy = width / 2.0, height / 2.0
    corners = [
        _unrotate_point(x, y, angle, cx, cy) for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    ]
    xs = [max(0.0, min(float(width), x)) for x, _ in corners]
    ys = [max(0.0, min(float(height), y)) for _, y in corners]
    return LineBox(
        x0=min(xs) / zoom + origin.x,
        y0=min(ys) / zoom + origin.y,
        x1=max(xs) / zoom + origin.x,
        y1=max(ys) / zoom + origin.y,
    )


def _column_gap(mask: BoolArray, bands: list[tuple[int, int]]) -> bool:
    """Detect an empty vertical corridor in the middle of the text block: columns or a table."""
    width = mask.shape[1]
    per_band = np.stack([mask[start:end].any(axis=0) for start, end in bands])
    ink_count = per_band.sum(axis=0)
    ink_columns = np.nonzero(ink_count > 0)[0]
    if ink_columns.size == 0:
        return False
    block_x0 = float(np.percentile(ink_columns, float(LINES_PARAMS["box_percentile_lo"])))
    block_x1 = float(np.percentile(ink_columns, float(LINES_PARAMS["box_percentile_hi"])))
    block_width = block_x1 - block_x0
    if block_width <= 0:
        return False
    max_ink_bands = (1.0 - float(LINES_PARAMS["column_empty_band_share"])) * len(bands)
    lo = math.floor(block_x0)
    hi = math.ceil(block_x1) + 1
    empty = ink_count[lo:hi] <= max_ink_bands
    zone_lo = block_x0 + float(LINES_PARAMS["column_gap_zone_lo"]) * block_width
    zone_hi = block_x0 + float(LINES_PARAMS["column_gap_zone_hi"]) * block_width
    min_gap = float(LINES_PARAMS["column_gap_min_width"]) * width
    for start, end in _runs(empty):
        gap_start, gap_end = lo + start, lo + end
        if gap_end - gap_start >= min_gap and gap_start >= zone_lo and gap_end <= zone_hi:
            return True
    return False


def detect_lines(pdf_path: Path, page_index: int) -> LineDetection:
    """Detect text line boxes on a page.

    :param pdf_path: source PDF.
    :param page_index: 0-based page index.
    :return: boxes in PDF points (page coordinate system) and the reliability verdict.
    """
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_index]
        origin = page.rect.tl
        gray, mask, threshold, zoom = _page_ink(page)
    correction = _estimate_skew(gray, threshold)
    # The page is skewed by the opposite of the angle that straightens it.
    skew = -correction
    rotated = _rotate_mask(mask, correction)
    bands = _find_bands(rotated)
    if not bands:
        return LineDetection(boxes=[], reliable=False, columns_detected=False, skew_deg=skew)

    boxes: list[LineBox] = []
    kept: list[tuple[int, int]] = []
    lo_pct = float(LINES_PARAMS["box_percentile_lo"])
    hi_pct = float(LINES_PARAMS["box_percentile_hi"])
    for start, end in bands:
        columns = np.nonzero(rotated[start:end].any(axis=0))[0]
        if columns.size == 0:
            continue
        x0 = float(np.percentile(columns, lo_pct))
        x1 = float(np.percentile(columns, hi_pct)) + 1.0
        boxes.append(
            _band_box(x0, float(start), x1, float(end), correction, rotated.shape, zoom, origin)
        )
        kept.append((start, end))

    columns_detected = _column_gap(rotated, kept) if kept else False
    heights = np.array([end - start for start, end in kept], dtype=np.float64)
    reliable = False
    if len(kept) >= int(LINES_PARAMS["min_bands"]):
        median_height = float(np.median(heights))
        variation = float(heights.std() / heights.mean()) if heights.mean() > 0 else math.inf
        tall_band = bool(
            (heights > float(LINES_PARAMS["max_band_height_ratio"]) * median_height).any()
        )
        reliable = (
            variation < float(LINES_PARAMS["max_height_cv"])
            and not tall_band
            and not columns_detected
        )
    logger.debug(
        "Detected %d lines on page %d: reliable=%s columns=%s skew=%.2f",
        len(boxes),
        page_index,
        reliable,
        columns_detected,
        skew,
    )
    return LineDetection(
        boxes=boxes, reliable=reliable, columns_detected=columns_detected, skew_deg=skew
    )


def _longest_run(flags: BoolArray) -> tuple[int, int] | None:
    runs = _runs(flags)
    if not runs:
        return None
    return max(runs, key=lambda run: run[1] - run[0])


def find_split_gap(
    pdf_path: Path, page_index: int, lines: LineDetection, cfg: ChecksConfig
) -> float | None:
    """Find a cut between two lines near the middle of the page that crosses no ink.

    Only the gap itself is checked here; the caller checks ``reliable``, the line count and
    tables (section 6.7).

    :param pdf_path: source PDF.
    :param page_index: 0-based page index.
    :param lines: detection of the same page.
    :param cfg: checks block; ``split_gap_max_ink`` is the allowed ink share of a gap row.
    :return: y of the cut in PDF points, or None when there is no suitable gap.
    """
    boxes = lines.boxes
    count = len(boxes)
    if count < 2:
        return None
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_index]
        origin_y = page.rect.y0
        _gray, mask, _threshold, zoom = _page_ink(page)
    height, width = mask.shape
    max_ink = cfg.split_gap_max_ink * width
    median_height = float(np.median([(box.y1 - box.y0) * zoom for box in boxes]))
    middle = count / 2.0
    first = max(0, int(middle) - 3)
    last = min(count - 2, int(middle) + 3)
    best: tuple[float, int, float] | None = None
    for k in range(first, last + 1):
        upper, lower = boxes[k], boxes[k + 1]
        top = round(((upper.y0 + upper.y1) / 2.0 - origin_y) * zoom)
        bottom = round(((lower.y0 + lower.y1) / 2.0 - origin_y) * zoom)
        top, bottom = max(0, top), min(height, bottom)
        if bottom - top < 1:
            continue
        clean = mask[top:bottom].sum(axis=1) <= max_ink
        run = _longest_run(clean)
        if run is None or run[1] - run[0] < 0.5 * median_height:
            continue
        center_px = top + (run[0] + run[1]) / 2.0
        distance = abs((k + 1) - middle)
        if best is None or distance < best[0]:
            best = (distance, k, center_px)
    if best is None:
        return None
    return best[2] / zoom + origin_y


def line_plausible(lines: list[str], boxes: list[LineBox], *, min_share: float) -> bool:
    """Check that text lines plausibly match line boxes one to one.

    For typewritten text the ratio "characters / box width" is almost constant. A missing line
    together with a split one keeps the count but shifts this ratio.

    :param lines: counted text lines; the caller ensures ``len(lines) == len(boxes)``.
    :param boxes: detected boxes in the same order.
    :param min_share: share of long lines that must be within the tolerance of the median
        (1.0 for crops, 0.9 for the PDF layer).
    :return: True when the share is reached, or when there are no lines long enough to judge.
    """
    min_chars = int(LINES_PARAMS["plausible_min_chars"])
    tolerance = float(LINES_PARAMS["plausible_tolerance"])
    ratios = [
        len(line.strip()) / (box.x1 - box.x0)
        for line, box in zip(lines, boxes, strict=False)
        if len(line.strip()) >= min_chars and box.x1 > box.x0
    ]
    if not ratios:
        return True
    median = float(np.median(ratios))
    if median <= 0:
        return False
    within = sum(1 for ratio in ratios if abs(ratio / median - 1.0) <= tolerance)
    return within / len(ratios) >= min_share


def line_detection_to_json(detection: LineDetection) -> str:
    """Serialize a detection for ``pages.line_boxes_json``."""
    return json.dumps(
        {
            "boxes": [
                [round(b.x0, 2), round(b.y0, 2), round(b.x1, 2), round(b.y1, 2)]
                for b in detection.boxes
            ],
            "reliable": detection.reliable,
            "columns_detected": detection.columns_detected,
            "skew_deg": detection.skew_deg,
        },
        separators=(",", ":"),
    )


def line_detection_from_json(data: str) -> LineDetection:
    """Restore a detection stored by :func:`line_detection_to_json`."""
    raw = json.loads(data)
    return LineDetection(
        boxes=[LineBox(*(float(v) for v in box)) for box in raw["boxes"]],
        reliable=bool(raw["reliable"]),
        columns_detected=bool(raw["columns_detected"]),
        skew_deg=float(raw["skew_deg"]),
    )
