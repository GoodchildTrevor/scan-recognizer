"""Page-level PDF operations: text layer kind, rendering, text removal (sections 6.1, 6.2, 9.3)."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import pymupdf

from ocr_archive.models import TextLayer

logger = logging.getLogger(__name__)

#: Resolution of the "before/after" renders that decide whether text is visible.
COMPARE_DPI = 150
#: Largest allowed difference of any channel of any pixel for "nothing changed visually".
COMPARE_TOLERANCE = 1
#: Run the LRU cleanup of the render cache once per this many writes.
_CACHE_CHECK_EVERY = 32
#: After a cleanup the cache occupies at most this share of the limit.
_CACHE_TARGET_SHARE = 0.9

_cache_writes = 0


def _render_rgb(page: pymupdf.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    data = np.frombuffer(pix.samples, dtype=np.uint8)
    return data.reshape(pix.height, pix.stride)[:, : pix.width * pix.n].reshape(
        pix.height, pix.width, pix.n
    )


def pages_look_equal(
    doc_a: pymupdf.Document,
    index_a: int,
    doc_b: pymupdf.Document,
    index_b: int,
    *,
    dpi: int = COMPARE_DPI,
    tolerance: int = COMPARE_TOLERANCE,
) -> bool:
    """Compare two pages pixel by pixel.

    :param dpi: render resolution.
    :param tolerance: largest allowed difference of any channel of any pixel.
    :return: True when sizes match and no channel differs by more than ``tolerance``.
    """
    first = _render_rgb(doc_a[index_a], dpi)
    second = _render_rgb(doc_b[index_b], dpi)
    if first.shape != second.shape:
        return False
    diff = np.abs(first.astype(np.int16) - second.astype(np.int16))
    return int(diff.max(initial=0)) <= tolerance


def _single_page_copy(doc: pymupdf.Document, page_index: int) -> pymupdf.Document:
    copy = pymupdf.open()
    copy.insert_pdf(doc, from_page=page_index, to_page=page_index)
    return copy


def _redact(page: pymupdf.Page, rects: list[pymupdf.Rect], text_mode: int) -> None:
    """Apply fill-less redactions that never touch images or vector graphics."""
    for rect in rects:
        # PyMuPDF documents fill=False as "no fill at all"; its type hint omits it.
        page.add_redact_annot(rect, fill=False, cross_out=False)  # type: ignore[arg-type]
    page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_NONE,
        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
        text=text_mode,
    )


def strip_text(doc: pymupdf.Document, page_index: int) -> None:
    """Remove all text of a page; images and graphics stay untouched.

    :param doc: document modified in place.
    :param page_index: 0-based page index.
    """
    page = doc[page_index]
    _redact(page, [page.rect], pymupdf.PDF_REDACT_TEXT_REMOVE)


def text_layer_kind(pdf_path: Path, page_index: int) -> TextLayer:
    """Classify the text of a source page: none, invisible (hidden) or visible.

    The text is removed from a single-page copy; if the render does not change, the text was
    invisible (old OCR in render mode 3, text under the scan image, white text).

    :param pdf_path: source PDF.
    :param page_index: 0-based page index.
    """
    with pymupdf.open(pdf_path) as doc:
        if not doc[page_index].get_texttrace():
            return TextLayer.NONE
        with (
            _single_page_copy(doc, page_index) as before,
            _single_page_copy(doc, page_index) as after,
        ):
            strip_text(after, 0)
            if pages_look_equal(before, 0, after, 0):
                return TextLayer.HIDDEN
            return TextLayer.VISIBLE


def _invisible_span_rects(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects: list[pymupdf.Rect] = []
    for span in page.get_texttrace():
        # Render mode 3 is "neither fill nor stroke"; zero opacity is invisible as well.
        if span.get("type") != 3 and float(span.get("opacity", 1.0)) != 0.0:
            continue
        rect = pymupdf.Rect(span["bbox"]) & page.rect
        if rect.is_empty or rect.is_infinite:
            continue
        rects.append(rect)
    return rects


def strip_invisible_spans(doc: pymupdf.Document, page_index: int) -> bool:
    """Remove invisible text spans (old OCR) of a page whose visible text must stay.

    The removal is tried on a copy first. If the render of the copy differs from the page,
    the page is left untouched (section 9.3, ``old_text_kept``).

    :param doc: document modified in place.
    :param page_index: 0-based page index.
    :return: False when removal would change the render and nothing was removed.
    """
    rects = _invisible_span_rects(doc[page_index])
    if not rects:
        return True
    # Only invisible text is removed inside the rectangles: a visible stamp overlapping the
    # old OCR keeps its characters.
    mode = pymupdf.PDF_REDACT_TEXT_REMOVE_INVISIBLE
    with _single_page_copy(doc, page_index) as trial:
        _redact(trial[0], rects, mode)
        if not pages_look_equal(doc, page_index, trial, 0):
            logger.info("Invisible text kept on page %d: removal changes the render", page_index)
            return False
    _redact(doc[page_index], rects, mode)
    return True


def image_area_share(page: pymupdf.Page) -> float:
    """Return the summed area of raster image boxes, clipped to the page, as a page share."""
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    if page_area <= 0:
        return 0.0
    total = 0.0
    for info in page.get_image_info():
        rect = pymupdf.Rect(info["bbox"]) & page_rect
        if not rect.is_empty and not rect.is_infinite:
            total += rect.width * rect.height
    return total / page_area


def _pixmap(page: pymupdf.Page, long_side: int, clip: pymupdf.Rect | None) -> pymupdf.Pixmap:
    area = clip if clip is not None else page.rect
    zoom = long_side / max(area.width, area.height)
    return page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom), clip=clip, colorspace=pymupdf.csRGB, alpha=False
    )


def render_page(
    pdf_path: Path, page_index: int, long_side: int, *, clip: pymupdf.Rect | None = None
) -> bytes:
    """Render a page (or a clip of it) for the model: RGB PNG, no image processing.

    :param long_side: long side in pixels of the page, or of ``clip`` when given.
    :param clip: region in PDF points.
    :return: PNG bytes.
    """
    with pymupdf.open(pdf_path) as doc:
        return bytes(_pixmap(doc[page_index], long_side, clip).tobytes("png"))


def _clip_key(clip: pymupdf.Rect) -> str:
    values = ",".join(f"{v:.3f}" for v in (clip.x0, clip.y0, clip.x1, clip.y1))
    return hashlib.sha1(values.encode("ascii")).hexdigest()[:12]


def _cleanup_cache(root: Path, max_bytes: float) -> None:
    files: list[tuple[float, int, Path]] = []
    total = 0
    for directory, _dirs, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            files.append((stat.st_mtime, stat.st_size, path))
            total += stat.st_size
    if total <= max_bytes:
        return
    target = max_bytes * _CACHE_TARGET_SHARE
    for _mtime, size, path in sorted(files):
        if total <= target:
            break
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        total -= size
    logger.info("Render cache trimmed to %d bytes", total)


def render_cached(
    cache_dir: Path,
    page_id: int,
    pdf_path: Path,
    page_index: int,
    long_side: int,
    *,
    clip: pymupdf.Rect | None = None,
    max_gb: float,
) -> bytes:
    """Render a page through the LRU cache ``<cache_dir>/render/<page_id>/``.

    :param cache_dir: the data cache directory; it may be deleted at any time.
    :param max_gb: cache size limit; the oldest files are removed when it is exceeded.
    :return: PNG bytes, as :func:`render_page`.
    """
    global _cache_writes
    root = cache_dir / "render"
    name = f"{long_side}_{_clip_key(clip)}.png" if clip is not None else f"{long_side}.png"
    path = root / str(page_id) / name
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        pass
    else:
        with contextlib.suppress(FileNotFoundError):
            os.utime(path)
        return data
    data = render_page(pdf_path, page_index, long_side, clip=clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    _cache_writes += 1
    if _cache_writes % _CACHE_CHECK_EVERY == 1:
        _cleanup_cache(root, max_gb * 1024**3)
    return data


def render_view(
    pdf_path: Path,
    page_index: int,
    long_side: int,
    *,
    clip_fractions: tuple[float, float, float, float] | None,
    fmt: str = "jpeg",
    quality: int = 85,
) -> bytes:
    """Render a page image for the review window.

    Unlike :func:`render_page`, ``long_side`` always refers to the whole page; the clip only
    cuts a part of that render (section 11).

    :param clip_fractions: ``x0, y0, x1, y1`` as fractions of the page, or None.
    :param fmt: ``jpeg`` or ``png``.
    :param quality: JPEG quality.
    """
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_index]
        rect = page.rect
        clip: pymupdf.Rect | None = None
        if clip_fractions is not None:
            x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in clip_fractions)
            clip = pymupdf.Rect(
                rect.x0 + x0 * rect.width,
                rect.y0 + y0 * rect.height,
                rect.x0 + x1 * rect.width,
                rect.y0 + y1 * rect.height,
            )
        zoom = long_side / max(rect.width, rect.height)
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(zoom, zoom), clip=clip, colorspace=pymupdf.csRGB, alpha=False
        )
        if fmt == "png":
            return bytes(pix.tobytes("png"))
        return bytes(pix.tobytes("jpeg", jpg_quality=quality))
