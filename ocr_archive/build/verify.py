"""Build verification (section 9.5): any failure fails the build.

The DOCX is compared with what was handed to pandoc (the normalized text, section 9.1), never
with pandoc's own reading of the Markdown, so a parsing error of pandoc cannot appear on both
sides. Error messages carry page numbers and counts only, never document text.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from rapidfuzz.distance import Levenshtein

from ocr_archive.build.markdown import SOURCE_PAGE_STYLE, PageReference, TableRef
from ocr_archive.exceptions import BuildError
from ocr_archive.pdfpages import pages_look_equal
from ocr_archive.textnorm import numbers

_WORD_RE = re.compile(r"[^\W_]+")
_DIGITS_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
# Characters that content_text() would read as markup in literal text: protect them so that
# numbers() sees the literal text as it is.
_LITERAL_PROTECT = str.maketrans({"<": "‹", "&": "＆"})
_PAGE_SIZE_TOLERANCE = 0.01


@dataclass
class PageContent:
    """Text read back from the DOCX for one page, in the shape of :class:`PageReference`."""

    texts: list[str] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)
    tables: list[TableRef] = field(default_factory=list)


# --- DOCX AST ----------------------------------------------------------------------------------


def _attr_style(attr: Any) -> str | None:
    if isinstance(attr, list) and len(attr) == 3:
        for key, value in attr[2]:
            if key == "custom-style":
                return str(value)
    return None


def _inlines_text(inlines: list[dict[str, Any]], formulas: list[str]) -> str:
    out: list[str] = []
    for node in inlines:
        kind = node.get("t")
        content: Any = node.get("c")
        if kind == "Str":
            out.append(str(content))
        elif kind in ("Space", "SoftBreak", "LineBreak"):
            out.append(" ")
        elif kind in (
            "Emph",
            "Strong",
            "Underline",
            "Strikeout",
            "Superscript",
            "Subscript",
            "SmallCaps",
        ):
            out.append(_inlines_text(content, formulas))
        elif kind == "Quoted":
            quote = '"' if content[0].get("t") == "DoubleQuote" else "'"
            out.append(quote + _inlines_text(content[1], formulas) + quote)
        elif kind in ("Cite", "Span"):
            out.append(_inlines_text(content[1], formulas))
        elif kind == "Link":
            out.append(_inlines_text(content[1], formulas))
        elif kind == "Code":
            out.append(str(content[1]))
        elif kind == "Math":
            formulas.append(str(content[1]))
            out.append(" ")
        # RawInline, Image and Note carry no text of the page.
    return "".join(out)


def _table_rows(table: list[Any]) -> list[list[Any]]:
    head, bodies, foot = table[3], table[4], table[5]
    rows: list[list[Any]] = list(head[1])
    for body in bodies:
        rows.extend(body[2])
        rows.extend(body[3])
    rows.extend(foot[1])
    return rows


def _table_content(table: list[Any], page: PageContent) -> None:
    rows = _table_rows(table)
    shape: list[int] = []
    carry: dict[int, int] = {}
    first_empty = False
    for number, row in enumerate(rows):
        occupied = {col for col, left in carry.items() if left > 0}
        next_carry = {col: left - 1 for col, left in carry.items() if left > 1}
        col = 0
        texts: list[str] = []
        for cell in row[1]:
            rowspan, colspan = int(cell[2]), int(cell[3])
            while col in occupied:
                col += 1
            for c in range(col, col + max(1, colspan)):
                occupied.add(c)
                if rowspan > 1:
                    next_carry[c] = rowspan - 1
            col += max(1, colspan)
            cell_page = PageContent()
            _blocks_content(cell[4], cell_page)
            page.formulas.extend(cell_page.formulas)
            page.tables.extend(cell_page.tables)
            texts.append(" ".join(cell_page.texts))
        shape.append(len(occupied))
        carry = next_carry
        row_text = " ".join(texts)
        if number == 0:
            first_empty = not row_text.strip() and not any(
                "Math" in str(cell[4]) for cell in row[1]
            )
        page.texts.append(row_text)
    page.tables.append(TableRef(shape, first_empty))


def _blocks_content(blocks: list[dict[str, Any]], page: PageContent) -> None:
    for node in blocks:
        kind = node.get("t")
        content: Any = node.get("c")
        if kind in ("Para", "Plain", "Header"):
            inlines = content[2] if kind == "Header" else content
            text = _inlines_text(inlines, page.formulas)
            if text.strip():
                page.texts.append(text)
        elif kind == "LineBlock":
            for line in content:
                page.texts.append(_inlines_text(line, page.formulas))
        elif kind == "CodeBlock":
            page.texts.append(str(content[1]))
        elif kind in ("BlockQuote",):
            _blocks_content(content, page)
        elif kind == "Div":
            _blocks_content(content[1], page)
        elif kind in ("BulletList", "OrderedList"):
            items = content if kind == "BulletList" else content[1]
            for item in items:
                _blocks_content(item, page)
        elif kind == "DefinitionList":
            for term, definitions in content:
                page.texts.append(_inlines_text(term, page.formulas))
                for definition in definitions:
                    _blocks_content(definition, page)
        elif kind == "Figure":
            _blocks_content(content[2], page)
        elif kind == "Table":
            _table_content(content, page)


def _flatten(blocks: list[dict[str, Any]], out: list[tuple[bool, dict[str, Any]]]) -> None:
    """Flatten style divs; mark every paragraph of a "Source Page" div as a page start."""
    for node in blocks:
        if node.get("t") == "Div":
            attr, inner = node["c"]
            if _attr_style(attr) == SOURCE_PAGE_STYLE:
                out.extend((True, child) for child in inner)
                continue
            _flatten(inner, out)
        else:
            out.append((False, node))


def docx_pages(ast: dict[str, Any]) -> list[PageContent]:
    """Split the DOCX AST into pages by the "Source Page" paragraphs.

    Each paragraph in a "Source Page" div starts a page (consecutive empty pages may share one
    div). A "Стр. N" line inside the page text is ordinary text and never splits pages.
    Everything before the first page (draft banner, title) is ignored.
    """
    flat: list[tuple[bool, dict[str, Any]]] = []
    _flatten(ast.get("blocks", []), flat)
    pages: list[PageContent] = []
    for is_start, node in flat:
        if is_start:
            pages.append(PageContent())
        elif pages:
            _blocks_content([node], pages[-1])
    return pages


# --- comparison --------------------------------------------------------------------------------


def _literal_numbers(texts: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(numbers(text.translate(_LITERAL_PROTECT)))
    return counter


def _words(texts: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(_WORD_RE.findall(unicodedata.normalize("NFC", text)))
    return counter


def _formula_digits(formulas: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for formula in formulas:
        counter.update(_DIGITS_RE.findall(formula))
    return counter


def _flat(texts: list[str]) -> str:
    return _WS_RE.sub("", unicodedata.normalize("NFC", "".join(texts)))


def _absorb_fallback_formulas(reference: PageReference, actual: PageContent) -> PageContent:
    """Treat formulas pandoc could not convert (written back as their TeX text) as formulas."""
    if len(actual.formulas) >= len(reference.formulas):
        return actual
    texts = list(actual.texts)
    formulas = list(actual.formulas)
    for body in reference.formulas:
        body = body.strip()
        if not body:
            continue
        written = [f"$${body}$$", f"${body}$", f"\\({body}\\)", f"\\[{body}\\]", body]
        found = next(
            ((i, form) for form in written for i, text in enumerate(texts) if form in text),
            None,
        )
        if found is not None:
            index, form = found
            texts[index] = texts[index].replace(form, " ", 1)
            formulas.append(body)
    return PageContent(texts, formulas, actual.tables)


def _counter_diff(expected: Counter[str], actual: Counter[str]) -> tuple[int, int]:
    missing = sum((expected - actual).values())
    extra = sum((actual - expected).values())
    return missing, extra


def compare_page(
    number: int, reference: PageReference, actual: PageContent, min_similarity: float
) -> list[str]:
    """Compare one page; return problems (empty when the page is fine).

    :param number: 1-based page number for the messages.
    """
    actual = _absorb_fallback_formulas(reference, actual)
    problems: list[str] = []
    missing, extra = _counter_diff(
        _literal_numbers(reference.texts), _literal_numbers(actual.texts)
    )
    if missing or extra:
        problems.append(f"page {number}: numbers differ ({missing} missing, {extra} extra)")
    missing, extra = _counter_diff(
        _formula_digits(reference.formulas), _formula_digits(actual.formulas)
    )
    if missing or extra:
        problems.append(f"page {number}: formula digits differ ({missing} missing, {extra} extra)")
    missing, extra = _counter_diff(_words(reference.texts), _words(actual.texts))
    if missing or extra:
        problems.append(f"page {number}: words differ ({missing} missing, {extra} extra)")
    expected_tables = [table.canonical() for table in reference.tables]
    actual_tables = [table.canonical() for table in actual.tables]
    if expected_tables != actual_tables:
        problems.append(
            f"page {number}: tables differ ({len(expected_tables)} expected, "
            f"{len(actual_tables)} found)"
        )
    expected_flat, actual_flat = _flat(reference.texts), _flat(actual.texts)
    if expected_flat or actual_flat:
        similarity = Levenshtein.normalized_similarity(expected_flat, actual_flat)
        if similarity < min_similarity:
            problems.append(f"page {number}: text order similarity {similarity:.4f}")
    return problems


def verify_docx(
    ast: dict[str, Any], references: list[PageReference], min_similarity: float
) -> None:
    """Verify the DOCX read back against the references of all pages (section 9.5).

    :raises BuildError: listing the problems found.
    """
    pages = docx_pages(ast)
    if len(pages) != len(references):
        raise BuildError(f"DOCX has {len(pages)} pages, expected {len(references)}")
    problems: list[str] = []
    for number, (reference, actual) in enumerate(zip(references, pages, strict=True), start=1):
        problems.extend(compare_page(number, reference, actual, min_similarity))
    if problems:
        shown = "; ".join(problems[:10])
        more = f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""
        raise BuildError(f"DOCX verification failed: {shown}{more}")


# --- PDF ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PdfPageExpectation:
    """Expected text of a result page (section 9.5).

    :ivar page_index: 0-based page index.
    :ivar text: text the page must hold: the text left from the source followed by our layer,
        or the unchanged source text.
    """

    page_index: int
    text: str


def comparable_text(text: str) -> str:
    """NFC text without any whitespace, as the PDF text check compares it."""
    return _WS_RE.sub("", unicodedata.normalize("NFC", text))


def verify_pdf(source_pdf: Path, result_pdf: Path, expectations: list[PdfPageExpectation]) -> None:
    """Verify the searchable PDF: structure, look and text of every page (section 9.5).

    :raises BuildError: on the first problem class found, with page numbers.
    """
    try:
        with pymupdf.open(source_pdf) as source, pymupdf.open(result_pdf) as result:
            if source.page_count != result.page_count:
                raise BuildError(
                    f"PDF has {result.page_count} pages, source has {source.page_count}"
                )
            problems: list[str] = []
            for index in range(source.page_count):
                a, b = source[index].rect, result[index].rect
                if (
                    abs(a.width - b.width) > _PAGE_SIZE_TOLERANCE
                    or abs(a.height - b.height) > _PAGE_SIZE_TOLERANCE
                ):
                    problems.append(f"page {index + 1}: size changed")
                elif not pages_look_equal(source, index, result, index):
                    problems.append(f"page {index + 1}: render changed")
            by_index = {e.page_index: e for e in expectations}
            for index in range(result.page_count):
                expectation = by_index.get(index)
                if expectation is None:
                    problems.append(f"page {index + 1}: no expected text")
                    continue
                actual = comparable_text(result[index].get_text())
                if actual != comparable_text(expectation.text):
                    problems.append(f"page {index + 1}: text differs from the expected layer")
    except (RuntimeError, ValueError) as err:
        raise BuildError(f"cannot verify PDF: {err}") from err
    if problems:
        shown = "; ".join(problems[:10])
        more = f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""
        raise BuildError(f"PDF verification failed: {shown}{more}")
