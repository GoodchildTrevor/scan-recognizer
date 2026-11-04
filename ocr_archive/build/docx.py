"""DOCX through pandoc (section 9.2) and reading it back for the verification (section 9.5)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from ocr_archive.exceptions import BuildError
from ocr_archive.settings import BuildConfig, resolve_asset

logger = logging.getLogger(__name__)

MARKDOWN_FORMAT = "markdown-smart-tex_math_dollars+tex_math_single_backslash"
PANDOC_TIMEOUT_S = 600

_CORE_PATH = "docProps/core.xml"
_CORE_NAMESPACES = {
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "dcmitype": "http://purl.org/dc/dcmitype/",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}


def run_pandoc(args: list[str], cfg: BuildConfig) -> bytes:
    """Run pandoc and return its stdout.

    :param args: arguments after the executable.
    :raises BuildError: when pandoc is missing, times out or exits with an error.
    """
    command = [cfg.pandoc_path, *args]
    try:
        completed = subprocess.run(
            command, capture_output=True, timeout=PANDOC_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError) as err:
        raise BuildError(f"pandoc failed to run: {err}") from err
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()[-2000:]
        raise BuildError(f"pandoc exited with {completed.returncode}: {stderr}")
    return completed.stdout


def pandoc_version(cfg: BuildConfig) -> str:
    """Return the first line of ``pandoc --version``."""
    output = run_pandoc(["--version"], cfg).decode("utf-8", errors="replace")
    return output.splitlines()[0].strip() if output else ""


def build_docx(markdown_path: Path, out_path: Path, cfg: BuildConfig) -> None:
    """Convert the pandoc Markdown to DOCX with the reference document and the HTML table filter.

    :param markdown_path: input Markdown file.
    :param out_path: DOCX to write.
    :raises BuildError: when pandoc fails.
    """
    run_pandoc(
        [
            "-f",
            MARKDOWN_FORMAT,
            "-t",
            "docx",
            f"--reference-doc={resolve_asset(cfg.reference_docx)}",
            f"--lua-filter={resolve_asset(cfg.lua_filter)}",
            "-o",
            str(out_path),
            str(markdown_path),
        ],
        cfg,
    )


def read_docx_ast(docx_path: Path, cfg: BuildConfig) -> dict[str, Any]:
    """Read a DOCX back into the pandoc JSON AST with paragraph and character styles.

    :raises BuildError: when pandoc fails or the output is not JSON.
    """
    output = run_pandoc([str(docx_path), "-f", "docx+styles", "-t", "json"], cfg)
    try:
        ast = json.loads(output)
    except ValueError as err:
        raise BuildError(f"pandoc returned invalid JSON: {err}") from err
    if not isinstance(ast, dict) or "blocks" not in ast:
        raise BuildError("pandoc returned an unexpected AST")
    return ast


def set_core_properties(docx_path: Path, *, title: str, subject: str) -> None:
    """Set the title and subject document properties (``docProps/core.xml``).

    Pandoc would render a ``title`` metadata field as a visible paragraph, so the properties of
    the draft are written into the package after conversion.

    :raises BuildError: when the package cannot be rewritten.
    """
    for prefix, uri in _CORE_NAMESPACES.items():
        ElementTree.register_namespace(prefix, uri)
    tmp_path = docx_path.with_name(docx_path.name + ".props.tmp")
    try:
        with zipfile.ZipFile(docx_path) as source:
            entries = [(info, source.read(info.filename)) for info in source.infolist()]
        names = {info.filename for info, _ in entries}
        core = next((data for info, data in entries if info.filename == _CORE_PATH), None)
        root = (
            ElementTree.fromstring(core)
            if core is not None
            else ElementTree.Element(f"{{{_CORE_NAMESPACES['cp']}}}coreProperties")
        )
        for tag, value in (("title", title), ("subject", subject)):
            element = root.find(f"dc:{tag}", _CORE_NAMESPACES)
            if element is None:
                element = ElementTree.SubElement(root, f"{{{_CORE_NAMESPACES['dc']}}}{tag}")
            element.text = value
        new_core = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as target:
            for info, data in entries:
                target.writestr(info, new_core if info.filename == _CORE_PATH else data)
            if _CORE_PATH not in names:
                # Without a relationship the part is ignored by Word, but pandoc always writes it.
                target.writestr(_CORE_PATH, new_core)
        os.replace(tmp_path, docx_path)
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError) as err:
        tmp_path.unlink(missing_ok=True)
        raise BuildError(f"cannot set DOCX properties: {err}") from err
