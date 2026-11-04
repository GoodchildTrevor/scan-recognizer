"""manifest.json of a build (section 9.5): the snapshot, file hashes and tool versions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pymupdf

from ocr_archive import __version__
from ocr_archive.db.connection import utc_now
from ocr_archive.settings import Settings

_HASH_BLOCK = 1 << 20


def file_sha256(path: Path) -> str:
    """sha256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def tool_versions(settings: Settings, pandoc: str) -> dict[str, Any]:
    """Versions of the tools and the model the build depends on."""
    return {
        "pandoc": pandoc,
        "pymupdf": pymupdf.VersionBind,
        "ocr_archive": __version__,
        "model": settings.vlm.model,
        "weights": settings.vlm.weights.model_dump(),
    }


def build_manifest(
    *,
    document_id: str,
    title: str,
    kind: str,
    version: int,
    rules_version: str | None,
    snapshot: list[dict[str, Any]],
    files: dict[str, str],
    output_dir: Path,
    versions: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the manifest; file hashes are computed from ``output_dir``.

    :param files: role (``pdf``, ``docx``, ``chunks``) to file name inside ``output_dir``.
    """
    return {
        "document_id": document_id,
        "title": title,
        "kind": kind,
        "build_version": version,
        "created_at": utc_now(),
        "rules_version": rules_version,
        "files": files,
        "sha256": {role: file_sha256(output_dir / name) for role, name in files.items()},
        "versions": versions,
        "snapshot": snapshot,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest as UTF-8 JSON."""
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
