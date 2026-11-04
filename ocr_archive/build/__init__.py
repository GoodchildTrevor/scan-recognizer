"""Build of the results: snapshot, normalization, DOCX, PDF, chunks, verification, publication."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ocr_archive.models import BuildKind

if TYPE_CHECKING:
    from ocr_archive.models import BuildContext


def build_document(
    document_id: str, ctx: BuildContext, *, kind: BuildKind = BuildKind.FINAL
) -> int:
    """Build the document from a snapshot and publish it atomically (section 9).

    :param document_id: document to build.
    :param ctx: build context.
    :param kind: final build (document in ``ready``) or draft DOCX.
    :return: build version.
    :raises BuildError: when a step or a verification fails.
    """
    # Imported lazily: textnorm imports build.markers, and the build modules import textnorm.
    from ocr_archive.build.document import run_build

    return run_build(document_id, ctx, kind=kind)
