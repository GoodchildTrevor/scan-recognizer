"""Exception hierarchy of the OCR archive."""


class OcrArchiveError(Exception):
    """Base class of all service errors."""


class StatusTransitionError(OcrArchiveError):
    """A document or page is not in a status that allows the operation.

    :param code: machine-readable reason returned by the API as ``detail``.
    """

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


class StaleRevisionError(StatusTransitionError):
    """The page changed since the review window loaded it."""

    def __init__(self, message: str = "") -> None:
        super().__init__("stale", message)


class DraftConflictError(StatusTransitionError):
    """The draft was saved from another tab since the window loaded it."""

    def __init__(self, message: str = "") -> None:
        super().__init__("draft_conflict", message)


class VlmTransportError(OcrArchiveError):
    """The model server is temporarily unavailable: timeout, connection error, 502/503/504."""


class VlmTimeoutError(VlmTransportError):
    """The model server did not answer within ``vlm.timeout_s``.

    Generation time is reproducible at temperature 0, so a page that times out repeatedly
    is treated as failed after a few attempts (section 12.3).
    """


class VlmRequestError(OcrArchiveError):
    """The model server rejected the request permanently.

    :param status_code: HTTP status, or None for an invalid response body.
    :param context_exceeded: the request did not fit into the context window.
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, context_exceeded: bool = False
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.context_exceeded = context_exceeded


class VlmInvalidResponseError(VlmRequestError):
    """The response has no choices or cannot be parsed."""


class BuildError(OcrArchiveError):
    """A build step or a build verification failed."""


class ExportError(OcrArchiveError):
    """Copying the final files to the export folder failed."""


class IngestError(OcrArchiveError):
    """The ingester did not accept the chunks."""


class ConfigError(OcrArchiveError):
    """The configuration is invalid."""
