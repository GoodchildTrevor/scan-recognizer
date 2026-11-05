"""Configuration: config.yaml plus environment variables for secrets (section 18)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ocr_archive.exceptions import ConfigError
from ocr_archive.models import MANDATORY_FLAGS, FlagCode

PACKAGE_DIR = Path(__file__).resolve().parent

#: Keys the ingester rejects in extra_payload with 422 (section 3.2).
RESERVED_PAYLOAD_KEYS = frozenset(
    {"name", "file_path", "file_format", "creation_date", "modification_date"}
)

#: Token probabilities above this are not stored (section 6.6), so thresholds cannot exceed it.
MAX_TOKEN_P = 0.99


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ApiConfig(_Block):
    host: str = "172.17.0.1"
    port: int = 8090


class VlmWeights(_Block):
    gguf_file: str = ""
    gguf_sha256: str = ""
    mmproj_file: str = ""
    mmproj_sha256: str = ""


class VlmConfig(_Block):
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "qwen3.6-35b-a3b"
    weights: VlmWeights = VlmWeights()
    timeout_s: float = 900
    max_backoff_s: float = 300


class ProfileConfig(_Block):
    prompt: str
    image_long_side: int
    max_tokens: int = 6000
    temperature: float = 0.0
    top_p: float = 1.0
    top_logprobs: int = 1
    # Repetition penalties punish every token that already occurred in the answer; on a page
    # of text they replace repeated words and digits. They are sent with every request so the
    # defaults of llama-server (models.ini) or of a proxy cannot change the recognition.
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repeat_penalty: float = 1.0  # llama.cpp extension, 1.0 means off
    split: Literal["auto", "never"] | None = None


def _default_profiles() -> dict[str, ProfileConfig]:
    return {
        "pass1": ProfileConfig(prompt="transcribe_v1", image_long_side=2000),
        "pass2": ProfileConfig(prompt="transcribe_v1", image_long_side=2800, split="auto"),
        "region": ProfileConfig(prompt="fragment_v1", image_long_side=2800, max_tokens=3000),
    }


class ChecksConfig(_Block):
    empty_min_chars: int = 50
    line_abs_tolerance: int = 3
    line_ratio_tolerance: float = 0.15
    low_conf_token_p: float = 0.6
    digit_token_p: float = 0.9
    split_min_lines: int = 40
    split_gap_max_ink: float = 0.001

    @field_validator("low_conf_token_p", "digit_token_p")
    @classmethod
    def _token_p_limit(cls, value: float) -> float:
        if not 0 < value <= MAX_TOKEN_P:
            raise ValueError(f"token probability threshold must be in (0, {MAX_TOKEN_P}]")
        return value


def _all_flags() -> list[FlagCode]:
    return list(FlagCode)


class DecideConfig(_Block):
    candidate: Literal["pass1", "pass2"] = "pass2"
    flags: list[FlagCode] = Field(default_factory=_all_flags)
    hard_markers: list[str] = Field(
        default_factory=lambda: [
            "неразб.",
            "рук.: неразб.",
            "таблица: проверить",
            "формула: проверить",
            "обрезано",
        ]
    )

    @field_validator("flags")
    @classmethod
    def _mandatory_flags(cls, value: list[FlagCode]) -> list[FlagCode]:
        missing = MANDATORY_FLAGS - set(value)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"decide.flags must include {names}")
        return value


class SignalsConfig(_Block):
    min_diff_words: int = 4
    min_low_digits: int = 1
    min_low_letters: int = 5


class HintsConfig(_Block):
    merge_gap_words: int = 3


class FastTrackConfig(_Block):
    enabled: bool = True
    max_image_area: float = 0.05
    min_chars: int = 20
    max_bad_char_ratio: float = 0.01
    math_font_hints: list[str] = Field(
        default_factory=lambda: ["Math", "MT Extra", "Euclid", "CMMI", "CMSY", "CMEX"]
    )


class ReviewConfig(_Block):
    page_long_side: int = 2000
    image_format: Literal["jpeg", "png"] = "jpeg"
    jpeg_quality: int = 85
    load_guideline: float = 0.10
    stale_draft_days: int = 3
    default_page_review_s: float = 60


class WorkerConfig(_Block):
    page_concurrency: int = 1
    poll_interval_s: float = 5


class RenderCacheConfig(_Block):
    max_gb: float = 5


class BuildConfig(_Block):
    pandoc_path: str = "pandoc"
    reference_docx: str = "ocr_archive/assets/reference.docx"
    lua_filter: str = "ocr_archive/assets/html_tables.lua"
    font_path: str = "ocr_archive/assets/fonts/DejaVuSans.ttf"
    docx_min_similarity: float = 0.99
    keep_versions: int = 2
    keep_drafts: int = 1


class ChunksConfig(_Block):
    max_chars: int = 1200


class IngesterConfig(_Block):
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8000"
    collection: str = "archive"
    timeout_s: float = 600
    extra_payload: dict[str, Any] = Field(default_factory=lambda: {"dataset": "ocr-archive"})
    api_key: SecretStr = SecretStr("")

    @field_validator("extra_payload")
    @classmethod
    def _no_reserved_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        reserved = RESERVED_PAYLOAD_KEYS & set(value)
        if reserved:
            raise ValueError(f"extra_payload must not contain {', '.join(sorted(reserved))}")
        return value


class Settings(BaseSettings):
    """Process configuration. Secrets come only from the environment."""

    model_config = SettingsConfigDict(extra="forbid", frozen=True, case_sensitive=False)

    data_dir: Path = Path("/data")
    db_path: Path | None = None
    export_dir: Path = Path("/export")
    max_upload_mb: int = 1024

    api: ApiConfig = ApiConfig()
    vlm: VlmConfig = VlmConfig()
    profiles: dict[str, ProfileConfig] = Field(default_factory=_default_profiles)
    checks: ChecksConfig = ChecksConfig()
    decide: DecideConfig = DecideConfig()
    signals: SignalsConfig = SignalsConfig()
    hints: HintsConfig = HintsConfig()
    fast_track: FastTrackConfig = FastTrackConfig()
    review: ReviewConfig = ReviewConfig()
    worker: WorkerConfig = WorkerConfig()
    render_cache: RenderCacheConfig = RenderCacheConfig()
    build: BuildConfig = BuildConfig()
    chunks: ChunksConfig = ChunksConfig()
    ingester: IngesterConfig = IngesterConfig()

    ocr_api_key: SecretStr = SecretStr("")
    llama_api_key: SecretStr = SecretStr("")
    ingester_api_key: SecretStr = SecretStr("")

    @model_validator(mode="after")
    def _required_profiles(self) -> Settings:
        missing = {"pass1", "pass2", "region"} - set(self.profiles)
        if missing:
            raise ValueError(f"profiles must define {', '.join(sorted(missing))}")
        return self

    @property
    def database_path(self) -> Path:
        """Path of state.db."""
        return self.db_path if self.db_path is not None else self.data_dir / "state.db"

    @property
    def sources_dir(self) -> Path:
        return self.data_dir / "sources"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def lock_path(self) -> Path:
        return self.data_dir / "worker.lock"

    @property
    def fingerprint_path(self) -> Path:
        return self.data_dir / "fingerprint.json"


def resolve_asset(path: str) -> Path:
    """Resolve a configured asset path: absolute, relative to the working directory, or packaged.

    :param path: value from the ``build`` block, e.g. ``ocr_archive/assets/reference.docx``.
    :return: existing path if found, otherwise the path inside the package.
    """
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    packaged = PACKAGE_DIR.parent / candidate
    if packaged.exists():
        return packaged
    prefix = "ocr_archive/"
    if path.startswith(prefix):
        return PACKAGE_DIR / path[len(prefix) :]
    return candidate


def default_config_path() -> Path | None:
    """Return the config file the processes use: $OCR_CONFIG, /app/config.yaml or ./config.yaml."""
    env_path = os.environ.get("OCR_CONFIG")
    if env_path:
        return Path(env_path)
    for candidate in (Path("/app/config.yaml"), Path("config.yaml")):
        if candidate.exists():
            return candidate
    return None


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from YAML; secrets are read from the environment.

    :param path: config file; defaults to :func:`default_config_path`.
    :raises ConfigError: when the file cannot be read or fails validation.
    """
    config_path = path if path is not None else default_config_path()
    data: dict[str, Any] = {}
    if config_path is not None:
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as err:
            raise ConfigError(f"cannot read config {config_path}: {err}") from err
        if loaded is not None and not isinstance(loaded, dict):
            raise ConfigError(f"config {config_path} must be a mapping")
        data = loaded or {}
    # Secrets in the YAML file are ignored on purpose: they come only from the environment.
    for secret in ("ocr_api_key", "llama_api_key", "ingester_api_key"):
        data.pop(secret, None)
    if isinstance(data.get("ingester"), dict):
        data["ingester"].pop("api_key", None)
    try:
        settings = Settings(**data)
    except ValueError as err:
        raise ConfigError(f"invalid config: {err}") from err
    if settings.ingester_api_key.get_secret_value() and not (
        settings.ingester.api_key.get_secret_value()
    ):
        ingester = settings.ingester.model_copy(update={"api_key": settings.ingester_api_key})
        settings = settings.model_copy(update={"ingester": ingester})
    return settings
