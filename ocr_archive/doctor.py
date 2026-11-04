"""Environment checks of ``ocr doctor`` (section 17)."""

from __future__ import annotations

import asyncio
import io
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

from ocr_archive.db.connection import SCHEMA_VERSION, open_db, schema_version
from ocr_archive.exceptions import IngestError, VlmRequestError, VlmTransportError
from ocr_archive.ingester_client import ingester_health
from ocr_archive.settings import PACKAGE_DIR, Settings, resolve_asset
from ocr_archive.vlm.client import VlmClient
from ocr_archive.vlm.logprobs import token_stats

ASSETS = PACKAGE_DIR / "assets"
SAMPLE_PNG = ASSETS / "doctor_sample.png"
SAMPLE_TXT = ASSETS / "doctor_sample.txt"
MIN_PANDOC = (3, 4)
REQUIRED_STYLES = ("Source Page", "Uncertain", "Unverified", "Draft Banner")
FINGERPRINT_TOKENS = 50
FINGERPRINT_TOLERANCE = 0.05
MIN_WORD_SHARE = 0.8
NETWORK_FS = ("nfs", "nfs4", "cifs", "smb3", "smbfs")
HOST_SHARE_FS = ("9p", "fakeowner", "virtiofs")


class Level(StrEnum):
    OK = "OK"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Check:
    name: str
    level: Level
    message: str


def _fs_type(path: Path) -> str | None:
    mounts = Path("/proc/mounts")
    if not mounts.exists():
        return None
    best, best_type = "", None
    resolved = str(path.resolve())
    for line in mounts.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if (
            len(parts) >= 3
            and (resolved == parts[1] or resolved.startswith(parts[1].rstrip("/") + "/"))
            and len(parts[1]) > len(best)
        ):
            best, best_type = parts[1], parts[2]
    return best_type


def check_storage(settings: Settings) -> list[Check]:
    """Checks 1–3: data directory, database, file system, flock, export folder."""
    checks: list[Check] = []
    data = settings.data_dir
    try:
        data.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=data, delete=True):
            pass
        conn = open_db(settings.database_path)
        version = schema_version(conn)
        conn.close()
        level = Level.OK if version == SCHEMA_VERSION else Level.FAIL
        checks.append(Check("data", level, f"{data} writable, schema version {version}"))
    except (OSError, ValueError) as err:
        checks.append(Check("data", Level.FAIL, f"{data}: {err}"))
        return checks
    fs = _fs_type(data)
    if fs is None:
        checks.append(Check("filesystem", Level.WARNING, "cannot read /proc/mounts"))
    elif fs.split(".")[0] in NETWORK_FS or fs in NETWORK_FS:
        checks.append(Check("filesystem", Level.FAIL, f"{fs}: network file system, no flock"))
    elif fs in HOST_SHARE_FS or fs.startswith("fuse."):
        checks.append(
            Check(
                "filesystem",
                Level.WARNING,
                f"{fs}: Docker Desktop host directory, locks not guaranteed; use a named volume",
            )
        )
    else:
        checks.append(Check("filesystem", Level.OK, fs))
    try:
        import fcntl

        with (data / ".doctor.lock").open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        checks.append(Check("flock", Level.OK, "lock taken and released"))
    except ImportError:
        checks.append(Check("flock", Level.WARNING, "fcntl unavailable (not Linux)"))
    except OSError as err:
        checks.append(Check("flock", Level.FAIL, str(err)))
    try:
        settings.export_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=settings.export_dir, prefix=".doctor", delete=True):
            pass
        checks.append(Check("export", Level.OK, f"{settings.export_dir} writable"))
    except OSError as err:
        checks.append(Check("export", Level.FAIL, f"{settings.export_dir}: {err}"))
    return checks


def _expected_words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _degrade(png: bytes) -> bytes:
    with Image.open(io.BytesIO(png)) as img:
        blurred = img.convert("RGB").filter(ImageFilter.GaussianBlur(1.6))
        noise = Image.effect_noise(blurred.size, 40).convert("RGB")
        mixed = Image.blend(blurred, noise, 0.25)
        out = io.BytesIO()
        mixed.save(out, format="PNG")
        return out.getvalue()


async def check_model(settings: Settings, *, record_fingerprint: bool) -> list[Check]:
    """Checks 4–10: model id, weights, vision, thinking, logprobs, fingerprint."""
    checks: list[Check] = []
    client = VlmClient(settings.vlm, settings.llama_api_key.get_secret_value())
    try:
        try:
            models = await client.models()
        except VlmTransportError as err:
            return [Check("vlm", Level.FAIL, str(err))]
        if settings.vlm.model in models:
            checks.append(Check("vlm", Level.OK, f"model {settings.vlm.model}"))
        else:
            checks.append(Check("vlm", Level.FAIL, f"{settings.vlm.model} not in {models}"))
        props = await client.props()
        model_path = str((props or {}).get("model_path") or "")
        expected = settings.vlm.weights.gguf_file
        if model_path and expected and Path(model_path).name == expected:
            checks.append(Check("weights", Level.OK, expected))
        else:
            checks.append(Check("weights", Level.WARNING, "could not verify the weights file"))

        png = SAMPLE_PNG.read_bytes()
        expected_words = _expected_words(SAMPLE_TXT.read_text(encoding="utf-8"))
        profile = settings.profiles["pass1"]
        body = client.request_body(png, profile, user_text="Страница 1.")
        try:
            raw = await client.chat(body)
            clean = await client.transcribe(png, profile, user_text="Страница 1.")
            degraded = await client.transcribe(_degrade(png), profile, user_text="Страница 1.")
        except (VlmTransportError, VlmRequestError) as err:
            checks.append(Check("vision", Level.FAIL, str(err)))
            return checks
        found = set(_expected_words(clean.text))
        share = sum(1 for w in expected_words if w in found) / max(1, len(expected_words))
        checks.append(
            Check(
                "vision",
                Level.OK if share >= MIN_WORD_SHARE else Level.FAIL,
                f"{share:.0%} of expected words",
            )
        )
        message = (raw.get("choices") or [{}])[0].get("message") or {}
        thinking = bool(message.get("reasoning_content")) or "<think>" in str(
            message.get("content")
        )
        checks.append(
            Check(
                "thinking",
                Level.FAIL if thinking else Level.OK,
                "reasoning present" if thinking else "disabled",
            )
        )
        if clean.token_probs is None:
            checks.append(
                Check(
                    "logprobs",
                    Level.WARNING,
                    "without logprobs uncertainty signals are not computed; flags use "
                    "the other rules",
                )
            )
        else:
            checks.append(Check("logprobs", Level.OK, "present and consistent with the text"))
            if degraded.token_probs is not None:
                good = token_stats(clean.token_probs, settings.checks)
                bad = token_stats(degraded.token_probs, settings.checks)
                useful = (
                    bad.letter_min_p < good.letter_min_p
                    and bad.digit_min_p <= good.digit_min_p
                    and bad.letter_low >= good.letter_low
                    and bad.digit_low >= good.digit_low
                )
                checks.append(
                    Check(
                        "logprobs_useful",
                        Level.OK if useful else Level.WARNING,
                        "degraded sample is less certain"
                        if useful
                        else "degraded sample is not less certain",
                    )
                )
        checks.append(_fingerprint(settings, raw, clean.text, record=record_fingerprint))
    finally:
        await client.aclose()
    return checks


def _first_probs(raw: dict[str, Any]) -> list[float]:
    import math

    choice = (raw.get("choices") or [{}])[0]
    entries = ((choice.get("logprobs") or {}).get("content")) or []
    return [round(math.exp(float(e.get("logprob", 0.0))), 4) for e in entries[:FINGERPRINT_TOKENS]]


def _fingerprint(settings: Settings, raw: dict[str, Any], text: str, *, record: bool) -> Check:
    path = settings.fingerprint_path
    current = {"text": text, "probs": _first_probs(raw)}
    if record:
        path.write_text(json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8")
        return Check("fingerprint", Level.OK, f"recorded to {path}")
    if not path.exists():
        return Check(
            "fingerprint", Level.WARNING, "no fingerprint; run ocr doctor --record-fingerprint"
        )
    stored = json.loads(path.read_text(encoding="utf-8"))
    same_text = stored.get("text") == text
    probs_a, probs_b = stored.get("probs", []), current["probs"]
    same_probs = len(probs_a) == len(probs_b) and all(
        abs(a - b) <= FINGERPRINT_TOLERANCE for a, b in zip(probs_a, probs_b, strict=True)
    )
    if same_text and same_probs:
        return Check("fingerprint", Level.OK, "matches")
    return Check(
        "fingerprint",
        Level.WARNING,
        "model, weights or server settings changed; if unintended, check before "
        "running the archive",
    )


def check_build_tools(settings: Settings) -> list[Check]:
    """Check 11: pandoc version, reference.docx styles, Lua filter and font."""
    checks: list[Check] = []
    pandoc = shutil.which(settings.build.pandoc_path)
    if pandoc is None:
        checks.append(Check("pandoc", Level.FAIL, "pandoc not found"))
    else:
        out = subprocess.run([pandoc, "--version"], capture_output=True, text=True, check=False)
        match = re.search(r"pandoc(?:\.exe)? (\d+)\.(\d+)", out.stdout)
        version = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
        checks.append(
            Check(
                "pandoc",
                Level.OK if version >= MIN_PANDOC else Level.FAIL,
                f"version {version[0]}.{version[1]}",
            )
        )
    reference = resolve_asset(settings.build.reference_docx)
    try:
        with zipfile.ZipFile(reference) as archive:
            styles = archive.read("word/styles.xml").decode("utf-8")
        missing = [s for s in REQUIRED_STYLES if f'w:val="{s}"' not in styles]
        checks.append(
            Check(
                "reference.docx",
                Level.FAIL if missing else Level.OK,
                f"missing styles {missing}" if missing else "styles present",
            )
        )
    except (OSError, KeyError, zipfile.BadZipFile) as err:
        checks.append(Check("reference.docx", Level.FAIL, str(err)))
    for name, value in (
        ("lua_filter", settings.build.lua_filter),
        ("font", settings.build.font_path),
    ):
        path = resolve_asset(value)
        checks.append(Check(name, Level.OK if path.exists() else Level.FAIL, str(path)))
    return checks


async def check_ingester(settings: Settings) -> list[Check]:
    """Check 12: ingester health and allowed collection, only when enabled."""
    if not settings.ingester.enabled:
        return [Check("ingester", Level.OK, "disabled")]
    try:
        health = await ingester_health(settings.ingester)
    except IngestError as err:
        return [Check("ingester", Level.FAIL, str(err))]
    allowed = health.get("allowed_collections") or health.get("collections")
    if isinstance(allowed, list) and settings.ingester.collection not in allowed:
        return [
            Check("ingester", Level.FAIL, f"collection {settings.ingester.collection} not allowed")
        ]
    return [Check("ingester", Level.OK, "healthy")]


def run_doctor(settings: Settings, *, record_fingerprint: bool = False) -> list[Check]:
    """Run all checks in order."""
    checks = check_storage(settings)

    async def _remote() -> list[Check]:
        return await check_model(
            settings, record_fingerprint=record_fingerprint
        ) + await check_ingester(settings)

    checks += asyncio.run(_remote())
    checks += check_build_tools(settings)
    return checks
