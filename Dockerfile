# ocr-archive: one image for the api and worker containers (section 12.4).
FROM python:3.12-slim

ARG PANDOC_VERSION=3.5
ARG PANDOC_SHA256=""
ARG TARGETARCH=amd64

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OCR_CONFIG=/app/config.yaml

# pandoc from the official .deb, pinned: the DOCX writer needs >= 3.4 for merged cells.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL -o /tmp/pandoc.deb \
      "https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-1-${TARGETARCH}.deb" \
 && if [ -n "${PANDOC_SHA256}" ]; then echo "${PANDOC_SHA256}  /tmp/pandoc.deb" | sha256sum -c -; fi \
 && apt-get install -y --no-install-recommends /tmp/pandoc.deb \
 && rm /tmp/pandoc.deb \
 && apt-get purge -y curl && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY ocr_archive ./ocr_archive
RUN pip install --no-cache-dir . \
 && useradd --uid 10001 --create-home --shell /usr/sbin/nologin ocr \
 && mkdir -p /data /inbox /export \
 && chown ocr:ocr /data /export

USER ocr
CMD ["ocr", "--help"]
