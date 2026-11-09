# Paradee TTS API — OpenAI-compatible speech synthesis on an 8 MB CPU model.
#
# Two stages: a builder that may use git/ca-certificates to fetch the upstream
# package and the weights, and a runtime that has no compilers, no git and no
# pip cache. The weights are verified by sha256 and the build fails unless the
# model loads and speaks with the Hugging Face hub disabled.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# git is needed only here: paradee has no PyPI release and is installed from a
# pinned commit. ca-certificates is needed for the wheel downloads. curl is what
# scripts/fetch_models.sh uses to pull the pinned weights — python:3.12-slim
# ships neither curl nor wget, so the build fails with exit 127 without it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Pinned upstream commit (tag v1.2.1). There is no PyPI package for paradee.
RUN pip install "paradee @ git+https://github.com/sahilmahendrakar/paradee@c2789ef9abf54a9384eef89cc7de3f1afe5ad01d"

# misaki calls spacy.cli.download("en_core_web_sm") when the model is absent,
# which would be a runtime network call. Install it as a package instead.
RUN pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

COPY scripts/fetch_models.sh /build/scripts/fetch_models.sh
RUN MODELS_DIR=/build/models bash /build/scripts/fetch_models.sh

# Build gate: loads the baked weights with hf_hub_download disabled, speaks all
# four voices, proves British phonemisation and the eSpeak fallback work, and
# checks the WAV + timestamp contracts. A packaging regression fails the build.
COPY app/ /build/app/
COPY scripts/warmup.py /build/scripts/warmup.py
RUN PARADEE_MODEL_PATH=/build/models/paradee_int8.onnx \
    PARADEE_CONFIG_PATH=/build/models/config.json \
    HF_HUB_OFFLINE=1 \
    python /build/scripts/warmup.py


FROM python:3.12-slim AS runtime

# PORT is only the local default: Railway injects its own at runtime (8080
# observed), and the CMD reads it. The model files are inside the image, so
# the process must never reach the network — HF_HUB_OFFLINE=1 makes an
# accidental fetch fail loudly instead of silently downloading.
# OMP_NUM_THREADS=1 keeps latency predictable on a shared vCPU, and matches
# the single-thread basis of the vendor's speed figure.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PARADEE_MODEL_PATH=/app/models/paradee_int8.onnx \
    PARADEE_CONFIG_PATH=/app/models/config.json \
    OMP_NUM_THREADS=1 \
    HOME=/home/app

# espeakng-loader ships its own libespeak-ng.so and 18 MB of data, but the
# library links libstdc++/libgcc, and onnxruntime needs libgomp on some bases.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 libgcc-s1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY --from=builder /build/models/ /app/models/
COPY app/ /app/app/

RUN useradd --create-home --uid 10001 app \
    && chown -R app:app /app /home/app

USER app
EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1"]
