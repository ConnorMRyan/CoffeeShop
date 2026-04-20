# ── CoffeeShop MARL — GCP Training Container ─────────────────────────────────
#
# Build:
#   docker build -t coffeeshop-train .
#
# Run locally:
#   docker run --gpus all coffeeshop-train env=nethack agent=ppo
#
# Run on GCP (pass GCS bucket for checkpoint sync):
#   docker run --gpus all \
#     -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json \
#     -v /path/to/sa.json:/secrets/sa.json:ro \
#     coffeeshop-train env=nethack gcs.bucket=my-bucket run.steps=1000000000

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ── System dependencies ───────────────────────────────────────────────────────
# NLE (NetHack Learning Environment) compiles a NetHack binary at pip-install
# time and needs cmake, bison, flex, and ncurses headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        bison \
        flex \
        make \
        libncurses5-dev \
        libncursesw5-dev \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only the dependency manifest first so this layer is cached as long as
# requirements.txt doesn't change (source edits won't bust the cache).
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# GCS client — installed separately so it can be omitted for non-GCP images
# without changing requirements.txt.
RUN pip install --no-cache-dir google-cloud-storage

# ── Project source ────────────────────────────────────────────────────────────
COPY . .

# Install the project itself without re-installing deps from PyPI.
RUN pip install --no-deps -e .

# Hydra resolves config paths relative to the working directory.
ENV PYTHONPATH=/workspace

# ── Entrypoint ────────────────────────────────────────────────────────────────
# All Hydra overrides are passed as extra arguments:
#   docker run ... coffeeshop-train env=nethack gcs.bucket=my-bucket
ENTRYPOINT ["python", "scripts/train.py"]
