# docker/Dockerfile
# Base: NVIDIA CUDA 11.8 + cuDNN 8 + Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# ── System deps ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl build-essential ninja-build \
    libopenblas-dev dos2unix \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3       1

# ── Install uv ────────────────────────────────────────────────────────────────
RUN pip install uv

# ── Heavy deps first (better layer caching) ───────────────────────────────────
WORKDIR /workspace

# Step 1: Pre-install torch (CUDA 11.8) and build tools
RUN uv pip install --system \
    torch==2.5.0 torchvision==0.20.0 \
    --index-url https://download.pytorch.org/whl/cu118
RUN uv pip install --system setuptools wheel packaging ninja hatchling editables

# Step 2: Build causal-conv1d and mamba-ssm from source (--no-deps avoids conflicts)
#   MAMBA_FORCE_BUILD=TRUE is required by official docs for Mamba-3 support
ENV MAX_JOBS=2 MAMBA_FORCE_BUILD=TRUE
RUN uv pip install --system --no-build-isolation --no-deps \
    "causal-conv1d>=1.4.0"
RUN uv pip install --system --no-build-isolation --no-deps \
    "mamba-ssm @ git+https://github.com/state-spaces/mamba.git"


# ── Project code & remaining deps ─────────────────────────────────────────────
COPY . /workspace/

# Fix Windows line endings on shell scripts (critical for Windows → Linux Docker)
RUN find /workspace -name "*.sh" -exec dos2unix {} \;

# Step 3: Install remaining project dependencies and edgemamba3 itself
RUN uv pip install --system --no-build-isolation -e .

# Step 4: Upgrade triton AFTER project install (Mamba-3 requires triton.set_allocator)
#   Must come last because uv's resolver downgrades triton to 3.1.0 (torch==2.5.0 pin)
RUN uv pip install --system --no-deps "triton>=3.5.0"

# ── Entrypoint ────────────────────────────────────────────────────────────────
RUN chmod +x /workspace/docker/entrypoint.sh
ENTRYPOINT ["/bin/bash", "/workspace/docker/entrypoint.sh"]
