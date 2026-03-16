# Base image: CUDA 12.8 + cuDNN 9 (matches torch 2.9.1 cu128)
# Must use devel (not runtime): torch.compile generates CUDA kernels via Triton at runtime
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 (ships with Ubuntu 22.04) and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy only dependency files first (cache-friendly)
WORKDIR /build
COPY pyproject.toml uv.lock ./

# Install dependencies with GPU (CUDA 12.8) extras into a venv
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv sync --extra gpu --no-dev --no-install-project

# Activate venv: put it first on PATH and symlink python for compatibility
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
# Ensure `python` command exists (Ubuntu 22.04 only ships `python3`)
RUN ln -sf /opt/venv/bin/python3 /opt/venv/bin/python 2>/dev/null || true

# Verify key dependencies are installed
RUN python -c "import torch; import requests; import mlflow; print(f'torch={torch.__version__}, requests OK, mlflow={mlflow.__version__}')"

WORKDIR /workspace
