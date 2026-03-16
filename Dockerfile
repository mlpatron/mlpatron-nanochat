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

# Install dependencies into default .venv (uv sync expects this path)
RUN uv venv && uv sync --extra gpu --no-dev --no-install-project

# Put venv on PATH; create `python` command (Ubuntu 22.04 only has `python3`)
ENV PATH="/build/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/build/.venv"
RUN ln -sf /usr/bin/python3 /build/.venv/bin/python

# Ensure PyTorch's bundled CUDA libs are on LD_LIBRARY_PATH (needed by torch.compile / Triton)
ENV LD_LIBRARY_PATH="/build/.venv/lib/python3.10/site-packages/torch/lib:/build/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"

# Verify key dependencies are installed
RUN /build/.venv/bin/python -c "import torch; import requests; import mlflow; print(f'torch={torch.__version__}, cuda={torch.version.cuda}, requests OK, mlflow={mlflow.__version__}')"

WORKDIR /workspace
