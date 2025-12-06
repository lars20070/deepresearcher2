ARG PYTHON_VERSION=3.12
FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VERSION}-trixie

# Copy uv and uvx (pinned version)
COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /uvx /bin/

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Enable Docker BuildKit for cache mounts
ENV DOCKER_BUILDKIT=1

# Install system dependencies with cache mount for faster rebuilds (pinned versions)
RUN --mount=type=cache,target=/var/cache/apt \
  --mount=type=cache,target=/var/lib/apt/lists \
  apt-get update \
  && apt-get install -y --no-install-recommends \
  cmake=3.31.6-2 \
  ninja-build=1.12.1-1 \
  libclang-dev=1:19.0-63 \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI
# The official install script usually detects the architecture (ARM64 vs AMD64) automatically.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -fsSL https://ollama.com/install.sh | sh

# The vscode user already exists in the base image with UID 1000.
USER vscode

WORKDIR /workspaces/deepresearcher2
