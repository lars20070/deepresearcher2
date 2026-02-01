ARG PYTHON_VERSION=3.12
FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VERSION}-trixie

# Copy uv and uvx (latest version)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Enable Docker BuildKit for cache mounts
ENV DOCKER_BUILDKIT=1

# Set shell with pipefail for all RUN commands
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies with cache mount for faster rebuilds (latest versions)
# Trixie is a testing distribution. Certain versions of packages might disappear from mirrors. Therefore we do not pin versions here.
# Remove yarn source list for now (GPG key missing, causes apt-get update to fail)
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt \
  --mount=type=cache,target=/var/lib/apt/lists \
  rm -f /etc/apt/sources.list.d/yarn.list \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
  cmake \
  ninja-build \
  libclang-dev \
  zstd \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

# Install Ollama CLI
# The official install script usually detects the architecture (ARM64 vs AMD64) automatically.
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Claude Code CLI
# Must be installed as non-root user
USER vscode
RUN curl -fsSL https://claude.ai/install.sh | bash
USER root

# The vscode user already exists in the base image with UID 1000.
USER vscode

WORKDIR /workspaces/deepresearcher2
