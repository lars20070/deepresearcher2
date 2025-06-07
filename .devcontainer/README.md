## Dev Container

The development container image is based on Debian 11 (Bullseye). The container provides a standard development environment.

The container also provides a secure sandbox environment for coding agents. Coding agents such as `Claude Code` might have read-write access to the entire file system. The dev container prevents the coding agent from messing up the host system.

home directory: `/home/vscode`<br>
project directory: `/workspaces/deepresearcher2`

## Docker

The setup has been tested with Docker Desktop on macOS. Adjust memory and disk usage in `Settings` -> `Resources` -> `Advanced` if necessary.

### Ollama

The project relies on local LLMs running in `ollama`. The Ollama CLI is installed in the container via `features`.

In order to avoid pulling models at container startup, we mount the `~/.ollama` folder of the host system into the container. Make sure the local Ollama installation exists.

The Ollama installation on the host system is GPU accelerated. The Ollama installation in the container is not. For that reason, we use the Ollama installation on the host system as external Ollama server by setting the `OLLAMA_HOST` environment variable. Strictly speaking, mounting of the external `~/.ollama` folder is therefore not necessary.