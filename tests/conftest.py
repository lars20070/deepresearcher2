#!/usr/bin/env python3

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "paid: tests requiring paid API keys")
    config.addinivalue_line("markers", "ollama: tests requiring a local Ollama instance")
    config.addinivalue_line("markers", "example: examples which are not testing deepresearcher2 functionality")


@pytest.fixture(autouse=True)
def skip_ollama_tests(request: pytest.FixtureRequest) -> None:
    """
    Skip tests marked with 'ollama' when running in CI environment.
    Run these tests only locally.
    """
    if request.node.get_closest_marker("ollama") and os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("Tests requiring Ollama skipped in CI environment")


@pytest.fixture
def topic() -> str:
    """Provide a research topic for unit testing."""
    return "petrichor"
