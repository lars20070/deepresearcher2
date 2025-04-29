#!/usr/bin/env python3

import os
from collections.abc import Generator

import pytest

from deepresearcher2.config import config


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


@pytest.fixture
def config_for_testing(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """
    Override the config for unit testing.
    """
    monkeypatch.setattr(config, "topic", "petrichor")
    monkeypatch.setattr(config, "max_research_loops", 3)
    monkeypatch.setattr(config, "max_web_search_results", 2)
    monkeypatch.setattr(config, "search_engine", "duckduckgo")
    monkeypatch.setattr(config, "reports_folder", "tests/reports/")
    monkeypatch.setattr(config, "logs2logfire", False)

    yield
