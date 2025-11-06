#!/usr/bin/env python3
import glob
import os
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from vcr.request import Request

from deepresearcher2.config import Model, SearchEngine, config
from deepresearcher2.evals.evals import EvalGame, EvalPlayer
from deepresearcher2.logger import logger


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "paid: tests requiring paid API keys")
    config.addinivalue_line("markers", "ollama: tests requiring a local Ollama instance")
    config.addinivalue_line("markers", "searxng: tests requiring a local SearXNG instance")
    config.addinivalue_line("markers", "example: examples which are not testing deepresearcher2 functionality")


@pytest.fixture(autouse=True)
def skip_ollama_tests(request: pytest.FixtureRequest) -> None:
    """
    Skip tests marked with 'ollama' when running in CI environment.
    Run these tests only locally.
    """
    if request.node.get_closest_marker("ollama") and os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("Tests requiring Ollama skipped in CI environment")


@pytest.fixture(autouse=True)
def skip_searxng_tests(request: pytest.FixtureRequest) -> None:
    """
    Skip tests marked with 'searxng' when running in CI environment.
    Run these tests only locally.
    """
    if request.node.get_closest_marker("searxng") and os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("Tests requiring SearXNG skipped in CI environment")


@pytest.fixture
def topic() -> str:
    """
    Provide a research topic for unit testing.
    """
    return "petrichor"


@pytest.fixture(autouse=True)
def config_for_testing(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """
    Override the config for unit testing.
    """
    monkeypatch.setattr(config, "topic", "petrichor")
    monkeypatch.setattr(config, "max_research_loops", 3)
    monkeypatch.setattr(config, "max_web_search_results", 2)
    monkeypatch.setattr(config, "search_engine", SearchEngine.serper)
    monkeypatch.setattr(config, "model", Model.llama33)
    monkeypatch.setattr(config, "model_timeout", 600)
    monkeypatch.setattr(config, "reports_folder", "tests/reports/")
    monkeypatch.setattr(config, "logs2logfire", False)

    yield


@pytest.fixture
def cleanup_reports_folder(config_for_testing: Generator[None, None, None]) -> None:
    """
    Remove all old report files
    """
    for pattern in ["*.md", "*.pdf"]:
        for f in glob.glob(os.path.join(config.reports_folder, pattern)):
            try:
                os.remove(f)
                logger.debug(f"Removed file {f}")
            except OSError as e:
                logger.error(f"Error removing file {f}: {e}")


@pytest.fixture
def ice_cream_players() -> list[EvalPlayer]:
    """
    Provide a list of EvalPlayer instances with ice cream flavours.
    """
    return [
        EvalPlayer(idx=0, item="vanilla"),
        EvalPlayer(idx=1, item="chocolate"),
        EvalPlayer(idx=2, item="strawberry"),
        EvalPlayer(idx=3, item="peach"),
        EvalPlayer(idx=4, item="toasted rice & miso caramel ice cream"),
    ]


@pytest.fixture
def ice_cream_game() -> EvalGame:
    """
    Provide an EvalGame instance for ice cream flavour comparison.
    """
    return EvalGame(criterion="Which of the two ice cream flavours A or B is more creative?")


@pytest.fixture
def mock_fetch_full_page_content(mocker: MockerFixture) -> MagicMock:
    """
    Mocks the fetch_full_page_content function.
    """
    return mocker.patch(
        "deepresearcher2.utils.fetch_full_page_content",
        return_value="Mocked long page content for testing purposes.",
    )


@pytest.fixture
def vcr_config() -> dict[str, object]:
    """
    Configure VCR recordings for tests with @pytest.mark.vcr() decorator.

    When on bare metal, our host is `localhost`. When in a dev container, our host is `host.docker.internal`.
    `uri_spoofing` ensures that VCR cassettes are read or recorded as if the host was `localhost`.
    See ./tests/cassettes/*/*.yaml.

    Returns:
        dict[str, object]: VCR configuration settings.
    """

    def uri_spoofing(request: Request) -> Request:
        if request.uri and "host.docker.internal" in request.uri:
            # Replace host.docker.internal with localhost.
            request.uri = request.uri.replace("host.docker.internal", "localhost")
        return request

    return {
        "ignore_localhost": False,  # We want to record local SearXNG and Ollama requests.
        "filter_headers": ["authorization", "x-api-key"],
        "decode_compressed_response": True,
        "before_record_request": uri_spoofing,
    }
