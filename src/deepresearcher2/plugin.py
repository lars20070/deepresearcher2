#!/usr/bin/env python3
from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import BaseModel, Field
from pydantic_evals import Dataset
from pytest import CallInfo, Config, Item, Parser, PytestPluginManager, Session

from .evals.evals import EvalPlayer
from .logger import logger

ASSAY_MODES = ("evaluate", "new_baseline")

PLAYERS_KEY = pytest.StashKey[list[EvalPlayer]]()


def pytest_addoption(parser: Parser) -> None:
    group = parser.getgroup("recording")
    group.addoption(
        "--assay-mode",
        action="store",
        default="evaluate",
        choices=ASSAY_MODES,
        help='Assay mode. Defaults to "evaluate".',
    )


def pytest_configure(config: Config) -> None:
    """
    Configurations at the start of the test session.
    For example, add custom markers here.
    """
    # config.addinivalue_line("markers", "vcr: Mark the test as using VCR.py.")
    # config.addinivalue_line("markers", "block_network: Block network access except for VCR recording.")
    # config.addinivalue_line("markers", "default_cassette: Override the default cassette name.")
    # config.addinivalue_line(
    #     "markers",
    #     "allowed_hosts: List of regexes to match hosts to where connection must be allowed.",
    # )
    # network.install_pycurl_wrapper()

    # Add marker @pytest.mark.assay
    config.addinivalue_line("markers", "assay: Mark the test for AI agent evaluation i.e. running an assay.")

    assay_mode = config.getoption("--assay-mode")
    logger.debug(f"assay_mode={assay_mode}")

    pass


def pytest_unconfigure() -> None:
    pass


def pytest_addhooks(pluginmanager: PytestPluginManager) -> None:
    # pluginmanager.add_hookspecs(hooks)
    pass


def pytest_sessionstart(session: Session) -> None:
    """
    Session start hook.
    """
    logger.info("Hello from `pytest_sessionstart` hook!")


def pytest_sessionfinish(session: Session, exitstatus: int) -> None:
    """
    Session finish hook.
    """
    logger.info("Hello from `pytest_sessionfinish` hook!")
    logger.info(f"Exit status: {exitstatus}")


class AssayContext(BaseModel):
    """
    Context for assay execution.

    All data and metadata required to run an assay.
    """

    dataset: Dataset = Field(..., description="The evaluation dataset for this assay")
    path: Path = Field(..., description="File path where the assay dataset is stored")
    record_mode: str = Field(default="evaluate", description='Recording mode: "evaluate" or "new_baseline"')


def _assay_path(item: Item) -> Path:
    """
    Compute the assay file path from test module and function name.
    """
    path = Path(item.fspath)
    module_name = path.stem
    test_name = item.name.split("[")[0]
    return path.parent / "assays" / module_name / f"{test_name}.json"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: Item) -> Generator[None, None, None]:
    """
    Here we will inject the Dataset input.
    """

    # Execute the hook only for tests marked with @pytest.mark.assay
    marker = item.get_closest_marker("assay")
    if marker is None:
        yield
        return

    logger.debug("In pytest_runtest_call hook - before test execution")

    # Compute path
    assay_path = _assay_path(item)
    logger.debug(f"assay path: {assay_path}")

    yield

    logger.debug("In pytest_runtest_call hook - after test execution")


@pytest.hookimpl(tryfirst=True)  # Executed before other hooks. Important for non-None return values.
def pytest_runtest_makereport(item: Item, call: CallInfo) -> None:
    """
    Hook to process test reports.

    Run the Bradley-Terry tournament on the model outputs after each test here.

    Args:
        item (Item): The test item.
        call (CallInfo): Information about the test call.
    """
    # pytest_runtest_makereport is called three times per test: setup, call, teardown
    # Here, we are interested in the "call" phase.
    # Use setup and teardown to report when a fixture or cleanup fails.
    if call.when == "call":
        outcome = call.excinfo  # Contains exceptions i.e. None if test passed.

        try:
            # Access the test ID (nodeid)
            test_id = item.nodeid

            # Access the test outcome (passed, failed, etc.)
            test_outcome = "failed" if outcome else "passed"

            # Access the test duration
            test_duration = call.duration

            # Print Test Outcome and Duration
            logger.info(f"Test: {test_id}")
            logger.info(f"Test Outcome: {test_outcome}")
            logger.info(f"Test Duration: {test_duration:.5f} seconds")

            # Access baseline and novel players
            all_players = item.stash.get(PLAYERS_KEY, None)
            logger.debug(f"number of players: {len(all_players) if all_players is not None else 'None'}")

        except Exception as e:
            logger.error("ERROR:", e)
