#!/usr/bin/env python3
import pytest
from _pytest.config import Config, PytestPluginManager
from _pytest.config.argparsing import Parser
from _pytest.main import Session
from _pytest.nodes import Item
from _pytest.runner import CallInfo

from .logger import logger

ASSAY_MODES = ("evaluate", "new_baseline")


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
    # config.addinivalue_line("markers", "vcr: Mark the test as using VCR.py.")
    # config.addinivalue_line("markers", "block_network: Block network access except for VCR recording.")
    # config.addinivalue_line("markers", "default_cassette: Override the default cassette name.")
    # config.addinivalue_line(
    #     "markers",
    #     "allowed_hosts: List of regexes to match hosts to where connection must be allowed.",
    # )
    # network.install_pycurl_wrapper()

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


@pytest.hookimpl(tryfirst=True)  # Executed before other hooks. Important for non-None return values.
def pytest_runtest_makereport(item: Item, call: CallInfo) -> None:
    """
    Hook to process test reports.

    Args:
        item (Item): The test item.
        call (CallInfo): Information about the test call.
    """
    # pytest_runtest_makereport is called three times per test: setup, call, teardown
    # Here, we are interested in the "call" phase.
    # Use setup and teardown to report when a fixture or cleanup fails.
    if call.when == "call":
        outcome = call.excinfo

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

        except Exception as e:
            logger.error("ERROR:", e)
