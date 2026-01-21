#!/usr/bin/env python3
import contextvars
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_evals import Dataset
from pytest import CallInfo, Config, Function, Item, Parser, PytestPluginManager, Session

from .evals.evals import EvalPlayer
from .logger import logger

PLAYERS_KEY = pytest.StashKey[list[EvalPlayer]]()

# Modes for the assay plugin. "evaluate" is the default mode.
ASSAY_MODES = ("evaluate", "new_baseline")

# Key to stash Agent responses during assay tests
AGENT_RESPONSES_KEY = pytest.StashKey[list[AgentRunResult[Any]]]()

# Items stashed by the _wrapped_run wrapper. Required for async safety.
# _current_item_var defined at module level. But items are stored locally to the current execution context.
_current_item_var: contextvars.ContextVar[Item | None] = contextvars.ContextVar("_current_item", default=None)


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
    # config.addinivalue_line(
    #         "markers",
    #         "assay(generator=None, assay_mode='evaluate'): Mark the test for AI agent evaluation. "
    #         "Args: generator - callable returning Dataset; assay_mode - 'evaluate' or 'new_baseline'.",
    #     )

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
    assay_mode: str = Field(default="evaluate", description='Assay mode: "evaluate" or "new_baseline"')


def _path(item: Item) -> Path:
    """
    Compute the assay file path from test module and function name.
    """
    path = Path(item.fspath)
    module_name = path.stem
    test_name = item.name.split("[")[0]
    return path.parent / "assays" / module_name / f"{test_name}.json"


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: Item) -> None:
    """
    Here we will inject the Dataset input.
    """

    # Execute the hook only for tests marked with @pytest.mark.assay
    marker = item.get_closest_marker("assay")
    if marker is None:
        return

    # Only inject for Function items i.e. actual test functions
    # For example, if @pytest.mark.assay decorates a class, we skip it here.
    if not isinstance(item, Function):
        return

    # Get generator from marker kwargs
    generator = marker.kwargs.get("generator")

    logger.info("Populating assay context with dataset and path")
    path = _path(item)
    if path.exists():
        logger.info(f"Loading assay dataset from {path}")
        dataset = Dataset[dict[str, str], type[None], Any].from_file(path)
    elif generator is not None:
        logger.info("Generating new assay dataset using custom generator")
        dataset = generator()

        if not isinstance(dataset, Dataset):
            raise TypeError(f"The generator {generator} must return a Dataset instance.")

        logger.info(f"Serialising generated assay dataset to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_file(path, schema_path=None)
    else:
        logger.info("No existing assay dataset file or generator found; using empty dataset")
        dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    # Inject assay context into the test function arguments
    item.funcargs["assay"] = AssayContext(  # type: ignore[attr-defined]
        dataset=dataset,
        path=path,
        assay_mode=item.config.getoption("--assay-mode"),
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: Item) -> Generator[None, None, None]:
    """
    Intercepts `Agent.run()` calls during test execution to record model outputs.

    Mechanism:
    1. Setup: Temporarily replaces `Agent.run` with an instrumented wrapper.
    2. Capture: When called, the wrapper retrieves the current test `item` via a `ContextVar`
       and saves the result to `item.stash[AGENT_RESPONSES_KEY]`.
    3. Teardown: Restores the original `Agent.run` method.

    Why ContextVar?
    The `Agent.run` method signature cannot be modified to accept the test `item`.
    `ContextVar` acts as a thread-safe tunnel, allowing the wrapper to access the
    current test context implicitly without threading arguments through the call stack.
    """

    # Execute the hook only for tests marked with @pytest.mark.assay
    marker = item.get_closest_marker("assay")
    if marker is None:
        yield
        return

    logger.info("Inside pytest_runtest_call hook")

    # Initialize the stash for capturing responses
    item.stash[AGENT_RESPONSES_KEY] = []

    # Capture the current method state
    # Prevents infinite recursion in the wrapper
    original_run = Agent.run
    # Capture the item in the context variable
    token = _current_item_var.set(item)

    # Define the wrapper function
    async def _instrumented_agent_run(
        self: Agent[Any, Any],
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> AgentRunResult[Any]:
        """
        Wrapped Agent.run() that captures responses to the current test item's stash.
        """
        # Transparently await the original method
        result = await original_run(self, *args, **kwargs)

        # Capture logic using the ContextVar
        current_item = _current_item_var.get()
        if current_item is not None:
            responses = current_item.stash.get(AGENT_RESPONSES_KEY, [])
            responses.append(result)
            logger.debug(f"Captured Agent.run() response #{len(responses)}: {result.output!r:.100}")

        return result

    # Apply the patch
    Agent.run = _instrumented_agent_run  # type: ignore[method-assign]
    logger.debug("Monkeypatched Agent.run() for automatic response capture")

    try:
        yield  # Run the test
    finally:
        # Restore method state
        Agent.run = original_run  # type: ignore[method-assign]
        # Restore context variable
        _current_item_var.reset(token)

        # Captured responses now in item.stash[AGENT_RESPONSES_KEY]
        captured_count = len(item.stash.get(AGENT_RESPONSES_KEY, []))
        logger.debug(f"Restored Agent.run(), captured {captured_count} responses.")


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: Item) -> None:
    """
    Here we serialize the Dataset if in 'new_baseline' mode.
    """

    # Execute the hook only for tests marked with @pytest.mark.assay
    marker = item.get_closest_marker("assay")
    if marker is None:
        return

    # Only inject for Function items i.e. actual test functions
    # For example, if @pytest.mark.assay decorates a class, we skip it here.
    if not isinstance(item, Function):
        return

    assay: AssayContext | None = item.funcargs.get("assay")  # type: ignore[attr-defined]
    if assay is None:
        logger.warning("No assay context found in test function arguments during teardown.")
        return

    if assay.assay_mode == "new_baseline":
        logger.info(f"Serializing assay dataset to {assay.path}")
        assay.path.parent.mkdir(parents=True, exist_ok=True)
        assay.dataset.to_file(assay.path, schema_path=None)


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

            # Number of intercepted Agent.run() calls
            responses = item.stash.get(AGENT_RESPONSES_KEY, [])

            # Access the test outcome (passed, failed, etc.)
            test_outcome = "failed" if outcome else "passed"

            # Access the test duration
            test_duration = call.duration

            # Print Test Outcome and Duration
            logger.info(f"Test: {test_id}")
            logger.debug(f"Number of Agent.run() calls during test: {len(responses)}")
            logger.info(f"Test Outcome: {test_outcome}")
            logger.info(f"Test Duration: {test_duration:.5f} seconds")

            # Access baseline and novel players
            all_players = item.stash.get(PLAYERS_KEY, None)
            logger.debug(f"number of players: {len(all_players) if all_players is not None else 'None'}")

        except Exception as e:
            logger.error("ERROR:", e)
