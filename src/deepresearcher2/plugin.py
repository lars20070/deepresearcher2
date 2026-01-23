#!/usr/bin/env python3
import asyncio
import contextvars
from collections.abc import Coroutine, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Dataset
from pytest import CallInfo, Config, Function, Item, Parser

from .agents import EVALUATION_AGENT
from .evals.evals import (
    EvalGame,
    EvalPlayer,
    EvalTournament,
    adaptive_uncertainty_strategy,
)
from .logger import logger

# Modes for the assay plugin. "evaluate" is the default mode.
ASSAY_MODES = ("evaluate", "new_baseline")

# Key to stash Agent responses during assay tests
AGENT_RESPONSES_KEY = pytest.StashKey[list[AgentRunResult[Any]]]()

# Items stashed by the _wrapped_run wrapper. Required for async safety.
# _current_item_var defined at module level. But items are stored locally to the current execution context.
_current_item_var: contextvars.ContextVar[Item | None] = contextvars.ContextVar("_current_item", default=None)


@dataclass
class Readout:
    """Result from an evaluator execution."""

    passed: bool = True
    details: dict[str, Any] | None = None


class Evaluator(Protocol):
    """Protocol for evaluation strategy callables.

    The Protocol defines ONLY what the plugin needs to call.
    Evaluator implementations configure themselves via __init__.
    """

    def __call__(self, item: Item) -> Coroutine[Any, Any, Readout]: ...


def pytest_addoption(parser: Parser) -> None:
    """
    Register the --assay-mode option (evaluate or new_baseline).

    Args:
        parser: The pytest argument parser.
    """
    parser.addoption(
        "--assay-mode",
        action="store",
        default="evaluate",
        choices=ASSAY_MODES,
        help='Assay mode. Defaults to "evaluate".',
    )


def pytest_configure(config: Config) -> None:
    """
    Register the @pytest.mark.assay marker.

    Args:
        config: The pytest configuration object.
    """
    logger.info("Registering the @pytest.mark.assay marker.")
    config.addinivalue_line(
        "markers",
        "assay(generator=None, evaluator=BradleyTerryEvaluator()): "
        "Mark the test for AI agent evaluation (assay). "
        "Args: "
        "generator - optional callable returning a Dataset for test cases; "
        "evaluator - optional Evaluator instance for custom evaluation strategy "
        "(defaults to BradleyTerryEvaluator with default settings). "
        "Configure evaluators by instantiating with parameters: "
        "evaluator=BradleyTerryEvaluator(criterion='Which response is better?')",
    )


class AssayContext(BaseModel):
    """
    Context injected into assay tests containing dataset, path, and mode.

    Attributes:
        dataset: The evaluation dataset with test cases.
        path: File path for dataset persistence.
        assay_mode: "evaluate" or "new_baseline".
    """

    dataset: Dataset = Field(..., description="The evaluation dataset for this assay")
    path: Path = Field(..., description="File path where the assay dataset is stored")
    assay_mode: str = Field(default="evaluate", description='Assay mode: "evaluate" or "new_baseline"')


def _path(item: Item) -> Path:
    """
    Compute assay file path: <test_dir>/assays/<module>/<test>.json.

    Args:
        item: The pytest test item.
    """
    path = Path(item.fspath)
    module_name = path.stem
    test_name = item.name.split("[")[0]
    return path.parent / "assays" / module_name / f"{test_name}.json"


def _is_assay(item: Item) -> bool:
    """
    Check if item is a valid assay unit test:
    - item is a Function (not a class or module)
    - item has the @pytest.mark.assay marker

    Args:
        item: The pytest collection item to check.
    """
    if not isinstance(item, Function):
        return False
    return item.get_closest_marker("assay") is not None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: Item) -> None:
    """
    Load dataset and inject AssayContext into the test function.
    Loads from file if exists, else calls generator, else uses empty dataset.

    Args:
        item: The pytest test item being set up.
    """
    if not _is_assay(item):
        return

    # Get generator from marker kwargs
    marker = item.get_closest_marker("assay")
    assert marker is not None
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

    # Only inject for Function items i.e. actual test functions
    # For example, if @pytest.mark.assay decorates a class, we skip it here.
    if not isinstance(item, Function):
        yield
        return

    # 1. Filter: Only run for tests marked with @pytest.mark.assay
    marker = item.get_closest_marker("assay")
    if marker is None:
        yield
        return

    logger.info("Intercepting Agent.run() calls in pytest_runtest_call hook")

    # 2. Initialize Storage: Prepare the specific stash key for this test item
    item.stash[AGENT_RESPONSES_KEY] = []

    # 3. Capture State: Save the *current* method implementation.
    # Crucial for compatibility: If another plugin has already patched Agent.run,
    # we capture that patch instead of the raw method, preserving the chain of command.
    original_agent_run = Agent.run

    # 4. Set Context: Push the current test item into the thread-local context.
    # This 'token' is required to cleanly pop the context later.
    token = _current_item_var.set(item)

    # 5. Define Wrapper: Create the interceptor closure locally
    async def _instrumented_agent_run(
        self: Agent[Any, Any],
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> AgentRunResult[Any]:
        """
        Wrapped Agent.run() that captures responses to the current test item's stash.
        """
        # A. Execute actual logic (awaiting the captured variable avoids infinite recursion)
        result = await original_agent_run(self, *args, **kwargs)

        # B. Capture result (retrieve item via ContextVar tunnel)
        current_item = _current_item_var.get()
        if current_item is not None:
            responses = current_item.stash.get(AGENT_RESPONSES_KEY, [])
            responses.append(result)
            logger.debug(f"Captured Agent.run() response #{len(responses)}: {repr(result.output)[:100]}")

        return result

    # 6. Apply Patch: Hot-swap the class method
    Agent.run = _instrumented_agent_run  # type: ignore[method-assign]
    logger.debug("Monkeypatched Agent.run() for automatic response capture")

    try:
        yield  # 7. Yield control to pytest to run the actual test function
    finally:
        # 8. Data Integrity: Restore the original method immediately
        Agent.run = original_agent_run  # type: ignore[method-assign]

        # 9. Cleanup: Pop the ContextVar value using the token to prevent state leaks.
        # This ensures the context is clean for the next test execution.
        _current_item_var.reset(token)

        captured_count = len(item.stash.get(AGENT_RESPONSES_KEY, []))
        logger.debug(f"Restored Agent.run(), captured {captured_count} responses.")


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: Item) -> None:
    """
    Serialize the dataset to disk when in new_baseline mode.

    Args:
        item: The pytest test item being torn down.
    """
    if not _is_assay(item):
        return

    # Check whether to serialize the dataset
    assay: AssayContext | None = item.funcargs.get("assay")  # type: ignore[attr-defined]
    if assay is None or assay.assay_mode != "new_baseline":
        return

    logger.info(f"Serializing assay dataset to {assay.path}")
    assay.path.parent.mkdir(parents=True, exist_ok=True)
    assay.dataset.to_file(assay.path, schema_path=None)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item: Item, call: CallInfo) -> None:
    """
    Hook to process test reports and run evaluations after test execution.

    Runs the configured evaluation strategy (default: Bradley-Terry tournament)
    on captured model outputs after the test's "call" phase completes.

    Args:
        item: The pytest test item containing test metadata and stashed data.
        call: Information about the test call phase (setup/call/teardown).
    """
    if not _is_assay(item):
        return

    # pytest_runtest_makereport is called three times per test: setup, call, teardown
    # Here, we are interested in the "call" phase.
    # Use setup and teardown to report when a fixture or cleanup fails.
    if call.when != "call":
        return

    # Log test execution summary
    logger.info(f"Test: {item.nodeid}")
    test_outcome = "failed" if call.excinfo else "passed"
    logger.info(f"Test Outcome: {test_outcome}")
    logger.info(f"Test Duration: {call.duration:.5f} seconds")

    # Check whether to run evaluation
    assay: AssayContext | None = item.funcargs.get("assay")  # type: ignore[attr-defined]
    if assay is None or assay.assay_mode != "evaluate":
        return

    # Retrieve evaluator from marker kwargs with type validation
    marker = item.get_closest_marker("assay")
    assert marker is not None
    evaluator: Evaluator = marker.kwargs.get("evaluator", BradleyTerryEvaluator())
    if not callable(evaluator):
        logger.error(f"Invalid evaluator type: {type(evaluator)}. Expected callable.")
        return

    # Run the evaluator asynchronously
    try:
        readout = asyncio.run(evaluator(item))
        logger.info(f"Evaluation result: passed={readout.passed}")

        # Serialize the readout
        readout_path = assay.path.with_suffix(".readout.json")
        readout_path.parent.mkdir(parents=True, exist_ok=True)
        # readout.to_file(readout_path, schema_path=None)

    except Exception:
        logger.exception("Error during evaluation in pytest_runtest_makereport.")


class BradleyTerryEvaluator:
    """Evaluates test outputs using Bradley-Terry tournament scoring.

    Configuration is set at instantiation; __call__ runs the evaluation.
    """

    def __init__(
        self,
        criterion: str = "Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?",
    ) -> None:
        """Configure the evaluator.

        Args:
            criterion: The evaluation criterion for pairwise comparison.
        """
        self.criterion = criterion

    async def __call__(self, item: Item) -> Readout:
        """Run Bradley-Terry tournament on baseline and novel responses.

        Args:
            item: The pytest test item with assay context and captured responses.

        Returns:
            Readout with passed status and details.
        """
        # Prepare the list of all players, baseline and novel
        players: list[EvalPlayer] = []

        # 1. Baseline players from previously serialized assay dataset
        assay: AssayContext | None = item.funcargs.get("assay")  # type: ignore[attr-defined]
        baseline_case_count = 0
        if assay is not None:
            for idx, case in enumerate(assay.dataset.cases):
                players.append(EvalPlayer(idx=idx, item=case.inputs["query"]))
            baseline_case_count = len(assay.dataset.cases)

        # 2. Novel players from current test run
        responses = item.stash.get(AGENT_RESPONSES_KEY, [])
        for idx, response in enumerate(responses):
            if response.output is None:
                logger.warning(f"Response #{idx} has None output.")
                continue
            players.append(EvalPlayer(idx=idx + baseline_case_count, item=response.output))

        # Log all players before tournament
        for player in players:
            logger.debug(f"Player #{player.idx} item: {repr(player.item)[:100]}")

        if not players:
            logger.debug("No players to evaluate in tournament.")
            return Readout(passed=True, details={"message": "No players to evaluate"})

        model_settings = ModelSettings(
            temperature=0.0,
            timeout=300,
        )

        # Run the Bradley-Terry tournament to score both baseline and novel queries
        game = EvalGame(criterion=self.criterion)
        tournament = EvalTournament(players=players, game=game)
        players_scored = await tournament.run(
            agent=EVALUATION_AGENT,
            model_settings=model_settings,
            strategy=adaptive_uncertainty_strategy,
            max_standard_deviation=2.0,
        )

        # Players sorted by score
        players_sorted = sorted(players_scored, key=lambda p: p.score if p.score is not None else float("-inf"))
        for player in players_sorted:
            logger.debug(f"Player {player.idx:4d}   score: {player.score:7.4f}   query: {player.item}")

        # Average score for both baseline and novel queries
        scores_baseline = [tournament.get_player_by_idx(idx=i).score or 0.0 for i in range(len(players) // 2)]
        scores_novel = [tournament.get_player_by_idx(idx=i + len(players) // 2).score or 0.0 for i in range(len(players) // 2)]
        if scores_baseline and scores_novel:
            logger.debug(f"Average score for baseline queries (Players 0 to 9): {np.mean(scores_baseline):7.4f}")
            logger.debug(f"Average score for novel queries  (Players 10 to 19): {np.mean(scores_novel):7.4f}")

        return Readout(
            passed=True,
            details={
                "baseline_scores": scores_baseline,
                "novel_scores": scores_novel,
                "player_count": len(players),
            },
        )
