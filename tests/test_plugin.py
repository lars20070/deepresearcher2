#!/usr/bin/env python3
"""
Unit tests for the pytest assay plugin.

These tests cover:
- Plugin hook functions (pytest_addoption, pytest_configure, etc.)
- Helper functions (_path, _is_assay)
- AssayContext model
- Agent.run() interception mechanism
- Evaluation strategies
"""

from __future__ import annotations as _annotations

import contextlib
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.agent import AgentRunResult
from pydantic_evals import Case, Dataset
from pytest import Function, Item

import deepresearcher2.plugin
from deepresearcher2.plugin import (
    ASSAY_MODES,
    AssayContext,
    _current_item_var,
    _is_assay,
    _path,
    bradley_terry_evaluation,
    pytest_addoption,
    pytest_configure,
    pytest_runtest_call,
    pytest_runtest_makereport,
    pytest_runtest_setup,
    pytest_runtest_teardown,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# =============================================================================
# Module Import Tests
# =============================================================================


def test_module_imports() -> None:
    """
    Test that the plugin module imports correctly and exports expected symbols.

    This test forces a module reload to ensure coverage tracks import-time code.
    """
    # Reload the module to capture import-time coverage
    module = importlib.reload(deepresearcher2.plugin)

    # Verify module-level exports
    assert module.ASSAY_MODES == ("evaluate", "new_baseline")
    assert callable(module.pytest_addoption)
    assert callable(module.pytest_configure)
    assert callable(module.pytest_runtest_setup)
    assert callable(module.pytest_runtest_call)
    assert callable(module.pytest_runtest_teardown)
    assert callable(module.pytest_runtest_makereport)
    assert callable(module.bradley_terry_evaluation)
    assert callable(module._path)
    assert callable(module._is_assay)


def test_assay_modes_constant() -> None:
    """Test that ASSAY_MODES contains the expected values."""
    assert ASSAY_MODES == ("evaluate", "new_baseline")
    assert len(ASSAY_MODES) == 2


# =============================================================================
# AssayContext Model Tests
# =============================================================================


def test_assay_context_model() -> None:
    """Test AssayContext model creation with valid data."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])
    path = Path("/tmp/test.json")

    context = AssayContext(
        dataset=dataset,
        path=path,
        assay_mode="evaluate",
    )

    assert context.dataset == dataset
    assert context.path == path
    assert context.assay_mode == "evaluate"


def test_assay_context_model_default_mode() -> None:
    """Test AssayContext uses 'evaluate' as default assay_mode."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])
    path = Path("/tmp/test.json")

    # Create without specifying assay_mode
    context = AssayContext(dataset=dataset, path=path)

    assert context.assay_mode == "evaluate"


def test_assay_context_model_new_baseline_mode() -> None:
    """Test AssayContext with new_baseline mode."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])
    path = Path("/tmp/test.json")

    context = AssayContext(
        dataset=dataset,
        path=path,
        assay_mode="new_baseline",
    )

    assert context.assay_mode == "new_baseline"


def test_assay_context_with_cases() -> None:
    """Test AssayContext with a dataset containing cases."""
    cases: list[Case[dict[str, str], type[None], Any]] = [
        Case(name="case_001", inputs={"query": "test query 1"}),
        Case(name="case_002", inputs={"query": "test query 2"}),
    ]
    dataset = Dataset[dict[str, str], type[None], Any](cases=cases)
    path = Path("/tmp/test.json")

    context = AssayContext(dataset=dataset, path=path)

    assert len(context.dataset.cases) == 2
    assert context.dataset.cases[0].inputs["query"] == "test query 1"


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_path_computation(mocker: MockerFixture) -> None:
    """Test _path computes the correct assay file path."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.fspath = "/project/tests/test_example.py"
    mock_item.name = "test_my_function"

    result = _path(mock_item)

    expected = Path("/project/tests/assays/test_example/test_my_function.json")
    assert result == expected


def test_path_computation_with_parametrized_test(mocker: MockerFixture) -> None:
    """Test _path strips parameter suffix from parametrized test names."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.fspath = "/project/tests/test_example.py"
    mock_item.name = "test_my_function[param1-param2]"

    result = _path(mock_item)

    # Should strip everything after the bracket
    expected = Path("/project/tests/assays/test_example/test_my_function.json")
    assert result == expected


def test_path_computation_nested_directory(mocker: MockerFixture) -> None:
    """Test _path with deeply nested test directories."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.fspath = "/project/tests/integration/api/test_endpoints.py"
    mock_item.name = "test_get_user"

    result = _path(mock_item)

    expected = Path("/project/tests/integration/api/assays/test_endpoints/test_get_user.json")
    assert result == expected


def test_is_assay_with_marker(mocker: MockerFixture) -> None:
    """Test _is_assay returns True for Function items with assay marker."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.get_closest_marker.return_value = mocker.MagicMock()

    assert _is_assay(mock_item) is True
    mock_item.get_closest_marker.assert_called_once_with("assay")


def test_is_assay_without_marker(mocker: MockerFixture) -> None:
    """Test _is_assay returns False for items without assay marker."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.get_closest_marker.return_value = None

    assert _is_assay(mock_item) is False


def test_is_assay_non_function_item(mocker: MockerFixture) -> None:
    """Test _is_assay returns False for non-Function items (e.g., class)."""
    mock_item = mocker.MagicMock(spec=Item)

    # Ensure it's not a Function instance
    assert not isinstance(mock_item, Function)
    assert _is_assay(mock_item) is False


# =============================================================================
# pytest_addoption Tests
# =============================================================================


def test_pytest_addoption(mocker: MockerFixture) -> None:
    """Test pytest_addoption registers the --assay-mode option correctly."""
    mock_group = mocker.MagicMock()
    mock_parser = mocker.MagicMock()
    mock_parser.getgroup.return_value = mock_group

    pytest_addoption(mock_parser)

    mock_parser.getgroup.assert_called_once_with("recording")
    mock_group.addoption.assert_called_once_with(
        "--assay-mode",
        action="store",
        default="evaluate",
        choices=ASSAY_MODES,
        help='Assay mode. Defaults to "evaluate".',
    )


# =============================================================================
# pytest_configure Tests
# =============================================================================


def test_pytest_configure(mocker: MockerFixture) -> None:
    """Test pytest_configure registers the assay marker."""
    mock_config = mocker.MagicMock()
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_configure(mock_config)

    mock_config.addinivalue_line.assert_called_once()
    call_args = mock_config.addinivalue_line.call_args
    assert call_args[0][0] == "markers"
    assert "assay" in call_args[0][1]


# =============================================================================
# pytest_runtest_setup Tests
# =============================================================================


def test_pytest_runtest_setup_non_assay_item(mocker: MockerFixture) -> None:
    """Test pytest_runtest_setup skips non-assay items."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.funcargs = {}
    mocker.patch("deepresearcher2.plugin._is_assay", return_value=False)

    # Should not raise and should not modify funcargs
    pytest_runtest_setup(mock_item)

    # funcargs should not have been accessed/modified
    assert "assay" not in mock_item.funcargs


def test_pytest_runtest_setup_with_existing_dataset(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup loads existing dataset from file."""
    # Create a temporary dataset file
    dataset_path = tmp_path / "assays" / "test_module" / "test_func.json"
    dataset_path.parent.mkdir(parents=True)
    dataset = Dataset[dict[str, str], type[None], Any](cases=[Case(name="case_001", inputs={"query": "existing query"})])
    dataset.to_file(dataset_path, schema_path=None)

    # Mock the item
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin._path", return_value=dataset_path)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_setup(mock_item)

    # Verify assay context was injected
    assert "assay" in mock_item.funcargs
    assay_ctx = mock_item.funcargs["assay"]
    # Check by attribute presence instead of isinstance (module reload can cause different class instances)
    assert hasattr(assay_ctx, "dataset")
    assert hasattr(assay_ctx, "path")
    assert hasattr(assay_ctx, "assay_mode")
    assert len(assay_ctx.dataset.cases) == 1
    assert assay_ctx.path == dataset_path


def test_pytest_runtest_setup_with_generator(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup calls generator when no dataset file exists."""
    dataset_path = tmp_path / "assays" / "test_module" / "test_func.json"

    # Mock generator that returns a dataset
    generated_dataset = Dataset[dict[str, str], type[None], Any](cases=[Case(name="generated", inputs={"query": "generated query"})])
    mock_generator = mocker.MagicMock(return_value=generated_dataset)

    # Mock the item
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"generator": mock_generator}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin._path", return_value=dataset_path)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_setup(mock_item)

    # Verify generator was called
    mock_generator.assert_called_once()

    # Verify dataset was serialized
    assert dataset_path.exists()

    # Verify assay context was injected
    assay_ctx = mock_item.funcargs["assay"]
    assert len(assay_ctx.dataset.cases) == 1


def test_pytest_runtest_setup_generator_invalid_return(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup raises TypeError for invalid generator return."""
    dataset_path = tmp_path / "assays" / "test_module" / "test_func.json"

    # Mock generator that returns invalid type
    mock_generator = mocker.MagicMock(return_value="not a dataset")

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"generator": mock_generator}
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin._path", return_value=dataset_path)
    mocker.patch("deepresearcher2.plugin.logger")

    with pytest.raises(TypeError, match="must return a Dataset instance"):
        pytest_runtest_setup(mock_item)


def test_pytest_runtest_setup_empty_dataset(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup uses empty dataset when no file or generator."""
    dataset_path = tmp_path / "assays" / "test_module" / "test_func.json"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}  # No generator
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin._path", return_value=dataset_path)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_setup(mock_item)

    assay_ctx = mock_item.funcargs["assay"]
    assert len(assay_ctx.dataset.cases) == 0


# =============================================================================
# pytest_runtest_call Tests
# =============================================================================


def test_pytest_runtest_call_non_function_item(mocker: MockerFixture) -> None:
    """Test pytest_runtest_call yields immediately for non-Function items."""
    mock_item = mocker.MagicMock(spec=Item)

    gen = pytest_runtest_call(mock_item)
    next(gen)  # Should yield immediately

    with pytest.raises(StopIteration):
        next(gen)


def test_pytest_runtest_call_without_assay_marker(mocker: MockerFixture) -> None:
    """Test pytest_runtest_call yields immediately without assay marker."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.get_closest_marker.return_value = None

    gen = pytest_runtest_call(mock_item)
    next(gen)

    with pytest.raises(StopIteration):
        next(gen)


def test_pytest_runtest_call_initializes_stash(mocker: MockerFixture) -> None:
    """Test pytest_runtest_call initializes the response stash."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("deepresearcher2.plugin.logger")

    gen = pytest_runtest_call(mock_item)
    next(gen)  # Run until yield

    # Verify stash was initialized - check that at least one key exists with empty list
    assert len(mock_item.stash) == 1
    stash_values = list(mock_item.stash.values())
    assert stash_values[0] == []

    # Clean up
    with contextlib.suppress(StopIteration):
        next(gen)


def test_pytest_runtest_call_sets_context_var(mocker: MockerFixture) -> None:
    """Test pytest_runtest_call sets the current item context variable."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("deepresearcher2.plugin.logger")

    gen = pytest_runtest_call(mock_item)
    next(gen)

    # During the yield, context var should be set - use the module's instance
    assert deepresearcher2.plugin._current_item_var.get() == mock_item

    # Clean up and verify context var is reset
    with contextlib.suppress(StopIteration):
        next(gen)

    assert deepresearcher2.plugin._current_item_var.get() is None


# =============================================================================
# pytest_runtest_teardown Tests
# =============================================================================


def test_pytest_runtest_teardown_non_assay_item(mocker: MockerFixture) -> None:
    """Test pytest_runtest_teardown skips non-assay items."""
    mock_item = mocker.MagicMock(spec=Item)
    mocker.patch("deepresearcher2.plugin._is_assay", return_value=False)

    # Should not raise
    pytest_runtest_teardown(mock_item)


def test_pytest_runtest_teardown_evaluate_mode(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown does not serialize in evaluate mode."""
    dataset_path = tmp_path / "assays" / "test.json"
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"assay": AssayContext(dataset=dataset, path=dataset_path, assay_mode="evaluate")}

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_teardown(mock_item)

    # File should not be created in evaluate mode
    assert not dataset_path.exists()


def test_pytest_runtest_teardown_new_baseline_mode(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown serializes dataset in new_baseline mode."""
    dataset_path = tmp_path / "assays" / "test.json"
    dataset = Dataset[dict[str, str], type[None], Any](cases=[Case(name="case_001", inputs={"query": "test"})])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"assay": AssayContext(dataset=dataset, path=dataset_path, assay_mode="new_baseline")}

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_teardown(mock_item)

    # File should be created in new_baseline mode
    assert dataset_path.exists()


def test_pytest_runtest_teardown_no_assay_context(mocker: MockerFixture) -> None:
    """Test pytest_runtest_teardown handles missing assay context gracefully."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}  # No assay context

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)

    # Should not raise
    pytest_runtest_teardown(mock_item)


# =============================================================================
# pytest_runtest_makereport Tests
# =============================================================================


def test_pytest_runtest_makereport_non_assay_item(mocker: MockerFixture) -> None:
    """Test pytest_runtest_makereport skips non-assay items."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_call = mocker.MagicMock()
    mock_call.when = "call"

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=False)
    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_makereport(mock_item, mock_call)

    # Logger should not be called for non-assay items
    mock_logger.info.assert_not_called()


def test_pytest_runtest_makereport_setup_phase(mocker: MockerFixture) -> None:
    """Test pytest_runtest_makereport ignores setup phase."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_call = mocker.MagicMock()
    mock_call.when = "setup"

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_makereport(mock_item, mock_call)

    mock_logger.info.assert_not_called()


def test_pytest_runtest_makereport_teardown_phase(mocker: MockerFixture) -> None:
    """Test pytest_runtest_makereport ignores teardown phase."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_call = mocker.MagicMock()
    mock_call.when = "teardown"

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_makereport(mock_item, mock_call)

    mock_logger.info.assert_not_called()


def test_pytest_runtest_makereport_passed_test(mocker: MockerFixture) -> None:
    """Test pytest_runtest_makereport logs passed test correctly."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.nodeid = "tests/test_example.py::test_foo"
    mock_item.funcargs = {"assay": None}  # No assay context
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}
    mock_item.get_closest_marker.return_value = mock_marker

    mock_call = mocker.MagicMock()
    mock_call.when = "call"
    mock_call.excinfo = None
    mock_call.duration = 0.12345

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_makereport(mock_item, mock_call)

    mock_logger.info.assert_any_call("Test: tests/test_example.py::test_foo")
    mock_logger.info.assert_any_call("Test Outcome: passed")
    mock_logger.info.assert_any_call("Test Duration: 0.12345 seconds")


def test_pytest_runtest_makereport_failed_test(mocker: MockerFixture) -> None:
    """Test pytest_runtest_makereport logs failed test correctly."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.nodeid = "tests/test_example.py::test_bar"
    mock_item.funcargs = {"assay": None}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}
    mock_item.get_closest_marker.return_value = mock_marker

    mock_call = mocker.MagicMock()
    mock_call.when = "call"
    mock_call.excinfo = mocker.MagicMock()  # Has exception
    mock_call.duration = 0.5

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_makereport(mock_item, mock_call)

    mock_logger.info.assert_any_call("Test Outcome: failed")


def test_pytest_runtest_makereport_runs_evaluation(mocker: MockerFixture) -> None:
    """Test pytest_runtest_makereport runs evaluation in evaluate mode."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.nodeid = "tests/test.py::test_func"
    mock_item.funcargs = {"assay": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_marker = mocker.MagicMock()

    # Custom evaluator mock
    mock_evaluator = AsyncMock()
    mock_marker.kwargs = {"evaluator": mock_evaluator}
    mock_item.get_closest_marker.return_value = mock_marker

    mock_call = mocker.MagicMock()
    mock_call.when = "call"
    mock_call.excinfo = None
    mock_call.duration = 0.1

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_makereport(mock_item, mock_call)

    # Evaluator should have been called
    mock_evaluator.assert_called_once_with(mock_item)


def test_pytest_runtest_makereport_skips_evaluation_in_new_baseline(mocker: MockerFixture) -> None:
    """Test pytest_runtest_makereport skips evaluation in new_baseline mode."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.nodeid = "tests/test.py::test_func"
    mock_item.funcargs = {"assay": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="new_baseline")}
    mock_marker = mocker.MagicMock()
    mock_evaluator = AsyncMock()
    mock_marker.kwargs = {"evaluator": mock_evaluator}
    mock_item.get_closest_marker.return_value = mock_marker

    mock_call = mocker.MagicMock()
    mock_call.when = "call"
    mock_call.excinfo = None
    mock_call.duration = 0.1

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_makereport(mock_item, mock_call)

    # Evaluator should NOT be called in new_baseline mode
    mock_evaluator.assert_not_called()


def test_pytest_runtest_makereport_invalid_evaluator(mocker: MockerFixture) -> None:
    """Test pytest_runtest_makereport handles non-callable evaluator."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.nodeid = "tests/test.py::test_func"
    mock_item.funcargs = {"assay": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"evaluator": "not_callable"}  # Invalid
    mock_item.get_closest_marker.return_value = mock_marker

    mock_call = mocker.MagicMock()
    mock_call.when = "call"
    mock_call.excinfo = None
    mock_call.duration = 0.1

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_makereport(mock_item, mock_call)

    # Should log error for invalid evaluator
    mock_logger.error.assert_called_once()
    assert "Invalid evaluator type" in str(mock_logger.error.call_args)


def test_pytest_runtest_makereport_evaluation_exception(mocker: MockerFixture) -> None:
    """Test pytest_runtest_makereport handles evaluation exceptions."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.nodeid = "tests/test.py::test_func"
    mock_item.funcargs = {"assay": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_marker = mocker.MagicMock()

    # Evaluator that raises an exception
    async def failing_evaluator(item: Item) -> None:
        raise RuntimeError("Evaluation failed")

    mock_marker.kwargs = {"evaluator": failing_evaluator}
    mock_item.get_closest_marker.return_value = mock_marker

    mock_call = mocker.MagicMock()
    mock_call.when = "call"
    mock_call.excinfo = None
    mock_call.duration = 0.1

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    # Should not raise, exception is handled internally
    pytest_runtest_makereport(mock_item, mock_call)

    # Should log exception
    mock_logger.exception.assert_called_once()


# =============================================================================
# bradley_terry_evaluation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_bradley_terry_evaluation_no_players(mocker: MockerFixture) -> None:
    """Test bradley_terry_evaluation handles empty player list."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"assay": None}
    # Use the module's AGENT_RESPONSES_KEY
    mock_item.stash = {deepresearcher2.plugin.AGENT_RESPONSES_KEY: []}

    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    await bradley_terry_evaluation(mock_item)

    mock_logger.debug.assert_any_call("No players to evaluate in tournament.")


@pytest.mark.asyncio
async def test_bradley_terry_evaluation_with_baseline_cases(mocker: MockerFixture) -> None:
    """Test bradley_terry_evaluation creates players from baseline cases."""
    cases: list[Case[dict[str, str], type[None], Any]] = [
        Case(name="case_001", inputs={"query": "baseline query 1"}),
        Case(name="case_002", inputs={"query": "baseline query 2"}),
    ]
    dataset = Dataset[dict[str, str], type[None], Any](cases=cases)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"assay": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    # Use the module's AGENT_RESPONSES_KEY
    mock_item.stash = {deepresearcher2.plugin.AGENT_RESPONSES_KEY: []}

    mocker.patch("deepresearcher2.plugin.logger")

    # Mock the tournament to avoid actual API calls
    mock_tournament_class = mocker.patch("deepresearcher2.plugin.EvalTournament")
    mock_tournament = mocker.MagicMock()
    mock_tournament.run = AsyncMock(return_value=[])
    mock_tournament.get_player_by_idx = MagicMock(return_value=MagicMock(score=0.5))
    mock_tournament_class.return_value = mock_tournament

    await bradley_terry_evaluation(mock_item)

    # Verify tournament was created with players
    mock_tournament_class.assert_called_once()
    call_kwargs = mock_tournament_class.call_args.kwargs
    assert len(call_kwargs["players"]) == 2


@pytest.mark.asyncio
async def test_bradley_terry_evaluation_with_novel_responses(mocker: MockerFixture) -> None:
    """Test bradley_terry_evaluation creates players from novel responses."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    # Mock agent responses
    mock_response1 = mocker.MagicMock(spec=AgentRunResult)
    mock_response1.output = "novel output 1"
    mock_response2 = mocker.MagicMock(spec=AgentRunResult)
    mock_response2.output = "novel output 2"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"assay": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    # Use the module's AGENT_RESPONSES_KEY
    mock_item.stash = {deepresearcher2.plugin.AGENT_RESPONSES_KEY: [mock_response1, mock_response2]}

    mocker.patch("deepresearcher2.plugin.logger")

    mock_tournament_class = mocker.patch("deepresearcher2.plugin.EvalTournament")
    mock_tournament = mocker.MagicMock()
    mock_tournament.run = AsyncMock(return_value=[])
    mock_tournament.get_player_by_idx = MagicMock(return_value=MagicMock(score=0.5))
    mock_tournament_class.return_value = mock_tournament

    await bradley_terry_evaluation(mock_item)

    # Verify tournament was created with 2 novel players
    mock_tournament_class.assert_called_once()
    call_kwargs = mock_tournament_class.call_args.kwargs
    assert len(call_kwargs["players"]) == 2


@pytest.mark.asyncio
async def test_bradley_terry_evaluation_skips_none_outputs(mocker: MockerFixture) -> None:
    """Test bradley_terry_evaluation skips responses with None output."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_response1 = mocker.MagicMock(spec=AgentRunResult)
    mock_response1.output = None  # Should be skipped
    mock_response2 = mocker.MagicMock(spec=AgentRunResult)
    mock_response2.output = "valid output"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"assay": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    # Use the module's AGENT_RESPONSES_KEY
    mock_item.stash = {deepresearcher2.plugin.AGENT_RESPONSES_KEY: [mock_response1, mock_response2]}

    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    mock_tournament_class = mocker.patch("deepresearcher2.plugin.EvalTournament")
    mock_tournament = mocker.MagicMock()
    mock_tournament.run = AsyncMock(return_value=[])
    mock_tournament.get_player_by_idx = MagicMock(return_value=MagicMock(score=0.5))
    mock_tournament_class.return_value = mock_tournament

    await bradley_terry_evaluation(mock_item)

    # Should log warning for None output
    mock_logger.warning.assert_called_once()
    assert "None output" in str(mock_logger.warning.call_args)

    # Only one player should be created
    mock_tournament_class.assert_called_once()
    call_kwargs = mock_tournament_class.call_args.kwargs
    assert len(call_kwargs["players"]) == 1


# =============================================================================
# Context Variable Tests
# =============================================================================


def test_current_item_var_default() -> None:
    """Test that _current_item_var has None as default."""
    # Reset to ensure clean state
    token = _current_item_var.set(None)
    try:
        assert _current_item_var.get() is None
    finally:
        _current_item_var.reset(token)


def test_current_item_var_set_and_get(mocker: MockerFixture) -> None:
    """Test setting and getting the current item context variable."""
    mock_item = mocker.MagicMock(spec=Item)

    token = _current_item_var.set(mock_item)
    try:
        assert _current_item_var.get() == mock_item
    finally:
        _current_item_var.reset(token)

    # After reset, should be None again
    assert _current_item_var.get() is None
