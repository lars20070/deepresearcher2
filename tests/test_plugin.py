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
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Case, Dataset
from pytest import Function, Item

import deepresearcher2.plugin
from deepresearcher2.plugin import (
    ASSAY_MODES,
    BASELINE_DATASET_KEY,
    AssayContext,
    BradleyTerryEvaluator,
    PairwiseEvaluator,
    Readout,
    _current_item_var,
    _is_assay,
    _path,
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
    assert hasattr(module, "BradleyTerryEvaluator")
    assert hasattr(module, "PairwiseEvaluator")
    assert hasattr(module, "Readout")
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
        Case(name="case_001", inputs={"topic": "test topic 1", "query": "test query 1"}),
        Case(name="case_002", inputs={"topic": "test topic 2", "query": "test query 2"}),
    ]
    dataset = Dataset[dict[str, str], type[None], Any](cases=cases)
    path = Path("/tmp/test.json")

    context = AssayContext(dataset=dataset, path=path)

    # Verify case count and structure
    assert len(context.dataset.cases) == 2

    # Verify first case content
    assert context.dataset.cases[0].name == "case_001"
    assert context.dataset.cases[0].inputs["topic"] == "test topic 1"
    assert context.dataset.cases[0].inputs["query"] == "test query 1"
    assert "topic" in context.dataset.cases[0].inputs
    assert "query" in context.dataset.cases[0].inputs

    # Verify dataset is mutable (required for update inside unit tests)
    context.dataset.cases.clear()
    assert len(context.dataset.cases) == 0

    # Verify we can extend with new cases
    new_case = Case(name="case_003", inputs={"query": "new query"})
    context.dataset.cases.append(new_case)
    assert len(context.dataset.cases) == 1
    assert context.dataset.cases[0].name == "case_003"


# =============================================================================
# Helper Function Tests
# =============================================================================


def test_path_computation(mocker: MockerFixture) -> None:
    """Test _path computes the correct assay file path."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.path = Path("/project/tests/test_example.py")
    mock_item.name = "test_my_function"

    result = _path(mock_item)

    expected = Path("/project/tests/assays/test_example/test_my_function.json")
    assert result == expected

    # Verify path is absolute
    assert result.is_absolute()

    # Verify path ends with .json
    assert result.suffix == ".json"

    # Verify assays directory is in the path
    assert "assays" in result.parts


def test_path_computation_with_parametrized_test(mocker: MockerFixture) -> None:
    """Test _path strips parameter suffix from parametrized test names."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.path = Path("/project/tests/test_example.py")
    mock_item.name = "test_my_function[param1-param2]"

    result = _path(mock_item)

    # Should strip everything after the bracket
    expected = Path("/project/tests/assays/test_example/test_my_function.json")
    assert result == expected


def test_path_computation_nested_directory(mocker: MockerFixture) -> None:
    """Test _path with deeply nested test directories."""
    mock_item = mocker.MagicMock(spec=Item)
    mock_item.path = Path("/project/tests/integration/api/test_endpoints.py")
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
    mock_parser = mocker.MagicMock()

    pytest_addoption(mock_parser)

    mock_parser.addoption.assert_called_once_with(
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
    assert "context" not in mock_item.funcargs


def test_pytest_runtest_setup_with_existing_dataset(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup loads existing dataset from file."""
    # Create a temporary dataset file with realistic topic data
    dataset_path = tmp_path / "assays" / "test_module" / "test_func.json"
    dataset_path.parent.mkdir(parents=True)
    dataset = Dataset[dict[str, str], type[None], Any](
        cases=[
            Case(name="case_001", inputs={"topic": "pangolin trafficking", "query": "existing query"}),
            Case(name="case_002", inputs={"topic": "molecular gastronomy", "query": "another query"}),
        ]
    )
    dataset.to_file(dataset_path, schema_path=None)

    # Mock the item
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin._path", return_value=dataset_path)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_setup(mock_item)

    # Verify baseline dataset was stashed for evaluators
    assert deepresearcher2.plugin.BASELINE_DATASET_KEY in mock_item.stash
    baseline = mock_item.stash[deepresearcher2.plugin.BASELINE_DATASET_KEY]
    assert len(baseline.cases) == 2
    assert baseline.cases[0].inputs["query"] == "existing query"

    # Verify assay context was injected
    assert "context" in mock_item.funcargs
    assay_ctx = mock_item.funcargs["context"]

    # Check by attribute presence instead of isinstance (module reload can cause different class instances)
    assert hasattr(assay_ctx, "dataset")
    assert hasattr(assay_ctx, "path")
    assert hasattr(assay_ctx, "assay_mode")

    # Verify dataset content was loaded correctly
    assert len(assay_ctx.dataset.cases) == 2
    assert assay_ctx.path == dataset_path
    assert assay_ctx.assay_mode == "evaluate"

    # Verify case data integrity
    assert assay_ctx.dataset.cases[0].name == "case_001"
    assert assay_ctx.dataset.cases[0].inputs["topic"] == "pangolin trafficking"
    assert assay_ctx.dataset.cases[0].inputs["query"] == "existing query"

    assert assay_ctx.dataset.cases[1].name == "case_002"
    assert assay_ctx.dataset.cases[1].inputs["topic"] == "molecular gastronomy"
    assert assay_ctx.dataset.cases[1].inputs["query"] == "another query"

    # Verify cases are iterable (as used in test_curiosity.py)
    topics = [case.inputs["topic"] for case in assay_ctx.dataset.cases]
    assert topics == ["pangolin trafficking", "molecular gastronomy"]


def test_pytest_runtest_setup_with_generator(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_setup calls generator when no dataset file exists."""
    dataset_path = tmp_path / "assays" / "test_module" / "test_func.json"

    # Mock generator that returns a dataset (like generate_evaluation_cases in test_curiosity.py)
    topics = ["pangolin trafficking", "molecular gastronomy", "dark kitchen economics"]
    generated_cases: list[Case[dict[str, str], type[None], Any]] = [
        Case(name=f"case_{idx:03d}", inputs={"topic": topic}) for idx, topic in enumerate(topics)
    ]
    generated_dataset = Dataset[dict[str, str], type[None], Any](cases=generated_cases)
    mock_generator = mocker.MagicMock(return_value=generated_dataset)

    # Mock the item
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"generator": mock_generator}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin._path", return_value=dataset_path)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_setup(mock_item)

    # Verify baseline dataset was stashed
    assert deepresearcher2.plugin.BASELINE_DATASET_KEY in mock_item.stash
    assert len(mock_item.stash[deepresearcher2.plugin.BASELINE_DATASET_KEY].cases) == 3

    # Verify generator was called exactly once
    mock_generator.assert_called_once()

    # Verify dataset file was created
    assert dataset_path.exists()
    assert dataset_path.suffix == ".json"

    # Verify file contains valid JSON that can be reloaded
    reloaded_dataset = Dataset[dict[str, str], type[None], Any].from_file(dataset_path)
    assert len(reloaded_dataset.cases) == 3

    # Verify reloaded data matches original
    for idx, topic in enumerate(topics):
        assert reloaded_dataset.cases[idx].name == f"case_{idx:03d}"
        assert reloaded_dataset.cases[idx].inputs["topic"] == topic

    # Verify assay context was injected with correct data
    assay_ctx = mock_item.funcargs["context"]
    assert len(assay_ctx.dataset.cases) == 3
    assert assay_ctx.path == dataset_path
    assert assay_ctx.assay_mode == "evaluate"

    # Verify case content
    assert assay_ctx.dataset.cases[0].inputs["topic"] == "pangolin trafficking"
    assert assay_ctx.dataset.cases[1].inputs["topic"] == "molecular gastronomy"
    assert assay_ctx.dataset.cases[2].inputs["topic"] == "dark kitchen economics"


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
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}  # No generator
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin._path", return_value=dataset_path)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_setup(mock_item)

    assert deepresearcher2.plugin.BASELINE_DATASET_KEY in mock_item.stash
    assert len(mock_item.stash[deepresearcher2.plugin.BASELINE_DATASET_KEY].cases) == 0

    assay_ctx = mock_item.funcargs["context"]
    assert len(assay_ctx.dataset.cases) == 0


def test_pytest_runtest_setup_baseline_stash_is_copy(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test that mutating assay.dataset does not change the stashed baseline."""
    dataset_path = tmp_path / "assays" / "test_module" / "test_func.json"
    dataset_path.parent.mkdir(parents=True)
    dataset = Dataset[dict[str, str], type[None], Any](
        cases=[
            Case(name="case_001", inputs={"topic": "topic A", "query": "baseline query A"}),
        ]
    )
    dataset.to_file(dataset_path, schema_path=None)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "evaluate"

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin._path", return_value=dataset_path)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_setup(mock_item)

    assay_ctx = mock_item.funcargs["context"]
    baseline = mock_item.stash[deepresearcher2.plugin.BASELINE_DATASET_KEY]

    # Simulate test mutation (like test_curiosity.py)
    assay_ctx.dataset.cases.clear()
    assay_ctx.dataset.cases.append(Case(name="case_001", inputs={"topic": "topic A", "query": "novel query A"}))

    # Stashed baseline must be unchanged
    assert len(baseline.cases) == 1
    assert baseline.cases[0].inputs["query"] == "baseline query A"
    assert assay_ctx.dataset.cases[0].inputs["query"] == "novel query A"


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

    # During the yield, context var should be set
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
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="evaluate")}

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_teardown(mock_item)

    # File should not be created in evaluate mode
    assert not dataset_path.exists()


def test_pytest_runtest_teardown_new_baseline_mode(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown merges captured responses and serializes dataset in new_baseline mode."""
    dataset_path = tmp_path / "assays" / "test.json"

    # Create dataset with cases that have empty expected_output (as generated)
    cases: list[Case[dict[str, str], str, Any]] = [
        Case(name="case_000", inputs={"topic": "topic A"}, expected_output=""),
        Case(name="case_001", inputs={"topic": "topic B"}, expected_output=""),
    ]
    dataset = Dataset[dict[str, str], str, Any](cases=cases)

    # Create mock AgentRunResult responses (captured by pytest_runtest_call)
    mock_response_a = mocker.MagicMock(spec=AgentRunResult)
    mock_response_a.output = "generated query A"
    mock_response_b = mocker.MagicMock(spec=AgentRunResult)
    mock_response_b.output = "generated query B"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="new_baseline")}
    mock_item.stash = {
        deepresearcher2.plugin.AGENT_RESPONSES_KEY: [mock_response_a, mock_response_b],
    }

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_teardown(mock_item)

    # File should be created in new_baseline mode
    assert dataset_path.exists()
    assert dataset_path.suffix == ".json"

    # Verify serialized content can be reloaded
    reloaded = Dataset[dict[str, str], str, Any].from_file(dataset_path)
    assert len(reloaded.cases) == 2

    # Verify expected_output was populated from captured responses
    assert reloaded.cases[0].name == "case_000"
    assert reloaded.cases[0].inputs["topic"] == "topic A"
    assert reloaded.cases[0].expected_output == "generated query A"

    assert reloaded.cases[1].name == "case_001"
    assert reloaded.cases[1].inputs["topic"] == "topic B"
    assert reloaded.cases[1].expected_output == "generated query B"


def test_pytest_runtest_teardown_new_baseline_response_count_mismatch(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown skips serialization when response count mismatches case count."""
    dataset_path = tmp_path / "assays" / "test.json"

    cases: list[Case[dict[str, str], str, Any]] = [
        Case(name="case_000", inputs={"topic": "topic A"}, expected_output=""),
        Case(name="case_001", inputs={"topic": "topic B"}, expected_output=""),
    ]
    dataset = Dataset[dict[str, str], str, Any](cases=cases)

    # Only one response for two cases
    mock_response = mocker.MagicMock(spec=AgentRunResult)
    mock_response.output = "query A"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="new_baseline")}
    mock_item.stash = {
        deepresearcher2.plugin.AGENT_RESPONSES_KEY: [mock_response],
    }

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_teardown(mock_item)

    # File should NOT be created due to mismatch
    assert not dataset_path.exists()
    mock_logger.error.assert_called_once()
    assert "Cannot merge responses" in str(mock_logger.error.call_args)


def test_pytest_runtest_teardown_new_baseline_none_output(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown uses empty string for None response output."""
    dataset_path = tmp_path / "assays" / "test.json"

    cases: list[Case[dict[str, str], str, Any]] = [
        Case(name="case_000", inputs={"topic": "topic A"}, expected_output=""),
    ]
    dataset = Dataset[dict[str, str], str, Any](cases=cases)

    mock_response = mocker.MagicMock(spec=AgentRunResult)
    mock_response.output = None

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="new_baseline")}
    mock_item.stash = {
        deepresearcher2.plugin.AGENT_RESPONSES_KEY: [mock_response],
    }

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_teardown(mock_item)

    assert dataset_path.exists()
    reloaded = Dataset[dict[str, str], str, Any].from_file(dataset_path)
    assert reloaded.cases[0].expected_output == ""


def test_pytest_runtest_teardown_new_baseline_no_responses(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_teardown skips serialization when no responses captured but cases exist."""
    dataset_path = tmp_path / "assays" / "test.json"

    cases: list[Case[dict[str, str], str, Any]] = [
        Case(name="case_000", inputs={"topic": "topic A"}, expected_output=""),
    ]
    dataset = Dataset[dict[str, str], str, Any](cases=cases)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=dataset_path, assay_mode="new_baseline")}
    mock_item.stash = {}  # No AGENT_RESPONSES_KEY

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_teardown(mock_item)

    # File should NOT be created (0 responses vs 1 case)
    assert not dataset_path.exists()
    mock_logger.error.assert_called_once()


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
    mock_item.funcargs = {"context": None}  # No assay context
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

    # Verify all expected log messages
    mock_logger.info.assert_any_call("Test: tests/test_example.py::test_foo")
    mock_logger.info.assert_any_call("Test Outcome: passed")
    mock_logger.info.assert_any_call("Test Duration: 0.12345 seconds")

    # Verify logger.info was called exactly 3 times for the summary
    assert mock_logger.info.call_count == 3

    # Verify error was not logged for passing test
    mock_logger.error.assert_not_called()
    mock_logger.exception.assert_not_called()


def test_pytest_runtest_makereport_failed_test(mocker: MockerFixture) -> None:
    """Test pytest_runtest_makereport logs failed test correctly."""
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.nodeid = "tests/test_example.py::test_bar"
    mock_item.funcargs = {"context": None}
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


def test_pytest_runtest_makereport_runs_evaluation(mocker: MockerFixture, tmp_path: Path) -> None:
    """Test pytest_runtest_makereport runs evaluation and serializes readout."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])
    assay_path = tmp_path / "assays" / "test_module" / "test_func.json"
    assay_path.parent.mkdir(parents=True, exist_ok=True)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.nodeid = "tests/test.py::test_func"
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=assay_path, assay_mode="evaluate")}
    mock_marker = mocker.MagicMock()

    # Custom evaluator mock - returns Readout with details
    mock_evaluator = AsyncMock(return_value=Readout(passed=True, details={"test_key": "test_value"}))
    mock_marker.kwargs = {"evaluator": mock_evaluator}
    mock_item.get_closest_marker.return_value = mock_marker

    mock_call = mocker.MagicMock()
    mock_call.when = "call"
    mock_call.excinfo = None
    mock_call.duration = 0.1

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_makereport(mock_item, mock_call)

    # Evaluator should have been called
    mock_evaluator.assert_called_once_with(mock_item)
    # Should log the result
    mock_logger.info.assert_any_call("Evaluation result: passed=True")

    # Verify readout file was created
    readout_path = assay_path.with_suffix(".readout.json")
    assert readout_path.exists()

    # Verify readout content
    with readout_path.open() as f:
        readout_data = json.load(f)
    assert readout_data["passed"] is True
    assert readout_data["details"] == {"test_key": "test_value"}


def test_pytest_runtest_makereport_uses_default_evaluator(mocker: MockerFixture) -> None:
    """Test pytest_runtest_makereport uses BradleyTerryEvaluator() as default."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.nodeid = "tests/test.py::test_func"
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_item.stash = {
        deepresearcher2.plugin.AGENT_RESPONSES_KEY: [],
        BASELINE_DATASET_KEY: dataset,
    }
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {}  # No evaluator specified - should use default
    mock_item.get_closest_marker.return_value = mock_marker

    mock_call = mocker.MagicMock()
    mock_call.when = "call"
    mock_call.excinfo = None
    mock_call.duration = 0.1

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mock_logger = mocker.patch("deepresearcher2.plugin.logger")
    # Mock the tournament to avoid actual API calls
    mock_tournament_class = mocker.patch("deepresearcher2.plugin.EvalTournament")
    mock_tournament = mocker.MagicMock()
    mock_tournament.run = AsyncMock(return_value=[])
    mock_tournament.get_player_by_idx = MagicMock(return_value=MagicMock(score=0.5))
    mock_tournament_class.return_value = mock_tournament

    pytest_runtest_makereport(mock_item, mock_call)

    # Should log evaluation completed (no error)
    assert any("Evaluation result" in str(call) for call in mock_logger.info.call_args_list)


def test_pytest_runtest_makereport_invalid_evaluator(mocker: MockerFixture) -> None:
    """Test pytest_runtest_makereport handles non-callable evaluator."""
    dataset = Dataset[dict[str, str], type[None], Any](cases=[])

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.nodeid = "tests/test.py::test_func"
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
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
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_marker = mocker.MagicMock()

    # Evaluator that raises an exception
    async def failing_evaluator(item: Item) -> Readout:
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
# Readout Tests
# =============================================================================


def test_readout_model_defaults() -> None:
    """Test Readout model initializes with default values."""
    readout = Readout()

    assert readout.passed is True
    assert readout.details is None


def test_readout_model_custom() -> None:
    """Test Readout model initializes with custom values."""
    readout = Readout(passed=False, details={"error": "test error"})

    assert readout.passed is False
    assert readout.details == {"error": "test error"}


def test_readout_to_file(tmp_path: Path) -> None:
    """Test Readout.to_file() serializes to JSON file."""
    readout = Readout(passed=True, details={"key": "value", "count": 42})
    file_path = tmp_path / "readout.json"

    readout.to_file(file_path)

    assert file_path.exists()

    # Verify content
    with file_path.open() as f:
        data = json.load(f)

    assert data["passed"] is True
    assert data["details"] == {"key": "value", "count": 42}


def test_readout_to_file_with_none_details(tmp_path: Path) -> None:
    """Test Readout.to_file() handles None details correctly."""
    readout = Readout(passed=False, details=None)
    file_path = tmp_path / "readout.json"

    readout.to_file(file_path)

    assert file_path.exists()

    with file_path.open() as f:
        data = json.load(f)

    assert data["passed"] is False
    assert data["details"] is None


# =============================================================================
# BradleyTerryEvaluator Tests
# =============================================================================


def test_bradley_terry_evaluator_init_defaults() -> None:
    """Test BradleyTerryEvaluator initializes with default values."""
    evaluator = BradleyTerryEvaluator()

    # Default model: OpenAIChatModel with qwen3:8b on Ollama
    assert evaluator.model is not None
    assert isinstance(evaluator.model, OpenAIChatModel)
    assert evaluator.model.model_name == "qwen3:8b"
    # model_settings (TypedDict)
    assert evaluator.model_settings.get("temperature") == 0.0
    assert evaluator.model_settings.get("timeout") == 300
    # system_prompt
    assert evaluator.system_prompt is not None
    assert "response" in evaluator.system_prompt
    assert "A" in evaluator.system_prompt and "B" in evaluator.system_prompt
    # agent
    assert evaluator.agent is not None
    # criterion and max_standard_deviation
    assert evaluator.criterion == "Which of the two search queries shows more genuine curiosity and creativity, and is less formulaic?"
    assert evaluator.max_standard_deviation == 2.0


def test_bradley_terry_evaluator_init_custom() -> None:
    """Test BradleyTerryEvaluator initializes with custom values."""
    evaluator = BradleyTerryEvaluator(criterion="Custom criterion", max_standard_deviation=1.5)

    assert evaluator.criterion == "Custom criterion"
    assert evaluator.max_standard_deviation == 1.5


def test_bradley_terry_evaluator_init_with_custom_model(mocker: MockerFixture) -> None:
    """Test BradleyTerryEvaluator uses custom model when provided."""
    custom_model = OpenAIChatModel(
        model_name="custom-model",
        provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
    )
    evaluator = BradleyTerryEvaluator(model=custom_model)

    assert evaluator.model is custom_model
    assert isinstance(evaluator.model, OpenAIChatModel)
    assert evaluator.model.model_name == "custom-model"
    assert evaluator.agent.model is custom_model


def test_bradley_terry_evaluator_init_with_model_string() -> None:
    """Test BradleyTerryEvaluator accepts model as string (e.g. for OpenAI default)."""
    evaluator = BradleyTerryEvaluator(model="openai:gpt-4o-mini")

    assert evaluator.model == "openai:gpt-4o-mini"


@pytest.mark.asyncio
async def test_bradley_terry_evaluator_call_no_players(mocker: MockerFixture) -> None:
    """Test BradleyTerryEvaluator.__call__ handles empty player list."""
    evaluator = BradleyTerryEvaluator()
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": None}
    mock_item.stash = {
        deepresearcher2.plugin.AGENT_RESPONSES_KEY: [],
        BASELINE_DATASET_KEY: Dataset[dict[str, str], type[None], Any](cases=[]),
    }

    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    result = await evaluator(mock_item)

    # Check result by attribute presence (module reload can cause isinstance to fail)
    assert type(result).__name__ == "Readout"
    assert result.passed is True
    assert result.details is not None
    assert result.details.get("message") == "No players to evaluate"
    mock_logger.debug.assert_any_call("No players to evaluate in tournament.")


@pytest.mark.asyncio
async def test_bradley_terry_evaluator_call_with_players(mocker: MockerFixture) -> None:
    """Test BradleyTerryEvaluator.__call__ runs tournament with players."""
    # Mock ModelSettings before creating the evaluator (it's now called in __init__)
    mock_model_settings = mocker.patch("deepresearcher2.plugin.ModelSettings")
    mock_model_settings.return_value = mocker.MagicMock()

    evaluator = BradleyTerryEvaluator(criterion="Test criterion", max_standard_deviation=1.5)

    # Verify ModelSettings was called with hard-coded values during __init__
    mock_model_settings.assert_called_once_with(temperature=0.0, timeout=300)

    cases: list[Case[dict[str, str], type[None], Any]] = [
        Case(name="case_001", inputs={"query": "baseline query"}),
    ]
    dataset = Dataset[dict[str, str], type[None], Any](cases=cases)

    mock_response = mocker.MagicMock(spec=AgentRunResult)
    mock_response.output = "novel output"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_item.stash = {
        deepresearcher2.plugin.AGENT_RESPONSES_KEY: [mock_response],
        BASELINE_DATASET_KEY: dataset,
    }

    mocker.patch("deepresearcher2.plugin.logger")

    # Mock the tournament - use SimpleNamespace for players to allow formatting
    mock_tournament_class = mocker.patch("deepresearcher2.plugin.EvalTournament")
    mock_tournament = mocker.MagicMock()
    mock_player = SimpleNamespace(idx=0, score=0.75, item="baseline query")
    mock_tournament.run = AsyncMock(return_value=[mock_player])
    mock_tournament.get_player_by_idx = MagicMock(return_value=mock_player)
    mock_tournament_class.return_value = mock_tournament

    result = await evaluator(mock_item)

    # Verify tournament was created with correct criterion
    mock_tournament_class.assert_called_once()
    call_kwargs = mock_tournament_class.call_args.kwargs
    assert call_kwargs["game"].criterion == "Test criterion"

    # Verify tournament.run was called with model_settings and max_standard_deviation
    mock_tournament.run.assert_called_once()
    run_kwargs = mock_tournament.run.call_args.kwargs
    assert "model_settings" in run_kwargs
    assert run_kwargs["max_standard_deviation"] == 1.5

    # Verify result (use type name check due to module reload)
    assert type(result).__name__ == "Readout"
    assert result.passed is False


@pytest.mark.asyncio
async def test_bradley_terry_evaluator_protocol_conformance() -> None:
    """Test BradleyTerryEvaluator conforms to Evaluator Protocol."""
    evaluator = BradleyTerryEvaluator()
    # Should be callable with Item and return Coroutine[Any, Any, Readout]
    assert callable(evaluator)


# =============================================================================
# PairwiseEvaluator Tests
# =============================================================================


def test_pairwise_evaluator_init_defaults() -> None:
    """Test PairwiseEvaluator initializes with default values."""
    evaluator = PairwiseEvaluator()

    # Default model: OpenAIChatModel with qwen3:8b on Ollama
    assert evaluator.model is not None
    assert isinstance(evaluator.model, OpenAIChatModel)
    assert evaluator.model.model_name == "qwen3:8b"
    # model_settings (TypedDict)
    assert evaluator.model_settings.get("temperature") == 0.0
    assert evaluator.model_settings.get("timeout") == 300
    # system_prompt
    assert evaluator.system_prompt is not None
    assert "response" in evaluator.system_prompt or "A" in evaluator.system_prompt
    assert "A" in evaluator.system_prompt and "B" in evaluator.system_prompt
    # agent
    assert evaluator.agent is not None
    # criterion
    assert "curiosity" in evaluator.criterion or "better" in evaluator.criterion.lower()


def test_pairwise_evaluator_init_custom() -> None:
    """Test PairwiseEvaluator initializes with custom values."""
    evaluator = PairwiseEvaluator(criterion="Custom criterion")

    assert evaluator.criterion == "Custom criterion"


def test_pairwise_evaluator_init_with_custom_model(mocker: MockerFixture) -> None:
    """Test PairwiseEvaluator uses custom model when provided."""
    custom_model = OpenAIChatModel(
        model_name="custom-model",
        provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
    )
    evaluator = PairwiseEvaluator(model=custom_model)

    assert evaluator.model is custom_model
    assert isinstance(evaluator.model, OpenAIChatModel)
    assert evaluator.model.model_name == "custom-model"
    assert evaluator.agent.model is custom_model


def test_pairwise_evaluator_init_with_model_string() -> None:
    """Test PairwiseEvaluator accepts model as string (e.g. for OpenAI default)."""
    evaluator = PairwiseEvaluator(model="openai:gpt-4o-mini")

    assert evaluator.model == "openai:gpt-4o-mini"


@pytest.mark.asyncio
async def test_pairwise_evaluator_call_no_pairs(mocker: MockerFixture) -> None:
    """Test PairwiseEvaluator.__call__ handles empty baseline and novel lists."""
    evaluator = PairwiseEvaluator()
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {
        "context": AssayContext(
            dataset=Dataset[dict[str, str], type[None], Any](cases=[]),
            path=Path("/tmp/test.json"),
            assay_mode="evaluate",
        )
    }
    mock_item.stash = {
        deepresearcher2.plugin.AGENT_RESPONSES_KEY: [],
        BASELINE_DATASET_KEY: Dataset[dict[str, str], type[None], Any](cases=[]),
    }

    mocker.patch("deepresearcher2.plugin.logger")

    result = await evaluator(mock_item)

    assert type(result).__name__ == "Readout"
    assert result.passed is False
    assert result.details is not None
    assert result.details.get("test_cases_count") == 0
    assert result.details.get("wins_baseline") == []
    assert result.details.get("wins_novel") == []


@pytest.mark.asyncio
async def test_pairwise_evaluator_call_with_pairs(mocker: MockerFixture) -> None:
    """Test PairwiseEvaluator.__call__ runs pairwise comparison with mocked agent."""
    evaluator = PairwiseEvaluator(criterion="Test criterion")

    cases: list[Case[dict[str, str], type[None], Any]] = [
        Case(name="case_001", inputs={"query": "baseline query 1"}),
        Case(name="case_002", inputs={"query": "baseline query 2"}),
    ]
    dataset = Dataset[dict[str, str], type[None], Any](cases=cases)

    mock_response1 = mocker.MagicMock(spec=AgentRunResult)
    mock_response1.output = "novel output 1"
    mock_response2 = mocker.MagicMock(spec=AgentRunResult)
    mock_response2.output = "novel output 2"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_item.stash = {
        deepresearcher2.plugin.AGENT_RESPONSES_KEY: [mock_response1, mock_response2],
        deepresearcher2.plugin.BASELINE_DATASET_KEY: dataset,
    }

    mocker.patch("deepresearcher2.plugin.logger")

    # Mock agent.run to return "B" (novel wins) for both comparisons
    mock_run_result = mocker.MagicMock()
    mock_run_result.output = "B"
    mocker.patch.object(evaluator.agent, "run", new_callable=AsyncMock, return_value=mock_run_result)

    result = await evaluator(mock_item)

    # Novel wins both -> passed=True
    assert type(result).__name__ == "Readout"
    assert result.passed is True
    assert result.details is not None
    assert result.details.get("test_cases_count") == 2
    assert result.details.get("wins_novel") == [True, True]
    assert result.details.get("wins_baseline") == [False, False]


@pytest.mark.asyncio
async def test_pairwise_evaluator_call_baseline_wins(mocker: MockerFixture) -> None:
    """Test PairwiseEvaluator.__call__ when baseline wins more comparisons."""
    evaluator = PairwiseEvaluator(criterion="Test criterion")

    cases: list[Case[dict[str, str], type[None], Any]] = [
        Case(name="case_001", inputs={"query": "baseline query"}),
    ]
    dataset = Dataset[dict[str, str], type[None], Any](cases=cases)

    mock_response = mocker.MagicMock(spec=AgentRunResult)
    mock_response.output = "novel output"

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_item.stash = {
        deepresearcher2.plugin.AGENT_RESPONSES_KEY: [mock_response],
        deepresearcher2.plugin.BASELINE_DATASET_KEY: dataset,
    }

    mocker.patch("deepresearcher2.plugin.logger")

    # Mock agent.run to return "A" (baseline wins)
    mock_run_result = mocker.MagicMock()
    mock_run_result.output = "A"
    mocker.patch.object(evaluator.agent, "run", new_callable=AsyncMock, return_value=mock_run_result)

    result = await evaluator(mock_item)

    assert type(result).__name__ == "Readout"
    assert result.passed is False
    assert result.details is not None
    assert result.details.get("wins_novel") == [False]
    assert result.details.get("wins_baseline") == [True]


@pytest.mark.asyncio
async def test_pairwise_evaluator_call_mismatch_raises(mocker: MockerFixture) -> None:
    """Test PairwiseEvaluator.__call__ raises when baseline and novel counts differ."""
    evaluator = PairwiseEvaluator()
    cases: list[Case[dict[str, str], type[None], Any]] = [
        Case(name="case_001", inputs={"query": "baseline query"}),
    ]
    dataset = Dataset[dict[str, str], type[None], Any](cases=cases)

    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {"context": AssayContext(dataset=dataset, path=Path("/tmp/test.json"), assay_mode="evaluate")}
    mock_item.stash = {
        deepresearcher2.plugin.AGENT_RESPONSES_KEY: [],  # No novel responses
        deepresearcher2.plugin.BASELINE_DATASET_KEY: dataset,
    }

    mocker.patch("deepresearcher2.plugin.logger")

    with pytest.raises(AssertionError, match="Mismatch in response counts"):
        await evaluator(mock_item)


@pytest.mark.asyncio
async def test_pairwise_evaluator_protocol_conformance() -> None:
    """Test PairwiseEvaluator conforms to Evaluator Protocol."""
    evaluator = PairwiseEvaluator()
    # Should be callable with Item and return Coroutine[Any, Any, Readout]
    assert callable(evaluator)


# =============================================================================
# Integration-style Tests (Realistic Workflows)
# =============================================================================


def test_full_assay_workflow_with_topic_generation(mocker: MockerFixture, tmp_path: Path) -> None:
    """
    Test a realistic assay workflow similar to test_curiosity.py.

    This test verifies the complete flow:
    1. Generator creates initial cases with topics
    2. Setup injects AssayContext
    3. Agent.run() responses are captured in AGENT_RESPONSES_KEY
    4. Teardown merges responses into expected_output and serializes
    """
    dataset_path = tmp_path / "assays" / "test_curiosity" / "test_search_queries.json"

    # Define topics like in test_curiosity.py
    topics = ["pangolin trafficking networks", "molecular gastronomy", "dark kitchen economics"]

    # Generator function (similar to generate_evaluation_cases)
    def generate_cases() -> Dataset[dict[str, str], str, Any]:
        cases: list[Case[dict[str, str], str, Any]] = []
        for idx, topic in enumerate(topics):
            case = Case(name=f"case_{idx:03d}", inputs={"topic": topic}, expected_output="")
            cases.append(case)
        return Dataset[dict[str, str], str, Any](cases=cases)

    # Setup: Create mock item with generator
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.funcargs = {}
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_marker.kwargs = {"generator": generate_cases}
    mock_item.get_closest_marker.return_value = mock_marker
    mock_item.config.getoption.return_value = "new_baseline"

    mocker.patch("deepresearcher2.plugin._is_assay", return_value=True)
    mocker.patch("deepresearcher2.plugin._path", return_value=dataset_path)
    mocker.patch("deepresearcher2.plugin.logger")

    # Run setup
    pytest_runtest_setup(mock_item)

    # Verify setup injected correct context
    assert "context" in mock_item.funcargs
    assay_ctx = mock_item.funcargs["context"]
    assert len(assay_ctx.dataset.cases) == 3
    assert assay_ctx.assay_mode == "new_baseline"

    # Simulate captured Agent.run() responses (populated by pytest_runtest_call)
    mock_responses = []
    for topic in topics:
        mock_response = mocker.MagicMock(spec=AgentRunResult)
        mock_response.output = f"search for: {topic}"
        mock_responses.append(mock_response)
    mock_item.stash[deepresearcher2.plugin.AGENT_RESPONSES_KEY] = mock_responses

    # Run teardown (should merge responses and serialize in new_baseline mode)
    pytest_runtest_teardown(mock_item)

    # Verify file was created with updated data
    assert dataset_path.exists()

    # Reload and verify data integrity
    reloaded = Dataset[dict[str, str], str, Any].from_file(dataset_path)
    assert len(reloaded.cases) == 3

    for idx, topic in enumerate(topics):
        assert reloaded.cases[idx].name == f"case_{idx:03d}"
        assert reloaded.cases[idx].inputs["topic"] == topic
        assert reloaded.cases[idx].expected_output == f"search for: {topic}"


def test_response_capture_simulation(mocker: MockerFixture) -> None:
    """
    Test that the response capture mechanism correctly stores Agent outputs.

    This simulates what happens during pytest_runtest_call when Agent.run() is called.
    """
    # Setup mock item with assay marker
    mock_item = mocker.MagicMock(spec=Function)
    mock_item.stash = {}
    mock_marker = mocker.MagicMock()
    mock_item.get_closest_marker.return_value = mock_marker

    mocker.patch("deepresearcher2.plugin.logger")

    # Run the hook to initialize stash
    gen = pytest_runtest_call(mock_item)
    next(gen)  # Run until yield

    # Verify stash was initialized with empty list
    assert deepresearcher2.plugin.AGENT_RESPONSES_KEY in mock_item.stash
    responses = mock_item.stash[deepresearcher2.plugin.AGENT_RESPONSES_KEY]
    assert responses == []

    # Verify context var was set
    assert deepresearcher2.plugin._current_item_var.get() == mock_item

    # Simulate adding responses (normally done by wrapped Agent.run)
    mock_response = mocker.MagicMock(spec=AgentRunResult)
    mock_response.output = "test output"
    responses.append(mock_response)

    # Verify response was captured
    assert len(mock_item.stash[deepresearcher2.plugin.AGENT_RESPONSES_KEY]) == 1
    assert mock_item.stash[deepresearcher2.plugin.AGENT_RESPONSES_KEY][0].output == "test output"

    # Clean up
    with contextlib.suppress(StopIteration):
        next(gen)

    # Verify context var was reset
    assert _current_item_var.get() is None


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
