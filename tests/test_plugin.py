#!/usr/bin/env python3

import importlib

from pytest_mock import MockerFixture

import deepresearcher2.plugin
from deepresearcher2.plugin import (
    ASSAY_MODES,
    pytest_addoption,
    pytest_configure,
    pytest_runtest_makereport,
)


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
    assert callable(module.pytest_unconfigure)
    assert callable(module.pytest_addhooks)
    assert callable(module.pytest_sessionstart)
    assert callable(module.pytest_sessionfinish)
    assert callable(module.pytest_runtest_makereport)


def test_pytest_addoption(mocker: MockerFixture) -> None:
    """
    Test the pytest_addoption() function registers the --assay-mode option correctly.
    """
    # Create mock objects for Parser and OptionGroup
    mock_group = mocker.MagicMock()
    mock_parser = mocker.MagicMock()
    mock_parser.getgroup.return_value = mock_group

    # Call the function under test
    pytest_addoption(mock_parser)

    # Verify parser.getgroup was called with "recording"
    mock_parser.getgroup.assert_called_once_with("recording")

    # Verify addoption was called with the correct arguments
    mock_group.addoption.assert_called_once_with(
        "--assay-mode",
        action="store",
        default="evaluate",
        choices=ASSAY_MODES,
        help='Assay mode. Defaults to "evaluate".',
    )


def test_pytest_configure(mocker: MockerFixture) -> None:
    """
    Test the pytest_configure() function retrieves and logs the --assay-mode option.
    """
    # Mock the config object
    mock_config = mocker.MagicMock()
    mock_config.getoption.return_value = "evaluate"

    # Mock the logger to verify it's called
    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    # Call the function under test
    pytest_configure(mock_config)

    # Verify getoption was called with "--assay-mode"
    mock_config.getoption.assert_called_once_with("--assay-mode")

    # Verify logger.debug was called with the assay_mode value
    mock_logger.debug.assert_called_once_with("assay_mode=evaluate")


def test_pytest_configure_new_baseline(mocker: MockerFixture) -> None:
    """
    Test the pytest_configure() function with new_baseline mode.
    """
    mock_config = mocker.MagicMock()
    mock_config.getoption.return_value = "new_baseline"

    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_configure(mock_config)

    mock_config.getoption.assert_called_once_with("--assay-mode")
    mock_logger.debug.assert_called_once_with("assay_mode=new_baseline")


def test_pytest_runtest_makereport_passed(mocker: MockerFixture) -> None:
    """
    Test the pytest_runtest_makereport() function for a passing test.
    """
    mock_item = mocker.MagicMock()
    mock_item.nodeid = "tests/test_example.py::test_foo"

    mock_call = mocker.MagicMock()
    mock_call.when = "call"
    mock_call.excinfo = None  # No exception means test passed
    mock_call.duration = 0.12345

    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_makereport(mock_item, mock_call)

    # Verify logger.info was called with test details
    assert mock_logger.info.call_count == 3
    mock_logger.info.assert_any_call("Test: tests/test_example.py::test_foo")
    mock_logger.info.assert_any_call("Test Outcome: passed")
    mock_logger.info.assert_any_call("Test Duration: 0.12345 seconds")


def test_pytest_runtest_makereport_failed(mocker: MockerFixture) -> None:
    """
    Test the pytest_runtest_makereport() function for a failing test.
    """
    mock_item = mocker.MagicMock()
    mock_item.nodeid = "tests/test_example.py::test_bar"

    mock_call = mocker.MagicMock()
    mock_call.when = "call"
    mock_call.excinfo = mocker.MagicMock()  # Exception info present means test failed
    mock_call.duration = 0.5

    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_makereport(mock_item, mock_call)

    assert mock_logger.info.call_count == 3
    mock_logger.info.assert_any_call("Test: tests/test_example.py::test_bar")
    mock_logger.info.assert_any_call("Test Outcome: failed")
    mock_logger.info.assert_any_call("Test Duration: 0.50000 seconds")


def test_pytest_runtest_makereport_setup_phase(mocker: MockerFixture) -> None:
    """
    Test that pytest_runtest_makereport() ignores non-call phases.
    """
    mock_item = mocker.MagicMock()
    mock_call = mocker.MagicMock()
    mock_call.when = "setup"  # Not the "call" phase

    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_makereport(mock_item, mock_call)

    # Logger should not be called during setup phase
    mock_logger.info.assert_not_called()


def test_pytest_runtest_makereport_teardown_phase(mocker: MockerFixture) -> None:
    """
    Test that pytest_runtest_makereport() ignores teardown phase.
    """
    mock_item = mocker.MagicMock()
    mock_call = mocker.MagicMock()
    mock_call.when = "teardown"  # Not the "call" phase

    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    pytest_runtest_makereport(mock_item, mock_call)

    # Logger should not be called during teardown phase
    mock_logger.info.assert_not_called()


def test_pytest_runtest_makereport_handles_exception(mocker: MockerFixture) -> None:
    """
    Test that pytest_runtest_makereport() handles exceptions gracefully.
    """
    mock_item = mocker.MagicMock()
    # Make nodeid property raise an exception
    type(mock_item).nodeid = mocker.PropertyMock(side_effect=AttributeError("test error"))

    mock_call = mocker.MagicMock()
    mock_call.when = "call"
    mock_call.excinfo = None

    mock_logger = mocker.patch("deepresearcher2.plugin.logger")

    # Should not raise, exception is caught internally
    pytest_runtest_makereport(mock_item, mock_call)

    # Verify error was logged
    mock_logger.error.assert_called_once()
