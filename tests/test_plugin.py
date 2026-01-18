#!/usr/bin/env python3


from dotenv import load_dotenv
from pytest_mock import MockerFixture

from deepresearcher2.plugin import ASSAY_MODES, pytest_addoption

load_dotenv()


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
