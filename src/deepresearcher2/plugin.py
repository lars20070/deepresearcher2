#!/usr/bin/env python3
from _pytest.config.argparsing import Parser

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
