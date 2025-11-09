#!/usr/bin/env python3

import subprocess
import sys

from .logger import logger

"""
CLI wrappers in order to add these commands to [project.scripts] in pyproject.toml
"""


def uml() -> None:
    """
    Generate UML diagrams.
    """
    logger.info("Generating UML diagrams")
    cmd = [
        "uv",
        "run",
        "pyreverse",
        "-o",
        "dot",
        "-A",
        "-S",
        "--only-classnames",
        "-d",
        "./uml",
        "./src/deepresearcher2",
    ]
    result = subprocess.call(cmd)
    if result != 0:
        logger.error("Failed to generate UML diagrams with pyreverse.")
        sys.exit(result)
    cmd = [
        "dot",
        "-Tpng",
        "-Grankdir=LR",
        "-o",
        "./uml/classes.png",
        "./uml/classes.dot",
    ]
    result = subprocess.call(cmd)
    if result != 0:
        logger.error("Failed to generate UML diagram for classes with Graphviz.")
        sys.exit(result)
    cmd = [
        "dot",
        "-Tpng",
        "-Grankdir=LR",
        "-o",
        "./uml/packages.png",
        "./uml/packages.dot",
    ]
    result = subprocess.call(cmd)
    if result != 0:
        logger.error("Failed to generate UML diagram for packages with Graphviz.")
    sys.exit(result)
