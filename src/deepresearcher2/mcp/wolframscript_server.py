#!/usr/bin/env python3
"""MCP server that provides the local date."""

import asyncio

import fastmcp

from deepresearcher2.logger import logger


def wolframscript_server() -> None:
    """
    Start an MCP server that wraps WolframScript.

    The server exposes a tool that runs the `wolframscript` command on the terminal
    to return the results.

    * Wolfram Language: symbolic programming language e.g. `Integrate[x*Sin[x], x]`
    * Wolfram Engine: kernel for running Wolfram Language code
    * WolframScript: command-line interface to Wolfram Engine
    * Mathematica: notebook interface to Wolfram Engine

    Both Wolfram Engine and WolframScript are freely available for personal use.
    https://www.wolfram.com/engine/

    The server is tested in test_wolframscript_server().
    In order to test the server manually in Claude Desktop, please extend the config as below.
    ~/Library/Application Support/Claude/claude_desktop_config.json

    {
        "mcpServers": {
            "wolframscript_server": {
                "command": "uv",
                "args": [
                    "--directory",
                    "/Users/lars/Code/deepresearcher2",
                    "run",
                    "wolframscript_server"
                ]
            }
        }
    }
    """
    server = fastmcp.FastMCP("WolframScript Server")

    @server.tool
    async def wolframscript(script: str) -> str:
        """Get the current local date and time by running the `wolframscript` command.

        Documentation for Wolfram Language:
        https://context7.com/websites/reference_wolfram_language/llms.txt

        Arguments:
            script (str): Wolfram Language script to execute.

        Returns:
            str: The current local date and time as a string.
        """
        logger.info(f"Calling 'wolframscript' tool with script: {script}")
        try:
            # Run the wolframscript command asynchronously
            process = await asyncio.create_subprocess_exec(
                "wolframscript",
                "--version",
                stdout=asyncio.subprocess.PIPE,  # Capture in pipe
                stderr=asyncio.subprocess.PIPE,  # Capture in pipe
            )
            stdout, stderr = await process.communicate()  # Read from both pipes

            if process.returncode != 0:
                error = stderr.decode() if stderr else "Unknown error"
                error_msg = f"'date' command failed: {error}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            date_output = stdout.decode().strip()
            logger.debug(f"'date' command output: {date_output}")
            return date_output
        except FileNotFoundError:
            error_msg = "'date' command not found. This tool requires a Unix-like system."
            logger.error(error_msg)
            raise RuntimeError(error_msg) from None
        except Exception as e:
            logger.error(f"Unexpected error getting date: {e}")
            raise

    server.run()
