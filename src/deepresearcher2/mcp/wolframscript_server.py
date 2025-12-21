#!/usr/bin/env python3
"""MCP server that provides the local date."""

import asyncio
import os
import tempfile

import fastmcp

from deepresearcher2.logger import logger


async def _run_wolframscript(args: list[str]) -> str:
    """Run a wolframscript command and return the output.

    Args:
        args: List of arguments to pass to wolframscript (e.g., ["--version"] or ["-print", "-file", "/path/to/file"]).

    Returns:
        str: The decoded stdout output from the command.

    Raises:
        RuntimeError: If the command fails or is not found.
    """
    try:
        # Run the wolframscript command asynchronously
        process = await asyncio.create_subprocess_exec(
            "wolframscript",
            *args,
            stdout=asyncio.subprocess.PIPE,  # Capture in pipe
            stderr=asyncio.subprocess.PIPE,  # Capture in pipe
        )
        stdout, stderr = await process.communicate()  # Read from both pipes

        if process.returncode != 0:
            error = stderr.decode() if stderr else "Unknown error"
            error_msg = f"'wolframscript' command failed: {error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        output = stdout.decode().strip()
        return output
    except FileNotFoundError:
        error_msg = "'wolframscript' command not found. This tool requires a Wolfram Engine installation. https://www.wolfram.com/engine/"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from None
    except Exception as e:
        logger.error(f"Unexpected error running wolframscript: {e}")
        raise


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
    async def evaluate(script: str) -> str:
        """Evaluate a Wolfram Language script by running the `wolframscript -print -file <script>` command.

        Documentation for Wolfram Language:
        https://context7.com/websites/reference_wolfram_language/llms.txt

        IMPORTANT: The tool is returning the result of the last line executed in the script, and any expression printed explicitly with `Print[]`.

        <example>
          <script>
            Integrate[x*Sin[x], x]
          </script>
          <output>
            -(x*Cos[x]) + Sin[x]
          </output>
        </example>

        <example>
          <script>
            r = D[Sin[x]^2, x]
            Integrate[r^2, x]
          </script>
          <output>
            x/2 - Sin[4*x]/8
          </output>
        </example>

        <example>
          <script>
            r = D[Sin[x]^2, x]
            Print[r]
            Integrate[r^2, x]
          </script>
          <output>
            2*Cos[x]*Sin[x]
            x/2 - Sin[4*x]/8
          </output>
        </example>

        Arguments:
            script (str): Wolfram Language script to execute.

        Returns:
            str: The result of the Wolfram Language script as a string.
        """
        logger.info(f"Calling 'wolframscript' tool with script: {script}")
        # Write script to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".wl") as tmp_file:
            tmp_file.write(script)
            tmp_file_path = tmp_file.name

        try:
            output = await _run_wolframscript(["-print", "-file", tmp_file_path])
            logger.debug(f"Script output: {output}")
            return output
        finally:
            # Clean up the temporary file
            os.unlink(tmp_file_path)

    @server.tool
    async def version_wolframscript() -> str:
        """Get the version of the `wolframscript` tool.

        Returns:
            str: Version of the `wolframscript` tool.
        """
        logger.info("Running 'wolframscript -version'")
        version = await _run_wolframscript(["-version"])
        logger.debug(f"WolframScript version: {version}")
        return version

    @server.tool
    async def version_wolframengine() -> str:
        """Get the version of the Wolfram Engine.

        Returns:
            str: Version of the Wolfram Engine.
        """
        logger.info("Running 'wolframscript -code '$Version''")
        version = await _run_wolframscript(["-code", "$Version"])
        logger.debug(f"Wolfram Engine version: {version}")
        return version

    @server.tool
    async def licensetype() -> str:
        """Get the license type of the Wolfram Engine.

        Returns:
            str: License type of the Wolfram Engine e.g. 'Professional'.
        """
        logger.info("Running 'wolframscript -code '$LicenseType''")
        license_type = await _run_wolframscript(["-code", "$LicenseType"])
        logger.debug(f"Wolfram Engine license type: {license_type}")
        return license_type

    server.run()
