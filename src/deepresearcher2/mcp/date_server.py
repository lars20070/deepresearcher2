#!/usr/bin/env python3
"""MCP server that provides the local date."""

import asyncio

import fastmcp

from deepresearcher2.logger import logger


def date_server() -> None:
    """
    Start an MCP server that provides the local date.

    The server exposes a tool that runs the `date` command on the terminal
    to return the current local date and time.

    The server is tested in test_date_server().
    In order to test the server manually in Claude Desktop, please extend the config as below.
    ~/Library/Application Support/Claude/claude_desktop_config.json

    {
        "mcpServers": {
            "date_server": {
                "command": "uv",
                "args": [
                    "--directory",
                    "/Users/lars/Code/deepresearcher2",
                    "run",
                    "date_server"
                ]
            }
        }
    }
    """
    server = fastmcp.FastMCP("Date Server")

    @server.tool
    async def date() -> str:
        """Get the current local date and time by running the `date` command.

        Returns:
            str: The current local date and time as a string.
        """
        logger.info("Calling 'date' tool")
        try:
            # Run the date command asynchronously
            process = await asyncio.create_subprocess_exec(
                "date",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

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
