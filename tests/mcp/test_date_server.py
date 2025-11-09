#!/usr/bin/env python3
import os

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from deepresearcher2.logger import logger


@pytest.mark.asyncio
async def test_date_server() -> None:
    """
    Test the date MCP server functionality defined in deepresearcher2.mcp.date_server.date_server()

    The MCP server wraps the 'date' command to return the current local date and time.
    The MCP server is started automatically.
    """
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "date_server"],
        env=dict(os.environ),
    )

    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        # List all available tools
        result = await session.list_tools()
        tools = result.tools
        assert len(tools) == 1
        assert tools[0].name == "date"
        logger.debug(f"Available tools on date server: {[tool.name for tool in tools]}")

        # Call the date tool
        result = await session.call_tool("date", {})

        # Extract text from result
        content = result.content[0]
        text = getattr(content, "text", str(content))

        logger.debug(f"Date output: {text}")
        assert len(text) > 0
        assert any(char.isdigit() for char in text), "Date output should contain digits"
