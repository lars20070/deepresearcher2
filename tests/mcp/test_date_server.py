#!/usr/bin/env python3
import os
from unittest.mock import AsyncMock

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pytest_mock import MockerFixture

from deepresearcher2.logger import logger
from deepresearcher2.mcp.date_server import _run_date


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


@pytest.mark.asyncio
async def test_run_date_success(mocker: MockerFixture) -> None:
    """
    Test _run_date() with successful command execution.
    """
    # Mock the subprocess to return successful output
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"Sun Nov  9 07:59:16 UTC 2025", b""))

    mock_create_subprocess = mocker.patch(
        "deepresearcher2.mcp.date_server.asyncio.create_subprocess_exec",
        return_value=mock_process,
    )

    result = await _run_date()

    assert result == "Sun Nov  9 07:59:16 UTC 2025"
    mock_create_subprocess.assert_called_once()
    call_args = mock_create_subprocess.call_args
    assert call_args[0][0] == "date"


@pytest.mark.asyncio
async def test_run_date_file_not_found(mocker: MockerFixture) -> None:
    """
    Test _run_date() when date command is not found.
    """
    # Mock the subprocess to raise FileNotFoundError
    mocker.patch(
        "deepresearcher2.mcp.date_server.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("date: command not found"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await _run_date()

    assert "'date' command not found" in str(exc_info.value)
    assert "Unix-like system" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_date_command_failure(mocker: MockerFixture) -> None:
    """
    Test _run_date() when the command fails with non-zero exit code.
    """
    # Mock the subprocess to return a failed process
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate = AsyncMock(return_value=(b"", b"Error: Invalid date format"))

    mocker.patch(
        "deepresearcher2.mcp.date_server.asyncio.create_subprocess_exec",
        return_value=mock_process,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await _run_date()

    assert "'date' command failed" in str(exc_info.value)
    assert "Error: Invalid date format" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_date_command_failure_no_stderr(mocker: MockerFixture) -> None:
    """
    Test _run_date() when the command fails but stderr is empty.
    """
    # Mock the subprocess to return a failed process with empty stderr
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate = AsyncMock(return_value=(b"", b""))

    mocker.patch(
        "deepresearcher2.mcp.date_server.asyncio.create_subprocess_exec",
        return_value=mock_process,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await _run_date()

    assert "'date' command failed" in str(exc_info.value)
    assert "Unknown error" in str(exc_info.value)
