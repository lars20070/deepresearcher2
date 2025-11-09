#!/usr/bin/env python3
import os
from unittest.mock import AsyncMock

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pytest_mock import MockerFixture

from deepresearcher2.logger import logger
from deepresearcher2.mcp.wolframscript_server import _run_wolframscript


@pytest.mark.wolframscript
@pytest.mark.asyncio
async def test_wolframscript_server() -> None:
    """
    Test the wolframscript MCP server functionality defined in deepresearcher2.mcp.wolframscript_server.wolframscript_server()

    The MCP server wraps the 'wolframscript' command to evaluate Wolfram Language code.
    The MCP server is started automatically.
    """
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "wolframscript_server"],
        env=dict(os.environ),
    )

    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        # List all available tools
        result = await session.list_tools()
        tools = result.tools
        assert len(tools) == 2
        assert tools[0].name == "evaluate"
        logger.debug(f"Available tools on wolframscript server: {[tool.name for tool in tools]}")

        # Call the evaluate tool
        result = await session.call_tool("evaluate", {"script": "Integrate[x*Sin[x], x]"})

        # Extract text from result
        content = result.content[0]
        text = getattr(content, "text", str(content))

        logger.debug(f"Script output: {text}")
        assert len(text) > 0


@pytest.mark.wolframscript
@pytest.mark.asyncio
async def test_wolframscript_server_version() -> None:
    """
    Test the wolframscript MCP server functionality defined in deepresearcher2.mcp.wolframscript_server.wolframscript_server()

    The MCP server wraps the 'wolframscript' command to return the version of WolframScript.
    The MCP server is started automatically.
    """
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "wolframscript_server"],
        env=dict(os.environ),
    )

    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        # List all available tools
        result = await session.list_tools()
        tools = result.tools
        assert len(tools) == 2
        assert tools[1].name == "version"
        logger.debug(f"Available tools on wolframscript server: {[tool.name for tool in tools]}")

        # Call the wolframscript tool
        result = await session.call_tool("version", {})

        # Extract text from result
        content = result.content[0]
        text = getattr(content, "text", str(content))

        logger.debug(f"WolframScript version: {text}")
        assert len(text) > 0
        assert any(char.isdigit() for char in text), "Version should contain digits"


@pytest.mark.asyncio
async def test_run_wolframscript_success(mocker: MockerFixture) -> None:
    """
    Test _run_wolframscript() with successful command execution.
    """
    # Mock the subprocess to return successful output
    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"WolframScript 1.13.0 for Mac OS X ARM (64-bit)", b""))

    mock_create_subprocess = mocker.patch(
        "deepresearcher2.mcp.wolframscript_server.asyncio.create_subprocess_exec",
        return_value=mock_process,
    )

    result = await _run_wolframscript(["--version"])

    assert result == "WolframScript 1.13.0 for Mac OS X ARM (64-bit)"
    mock_create_subprocess.assert_called_once()
    call_args = mock_create_subprocess.call_args
    assert call_args[0][0] == "wolframscript"
    assert call_args[0][1] == "--version"


@pytest.mark.asyncio
async def test_run_wolframscript_file_not_found(mocker: MockerFixture) -> None:
    """
    Test _run_wolframscript() when wolframscript command is not found.
    """
    # Mock the subprocess to raise FileNotFoundError
    mocker.patch(
        "deepresearcher2.mcp.wolframscript_server.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("wolframscript: command not found"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await _run_wolframscript(["--version"])

    assert "wolframscript' command not found" in str(exc_info.value)
    assert "Wolfram Engine installation" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_wolframscript_command_failure(mocker: MockerFixture) -> None:
    """
    Test _run_wolframscript() when the command fails with non-zero exit code.
    """
    # Mock the subprocess to return a failed process
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate = AsyncMock(return_value=(b"", b"Error: Invalid syntax"))

    mocker.patch(
        "deepresearcher2.mcp.wolframscript_server.asyncio.create_subprocess_exec",
        return_value=mock_process,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await _run_wolframscript(["-file", "nonexistent.wl"])

    assert "'wolframscript' command failed" in str(exc_info.value)
    assert "Error: Invalid syntax" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_wolframscript_command_failure_no_stderr(mocker: MockerFixture) -> None:
    """
    Test _run_wolframscript() when the command fails but stderr is empty.
    """
    # Mock the subprocess to return a failed process with empty stderr
    mock_process = AsyncMock()
    mock_process.returncode = 1
    mock_process.communicate = AsyncMock(return_value=(b"", b""))

    mocker.patch(
        "deepresearcher2.mcp.wolframscript_server.asyncio.create_subprocess_exec",
        return_value=mock_process,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await _run_wolframscript(["--invalid-flag"])

    assert "'wolframscript' command failed" in str(exc_info.value)
    assert "Unknown error" in str(exc_info.value)
