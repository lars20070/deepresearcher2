## MCP servers

Check that the MCP servers are running by opening a Copilot chat in `Agent` mode and clicking on the `Configure Tools ...` button in the lower right corner. In the appearing list you can toggle individual MCP servers and tools on or off.

## Context7 MCP server

In VS Code, the Context7 MCP server needs to be invoked explicitly with `#context7` in the prompts.

## GitHub MCP server

In VS Code, the GitHub MCP server needs to be invoked explicitly with `#github` in the prompts.<br>
Note: Use the MCP server in `Agent` mode and NOT `Ask` mode.

Alternatively, you can install the GitHub MCP server at user level. Go to the *Extensions* tab, search for `@mcp GitHub` and install it. The GitHub MCP server is available under the name `#github/github-mcp-server`. See `~/Library/Application Support/Code/User/mcp.json` on macOS.