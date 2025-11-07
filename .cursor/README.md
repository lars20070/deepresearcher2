## MDC Files
* `*.mdc` == metadata + content
* Markdown file with a preamble. The preamble explain to Cursor when and for which files to apply the rules.

## MCP servers
Create a `mcp.json` file and fill in the placeholders.
```bash
cp mcp.json.example mcp.json
```

## Context7 MCP server

In Cursor, the Context7 MCP server is invoked automatically. Do NOT add `@context7` to the prompts.

For checking that the Context7 MCP server is connected:
* Open Command Palette (Ctrl/Cmd+Shift+P)
* Search for "Output: Show Output Channels"
* Select "MCP Logs" from dropdown

## GitHub MCP server

Please fill in the `GITHUB_PAT` placeholder in `mcp.json` for the [GitHub Personal Access Token](https://github.com/settings/tokens).