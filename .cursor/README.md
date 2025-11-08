## MDC Rules
* `*.mdc` == metadata + content
* Markdown file with a preamble. The preamble explain to Cursor when and for which files to apply the rules.
* Rules should be modular and as specific as possible. For example, `python-testing.mdc` is only applied in the `tests/` folder. No tokens are wasted when working in say `src/`.

## MCP servers
Create a file `mcp.json` and fill in any placeholders.
```bash
cp mcp.json.example mcp.json
```
Check that the MCP servers are running by opening the Cursor Settings and navigating to *Tools & MCP* tab. Here you can switch individual MCP servers on or off, and check the available tools.

## Context7 MCP server

In Cursor, the Context7 MCP server is invoked automatically. Do NOT add `@context7` to the prompts.

## GitHub MCP server

In Cursor, the GitHub MCP server is invoked automatically. Do NOT add `@github` to the prompts.<br>
Please fill in the `YOUR_GITHUB_PAT` placeholder in `mcp.json` for the [GitHub Personal Access Token](https://github.com/settings/tokens).<br>
Note: Use the MCP server in `Agent` mode and NOT `Ask` mode.