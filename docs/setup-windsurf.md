# Setting up OMEGA with Windsurf

This guide explains how to configure OMEGA memory in Windsurf using MCP (Model Context Protocol).

## Prerequisites

- Windsurf installed
- OMEGA installed: `pip install omega-memory`
- Python 3.11+ installed

## Step 1: Locate MCP Settings

Windsurf stores MCP server configuration in a JSON file at:

**macOS/Linux:** `~/.windsurf/mcp.json`

**Windows:** `%USERPROFILE%\.windsurf\mcp.json`

If the file doesn't exist, create it manually.

## Step 2: Add OMEGA Configuration

Add the following configuration to your `mcp.json` file:

```json
{
  "mcpServers": {
    "omega-memory": {
      "command": "python3",
      "args": ["-m", "omega.server.mcp_server"]
    }
  }
}
```

**Important:** If you already have other MCP servers configured, add the `omega-memory` entry to the existing `mcpServers` object.

Example with multiple servers:

```json
{
  "mcpServers": {
    "omega-memory": {
      "command": "python3",
      "args": ["-m", "omega.server.mcp_server"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
    }
  }
}
```

## Step 3: Verify Installation

1. **Restart Windsurf** - Close and reopen Windsurf to load the new MCP configuration.

2. **Check MCP Status**
   - Open Windsurf settings (Cmd+, on macOS, Ctrl+, on Linux/Windows)
   - Navigate to the MCP/Extensions section
   - Verify that `omega-memory` appears in the list of connected servers

3. **Test the Connection**
   - Start a new chat in Windsurf
   - Ask: "What do you remember about my project?"
   - OMEGA should respond with relevant memories from previous sessions

## Step 4: Initialize OMEGA (Optional)

If you haven't run OMEGA setup yet, initialize it in your terminal:

```bash
omega setup
omega doctor
```

This ensures OMEGA is properly configured and ready to capture memories.

## Troubleshooting

### OMEGA not appearing in Windsurf

1. Verify `mcp.json` syntax is valid (use a JSON validator)
2. Ensure Python 3.11+ is installed and accessible as `python3`
3. Check that OMEGA is installed: `pip show omega-memory`
4. Restart Windsurf after making any changes

### Python command not found

If Windsurf can't find `python3`, use the full path to your Python executable:

```json
{
  "mcpServers": {
    "omega-memory": {
      "command": "/usr/bin/python3",
      "args": ["-m", "omega.server.mcp_server"]
    }
  }
}
```

Find your Python path with:
```bash
which python3
```

### No memories being recalled

1. Run `omega doctor` to verify OMEGA status
2. Check that you have memories stored: `omega list`
3. Ensure you've had previous sessions with memories captured

## Alternative: Using Claude MCP CLI

Windsurf also supports MCP configuration via the Claude MCP CLI:

```bash
claude mcp add omega-memory -- python3 -m omega.server.mcp_server
```

Then restart Windsurf to load the configuration.

## Next Steps

Once OMEGA is connected to Windsurf:

- Use `omega checkpoint` to mark important milestones
- Let OMEGA auto-capture decisions and lessons
- Ask Windsurf to remember things: "Remember that we use PostgreSQL for orders"

OMEGA works identically across all editors — only the setup differs.
