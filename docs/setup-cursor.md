# Setting up OMEGA with Cursor

This guide explains how to configure OMEGA memory in Cursor using MCP (Model Context Protocol).

## Prerequisites

- Cursor installed
- OMEGA installed: `pip install omega-memory`
- Python 3.11+ installed

## Step 1: Locate MCP Settings

Cursor stores MCP server configuration in a JSON file at:

**macOS/Linux:** `~/.cursor/mcp.json`

**Windows:** `%USERPROFILE%\.cursor\mcp.json`

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

1. **Restart Cursor** - Close and reopen Cursor to load the new MCP configuration.

2. **Check MCP Status**
   - Open Cursor settings
   - Navigate to MCP/Extensions section
   - Verify that `omega-memory` appears in the list of connected servers

3. **Test the Connection**
   - Start a new chat in Cursor
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

### OMEGA not appearing in Cursor

1. Verify `mcp.json` syntax is valid (use a JSON validator)
2. Ensure Python 3.11+ is installed and accessible as `python3`
3. Check that OMEGA is installed: `pip show omega-memory`
4. Restart Cursor after making any changes

### Python command not found

If Cursor can't find `python3`, use the full path to your Python executable:

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

## Next Steps

Once OMEGA is connected to Cursor:

- Use `omega checkpoint` to mark important milestones
- Let OMEGA auto-capture decisions and lessons
- Ask Cursor to remember things: "Remember that we use PostgreSQL for orders"

OMEGA works identically across all editors — only the setup differs.
