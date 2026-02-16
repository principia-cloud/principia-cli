"""Scene Generation MCP Server entry point for principia-cli.

Thin wrapper that imports the SAGE-based MCP server from layout_wo_robot.py.
"""
import builtins
import os
import sys

# CRITICAL: Redirect print() to stderr BEFORE any imports.
# The MCP stdio transport uses stdout for JSON-RPC messages.  There are ~280
# bare print() calls across the scene-gen codebase (room_solver, object_planner,
# etc.) that would write to stdout and corrupt the JSON-RPC stream, killing
# the MCP connection.  By patching print() to default to stderr, all debug
# output goes to the log stream while the MCP transport keeps working.
_builtin_print = builtins.print


def _safe_print(*args, **kwargs):
    if "file" not in kwargs:
        kwargs["file"] = sys.stderr
    _builtin_print(*args, **kwargs)


builtins.print = _safe_print

# Ensure package root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layout_wo_robot import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
