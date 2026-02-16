"""Scene Generation MCP Server entry point for principia-cli.

Thin wrapper that imports the SAGE-based MCP server from layout_wo_robot.py.
"""
import os
import sys

# Ensure package root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layout_wo_robot import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
