"""MCP server exposing the same read-only tools (Section 4: one standard, any client).

Any MCP client -- Claude Desktop, Claude Code, an IDE -- can discover and call
these tools without custom integration code. Run with:

    python -m freight_watch mcp

Requires the optional `mcp` package (v2 SDK).
"""

from __future__ import annotations

from .tools import TOOLS, FindingsStore, execute_tool


def build_server(store: FindingsStore | None = None):
    from mcp.server.mcpserver import MCPServer

    store = store or FindingsStore()
    server = MCPServer(
        name="freight-watch",
        instructions="Read-only access to the freight cost anomaly report and its context notes.",
    )

    @server.tool(description=TOOLS["list_flags"].description)
    def list_flags(route: str | None = None, month: str | None = None, status: str = "all") -> dict:
        return execute_tool(store, "list_flags", {"route": route, "month": month, "status": status})

    @server.tool(description=TOOLS["get_route_weeks"].description)
    def get_route_weeks(route: str, month: str) -> dict:
        return execute_tool(store, "get_route_weeks", {"route": route, "month": month})

    @server.tool(description=TOOLS["search_notes"].description)
    def search_notes(query: str, route: str | None = None) -> dict:
        return execute_tool(store, "search_notes", {"query": query, "route": route})

    @server.tool(description=TOOLS["get_note"].description)
    def get_note(note_id: str) -> dict:
        return execute_tool(store, "get_note", {"note_id": note_id})

    return server


def main() -> None:
    build_server().run("stdio")
