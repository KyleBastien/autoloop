"""Regression tests for mcp version pin (issue #59).

mcp v2.0.0 removed mcp.server.fastmcp.FastMCP. The optional dependency
must stay pinned to mcp<2 until we migrate to the fastmcp package.
"""

from __future__ import annotations

import sys
import types
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def test_pyproject_pins_mcp_below_2():
    """pyproject.toml must pin mcp<2 to avoid the broken v2.0.0 import."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    mcp_deps = data["project"]["optional-dependencies"]["mcp"]
    assert len(mcp_deps) == 1
    spec = mcp_deps[0]
    assert "<2" in spec, f"mcp dep must have upper bound <2, got: {spec}"
    assert ">=1.2" in spec, f"mcp dep must require >=1.2, got: {spec}"


def test_main_exits_when_fastmcp_missing(monkeypatch):
    """main() prints install instructions and exits 1 when FastMCP is missing."""
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server", None)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)

    from importlib import reload

    import autoloop.mcp_server

    reload(autoloop.mcp_server)

    captured = StringIO()
    with patch("sys.stdout", captured), pytest.raises(SystemExit, match="1"):
        autoloop.mcp_server.main()

    output = captured.getvalue()
    assert "mcp" in output
    assert "uv tool install" in output


def test_main_succeeds_with_mcp_v1(monkeypatch):
    """main() creates a FastMCP server when mcp.server.fastmcp is importable."""
    fake_mcp = types.ModuleType("mcp")
    fake_server = types.ModuleType("mcp.server")
    fake_fastmcp = types.ModuleType("mcp.server.fastmcp")

    class FakeFastMCP:
        def __init__(self, name):
            self.name = name
            self.tools = {}

        def tool(self):
            def decorator(fn):
                self.tools[fn.__name__] = fn
                return fn

            return decorator

        def run(self):
            pass

    fake_fastmcp.FastMCP = FakeFastMCP

    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_fastmcp)

    from importlib import reload

    import autoloop.mcp_server

    reload(autoloop.mcp_server)
    autoloop.mcp_server.main()
