"""Tests for fastmcp dependency and import fallback."""

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


def test_pyproject_requires_fastmcp_v3():
    """pyproject.toml must depend on fastmcp>=3,<4."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)

    mcp_deps = data["project"]["optional-dependencies"]["mcp"]
    assert len(mcp_deps) == 1
    spec = mcp_deps[0]
    assert "fastmcp" in spec, f"mcp extra must depend on fastmcp, got: {spec}"
    assert ">=3" in spec, f"fastmcp dep must require >=3, got: {spec}"
    assert "<4" in spec, f"fastmcp dep must have upper bound <4, got: {spec}"


def test_main_exits_when_fastmcp_missing(monkeypatch):
    """main() prints install instructions and exits 1 when fastmcp is missing."""
    monkeypatch.setitem(sys.modules, "fastmcp", None)

    from importlib import reload

    import autoloop.mcp_server

    reload(autoloop.mcp_server)

    captured = StringIO()
    with patch("sys.stdout", captured), pytest.raises(SystemExit, match="1"):
        autoloop.mcp_server.main()

    output = captured.getvalue()
    assert "fastmcp" in output
    assert "uv tool install" in output


def test_main_succeeds_with_fastmcp(monkeypatch):
    """main() creates a FastMCP server when fastmcp is importable."""
    fake_fastmcp = types.ModuleType("fastmcp")

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

    monkeypatch.setitem(sys.modules, "fastmcp", fake_fastmcp)

    from importlib import reload

    import autoloop.mcp_server

    reload(autoloop.mcp_server)
    autoloop.mcp_server.main()
