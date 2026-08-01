"""Doctor subcommand — check runner framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Check:
    name: str
    fn: Callable[[], tuple[bool, str]]
    fix_hint: str


@dataclass
class Result:
    name: str
    passed: bool
    message: str
    fix_hint: str


def check_autoloop_toml(repo_dir: Path | None = None) -> tuple[bool, str]:
    """Validate that autoloop.toml exists and parses without error."""
    from autoloop.config import load_config

    config_path = (repo_dir or Path.cwd()) / "autoloop.toml"
    try:
        load_config(config_path)
    except FileNotFoundError:
        return False, "autoloop.toml not found"
    except Exception as exc:
        return False, f"autoloop.toml invalid: {exc}"
    return True, "autoloop.toml found and valid"


def check_claude_settings(repo_dir: Path | None = None) -> tuple[bool, str]:
    """Validate that .claude/settings.json exists in the repo root."""
    settings_path = (repo_dir or Path.cwd()) / ".claude" / "settings.json"
    if not settings_path.exists():
        return False, ".claude/settings.json not found"
    return True, ".claude/settings.json found"


def get_checks(repo_dir: Path | None = None) -> list[Check]:
    """Return the default set of doctor checks."""
    return [
        Check(
            name="autoloop.toml",
            fn=lambda: check_autoloop_toml(repo_dir),
            fix_hint='run "autoloop init" to generate it',
        ),
        Check(
            name=".claude/settings.json",
            fn=lambda: check_claude_settings(repo_dir),
            fix_hint='run "autoloop init" to scaffold it, or create manually',
        ),
    ]


def run_checks(checks: list[Check]) -> list[Result]:
    """Run each check, print pass/fail, return results."""
    if not checks:
        print("No checks registered.")
        return []

    results: list[Result] = []
    for check in checks:
        try:
            passed, message = check.fn()
        except Exception as exc:
            passed = False
            message = str(exc)

        results.append(
            Result(name=check.name, passed=passed, message=message, fix_hint=check.fix_hint)
        )

        if passed:
            print(f"\033[32m✓\033[0m {check.name}: {message}")
        else:
            print(f"\033[31m✗\033[0m {check.name}: {message}")
            print(f"  hint: {check.fix_hint}")

    return results
