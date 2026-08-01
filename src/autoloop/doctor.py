"""Doctor subcommand — check runner framework."""

from __future__ import annotations

from dataclasses import dataclass
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
