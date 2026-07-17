"""Auto-close a parent issue once all of its sub-issues are complete.

Sub-issues carry a ``Parent issue: <ref>`` reference in their body. When the last
open sub-issue of a parent is closed, the parent is closed automatically with a
summary comment. Issues are read/written through the configured issue source
(GitHub or Linear); the triggering PR always lives on GitHub, so its body is read
directly via ``gh``.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Protocol

from autoloop.sources import get_source

_PARENT_RE = re.compile(r"Parent issue:\s*(#\d+|[A-Za-z][A-Za-z0-9]*-\d+)")
_CLOSES_RE = re.compile(r"Closes\s+(#\d+|[A-Za-z][A-Za-z0-9]*-\d+)")


class _HasRepoSource(Protocol):
    repo: str
    source: str


def get_pr_body(pr_number: int, repo: str) -> str:
    """Return the body of the given pull request (always GitHub), or ''."""
    result = subprocess.run(
        ["gh", "pr", "view", str(pr_number), "--repo", repo, "--json", "body"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return json.loads(result.stdout).get("body", "") or ""


def _coerce(token: str) -> int | str:
    token = token.lstrip("#")
    return int(token) if token.isdigit() else token


def parse_parent_ref(body: str) -> int | str | None:
    """Extract the parent issue ref from a ``Parent issue: <ref>`` line."""
    if not body:
        return None
    match = _PARENT_RE.search(body)
    return _coerce(match.group(1)) if match else None


def parse_closes_ref(body: str) -> int | str | None:
    """Extract the issue ref from a ``Closes <ref>`` reference in a PR body."""
    if not body:
        return None
    match = _CLOSES_RE.search(body)
    return _coerce(match.group(1)) if match else None


def all_siblings_closed(source, parent_ref) -> bool:
    """Return True only when no open issue references ``Parent issue: <parent_ref>``."""
    for issue in source.list_issues(state="open", limit=100):
        if parse_parent_ref(issue.get("body", "") or "") == parent_ref:
            return False
    return True


def count_subissues(source, parent_ref) -> int:
    """Return the total number of issues referencing ``Parent issue: <parent_ref>``."""
    return sum(
        1
        for issue in source.list_issues(state="all", limit=100)
        if parse_parent_ref(issue.get("body", "") or "") == parent_ref
    )


def close_parent_with_comment(source, parent_ref, sibling_count: int) -> None:
    """Close the parent issue and post an auto-close summary comment."""
    source.close_issue(parent_ref)
    source.comment(
        parent_ref,
        f"Auto-closed: All {sibling_count} sub-issues are now complete.",
    )


def _issue_body(source, ref) -> str:
    issue = source.get_issue(ref)
    return (issue.get("body", "") or "") if issue else ""


def close_parent_chain(source, issue_ref, max_depth: int = 5) -> list:
    """Walk up the parent chain, closing each parent whose sub-issues are all done.

    Returns a list of closed parent issue refs (innermost first).
    """
    closed = []
    current = issue_ref
    seen = set()
    for _ in range(max_depth):
        parent_ref = parse_parent_ref(_issue_body(source, current))
        if parent_ref is None or parent_ref in seen:
            break
        seen.add(parent_ref)
        if not all_siblings_closed(source, parent_ref):
            break
        close_parent_with_comment(source, parent_ref, count_subissues(source, parent_ref))
        closed.append(parent_ref)
        current = parent_ref
    return closed


def check_and_close_parent(
    pr_number: int,
    source=None,
    cfg: _HasRepoSource | None = None,
) -> int | str | None:
    """Close parent issues when a merged PR completes its last open sub-issue.

    Reads the (GitHub) PR body for its ``Closes <ref>`` link, then walks up the
    parent chain closing each parent whose sub-issues are all closed. Returns the
    first closed parent ref, or None when nothing was modified.
    """
    if source is None:
        if cfg is None:
            raise ValueError("Either source or cfg must be provided")
        source = get_source(cfg)

    pr_body = get_pr_body(pr_number, cfg.repo) if cfg is not None else ""
    closed_issue = parse_closes_ref(pr_body)
    if closed_issue is None:
        return None

    closed = close_parent_chain(source, closed_issue)
    return closed[0] if closed else None


def main():
    import sys

    from autoloop.config import load_config

    check_and_close_parent(int(sys.argv[1]), cfg=load_config())


if __name__ == "__main__":
    main()
