from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from autoloop.auto_close_parent import (
    all_siblings_closed,
    check_and_close_parent,
    close_parent_chain,
    close_parent_with_comment,
    count_subissues,
    get_pr_body,
    parse_closes_ref,
    parse_parent_ref,
)


class FakeSource:
    """In-memory IssueSource stand-in for orchestration tests."""

    def __init__(self, *, open_issues=None, all_issues=None, bodies=None):
        self.open_issues = open_issues or []
        self.all_issues = all_issues or []
        self.bodies = bodies or {}
        self.closed: list = []
        self.comments: list = []

    def list_issues(self, *, labels=None, state="open", limit=50):
        if state == "open":
            return self.open_issues
        if state == "all":
            return self.all_issues
        return []

    def get_issue(self, ref, *, include_comments=False):
        if ref in self.bodies:
            return {"number": ref, "body": self.bodies[ref]}
        return None

    def close_issue(self, ref):
        self.closed.append(ref)

    def comment(self, ref, body):
        self.comments.append((ref, body))


# --- parse_parent_ref ---


def test_parse_parent_ref_github():
    assert parse_parent_ref("Some text\nParent issue: #42\nmore text") == 42


def test_parse_parent_ref_linear():
    assert parse_parent_ref("Parent issue: ENG-42") == "ENG-42"


def test_parse_parent_ref_missing_pattern():
    assert parse_parent_ref("This body has no parent reference at all.") is None


def test_parse_parent_ref_empty_body():
    assert parse_parent_ref("") is None


def test_parse_parent_ref_malformed():
    assert parse_parent_ref("Parent issue: #") is None
    assert parse_parent_ref("Parent issue: 42") is None


def test_parse_closes_ref_github():
    assert parse_closes_ref("This PR does stuff.\nCloses #57") == 57


def test_parse_closes_ref_linear():
    assert parse_closes_ref("Closes ENG-57") == "ENG-57"


def test_parse_closes_ref_missing_pattern():
    assert parse_closes_ref("No linked issue in this body.") is None


def test_parse_closes_ref_empty_body():
    assert parse_closes_ref("") is None


def test_parse_closes_ref_malformed():
    assert parse_closes_ref("Closes #") is None
    assert parse_closes_ref("Closes 57") is None


# --- all_siblings_closed ---


def test_all_siblings_closed_zero_open_siblings():
    src = FakeSource(open_issues=[{"number": 7, "body": "Unrelated open issue"}])
    assert all_siblings_closed(src, 55) is True


def test_all_siblings_closed_one_open_sibling():
    src = FakeSource(open_issues=[{"number": 56, "body": "Parent issue: #55"}])
    assert all_siblings_closed(src, 55) is False


def test_all_siblings_closed_ignores_other_parents():
    src = FakeSource(open_issues=[{"number": 99, "body": "Parent issue: #12"}])
    assert all_siblings_closed(src, 55) is True


def test_all_siblings_closed_empty():
    src = FakeSource(open_issues=[])
    assert all_siblings_closed(src, 55) is True


def test_all_siblings_closed_linear():
    src = FakeSource(open_issues=[{"number": "ENG-2", "body": "Parent issue: ENG-1"}])
    assert all_siblings_closed(src, "ENG-1") is False


# --- count_subissues ---


def test_count_subissues_counts_matching_parents():
    src = FakeSource(
        all_issues=[
            {"number": 56, "body": "Parent issue: #55"},
            {"number": 57, "body": "Parent issue: #55"},
            {"number": 99, "body": "Parent issue: #12"},
            {"number": 7, "body": "Unrelated"},
        ]
    )
    assert count_subissues(src, 55) == 2


def test_count_subissues_none_match():
    src = FakeSource(all_issues=[{"number": 7, "body": "Unrelated"}])
    assert count_subissues(src, 55) == 0


# --- close_parent_with_comment ---


def test_close_parent_with_comment_invokes_both():
    src = FakeSource()
    close_parent_with_comment(src, 55, 3)

    assert src.closed == [55]
    assert len(src.comments) == 1
    num, body = src.comments[0]
    assert num == 55
    assert re.search(r"Auto-closed: All \d+ sub-issues are now complete\.", body)
    assert "3" in body


# --- check_and_close_parent ---


def test_check_and_close_parent_closes_when_last_sibling():
    src = FakeSource(
        open_issues=[],
        all_issues=[
            {"number": 56, "body": "Parent issue: #55"},
            {"number": 57, "body": "Parent issue: #55"},
        ],
        bodies={57: "Parent issue: #55", 55: ""},
    )
    with patch("autoloop.auto_close_parent.get_pr_body", return_value="Closes #57"):
        result = check_and_close_parent(42, source=src, cfg=SimpleNamespace(repo="o/r"))

    assert result == 55
    assert src.closed == [55]
    assert src.comments[0][0] == 55
    assert "All 2 sub-issues are now complete." in src.comments[0][1]


def test_check_and_close_parent_skips_when_sibling_open():
    src = FakeSource(
        open_issues=[{"number": 56, "body": "Parent issue: #55"}],
        bodies={57: "Parent issue: #55"},
    )
    with patch("autoloop.auto_close_parent.get_pr_body", return_value="Closes #57"):
        result = check_and_close_parent(42, source=src, cfg=SimpleNamespace(repo="o/r"))

    assert result is None
    assert src.closed == []


def test_check_and_close_parent_skips_when_no_parent_ref():
    src = FakeSource(bodies={57: "A sub-issue with no parent reference."})
    with patch("autoloop.auto_close_parent.get_pr_body", return_value="Closes #57"):
        result = check_and_close_parent(42, source=src, cfg=SimpleNamespace(repo="o/r"))

    assert result is None
    assert src.closed == []


def test_check_and_close_parent_skips_when_no_closes_ref():
    src = FakeSource()
    with patch("autoloop.auto_close_parent.get_pr_body", return_value="No Closes reference."):
        result = check_and_close_parent(42, source=src, cfg=SimpleNamespace(repo="o/r"))

    assert result is None
    assert src.closed == []


def test_check_and_close_parent_constructs_source_from_cfg():
    fake_cfg = SimpleNamespace(repo="owner/other", source="github")
    src = FakeSource()

    with (
        patch("autoloop.auto_close_parent.get_source", return_value=src) as mock_get_source,
        patch("autoloop.auto_close_parent.get_pr_body", return_value=""),
    ):
        check_and_close_parent(1, cfg=fake_cfg)

    mock_get_source.assert_called_once_with(fake_cfg)


def test_check_and_close_parent_raises_without_source_or_cfg():
    with pytest.raises(ValueError, match="Either source or cfg must be provided"):
        check_and_close_parent(1)


def test_get_pr_body_uses_repo():
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout='{"body": "Closes #10"}')

    with patch("autoloop.auto_close_parent.subprocess.run", side_effect=fake_run):
        body = get_pr_body(7, "owner/other")

    assert body == "Closes #10"
    assert calls[0][calls[0].index("--repo") + 1] == "owner/other"


# --- close_parent_chain ---


def test_close_parent_chain_single_level():
    src = FakeSource(
        open_issues=[],
        all_issues=[
            {"number": 20, "body": "Parent issue: #10"},
            {"number": 21, "body": "Parent issue: #10"},
        ],
        bodies={20: "Parent issue: #10", 10: ""},
    )
    closed = close_parent_chain(src, 20)
    assert closed == [10]
    assert src.closed == [10]


def test_close_parent_chain_two_levels():
    src = FakeSource(
        open_issues=[],
        all_issues=[
            {"number": 30, "body": "Parent issue: #20"},
            {"number": 20, "body": "Parent issue: #10"},
        ],
        bodies={30: "Parent issue: #20", 20: "Parent issue: #10", 10: ""},
    )
    closed = close_parent_chain(src, 30)
    assert closed == [20, 10]
    assert src.closed == [20, 10]


def test_close_parent_chain_stops_when_sibling_open():
    src = FakeSource(
        open_issues=[{"number": 31, "body": "Parent issue: #20"}],
        bodies={30: "Parent issue: #20", 20: "Parent issue: #10"},
    )
    closed = close_parent_chain(src, 30)
    assert closed == []
    assert src.closed == []


def test_close_parent_chain_stops_when_no_parent():
    src = FakeSource(bodies={50: "No parent reference here"})
    closed = close_parent_chain(src, 50)
    assert closed == []


def test_close_parent_chain_respects_max_depth():
    bodies = {n: f"Parent issue: #{n - 1}" for n in range(2, 12)}
    src = FakeSource(open_issues=[], all_issues=[], bodies=bodies)
    closed = close_parent_chain(src, 10, max_depth=3)
    assert len(closed) == 3


def test_close_parent_chain_partial_close():
    """#30 → parent #20 (all closed) → parent #10 (sibling #21 open). Only #20 closes."""
    src = FakeSource(
        open_issues=[{"number": 21, "body": "Parent issue: #10"}],
        all_issues=[{"number": 30, "body": "Parent issue: #20"}],
        bodies={30: "Parent issue: #20", 20: "Parent issue: #10"},
    )
    closed = close_parent_chain(src, 30)
    assert closed == [20]


def test_close_parent_chain_linear_ids():
    src = FakeSource(
        open_issues=[],
        all_issues=[{"number": "ENG-20", "body": "Parent issue: ENG-10"}],
        bodies={"ENG-20": "Parent issue: ENG-10", "ENG-10": ""},
    )
    closed = close_parent_chain(src, "ENG-20")
    assert closed == ["ENG-10"]
