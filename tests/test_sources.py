from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import autoloop.sources as sources
from autoloop.sources import (
    GitHubSource,
    LinearSource,
    first_ref,
    get_source,
    linear_api_key,
    parse_refs,
)


# --------------------------------------------------------------------------- #
# parse_refs / first_ref
# --------------------------------------------------------------------------- #


def test_parse_refs_github():
    assert parse_refs("Depends on: #43 and #44") == [43, 44]


def test_parse_refs_linear():
    assert parse_refs("Depends on: ENG-43, blocked by ABC-7") == ["ENG-43", "ABC-7"]


def test_parse_refs_mixed():
    assert parse_refs("#1 then ENG-2") == [1, "ENG-2"]


def test_parse_refs_empty():
    assert parse_refs("") == []
    assert parse_refs("nothing here") == []


def test_first_ref():
    assert first_ref("Closes #5") == 5
    assert first_ref("Closes ENG-5") == "ENG-5"
    assert first_ref("none") is None


# --------------------------------------------------------------------------- #
# GitHubSource — asserts on the gh argv it emits
# --------------------------------------------------------------------------- #


def _capture(stdout="", returncode=0):
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    return calls, fake_run


def test_github_list_issues_argv_and_parse():
    calls, fake = _capture(stdout=json.dumps([{"number": 1, "title": "x"}]))
    with patch("autoloop.sources.subprocess.run", side_effect=fake):
        result = GitHubSource("acme/widgets").list_issues(
            labels=["ready", "p1"], state="open", limit=10
        )
    assert result == [{"number": 1, "title": "x"}]
    cmd = calls[0]
    assert cmd[:3] == ["gh", "issue", "list"]
    assert cmd[cmd.index("--repo") + 1] == "acme/widgets"
    assert cmd[cmd.index("--state") + 1] == "open"
    assert cmd[cmd.index("--limit") + 1] == "10"
    assert "ready" in cmd and "p1" in cmd


def test_github_list_issues_returns_empty_on_error():
    _calls, fake = _capture(returncode=1)
    with patch("autoloop.sources.subprocess.run", side_effect=fake):
        assert GitHubSource("acme/widgets").list_issues() == []


def test_github_get_issue_includes_comments_field():
    calls, fake = _capture(stdout=json.dumps({"number": 5}))
    with patch("autoloop.sources.subprocess.run", side_effect=fake):
        GitHubSource("acme/widgets").get_issue(5, include_comments=True)
    cmd = calls[0]
    assert cmd[:3] == ["gh", "issue", "view"]
    assert "comments" in cmd[cmd.index("--json") + 1]


def test_github_get_state():
    _calls, fake = _capture(stdout=json.dumps({"state": "CLOSED"}))
    with patch("autoloop.sources.subprocess.run", side_effect=fake):
        assert GitHubSource("acme/widgets").get_state(5) == "CLOSED"


def test_github_create_issue_parses_number():
    _calls, fake = _capture(stdout="https://github.com/acme/widgets/issues/99\n")
    with patch("autoloop.sources.subprocess.run", side_effect=fake):
        assert GitHubSource("acme/widgets").create_issue("t", "b") == 99


def test_github_create_issue_none_on_failure():
    _calls, fake = _capture(returncode=1)
    with patch("autoloop.sources.subprocess.run", side_effect=fake):
        assert GitHubSource("acme/widgets").create_issue("t", "b") is None


def test_github_edit_issue_combines_flags():
    calls, fake = _capture()
    with patch("autoloop.sources.subprocess.run", side_effect=fake):
        GitHubSource("acme/widgets").edit_issue(
            5, body="new", add_labels=["ready", "p1"], remove_labels=["rejected"]
        )
    cmd = calls[0]
    assert cmd[:3] == ["gh", "issue", "edit"]
    assert cmd[cmd.index("--add-label") + 1] == "ready,p1"
    assert cmd[cmd.index("--remove-label") + 1] == "rejected"
    assert cmd[cmd.index("--body") + 1] == "new"


def test_github_edit_issue_noop_when_nothing():
    calls, fake = _capture()
    with patch("autoloop.sources.subprocess.run", side_effect=fake):
        GitHubSource("acme/widgets").edit_issue(5)
    assert calls == []


def test_github_comment_and_close_argv():
    calls, fake = _capture()
    with patch("autoloop.sources.subprocess.run", side_effect=fake):
        src = GitHubSource("acme/widgets")
        src.comment(5, "hi")
        src.close_issue(5)
    assert calls[0][:3] == ["gh", "issue", "comment"]
    assert calls[1][:3] == ["gh", "issue", "close"]


def test_github_ref():
    assert GitHubSource("acme/widgets").ref(42) == "#42"


def test_github_ref_link_is_hash():
    assert GitHubSource("acme/widgets").ref_link(42) == "#42"


def test_github_create_labels_argv():
    calls, fake = _capture()
    with patch("autoloop.sources.subprocess.run", side_effect=fake):
        GitHubSource("acme/widgets").create_labels([("ready", "0E8A16", "desc")])
    cmd = calls[0]
    assert cmd[:3] == ["gh", "label", "create"]
    assert "--force" in cmd
    assert cmd[cmd.index("--color") + 1] == "0E8A16"


# --------------------------------------------------------------------------- #
# LinearSource — GraphQL mapping with _gql stubbed
# --------------------------------------------------------------------------- #

_NODE = {
    "id": "uuid-1",
    "identifier": "ENG-12",
    "title": "Add flag",
    "description": "the body",
    "state": {"type": "started"},
    "labels": {"nodes": [{"name": "ready"}, {"name": "p1"}]},
}


def _linear(dispatch):
    src = LinearSource("ENG", "key")
    src._gql = dispatch  # type: ignore[method-assign]
    return src


def test_linear_to_issue_mapping():
    src = LinearSource("ENG", "key")
    issue = src._to_issue(_NODE)
    assert issue["number"] == "ENG-12"
    assert issue["body"] == "the body"
    assert issue["state"] == "OPEN"
    assert {lbl["name"] for lbl in issue["labels"]} == {"ready", "p1"}


def test_linear_to_issue_completed_is_closed():
    node = {**_NODE, "state": {"type": "completed"}}
    assert LinearSource("ENG", "key")._to_issue(node)["state"] == "CLOSED"


def test_linear_to_issue_duplicate_is_closed():
    # Linear's Duplicate state has type "duplicate" (not "canceled") — must count
    # as closed so duplicate-closed issues don't leak back in as ready.
    node = {**_NODE, "state": {"type": "duplicate"}}
    assert LinearSource("ENG", "key")._to_issue(node)["state"] == "CLOSED"


def test_linear_num_from_identifier():
    assert LinearSource._num("ENG-123") == 123
    assert LinearSource._num(123) == 123


def test_linear_list_issues_filters_server_side():
    captured = {}

    def dispatch(q, v=None):
        captured["filter"] = v["filter"]
        captured["first"] = v["first"]
        return {"issues": {"nodes": [_NODE]}}

    src = _linear(dispatch)
    result = src.list_issues(labels=["ready"], state="open", limit=10)

    # Maps whatever the server returns (no client-side re-filtering).
    assert [i["number"] for i in result] == ["ENG-12"]
    # State + label + team pushed into the GraphQL filter, and first == limit.
    assert captured["first"] == 10
    assert captured["filter"]["team"]["key"]["eq"] == "ENG"
    assert captured["filter"]["state"]["type"]["nin"] == list(sources._CLOSED_TYPES)
    assert captured["filter"]["and"] == [{"labels": {"some": {"name": {"eq": "ready"}}}}]


def test_linear_list_issues_closed_filter():
    captured = {}
    src = _linear(
        lambda q, v=None: captured.update(filter=v["filter"]) or {"issues": {"nodes": []}}
    )
    src.list_issues(state="closed")
    assert captured["filter"]["state"]["type"]["in"] == list(sources._CLOSED_TYPES)


def test_linear_get_issue_and_state():
    src = _linear(lambda q, v=None: {"issues": {"nodes": [_NODE]}})
    assert src.get_issue("ENG-12")["title"] == "Add flag"
    assert src.get_state("ENG-12") == "OPEN"


def test_linear_get_issue_missing_returns_none():
    src = _linear(lambda q, v=None: {"issues": {"nodes": []}})
    assert src.get_issue("ENG-99") is None


def test_linear_create_issue_returns_identifier():
    def dispatch(q, v=None):
        if "teams" in q:
            return {"teams": {"nodes": [{"id": "team-1"}]}}
        if "issueCreate" in q:
            assert v["i"]["teamId"] == "team-1"
            return {"issueCreate": {"issue": {"identifier": "ENG-50"}}}
        raise AssertionError(q)

    assert _linear(dispatch).create_issue("t", "b") == "ENG-50"


def test_linear_edit_issue_computes_label_ids():
    seen = {}

    def dispatch(q, v=None):
        if "number:{eq" in q or "number:{ eq" in q or "issues(filter" in q:
            return {"issues": {"nodes": [_NODE]}}
        if "issueLabels" in q:
            return {"issueLabels": {"nodes": [
                {"id": "L-ready", "name": "ready"},
                {"id": "L-inprog", "name": "in-progress"},
                {"id": "L-p1", "name": "p1"},
            ]}}  # fmt: skip
        if "issueUpdate" in q:
            seen["labelIds"] = set(v["i"]["labelIds"])
            return {"issueUpdate": {"success": True}}
        raise AssertionError(q)

    _linear(dispatch).edit_issue("ENG-12", remove_labels=["ready"], add_labels=["in-progress"])
    # started with {ready, p1}; remove ready, add in-progress → {in-progress, p1}
    assert seen["labelIds"] == {"L-inprog", "L-p1"}


def test_linear_close_issue_sets_done_state():
    seen = {}

    def dispatch(q, v=None):
        if "issues(filter" in q:
            return {"issues": {"nodes": [_NODE]}}
        if "workflowStates" in q:
            return {"workflowStates": {"nodes": [
                {"id": "s-todo", "type": "unstarted"},
                {"id": "s-done", "type": "completed"},
            ]}}  # fmt: skip
        if "issueUpdate" in q:
            seen["stateId"] = v["s"]
            return {"issueUpdate": {"success": True}}
        raise AssertionError(q)

    _linear(dispatch).close_issue("ENG-12")
    assert seen["stateId"] == "s-done"


def test_linear_ref_is_identifier():
    assert LinearSource("ENG", "key").ref("ENG-9") == "ENG-9"


def test_linear_ref_link_builds_markdown_url():
    src = _linear(lambda q, v=None: {"organization": {"urlKey": "pushpress"}})
    assert (
        src.ref_link("DANBOT-336") == "[DANBOT-336](https://linear.app/pushpress/issue/DANBOT-336)"
    )


class _Resp:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def read(self):
        return self._body


def test_linear_gql_retries_transient(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError("u", 504, "Gateway Timeout", {}, None)
        return _Resp(json.dumps({"data": {"ok": 1}}).encode())

    monkeypatch.setattr(sources.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sources.time, "sleep", lambda _s: None)
    assert LinearSource("ENG", "key")._gql("query{}") == {"ok": 1}
    assert calls["n"] == 3  # two 504s, third succeeds


def test_linear_gql_raises_on_non_transient(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError("u", 403, "Forbidden", {}, None)

    monkeypatch.setattr(sources.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sources.time, "sleep", lambda _s: None)
    with pytest.raises(urllib.error.HTTPError):
        LinearSource("ENG", "key")._gql("q", attempts=2)


def test_linear_gql_retries_socket_timeout(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("The read operation timed out")
        return _Resp(json.dumps({"data": {"ok": 1}}).encode())

    monkeypatch.setattr(sources.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sources.time, "sleep", lambda _s: None)
    assert LinearSource("ENG", "key")._gql("q") == {"ok": 1}
    assert calls["n"] == 2


def test_linear_gql_gives_up_after_attempts(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 504, "Gateway Timeout", {}, None)

    monkeypatch.setattr(sources.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(sources.time, "sleep", lambda _s: None)
    with pytest.raises(urllib.error.HTTPError):
        LinearSource("ENG", "key")._gql("q", attempts=3)
    assert calls["n"] == 3


def test_linear_create_labels_survives_forbidden(capsys):
    def dispatch(q, v=None):
        if "teams" in q:
            return {"teams": {"nodes": [{"id": "team-1"}]}}
        if "issueLabels" in q:
            return {"issueLabels": {"nodes": []}}
        if "issueLabelCreate" in q:
            raise RuntimeError("Linear API error: forbidden")
        raise AssertionError(q)

    # Must not raise even though label creation is denied.
    _linear(dispatch).create_labels([("ready", "0E8A16", "desc")])
    assert "could not create label ready" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# get_source factory
# --------------------------------------------------------------------------- #


def test_get_source_github(monkeypatch):
    sources._cache.clear()
    cfg = SimpleNamespace(source="github", repo="acme/widgets", linear_team="")
    src = get_source(cfg)
    assert isinstance(src, GitHubSource)
    assert src.repo == "acme/widgets"


def test_get_source_linear(monkeypatch):
    sources._cache.clear()
    monkeypatch.setenv("LINEAR_API_KEY", "sekret")
    cfg = SimpleNamespace(source="linear", repo="acme/widgets", linear_team="ENG")
    src = get_source(cfg)
    assert isinstance(src, LinearSource)
    assert src.team_key == "ENG"
    assert src.api_key == "sekret"


def test_get_source_default_is_github():
    sources._cache.clear()
    cfg = SimpleNamespace(repo="acme/widgets")  # no source attr
    assert isinstance(get_source(cfg), GitHubSource)


def test_get_source_caches(monkeypatch):
    sources._cache.clear()
    cfg = SimpleNamespace(source="github", repo="acme/widgets", linear_team="")
    assert get_source(cfg) is get_source(cfg)


# --------------------------------------------------------------------------- #
# linear_api_key: env wins, else <repo>/.env fallback
# --------------------------------------------------------------------------- #


def test_linear_api_key_env_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("LINEAR_API_KEY", "from-env")
    (tmp_path / ".env").write_text("LINEAR_API_KEY=from-file\n")
    monkeypatch.setattr(sources, "REPO_DIR", tmp_path)
    assert linear_api_key() == "from-env"


def test_linear_api_key_reads_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    (tmp_path / ".env").write_text('FOO=bar\nexport LINEAR_API_KEY="lin_api_xyz"\n')
    monkeypatch.setattr(sources, "REPO_DIR", tmp_path)
    assert linear_api_key() == "lin_api_xyz"


def test_linear_api_key_missing_returns_empty(monkeypatch, tmp_path):
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    monkeypatch.setattr(sources, "REPO_DIR", tmp_path)  # no .env here
    assert linear_api_key() == ""
