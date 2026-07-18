"""Issue-source abstraction: GitHub (via `gh`) or Linear (via GraphQL).

Every module resolves its client with ``get_source(cfg)`` and works in terms of
the normalized issue dict the pipeline already uses::

    {"number": int|str, "title": str, "body": str,
     "labels": [{"name": str}], "comments": [{"body": str}], "state": "OPEN"|"CLOSED"}

GitHub issue numbers are ints; Linear identifiers (``ENG-123``) are strings. PR
operations are NOT part of this interface — PRs always live on GitHub.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

from autoloop.config import REPO_DIR

_REF_RE = re.compile(r"#(\d+)|\b([A-Z][A-Z0-9]*-\d+)\b")


def linear_api_key() -> str:
    """Return the Linear API key: env ``LINEAR_API_KEY`` wins, else a
    ``LINEAR_API_KEY=`` line in ``<repo>/.env`` (git-ignored)."""
    key = os.environ.get("LINEAR_API_KEY")
    if key:
        return key
    env_file = REPO_DIR / ".env"
    if env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if line.startswith("LINEAR_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def parse_refs(text: str) -> list[str | int]:
    """Return every issue reference in *text*, GitHub ``#N`` and Linear ``ABC-123``.

    GitHub refs come back as ints (``42``); Linear identifiers as strings.
    """
    if not text:
        return []
    refs: list[str | int] = []
    for gh, lin in _REF_RE.findall(text):
        refs.append(int(gh) if gh else lin)
    return refs


def first_ref(text: str) -> str | int | None:
    """First issue reference in *text*, or None."""
    refs = parse_refs(text)
    return refs[0] if refs else None


@runtime_checkable
class IssueSource(Protocol):
    """Backend-agnostic issue operations. Numbers are int (GitHub) or str (Linear)."""

    def list_issues(
        self, *, labels: list[str] | None = None, state: str = "open", limit: int = 50
    ) -> list[dict]: ...
    def get_issue(self, number, *, include_comments: bool = False) -> dict | None: ...
    def get_state(self, number) -> str: ...
    def create_issue(self, title: str, body: str) -> str | int | None: ...
    def edit_issue(
        self,
        number,
        *,
        title: str | None = None,
        body: str | None = None,
        add_labels: list[str] | None = None,
        remove_labels: list[str] | None = None,
    ) -> None: ...
    def comment(self, number, body: str) -> None: ...
    def close_issue(self, number) -> None: ...
    def ref(self, number) -> str: ...
    def create_labels(self, labels: list[tuple[str, str, str]]) -> None: ...


# --------------------------------------------------------------------------- #
# GitHub
# --------------------------------------------------------------------------- #


class GitHubSource:
    """Issue operations backed by the ``gh`` CLI."""

    def __init__(self, repo: str) -> None:
        self.repo = repo

    def list_issues(self, *, labels=None, state="open", limit=50) -> list[dict]:
        cmd = [
            "gh", "issue", "list", "--repo", self.repo, "--state", state,
            "--json", "number,title,body,labels", "--limit", str(limit),
        ]  # fmt: skip
        for lbl in labels or []:
            cmd += ["--label", lbl]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return []
        return json.loads(result.stdout)

    def get_issue(self, number, *, include_comments=False) -> dict | None:
        fields = "number,title,body,labels"
        if include_comments:
            fields += ",comments"
        result = subprocess.run(
            ["gh", "issue", "view", str(number), "--repo", self.repo, "--json", fields],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)

    def get_state(self, number) -> str:
        result = subprocess.run(
            ["gh", "issue", "view", str(number), "--repo", self.repo, "--json", "state"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return ""
        return json.loads(result.stdout).get("state", "")

    def create_issue(self, title, body) -> int | None:
        result = subprocess.run(
            ["gh", "issue", "create", "--repo", self.repo, "--title", title, "--body", body],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        return int(result.stdout.strip().rstrip("/").split("/")[-1])

    def edit_issue(
        self, number, *, title=None, body=None, add_labels=None, remove_labels=None
    ) -> None:
        args = ["gh", "issue", "edit", str(number), "--repo", self.repo]
        if title is not None:
            args += ["--title", title]
        if body is not None:
            args += ["--body", body]
        if add_labels:
            args += ["--add-label", ",".join(add_labels)]
        if remove_labels:
            args += ["--remove-label", ",".join(remove_labels)]
        if len(args) == 6:  # base argv only — nothing to change
            return
        subprocess.run(args, capture_output=True, text=True)

    def comment(self, number, body) -> None:
        subprocess.run(
            ["gh", "issue", "comment", str(number), "--repo", self.repo, "--body", body],
            capture_output=True,
            text=True,
        )

    def close_issue(self, number) -> None:
        subprocess.run(
            ["gh", "issue", "close", str(number), "--repo", self.repo],
            capture_output=True,
            text=True,
        )

    def ref(self, number) -> str:
        return f"#{number}"

    def create_labels(self, labels) -> None:
        for name, color, description in labels:
            subprocess.run(
                [
                    "gh",
                    "label",
                    "create",
                    name,
                    "--repo",
                    self.repo,
                    "--color",
                    color,
                    "--description",
                    description,
                    "--force",
                ],  # fmt: skip
                capture_output=True,
                text=True,
            )


# --------------------------------------------------------------------------- #
# Linear
# --------------------------------------------------------------------------- #

_LINEAR_URL = "https://api.linear.app/graphql"
_CLOSED_TYPES = {"completed", "canceled"}
_RETRY_STATUS = {429, 500, 502, 503, 504}  # transient; retry with backoff


class LinearSource:
    """Issue operations backed by the Linear GraphQL API (stdlib urllib, no deps).

    Lifecycle labels (ready/in-progress/...) are plain Linear labels, matching the
    GitHub model. Closing an issue moves it to the team's first completed-type
    workflow state.
    """

    def __init__(self, team_key: str, api_key: str) -> None:
        self.team_key = team_key
        self.api_key = api_key
        self._team_id: str | None = None
        self._label_ids: dict[str, str] | None = None
        self._done_state: str | None = None

    # --- transport ---

    def _gql(self, query: str, variables: dict | None = None, *, attempts: int = 4) -> dict:
        payload = json.dumps({"query": query, "variables": variables or {}}).encode()
        headers = {"Authorization": self.api_key, "Content-Type": "application/json"}
        for attempt in range(attempts):
            req = urllib.request.Request(_LINEAR_URL, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (fixed https)
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                # Retry transient gateway/rate-limit errors; a single blip must
                # not abort a whole triage/implement batch.
                if e.code in _RETRY_STATUS and attempt < attempts - 1:
                    time.sleep(2**attempt)
                    continue
                raise
            except urllib.error.URLError:
                if attempt < attempts - 1:
                    time.sleep(2**attempt)
                    continue
                raise
        if data.get("errors"):
            raise RuntimeError(f"Linear API error: {data['errors']}")
        return data["data"]

    # --- lazy id resolution (cached per instance) ---

    def _team(self) -> str:
        if self._team_id is None:
            data = self._gql(
                "query($k:String!){teams(filter:{key:{eq:$k}}){nodes{id}}}",
                {"k": self.team_key},
            )
            nodes = data["teams"]["nodes"]
            if not nodes:
                raise RuntimeError(f"Linear team not found: {self.team_key}")
            self._team_id = nodes[0]["id"]
        return self._team_id

    def _labels(self) -> dict[str, str]:
        if self._label_ids is None:
            data = self._gql(
                "query($k:String!){issueLabels(filter:{team:{key:{eq:$k}}}){nodes{id name}}}",
                {"k": self.team_key},
            )
            self._label_ids = {n["name"]: n["id"] for n in data["issueLabels"]["nodes"]}
        return self._label_ids

    def _done_state_id(self) -> str:
        if self._done_state is None:
            data = self._gql(
                "query($k:String!){workflowStates(filter:{team:{key:{eq:$k}}}){nodes{id type}}}",
                {"k": self.team_key},
            )
            states = data["workflowStates"]["nodes"]
            done = next((s for s in states if s["type"] == "completed"), None)
            if done is None:
                raise RuntimeError(f"No completed workflow state for team {self.team_key}")
            self._done_state = done["id"]
        return self._done_state

    # --- helpers ---

    @staticmethod
    def _num(identifier) -> int:
        """Extract the numeric part of an identifier like ``ENG-123`` (or an int)."""
        return int(str(identifier).rsplit("-", 1)[-1])

    def _node(self, identifier, *, include_comments=False) -> dict | None:
        comments = "comments{nodes{body}}" if include_comments else ""
        data = self._gql(
            "query($k:String!,$n:Float!){issues(filter:{team:{key:{eq:$k}},"
            "number:{eq:$n}}){nodes{id identifier title description "
            f"state{{type}} labels{{nodes{{name}}}} {comments}}}}}}}",
            {"k": self.team_key, "n": self._num(identifier)},
        )
        nodes = data["issues"]["nodes"]
        return nodes[0] if nodes else None

    @staticmethod
    def _to_issue(node: dict) -> dict:
        issue = {
            "number": node["identifier"],
            "title": node.get("title") or "",
            "body": node.get("description") or "",
            "labels": [{"name": lbl["name"]} for lbl in node.get("labels", {}).get("nodes", [])],
            "state": "CLOSED" if node.get("state", {}).get("type") in _CLOSED_TYPES else "OPEN",
        }
        if "comments" in node:
            issue["comments"] = [{"body": c["body"]} for c in node["comments"]["nodes"]]
        return issue

    def _uuid(self, identifier) -> str | None:
        node = self._node(identifier)
        return node["id"] if node else None

    # --- IssueSource interface ---

    def list_issues(self, *, labels=None, state="open", limit=50) -> list[dict]:
        # ponytail: fetch team issues, filter labels/state client-side (small lists).
        # `first` is a soft cap; very large backlogs beyond it are not paged.
        data = self._gql(
            "query($k:String!,$first:Int!){issues(first:$first,"
            "filter:{team:{key:{eq:$k}}}){nodes{identifier title description "
            "state{type} labels{nodes{name}}}}}",
            {"k": self.team_key, "first": max(limit, 50)},
        )
        issues = [self._to_issue(n) for n in data["issues"]["nodes"]]
        if state == "open":
            issues = [i for i in issues if i["state"] == "OPEN"]
        elif state == "closed":
            issues = [i for i in issues if i["state"] == "CLOSED"]
        if labels:
            want = set(labels)
            issues = [i for i in issues if want <= {lbl["name"] for lbl in i["labels"]}]
        return issues[:limit]

    def get_issue(self, number, *, include_comments=False) -> dict | None:
        node = self._node(number, include_comments=include_comments)
        return self._to_issue(node) if node else None

    def get_state(self, number) -> str:
        issue = self.get_issue(number)
        return issue["state"] if issue else ""

    def create_issue(self, title, body) -> str | None:
        data = self._gql(
            "mutation($i:IssueCreateInput!){issueCreate(input:$i){issue{identifier}}}",
            {"i": {"teamId": self._team(), "title": title, "description": body}},
        )
        return data["issueCreate"]["issue"]["identifier"]

    def edit_issue(
        self, number, *, title=None, body=None, add_labels=None, remove_labels=None
    ) -> None:
        node = self._node(number)
        if node is None:
            return
        payload: dict = {}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["description"] = body
        if add_labels or remove_labels:
            label_ids = self._labels()
            current = {lbl["name"] for lbl in node.get("labels", {}).get("nodes", [])}
            current |= set(add_labels or [])
            current -= set(remove_labels or [])
            payload["labelIds"] = [label_ids[n] for n in current if n in label_ids]
        if payload:
            self._gql(
                "mutation($id:String!,$i:IssueUpdateInput!){issueUpdate(id:$id,input:$i){success}}",
                {"id": node["id"], "i": payload},
            )

    def comment(self, number, body) -> None:
        uuid = self._uuid(number)
        if uuid:
            self._gql(
                "mutation($i:CommentCreateInput!){commentCreate(input:$i){success}}",
                {"i": {"issueId": uuid, "body": body}},
            )

    def close_issue(self, number) -> None:
        uuid = self._uuid(number)
        if uuid:
            self._gql(
                "mutation($id:String!,$s:String!){issueUpdate(id:$id,input:{stateId:$s}){success}}",
                {"id": uuid, "s": self._done_state_id()},
            )

    def ref(self, number) -> str:
        return str(number)

    def create_labels(self, labels) -> None:
        team = self._team()
        existing = self._labels()
        for name, color, _description in labels:
            if name in existing:
                continue
            try:
                self._gql(
                    "mutation($i:IssueLabelCreateInput!){issueLabelCreate(input:$i){success}}",
                    {"i": {"teamId": team, "name": name, "color": f"#{color}"}},
                )
            except RuntimeError as e:
                # Label management may be admin-gated; don't abort init on one label.
                print(f"  could not create label {name}: {e}")
        self._label_ids = None  # force refresh on next use


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

_cache: dict[tuple, IssueSource] = {}


def get_source(cfg) -> IssueSource:
    """Return the issue source for *cfg* (cached per process by source/repo/team)."""
    source = getattr(cfg, "source", "github")
    key = (source, getattr(cfg, "repo", ""), getattr(cfg, "linear_team", ""))
    if key not in _cache:
        if source == "linear":
            _cache[key] = LinearSource(cfg.linear_team, linear_api_key())
        else:
            _cache[key] = GitHubSource(cfg.repo)
    return _cache[key]
