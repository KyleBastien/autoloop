"""Implement the top ready GitHub issue via Claude.

Config-driven pipeline: all repo-specific constants are read from autoloop.toml
via load_config().
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import time
from datetime import UTC, datetime

from autoloop.claude_runner import ClaudeResult, run_claude
from autoloop.config import REPO_DIR, load_config
from autoloop.sources import get_source

cfg = None

LOCKFILE = REPO_DIR / ".autoloop.lock"
LOG_FILE = REPO_DIR / "autoloop" / "run_history.jsonl"

EMPTY_BRANCH_DIAGNOSTIC = """\
No changes were produced by the implementation agent.
This usually means the agent could not act, not that the code is wrong.
Likely causes:
 1. Missing .claude/settings.json permissions (run: autoloop init to scaffold)
 2. An active Claude Code session in this directory (close it or run elsewhere)
 3. The inner claude invocation failed to start (check claude CLI auth)"""


# --- Pure functions (testable without mocking) ---


def parse_dependency_numbers(body: str) -> list[str]:
    """Extract dependency issue refs (GitHub ``#N`` or Linear ``ABC-123``) from a body."""
    matches = re.findall(r"Depends on:?\s*(#\d+|[A-Za-z][A-Za-z0-9]*-\d+)", body, re.IGNORECASE)
    return [m.lstrip("#") for m in matches]


def build_branch_name(issue: dict) -> str:
    """Slugify issue into a branch name (embeds the issue id for Linear auto-linking)."""
    slug = re.sub(r"[^a-z0-9]+", "-", issue["title"].lower()).strip("-")[:50]
    return f"autoloop/{str(issue['number']).lower()}-{slug}"


def truncate_spec(body: str, max_chars: int, issue_url: str = "") -> str:
    """Truncate an issue spec to *max_chars*, preserving the beginning."""
    if max_chars <= 0 or len(body) <= max_chars:
        return body
    truncated = body[:max_chars]
    note = "\n\n[Issue body truncated."
    if issue_url:
        note += f" Full issue: {issue_url}"
    note += "]"
    return truncated + note


def parse_and_strip_metric_targets(body: str) -> tuple[str, list[str]]:
    """Strip **Metric Target:** lines from an issue body."""
    targets = []
    cleaned_lines = []
    for line in body.splitlines(keepends=True):
        if re.match(r"\s*\*\*Metric Target:\*\*", line):
            targets.append(line.rstrip("\n").rstrip("\r"))
        else:
            cleaned_lines.append(line)
    return "".join(cleaned_lines), targets


def detect_issue_type(body: str) -> str:
    """Determine conventional commit type from issue body."""
    body_lower = (body or "").lower()
    if "## type\nbug" in body_lower:
        return "fix"
    if "## type\nrefactor" in body_lower:
        return "refactor"
    if "## type\nmigration" in body_lower:
        return "refactor"
    if "## type\ndocs" in body_lower:
        return "docs"
    if "## type\nchore" in body_lower:
        return "chore"
    return "feat"


def build_pr_body(
    issue: dict,
    attempts: int = 0,
    duration: float = 0,
    cost_usd: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> str:
    """Build the PR description markdown."""
    src = get_source(cfg)
    body = f"Closes {src.ref(issue['number'])}\n"
    parent = parent_issue_number(issue)
    if parent is not None:
        body += f"Parent: {src.ref_link(parent)}\n"
    body += (
        f"\n## Summary\n{issue['title']}\n\n## Test Plan\n- `{cfg.verify_cmd}` — all tests pass\n"
    )
    if cfg.lint_command:
        body += f"- `{cfg.lint_command}` — clean\n"
    body += "\n"
    if attempts > 0:
        body += (
            f"## AutoLoop Run Stats\n"
            f"- Attempts: {attempts}/{cfg.max_retries}\n"
            f"- Duration: {duration:.0f}s\n"
            f"- Input tokens: {input_tokens:,}\n"
            f"- Output tokens: {output_tokens:,}\n"
            f"- Cost: ${cost_usd:.2f}\n\n"
        )
    body += "Automated implementation by AutoLoop."
    return body


def collect_verification_errors(
    ahead_count: str,
    test_rc: int,
    test_out: str,
    lint_rc: int,
    changed_files: list[str],
    test_file_pattern: str = r"^tests/.*\.py$",
    issue_type: str = "feat",
    test_gate_skip_types: list[str] | None = None,
) -> list[str]:
    """Build error list from verification subprocess results."""
    errors = []
    if ahead_count.strip() == "0" or not ahead_count.strip():
        errors.append("No commits on branch")
    if test_rc != 0:
        errors.append(f"Tests failed:\n{test_out[-500:]}")
    if lint_rc != 0:
        errors.append("Lint or format check failed")
    skip_types = test_gate_skip_types if test_gate_skip_types is not None else []
    if test_file_pattern and issue_type not in skip_types:
        test_files = [f for f in changed_files if re.search(test_file_pattern, f)]
        if not test_files:
            errors.append("No test files were added or modified")
    return errors


# --- Lockfile ---


def acquire_lock() -> bool:
    """Acquire lockfile. Returns False if another run is active."""
    if LOCKFILE.exists():
        try:
            pid = int(LOCKFILE.read_text().strip())
            os.kill(pid, 0)
            return False
        except (ProcessLookupError, ValueError):
            pass
    LOCKFILE.write_text(str(os.getpid()))
    return True


def release_lock():
    """Remove the lockfile."""
    LOCKFILE.unlink(missing_ok=True)


def log_run(
    issue_number: int,
    success: bool,
    attempts: int,
    duration: float,
    cost_usd: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
):
    """Append a JSON entry to the run history log."""
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "issue": issue_number,
        "success": success,
        "attempts": attempts,
        "duration_seconds": round(duration),
        "cost_usd": round(cost_usd, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# --- Active session detection ---


def detect_active_claude_session(project_dir: str | None = None) -> bool | None:
    """Check if an interactive Claude Code session is active in the project directory.

    Returns True if a session is detected, False if none found, or None if
    detection is inconclusive (tools unavailable).
    """
    if project_dir is None:
        project_dir = str(REPO_DIR)

    project_dir = os.path.realpath(project_dir)
    logging.debug("detect_active_claude_session: resolved project_dir=%s", project_dir)

    try:
        result = subprocess.run(
            ["pgrep", "-af", "claude"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return False

    pids = []
    for line in result.stdout.strip().splitlines():
        parts = line.split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if len(parts) > 1 and "--dangerously-skip-permissions" not in parts[1]:
            pids.append(pid)

    if not pids:
        return False

    if platform.system() == "Darwin":
        return _check_cwd_lsof(pids, project_dir)
    return _check_cwd_proc(pids, project_dir)


def _check_cwd_lsof(pids: list[int], project_dir: str) -> bool | None:
    """Use lsof to check if any pid has cwd matching project_dir (macOS)."""
    try:
        result = subprocess.run(
            ["lsof", "-a", "-d", "cwd", "-Fn"]
            + [item for pid in pids for item in ("-p", str(pid))],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        if line.startswith("n"):
            cwd = os.path.realpath(line[1:])
            logging.debug("detect_active_claude_session: session cwd=%s", cwd)
            if cwd == project_dir:
                return True

    return False


def _check_cwd_proc(pids: list[int], project_dir: str) -> bool | None:
    """Use /proc to check if any pid has cwd matching project_dir (Linux)."""
    checked_any = False
    for pid in pids:
        try:
            cwd = os.path.realpath(f"/proc/{pid}/cwd")
            checked_any = True
            logging.debug("detect_active_claude_session: session cwd=%s", cwd)
            if cwd == project_dir:
                return True
        except (OSError, PermissionError):
            continue

    return False if checked_any else None


# --- Subprocess functions ---


def parent_issue_number(issue: dict) -> int | str | None:
    """Extract the parent issue ref (``#N`` or ``ABC-123``) from a sub-issue body."""
    body = issue.get("body", "") or ""
    match = re.search(r"Parent issue:\s*(#\d+|[A-Za-z][A-Za-z0-9]*-\d+)", body)
    if not match:
        return None
    token = match.group(1).lstrip("#")
    return int(token) if token.isdigit() else token


def priority_rank(issue: dict) -> int:
    """Rank an issue by its priority label (p0 < p1 < p2 < unlabeled)."""
    priority_order = {"p0": 0, "p1": 1, "p2": 2}
    labels = {lbl["name"] for lbl in issue.get("labels", [])}
    for p, rank in priority_order.items():
        if p in labels:
            return rank
    return 99


def select_top_issue(issues: list[dict]) -> dict | None:
    """Pick the top issue, keeping sub-issues of one parent together."""
    if not issues:
        return None

    eligible = [i for i in issues if dependencies_met(i)]
    if not eligible:
        return None

    groups: dict[int | str, list[dict]] = {}
    standalone: list[dict] = []
    for issue in eligible:
        parent = parent_issue_number(issue)
        if parent is not None:
            groups.setdefault(parent, []).append(issue)
        else:
            standalone.append(issue)

    best_sub = None
    if groups:
        # Prefer the largest group; on a tie, the lowest parent, then lowest sub id.
        # min() over a homogeneous run (all int or all str ids) matches the old -p rule.
        max_size = max(len(v) for v in groups.values())
        best_parent = min(p for p, v in groups.items() if len(v) == max_size)
        best_sub = min(groups[best_parent], key=lambda i: i["number"])

    best_standalone = None
    if standalone:
        best_standalone = sorted(standalone, key=priority_rank)[0]

    if best_sub and best_standalone:
        if priority_rank(best_standalone) < priority_rank(best_sub):
            return best_standalone
        return best_sub

    return best_sub or best_standalone


def get_top_ready_issue(exclude: set | None = None) -> dict | None:
    """Pick the top ready issue, grouping sub-issues by parent.

    ``exclude`` skips issue numbers already attempted this run so a failing
    issue doesn't get re-picked and stall the batch.
    """
    issues = get_source(cfg).list_issues(labels=["ready"], state="open", limit=10)
    if exclude:
        issues = [i for i in issues if i["number"] not in exclude]
    return select_top_issue(issues)


def get_issue_by_number(number) -> dict | None:
    """Fetch a specific issue by number, ignoring labels and story points."""
    return get_source(cfg).get_issue(number)


def dependencies_met(issue: dict) -> bool:
    """Check if all issues in Dependencies field are closed."""
    body = issue.get("body", "") or ""
    src = get_source(cfg)
    for dep_num in parse_dependency_numbers(body):
        if src.get_state(dep_num) not in ("CLOSED", ""):
            return False
    return True


def create_branch(issue: dict) -> str:
    """Create feature branch from latest main."""
    branch = build_branch_name(issue)
    subprocess.run(["git", "checkout", "main"], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "pull", "origin", "main"], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "checkout", "-b", branch], cwd=REPO_DIR, check=True)
    return branch


def build_implementation_prompt(issue: dict) -> str:
    """Build the full prompt for the implementation agent."""
    claude_md = (REPO_DIR / "CLAUDE.md").read_text()

    data = get_source(cfg).get_issue(issue["number"], include_comments=True)
    full_context = issue["body"] or ""
    if data:
        for c in data.get("comments", []):
            body = c.get("body", "")
            if any(
                tag in body
                for tag in (
                    "Auto-triage",
                    "AutoLoop Attempt",
                    "Implementation Detail",
                    DESIGN_COMMENT_MARKER,
                )
            ):
                full_context += f"\n\n{body}"

    full_context, metric_targets = parse_and_strip_metric_targets(full_context)
    if metric_targets:
        logging.info(
            "Stripped %d metric target(s) from issue #%s: %s",
            len(metric_targets),
            issue["number"],
            metric_targets,
        )

    src = get_source(cfg)
    issue_ref = src.ref(issue["number"])
    full_context = truncate_spec(full_context, cfg.spec_truncation, src.url(issue["number"]))

    prompt = (
        f"## Task\n\n"
        f"Implement issue {issue_ref}: {issue['title']}\n\n"
        f"## Issue Details\n\n{full_context}\n\n"
        f"## Project Conventions\n\n{claude_md}\n\n"
        f"## Implementation Checklist\n\n"
        f"1. Read the files listed in 'Files to Modify'\n"
        f"2. Implement the changes described in the issue\n"
    )

    step = 3
    if cfg.test_file_pattern:
        prompt += (
            f"{step}. Write unit tests for every new/changed function. At least one test"
            f" must reach the new code through its real caller — never mock, stub or patch"
            f" a function this change itself adds or edits, or the test still passes when"
            f" the wiring is wrong. Mock only what a test cannot reach: network, clock,"
            f" filesystem, subprocess.\n"
        )
        step += 1
    prompt += f"{step}. Run `{cfg.verify_cmd}` — all tests must pass\n"
    step += 1
    if cfg.lint_command:
        prompt += f"{step}. Run `{cfg.lint_command}` — must be clean\n"
        step += 1
    prompt += f"{step}. If README.md needs updating (new tools, commands), update it\n"
    step += 1
    prompt += (
        f"{step}. Stage and commit:\n"
        f"   `git add <specific files>`\n"
        f"   `git commit -m '<type>: <description> ({issue_ref})'\n"
        f"   Types: fix (bugs), feat (features), refactor\n"
        f"   Keep first line under 70 chars\n\n"
        f"## Rules\n\n"
        f"- Never use real person or company names in test data\n"
        f"- Follow existing code patterns in this repo\n"
        f"- Do not add features beyond what the issue asks for\n"
        f"- If the change can suppress an output (a guard, a filter, an early return),"
        f" test both directions: that it stays quiet when it should, and that it still"
        f" speaks when it should. Silencing a false alarm is how a silent failure is built\n"
        f"- When code has to recognize its own output, match on a stable marker you set,"
        f" never on user-facing text — display copy gets reworded and decorated\n"
        f"- When you filter out values the system itself produces, search the codebase for"
        f" every producer of them, not only the ones the issue names\n"
        f"- Before committing, search for other call sites with the same defect the issue"
        f" describes. Fix the ones the issue covers; list the rest in the commit body\n"
    )
    if cfg.test_file_pattern or cfg.lint_command:
        skippable = " or ".join(
            part
            for part in (
                "tests" if cfg.test_file_pattern else "",
                "lint" if cfg.lint_command else "",
            )
            if part
        )
        prompt += f"- Do not skip {skippable}\n"
    prompt += "- Do not run git push\n"

    return prompt


DESIGN_PROMPT = (
    "## Task\n\n"
    "Propose an implementation design for GitHub issue #{number}: {title}\n\n"
    "## Issue Details\n\n{body}\n\n"
    "## Project Conventions\n\n{conventions}\n\n"
    "## Instructions\n\n"
    "Write a concise implementation design proposal. Describe the approach, the\n"
    "functions or files to add or change, and the key edge cases to handle.\n"
    "Do not write the code — only the design. Do not modify any files.\n"
)


DESIGN_COMMENT_MARKER = "Implementation Design:"


def design_issue(issue: dict) -> str:
    """Generate an implementation design proposal for the issue via Claude."""
    claude_md = (REPO_DIR / "CLAUDE.md").read_text()
    prompt = DESIGN_PROMPT.format(
        number=issue["number"],
        title=issue["title"],
        body=issue.get("body", "") or "",
        conventions=claude_md,
    )
    return run_claude(prompt, cfg.impl_model, cfg.impl_timeout).text


def design_required(issue: dict, require_design: bool = False) -> bool:
    """Whether the issue must pass a design review before implementation."""
    if require_design:
        return True
    labels = {lbl["name"] for lbl in issue.get("labels", [])}
    return "design-required" in labels


def has_needs_design_label(issue: dict) -> bool:
    """Whether the issue still carries the 'needs-design' label."""
    labels = {lbl["name"] for lbl in issue.get("labels", [])}
    return "needs-design" in labels


def has_design_comment(number) -> bool:
    """Check the issue's comments for an existing Implementation Design."""
    data = get_source(cfg).get_issue(number, include_comments=True)
    if not data:
        return False
    return any(DESIGN_COMMENT_MARKER in c.get("body", "") for c in data.get("comments", []))


def post_design(number, design: str):
    """Post the implementation design as a comment on the issue."""
    get_source(cfg).comment(number, f"**{DESIGN_COMMENT_MARKER}**\n\n{design}")


def add_needs_design_label(number):
    """Add the 'needs-design' label to flag the issue for human review."""
    get_source(cfg).edit_issue(number, add_labels=["needs-design"])


def design_gate(issue: dict, require_design: bool = False) -> bool:
    """Enforce the optional design review before implementation.

    Returns True if implementation should proceed, False if it should be
    skipped this run.
    """
    if not design_required(issue, require_design):
        return True

    number = issue["number"]
    if has_design_comment(number):
        if has_needs_design_label(issue):
            print(f"#{number}: design awaiting human approval, skipping.")
            return False
        return True

    print(f"#{number}: no design found, generating implementation design.")
    design = design_issue(issue)
    if design:
        post_design(number, design)
    add_needs_design_label(number)
    print(f"#{number}: design generated and needs-design added, skipping implementation.")
    return False


def build_timeout_comment(attempt: int, timeout_seconds: int) -> str:
    """Build an actionable guidance comment for implementation timeout."""
    return (
        f"**AutoLoop Attempt {attempt} failed: implementation timeout ({timeout_seconds}s)**\n\n"
        f"Possible fixes:\n"
        f"- Increase timeout: set `impl_timeout = {timeout_seconds * 2}` in autoloop.toml\n"
        f"- Or set env var: `AUTOLOOP_TIMEOUT={timeout_seconds * 2}`\n"
        f"- Decompose the issue into smaller sub-issues (target ≤ 2 story points)\n"
        f"- Add implementation hints to the issue body to reduce exploration time"
    )


def post_timeout_failure(number, attempt: int, timeout_seconds: int):
    """Post timeout failure with actionable guidance as a comment on the issue."""
    get_source(cfg).comment(number, build_timeout_comment(attempt, timeout_seconds))


def post_attempt_failure(number, attempt: int, errors: str):
    """Post verification failure as a comment on the issue."""
    comment = f"**AutoLoop Attempt {attempt} failed:**\n\n```\n{errors[-2000:]}\n```"
    get_source(cfg).comment(number, comment)


def implement(issue: dict, previous_errors: str | None = None) -> ClaudeResult:
    """Run Claude to implement the issue. Optionally includes prior failure context."""
    prompt = build_implementation_prompt(issue)
    if previous_errors:
        prompt += (
            f"\n\n## Previous Attempt Failed\n\n"
            f"The last implementation attempt failed verification with these errors:\n"
            f"```\n{previous_errors[-2000:]}\n```\n\n"
            f"Fix these specific issues. Do not start from scratch"
            f" — build on what's already there.\n"
        )
    return run_claude(prompt, cfg.impl_model, cfg.impl_timeout)


def is_branch_empty(branch: str) -> bool:
    """Return True if the branch has zero commits ahead of main."""
    result = subprocess.run(
        ["git", "rev-list", "--count", f"main..{branch}"],
        capture_output=True,
        text=True,
        cwd=REPO_DIR,
    )
    count = result.stdout.strip() if result.returncode == 0 else ""
    return count == "0" or count == ""


def verify_implementation(branch: str, issue_body: str = "") -> tuple[bool, str]:
    """Verify the agent actually produced valid work."""
    ahead = subprocess.run(
        ["git", "rev-list", "--count", f"main..{branch}"],
        capture_output=True,
        text=True,
        cwd=REPO_DIR,
    )

    # A verify command that times out is a FAILED ATTEMPT, not a run-killer:
    # catch TimeoutExpired so the retry loop keeps going (it used to propagate
    # to the outer except, abandoning the issue and skipping remaining retries).
    def _run_check(cmd) -> tuple[int, str]:
        try:
            r = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=REPO_DIR,
                timeout=cfg.test_timeout,
            )
            return r.returncode, r.stdout
        except subprocess.TimeoutExpired:
            return 1, f"`{cmd}` timed out after {cfg.test_timeout}s"

    test_rc, test_out = _run_check(cfg.verify_cmd)
    # The configured lint command covers format too, and is optional: repos that
    # set none skip the check rather than failing it.
    lint_rc = _run_check(cfg.lint_command)[0] if cfg.lint_command else 0
    diff = subprocess.run(
        ["git", "diff", "--name-only", "main"],
        capture_output=True,
        text=True,
        cwd=REPO_DIR,
    )
    changed = [f for f in diff.stdout.strip().split("\n") if f]

    errors = collect_verification_errors(
        ahead_count=ahead.stdout if ahead.returncode == 0 else "",
        test_rc=test_rc,
        test_out=test_out,
        lint_rc=lint_rc,
        changed_files=changed,
        test_file_pattern=cfg.test_file_pattern,
        issue_type=detect_issue_type(issue_body),
        test_gate_skip_types=cfg.test_gate_skip_types,
    )
    if errors:
        return False, "\n".join(errors)
    return True, ""


REVIEW_PROMPT = """\
Review this implementation against the original issue.

Issue #{number}: {title}
{issue_body}

Diff:
{diff}

Evaluate:
1. Does the implementation satisfy each acceptance criterion?
2. Are the tests meaningful (not just pass-through stubs)?
3. Does any test mock, stub or patch a function this diff itself adds or changes?
   Name it — such a test passes even when the wiring is wrong, so the criterion it
   claims to cover is not actually covered.
4. If the change can suppress an output, is there a test proving it still produces
   that output when it should?
5. If the change recognizes the system's own output, does it match a stable marker
   rather than user-facing text, and does it account for every producer?
6. Does the code follow existing patterns in the codebase?

Respond with JSON only:
{{
  "approved": true | false,
  "issues": ["issue 1", "issue 2"],
  "summary": "one line"
}}
"""


def parse_review_response(text: str) -> tuple[bool, str]:
    """Parse the review verdict JSON into an (approved, feedback) pair."""
    stripped = text.strip()
    if "```" in stripped:
        stripped = stripped.split("```")[1].replace("json", "").strip()
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, IndexError):
        return False, f"Review response was not valid JSON:\n{text[:500]}"
    if not isinstance(data, dict) or "approved" not in data:
        return False, f"Review response was malformed:\n{text[:500]}"
    if data.get("approved"):
        return True, data.get("summary", "") or ""
    issues = data.get("issues") or []
    if isinstance(issues, list) and issues:
        feedback = "Review found issues:\n" + "\n".join(f"- {i}" for i in issues)
    else:
        feedback = data.get("summary") or "Review rejected the implementation."
    return False, feedback


def review_implementation(issue: dict, branch: str) -> tuple[bool, str]:
    """Review the implementation for semantic quality via Claude."""
    diff = subprocess.run(
        ["git", "diff", f"main..{branch}"],
        capture_output=True,
        text=True,
        cwd=REPO_DIR,
    ).stdout

    prompt = REVIEW_PROMPT.format(
        number=issue["number"],
        title=issue["title"],
        issue_body=issue.get("body", "") or "",
        diff=diff[:8000],
    )
    result = run_claude(prompt, cfg.impl_model, cfg.impl_timeout)
    if not result.success:
        return False, "Review call failed (timeout or non-zero exit)."
    return parse_review_response(result.text)


def ensure_clean_main():
    """Reset to a clean main branch, discarding any leftover state."""
    subprocess.run(["git", "checkout", "--", "."], cwd=REPO_DIR)
    subprocess.run(["git", "checkout", "main"], cwd=REPO_DIR)
    subprocess.run(["git", "pull", "--ff-only", "origin", "main"], cwd=REPO_DIR)


def cleanup_branch(branch: str):
    """Delete failed branch locally and remotely."""
    subprocess.run(["git", "checkout", "main"], cwd=REPO_DIR)
    subprocess.run(["git", "branch", "-D", branch], cwd=REPO_DIR)
    subprocess.run(
        ["git", "push", "origin", "--delete", branch],
        cwd=REPO_DIR,
        capture_output=True,
    )


def create_pr(
    issue: dict,
    branch: str,
    attempts: int = 0,
    duration: float = 0,
    cost_usd: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
):
    """Create PR with conventional format."""
    issue_type = detect_issue_type(issue.get("body", ""))
    title = f"{issue_type}: {issue['title'][:60]} ({get_source(cfg).ref(issue['number'])})"
    body = build_pr_body(
        issue,
        attempts=attempts,
        duration=duration,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            cfg.repo,
            "--title",
            title,
            "--body",
            body,
            "--head",
            branch,
            "--base",
            "main",
            "--assignee",
            cfg.pr_reviewer,
        ],
        cwd=REPO_DIR,
    )


def unblock_ready_issues():
    """Re-check blocked issues and restore ready label if deps are met."""
    src = get_source(cfg)
    for issue in src.list_issues(labels=["blocked"], state="open", limit=50):
        if dependencies_met(issue):
            src.edit_issue(issue["number"], remove_labels=["blocked"], add_labels=["ready"])
            print(f"  Unblocked #{issue['number']}: {issue['title']}")


def cleanup_merged_labels():
    """Remove in-review label from closed issues whose PR already merged."""
    src = get_source(cfg)
    for issue in src.list_issues(labels=["in-review"], state="closed", limit=50):
        src.edit_issue(issue["number"], remove_labels=["in-review"])


def post_in_progress_comment(number):
    """Comment on the issue noting the bot has started implementing it."""
    get_source(cfg).comment(
        number,
        "**AutoLoop:** The implementation bot has started working on "
        "this issue. It will open a PR when implementation is complete.",
    )


def label_in_review(number):
    """Move issue from in-progress to in-review."""
    get_source(cfg).edit_issue(number, remove_labels=["in-progress"], add_labels=["in-review"])


# --- Orchestration ---


def implement_single_issue(issue: dict, require_design: bool = False) -> bool:
    """Implement one issue end-to-end. Returns True if PR created successfully."""
    try:
        from autoloop.config import touches_protected_path
        from autoloop.create_issue import extract_files_from_spec
        from autoloop.triage_issues import find_duplicate, mark_duplicate

        # Dedup safety net: the ready backlog can hold pre-existing duplicate
        # pairs (both approved before triage-time dedup, or approved together).
        # Catch them here before wasting an implement + opening a duplicate PR.
        dup_ref, _dup_result = find_duplicate(issue, cfg)
        if dup_ref:
            print(f"  #{issue['number']}: duplicate of {dup_ref}, skipping.")
            mark_duplicate(issue["number"], dup_ref, cfg)
            return False

        body = issue.get("body") or ""
        mentioned_files = extract_files_from_spec(body)
        if touches_protected_path(mentioned_files, cfg.protected_paths):
            print(f"  #{issue['number']}: touches protected path, skipping.")
            get_source(cfg).edit_issue(issue["number"], add_labels=["needs-human"])
            return False

        ensure_clean_main()

        if not design_gate(issue, require_design):
            return False

        start_time = time.time()
        claude_results: list[ClaudeResult] = []
        final_attempt = 0
        success = False

        print(f"Implementing #{issue['number']}: {issue['title']}")

        get_source(cfg).edit_issue(
            issue["number"], remove_labels=["ready"], add_labels=["in-progress"]
        )
        post_in_progress_comment(issue["number"])

        branch = create_branch(issue)
        print(f"  Branch: {branch}")

        last_errors = None
        empty_branch_failure = False
        timeout_failure = False
        for attempt in range(1, cfg.max_retries + 1):
            print(f"  Attempt {attempt}/{cfg.max_retries}...")
            result = implement(issue, previous_errors=last_errors)
            claude_results.append(result)
            final_attempt = attempt

            if result.timed_out:
                print(f"  Implementation timed out after {cfg.impl_timeout}s.")
                post_timeout_failure(issue["number"], attempt, cfg.impl_timeout)
                timeout_failure = True
                break

            if is_branch_empty(branch):
                print(f"  {EMPTY_BRANCH_DIAGNOSTIC}")
                post_attempt_failure(issue["number"], attempt, EMPTY_BRANCH_DIAGNOSTIC)
                empty_branch_failure = True
                break

            valid, errors = verify_implementation(branch, issue_body=issue.get("body", ""))
            if not valid:
                print(f"  Verification failed:\n{errors}")
                last_errors = errors
                post_attempt_failure(issue["number"], attempt, errors)
                continue

            print("  Verification passed. Reviewing implementation...")
            approved, feedback = review_implementation(issue, branch)
            if not approved:
                print(f"  Review failed:\n{feedback}")
                last_errors = feedback
                post_attempt_failure(issue["number"], attempt, feedback)
                continue

            success = True
            print("  Review passed.")
            break

        elapsed = time.time() - start_time
        total_cost = sum(r.cost_usd for r in claude_results)
        total_input = sum(r.input_tokens for r in claude_results)
        total_output = sum(r.output_tokens for r in claude_results)
        total_cache_read = sum(r.cache_read_tokens for r in claude_results)

        if not success:
            if timeout_failure:
                print("  Implementation timed out. Flagging needs-human; keeping ready.")
            elif empty_branch_failure:
                print("  Implementation produced no changes. Flagging needs-human; keeping ready.")
            else:
                print("  All retries exhausted. Flagging needs-human; keeping ready.")
            # Keep 'ready' so the agent keeps taking swings at it on future runs
            # (needs-human is just a visibility flag). Within a single run the
            # attempted-set skip prevents re-picking it, so the batch still moves on.
            get_source(cfg).edit_issue(
                issue["number"],
                remove_labels=["in-progress"],
                add_labels=["ready", "needs-human"],
            )
            cleanup_branch(branch)
            log_run(
                issue["number"],
                False,
                final_attempt,
                elapsed,
                total_cost,
                total_input,
                total_output,
                total_cache_read,
            )
            return False

        subprocess.run(["git", "push", "-u", "origin", branch], cwd=REPO_DIR)
        create_pr(
            issue,
            branch,
            attempts=final_attempt,
            duration=elapsed,
            cost_usd=total_cost,
            input_tokens=total_input,
            output_tokens=total_output,
        )
        label_in_review(issue["number"])
        print(f"  PR created for #{issue['number']}.")

        subprocess.run(["git", "checkout", "main"], cwd=REPO_DIR)

        print(f"\n--- AutoLoop Run Stats (#{issue['number']}) ---")
        print(f"  Duration: {elapsed:.0f}s")
        print(f"  Claude calls: {len(claude_results)}")
        print(f"  Input tokens: {total_input:,}")
        print(f"  Output tokens: {total_output:,}")
        print(f"  Cost: ${total_cost:.2f}")
        log_run(
            issue["number"],
            True,
            final_attempt,
            elapsed,
            total_cost,
            total_input,
            total_output,
            total_cache_read,
        )

        return True
    except Exception:
        logging.exception("implement_single_issue failed for #%s", issue.get("number"))
        # Don't orphan the issue as in-progress on an unexpected crash: put it
        # back to ready so a later run retries it, and clean up the branch.
        try:
            get_source(cfg).edit_issue(
                issue["number"], remove_labels=["in-progress"], add_labels=["ready"]
            )
            branch_name = build_branch_name(issue)
            if (
                branch_name
                in subprocess.run(
                    ["git", "branch"], capture_output=True, text=True, cwd=REPO_DIR
                ).stdout
            ):
                cleanup_branch(branch_name)
        except Exception:
            logging.exception("cleanup after failed #%s also failed", issue.get("number"))
        return False


def implement_targeted_issue(number, require_design: bool = False) -> bool:
    """Implement a specific issue by number, bypassing label and point checks."""
    issue = get_issue_by_number(number)
    if not issue:
        print(f"#{number}: could not fetch issue, aborting.")
        return False

    if not dependencies_met(issue):
        print(f"#{number}: dependencies not met, aborting.")
        return False

    success = implement_single_issue(issue, require_design=require_design)
    print(f"\nImplemented {1 if success else 0} issue(s) this run.")
    return success


def main(issue=None, max_issues=1, require_design=False):
    global cfg
    if cfg is None:
        cfg = load_config()

    logging.debug("main: resolved project_dir=%s from cfg", cfg.project_dir)
    session_detected = detect_active_claude_session(cfg.project_dir)
    if session_detected is True:
        print(
            "Active Claude Code session detected in this directory.\n"
            "Close it, or move the Claude Code session to a parent folder."
        )
        return

    if not acquire_lock():
        print("Another implementation is running. Exiting.")
        return

    try:
        cleanup_merged_labels()
        unblock_ready_issues()

        if issue is not None:
            implement_targeted_issue(issue, require_design=require_design)
            return

        implemented = 0
        attempted: set = set()
        while implemented < max_issues:
            top_issue = get_top_ready_issue(exclude=attempted)
            if not top_issue:
                print("No more ready issues.")
                break

            attempted.add(top_issue["number"])
            if implement_single_issue(top_issue, require_design=require_design):
                implemented += 1
            # On failure, continue to the next ready issue instead of stopping
            # the batch. The failed issue is skipped (attempted) this run and,
            # once labeled needs-human, drops out of the ready pool.

        print(f"\nImplemented {implemented} issue(s) this run.")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
