"""Cron 1: Triage untriaged GitHub issues via Claude."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoloop.config import AutoLoopConfig

from autoloop.claude_runner import ClaudeResult, run_claude
from autoloop.config import REPO_DIR
from autoloop.create_issue import build_issue_body
from autoloop.sources import get_source

LOG_FILE = REPO_DIR / "autoloop" / "run_history.jsonl"


def build_triage_prompt(cfg: AutoLoopConfig) -> str:
    """Build the triage prompt with config-supplied thresholds and commands."""
    return f"""\
Evaluate this GitHub issue for implementation readiness.

The project source tree and CLAUDE.md are provided below the requirements. Use
them as ground truth about the codebase:
- Validate file references. If the issue names a module, path, or function that
  does not appear in the project tree, treat the reference as invalid: lower your
  confidence in the estimate and call out the invalid reference in "reason".
- Assess feasibility. Judge whether the acceptance criteria are realistic for the
  codebase's actual architecture.
- Detect duplication. If the issue requests functionality that already exists in
  the tree, note the duplication in "reason" and reduce the readiness accordingly.

TEMPLATE REQUIREMENTS — reject if missing:
- Summary (one clear sentence)
- Type (bug/feature/refactor)
- Expected Behavior (specific and testable)
- Acceptance Criteria (at least one checkbox item)

"Files to Modify" is optional — do NOT reject for missing files.

REJECTION GUIDANCE:
- When rejecting, explain what is missing or vague at the module or function level.
- Do NOT suggest specific line numbers, variable names, or exact assertion text.
- Good: "Expected Behavior should describe observable output, not internal state"
- Bad: "add assertion 'PROFILE.md content appears at index 3 of system_prompt'"
- The goal is to tell the submitter WHAT to fix, not HOW to implement it.

SIZE ESTIMATION:
- 1 point: single file change, <50 lines
- 2 points: 1-3 files, new function + tests, <150 lines
- 3+ points: 4+ files, schema changes, new module, >150 lines

PROJECT COMMANDS:
- Test: {cfg.verify_cmd}
- Lint: {cfg.lint_command}

VERDICT:
- "not-code-work" if the issue's deliverable is not a change to this repo's code.
  This takes precedence over every other verdict — check it first, and judge it on
  the deliverable, not on how well-written the issue is. A complete, specific,
  perfectly-templated issue is still "not-code-work" if finishing it would leave
  the tree untouched. Examples: filing or updating a ticket somewhere, sending a
  message, running a query, provisioning access, or a decision. Put what the
  deliverable actually is in "reason".
- "ready" if template complete AND estimated ≤{cfg.max_story_points} points
- "needs-decomposition" if template complete BUT >{cfg.max_story_points} points
- "rejected" if template incomplete or vague

"rejected" means the issue is badly written and a rewrite could fix it. Do NOT use
it for an issue that is well written but asks for something other than a code
change — that is "not-code-work", and rewriting it cannot help.

Respond with JSON only:
{{{{
  "verdict": "ready" | "needs-decomposition" | "rejected" | "not-code-work",
  "points": 1 | 2 | 3 | 5 | 8,
  "priority": "p0" | "p1" | "p2",
  "reason": "one line",
  "files_missing": true | false,
  "decomposition": [...]
}}}}

Include "decomposition" only if verdict is "needs-decomposition".
Each sub-issue: {{{{order, title, points, depends_on, files, why_first/why_after, code_work}}}}.

"code_work" is false when that step's deliverable is not a change to this repo's
code — filing or updating a ticket elsewhere, sending a message, running a query,
provisioning access, a decision. Those steps are handed to a human instead of the
implement pipeline, so mark them honestly rather than dressing them up as code.
"""


FILE_DISCOVERY_PROMPT = """\
Given this issue and the project structure, identify files to modify and test.

Project structure:
{tree}

CLAUDE.md:
{claude_md}

Issue #{number}: {title}
{body}

Respond with JSON only:
{{
  "files_to_modify": [
    {{"path": "src/patina/example.py", "reason": "main"}},
    {{"path": "tests/test_example.py", "reason": "test coverage"}}
  ]
}}
"""

SUB_ISSUE_PROMPT = """\
Generate structured issue fields for this sub-issue of a decomposed parent.

Parent issue: #{parent_number}
Parent summary: {parent_summary}

Sub-issue: {step_title}
Files: {step_files}
Reason for ordering: {step_reason}

Respond with JSON only:
{{
  "expected_behavior": "specific, testable description",
  "acceptance_criteria": ["criterion 1", "criterion 2"]
}}

Rules:
- Expected behavior must describe observable outputs, not repeat the title.
- Acceptance criteria must be verifiable by running a test or command.
- Do not include generic criteria like "tests pass" or "lint clean".
- Reference function names and modules, not line numbers.
"""

REWRITE_PROMPT = """\
This GitHub issue was rejected by automated triage. Rewrite the issue body so it
addresses the rejection reason and passes triage on the next attempt.

REJECTION REASON:
{reason}

CURRENT ISSUE BODY:
{body}

The rewritten body MUST satisfy every template requirement:
- Summary (one clear sentence)
- Type (bug/feature/refactor)
- Expected Behavior (specific and testable — describe observable output)
- Acceptance Criteria (at least one checkbox item, each verifiable)

Rules:
- Fix only what the rejection flagged; preserve the original intent and scope.
- Keep the existing `## ` section headers.
- Reference function names and modules, not line numbers.
- Respond with the full rewritten issue body as markdown ONLY — no preamble,
  no surrounding code fences, no commentary.
"""


# --- Pure functions (testable without mocking) ---


def parse_triage_response(stdout: str) -> dict:
    """Extract JSON verdict from Claude's triage output."""
    text = stdout.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {"verdict": "rejected", "reason": "Failed to parse triage response"}


def parse_file_discovery_response(stdout: str) -> list[dict]:
    """Extract file list JSON from Claude's file discovery output."""
    text = stdout.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()
    try:
        return json.loads(text).get("files_to_modify", [])
    except (json.JSONDecodeError, IndexError):
        return []


def validate_discovered_files(files: list[dict], repo_dir: Path) -> list[dict]:
    """Filter to files that exist or are new test files."""
    return [f for f in files if (repo_dir / f["path"]).exists() or f["path"].startswith("tests/")]


def parse_sub_issue_response(stdout: str) -> dict | None:
    """Extract sub-issue fields JSON from Claude's output."""
    text = stdout.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return None
    if not isinstance(data, dict) or "expected_behavior" not in data:
        return None
    return data


def parse_rewritten_body(stdout: str) -> str | None:
    """Extract a rewritten issue body from Claude's output."""
    text = stdout.strip()
    if text.startswith("```"):
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    if "## " not in text:
        return None
    return text


def build_decomposition_comment(result: dict) -> str:
    """Build markdown table from decomposition array."""
    rows = []
    for step in result.get("decomposition", []):
        deps = ", ".join(f"Step {d}" for d in step.get("depends_on", [])) or "—"
        files = ", ".join(f"`{f}`" for f in step.get("files", []))
        rows.append(f"| {step['order']} | {step['title']} | {step['points']} | {deps} | {files} |")
    table = (
        f"**Auto-triage:** Estimated at {result['points']} points"
        f" — needs decomposition.\n\n"
        f"| Order | Sub-issue | Pts | Depends on | Files |\n"
        f"|-------|-----------|-----|------------|-------|\n" + "\n".join(rows)
    )
    why_lines = []
    for step in result.get("decomposition", []):
        reason = step.get("why_first") or step.get("why_after", "")
        if reason:
            why_lines.append(f"- Step {step['order']}: {reason}")
    if why_lines:
        table += "\n\n**Why this order:**\n" + "\n".join(why_lines)
    table += (
        "\n\nCreate sub-issues using the issue template. Use `Depends on: #N`\n"
        "(real issue numbers) in the Dependencies field. The implementation bot\n"
        "skips issues whose dependencies aren't merged yet."
    )
    return table


def build_sub_issue_summary_comment(parent_number: int, sub_issues: list[int]) -> str:
    """Build the parent comment listing the created sub-issue numbers."""
    lines = "\n".join(f"- #{n}" for n in sub_issues)
    return (
        f"**Auto-triage — Sub-issues created:**\n\n"
        f"Decomposed #{parent_number} into {len(sub_issues)} sub-issue(s):\n{lines}"
    )


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


# --- Subprocess functions ---


def load_project_context() -> tuple[str, str]:
    """Return the project source tree and CLAUDE.md contents for prompt context."""
    tree = subprocess.run(
        ["find", "src/", "tests/", "-name", "*.py", "-not", "-path", "*__pycache__*"],
        capture_output=True,
        text=True,
        cwd=REPO_DIR,
    ).stdout
    claude_md = (REPO_DIR / "CLAUDE.md").read_text()
    return tree, claude_md


def list_untriaged_issues(cfg: AutoLoopConfig) -> list[dict]:
    """Fetch open issues that have no triage labels yet."""
    triage_labels = set(cfg.triage_labels)
    issues = get_source(cfg).list_issues(state="open", limit=50)
    return [
        i for i in issues if not any(lbl["name"] in triage_labels for lbl in i.get("labels", []))
    ]


def evaluate_issue(issue: dict, cfg: AutoLoopConfig) -> tuple[dict, ClaudeResult]:
    """Run Claude to evaluate an issue against the triage prompt."""
    tree, claude_md = load_project_context()
    triage_prompt = build_triage_prompt(cfg)
    prompt = (
        triage_prompt
        + "\n\nPROJECT STRUCTURE:\n"
        + tree[: cfg.tree_truncation]
        + "\n\nCLAUDE.md:\n"
        + claude_md
        + f"\n\nIssue #{issue['number']}: {issue['title']}\n\n"
        + (issue.get("body") or "")
    )
    result = run_claude(prompt, cfg.triage_model, cfg.triage_timeout)
    if not result.success:
        verdict = {
            "verdict": "rejected",
            "reason": "Triage timed out — issue body may be too large",
        }
        return verdict, result
    return parse_triage_response(result.text), result


def discover_files(issue: dict, cfg: AutoLoopConfig) -> tuple[list[dict], ClaudeResult]:
    """Ask Claude to identify relevant files for an issue."""
    tree, claude_md = load_project_context()
    prompt = FILE_DISCOVERY_PROMPT.format(
        tree=tree[: cfg.tree_truncation],
        claude_md=claude_md,
        number=issue["number"],
        title=issue["title"],
        body=issue["body"] or "",
    )

    result = run_claude(prompt, cfg.triage_model, cfg.triage_timeout)
    if not result.success:
        return [], result

    files = parse_file_discovery_response(result.text)
    return validate_discovered_files(files, REPO_DIR), result


def enrich_issue_with_files(number: int, files: list[dict], cfg: AutoLoopConfig):
    """Comment with discovered files so the implementation agent sees them."""
    file_lines = "\n".join(f"- `{f['path']}` — {f['reason']}" for f in files)
    comment = f"**Auto-triage — File Discovery:**\n\nIdentified files to modify:\n{file_lines}"
    get_source(cfg).comment(number, comment)


def reject_issue(number: int, reason: str, cfg: AutoLoopConfig):
    """Label issue as rejected and comment with the reason."""
    src = get_source(cfg)
    src.edit_issue(number, add_labels=["rejected"])
    src.comment(number, f"**Auto-triage — Rejected:** {reason}")


def rewrite_issue_body(
    issue: dict, reason: str, cfg: AutoLoopConfig
) -> tuple[str | None, ClaudeResult]:
    """Ask Claude to rewrite a rejected issue body to address the reason."""
    prompt = REWRITE_PROMPT.format(reason=reason, body=issue.get("body") or "")
    result = run_claude(prompt, cfg.triage_model, cfg.triage_timeout)
    if not result.success:
        return None, result
    return parse_rewritten_body(result.text), result


def apply_rewrite(number: int, body: str, cfg: AutoLoopConfig):
    """Update the issue body, drop the 'rejected' label, and note the auto-fix."""
    src = get_source(cfg)
    src.edit_issue(number, body=body, remove_labels=["rejected"])
    src.comment(
        number,
        "**Auto-triage — Auto-fix:** Rewrote the issue body to address the "
        "rejection reason and re-triaging once.",
    )


def route_to_human(number: int, reason: str, cfg: AutoLoopConfig):
    """Label issue needs-human and comment why the pipeline stopped."""
    src = get_source(cfg)
    src.edit_issue(number, add_labels=["needs-human"])
    src.comment(number, f"**Auto-triage — needs-human:** {reason}")


def approve_issue(number: int, priority: str, reason: str, cfg: AutoLoopConfig):
    """Label issue as ready with priority and comment."""
    src = get_source(cfg)
    src.edit_issue(number, add_labels=["ready", priority])
    src.comment(number, f"**Auto-triage — Ready ({priority}):** {reason}")


def suggest_sub_issue_fields(
    parent_number: int,
    parent_summary: str,
    step: dict,
    cfg: AutoLoopConfig,
) -> dict | None:
    """Ask Claude for a specific Expected Behavior + Acceptance Criteria."""
    if not shutil.which("claude"):
        return None

    why = step.get("why_first") or step.get("why_after", "")
    prompt = SUB_ISSUE_PROMPT.format(
        parent_number=parent_number,
        parent_summary=parent_summary,
        step_title=step["title"],
        step_files=", ".join(step.get("files", [])),
        step_reason=why,
    )
    result = run_claude(prompt, cfg.triage_model, cfg.triage_timeout)
    if not result.success:
        return None
    return parse_sub_issue_response(result.text)


def create_sub_issues(
    parent_number: int,
    result: dict,
    cfg: AutoLoopConfig,
    parent_summary: str = "",
) -> list[int]:
    """Create sub-issues from a decomposition and return their numbers."""
    src = get_source(cfg)
    step_to_issue: dict[int, int | str] = {}
    created: list[int | str] = []
    for step in result.get("decomposition", []):
        dep_refs = [
            src.ref(step_to_issue[d]) for d in step.get("depends_on", []) if d in step_to_issue
        ]
        deps = "Depends on: " + ", ".join(dep_refs) if dep_refs else ""

        fields = suggest_sub_issue_fields(parent_number, parent_summary, step, cfg)
        if fields:
            expected = fields.get("expected_behavior") or step["title"]
            extra_criteria = "\n".join(fields.get("acceptance_criteria", []))
        else:
            expected = step["title"]
            extra_criteria = ""

        why = step.get("why_first") or step.get("why_after", "")
        code_work = step.get("code_work", True)
        body = build_issue_body(
            summary=step["title"],
            issue_type="feature",
            files="\n".join(step.get("files", [])),
            current_behavior="",
            expected=expected,
            extra_criteria=extra_criteria,
            hints=f"Sub-issue of {src.ref(parent_number)}. {why}".strip(),
            deps=deps,
            context=f"Parent issue: {src.ref(parent_number)}",
            verify_cmd=cfg.verify_cmd,
            lint_command=cfg.lint_command,
            code_work=code_work,
        )

        issue_num = src.create_issue(step["title"], body)
        if issue_num is not None:
            step_to_issue[step["order"]] = issue_num
            created.append(issue_num)
            # Label the sub-issue now from its own point estimate so it is NOT
            # re-triaged (and re-decomposed into yet another near-identical
            # child) on the next run. This is what stops the decomposition loop.
            # A non-code step goes to a human whatever its size: it can never
            # satisfy the implement gate (commits + a changed test file).
            points = step.get("points", cfg.max_story_points + 1)
            if code_work and points <= cfg.max_story_points:
                src.edit_issue(issue_num, add_labels=["ready", "p2"])
            else:
                src.edit_issue(issue_num, add_labels=["needs-human"])

    return created


def decompose_issue(
    number: int,
    result: dict,
    cfg: AutoLoopConfig,
    parent_summary: str = "",
):
    """Label the parent needs-decomposition, create sub-issues, post summary."""
    src = get_source(cfg)
    src.edit_issue(number, add_labels=["needs-decomposition"])
    src.comment(number, build_decomposition_comment(result))
    sub_issues = create_sub_issues(number, result, cfg, parent_summary)
    if sub_issues:
        src.comment(number, build_sub_issue_summary_comment(number, sub_issues))


# --- Dedup (avoid triaging/implementing the same work twice) ---

DEDUP_PROMPT = """\
You are deduplicating an engineering backlog. Decide whether the NEW issue is
essentially the SAME work (same change, same outcome) as one of the EXISTING
issues — not merely related or in the same area.

NEW issue:
{title}
{body}

EXISTING issues:
{candidates}

Respond with JSON only:
{{"duplicate_of": "<exact id from the EXISTING list>" or null}}

Only set duplicate_of when implementing the NEW issue would redo the EXISTING
one. When in doubt, use null.
"""

_STOPWORDS = {"the", "a", "an", "to", "of", "in", "into", "and", "for", "with", "add", "+"}


def _title_tokens(title: str) -> set[str]:
    """Normalized significant word set of a title, for cheap similarity."""
    words = re.findall(r"[a-z0-9]+", (title or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def is_sub_issue(issue: dict) -> bool:
    """True if the issue is itself a decomposition product (has a parent ref)."""
    return bool(re.search(r"Parent issue:", issue.get("body") or ""))


def parse_duplicate_response(text: str, valid_refs: set[str]) -> str | None:
    """Extract a validated ``duplicate_of`` ref from Claude's dedup output."""
    stripped = text.strip()
    if "```" in stripped:
        stripped = stripped.split("```")[1].replace("json", "").strip()
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, IndexError):
        return None
    ref = data.get("duplicate_of") if isinstance(data, dict) else None
    return ref if ref in valid_refs else None


def find_duplicate(issue: dict, cfg: AutoLoopConfig) -> tuple[str | None, ClaudeResult | None]:
    """Return (canonical_ref, claude_result) if *issue* duplicates existing work.

    Cheap title-token prefilter narrows the candidate set, then Claude confirms
    true equivalence. Existing duplicate/rejected issues are never candidates.
    """
    src = get_source(cfg)
    tokens = _title_tokens(issue.get("title", ""))
    if not tokens:
        return None, None

    candidates = []
    for other in src.list_issues(state="all", limit=100):
        if str(other.get("number")) == str(issue.get("number")):
            continue
        labels = {lbl["name"] for lbl in other.get("labels", [])}
        if labels & {"duplicate", "rejected"}:
            continue
        other_tokens = _title_tokens(other.get("title", ""))
        if not other_tokens:
            continue
        overlap = len(tokens & other_tokens) / len(tokens | other_tokens)
        if overlap >= 0.5:
            candidates.append(other)
    if not candidates:
        return None, None
    candidates = candidates[:12]

    ref_by_num = {c["number"]: src.ref(c["number"]) for c in candidates}
    prompt = DEDUP_PROMPT.format(
        title=issue.get("title", ""),
        body=(issue.get("body") or "")[:1000],
        candidates="\n".join(
            f"- {ref_by_num[c['number']]}: {c.get('title', '')}" for c in candidates
        ),
    )
    result = run_claude(prompt, cfg.triage_model, cfg.triage_timeout)
    if not result.success:
        return None, result
    return parse_duplicate_response(result.text, set(ref_by_num.values())), result


def mark_duplicate(number, canonical_ref: str, cfg: AutoLoopConfig):
    """Label an issue a duplicate (dropping ready) and comment with the canonical."""
    src = get_source(cfg)
    src.edit_issue(number, add_labels=["duplicate"], remove_labels=["ready"])
    src.comment(
        number,
        f"**Auto-triage — Duplicate:** this appears to duplicate {canonical_ref}. "
        "Not triaging or implementing it; close it if that's correct.",
    )


# --- Orchestration ---


def triage_issue(issue: dict, cfg: AutoLoopConfig, auto_fix: bool = True) -> list[ClaudeResult]:
    """Evaluate a single issue and apply the appropriate label."""
    results: list[ClaudeResult] = []

    # Dedup guard: don't triage/implement work that already exists.
    if auto_fix:  # only on the first pass, not the post-rewrite re-triage
        dup_ref, dup_result = find_duplicate(issue, cfg)
        if dup_result:
            results.append(dup_result)
        if dup_ref:
            print(f"  #{issue['number']}: duplicate of {dup_ref}, skipping.")
            mark_duplicate(issue["number"], dup_ref, cfg)
            return results

    verdict, eval_result = evaluate_issue(issue, cfg)
    results.append(eval_result)

    if verdict["verdict"] == "rejected":
        if auto_fix:
            new_body, rewrite_result = rewrite_issue_body(issue, verdict["reason"], cfg)
            results.append(rewrite_result)
            if new_body:
                apply_rewrite(issue["number"], new_body, cfg)
                results.extend(triage_issue({**issue, "body": new_body}, cfg, auto_fix=False))
                return results
        reject_issue(issue["number"], verdict["reason"], cfg)
        return results

    if verdict["verdict"] == "not-code-work":
        print(f"  #{issue['number']}: not code work, routing to needs-human")
        route_to_human(
            issue["number"],
            f"this issue's deliverable is not a code change to this repo —"
            f" {verdict['reason']}. Handle it by hand rather than via implement.",
            cfg,
        )
        return results

    if verdict.get("files_missing", False):
        files, disc_result = discover_files(issue, cfg)
        results.append(disc_result)
        if files:
            enrich_issue_with_files(issue["number"], files, cfg)

    if verdict["verdict"] == "ready":
        from autoloop.config import touches_protected_path
        from autoloop.create_issue import extract_files_from_spec

        body = issue.get("body") or ""
        mentioned_files = extract_files_from_spec(body)
        if touches_protected_path(mentioned_files, cfg.protected_paths):
            print(f"  #{issue['number']}: touches protected path, routing to needs-human")
            route_to_human(
                issue["number"],
                f"issue targets protected paths ({', '.join(mentioned_files)})."
                " Requires manual implementation.",
                cfg,
            )
            return results
        approve_issue(issue["number"], verdict["priority"], verdict["reason"], cfg)
    elif verdict["verdict"] == "needs-decomposition":
        if is_sub_issue(issue):
            # Already a decomposition product — recursing would spawn another
            # near-identical child (the loop we're fixing). Send to a human.
            route_to_human(
                issue["number"],
                "this is already a sub-issue but is still estimated too large;"
                " not decomposing further.",
                cfg,
            )
        else:
            decompose_issue(issue["number"], verdict, cfg, issue.get("body") or "")

    return results


def main():
    from autoloop.config import load_config

    cfg = load_config()
    start_time = time.time()
    results: list[ClaudeResult] = []

    issues = list_untriaged_issues(cfg)
    if not issues:
        print("No untriaged issues found.")
        return
    for issue in issues:
        print(f"Triaging #{issue['number']}: {issue['title']}", flush=True)
        try:
            results.extend(triage_issue(issue, cfg))
        except Exception:
            # One issue's failure (API blip, bad decomposition) must not abort
            # the whole batch — the next scheduled run re-triages it.
            logging.exception("triage failed for #%s", issue.get("number"))

    if results:
        elapsed = time.time() - start_time
        total_cost = sum(r.cost_usd for r in results)
        total_input = sum(r.input_tokens for r in results)
        total_output = sum(r.output_tokens for r in results)
        total_cache_read = sum(r.cache_read_tokens for r in results)
        print("\n--- AutoLoop Triage Stats ---")
        print(f"  Duration: {elapsed:.0f}s")
        print(f"  Claude calls: {len(results)}")
        print(f"  Input tokens: {total_input:,}")
        print(f"  Output tokens: {total_output:,}")
        print(f"  Cost: ${total_cost:.2f}")
        log_run(
            0,
            True,
            len(issues),
            elapsed,
            total_cost,
            total_input,
            total_output,
            total_cache_read,
        )


if __name__ == "__main__":
    main()
