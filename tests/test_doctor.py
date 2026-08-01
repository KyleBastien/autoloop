"""Tests for autoloop doctor check runner."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from autoloop.doctor import (
    Check,
    Result,
    check_autoloop_toml,
    check_claude_cli_authenticated,
    check_claude_cli_installed,
    check_claude_settings,
    check_gh_cli_installed,
    get_checks,
    run_checks,
)


def test_run_checks_all_pass(capsys):
    checks = [
        Check(name="alpha", fn=lambda: (True, "ok"), fix_hint="fix alpha"),
        Check(name="beta", fn=lambda: (True, "good"), fix_hint="fix beta"),
    ]
    results = run_checks(checks)

    assert len(results) == 2
    assert all(r.passed for r in results)
    assert results[0].name == "alpha"
    assert results[1].name == "beta"

    out = capsys.readouterr().out
    assert "✓" in out
    assert "✗" not in out


def test_run_checks_one_fails(capsys):
    checks = [
        Check(name="good", fn=lambda: (True, "fine"), fix_hint="n/a"),
        Check(name="bad", fn=lambda: (False, "broken"), fix_hint="run repair"),
    ]
    results = run_checks(checks)

    assert results[0].passed is True
    assert results[1].passed is False
    assert results[1].fix_hint == "run repair"

    out = capsys.readouterr().out
    assert "✓" in out
    assert "✗" in out
    assert "hint: run repair" in out


def test_run_checks_exception_in_fn(capsys):
    def boom():
        raise RuntimeError("unexpected error")

    checks = [
        Check(name="exploder", fn=boom, fix_hint="check logs"),
    ]
    results = run_checks(checks)

    assert len(results) == 1
    assert results[0].passed is False
    assert "unexpected error" in results[0].message

    out = capsys.readouterr().out
    assert "✗" in out
    assert "hint: check logs" in out


def test_run_checks_empty(capsys):
    results = run_checks([])

    assert results == []
    out = capsys.readouterr().out
    assert "No checks registered." in out


def test_result_dataclass():
    r = Result(name="test", passed=True, message="ok", fix_hint="none")
    assert r.name == "test"
    assert r.passed is True
    assert r.message == "ok"
    assert r.fix_hint == "none"


# --- autoloop.toml check ---


def test_check_autoloop_toml_exists_and_valid(tmp_path):
    toml_file = tmp_path / "autoloop.toml"
    toml_file.write_text('repo = "acme/widgets"\n')
    passed, msg = check_autoloop_toml(tmp_path)
    assert passed is True
    assert "found and valid" in msg


def test_check_autoloop_toml_missing(tmp_path):
    passed, msg = check_autoloop_toml(tmp_path)
    assert passed is False
    assert "not found" in msg


def test_check_autoloop_toml_invalid_syntax(tmp_path):
    toml_file = tmp_path / "autoloop.toml"
    toml_file.write_text("[invalid toml\nno closing bracket")
    passed, msg = check_autoloop_toml(tmp_path)
    assert passed is False
    assert "invalid" in msg


# --- .claude/settings.json check ---


def test_check_claude_settings_exists(tmp_path):
    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir()
    (settings_dir / "settings.json").write_text("{}")
    passed, msg = check_claude_settings(tmp_path)
    assert passed is True
    assert "found" in msg


def test_check_claude_settings_missing(tmp_path):
    passed, msg = check_claude_settings(tmp_path)
    assert passed is False
    assert "not found" in msg


# --- get_checks integration ---


def test_get_checks_returns_registered_checks():
    checks = get_checks()
    assert len(checks) == 5
    assert checks[0].name == "autoloop.toml"
    assert checks[1].name == ".claude/settings.json"
    assert checks[2].name == "claude CLI installed"
    assert checks[3].name == "claude CLI authenticated"
    assert checks[4].name == "gh CLI installed and authenticated"


def test_get_checks_all_pass(tmp_path, capsys):
    toml_file = tmp_path / "autoloop.toml"
    toml_file.write_text('repo = "acme/widgets"\n')
    settings_dir = tmp_path / ".claude"
    settings_dir.mkdir()
    (settings_dir / "settings.json").write_text("{}")

    checks = get_checks(tmp_path)
    results = run_checks(checks)

    assert len(results) == 5
    file_results = [r for r in results if r.name in ("autoloop.toml", ".claude/settings.json")]
    assert all(r.passed for r in file_results)
    out = capsys.readouterr().out
    assert "✓" in out


def test_get_checks_all_fail(tmp_path, capsys):
    checks = get_checks(tmp_path)
    results = run_checks(checks)

    assert len(results) == 5
    out = capsys.readouterr().out
    assert "✗" in out


# --- claude CLI installed check ---


def test_check_claude_cli_installed_success():
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="v2.1.160\n", stderr="")
    with patch("autoloop.doctor.subprocess.run", return_value=completed):
        passed, msg = check_claude_cli_installed()
    assert passed is True
    assert "v2.1.160" in msg


def test_check_claude_cli_installed_not_found():
    with patch("autoloop.doctor.subprocess.run", side_effect=FileNotFoundError):
        passed, msg = check_claude_cli_installed()
    assert passed is False
    assert "not found" in msg


def test_check_claude_cli_installed_nonzero_exit():
    completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")
    with patch("autoloop.doctor.subprocess.run", return_value=completed):
        passed, msg = check_claude_cli_installed()
    assert passed is False
    assert "failed to run" in msg


def test_check_claude_cli_installed_version_on_stderr():
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="v2.1.160")
    with patch("autoloop.doctor.subprocess.run", return_value=completed):
        passed, msg = check_claude_cli_installed()
    assert passed is True
    assert "v2.1.160" in msg


# --- claude CLI authenticated check ---


def test_check_claude_cli_authenticated_success():
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="Logged in\n", stderr="")
    with patch("autoloop.doctor.subprocess.run", return_value=completed):
        passed, msg = check_claude_cli_authenticated()
    assert passed is True
    assert "authenticated" in msg


def test_check_claude_cli_authenticated_not_found():
    with patch("autoloop.doctor.subprocess.run", side_effect=FileNotFoundError):
        passed, msg = check_claude_cli_authenticated()
    assert passed is False
    assert "not found" in msg


def test_check_claude_cli_authenticated_not_authed():
    completed = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="not logged in"
    )
    with patch("autoloop.doctor.subprocess.run", return_value=completed):
        passed, msg = check_claude_cli_authenticated()
    assert passed is False
    assert "not authenticated" in msg


# --- gh CLI installed and authenticated check ---


def test_check_gh_cli_installed_success():
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Logged in to github.com\n", stderr=""
    )
    with patch("autoloop.doctor.subprocess.run", return_value=completed):
        passed, msg = check_gh_cli_installed()
    assert passed is True
    assert "installed and authenticated" in msg


def test_check_gh_cli_not_found():
    with patch("autoloop.doctor.subprocess.run", side_effect=FileNotFoundError):
        passed, msg = check_gh_cli_installed()
    assert passed is False
    assert "not found" in msg


def test_check_gh_cli_not_authenticated():
    completed = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="not logged in"
    )
    with patch("autoloop.doctor.subprocess.run", return_value=completed):
        passed, msg = check_gh_cli_installed()
    assert passed is False
    assert "not authenticated" in msg
