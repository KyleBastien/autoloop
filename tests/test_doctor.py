"""Tests for autoloop doctor check runner."""

from __future__ import annotations

from autoloop.doctor import Check, Result, run_checks


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
