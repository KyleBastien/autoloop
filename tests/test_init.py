from __future__ import annotations

from autoloop.init import run_init


def _run(tmp_path, monkeypatch, **kw):
    monkeypatch.chdir(tmp_path)
    run_init("acme/widgets", skip_labels=True, **kw)
    toml = (tmp_path / "autoloop.toml").read_text()
    workflow = (tmp_path / ".github" / "workflows" / "autoloop-cleanup.yml").read_text()
    return toml, workflow


def test_run_init_linear_gitignores_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_init("acme/widgets", skip_labels=True, source="linear", linear_team="ENG")
    gitignore = (tmp_path / ".gitignore").read_text()
    assert ".env" in gitignore


def test_run_init_github_does_not_gitignore_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_init("acme/widgets", skip_labels=True)
    gitignore = (tmp_path / ".gitignore").read_text()
    assert ".env" not in gitignore


def test_run_init_github_default(tmp_path, monkeypatch):
    toml, workflow = _run(tmp_path, monkeypatch)
    assert 'source = "github"' in toml
    assert 'linear_team = ""' in toml
    # GitHub variant keeps the label-cleanup github-script job.
    assert "github-script" in workflow
    assert "removeLabel" in workflow


def test_run_init_linear_variant(tmp_path, monkeypatch):
    toml, workflow = _run(tmp_path, monkeypatch, source="linear", linear_team="ENG")
    assert 'source = "linear"' in toml
    assert 'linear_team = "ENG"' in toml
    # Linear variant runs the Python auto-close step with the API key, no github-script.
    assert "github-script" not in workflow
    assert "LINEAR_API_KEY" in workflow
    assert "auto-close-parent" in workflow


def test_run_init_linear_workflow_renders_actions_expressions(tmp_path, monkeypatch):
    _toml, workflow = _run(tmp_path, monkeypatch, source="linear", linear_team="ENG")
    # .format() must collapse the doubled braces to valid Actions syntax.
    assert "${{ secrets.LINEAR_API_KEY }}" in workflow
