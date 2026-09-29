"""Tests for tools/sync_notebooks.py (example notebooks generated from the scripts)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "sync_notebooks.py"

if not TOOL.exists():  # e.g. running from an installed wheel
    pytest.skip("tools/sync_notebooks.py is not available", allow_module_level=True)

SCRIPT = '''"""Example 01: a tiny example.

Longer description.
"""

import sys

VALUE = 3


def helper():
    """Return the value."""
    return VALUE


if __name__ == "__main__":
    sys.exit(helper() - VALUE)
'''


def _load_tool():
    spec = importlib.util.spec_from_file_location("sync_notebooks", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tool():
    return _load_tool()


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "examples" / "notebooks").mkdir(parents=True)
    (tmp_path / "examples" / "01_tiny.py").write_text(SCRIPT)
    return tmp_path


def test_write_then_check_passes(tool, repo, capsys):
    assert tool.main(["--repo", str(repo)]) == 0
    notebook = repo / "examples" / "notebooks" / "01_tiny.ipynb"
    nb = nbformat.read(notebook, as_version=4)
    headings = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown"]
    assert headings == ["# Example 01: a tiny example.", "## Setup", "## Configuration",
                        "## `helper()`", "## Run"]  # fmt: skip
    assert tool.main(["--repo", str(repo), "--check"]) == 0
    assert "all 1 notebooks match their scripts" in capsys.readouterr().out
    # Rewriting an up-to-date notebook leaves the file (and its cell ids) alone.
    before = notebook.read_bytes()
    assert tool.main(["--repo", str(repo)]) == 0
    assert notebook.read_bytes() == before


def test_check_fails_when_the_script_changes(tool, repo, capsys):
    tool.main(["--repo", str(repo)])
    script = repo / "examples" / "01_tiny.py"
    script.write_text(SCRIPT.replace("VALUE = 3", "VALUE = 4"))
    notebook = repo / "examples" / "notebooks" / "01_tiny.ipynb"
    before = notebook.read_bytes()
    assert tool.main(["--repo", str(repo), "--check"]) == 1
    err = capsys.readouterr().err
    assert "OUT OF DATE 01_tiny.ipynb" in err
    assert notebook.read_bytes() == before  # --check writes nothing


def test_check_fails_on_docstring_only_change(tool, repo):
    tool.main(["--repo", str(repo)])
    script = repo / "examples" / "01_tiny.py"
    script.write_text(SCRIPT.replace("Longer description.", "Another description."))
    assert tool.main(["--repo", str(repo), "--check"]) == 1


def test_check_fails_on_outputs(tool, repo):
    tool.main(["--repo", str(repo)])
    notebook = repo / "examples" / "notebooks" / "01_tiny.ipynb"
    data = json.loads(notebook.read_text())
    code = next(c for c in data["cells"] if c["cell_type"] == "code")
    code["execution_count"] = 1
    code["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["x\n"]}]
    notebook.write_text(json.dumps(data))
    assert tool.main(["--repo", str(repo), "--check"]) == 1


def test_check_fails_on_missing_orphan_or_unreadable_notebooks(tool, repo, capsys):
    assert tool.main(["--repo", str(repo), "--check"]) == 1  # 01_tiny.ipynb missing
    assert "01_tiny.ipynb: missing" in capsys.readouterr().err
    tool.main(["--repo", str(repo)])
    orphan = repo / "examples" / "notebooks" / "02_gone.ipynb"
    orphan.write_text("{}")
    assert tool.main(["--repo", str(repo), "--check"]) == 1
    assert "02_gone.ipynb: no examples/02_gone.py" in capsys.readouterr().err
    orphan.unlink()
    (repo / "examples" / "notebooks" / "01_tiny.ipynb").write_text("not json")
    assert tool.main(["--repo", str(repo), "--check"]) == 1
    assert "cannot be read" in capsys.readouterr().err


def test_repository_notebooks_match_their_scripts():
    """The committed notebooks are in sync (the CI lint job runs the same check)."""
    if not any((ROOT / "examples").glob("[0-9][0-9]_*.py")):
        pytest.skip("examples/ is not available")
    result = subprocess.run(
        [sys.executable, str(TOOL), "--check"],
        cwd=ROOT / "tests",  # paths are resolved from the repository root, not the cwd
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
