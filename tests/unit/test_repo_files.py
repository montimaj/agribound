"""Repository hygiene checks: example script modes and line endings, documentation links.

These tests read the source checkout (``examples/``, ``docs/``, ``README.md``,
``.lycheeignore``) and are skipped when it is not available, e.g. when the
suite runs from an installed wheel.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"

#: Links to files of this repository on the main branch: GitHub pages
#: (``blob``/``tree``) and raw file URLs, such as the images in README.md.
_REPO_LINK = re.compile(
    r"https://(?:github\.com/montimaj/agribound/(?:blob|tree)/main"
    r"|raw\.githubusercontent\.com/montimaj/agribound/main)/([^\s)\"'<>`#\]]+)"
)


def _example_scripts() -> list[Path]:
    scripts = [
        *sorted((EXAMPLES / "hpc").glob("*.sh")),
        *sorted((EXAMPLES / "regions").glob("*.sh")),
    ]
    runner = EXAMPLES / "run_region_delineation.sh"
    if runner.exists():
        scripts.append(runner)
    return scripts


def _git_index_modes(paths: list[Path]) -> dict[str, str] | None:
    """Git index mode (e.g. ``"100755"``) of each tracked path, or None without git."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        return None
    rel = [p.relative_to(ROOT).as_posix() for p in paths]
    result = subprocess.run(
        ["git", "ls-files", "-s", "--", *rel],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    modes = {}
    for line in result.stdout.splitlines():
        meta, _, path = line.partition("\t")
        modes[path] = meta.split()[0]
    return modes


def test_example_shell_scripts_are_executable():
    scripts = _example_scripts()
    if not scripts:
        pytest.skip("examples/ shell scripts are not available")
    if sys.platform == "win32":
        # Windows stat() never reports execute bits for .sh files; the mode that
        # reaches users (and the sdist) is the one recorded in the git index.
        modes = _git_index_modes(scripts)
        if modes is None:
            pytest.skip("git is not available to read the index modes")
        tracked = {
            p.relative_to(ROOT).as_posix(): modes[p.relative_to(ROOT).as_posix()]
            for p in scripts
            if p.relative_to(ROOT).as_posix() in modes
        }
        if not tracked:
            pytest.skip("the example scripts are not tracked by git yet")
        not_executable = sorted(path for path, mode in tracked.items() if mode != "100755")
        assert not not_executable, f"git update-index --chmod=+x {' '.join(not_executable)}"
        return
    not_executable = [
        str(p.relative_to(ROOT)) for p in scripts if not p.stat().st_mode & stat.S_IXUSR
    ]
    assert not not_executable, f"chmod +x {' '.join(not_executable)}"


def _bash_files() -> list[Path]:
    """Scripts run or sourced by bash: example scripts, sbatch scripts, HPC profiles."""
    return [
        *_example_scripts(),
        *sorted((EXAMPLES / "hpc").glob("*.sbatch")),
        *sorted((EXAMPLES / "hpc" / "profiles").glob("*.env")),
    ]


def test_bash_files_have_lf_line_endings():
    """CRLF endings break bash ($'\\r': command not found)."""
    files = _bash_files()
    if not files:
        pytest.skip("examples/ shell scripts are not available")
    crlf = [str(p.relative_to(ROOT)) for p in files if b"\r\n" in p.read_bytes()]
    assert not crlf, f"CRLF line endings: {crlf}"


def test_gitattributes_keep_lf_for_bash_files():
    """.gitattributes makes Windows checkouts (core.autocrlf=true) keep LF in the scripts."""
    files = _bash_files()
    if not files or not (ROOT / ".gitattributes").exists():
        pytest.skip(".gitattributes or the example scripts are not available")
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    rel = [p.relative_to(ROOT).as_posix() for p in files]
    result = subprocess.run(
        ["git", "check-attr", "eol", "--", *rel],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    eol = dict(line.rsplit(": eol: ", 1) for line in result.stdout.splitlines())
    assert set(eol) == set(rel)
    not_lf = sorted(path for path, value in eol.items() if value != "lf")
    assert not not_lf, f"add an 'eol=lf' rule to .gitattributes for: {not_lf}"


def _linked_repo_paths() -> dict[str, str]:
    """Map each repository link in the docs and README to the linked path."""
    sources = [ROOT / "README.md", ROOT / "mkdocs.yml", *sorted((ROOT / "docs").rglob("*.md"))]
    links: dict[str, str] = {}
    for source in sources:
        if source.exists():
            for match in _REPO_LINK.finditer(source.read_text(encoding="utf-8")):
                links[match.group(0)] = match.group(1).rstrip("/")
    return links


def _lychee_patterns() -> list[re.Pattern[str]]:
    lines = (ROOT / ".lycheeignore").read_text(encoding="utf-8").splitlines()
    return [re.compile(line.strip()) for line in lines if line.strip() and not line.startswith("#")]


def test_repo_links_point_at_existing_files():
    links = _linked_repo_paths()
    if not links:
        pytest.skip("docs/ and README.md are not available")
    missing = sorted(url for url, path in links.items() if not (ROOT / path).exists())
    assert not missing, "links to files that do not exist:\n" + "\n".join(missing)


def _on_origin_main(path: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"origin/main:{path}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def test_links_to_unpushed_files_are_ignored_by_lychee():
    """The docs link check fails on files that are not on main yet."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "origin/main"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if probe.returncode != 0:
        pytest.skip("origin/main is not available in this checkout")
    links = _linked_repo_paths()
    if not links:
        pytest.skip("docs/ and README.md are not available")
    patterns = _lychee_patterns()
    unignored = sorted(
        url
        for url, path in links.items()
        if not _on_origin_main(path) and not any(p.search(url) for p in patterns)
    )
    assert not unignored, (
        "linked files are not on origin/main; add them to .lycheeignore:\n" + "\n".join(unignored)
    )


@pytest.mark.parametrize("name", ["NM_example.png", "Pampas_example.png"])
def test_legacy_readme_images_stay_at_their_published_paths(name):
    """The READMEs of the published 0.1.x releases (on PyPI) link assets/<name>."""
    legacy, archived = ROOT / "assets" / name, ROOT / "assets" / "gallery_0.1x" / name
    if not archived.exists():
        pytest.skip("assets/ is not part of this checkout")
    assert legacy.is_file(), f"assets/{name} is linked by the 0.1.x READMEs on PyPI; keep it"
    assert legacy.read_bytes() == archived.read_bytes()
