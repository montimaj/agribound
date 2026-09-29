"""Generate ``examples/notebooks/NN_*.ipynb`` from ``examples/NN_*.py``.

Each notebook holds the same code as its script. Cells:

- markdown ``# <title>`` plus the rest of the module docstring in a text block
- markdown ``## Setup``; code: imports, chdir, warnings, logging
- markdown ``## Configuration``; code: module-level constants
- markdown ``## `f()` `` plus the first docstring line; code: one cell per
  top-level ``def``/``class``
- markdown ``## Run``; code: the ``if __name__ == "__main__":`` block

Every notebook is checked before it is written: its code cells, joined, parse
to the same AST as the script without its module docstring and equal it as
text up to blank lines; it has no outputs; and it passes
``nbformat.validate``.

Paths are resolved from the repository root (the parent of this file's
directory), so the tool can be run from any directory::

    python tools/sync_notebooks.py           # (re)write out-of-date notebooks
    python tools/sync_notebooks.py --check   # exit 1 if any notebook is out of date

``--check`` writes nothing. It compares every stored notebook with the one this
tool would write (cell types, cell sources and notebook metadata; the random
cell ids are ignored), runs the checks above on the stored notebook, and
reports scripts without a notebook and notebooks without a script. It exits
with status 1 when anything differs and 0 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

#: Repository root: the parent of the ``tools/`` directory.
REPO_ROOT = Path(__file__).resolve().parents[1]

CONST_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
SCRIPT_GLOB = "[0-9][0-9]_*.py"


def _is_constant(node: ast.stmt) -> bool:
    if isinstance(node, ast.Assign):
        return all(isinstance(t, ast.Name) and CONST_RE.match(t.id) for t in node.targets)
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and bool(CONST_RE.match(node.target.id))
    return False


def _kind(node: ast.stmt) -> str:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return "def"
    if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
        return "main"
    if _is_constant(node):
        return "const"
    return "setup"


def split_script(path: Path) -> tuple[str, list[tuple[str, str, str]]]:
    """Return ``(docstring, [(kind, heading, code), ...])`` for a script."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    body = list(tree.body)
    doc = ""
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        doc = body[0].value.value
        start_line = body[0].end_lineno  # 1-based, inclusive end of the docstring
        body = body[1:]
    else:
        start_line = 0
    # Each node owns the lines from the end of the previous node to its own end,
    # so comments before a node go with it.
    groups: list[tuple[str, str, int, int]] = []  # kind, name, first line index, end index
    prev_end = start_line
    for node in body:
        kind = _kind(node)
        name = getattr(node, "name", "")
        end = node.end_lineno
        if groups and kind in ("setup", "const") and groups[-1][0] == kind:
            k, n, first, _ = groups[-1]
            groups[-1] = (k, n, first, end)
        elif groups and kind == "setup" and groups[-1][0] == "const":
            # a non-constant statement among constants stays in the configuration cell
            k, n, first, _ = groups[-1]
            groups[-1] = (k, n, first, end)
        else:
            groups.append((kind, name, prev_end, end))
        prev_end = end
    if prev_end < len(lines) and groups:
        k, n, first, _ = groups[-1]
        groups[-1] = (k, n, first, len(lines))
    cells = []
    for kind, name, first, end in groups:
        code = "".join(lines[first:end]).strip("\n")
        if kind == "setup":
            heading = "## Setup"
        elif kind == "const":
            heading = "## Configuration"
        elif kind == "main":
            heading = "## Run"
        else:
            node = next(n for n in body if getattr(n, "name", None) == name)
            summary = (ast.get_docstring(node) or "").strip().splitlines()
            heading = f"## `{name}()`" + (f"\n\n{summary[0]}" if summary else "")
        cells.append((kind, heading, code))
    return doc, cells


def build_notebook(path: Path) -> nbformat.NotebookNode:
    """Build the notebook for the script at *path*."""
    doc, cells = split_script(path)
    doc_lines = doc.strip("\n").splitlines()
    title = doc_lines[0].strip() if doc_lines else path.stem
    rest = "\n".join(doc_lines[1:]).strip("\n")
    intro = (
        f"# {title}\n\nNotebook copy of `examples/{path.name}` (same code). Run it from "
        "`examples/notebooks/` or the repository root; the setup cell changes to the repository "
        "root. Command-line options keep their defaults here: edit the configuration cell, and "
        "set `GEE_PROJECT` (or `gcloud config set project ...`) for Earth Engine.\n"
    )
    if rest:
        intro += f"\n```text\n{rest}\n```"
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    nb.cells.append(new_markdown_cell(intro))
    for _, heading, code in cells:
        nb.cells.append(new_markdown_cell(heading))
        nb.cells.append(new_code_cell(code))
    return nb


def check(path: Path, nb: nbformat.NotebookNode) -> None:
    """Raise :class:`AssertionError` unless *nb* holds exactly the code of *path*."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    has_doc = (
        bool(tree.body)
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
    )
    if has_doc:
        tree.body = tree.body[1:]
    code = "\n\n".join(c.source for c in nb.cells if c.cell_type == "code")
    try:
        code_tree = ast.parse(code)
    except SyntaxError as exc:
        raise AssertionError(f"{path.name}: notebook code does not parse: {exc}") from None
    if ast.dump(code_tree) != ast.dump(tree):
        raise AssertionError(f"{path.name}: notebook code differs from the script (AST)")

    def norm(text: str) -> list[str]:
        return [ln.rstrip() for ln in text.splitlines() if ln.strip()]

    doc_end = ast.parse(source).body[0].end_lineno if has_doc else 0
    script_body = "".join(source.splitlines(keepends=True)[doc_end:])
    if norm(code) != norm(script_body):
        raise AssertionError(f"{path.name}: notebook code differs from the script (text)")
    for cell in nb.cells:
        if cell.cell_type == "code" and (cell.outputs or cell.execution_count is not None):
            raise AssertionError(f"{path.name}: outputs present")
    nbformat.validate(nb)


def compare(stored: nbformat.NotebookNode, built: nbformat.NotebookNode) -> str | None:
    """Describe the first difference between two notebooks (cell ids ignored)."""
    if len(stored.cells) != len(built.cells):
        return f"{len(stored.cells)} cells, expected {len(built.cells)}"
    for index, (a, b) in enumerate(zip(stored.cells, built.cells, strict=True)):
        if a.cell_type != b.cell_type:
            return f"cell {index} is {a.cell_type}, expected {b.cell_type}"
        if a.source != b.source:
            first_line = b.source.splitlines()[0] if b.source else ""
            return f"cell {index} ({b.cell_type}, expected to start {first_line!r}) differs"
    if dict(stored.metadata) != dict(built.metadata):
        return "notebook metadata differs"
    return None


def sync(repo: Path, only_check: bool) -> int:
    """Check or write every notebook under *repo*; return the exit status."""
    examples = repo / "examples"
    out_dir = examples / "notebooks"
    scripts = sorted(examples.glob(SCRIPT_GLOB))
    if not scripts:
        print(f"error: no {SCRIPT_GLOB} scripts in {examples}", file=sys.stderr)
        return 1
    problems: list[str] = []
    for script in scripts:
        target = out_dir / f"{script.stem}.ipynb"
        built = build_notebook(script)
        try:
            stored = nbformat.read(target, as_version=4) if target.exists() else None
        except Exception as exc:  # unreadable JSON or notebook format
            if not only_check:
                raise
            problems.append(f"{target.name}: cannot be read ({exc})")
            continue
        if only_check:
            if stored is None:
                problems.append(f"{target.name}: missing")
                continue
            difference = compare(stored, built)
            try:
                check(script, stored)
            except (AssertionError, nbformat.ValidationError) as exc:
                difference = difference or str(exc)
            if difference:
                problems.append(f"{target.name}: {difference}")
            else:
                print(f"ok {target.name}")
            continue
        check(script, built)
        if stored is not None and compare(stored, built) is None:
            print(f"unchanged {target.name}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(built, target)
        n_code = sum(c.cell_type == "code" for c in built.cells)
        print(f"wrote {target.name}: {n_code} code cells")
    if only_check:
        stems = {script.stem for script in scripts}
        for notebook in sorted(out_dir.glob("[0-9][0-9]_*.ipynb")):
            if notebook.stem not in stems:
                problems.append(f"{notebook.name}: no examples/{notebook.stem}.py")
        for problem in problems:
            print(f"OUT OF DATE {problem}", file=sys.stderr)
        if problems:
            print(
                f"{len(problems)} notebook(s) differ from their scripts; run "
                "'python tools/sync_notebooks.py' and commit the result.",
                file=sys.stderr,
            )
            return 1
        print(f"all {len(scripts)} notebooks match their scripts")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if any notebook differs from its script",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=REPO_ROOT,
        help="repository root (default: the parent of this file's directory)",
    )
    args = parser.parse_args(argv)
    return sync(args.repo.resolve(), args.check)


if __name__ == "__main__":
    raise SystemExit(main())
