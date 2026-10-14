"""The examples write their metrics files as JSON (RFC 8259: no NaN or Infinity).

``agribound.evaluate`` reports an undefined score as NaN and the open last size class
as ``upper_ha = inf``. The 1.1 re-run's metrics files of examples 23 and 24 held
``NaN`` and ``Infinity`` (37 of 189 files), which strict JSON parsers reject. The
examples convert with :func:`agribound.provenance.to_jsonable` (non-finite -> null),
as agribound's own JSON writers do, and pass ``allow_nan=False``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))


def _metrics_writers(script: Path) -> list[ast.Call]:
    """The ``json.dumps(...)`` calls whose text goes to a ``metrics_*`` file via ``write_text``."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    # Names bound to a metrics file path or name (metrics_path = ...; name = f"metrics_...").
    named = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and "metrics_" in ast.unparse(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    calls = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "write_text"
            and node.args
        ):
            continue
        text, receiver = node.args[0], node.func.value
        is_dumps = (
            isinstance(text, ast.Call)
            and isinstance(text.func, ast.Attribute)
            and text.func.attr == "dumps"
            and isinstance(text.func.value, ast.Name)
            and text.func.value.id == "json"
        )
        to_metrics = "metrics_" in ast.unparse(receiver) or any(
            isinstance(n, ast.Name) and n.id in named for n in ast.walk(receiver)
        )
        if is_dumps and to_metrics:
            calls.append(text)
    return calls


def test_the_metrics_writers_are_found():
    found = {s.name[:2]: len(_metrics_writers(s)) for s in SCRIPTS if _metrics_writers(s)}
    # Example 24 writes the main and the irrigated-only metrics of each run, and the paired
    # change of each "-opt" run (--options).
    assert found == {"20": 3, "21": 1, "23": 1, "24": 3, "25": 1, "26": 1, "27": 1, "28": 1}


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.name[:2])
def test_metrics_files_are_strict_json(script):
    for call in _metrics_writers(script):
        where = f"{script.name}:{call.lineno}"
        keywords = {k.arg: k.value for k in call.keywords}
        assert "allow_nan" in keywords, f"{where}: pass allow_nan=False"
        assert ast.literal_eval(keywords["allow_nan"]) is False, where
        first = call.args[0]
        assert (
            isinstance(first, ast.Call)
            and isinstance(first.func, ast.Name)
            and first.func.id == "to_jsonable"
        ), f"{where}: wrap the metrics in to_jsonable (NaN and inf -> null)"


def test_to_jsonable_makes_evaluate_style_metrics_strict():
    from agribound.provenance import to_jsonable

    metrics = {"f1": float("nan"), "per_size_class": {"50-inf": {"upper_ha": float("inf")}}}
    text = json.dumps(to_jsonable(metrics), allow_nan=False)
    assert json.loads(text) == {"f1": None, "per_size_class": {"50-inf": {"upper_ha": None}}}
