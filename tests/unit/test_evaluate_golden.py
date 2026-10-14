"""Golden outputs of agribound.evaluate without ``ignore``.

``tests/data/evaluate_golden.json`` holds the full results of :func:`evaluate`,
:func:`evaluate_frame` and :func:`chance_baselines` on fixed layouts, with most
options set, as computed by the code before ``ignore`` was added (2026-10-11).
A call without ``ignore`` must keep returning them: the same keys in the same
order, the same counts and labels, and the same numbers up to floating-point
noise (relative 1e-6). Non-finite values are stored as null.
``proj_network`` depends on the environment and is not compared.

To write the file again after an intended change of the outputs::

    python tests/unit/test_evaluate_golden.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry import box

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "evaluate_golden.json"
CRS = "EPSG:32613"
X0, Y0 = 500_000.0, 3_800_000.0
SKIPPED_KEYS = {"proj_network"}


def sq(x: float, y: float, w: float = 100.0, h: float | None = None):
    """Axis-aligned rectangle with lower-left corner (X0 + x, Y0 + y)."""
    return box(X0 + x, Y0 + y, X0 + x + w, Y0 + y + (w if h is None else h))


def gdf(geoms, crs=CRS, **columns) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(columns, geometry=list(geoms), crs=crs)


def _stratified():
    ref = gdf([sq(0, 0), sq(200, 0), sq(1000, 0), sq(1200, 0)], basin=["a", "a", "b", "b"])
    pred = gdf([sq(0, 0), sq(200, 0), sq(1000, 0), sq(1200, 0, 30), sq(5000, 5000)])
    return pred, ref


def _merges_and_splits():
    ref = gdf(
        [sq(0, 0), sq(100, 0, 60, 100), sq(400, 0, 200), sq(700, 0, 150, 100), sq(900, 0, 40)],
        system=["pivot", "pivot", "flood", "flood", None],
    )
    pred = gdf(
        [
            sq(0, 0, 160, 100),  # merges two fields
            sq(400, 0, 120, 200),  # splits one field ...
            sq(520, 0, 80, 200),  # ... in two
            sq(705, 5, 150, 100),
            sq(2000, 0),  # overlaps no field
            None,  # dropped
        ]
    )
    return pred, ref


def _baseline_layers():
    ref = gdf([sq(0, 0, 130, 90), sq(140, 0, 80, 200), sq(0, 100, 120, 120)])
    pred = gdf([sq(5, 5, 120, 80), sq(150, 10, 60, 180), sq(10, 110, 100, 100)])
    return pred, ref


def _frame_records(frame) -> dict[str, Any]:
    return {
        "index": [str(label) for label in frame.index],
        "columns": list(frame.columns),
        "records": frame.astype(object).where(frame.notna(), None).to_dict("records"),
        "attrs": dict(frame.attrs),
    }


def golden_outputs(ev: ModuleType) -> dict[str, Any]:
    """The golden calls, made with the evaluate module *ev*."""
    out: dict[str, Any] = {}
    pred, ref = _stratified()
    out["stratified"] = ev.evaluate(
        pred, ref, strata="basin", size_bins="auto", bootstrap=20, boundary_tolerance_m=5.0
    )
    out["stratified_frame"] = _frame_records(
        ev.evaluate_frame(
            pred, ref, strata="basin", size_bins=[0, 0.5, 2], boundary_tolerance_m=5.0
        )
    )
    out["geographic"] = ev.evaluate(pred.to_crs(4326), ref.to_crs(4326))
    pred, ref = _merges_and_splits()
    out["many_to_one"] = ev.evaluate(
        pred,
        ref,
        iou_threshold=0.3,
        matching="many_to_one",
        strata="system",
        size_bins=[0, 1, 5, float("inf")],
        bootstrap=10,
        bootstrap_seed=7,
        boundary_tolerance_m=10.0,
        boundary_sample_spacing_m=None,
    )
    out["one_to_one"] = ev.evaluate(pred, ref, boundary_tolerance_m=10.0, equal_area_crs=5070)
    out["one_to_one_frame"] = _frame_records(
        ev.evaluate_frame(pred, ref, boundary_tolerance_m=10.0)
    )
    pred, ref = _baseline_layers()
    out["baselines"] = ev.evaluate(
        pred, ref, boundary_tolerance_m=5.0, baselines={"offset_steps": 2, "shift_m": 20.0}
    )
    out["chance_baselines"] = ev.chance_baselines(
        pred,
        ref,
        boundary_tolerance_m=5.0,
        footprint=True,
        offset_steps=2,
        min_area_m2=0.0,
        shift_m="auto",
        study_area=gdf([sq(-20, -20, 260, 260)]),
    )
    out["empty_predictions"] = ev.evaluate(gdf([]), ref, boundary_tolerance_m=5.0)
    return out


def _jsonable(value: Any) -> Any:
    """Builtins only; non-finite floats become None (as agribound's JSON writers do)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, str):  # numpy scalars
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _compare(expected: Any, actual: Any, where: str, problems: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            problems.append(f"{where}: expected a dict, got {type(actual).__name__}")
            return
        keys = [k for k in expected if k not in SKIPPED_KEYS]
        got = [k for k in actual if k not in SKIPPED_KEYS]
        if keys != got:
            problems.append(f"{where}: keys {got} != {keys}")
            return
        for key in keys:
            _compare(expected[key], actual[key], f"{where}.{key}", problems)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            problems.append(f"{where}: {actual!r} != {expected!r}")
            return
        for k, (e, a) in enumerate(zip(expected, actual, strict=True)):
            _compare(e, a, f"{where}[{k}]", problems)
    elif expected is None:
        if actual is not None:
            problems.append(f"{where}: {actual!r} != None (non-finite)")
    elif isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            problems.append(f"{where}: {actual!r} != {expected!r}")
    elif isinstance(expected, int) and isinstance(actual, int):
        if expected != actual:
            problems.append(f"{where}: {actual!r} != {expected!r}")
    elif isinstance(expected, float) and isinstance(actual, float):
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-9):
            problems.append(f"{where}: {actual!r} != {expected!r}")
    elif type(expected) is not type(actual) or expected != actual:
        problems.append(f"{where}: {actual!r} ({type(actual).__name__}) != {expected!r}")


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def current() -> dict[str, Any]:
    import importlib

    return _jsonable(golden_outputs(importlib.import_module("agribound.evaluate")))


@pytest.mark.parametrize(
    "case",
    [
        "stratified",
        "stratified_frame",
        "geographic",
        "many_to_one",
        "one_to_one",
        "one_to_one_frame",
        "baselines",
        "chance_baselines",
        "empty_predictions",
    ],
)
def test_outputs_without_ignore_are_unchanged(golden, current, case):
    problems: list[str] = []
    _compare(golden[case], current[case], case, problems)
    assert not problems, "\n".join(problems[:20])


def test_the_golden_cases_are_all_tested(golden, current):
    assert set(golden) == set(current)


if __name__ == "__main__":
    import importlib

    outputs = _jsonable(golden_outputs(importlib.import_module("agribound.evaluate")))
    GOLDEN.write_text(json.dumps(outputs, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN}")
