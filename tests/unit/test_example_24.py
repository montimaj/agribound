"""Example 24's irrigated-only score (examples/24_nevada_field_boundaries.py); no data needed."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "24_nevada_field_boundaries.py"
X0, Y0 = 584_000.0, 4_389_000.0  # corner of the Diamond Valley evaluation square (UTM 11N)
NAN = float("nan")

# The columns evaluate_run gave each run of summary.csv before the irrigated-only score.
MAIN_ROW = [
    "fields", "precision", "recall", "recall_ci_low", "recall_ci_high", "f1", "f1_ci_low",
    "f1_ci_high", "iou_mean", "best_iou_mean", "boundary_f1", "fp_noref", "merged", "split",
    "recall_pivot", "recall_sprinkler", "recall_flood_ctl", "recall_flood_unc_none",
    "recall_retired", "recall_micro", "recall_0-2_ha", "recall_2-10_ha", "recall_10-50_ha",
    "recall_50-inf_ha", "recall_irrigated", "n_irrigated", "recall_not_irrigated",
    "n_not_irrigated",
]  # fmt: skip
IRRIGATED_ROW = [
    "n_reference_irrigated", "n_predicted_scored", "n_predicted_ignored", "precision_irrigated",
    "recall_irrigated_score", "recall_irrigated_score_ci_low", "recall_irrigated_score_ci_high",
    "f1_irrigated", "f1_irrigated_ci_low", "f1_irrigated_ci_high", "iou_mean_irrigated",
    "boundary_f1_irrigated", "fp_noref_irrigated",
]  # fmt: skip


@pytest.fixture(scope="module")
def ex24():
    spec = importlib.util.spec_from_file_location("example_24_nevada", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sq(x: float, y: float, w: float = 200.0, h: float | None = None):
    """Axis-aligned rectangle with lower-left corner (X0 + x, Y0 + y)."""
    return box(X0 + x, Y0 + y, X0 + x + w, Y0 + y + (w if h is None else h))


@pytest.fixture
def layers(ex24):
    # Two pivots irrigated in 2022, a dry sprinkler field and a field without an
    # IrrMapper share in 2022; the predictions: the two pivots, one in each other field
    # and one on unmapped ground.
    reference = gpd.GeoDataFrame(
        {
            "ITYPE_2024": [1, 1, 2, 3],
            "irrmapper_2022": [0.9, 0.8, 0.1, NAN],
            "irrmapper_2023": [0.9, 0.8, 0.6, 0.0],
            "irrmapper_2024": [0.9, 0.2, 0.7, 0.0],
        },
        geometry=[sq(0, 0), sq(300, 0), sq(600, 0), sq(900, 0)],
        crs=ex24.UTM_CRS,
    )
    reference = ex24.add_strata(reference)
    pred = gpd.GeoDataFrame(
        geometry=[sq(0, 0), sq(310, 10, 190), sq(610, 10, 100), sq(910, 10, 100), sq(1500, 0, 100)],
        crs=ex24.UTM_CRS,
    )
    return pred, reference


def test_the_irrigated_fields_of_the_imagery_year(ex24, layers):
    _, reference = layers
    status = reference["irrigated_2022"].tolist()
    assert status[:3] == ["irrigated", "irrigated", "not irrigated"] and pd.isna(status[3])
    for year, scored in ((2022, [0, 1]), (2023, [0, 1, 2]), (2024, [0, 2])):
        irrigated, other = ex24.irrigated_only(reference, year)
        assert irrigated.index.tolist() == scored
        # Every other field is ignored, a field without an IrrMapper share included.
        assert sorted(other.index.tolist() + scored) == [0, 1, 2, 3]


def test_the_irrigated_only_score(ex24, layers):
    pred, reference = layers
    row, record = ex24.evaluate_irrigated(pred, reference, 2022, argparse.Namespace(bootstrap=0))
    assert list(row) == IRRIGATED_ROW
    assert (row["n_reference_irrigated"], row["n_predicted_scored"]) == (2, 3)
    assert row["n_predicted_ignored"] == 2  # the predictions in the two other fields
    assert row["fp_noref_irrigated"] == 1  # the prediction on unmapped ground
    assert row["precision_irrigated"] == pytest.approx(2 / 3)
    assert row["recall_irrigated_score"] == 1.0 and row["f1_irrigated"] == pytest.approx(0.8)
    assert row["recall_irrigated_score_ci_low"] is None and row["f1_irrigated_ci_high"] is None
    metrics = record["metrics"]
    assert metrics["count_predicted_ignored"] == 2 and metrics["count_reference"] == 2
    assert list(metrics["per_stratum"]) == ["pivot"]
    assert record["score"].startswith(
        "irrigated only: reference = the 2 reference fields irrigated in 2022 "
        "(irrmapper_2022 >= 0.5); ignore = the other 2 reference fields"
    )
    # The main score is unchanged, and keeps the predictions in the other fields.
    main, _ = ex24.evaluate_run(pred, reference, 2022, argparse.Namespace(bootstrap=0))
    assert list(main) == MAIN_ROW and not set(main) & set(row)
    assert main["precision"] == pytest.approx(2 / 5) and main["recall"] == pytest.approx(2 / 4)
    assert main["recall_irrigated"] == row["recall_irrigated_score"]


def test_the_intervals_and_the_record_are_strict_json(ex24, layers):
    from agribound.provenance import to_jsonable

    pred, reference = layers
    row, record = ex24.evaluate_irrigated(pred, reference, 2024, argparse.Namespace(bootstrap=20))
    assert row["n_reference_irrigated"] == 2 and row["n_predicted_ignored"] == 2
    for key in ("recall_irrigated_score", "f1_irrigated"):
        low, high = row[f"{key}_ci_low"], row[f"{key}_ci_high"]
        assert 0.0 <= low <= row[key] <= high <= 1.0, key
    text = json.dumps(to_jsonable({"row": row, **record}), allow_nan=False)
    assert json.loads(text)["metrics"]["bootstrap"]["n_resamples"] == 20


def test_the_second_table(ex24, layers, capsys):
    pred, reference = layers
    row, _ = ex24.evaluate_irrigated(pred, reference, 2022, argparse.Namespace(bootstrap=0))
    info = {"basin": "diamond_valley", "run": "naip da", "year": 2022, "status": "ok"}
    failed = {"basin": "diamond_valley", "run": "spot da", "year": 2023, "status": "failed: X"}
    other_basin = {**info, "basin": "mason_valley", **row}
    ex24.print_irrigated_table("diamond_valley", [{**info, **row}, failed, other_basin])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("Irrigated-only score: reference = the fields irrigated")
    assert lines[2].split() == [
        "Run", "Year", "Ref", "Scored", "Ignored", "P", "R", "F1", "IoU", "bF1", "FP-noref",
    ]  # fmt: skip
    assert len(lines) == 4  # one row: the failed run and the other basin are left out
    # Mean IoU of the two matches, 1 and 0.9025.
    assert lines[3].split()[:10] == [
        "naip", "da", "2022", "2", "3", "2", "0.67", "1.00", "0.80", "0.95",
    ]  # fmt: skip
    assert not math.isnan(row["boundary_f1_irrigated"])


def _run_args(ex24, *argv, spot=True):
    args = ex24.parse_args(["--gee-project", "test-project", *argv])
    args.spot = spot  # set by main() after the SPOT access check
    return args


def test_the_sam_auto_runs_are_opt_in(ex24):
    """--sam-auto adds the direct SAM runs after the default runs, which do not change."""
    default = ex24.basin_runs("diamond_valley", _run_args(ex24), {})
    assert not any(engine == "sam-auto" for _, _, engine, _ in default)
    runs = ex24.basin_runs("diamond_valley", _run_args(ex24, "--sam-auto"), {})
    assert runs[: len(default)] == default
    # The imagery types the engine's pre-registered evaluation kept: not Sentinel-2.
    assert runs[len(default) :] == [
        ("sam-auto", source, "sam-auto", {}) for source in ("naip", "spot", "landsat")
    ]
    no_spot = ex24.basin_runs("diamond_valley", _run_args(ex24, "--sam-auto", spot=False), {})
    assert [s for label, s, _, _ in no_spot if label == "sam-auto"] == ["naip", "landsat"]
    assert "sentinel2" not in ex24.SAM_AUTO_SOURCES


#: Stand-in checkpoints of the fine-tuned models (basin_runs only passes them on).
CHECKPOINTS = {
    (slug, source, engine): f"/ck/{slug}_{source}_{engine}.ckpt"
    for slug in ("diamond_valley", "mason_valley")
    for source in ("naip", "spot")
    for engine in ("delineate-anything", "dinov3")
}
DA = "delineate-anything"


def test_the_option_runs_are_opt_in(ex24):
    """--options adds four "-opt" runs per basin after the other runs, which do not change."""
    for slug in ("diamond_valley", "mason_valley"):
        default = ex24.basin_runs(slug, _run_args(ex24), CHECKPOINTS)
        assert not any(label.endswith("-opt") for label, _, _, _ in default)
        runs = ex24.basin_runs(slug, _run_args(ex24, "--options"), CHECKPOINTS)
        assert runs[: len(default)] == default
        assert [f"{s} {label}" for label, s, _, _ in runs[len(default) :]] == [
            f"{name}-opt" for name in ex24.OPTION_BASES[slug]
        ]
        # With --sam-auto too, the "-opt" runs come last.
        both = ex24.basin_runs(slug, _run_args(ex24, "--options", "--sam-auto"), CHECKPOINTS)
        assert both[: len(default)] == default and both[-4:] == runs[len(default) :]
    dv = ex24.basin_runs("diamond_valley", _run_args(ex24, "--options"), CHECKPOINTS)[-4:]
    assert dv == [
        ("da-opt", "naip", DA, {"engine_params": {"processing_gsd_m": 3.0, "pivots": "snap"}}),
        (
            "dinov3-ft-opt",
            "naip",
            "dinov3",
            {
                "engine_params": {
                    "checkpoint_path": "/ck/diamond_valley_naip_dinov3.ckpt",
                    "dilate_interior_px": "half_band",
                    "pivots": "snap",
                }
            },
        ),
        (
            "dinov3-ft-opt",
            "spot",
            "dinov3",
            {
                "engine_params": {
                    "checkpoint_path": "/ck/diamond_valley_spot_dinov3.ckpt",
                    "dilate_interior_px": "half_band",
                    "pivots": "snap",
                }
            },
        ),
        ("da-opt", "sentinel1", DA, {"engine_params": {"max_detections": 1000, "pivots": "snap"}}),
    ]
    mv = ex24.basin_runs("mason_valley", _run_args(ex24, "--options"), CHECKPOINTS)[-4:]
    assert mv[1][3]["engine_params"]["checkpoint_path"] == "/ck/mason_valley_naip_dinov3.ckpt"
    # Mason Valley's SPOT run is Delineate-Anything as released: the half band is DINOv3's.
    assert mv[2] == ("da-opt", "spot", DA, {"engine_params": {"pivots": "snap"}})


def test_an_option_run_is_skipped_with_its_base_run(ex24):
    """No SPOT access, no fine-tuned DINOv3: the "-opt" runs of those models are left out."""
    no_spot = ex24.basin_runs("diamond_valley", _run_args(ex24, "--options", spot=False), {})
    assert [f"{s} {label}" for label, s, _, _ in no_spot if label.endswith("-opt")] == [
        "naip da-opt",
        "sentinel1 da-opt",
    ]
    only_da = {k: v for k, v in CHECKPOINTS.items() if k[2] == DA}
    runs = ex24.basin_runs("mason_valley", _run_args(ex24, "--options"), only_da)
    assert [f"{s} {label}" for label, s, _, _ in runs if label.endswith("-opt")] == [
        "naip da-opt",
        "spot da-opt",
        "sentinel1 da-opt",
    ]


def test_an_option_run_has_the_configuration_of_its_base_run(ex24):
    """Only engine_params (the base run's plus the options) and the output differ; the crop
    filter is the result of the optical runs, never of Sentinel-1."""
    import warnings

    args = _run_args(ex24, "--options")
    for slug in ("diamond_valley", "mason_valley"):
        out_dir = ex24.OUTPUT_DIR / slug
        runs = ex24.basin_runs(slug, args, CHECKPOINTS)
        by_name = {f"{s} {label}": (label, s, e, over) for label, s, e, over in runs}
        for name in ex24.OPTION_BASES[slug]:
            label, source, engine, over = by_name[f"{name}-opt"]
            base_label, _, base_engine, base_over = by_name[name]
            assert base_engine == engine
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the SPOT access warning
                opt = ex24.run_config(out_dir, source, engine, label, args, **over).to_dict()
                base = ex24.run_config(
                    out_dir, source, engine, base_label, args, **base_over
                ).to_dict()
            assert {k for k in opt if opt[k] != base[k]} == {"engine_params", "output_path"}
            assert opt["output_path"].endswith(
                f"fields_{source}_{base_label}-opt_{ex24.SOURCE_YEARS[source]}.gpkg"
            )
            extra = ex24.option_params(source, engine)
            assert opt["engine_params"] == {**base["engine_params"], **extra}
            assert list(extra)[-1] == "pivots"  # the pivot snap after the engine's options
            steps = ex24.option_steps(source, engine)
            assert ex24.option_crop(source) == (source != "sentinel1")
            assert steps[-1].startswith("crop filter") == ex24.option_crop(source)


def test_the_change_of_an_option_run(ex24, layers):
    """The paired change: result minus base, all fields and irrigated only, 1 km blocks."""
    pred, reference = layers
    base = pred.iloc[[1, 2, 3, 4]]  # without the first pivot
    change = ex24.option_change(pred, base, reference, 2022)
    assert set(change) == {"all", "irrigated"}
    for part, ref in (("all", reference), ("irrigated", ex24.irrigated_only(reference, 2022)[0])):
        record = change[part]
        assert record["n_resamples"] == ex24.N_PAIRED and record["seed"] == 42
        assert record["block_size_m"] == 1000.0 and record["block_crs"] == "EPSG:32611"
        for key in ("f1", "boundary_f1"):
            low, high = record["ci"][key]
            assert low <= record["difference"][key] <= high, (part, key)
        # The point estimate is the difference of the two layers' own scores.
        a = ex24.evaluate_run(pred, reference, 2022, argparse.Namespace(bootstrap=0))[0]
        b = ex24.evaluate_run(base, reference, 2022, argparse.Namespace(bootstrap=0))[0]
        if part == "all":
            assert record["difference"]["f1"] == pytest.approx(a["f1"] - b["f1"])
        assert record["a"]["count_reference"] == len(ref)
    irr_a = ex24.evaluate_irrigated(pred, reference, 2022, argparse.Namespace(bootstrap=0))[0]
    irr_b = ex24.evaluate_irrigated(base, reference, 2022, argparse.Namespace(bootstrap=0))[0]
    assert change["irrigated"]["difference"]["f1"] == pytest.approx(
        irr_a["f1_irrigated"] - irr_b["f1_irrigated"]
    )


def test_the_options_row_and_table(ex24, layers, capsys):
    pred, reference = layers
    base = pred.iloc[[1, 2, 3, 4]]
    rows = {}
    for name, layer in (("naip dinov3-ft", base), ("naip dinov3-ft-opt", pred)):
        info = {"basin": "diamond_valley", "run": name, "source": "naip", "engine": "dinov3"}
        main, _ = ex24.evaluate_run(layer, reference, 2022, argparse.Namespace(bootstrap=0))
        irr, _ = ex24.evaluate_irrigated(layer, reference, 2022, argparse.Namespace(bootstrap=0))
        rows[name] = {**info, "year": 2022, "status": "ok", **main, **irr}
    change = ex24.option_change(pred, base, reference, 2022)
    pivots = {"n_groups": 2, "n_groups_multi": 1, "n_pieces_merged": 3}
    meta = {"dilate_interior_px": 6, "dilate_interior_rule": "half_band", "band_thickness_px": 12}
    record = {"facts": {"n_output": 7, "pivots": pivots}, "engine_meta": meta}
    row = ex24.option_row(rows["naip dinov3-ft-opt"], rows["naip dinov3-ft"], change, record)
    assert (row["run"], row["base"], row["result"]) == (
        "naip dinov3-ft-opt",
        "naip dinov3-ft",
        "crop-filter layer",
    )
    options = "dilate_interior_px=half_band; pivots=snap; crop filter (threshold 0.3)"
    assert row["options"] == options
    assert (row["n_output"], row["n_result"], row["growth_px"], row["band_px"]) == (7, 5, 6, 12)
    assert (row["pivot_groups_multi"], row["pivot_pieces_merged"]) == (1, 3)
    assert row["d_f1"] == pytest.approx(row["f1"] - row["f1_base"])
    assert row["d_f1_irrigated"] == pytest.approx(row["f1_irrigated"] - row["f1_irrigated_base"])
    assert row["d_boundary_f1"] == pytest.approx(row["boundary_f1"] - row["boundary_f1_base"])
    assert row["n_blocks"] == change["all"]["n_blocks"]
    # Without a base run: no change, and the table prints "-".
    alone = ex24.option_row(rows["naip dinov3-ft-opt"], {"run": "naip dinov3-ft"}, None, None)
    assert alone["d_f1"] is None and alone["f1_base"] is None and alone["growth_px"] is None
    ex24.print_option_table("diamond_valley", [row, alone, {**row, "basin": "mason_valley"}])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith('"-opt" runs: change against the base run')
    assert len(lines) == 5  # two lines of text, the header and the two Diamond Valley rows
    first = ["naip", "dinov3-ft-opt", "2022", "5", f"{row['f1_base']:.3f}", "->"]
    assert lines[3].split()[:6] == first
    assert lines[4].split()[-3:] == ["-", "-", "-"]
    ex24.print_option_table("diamond_valley", [])  # no "-opt" run: nothing printed
    assert capsys.readouterr().out == ""


def test_the_dry_run_lists_the_option_runs(ex24, capsys):
    """The dry run shows each "-opt" run's options; without --options nothing changes."""
    labels = gpd.GeoDataFrame(geometry=[sq(-6000, -6000)], crs=ex24.UTM_CRS)
    basins = {"diamond_valley": {"labels": labels}}
    ex24.dry_run(["diamond_valley"], basins, _run_args(ex24))
    default = capsys.readouterr().out
    assert "-opt_" not in default and "pivots=" not in default
    ex24.dry_run(["diamond_valley"], basins, _run_args(ex24, "--options"))
    out = capsys.readouterr().out
    assert out.startswith(default.rstrip("\n"))  # the default runs first, unchanged
    added = out[len(default.rstrip("\n")) :].strip().splitlines()
    assert len(added) == 8  # four runs, each with a line of options
    assert added[0].split()[-1].endswith("fields_naip_da-opt_2022.gpkg")
    assert added[1].strip() == (
        "(processing_gsd_m=3.0; pivots=snap; then the crop filter, whose layer is scored)"
    )
    assert "dilate_interior_px=half_band; pivots=snap; then the crop filter" in added[3]
    assert added[7].strip() == "(pivots=snap)"  # Sentinel-1: never crop-filtered


def test_a_sam_auto_run_has_the_configuration_of_the_released_run(ex24):
    """Only the engine, its engine_params and the output differ from Delineate-Anything's run
    on the same source (as the pre-registered evaluation of the engine requires)."""
    import warnings

    args = _run_args(ex24, "--sam-auto")
    out_dir = ex24.OUTPUT_DIR / "diamond_valley"
    released = {
        source: overrides
        for label, source, engine, overrides in ex24.basin_runs("diamond_valley", args, {})
        if label == "da" and engine == "delineate-anything"
    }
    for source in ex24.SAM_AUTO_SOURCES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the SPOT access warning
            sam = ex24.run_config(out_dir, source, "sam-auto", "sam-auto", args).to_dict()
            da = ex24.run_config(
                out_dir, source, "delineate-anything", "da", args, **released[source]
            ).to_dict()
        assert sam["engine_params"] == {}
        assert sam["output_path"].endswith(
            f"fields_{source}_sam-auto_{ex24.SOURCE_YEARS[source]}.gpkg"
        )
        differ = {k for k in sam if sam[k] != da[k]}
        assert differ == {"engine", "engine_params", "output_path"} - (
            {"engine_params"} if not da["engine_params"] else set()
        ), source
