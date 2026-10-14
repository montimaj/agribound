"""Example 28 (examples/28_new_york_field_boundaries.py) on synthetic data; no network, no model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "28_new_york_field_boundaries.py"
UTM = "EPSG:32618"


@pytest.fixture(scope="module")
def ex28():
    spec = importlib.util.spec_from_file_location("example_28_new_york", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(ex28, *argv, **changes):
    args = ex28.parse_args(["--gee-project", "test-project", *argv])
    args.sam_auto_skipped = None
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def _fields(rectangles):
    """OpenET-like fields in EPSG:4326 from (x0, y0, w, h) rectangles in UTM zone 18N, with the
    file's columns: CDL class 1 (corn) and PP_SCORE 0.6, unless a 5th and a 6th value give them."""
    rows, geoms = [], []
    for i, (x0, y0, w, h, *rest) in enumerate(rectangles):
        crop = rest[0] if rest else 1
        pp = rest[1] if len(rest) > 1 else 0.6
        geoms.append(box(x0, y0, x0 + w, y0 + h))
        rows.append(
            {
                "Unique_ID": 21136000001 + i,
                "SOURCECODE": "CLU",
                "MOD_DATE": "2021-04-21",
                "FIPS": "36011",
                "PP_SCORE": pp,
                "CROP_2021": crop,
                "CROP_2022": crop,
                "CROP_2024": crop,
            }
        )
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=UTM).to_crs(4326)


def _region_fields(ex28, n=4):
    """A few 200 m fields in every evaluation and training square of every region."""
    rectangles = []
    for spec in ex28.REGIONS.values():
        for kind in ("evaluation", *ex28.ARMS):
            if kind in spec:
                x0, y0 = spec[kind]
                rectangles += [(x0 + 500 + 300 * i, y0 + 500, 200, 200) for i in range(n)]
    return _fields(rectangles)


def _openet_dir(tmp_path, fields):
    """A folder with NY.zip (a shapefile at the root of the archive), as the delivery has it."""
    build, folder = tmp_path / "build", tmp_path / "openet"
    build.mkdir()
    folder.mkdir()
    fields.to_file(build / "NY.shp")
    with zipfile.ZipFile(folder / "NY.zip", "w") as archive:
        for part in sorted(build.iterdir()):
            archive.write(part, part.name)
    return folder


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("example 28 tried to use the network")

    import urllib.request

    import agribound.auth

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(agribound.auth, "setup_gee", refuse)


# --- Squares, years, runs -----------------------------------------------------------------


def test_the_regions_squares_and_arms(ex28):
    assert list(ex28.REGIONS) == ["black_dirt", "finger_lakes", "north_country"]
    r1 = ex28.REGIONS["black_dirt"]
    assert r1["evaluation"] == (545000, 4575000)  # example 25's New York square
    assert (r1["training"], r1["near_sample"]) == ((535000, 4575000), (539000, 4571000))
    assert ex28.REGIONS["finger_lakes"]["training"] == (370000, 4725000)
    assert ex28.REGIONS["north_country"]["training"] == (415000, 4860000)
    # Edge-to-edge gaps: A0 5 km (the rule's), A1 1 km, R2 15 km, R3 10 km.
    gaps = {
        (slug, arm): ex28.square(spec[arm]).distance(ex28.square(spec["evaluation"]))
        for slug, spec in ex28.REGIONS.items()
        for arm in ex28.ARMS
        if arm in spec
    }
    assert gaps == {
        ("black_dirt", "training"): 5000.0,
        ("black_dirt", "near_sample"): 1000.0,
        ("finger_lakes", "training"): 15000.0,
        ("north_country", "training"): 10000.0,
    }
    assert ex28.region_arms("black_dirt", _args(ex28)) == ["training", "near_sample"]
    assert ex28.region_arms("black_dirt", _args(ex28, "--no-near-sample")) == ["training"]
    assert ex28.region_arms("finger_lakes", _args(ex28)) == ["training"]
    assert ex28.train_name("black_dirt", "near_sample") == "black_dirt_train_a1"
    assert ex28.train_name("north_country", "training") == "north_country_train"


def test_the_year_rule_and_the_output_names(ex28, tmp_path):
    args = _args(ex28)
    expected = {
        "naip": (2022, None),
        "spot": (2021, ("2021-04-01", "2021-10-31")),
        "sentinel2": (2024, ("2024-04-01", "2024-10-31")),
        "landsat-pan": (2024, ("2024-04-01", "2024-10-31")),
        "sentinel1": (2024, ("2024-04-01", "2024-10-31")),
    }
    for source, (year, window) in expected.items():
        config = ex28.run_config(tmp_path, source, "delineate-anything", "da", args)
        assert (config.year, config.date_range, config.lulc_filter) == (year, window, False)
        assert config.output_path.endswith(f"fields_{source}_da_{year}.gpkg")
    assert ex28.run_config(tmp_path, "spot", "delineate-anything", "da", args).cloud_cover_max == 15
    ftw = ex28.run_config(tmp_path, "sentinel2", "ftw", "ftw", args)
    assert ftw.date_range is None  # FTW builds its own two windows
    tune = ex28.fine_tune_config(
        "black_dirt", "near_sample", "naip", "delineate-anything", tmp_path / "labels.gpkg", args
    )
    assert tune.fine_tune and tune.fine_tune_split == "random" and tune.fine_tune_val_split == 0.2
    assert tune.fine_tune_epochs == 20 and tune.engine_params == {"yolo_lr0": 1e-4}
    assert tune.output_path.endswith("black_dirt_train_a1/fields_naip_da-finetune_2022.gpkg")


def _labels(runs):
    return [f"{source} {label}" for label, source, _, _ in runs]


def test_the_runs_of_a_region(ex28):
    checkpoints = {
        ("black_dirt", arm, "naip", engine): f"{arm}-{engine}.ckpt"
        for arm in ex28.ARMS
        for engine in ("delineate-anything", "dinov3")
    }
    runs = ex28.region_runs("black_dirt", _args(ex28), checkpoints)
    assert _labels(runs) == [
        "naip da", "naip da-sam2", "naip da-ft", "naip dinov3-ft", "naip da-ft-a1",
        "naip dinov3-ft-a1", "naip sam-auto", "sentinel2 da", "sentinel2 ftw",
    ]  # fmt: skip
    by_label = {f"{source} {label}": (engine, o) for label, source, engine, o in runs}
    assert by_label["naip sam-auto"] == ("sam-auto", {})  # the engine's own NAIP default
    assert by_label["sentinel2 da"][1] == {"engine_params": {"max_detections": 1000}}
    assert by_label["naip da-sam2"][1] == {"sam_refine": True}
    assert by_label["naip dinov3-ft-a1"][1]["engine_params"]["checkpoint_path"] == (
        "near_sample-dinov3.ckpt"
    )
    # A failed fine-tuning leaves its run out; the region without A1 has no -a1 runs.
    assert _labels(ex28.region_runs("finger_lakes", _args(ex28), {})) == [
        "naip da", "naip da-sam2", "naip sam-auto", "sentinel2 da", "sentinel2 ftw",
    ]  # fmt: skip
    flags = _args(ex28, "--no-sam", "--no-sam-auto", "--sam-auto-sentinel2", "--extended")
    assert _labels(ex28.region_runs("north_country", flags, {})) == [
        "naip da", "sentinel2 da", "sentinel2 ftw", "sentinel2 sam-auto", "landsat-pan da",
        "sentinel1 da", "google-embedding k20", "tessera-embedding k20",
    ]  # fmt: skip
    spot = ex28.region_runs("north_country", _args(ex28, "--spot", "--no-sam"), {})
    assert _labels(spot)[:3] == ["naip da", "naip sam-auto", "spot da"]
    check = _args(ex28, "--pipeline-check", "--spot", "--extended")
    assert _labels(ex28.region_runs("black_dirt", check, checkpoints)) == [
        "naip da", "sentinel2 da",
    ]  # fmt: skip
    assert ex28.fine_tune_plan(check) == []


def test_the_fine_tuning_plan(ex28):
    assert ex28.fine_tune_plan(_args(ex28)) == [
        ("naip", "delineate-anything"),
        ("naip", "dinov3"),
    ]
    assert ex28.fine_tune_plan(_args(ex28, "--spot", "--no-dinov3")) == [
        ("naip", "delineate-anything"),
        ("spot", "delineate-anything"),
    ]
    assert ex28.fine_tune_plan(_args(ex28, "--no-fine-tune")) == []


def test_sam_auto_is_skipped_with_a_message_when_it_cannot_run(ex28, monkeypatch):
    import agribound.registry

    skipped = _args(ex28, "--sam-auto-sentinel2", sam_auto_skipped="no engine here")
    assert "naip sam-auto" not in _labels(ex28.region_runs("finger_lakes", skipped, {}))
    assert "sentinel2 sam-auto" not in _labels(ex28.region_runs("finger_lakes", skipped, {}))
    assert ex28.skipped_runs(skipped) == [
        ("naip sam-auto", "naip", "no engine here"),
        ("sentinel2 sam-auto", "sentinel2", "no engine here"),
    ]
    assert ex28.skipped_runs(_args(ex28)) == []

    registry = {k: v for k, v in agribound.registry.ENGINE_REGISTRY.items() if k != "sam-auto"}
    monkeypatch.setattr(agribound.registry, "ENGINE_REGISTRY", registry)
    assert ex28.sam_auto_missing().endswith("has no 'sam-auto' engine")
    registry["sam-auto"] = {"install_extra": "sam-auto"}
    assert ex28.SAM_AUTO_MODULES == ("torch", "sam2", "cv2")

    def find_spec(absent):
        # The other modules count as installed whatever this environment has (CI's core job
        # has no torch).
        return lambda name: None if name in absent else object()

    monkeypatch.setattr(importlib.util, "find_spec", find_spec({"sam2"}))
    assert ex28.sam_auto_missing() == 'sam2 not installed (pip install "agribound[sam-auto]")'
    monkeypatch.setattr(importlib.util, "find_spec", find_spec({"cv2"}))
    assert ex28.sam_auto_missing() == 'cv2 not installed (pip install "agribound[sam-auto]")'
    monkeypatch.setattr(importlib.util, "find_spec", find_spec({"torch", "sam2", "cv2"}))
    assert ex28.sam_auto_missing().startswith("torch, sam2 and cv2 not installed")
    monkeypatch.setattr(importlib.util, "find_spec", find_spec(set()))
    assert ex28.sam_auto_missing() is None


def test_the_sample_labels(ex28):
    assert ex28.sample_of("da", "delineate-anything") == "unseen"
    assert ex28.sample_of("da-sam2", "delineate-anything") == "unseen"
    assert ex28.sample_of("ftw", "ftw") == "unseen"
    assert ex28.sample_of("da-ft", "delineate-anything") == "held out"
    assert ex28.sample_of("dinov3-ft", "dinov3") == "held out"
    assert ex28.sample_of("da-ft-a1", "delineate-anything") == "near-sample"
    assert ex28.sample_of("dinov3-ft-a1", "dinov3") == "near-sample"
    assert ex28.sample_of("sam-auto", "sam-auto") == "label-free"
    assert ex28.sample_of("k20", "embedding") == "label-free"


def test_r1_reads_example_25s_composites(ex28, tmp_path):
    """R1's study area is written as example 25 writes it, so its NAIP 2022 and Sentinel-2 2024
    composites have the cache keys of example 25's New York files (the pipeline check)."""
    from agribound._cache import cache_key, clear_fingerprint_cache
    from agribound.composites.gee import _recipe_parts

    area = ex28.square(ex28.REGIONS["black_dirt"]["evaluation"])
    (tmp_path / "study_area.geojson").write_text(json.dumps(ex28.square_study_area(area, "R1")))
    clear_fingerprint_cache()
    args = _args(ex28)
    naip = ex28.run_config(tmp_path, "naip", "delineate-anything", "da", args)
    s2 = ex28.run_config(
        tmp_path, "sentinel2", "delineate-anything", "da", args,
        engine_params={"max_detections": 1000},
    )  # fmt: skip
    # outputs/examples_1.1/outputs/multistate_openet/ny/.agribound_cache (example 25, 1.1 run)
    assert cache_key(naip, *_recipe_parts("naip")) == "61af56d9eb7b"
    assert cache_key(s2, *_recipe_parts("sentinel2")) == "357fc6cfc395"


# --- Reference --------------------------------------------------------------------------------


def test_land_use_groups_and_strata(ex28):
    assert ex28.land_use(1) == "row crops" and ex28.land_use(5) == "row crops"
    assert ex28.land_use(49) == "vegetables and sod" and ex28.land_use(59) == "vegetables and sod"
    assert ex28.land_use(176) == "hay and pasture" and ex28.land_use(37) == "hay and pasture"
    assert ex28.land_use(68) == "fruit and vines" and ex28.land_use(61) == "fallow"
    assert ex28.land_use(31) == "other farmed"  # canola: farmed, in no group
    assert ex28.land_use(141) == "not farmed" and ex28.land_use(121) == "not farmed"
    assert ex28.land_use(0) is None  # no class assigned
    assert set(ex28.LAND_USE_KEYS) == {*ex28.LAND_USE, "other farmed", "not farmed"}
    reference = _fields([(0, 0, 100, 100, 49, 0.2), (200, 0, 100, 100, 0, float("nan"))])
    reference = ex28.add_strata(reference)
    # A missing group or shape (no class, no PP_SCORE): the field is in no stratum.
    assert reference["land_use_2022"][0] == "vegetables and sod"
    assert reference["shape"][0] == "strip-like"
    assert reference["land_use_2022"].isna()[1] and reference["shape"].isna()[1]
    assert {f"land_use_{y}" for y in (2021, 2022, 2024)} <= set(reference.columns)


def test_prepare_region_reads_the_squares(ex28, tmp_path, capsys):
    x0, y0 = ex28.REGIONS["finger_lakes"]["evaluation"]
    tx, ty = ex28.REGIONS["finger_lakes"]["training"]
    fields = _fields(
        [
            (x0 + 100, y0 + 100, 300, 300),
            (x0 + 600, y0 + 100, 40, 40),  # under 2,500 m2: dropped from the reference
            (x0 - 250, y0 + 1000, 300, 300),  # representative point outside the square
            (tx + 100, ty + 100, 300, 300),
            (tx - 290, ty + 100, 300, 300),  # reaches into the training square: a label
            (tx + 4990, ty + 4990, 5, 5),  # a small label: labels have no minimum size
        ]
    )
    fields.to_file(tmp_path / "ny.gpkg")
    region = ex28.prepare_region("finger_lakes", str(tmp_path / "ny.gpkg"))
    assert len(region["reference"]) == 1 and region["n_small"] == 1
    assert len(region["labels"]["training"]) == 3 and region["training_fields"]["training"] == 1
    assert region["reference"].crs == UTM
    out = capsys.readouterr().out
    assert "finger_lakes evaluation square: 1 fields, not the recorded 482" in out
    assert "finger_lakes training square: 3 fields, not the recorded 505" in out


def test_a_training_label_in_the_evaluation_square_is_refused(ex28, tmp_path):
    x0, y0 = ex28.REGIONS["finger_lakes"]["evaluation"]
    tx, ty = ex28.REGIONS["finger_lakes"]["training"]
    # One long field from the training square into the evaluation square (15 km apart).
    fields = _fields([(tx + 100, ty + 100, 300, 300), (tx - 100, ty + 4000, 200, 16500)])
    assert x0 + 5000 == tx and ty + 20500 > y0
    fields.to_file(tmp_path / "ny.gpkg")
    with pytest.raises(ValueError, match="reaches into the evaluation square"):
        ex28.prepare_region("finger_lakes", str(tmp_path / "ny.gpkg"))


def test_the_dry_run_reads_the_file_and_writes_nothing(
    ex28, tmp_path, monkeypatch, capsys, no_network
):
    folder = _openet_dir(tmp_path, _region_fields(ex28))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    # A project is given, as on a CI runner, where neither $GEE_PROJECT nor gcloud has one.
    argv = ["28", "--dry-run", "--openet-dir", str(folder), "--gee-project", "test-project"]
    monkeypatch.setattr(sys, "argv", argv)
    ex28.main()
    out = capsys.readouterr().out
    assert list(work.iterdir()) == []  # nothing written
    assert "black_dirt evaluation square: 4 fields, not the recorded 668" in out
    assert "near_sample square: corner (539000, 4571000)" in out
    assert "1.0 km from the evaluation square" in out
    assert "Dry run: configurations" in out
    assert "black_dirt_train_a1/fields_naip_dinov3-finetune_2022.gpkg" in out
    assert "north_country/fields_sentinel2_ftw_2024.gpkg" in out
    assert "pipeline check: naip da must give 585 polygons, 92 matches and F1 0.1468" in out
    # The direct SAM run is configured, or skipped with the reason.
    assert "black_dirt/fields_naip_sam-auto_2022.gpkg" in out or "naip sam-auto: skipped:" in out


def test_a_run_writes_rows_metrics_panels_and_the_pipeline_check(
    ex28, tmp_path, monkeypatch, capsys, no_network
):
    """main() with a fake agribound.delineate and crop filter: R1's released runs, SAM 2
    refinement and FTW, scored and compared as in a real run."""
    import pandas as pd

    import agribound
    import agribound.postprocess.lulc_filter as lulc
    import agribound.visualize as visualize

    fields = _region_fields(ex28)
    reference = fields.to_crs(UTM)
    folder = _openet_dir(tmp_path, fields)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    def fake_delineate(config):
        """The reference fields of the square (shrunk to no match on Sentinel-2 with
        Delineate-Anything) and five 1 km blocks on no field."""
        area = gpd.read_file(config.study_area).to_crs(UTM).geometry.iloc[0]
        x0, y0 = (round(v) for v in area.bounds[:2])
        inside = reference[reference.representative_point().within(area).to_numpy()]
        exact = "sentinel2_da_" not in Path(config.output_path).name
        geoms = [g if exact else g.buffer(-70) for g in inside.geometry]
        geoms += [box(x0 + 1000 * i, y0 + 3000, x0 + 1000 * i + 990, y0 + 3990) for i in range(5)]
        out = gpd.GeoDataFrame({"id": range(len(geoms))}, geometry=geoms, crs=UTM)
        out.to_crs(4326).to_file(config.output_path, layer="fields")
        return out.to_crs(4326)

    monkeypatch.setattr(agribound, "delineate", fake_delineate)
    monkeypatch.setattr(lulc, "filter_by_lulc", lambda gdf, config: gdf.iloc[: len(gdf) // 2])
    monkeypatch.setattr(visualize, "show_comparison", lambda *args, **kwargs: None)
    common = ["--openet-dir", str(folder), "--gee-project", "test-project", "--bootstrap", "20"]

    monkeypatch.setattr(sys, "argv", ["28", "--pipeline-check", *common])
    ex28.main()
    out_dir = work / "outputs" / "new_york_field_boundaries"
    check = json.loads((out_dir / "pipeline_check.json").read_text())
    assert set(check["result"]) == {"naip da", "sentinel2 da"}
    assert check["result"]["naip da"].startswith("FAILED: (polygons, matches, F1) (9, 4, 0.6154)")
    assert not (out_dir / "summary.csv").exists() and not (out_dir / "finger_lakes").exists()
    assert "Pipeline check (pre-registered): FAILED for naip da, sentinel2 da" in (
        capsys.readouterr().out
    )

    argv = ["28", "--only", "black_dirt", "--no-fine-tune", "--no-sam-auto", *common]
    monkeypatch.setattr(sys, "argv", argv)
    ex28.main()
    summary = pd.read_csv(out_dir / "summary.csv").set_index("run")
    assert list(summary.index) == ["naip da", "naip da-sam2", "sentinel2 da", "sentinel2 ftw"]
    assert set(summary["status"]) == {"ok"} and set(summary["sample"]) == {"unseen"}
    assert summary.loc["naip da", "f1"] == pytest.approx(8 / 13)  # 4 matches, 4 + 9 polygons
    assert summary.loc["naip da", "kept_crop"] == 4 and summary.loc["sentinel2 da", "f1"] == 0
    assert summary.loc["naip da-sam2", "compared_with"] == "naip da"
    assert summary.loc["naip da-sam2", "better_by_rule"] == "no clear difference"
    assert summary.loc["sentinel2 ftw", "better_by_rule"] == "better"
    assert pd.isna(summary.loc["naip da", "compared_with"])
    assert summary.loc["naip da", "pipeline_check"].startswith("FAILED")
    assert summary.loc["naip da", "grid_cell_m"] == pytest.approx(990, rel=0.01)
    panels = json.loads((out_dir / "panels.json").read_text())
    assert panels == {
        "black_dirt": {
            "top_left": "naip da",
            "top_right": "naip da-sam2",
            "bottom_left": "sentinel2 ftw",
            "bottom_right": "sentinel2 da",  # no sam-auto, no SPOT: the other Sentinel-2 run
        }
    }
    record = json.loads(
        (out_dir / "black_dirt" / "metrics_fields_sentinel2_ftw_2024.json").read_text()
    )
    assert record["row"]["better_by_rule"] == "better"
    assert set(record["paired"]) == {"field", "block"}
    assert record["paired"]["block"]["block_crs"] == UTM
    assert {"metrics", "metrics_land_use", "metrics_shape"} <= set(record)
    references = sorted(p.name for p in (out_dir / "references").glob("black_dirt_reference_*"))
    assert len(references) == 2 and references[1].endswith("_2024_crop_t0.3.gpkg")


def test_without_the_openet_folder_the_script_stops(ex28, monkeypatch, tmp_path):
    monkeypatch.delenv(ex28.DATA_ENV, raising=False)
    with pytest.raises(SystemExit, match="not public yet"):
        ex28.openet_dir(argparse.Namespace(openet_dir=None))
    with pytest.raises(SystemExit, match="NY.zip not found"):
        ex28.openet_path(tmp_path)


# --- Scores -----------------------------------------------------------------------------------


def _square_reference(ex28, n=4, crop=1):
    """*n* x *n* fields of 200 m in the corner of R2's evaluation square, with strata."""
    x0, y0 = ex28.REGIONS["finger_lakes"]["evaluation"]
    cells = [(x0 + 200 * i, y0 + 200 * j, 200, 200, crop) for j in range(n) for i in range(n)]
    reference = ex28.add_strata(_fields(cells).to_crs(UTM))
    return reference, ex28.square((x0, y0))


def _layer(geoms):
    return gpd.GeoDataFrame({"id": range(len(geoms))}, geometry=geoms, crs=UTM)


def test_merge_and_split_rates(ex28):
    ref = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)], crs=UTM)
    assert ex28.merge_split_rates(_layer([box(0, 0, 200, 100)]), ref) == (1.0, 0.0)
    halves = _layer([box(0, 0, 50, 100), box(50, 0, 100, 100)])
    assert ex28.merge_split_rates(halves, ref) == (0.0, 0.5)
    # A prediction that only touches a field does not merge it with the field it covers.
    touching = _layer([box(200, 0, 300, 100)])
    assert ex28.merge_split_rates(touching, ref) == (0.0, 0.0)


def test_evaluate_run_gives_strata_baselines_and_the_chance_rule(ex28):
    reference, area = _square_reference(ex28)
    x0, y0 = area.bounds[:2]
    far = box(x0 + 4000, y0 + 4000, x0 + 4500, y0 + 4500)  # on no field
    pred = _layer(list(reference.geometry[:12]) + [far])
    row, records = ex28.evaluate_run(pred, reference, 2022, area, argparse.Namespace(bootstrap=20))
    assert (row["fields"], row["tp"]) == (13, 12)
    assert row["recall"] == pytest.approx(12 / 16) and row["precision"] == pytest.approx(12 / 13)
    assert (row["recall_row"], row["n_row"]) == (pytest.approx(0.75), 16)
    assert row["n_veg"] == 0 and row["recall_veg"] is None
    assert (row["tp_compact"], row["n_compact"], row["n_strip"]) == (12, 16, 0)
    assert math.isnan(row["recall_0-1_ha"]) and row["recall_2-5_ha"] == pytest.approx(0.75)
    assert row["fp_noref"] == 1
    assert row["grid_cell_m"] == pytest.approx(200, rel=0.01) and row["shift_m"] > 0
    assert 0 <= row["f1_ci_low"] <= row["f1"] <= row["f1_ci_high"] <= 1
    assert row["above_chance"] == (row["f1_ci_low"] > max(row["grid_f1"], row["shifted_f1"]))
    assert set(records) == {"metrics", "metrics_land_use", "metrics_shape"}
    assert records["metrics"]["bootstrap"]["resampling"].endswith("within strata")


def test_the_chance_rule(ex28):
    row = {"f1_ci_low": 0.30, "grid_f1": 0.10, "shifted_f1": 0.25}
    assert ex28.above_chance(row) is True
    assert ex28.above_chance({**row, "shifted_f1": 0.31}) is False
    assert ex28.above_chance({**row, "f1_ci_low": None}) is None
    assert ex28.above_chance({**row, "grid_f1": math.nan}) is None


def test_paired_differences_and_the_claim_rule(ex28):
    # 16 fields of 200 m, one in each of 16 blocks of 1 km of R2's evaluation square.
    x0, y0 = ex28.REGIONS["finger_lakes"]["evaluation"]
    cells = [
        (x0 + 1200 * i + 100, y0 + 1200 * j + 100, 200, 200) for j in range(4) for i in range(4)
    ]
    reference = ex28.add_strata(_fields(cells).to_crs(UTM))
    perfect = _layer(list(reference.geometry))
    poor = _layer([g.buffer(-70) for g in reference.geometry])  # IoU 0.09: no match
    same, records = ex28.paired_difference(perfect, perfect, reference)
    assert same["f1_diff"] == 0 and same["better_by_rule"] == "no clear difference"
    assert same["f1_diff_field_ci_low"] == same["f1_diff_block_ci_high"] == 0
    assert records["field"]["n_resamples"] == 2000 and records["field"]["block_size_m"] is None
    assert records["block"]["n_blocks"] == 16 and records["block"]["block_crs"] == UTM
    better, _ = ex28.paired_difference(perfect, poor, reference)
    assert better["f1_diff"] == pytest.approx(1.0) and better["better_by_rule"] == "better"
    assert ex28.paired_difference(poor, perfect, reference)[0]["better_by_rule"] == "worse"
    # Each layer matches half of the fields: the same F1, intervals on both sides of 0.
    half_a = _layer(list(reference.geometry[:8]) + [g.buffer(-70) for g in reference.geometry[8:]])
    half_b = _layer([g.buffer(-70) for g in reference.geometry[:8]] + list(reference.geometry[8:]))
    split, _ = ex28.paired_difference(half_a, half_b, reference)
    assert split["f1_diff"] == 0 and split["better_by_rule"] == "no clear difference"
    assert split["f1_diff_field_ci_low"] < 0 < split["f1_diff_block_ci_high"]


def test_the_claim_rule_needs_both_intervals(ex28):
    diff = {
        "f1_diff_field_ci_low": 0.01, "f1_diff_field_ci_high": 0.2,
        "f1_diff_block_ci_low": 0.02, "f1_diff_block_ci_high": 0.2,
    }  # fmt: skip
    assert ex28.better_by_rule(diff) == "better"
    assert ex28.better_by_rule({**diff, "f1_diff_block_ci_low": -0.02}) == "no clear difference"
    assert ex28.better_by_rule({**diff, "f1_diff_block_ci_low": math.nan}) == "no clear difference"
    worse = {
        "f1_diff_field_ci_low": -0.2, "f1_diff_field_ci_high": -0.01,
        "f1_diff_block_ci_low": -0.2, "f1_diff_block_ci_high": -0.02,
    }  # fmt: skip
    assert ex28.better_by_rule(worse) == "worse"


def test_the_gallery_panel_rule(ex28):
    def row(run, f1, bf1=0.5, status="ok", region="black_dirt"):
        return {"region": region, "run": run, "f1": f1, "boundary_f1": bf1, "status": status}

    rows = [
        row("naip da", 0.15),
        row("naip da-sam2", 0.20, bf1=0.40),
        row("naip da-ft", 0.20, bf1=0.45),  # ties with da-sam2 on F1: the higher bF1 wins
        row("naip dinov3-ft", 0.90, status="failed: ValueError"),
        row("naip da-ft-a1", 0.95),  # the near-sample arm is never a panel
        row("naip sam-auto", 0.30),
        row("sentinel2 da", 0.11),
        row("sentinel2 ftw", 0.12),
        row("sentinel2 sam-auto", 0.99),
        row("naip da", 0.5, region="finger_lakes"),
    ]
    assert ex28.gallery_panels(rows, "black_dirt") == {
        "top_left": "naip da",
        "top_right": "naip da-ft",
        "bottom_left": "sentinel2 ftw",
        "bottom_right": "naip sam-auto",
    }
    no_sam = [r for r in rows if r["run"] != "naip sam-auto"]
    assert ex28.gallery_panels(no_sam, "black_dirt")["bottom_right"] == "sentinel2 da"
    spot = no_sam + [row("spot da", 0.1), row("spot dinov3-ft", 0.2)]
    assert ex28.gallery_panels(spot, "black_dirt")["bottom_right"] == "spot dinov3-ft"
    tie = [row("sentinel2 da", 0.1, bf1=0.3), row("sentinel2 ftw", 0.1, bf1=0.3)]
    assert ex28.gallery_panels(tie, "black_dirt")["bottom_left"] == "sentinel2 da"
    assert ex28.gallery_panels([], "north_country") == dict.fromkeys(
        ("top_left", "top_right", "bottom_left", "bottom_right")
    )


def test_the_pipeline_check(ex28, tmp_path, monkeypatch):
    theirs = tmp_path / "example_25"
    theirs.mkdir()
    monkeypatch.setattr(ex28, "EXAMPLE_25_DIR", theirs)
    layer = _layer([box(0, 0, 100, 100), box(100, 0, 200, 100)])
    ours = tmp_path / "fields_naip_da_2022.gpkg"
    layer.to_file(ours)
    row = {"run": "naip da", "fields": 585, "tp": 92, "f1": 0.14684756584197925}
    assert ex28.pipeline_check(row, ours).startswith("passed (")
    assert "not found, polygons not compared" in ex28.pipeline_check(row, ours)
    layer.iloc[::-1].to_file(theirs / ours.name)  # the same polygons in another order
    assert ex28.pipeline_check(row, ours).startswith("passed (polygons compared")
    assert ex28.pipeline_check({**row, "tp": 91}, ours).startswith("FAILED: (polygons, matches")
    _layer([box(0, 0, 100, 100), box(100, 0, 201, 100)]).to_file(theirs / ours.name)
    assert "the polygons differ" in ex28.pipeline_check(row, ours)


def test_the_summary_keeps_the_rows_of_regions_not_run(ex28, tmp_path):
    import pandas as pd

    path = tmp_path / "summary.csv"
    earlier = [("north_country", "naip da", 0.3), ("black_dirt", "naip da", 0.1)]
    pd.DataFrame(earlier, columns=["region", "run", "f1"]).to_csv(path, index=False)
    rows = [{"region": "black_dirt", "run": "naip da", "f1": 0.2}]
    summary = ex28.merged_rows(path, rows, ["black_dirt"])
    assert list(summary.itertuples(index=False, name=None)) == [
        ("black_dirt", "naip da", 0.2),
        ("north_country", "naip da", 0.3),
    ]


# --- --select ---------------------------------------------------------------------------------


def test_the_square_rule(ex28):
    x0, y0 = 500000.0, 4600000.0  # x0 is the central meridian of UTM zone 18 (-75 degrees)
    county = box(x0, y0, x0 + 30000, y0 + 30000)

    def cluster(cx, cy, n):
        return [(cx + 100 + 50 * i, cy + 100) for i in range(n)]

    a, b, c = (x0 + 10000, y0 + 10000), (x0 + 20000, y0 + 10000), (x0 + 10000, y0 + 20000)
    points = cluster(*a, 30) + cluster(*b, 30) + cluster(*c, 30)
    points += cluster(x0 + 25000, y0 + 25000, 25)
    x, y = (np.array(v, dtype=float) for v in zip(*points, strict=True))
    farmed = np.ones(len(x), dtype=bool)
    # Three squares tie (30 farmed fields each); the southernmost square centre wins, by
    # latitude: east of the central meridian latitude falls along a row of UTM squares, so
    # b is south of a, and c lies a row further north.
    assert ex28.rule_square(x, y, farmed, county) == (b, 30, 30)
    gap = ex28.square(b)
    assert ex28.rule_square(x, y, farmed, county, exclude=gap, min_gap_m=5000) == (a, 30, 30)
    assert ex28.rule_square(x, y, farmed, county, exclude=gap, min_gap_m=10000) == (
        (x0 + 25000, y0 + 25000), 25, 25,
    )  # fmt: skip
    # Only farmed fields count, and the square must lie inside the county.
    not_farmed = farmed.copy()
    not_farmed[30:60] = False
    assert ex28.rule_square(x, y, not_farmed, county) == (a, 30, 30)
    notch = county.difference(box(x0 + 22000, y0 + 12000, x0 + 23000, y0 + 13000))
    assert ex28.rule_square(x, y, farmed, notch) == (a, 30, 30)
    # On the 1 km grid (R1's A1 rule) the southernmost square that holds all of a's fields,
    # 1 km or more from b.
    near = ex28.rule_square(x, y, farmed, county, ex28.NEAR_GRID_M, gap, ex28.NEAR_GAP_M)
    assert near == ((x0 + 10000, y0 + 6000), 30, 30)
    assert ex28.rule_square(x, y, np.zeros(len(x), dtype=bool), county) == (None, 0, 0)


def test_completeness(ex28):
    area = ex28.square((365000, 4745000))
    classes = np.full((500, 500), 141)  # forest
    classes[:, :250] = 1  # corn on the western half
    fields = gpd.GeoDataFrame(geometry=[box(365000, 4745000, 366250, 4750000)], crs=UTM)
    assert ex28.completeness(classes, fields, area) == pytest.approx(0.5)
    assert ex28.completeness(classes, fields.iloc[:0], area) == 0.0
    assert math.isnan(ex28.completeness(np.full((500, 500), 141), fields, area))
