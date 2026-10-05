"""Offline tutorial integrity and end-to-end evaluation without engine extras."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "examples/data/landsat_tutorial"


@pytest.fixture(scope="module")
def tutorial():
    sys.path.insert(0, str(ROOT / "examples"))
    import landsat_tutorial_support

    return landsat_tutorial_support


def test_bundle_is_small_hash_pinned_and_contains_only_public_reference_geometry(tutorial):
    manifest = tutorial.load_bundle(BUNDLE)
    assert sum(r["bytes"] for r in manifest["files"]) < 4_000_000
    assert all("\\Users\\" not in json.dumps(r) for r in manifest["files"])
    assert manifest["selected_summary_sites"] == [
        "fr_camargue_2024",
        "fr_bordeaux_2024",
        "nl_flevoland_2025",
        "nl_bollenstreek_2025",
        "qc_yamaska_2024",
        "vn_mekong_adjacent_2024",
    ]


def test_tutorial_rejects_changed_reference_before_reusing_predictions(tutorial, tmp_path):
    shutil.copytree(BUNDLE, tmp_path / "bundle")
    (tmp_path / "bundle/reference.gpkg").write_bytes(b"different vintage")
    with pytest.raises(ValueError, match="reference.gpkg"):
        tutorial.load_bundle(tmp_path / "bundle")


def test_controls_reproduce_visible_fusion_and_preserve_common_support(tutorial, tmp_path):
    paths, diagnostic = tutorial.prepare_inputs(BUNDLE, tmp_path)
    assert len(paths) == 8
    integrity = diagnostic["integrity"]
    assert integrity["nir_max_abs_change_after_resampling"] == 0
    assert integrity["hybrid_visible_max_abs_change"] == 0
    assert integrity["coarse_pan_mean_max_abs_error"] == 0
    assert integrity["valid_fraction"] == 1
    assert diagnostic["visible_fusion"]["coarse_max_abs_error"] < 1e-7


def test_cached_reference_evaluation_reproduces_published_measurements(tutorial, tmp_path):
    paths = tutorial.run_landsat(BUNDLE, tmp_path, {}, live=False)
    original = {k: tutorial.sha256(p) for k, p in paths.items()}
    products = tutorial.load_products(BUNDLE, paths)
    result = tutorial.evaluate_products(BUNDLE, tmp_path, products)
    expected = pd.read_csv(BUNDLE / "reference_headline_15m.csv")
    expected = expected[(expected.site == "fr_camargue_2024") & expected.primary_scope]
    expected = expected.set_index("product")
    actual = result[(result.boundary_tolerance_m == 15) & (result.size_class_ha == "all")]
    for row in actual.itertuples():
        assert row.boundary_f1 == pytest.approx(expected.loc[row.product, "boundary_f1"])
        assert row.detection_f1 == pytest.approx(expected.loc[row.product, "f1"])
    assert original == {k: tutorial.sha256(p) for k, p in paths.items()}


def test_local_ftw_query_is_complete_and_agreement_is_separate(tutorial, tmp_path):
    ftw = tutorial.query_product(BUNDLE, tmp_path)
    raw = gpd.read_parquet(BUNDLE / "ftw_raw.parquet")
    expected = tutorial.inside_aoi(raw, tutorial.load_bundle(BUNDLE)["site"]["bbox"])
    assert len(ftw["ftw"]) == len(expected)
    assert set(ftw["ftw"].id) == set(expected.id)
    assert ftw["ftw"].to_crs(6933).area.sum() == pytest.approx(expected.to_crs(6933).area.sum())
    left = {
        "pan": tutorial.load_products(BUNDLE, {"pan": BUNDLE / "predictions/fields_pan.gpkg"})[
            "pan"
        ]
    }
    agreement = tutorial.agreement_products(tmp_path, left, ftw["ftw"])
    assert set(agreement.track) == {"prediction_agreement"}
    assert "accuracy" not in agreement.columns
    assert "boundary_f1" not in agreement.columns
    assert np.isfinite(agreement.correspondence_f1).all()
    assert not (tmp_path / "reference_accuracy.csv").exists()


def test_missing_live_snapshot_cannot_trigger_evaluation_download(tutorial, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("evaluation must not download")

    monkeypatch.setattr(tutorial, "query_ftw", fail)
    with pytest.raises(ValueError, match="stage query"):
        tutorial.query_product(BUNDLE, tmp_path, live=True, query_if_missing=False)


def test_live_resume_rejects_changed_input_without_repeating_inference(
    tutorial, tmp_path, monkeypatch
):
    from shapely.geometry import box

    import agribound.engines

    checkpoint = tmp_path / "mock_weights.pt"
    checkpoint.write_bytes(b"synthetic test-only weights; never run a model")
    real_hash = tutorial.sha256
    expected = tutorial.load_bundle(BUNDLE)["comparison"]["checkpoint_sha256"]
    monkeypatch.setattr(
        tutorial, "sha256", lambda p: expected if Path(p) == checkpoint else real_hash(p)
    )
    raster = tmp_path / "pan.tif"
    shutil.copyfile(BUNDLE / "inputs/landsat-pan.tif", raster)
    calls = []

    def fake_delineate(path, config):
        calls.append(path)
        return gpd.GeoDataFrame(geometry=[box(4.56, 43.56, 4.562, 43.562)], crs=4326).to_crs(32631)

    monkeypatch.setattr(
        agribound.engines, "get_engine", lambda _: SimpleNamespace(delineate=fake_delineate)
    )
    tutorial.run_landsat(
        BUNDLE, tmp_path, {"pan": raster}, live=True, checkpoint=checkpoint, project="test-project"
    )
    tutorial.run_landsat(
        BUNDLE, tmp_path, {"pan": raster}, live=True, checkpoint=checkpoint, project="test-project"
    )
    assert len(calls) == 1
    raster.write_bytes(raster.read_bytes() + b"changed input")
    with pytest.raises(ValueError, match="cache mismatch"):
        tutorial.run_landsat(
            BUNDLE,
            tmp_path,
            {"pan": raster},
            live=True,
            checkpoint=checkpoint,
            project="test-project",
        )
    assert len(calls) == 1


@pytest.fixture
def live_cache(tutorial, tmp_path, monkeypatch, request):
    """Create a real pinned cache with synthetic inference and actual input files."""
    from shapely.geometry import box

    import agribound.engines

    out = tmp_path
    if getattr(request, "param", None) == "relative":
        monkeypatch.chdir(tmp_path.parent)
        out = Path(tmp_path.name)
    checkpoint = out / "mock_weights.pt"
    checkpoint.write_bytes(b"synthetic test-only weights; never run a model")
    real_hash = tutorial.sha256
    expected = tutorial.load_bundle(BUNDLE)["comparison"]["checkpoint_sha256"]
    monkeypatch.setattr(
        tutorial, "sha256", lambda p: expected if Path(p) == checkpoint else real_hash(p)
    )
    paths, _ = tutorial.prepare_inputs(BUNDLE, out)
    inputs = out / "live_inputs"
    inputs.mkdir()
    for method in ("pan", "sr", "combined"):
        path = inputs / paths[method].name
        shutil.copyfile(paths[method], path)
        paths[method] = path

    def fake_delineate(path, config):
        return gpd.GeoDataFrame(geometry=[box(4.56, 43.56, 4.562, 43.562)], crs=4326).to_crs(32631)

    monkeypatch.setattr(
        agribound.engines, "get_engine", lambda _: SimpleNamespace(delineate=fake_delineate)
    )
    tutorial.run_landsat(
        BUNDLE, out, paths, live=True, checkpoint=checkpoint, project="test-project"
    )
    checkpoint.unlink()
    monkeypatch.setattr(tutorial, "sha256", real_hash)

    def fail(*args, **kwargs):
        raise AssertionError("cached replay must not download or infer")

    monkeypatch.setattr(agribound.engines, "get_engine", fail)
    monkeypatch.setattr(tutorial, "query_ftw", fail)
    return tmp_path


def test_stage_only_live_replay_validates_all_eight_without_weights_or_network(
    tutorial, live_cache
):
    paths = tutorial.cached_landsat_paths(BUNDLE, live_cache, live=True)
    assert set(paths) == set(tutorial.METHODS)
    assert all(path.is_file() for path in paths.values())
    assert len(tutorial.load_products(BUNDLE, paths)) == 8


@pytest.mark.parametrize("live_cache", ["relative"], indirect=True)
def test_relative_live_cache_replays_from_another_working_directory(
    tutorial, live_cache, monkeypatch
):
    record = json.loads((live_cache / "fields_pan_raw.resume.json").read_text(encoding="utf-8"))
    config = record["pin"]["config"]
    for value in (
        config["local_tif_path"],
        config["output_path"],
        config["cache_dir"],
        config["engine_params"]["checkpoint_path"],
    ):
        assert Path(value).is_absolute()
    monkeypatch.chdir(BUNDLE)
    paths = tutorial.cached_landsat_paths(BUNDLE, live_cache, live=True)
    assert set(paths) == set(tutorial.METHODS)
    assert len(tutorial.load_products(BUNDLE, paths)) == 8


@pytest.mark.parametrize("changed", ["output", "input", "config", "code", "missing_record"])
def test_stage_only_live_replay_rejects_changed_artifacts_and_pins(tutorial, live_cache, changed):
    target = live_cache / "fields_pan_raw.gpkg"
    record = target.with_suffix(".resume.json")
    if changed == "missing_record":
        record.unlink()
    elif changed in ("output", "input"):
        path = target if changed == "output" else live_cache / "live_inputs/landsat-pan.tif"
        path.write_bytes(path.read_bytes() + b"changed artifact")
    else:
        old = json.loads(record.read_text(encoding="utf-8"))
        if changed == "config":
            old["pin"]["config"]["engine_params"]["conf_threshold"] = 0.9
        else:
            old["pin"]["scientific_code_sha256"]["evaluate.py"] = "changed implementation"
        tutorial.save_json(record, old)
    with pytest.raises(ValueError, match="Live cache mismatch for pan"):
        tutorial.cached_landsat_paths(BUNDLE, live_cache, live=True)


def test_imports_do_not_require_engine_gee_or_plotting():
    code = """
import sys
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','ultralytics','cv2','ee','matplotlib'}:
            raise ImportError('optional dependency blocked: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
sys.path.insert(0, 'examples')
import agribound
import landsat_tutorial_support
from agribound.composites.landsat_multispectral import build_inputs
from agribound.comparison_ftw import ftw_variant
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("number", [29, 30])
def test_notebook_style_import_does_not_download_or_infer(number, monkeypatch):
    script = next((ROOT / "examples").glob(f"{number}_*.py"))
    spec = importlib.util.spec_from_file_location(f"tutorial_{number}", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec.loader.exec_module(module)
    assert module.NOTEBOOK_ARGS == []
    assert module.BUNDLE == BUNDLE


@pytest.mark.parametrize("number", [29, 30])
def test_evaluation_only_runs_from_notebook_directory_without_weights(number, tmp_path, tutorial):
    script = next((ROOT / "examples").glob(f"{number}_*.py"))
    result = subprocess.run(
        [sys.executable, str(script), "--stage", "evaluate", "--output-dir", str(tmp_path)],
        cwd=ROOT / "examples/notebooks",
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "reference_accuracy.csv").exists()
    assert (tmp_path / "prediction_agreement.csv").exists() == (number == 30)
    accuracy = pd.read_csv(tmp_path / "reference_accuracy.csv")
    assert set(accuracy[accuracy["product"].isin(tutorial.METHODS)]["product"]) == set(
        tutorial.METHODS
    )
    if number == 30:
        agreement = pd.read_csv(tmp_path / "prediction_agreement.csv")
        assert set(agreement["method"]) == set(tutorial.METHODS)
