"""Offline controls for new Landsat inputs and model channel semantics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio import Affine
from rasterio.io import MemoryFile
from shapely.geometry import box

from agribound.composites.landsat_multispectral import (
    build_inputs,
    hybrid_reduced_validation,
    repeat_2x,
    validate_grids,
)
from agribound.composites.pan_fusion import block_mean, inject_pan_detail
from agribound.engines.delineate_anything import _read_tile, stretch_with_bounds
from agribound.evaluate import evaluate


@pytest.fixture
def arrays():
    level = 0.1 + np.arange(16).reshape(4, 4) / 100
    sr = np.stack([level * factor for factor in (0.5, 0.8, 1.0, 3.0, 1.3, 1.2)]).astype("float32")
    pan = repeat_2x(level + 0.05) + np.tile([[0.01, -0.01], [-0.01, 0.01]], (4, 4))
    fused, _ = inject_pan_detail(sr[[2, 1, 0]], pan)
    return sr, pan.astype("float32"), fused


def test_native_pan_control_isolates_detail_and_preserves_other_channels(arrays):
    sr, pan, fused = arrays
    outputs, diagnostic = build_inputs(sr, pan, fused)
    native, coarse = outputs["pan_nir_red"], outputs["coarse_pan_nir_red"]
    np.testing.assert_array_equal(native[1:], coarse[1:])
    np.testing.assert_array_equal(native[0], pan)
    np.testing.assert_allclose(block_mean(native[0]), block_mean(coarse[0]), atol=1e-8)
    assert np.ptp(native[0, :2, :2]) > 0
    assert np.ptp(coarse[0, :2, :2]) == 0
    assert diagnostic["coarse_pan_mean_max_abs_error"] == 0


def test_false_color_and_hybrid_preserve_nir_and_existing_visible_fusion(arrays):
    sr, pan, fused = arrays
    out, diagnostic = build_inputs(sr, pan, fused)
    np.testing.assert_array_equal(out["false_color"], sr[[3, 2, 1]])
    np.testing.assert_array_equal(out["false_color_15m"], repeat_2x(sr[[3, 2, 1]]))
    np.testing.assert_array_equal(out["hybrid"][0], repeat_2x(sr[3]))
    np.testing.assert_array_equal(out["hybrid"][1:], fused[:2])
    assert diagnostic["nir_max_abs_change_after_resampling"] == 0
    assert diagnostic["hybrid_visible_max_abs_change"] == 0


@pytest.mark.parametrize("missing", ["pan", "nir", "swir", "fused"])
def test_incomplete_support_masks_whole_cell_in_every_method(arrays, missing):
    sr, pan, fused = [a.copy() for a in arrays]
    if missing == "pan":
        pan[0, 0] = np.nan
    elif missing == "nir":
        sr[3, 0, 0] = np.nan
    elif missing == "swir":
        sr[5, 0, 0] = np.nan
    else:
        fused[0, 0, 0] = np.nan
    out, diag = build_inputs(sr, pan, fused)
    assert diag["valid_30m_cells"] == 15
    assert np.isnan(out["false_color"][:, 0, 0]).all()
    for name, data in out.items():
        if name != "false_color":
            assert np.isnan(data[:, :2, :2]).all()
            assert np.isfinite(data[:, 2:, 2:]).all()


def test_no_support_and_wrong_shapes_fail(arrays):
    sr, pan, fused = arrays
    with pytest.raises(ValueError, match="complete common"):
        build_inputs(sr, pan * np.nan, fused)
    with pytest.raises(ValueError, match="six|6,H,W"):
        build_inputs(sr[:3], pan, fused)
    with pytest.raises(ValueError, match="aligned"):
        build_inputs(sr, pan[:-1], fused)


@pytest.mark.parametrize("problem", ["crs", "shift", "extent", "orientation", "resolution"])
def test_grid_mismatch_is_rejected(problem):
    from types import SimpleNamespace

    crs = rasterio.crs.CRS.from_epsg(32631)
    sr = SimpleNamespace(
        crs=crs,
        transform=Affine(30, 0, 500000, 0, -30, 5500000),
        height=4,
        width=4,
        res=(30, 30),
        count=6,
    )
    pan = SimpleNamespace(
        crs=crs,
        transform=sr.transform * Affine.scale(0.5),
        height=8,
        width=8,
        res=(15, 15),
        count=1,
    )
    if problem == "crs":
        pan.crs = rasterio.crs.CRS.from_epsg(32632)
    elif problem == "shift":
        pan.transform *= Affine.translation(1, 0)
    elif problem == "extent":
        pan.width = 6
    elif problem == "orientation":
        sr.transform = Affine(30, 0, 500000, 0, 30, 5500000)
        pan.transform = sr.transform * Affine.scale(0.5)
    else:
        sr.res = (20, 20)
    with pytest.raises(ValueError):
        validate_grids(sr, pan)


def test_reduced_hybrid_validation_is_visible_only_and_retains_baseline(arrays):
    sr, pan, _ = arrays
    diagnostic = hybrid_reduced_validation(sr[[2, 1, 0]], pan)
    assert len(diagnostic["fused_visible"]["red_green_rmse"]) == 2
    assert len(diagnostic["resampled_visible"]["red_green_rmse"]) == 2
    assert "no NIR sharpening" in diagnostic["nir"]


def test_engine_bgr_read_then_ultralytics_rgb_conversion_restores_requested_order(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path))
    (tmp_path / "Ultralytics").mkdir()
    pytest.importorskip("ultralytics")
    logical_rgb = np.stack([np.full((2, 2), v, "float32") for v in (40, 90, 180)])
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            width=2,
            height=2,
            count=3,
            dtype="float32",
            crs="EPSG:32631",
            transform=Affine(15, 0, 500000, 0, -15, 5500000),
        ) as dst:
            dst.write(logical_rgb)
        with mem.open() as src:
            data, valid = _read_tile(src, [3, 2, 1], 0, 0, 2)
            stretched = stretch_with_bounds(data, [0] * 3, [255] * 3, valid)
            bgr_hwc = stretched.transpose(1, 2, 0)
            # Actual Ultralytics preprocessing, with no loaded model or weights.
            from ultralytics.models.yolo.segment.predict import SegmentationPredictor

            predictor = SegmentationPredictor(overrides={"verbose": False})
            from types import SimpleNamespace

            import torch

            predictor.device = torch.device("cpu")
            predictor.model = SimpleNamespace(fp16=False)
            predictor.pre_transform = lambda images: images
            tensor = predictor.preprocess([bgr_hwc])
            np.testing.assert_allclose(tensor[0].numpy() * 255, logical_rgb, atol=1e-5)


def test_coverage_metrics_keep_whole_polygons_without_creating_clipping_edges():
    ref = gpd.GeoDataFrame(geometry=[box(500000, 5500000, 500060, 5500060)], crs=32631)
    pred = gpd.GeoDataFrame(
        geometry=[box(500000, 5500000, 500060, 5500060), box(500120, 5500000, 500180, 5500060)],
        crs=32631,
    )
    spec = importlib.util.spec_from_file_location(
        "ms_adapters", Path(__file__).parents[2] / "examples/landsat_reference_adapters.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    selected, info = module.select_coverage(pred, ref)
    assert len(selected) == 1 and info["n_predictions_unknown_coverage"] == 1
    assert selected.geometry.iloc[0].equals(pred.geometry.iloc[0])
    scores = evaluate(selected, ref, boundary_tolerance_m=15, boundary_mask=ref)
    assert scores["f1"] == 1
    assert scores["boundary_f1"] == pytest.approx(1)
    empty = selected.iloc[:0]
    scores = evaluate(empty, ref, boundary_tolerance_m=15, boundary_mask=ref)
    assert scores["count_predicted"] == 0 and scores["recall"] == 0
