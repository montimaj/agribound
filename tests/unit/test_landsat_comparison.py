"""Offline checks of prepared inputs and the example's actual run/export path."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from agribound.composites.gee import compute_export_grid
from agribound.composites.landsat_matched import validate_matched_inputs
from agribound.config import AgriboundConfig
from agribound.io.raster import write_raster


def load_example(number):
    """Load a runnable example without executing its main function."""
    root = Path(__file__).resolve().parents[2]
    path = next((root / "examples").glob(f"{number}_*.py"))
    spec = importlib.util.spec_from_file_location(f"example_{number}", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ((-2.88, 48.16, -2.82, 48.20), "EPSG:32630"),
        ((7.49, 48.35, 7.55, 48.39), "EPSG:32632"),
        ((1.62, 48.13, 1.68, 48.17), "EPSG:32631"),
        ((20.90, -34.20, 20.95, -34.16), "EPSG:32734"),
        ((106.38, 20.615, 106.405, 20.64), "EPSG:32648"),
    ],
)
def test_comparison_buffer_uses_the_local_utm_zone(bbox, expected):
    aoi = box(*bbox)
    buffered, crs = load_example("23").buffered_study_area(aoi)
    assert crs == expected
    assert buffered.contains(aoi)


def test_reference_profile_distinguishes_elongated_and_compact_parcels(tmp_path):
    example = load_example("24")
    path = tmp_path / "reference.gpkg"
    gpd.GeoDataFrame(
        geometry=[box(400000, 5330000, 400100, 5330100), box(400200, 5330000, 400210, 5330100)],
        crs=32631,
    ).to_file(path, driver="GPKG")
    profile = example.reference_profile(path)
    assert profile["n_reference"] == 2
    assert profile["median_elongation"] == pytest.approx(5.5)
    assert profile["median_area_ha"] == pytest.approx(0.55)


@pytest.fixture
def prepared(tmp_path):
    geometry = box(1.65, 48.15, 1.651, 48.151)
    manifest = {"crs": "EPSG:32631", "pairs": [{"scene_id": "test"}]}
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    grid = compute_export_grid(geometry, manifest["crs"], 30)
    paths = []
    for name, count, factor in (("sr", 6, 1), ("pan", 1, 2)):
        path = tmp_path / f"{name}.tif"
        data = np.ones((count, grid.height * factor, grid.width * factor), dtype=np.float32)
        write_raster(
            path, data, grid.crs, grid.transform * rasterio.Affine.scale(1 / factor), nodata=np.nan
        )
        with rasterio.open(path, "r+") as dst:
            dst.update_tags(AGRIBOUND_SCENE_MANIFEST_SHA256=manifest["sha256"])
        paths.append(path)
    return paths[1], paths[0], manifest, geometry


def test_prepared_inputs_require_same_manifest_grid_and_support(prepared):
    validate_matched_inputs(*prepared)
    pan, sr, manifest, geometry = prepared
    with rasterio.open(pan, "r+") as dst:
        data = dst.read()
        data[0, 0, 0] = np.nan
        dst.write(data)
    with pytest.raises(ValueError, match="valid support"):
        validate_matched_inputs(pan, sr, manifest, geometry)


def test_prepared_inputs_reject_changed_manifest_and_study_area(prepared):
    pan, sr, manifest, geometry = prepared
    with pytest.raises(ValueError, match="study area"):
        validate_matched_inputs(pan, sr, manifest, box(1.66, 48.15, 1.67, 48.16))
    manifest["pairs"].append({"scene_id": "changed"})
    with pytest.raises(ValueError, match="checksum"):
        validate_matched_inputs(pan, sr, manifest, geometry)


def test_example_exports_metrics_polygons_and_provenance_without_live_services(
    tmp_path, monkeypatch
):
    path = Path(__file__).resolve().parents[2] / "examples" / "23_landsat_pan_sr_comparison.py"
    spec = importlib.util.spec_from_file_location("landsat_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    reference = gpd.GeoDataFrame(geometry=[box(400000, 5330000, 400100, 5330100)], crs=32631)
    aoi = reference.to_crs(4326).geometry.iloc[0].buffer(0.001)
    cfg = AgriboundConfig(
        source="landsat-pan",
        gee_project="test",
        year=2023,
        study_area=aoi.wkt,
        lulc_filter=False,
        device="cpu",
    )
    raster = tmp_path / "pan.tif"
    write_raster(
        raster,
        np.ones((1, 32, 32), np.float32),
        "EPSG:32631",
        rasterio.Affine(15, 0, 399900, 0, -15, 5330300),
        nodata=np.nan,
    )

    class FakeEngine:
        def delineate(self, raster_path, config):
            assert config.source == "landsat-pan"
            result = reference.copy()
            result.attrs["engine_meta"] = {"checkpoint_sha256": "offline-test"}
            return result

    monkeypatch.setattr(example, "get_engine", lambda name: FakeEngine())
    predicted, rows = example.run_experiment(
        "pan", raster, cfg, reference, aoi, {"pairs": []}, tmp_path
    )
    assert len(predicted) == 1
    assert {row["boundary_tolerance_m"] for row in rows} == {10, 15, 30}
    assert all(row["f1"] == 1 for row in rows if row["size_class_ha"] == "all")
    assert len(gpd.read_file(tmp_path / "fields_pan.gpkg")) == 1
    provenance = json.loads((tmp_path / "fields_pan.gpkg.provenance.json").read_text())
    assert provenance["status"] == "success"
    assert provenance["engine_meta"]["checkpoint_sha256"] == "offline-test"
    assert provenance["facts"]["lulc_status"] == "disabled for all experiments"
