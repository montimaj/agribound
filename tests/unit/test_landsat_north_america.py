"""Offline adapter checks: pagination, source identity, vintage, CRS and coverage."""

import importlib.util
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box, mapping


@pytest.fixture
def adapters():
    path = Path(__file__).resolve().parents[2] / "examples" / "landsat_reference_adapters.py"
    spec = importlib.util.spec_from_file_location("reference_adapters", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def service(truncate=False):
    calls = []

    def request(url, params):
        calls.append(params)
        if not url.endswith("/query"):
            return {"objectIdField": "OBJECTID", "maxRecordCount": 2}
        if params.get("returnIdsOnly"):
            return {"objectIds": [5, 2, 3]}
        ids = list(reversed([int(i) for i in params["objectIds"].split(",")]))
        if truncate:
            ids = ids[:-1]
        return {
            "features": [
                {
                    "type": "Feature",
                    "properties": {"OBJECTID": i},
                    "geometry": mapping(box(i, 0, i + 1, 1)),
                }
                for i in ids
            ]
        }

    return request, calls


def test_arcgis_fetches_exact_ids_in_stable_order_and_wgs84(adapters):
    request, calls = service()
    data, manifest = adapters.arcgis_snapshot(
        "https://example.invalid/0", [0, 0, 10, 10], request=request
    )
    assert [f["properties"]["OBJECTID"] for f in data["features"]] == [2, 3, 5]
    assert manifest["object_ids"] == [2, 3, 5]
    assert [p["params"]["objectIds"] for p in manifest["pages"]] == ["2,3", "5"]
    assert all(p["params"]["outSR"] == 4326 for p in manifest["pages"])
    assert "resultOffset" not in json.dumps(calls)


def test_truncated_service_page_is_rejected(adapters):
    request, _ = service(truncate=True)
    with pytest.raises(ValueError, match="truncated"):
        adapters.arcgis_snapshot("https://example.invalid/0", [0, 0, 10, 10], request=request)


def test_duplicate_service_ids_are_rejected(adapters):
    def request(url, params):
        if url.endswith("/query"):
            return {"objectIds": [1, 1]}
        return {"objectIdField": "OBJECTID"}

    with pytest.raises(ValueError, match="Duplicate IDs"):
        adapters.arcgis_snapshot("https://example.invalid/0", [0, 0, 1, 1], request=request)


def test_active_declarations_and_survey_years(adapters):
    frame = pd.DataFrame(
        {
            "NBPRO": [0, 1, 2],
            "AN": [2022, 2023, 2023],
            "LastSurveyDate": [None, "2023-06-01", "2022-07-01"],
        }
    )
    rules = [
        {"column": "NBPRO", "op": "ge", "value": 1},
        {"column": "LastSurveyDate", "op": "year", "value": 2023},
    ]
    assert adapters.filter_reference(frame, rules, year=2023).index.tolist() == [1]
    with pytest.raises(ValueError, match="Declaration year"):
        adapters.filter_reference(frame, [], year=2023)
    with pytest.raises(ValueError, match="filter column"):
        adapters.filter_reference(frame, [{"column": "missing", "op": "in", "value": [1]}])


def test_coverage_preserves_whole_polygons_and_excludes_unknown_territory(adapters):
    polygons = [box(500000, 4500000, 500100, 4500100), box(500500, 4500000, 500600, 4500100)]
    predicted = gpd.GeoDataFrame({"name": ["known", "unknown"]}, geometry=polygons, crs=32612)
    coverage = gpd.GeoDataFrame(geometry=[box(500040, 4500040, 500060, 4500060)], crs=32612).to_crs(
        4326
    )
    selected, counts = adapters.select_coverage(predicted, coverage)
    assert selected.name.tolist() == ["known"]
    assert selected.geometry.iloc[0].equals(polygons[0])
    assert counts["n_predictions_unknown_coverage"] == 1


def test_safe_archive_extraction_rejects_escape_before_any_extraction(adapters, tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("fine.txt", "fine")
        stream.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="escapes"):
        adapters.safe_extract(archive, tmp_path / "data")
    assert not (tmp_path / "data" / "fine.txt").exists()


def test_spatial_subset_transforms_bbox_and_keeps_accents(adapters, tmp_path):
    frame = gpd.GeoDataFrame(
        {"crop": ["Maïs", "Soya"]},
        geometry=[box(-73, 45, -72.99, 45.01), box(-70, 45, -69.99, 45.01)],
        crs=4326,
    ).to_crs(32198)
    path = tmp_path / "reference.gpkg"
    frame.to_file(path, driver="GPKG")
    result = adapters.read_reference(path, {"adapter": "local"}, [-73.01, 44.99, -72.98, 45.02])
    assert result.crop.tolist() == ["Maïs"]
    assert result.crs.to_epsg() == 32198


def test_cached_sources_require_matching_checksum(adapters, tmp_path):
    path = tmp_path / "reference.geojson"
    path.write_text("pinned", encoding="utf-8")
    source = {"filename": path.name, "sha256": adapters.file_sha256(path)}
    assert adapters.fetch_source(source, tmp_path, offline=True) == path
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        adapters.fetch_source(source, tmp_path, offline=True)


def test_aggregation_keeps_reference_types_and_scopes_separate():
    path = (
        Path(__file__).resolve().parents[2] / "examples" / "26_landsat_north_america_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("north_america", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    common = {
        "size_class_ha": "all",
        "cohort": "north_america",
        "experiment": "pan",
        "boundary_tolerance_m": 15,
    }
    metrics = {
        m: 0.2
        for m in [
            "f1",
            "matched_iou",
            "boundary_f1",
            "precision",
            "recall",
            "boundary_precision",
            "boundary_recall",
            "oversegmentation",
            "undersegmentation",
            "inference_s",
        ]
    }
    table = pd.DataFrame(
        [
            {
                **common,
                **metrics,
                "reference_kind": "mapped_crop_unit",
                "evaluation_scope": "known_footprints",
                "n_reference": 10,
            },
            {
                **common,
                **metrics,
                "f1": 0.8,
                "reference_kind": "mapped_crop_unit",
                "evaluation_scope": "known_footprints",
                "n_reference": 30,
            },
            {
                **common,
                **metrics,
                "reference_kind": "declaration_parcel",
                "evaluation_scope": "known_footprints",
                "n_reference": 100,
            },
            {
                **common,
                **metrics,
                "reference_kind": "mapped_crop_unit",
                "evaluation_scope": "aoi",
                "n_reference": 10,
            },
        ]
    )
    summary = module.aggregate_metrics(table)
    selected = summary[
        (summary.reference_kind == "mapped_crop_unit")
        & (summary.evaluation_scope == "known_footprints")
    ]
    assert selected[selected.weighting == "equal_site"].f1.iloc[0] == pytest.approx(0.5)
    assert selected[selected.weighting == "reference_count"].f1.iloc[0] == pytest.approx(0.65)
    assert len(summary) == 6


def test_valid_imagery_coverage_excludes_nodata_and_reports_export_extent(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    path = (
        Path(__file__).resolve().parents[2] / "examples" / "26_landsat_north_america_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("coverage_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raster_path = tmp_path / "sr.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=6,
        dtype="float32",
        crs=32612,
        transform=from_origin(500000, 4500060, 30, 30),
        nodata=-9999,
    ) as raster:
        data = np.ones((6, 2, 2), dtype="float32")
        data[4, 0, 0] = -9999  # One missing non-RGB band must invalidate the cell.
        raster.write(data)
    whole = gpd.GeoDataFrame(geometry=[box(500000, 4500000, 500120, 4500060)], crs=32612)
    one_cell = gpd.GeoDataFrame(geometry=[box(500000, 4500030, 500030, 4500060)], crs=32612)
    result = module.imagery_coverage(raster_path, {"aoi": whole.to_crs(4326), "known": one_cell})
    assert result["aoi"]["valid_fraction"] == pytest.approx(0.75)
    assert result["known"]["valid_fraction"] == 0
    assert result["aoi"]["geometry_area_within_export_fraction"] == pytest.approx(0.5, abs=1e-5)


def test_annotation_template_never_certifies_unreviewed_coverage(tmp_path):
    path = Path(__file__).resolve().parents[2] / "examples" / "prepare_manual_field_reference.py"
    spec = importlib.util.spec_from_file_location("manual_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "manual.gpkg"
    options = {
        "reference_year": 2022,
        "imagery_source": "USDA NAIP dated item",
        "imagery_date": "2022-07-09",
        "imagery_license": "Public domain",
        "positional_accuracy": "Unknown for this item",
    }
    module.prepare_template(output, [-122.04, 39.08, -121.99, 39.12], **options)
    assert gpd.read_file(output, layer="reviewed_coverage").empty
    assert gpd.read_file(output, layer="fields").empty
    area = gpd.read_file(output, layer="evaluation_area")
    assert area.crs.to_epsg() == 32610
    meta = json.loads(output.with_suffix(".json").read_text())
    assert meta["status"] == "unannotated_unreviewed"
    with pytest.raises(ValueError, match="Preserve existing"):
        module.prepare_template(output, [-122.04, 39.08, -121.99, 39.12], **options)
