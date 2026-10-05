"""Offline product/reference separation, local FTW retrieval and cache controls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pytest
from shapely.geometry import box

from agribound.comparison_ftw import (
    ftw_variant,
    inside_aoi,
    product_agreement,
    select_product_coverage,
    validate_ftw_snapshot,
)
from agribound.ftw_query import query_ftw


@pytest.fixture
def fields():
    return gpd.GeoDataFrame(
        {
            "id": ["high", "low", "null", "edge"],
            "confidence": [80, 20, None, 69],
            "year": [2024] * 4,
        },
        geometry=[
            box(0, 0, 100, 100),
            box(120, 0, 220, 100),
            box(240, 0, 250, 10),
            box(300, 0, 400, 100),
        ],
        crs=32631,
    )


def test_confidence_null_is_unknown_not_low(fields):
    retained, diag = ftw_variant(fields, "ftw_conf69")
    assert list(retained.id) == ["high", "null", "edge"]
    assert diag["n_null_confidence_retained"] == 1
    strict, diag = ftw_variant(fields, "ftw_conf69_known")
    assert list(strict.id) == ["high", "edge"]
    assert diag["n_excluded"] == 2
    assert all(
        a.equals(b)
        for a, b in zip(retained.geometry, fields.loc[retained.index].geometry, strict=True)
    )


def test_area_filter_uses_metric_area_even_in_geographic_crs(fields):
    geographic = fields.to_crs(4326)
    retained, _ = ftw_variant(geographic, "ftw_area1000")
    assert set(retained.id) == {"high", "low", "edge"}
    assert retained.crs == geographic.crs
    with pytest.raises(ValueError, match="Unknown"):
        ftw_variant(fields, "optimized_threshold")


@pytest.mark.parametrize(
    "year,info,error",
    [
        (2023, {"n_files_opened": 1}, "Unsupported"),
        (2024, {"n_files_opened": 0}, "coverage"),
        (2024, {"n_files_opened": 1, "max_features": 10}, "Truncated"),
    ],
)
def test_unavailable_or_incomplete_is_not_empty(fields, year, info, error):
    with pytest.raises(ValueError, match=error):
        validate_ftw_snapshot(fields.iloc[:0], year, info)


def test_supported_complete_empty_and_wrong_year(fields):
    validate_ftw_snapshot(fields.iloc[:0], 2024, {"n_files_opened": 1})
    validate_ftw_snapshot(fields, 2024, {"n_files_opened": 1})
    with pytest.raises(ValueError, match="years"):
        validate_ftw_snapshot(fields, 2025, {"n_files_opened": 1})


def test_whole_polygon_aoi_policy_no_clipping():
    frame = gpd.GeoDataFrame(
        {"id": [1, 2]}, geometry=[box(-0.2, 0.2, 0.6, 0.8), box(0.9, 0.2, 1.5, 0.8)], crs=4326
    )
    selected = inside_aoi(frame, [0, 0, 1, 1])
    assert list(selected.id) == [1]
    assert selected.geometry.iloc[0].bounds[0] == -0.2
    with pytest.raises(ValueError, match="CRS"):
        inside_aoi(frame.set_crs(None, allow_override=True), [0, 0, 1, 1])


def test_agreement_is_separately_named_and_directional(fields):
    result = product_agreement(
        fields.iloc[:1], fields.iloc[:2], tolerance_m=15, size_bins=[0, 1, 5, 100]
    )
    assert result["track"] == "prediction_agreement"
    assert result["left_correspondence_fraction"] == 1
    assert result["ftw_correspondence_fraction"] == 0.5
    assert result["n_corresponding"] == 1
    assert result["count_difference_left_minus_ftw"] == -1
    assert not {"accuracy", "precision", "recall", "boundary_f1"}.intersection(result)
    empty = product_agreement(fields.iloc[:0], fields, tolerance_m=15, size_bins=[0, 1, 5, 100])
    assert empty["n_left"] == 0 and empty["n_corresponding"] == 0


def test_local_tiles_complete_polygons_duplicate_year_and_confidence(tmp_path):
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    a = gpd.GeoDataFrame(
        {
            "id": ["duplicate", "old", "null", "low"],
            "time": ["2024-01-01", "2025-01-01", "2024-01-01", "2024-01-01"],
            "label": ["field"] * 4,
            "confidence": [80, 90, None, 20],
        },
        geometry=[
            box(0.1, 0.1, 1.2, 0.8),
            box(0.2, 0.2, 0.4, 0.4),
            box(0.3, 0.3, 0.5, 0.5),
            box(0.6, 0.3, 0.8, 0.5),
        ],
        crs=4326,
    )
    a.to_parquet(tile_dir / "a.parquet")
    a.iloc[[0]].to_parquet(tile_dir / "b.parquet")
    manifest = gpd.GeoDataFrame(
        {"tile_id": ["a", "b"], "out_path": ["a.parquet", "b.parquet"], "status": ["ok"] * 2},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs=4326,
    )
    manifest.to_parquet(tmp_path / "manifest.parquet")
    result = query_ftw(
        study_area=[0, 0, 1.1, 1],
        year=2024,
        clip=False,
        deduplicate=True,
        manifest_path=tmp_path / "manifest.parquet",
        tile_dir=tile_dir,
        min_confidence=69,
        keep_null_confidence=True,
    )
    assert set(result.id) == {"duplicate", "null"}
    assert result[result.id == "duplicate"].geometry.iloc[0].bounds[2] == 1.2
    validate_ftw_snapshot(result, 2024, result.attrs["ftw_query"])


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[2] / "examples/28_landsat_ftw_comparison.py"
    spec = importlib.util.spec_from_file_location("ftw_comparison_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_snapshot_cache_rejects_changes_and_never_refreshes(runner, tmp_path, fields):
    site = {"id": "demo", "ftw_year": 2024}
    config = {"ftw": {}}
    folder = tmp_path / "demo"
    folder.mkdir()
    target = folder / "ftw_raw.parquet"
    fields.to_parquet(target)
    pin = {
        "signature": runner.ftw_query_signature(site, config),
        "sha256": runner.file_sha256(target),
        "query": {"n_files_opened": 1},
    }
    runner.save_json(folder / "ftw_snapshot.json", pin)
    assert runner.retrieve_ftw(site, config, tmp_path, SimpleNamespace(offline=True)) == pin
    fields.iloc[:1].to_parquet(target)
    with pytest.raises(ValueError, match="checksum"):
        runner.retrieve_ftw(site, config, tmp_path, SimpleNamespace(offline=True))


def test_unknown_reference_coverage_excludes_unmapped_predictions(runner, fields):
    selected, diagnostic = runner.ADAPTERS.select_coverage(fields, fields.iloc[:0])
    assert selected.empty
    assert diagnostic["n_predictions_unknown_coverage"] == len(fields)


def test_coverage_evaluation_does_not_create_cut_edges(runner):
    pred = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=32631)
    ref = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=32631)
    mask = gpd.GeoDataFrame(geometry=[box(20, 20, 80, 80)], crs=32631)
    result = runner.evaluate(pred, ref, boundary_mask=mask, boundary_tolerance_m=10, bootstrap=0)
    assert result["count_reference"] == 1
    # The established evaluator returns zero for no supported line length.
    # Clipped rectangle edges would instead spuriously produce perfect F1.
    assert result["boundary_f1"] == 0


def test_frozen_sites_retain_historical_bounds_and_six_new_areas(runner):
    config = runner.read_json(runner.ROOT / "examples/landsat_ftw_comparison_sites.json")
    original = runner.read_json(runner.ROOT / config["inherited_config"]["path"])
    old = {s["id"]: s for s in original["sites"]}
    historical = [s for s in config["sites"] if s["historical"]]
    assert len(historical) == len(old)
    for site in historical:
        assert site["bbox"] == old[site["id"]]["bbox"]
        assert site["date_start"] == old[site["id"]]["date_start"]
        assert site["date_end"] == old[site["id"]]["date_end"]
    new = [s for s in config["sites"] if not s["historical"]]
    assert len(new) >= 6 and len({s["country"] for s in new}) >= 4


def test_mask_projection_repair_preserves_source_geometry(runner):
    from shapely.geometry import Polygon

    mask = gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (100, 100), (0, 100), (100, 0), (0, 0)])], crs=32631
    )
    fields = gpd.GeoDataFrame(geometry=[box(20, 0, 40, 20)], crs=32631)
    original = mask.geometry.iloc[0].wkb
    selected, diagnostic = select_product_coverage(fields, mask, runner.ADAPTERS.select_coverage)
    assert len(selected) == 1
    assert diagnostic["coverage_repaired_after_projection"] == 1
    assert mask.geometry.iloc[0].wkb == original and not mask.is_valid.all()


def test_dutch_cached_year_categories_and_projected_crs(runner, tmp_path):
    source = {"crs": 28992}
    site = {"reference_year": 2025}
    records = gpd.GeoDataFrame(
        {
            "jaar": [2025] * 2,
            "category": ["Bouwland", "Landschapselement"],
            "gewas": ["Tarwe", "Sloot"],
        },
        geometry=[box(180000, 500000, 180100, 500100), box(180200, 500000, 180210, 500010)],
        crs=28992,
    )
    path = tmp_path / "provider_snapshot.geojson"
    path.write_text(json.dumps(records.__geo_interface__), encoding="utf-8")
    runner.save_json(path.with_suffix(".pin.json"), {"sha256": runner.file_sha256(path)})
    frame, metadata = runner.dutch_reference(source, site, tmp_path, offline=True)
    assert list(frame.gewas) == ["Tarwe"] and frame.crs.to_epsg() == 28992
    centre = frame.to_crs(4326).geometry.iloc[0].centroid
    assert 4 < centre.x < 7 and 51 < centre.y < 54
    assert metadata["crop_column"] == "gewas"
    with pytest.raises(ValueError, match="year"):
        runner.dutch_reference(source, {"reference_year": 2024}, tmp_path, offline=True)


def test_inference_reuse_checks_settings_before_copy(runner, tmp_path):
    site = {"id": "demo", "historical": False}
    method = {"id": "pan", "channels": ["PAN"] * 3, "resolution_m": 15}
    folder = tmp_path / "landsat_runs/demo"
    folder.mkdir(parents=True)
    path = folder / "fields_pan.gpkg"
    path.write_bytes(b"unchanged prediction")
    runner.save_json(folder / "run_status.json", {"methods": {"pan": {"status": "complete"}}})
    runner.save_json(
        path.with_suffix(".gpkg.provenance.json"),
        {
            "multispectral_comparison": {
                "output_sha256": runner.file_sha256(path),
                "logical_channels": method["channels"],
            },
            "engine_meta": {"conf_threshold": 0.99},
            "config": {},
        },
    )
    with pytest.raises(ValueError, match="settings"):
        runner.copy_landsat(site, method, {"comparison": {}}, tmp_path)
    assert not (tmp_path / "demo/fields_pan.gpkg").exists()


def test_reference_free_agreement_never_writes_accuracy(runner, tmp_path, fields):
    folder = tmp_path / "demo"
    folder.mkdir()
    runner.write_gpkg(fields, folder / "fields_ftw.gpkg")
    runner.write_gpkg(fields.iloc[:1], folder / "fields_pan.gpkg")
    runner.save_json(
        folder / "product_status.json",
        {"products": {"ftw": {"status": "complete"}, "pan": {"status": "complete"}}},
    )
    site = {
        "id": "demo",
        "country": "synthetic",
        "ftw_year": 2024,
        "date_start": "2024-01-01",
        "temporal_category": "same_year",
        "bbox": list(fields.to_crs(4326).total_bounds),
    }
    config = {"methods": [{"id": "pan"}], "comparison": {"boundary_tolerances_m": [10, 15, 30]}}
    runner.evaluate_reference_free(site, config, tmp_path)
    assert (folder / "prediction_agreement.csv").exists()
    assert not (folder / "reference_accuracy.csv").exists()
    assert runner.read_json(folder / "reference_evaluation_status.json")["status"] == "unmeasured"


def test_empty_confidence_variant_has_measured_coverage_selection(runner, fields):
    selected, diagnostic = select_product_coverage(
        fields.iloc[:0], fields, runner.ADAPTERS.select_coverage
    )
    assert selected.empty and diagnostic["n_predictions_evaluated"] == 0
    assert diagnostic["n_predictions_unknown_coverage"] == 0


def test_summary_uses_product_column_and_hides_failed_tables(runner, tmp_path):
    config = runner.read_json(runner.ROOT / "examples/landsat_ftw_comparison_sites.json")
    config = {**config, "sites": config["sites"][:1]}
    site = config["sites"][0]
    folder = tmp_path / site["id"]
    folder.mkdir()
    products = {"pan": {"status": "complete"}, "ftw": {"status": "complete"}}
    runner.save_json(folder / "product_status.json", {"products": products})
    import pandas as pd

    rows = [
        {
            "site": site["id"],
            "product": name,
            "size_class_ha": "all",
            "crop_stratum": "all_crops",
            "boundary_tolerance_m": 15,
            "primary_scope": True,
            "temporal_category": "historical_cross_year",
            "reference_kind": "declaration",
            "training_overlap": "unknown",
            "n_reference": 10,
            **{metric: score for metric in runner.METRICS},
        }
        for name, score in (("pan", 0.4), ("ftw", 0.5))
    ]
    pd.DataFrame(rows).to_csv(folder / "reference_accuracy.csv", index=False)
    runner.summarize(config, tmp_path)
    pairs = pd.read_csv(tmp_path / "reference_paired_differences.csv")
    assert pairs.iloc[0].boundary_f1 == pytest.approx(-0.1)


def test_benchmark_split_footprints_do_not_establish_independence(runner):
    module = runner.helper("landsat_ftw_training_audit.py")
    ref = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100), box(200, 0, 300, 100)], crs=32631)
    chips = gpd.GeoDataFrame(
        {"split": ["train", "val", "test"]},
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100), box(200, 0, 300, 100)],
        crs=32631,
    )
    bbox = tuple(ref.to_crs(4326).total_bounds)
    result = module.chip_overlap(ref, chips, bbox)
    assert result["n_reference_intersecting_train_footprints"] == 1
    assert result["n_reference_intersecting_test_footprints"] == 1
    assert "membership" in result["interpretation"]
    assert (
        module.spatial_category({**result, "status": "complete"})
        == "overlap_with_published_v1_training_footprints"
    )
    assert (
        module.spatial_category({"status": "not_in_published_v1_country_inventory"})
        == "not_in_published_v1_country_inventory"
    )


def test_edge_inventory_distinguishes_retained_crossings_from_exclusions(runner):
    report = runner.helper("landsat_ftw_report.py")
    frame = gpd.GeoDataFrame(
        geometry=[
            box(0.1, 0.1, 0.3, 0.3),
            box(0.8, 0.2, 1.1, 0.4),
            box(0.9, 0.5, 1.4, 0.7),
            box(2, 2, 3, 3),
        ],
        crs=4326,
    )
    before = frame.geometry.to_wkb().tolist()
    result = report.aoi_edge_counts(frame.to_crs(6933), [0, 0, 1, 1])
    assert result["n_selected_by_representative_point"] == 2
    assert result["n_selected_crossing_aoi_edge"] == 1
    assert result["n_intersecting_excluded_by_policy"] == 1
    assert result["n_outside_aoi"] == 1
    assert frame.geometry.to_wkb().tolist() == before
    assert all(
        value == 0 for value in report.aoi_edge_counts(frame.iloc[:0], [0, 0, 1, 1]).values()
    )


def test_final_aoi_policy_applies_after_postprocessing_and_preserves_raw(runner, tmp_path):
    delivered = gpd.GeoDataFrame(
        geometry=[box(0.8, 0.2, 1.1, 0.4), box(0.9, 0.5, 1.4, 0.7)], crs=4326
    )
    raw = tmp_path / "fields_pan.gpkg"
    runner.write_gpkg(delivered, raw)
    original_hash = runner.file_sha256(raw)
    selected = runner.select_final_aoi(delivered, {"bbox": [0, 0, 1, 1]}, tmp_path, "pan")
    assert len(selected) == 1
    assert selected.geometry.iloc[0].equals(delivered.geometry.iloc[0])
    assert runner.file_sha256(raw) == original_hash
    provenance = runner.read_json(tmp_path / "fields_pan_aoi.gpkg.provenance.json")
    assert provenance["n_excluded"] == 1
    assert not provenance["geometry_modified"] and not provenance["inference_repeated"]
