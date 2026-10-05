"""Offline checks for reference identity, reprojection, vintage and suite summaries."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box


@pytest.fixture
def example():
    root = Path(__file__).resolve().parents[2]
    path = root / "examples" / "25_landsat_international_comparison.py"
    spec = importlib.util.spec_from_file_location("international_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_checksum_is_required_for_network_and_cached_files(example, tmp_path, monkeypatch):
    content = b"pinned public reference"
    source = {
        "filename": "ref.parquet",
        "url": "https://example.invalid/ref",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    monkeypatch.setattr(example, "urlopen", lambda *args, **kwargs: io.BytesIO(content))
    path = example.fetch_reference(source, tmp_path)
    assert path.read_bytes() == content
    monkeypatch.setattr(example, "urlopen", lambda *args, **kwargs: pytest.fail("Offline download"))
    assert example.fetch_reference(source, tmp_path, offline=True) == path
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        example.fetch_reference(source, tmp_path, offline=True)


def test_bad_download_never_becomes_a_reference_cache(example, tmp_path, monkeypatch):
    source = {"filename": "ref.parquet", "url": "https://example.invalid/ref", "sha256": "0" * 64}
    monkeypatch.setattr(example, "urlopen", lambda *args, **kwargs: io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="checksum"):
        example.fetch_reference(source, tmp_path)
    assert not (tmp_path / "ref.parquet").exists()


def test_reference_selection_reprojects_and_keeps_whole_polygons(example):
    frame = gpd.GeoDataFrame(
        {"id": ["a", "b", "outside"]},
        geometry=[
            box(5.49, 51.56, 5.58, 51.58),
            box(5.58, 51.56, 5.59, 51.57),
            box(6, 52, 6.1, 52.1),
        ],
        crs=4326,
    ).to_crs(32631)
    selected, repaired = example.select_reference(frame, {"bbox": [5.53, 51.55, 5.60, 51.59]})
    assert selected.id.tolist() == ["a", "b"]
    assert selected.crs == frame.crs
    assert selected.geometry.iloc[0].equals(frame.geometry.iloc[0])
    assert selected.to_crs(4326).total_bounds[0] < 5.53
    assert repaired == 0


def test_reference_repair_and_identity_checks(example):
    invalid = Polygon([(5.54, 51.56), (5.56, 51.58), (5.54, 51.58), (5.56, 51.56), (5.54, 51.56)])
    frame = gpd.GeoDataFrame(
        {"id": ["a", "b"]}, geometry=[invalid, box(5.58, 51.56, 5.59, 51.57)], crs=4326
    )
    site = {"bbox": [5.53, 51.55, 5.60, 51.59]}
    selected, repaired = example.select_reference(frame, site)
    assert repaired == 1
    assert selected.geometry.is_valid.all()
    frame["id"] = "same"
    with pytest.raises(ValueError, match="Duplicate"):
        example.select_reference(frame, site)
    with pytest.raises(ValueError, match="CRS"):
        example.select_reference(frame.set_crs(None, allow_override=True), site)


def test_reference_vintage_and_crop_provenance(example, tmp_path):
    path = tmp_path / "reference.parquet"
    frame = gpd.GeoDataFrame(
        {
            "id": ["a", "b"],
            "crop_name": ["Wheat", "Barley"],
            "determination_datetime": pd.to_datetime(["2018-08-01"] * 2, utc=True),
        },
        geometry=[box(20.91, -34.19, 20.92, -34.18), box(20.93, -34.19, 20.94, -34.18)],
        crs=4326,
    )
    frame.to_parquet(path)
    source = {"reference_year": 2018, "crop_column": "crop_name"}
    site = {"bbox": [20.90, -34.20, 20.95, -34.16]}
    output, metadata = example.prepare_reference(source, site, path, tmp_path)
    assert len(gpd.read_file(output)) == 2
    meta = json.loads(metadata.read_text())
    assert meta["crop_or_use_class_counts"] == {"Wheat": 1, "Barley": 1}
    assert meta["source_file_sha256"] == example.sha256(path)
    with pytest.raises(ValueError, match="vintage"):
        example.prepare_reference({**source, "reference_year": 2023}, site, path, tmp_path)


def test_frozen_selection_cannot_be_changed_after_a_run(example, tmp_path):
    config = Path(__file__).resolve().parents[2] / "examples" / "landsat_international_sites.json"
    data, digest = example.freeze_config(config, tmp_path / "out")
    assert len(data["sites"]) >= 6
    assert len({s["country"] for s in data["sites"]}) >= 4
    assert "France" not in {s["country"] for s in data["sites"]}
    assert len(digest) == 64
    data["sites"][0]["date_start"] = "2022-04-01"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Frozen"):
        example.freeze_config(changed, tmp_path / "out")
    data["comparison"]["cloud_cover_max"] = 50
    changed.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="fixed example 23"):
        example.freeze_config(changed, tmp_path / "fresh_out")


def test_site_means_and_field_weighting_are_distinct_and_skip_undefined(example):
    base = {
        "cohort": "international",
        "experiment": "pan",
        "boundary_tolerance_m": 15,
        "size_class_ha": "all",
        "precision": 0.5,
        "recall": 0.5,
        "matched_iou": np.nan,
        "boundary_precision": 0.5,
        "boundary_recall": 0.5,
        "boundary_f1": 0.5,
        "oversegmentation": 0.2,
        "undersegmentation": 0.3,
        "inference_s": 10.0,
    }
    table = pd.DataFrame(
        [
            {**base, "site": "small", "n_reference": 10, "f1": 0.2},
            {**base, "site": "large", "n_reference": 90, "f1": 0.8},
        ]
    )
    results = example.aggregate_metrics(table)
    new = results[results.cohort == "international"].set_index("weighting")
    assert new.loc["equal_site", "f1"] == pytest.approx(0.5)
    assert new.loc["reference_count", "f1"] == pytest.approx(0.74)
    assert new.loc["reference_count", "n_reference"] == 100
    assert new.loc["equal_site", "matched_iou_n_sites"] == 0
    assert np.isnan(new.loc["equal_site", "matched_iou"])


def test_provider_vintage_clarification_preserves_metrics_and_geometry(example, tmp_path):
    source = {"filename": "boundaries_south_africa_2018.parquet", "reference_year": 2018}
    (tmp_path / "reference_source.json").write_text(json.dumps(source))
    (tmp_path / "reference_evaluation.json").write_text(
        json.dumps({**source, "vintage_matches": True})
    )
    (tmp_path / "run_status.json").write_text(
        json.dumps({"status": "complete", "reference": source})
    )
    provenance = {"facts": {"reference": source}, "status": "success"}
    path = tmp_path / "fields_pan.gpkg.provenance.json"
    path.write_text(json.dumps(provenance))
    metric_path = tmp_path / "comparison.csv"
    metric_path.write_text("f1\n0.5\n")
    vector_path = tmp_path / "reference.gpkg"
    vector_path.write_bytes(b"unchanged geometry bytes")
    before = example.sha256(metric_path), example.sha256(vector_path)
    example.clarify_reference_metadata(tmp_path)
    assert before == (example.sha256(metric_path), example.sha256(vector_path))
    assert (
        json.loads((tmp_path / "reference_evaluation.json").read_text())["vintage_matches"] is False
    )
    assert json.loads(path.read_text())["facts"]["reference"]["boundary_imagery_years"] == [
        2016,
        2018,
    ]


def test_no_scene_failure_exports_selection_evidence_without_rasters(tmp_path, monkeypatch):
    from agribound.composites import landsat_matched

    manifest = {
        "pairs": [],
        "unmatched_pan": [{"scene_id": "only_pan"}],
        "unmatched_sr": [],
        "cloud_cover_max": 20,
    }

    def no_scenes(*args):
        raise landsat_matched.NoMatchedScenesError(manifest)

    monkeypatch.setattr(landsat_matched, "paired_collections", no_scenes)
    with pytest.raises(ValueError, match="No matched"):
        landsat_matched.export_matched(None, None, tmp_path)
    evidence = json.loads((tmp_path / "scene_selection_failure.json").read_text())
    assert evidence["status"] == "unmeasured"
    assert evidence["unmatched_pan"] == manifest["unmatched_pan"]
    assert not list(tmp_path.glob("*.tif"))
