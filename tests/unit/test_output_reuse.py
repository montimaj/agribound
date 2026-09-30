"""Output reuse: study-area file contents and results versions (agribound._results).

Stubbed engine on the local test raster (no GPU, GEE or ML dependencies).
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box

from agribound import _results
from agribound._cache import aoi_fingerprint
from agribound._results import RESULTS_VERSIONS, results_versions
from agribound.config import AgriboundConfig
from agribound.pipeline import delineate
from agribound.provenance import provenance_path, read_provenance, reuse_mismatch

X0, Y0 = 500000.0, 4000000.0  # lower-left corner of the 640 m test raster (EPSG:32611)
BBOX = "bbox:-117.0,36.1,-116.99,36.2"


class StubEngine:
    """Returns two 200 m squares inside the test raster and counts its calls."""

    def __init__(self):
        self.calls = 0

    def delineate(self, raster_path, config):
        self.calls += 1
        boxes = [
            box(X0 + 20, Y0 + 20, X0 + 220, Y0 + 220),
            box(X0 + 300, Y0 + 300, X0 + 500, Y0 + 500),
        ]
        gdf = gpd.GeoDataFrame(geometry=boxes, crs="EPSG:32611")
        gdf.attrs["engine_meta"] = {"backend": "stub"}
        return gdf


@pytest.fixture
def engine(monkeypatch):
    stub = StubEngine()
    monkeypatch.setattr("agribound.engines.get_engine", lambda name: stub)
    return stub


@pytest.fixture
def fake_refine(monkeypatch):
    calls = []

    def refine(gdf, raster_path, config, **kwargs):
        calls.append(len(gdf))
        out = gdf.copy()
        out.attrs["sam_stats"] = {"n_total": len(gdf)}
        return out

    monkeypatch.setattr("agribound.engines.samgeo_engine.refine_boundaries", refine)
    return calls


def _write_square_aoi(path, half_side_m: float, name: str = "aoi") -> str:
    """Write a square study area (EPSG:4326 GeoJSON) centred on the test raster."""
    cx, cy = X0 + 320, Y0 + 320
    square = box(cx - half_side_m, cy - half_side_m, cx + half_side_m, cy + half_side_m)
    gdf = gpd.GeoDataFrame({"name": [name]}, geometry=[square], crs="EPSG:32611")
    gdf.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
    return str(path)


@pytest.fixture
def run_kwargs(sample_rgb_tif, tmp_path):
    return dict(
        source="local",
        local_tif_path=sample_rgb_tif,
        study_area=_write_square_aoi(tmp_path / "aoi.geojson", 1500),  # 3 km square
        engine="delineate-anything",
        output_path=str(tmp_path / "out" / "fields.gpkg"),
        device="cpu",
        lulc_filter=False,
    )


def _bump_mtime(path) -> None:
    """Move the modification time 1 s ahead (file systems with a coarse clock)."""
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))


def _as_1_0_0_record(output_path) -> None:
    """Rewrite the provenance record as agribound 1.0.0 wrote it (no reuse facts)."""
    path = provenance_path(output_path)
    record = json.loads(path.read_text())
    record["agribound_version"] = "1.0.0"
    record["facts"].pop("aoi_fingerprint")
    record["facts"].pop("results_versions")
    path.write_text(json.dumps(record))


def _reuse_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "agribound.provenance"
    ]


# ---------------------------------------------------------------------------
# Results versions
# ---------------------------------------------------------------------------


class TestResultsVersions:
    def test_registry(self):
        assert all(isinstance(v, int) and v >= 1 for v in RESULTS_VERSIONS.values())
        # agribound 1.0.1 changed the embedding clustering and the SAM refinement.
        assert RESULTS_VERSIONS["embedding"] >= 2 and RESULTS_VERSIONS["sam_refine"] >= 2

    @pytest.mark.parametrize(
        ("fields", "components"),
        [
            ({}, []),
            ({"sam_refine": True}, ["sam_refine"]),
            ({"engine_params": {"sam_refine": True}}, ["sam_refine"]),  # legacy switch
            ({"engine": "ensemble"}, []),
            ({"engine": "ensemble", "sam_refine": True}, ["sam_refine"]),
        ],
    )
    def test_imagery_engines(self, fields, components):
        config = AgriboundConfig(source="local", local_tif_path="x.tif", **fields)
        assert results_versions(config) == {c: RESULTS_VERSIONS[c] for c in components}

    @pytest.mark.parametrize(
        ("fields", "components"),
        [
            ({}, ["embedding"]),
            (
                {"sam_refine": True, "engine_params": {"sam_rgb_bands": [1, 2, 3]}},
                ["embedding", "sam_refine"],
            ),
        ],
    )
    def test_embedding_engine(self, fields, components):
        config = AgriboundConfig(
            source="google-embedding", engine="embedding", study_area=BBOX, **fields
        )
        assert results_versions(config) == {c: RESULTS_VERSIONS[c] for c in components}

    @pytest.mark.parametrize(
        ("members", "expected"),
        [
            (["embedding"], True),
            ([{"engine": " Embedding ", "engine_params": {}}], True),
            (["delineate-anything", {"engine": "ftw"}], False),
            (None, False),  # default members
        ],
    )
    def test_ensemble_members(self, members, expected):
        # No source supports the ensemble and the embedding engine today, so the
        # configuration is not validated here.
        params = {} if members is None else {"engines": members}
        config = SimpleNamespace(engine="ensemble", engine_params=params, sam_refine=False)
        assert ("embedding" in results_versions(config)) is expected


# ---------------------------------------------------------------------------
# Pipeline reuse
# ---------------------------------------------------------------------------


class TestPipelineReuse:
    def test_unchanged_output_is_reused_and_facts_are_recorded(self, engine, run_kwargs, caplog):
        first = delineate(**run_kwargs)
        facts = read_provenance(run_kwargs["output_path"])["facts"]
        assert facts["aoi_fingerprint"] == aoi_fingerprint(AgriboundConfig(**run_kwargs))
        assert facts["results_versions"] == {}
        with caplog.at_level(logging.WARNING):
            second = delineate(**run_kwargs)
        assert engine.calls == 1
        assert second.attrs["reused"] is True and second.attrs["run_id"] == first.attrs["run_id"]
        assert _reuse_warnings(caplog) == []

    def test_changed_study_area_file_is_not_reused(self, engine, run_kwargs):
        """Regression: a 6 km square written over a 3 km one reused the 3 km output."""
        first = delineate(**run_kwargs)
        _write_square_aoi(run_kwargs["study_area"], 3000)  # same path, 6 km square
        _bump_mtime(run_kwargs["study_area"])
        with pytest.raises(FileExistsError) as info:
            delineate(**run_kwargs)
        message = str(info.value)
        assert "different study area" in message and "aoi.geojson" in message
        assert "overwrite=True" in message
        assert engine.calls == 1

        rerun = delineate(**run_kwargs, overwrite=True)
        assert engine.calls == 2 and rerun.attrs["run_id"] != first.attrs["run_id"]
        # The new output is reused for the new study area.
        assert delineate(**run_kwargs).attrs["reused"] is True
        assert engine.calls == 2

    def test_rewritten_file_with_the_same_geometry_is_reused(self, engine, run_kwargs):
        delineate(**run_kwargs)
        # Other attributes and a new modification time, same geometry.
        _write_square_aoi(run_kwargs["study_area"], 1500, name="renamed")
        _bump_mtime(run_kwargs["study_area"])
        assert delineate(**run_kwargs).attrs["reused"] is True
        assert engine.calls == 1

    def test_results_version_bump_is_not_reused(self, engine, run_kwargs, fake_refine, monkeypatch):
        kwargs = {**run_kwargs, "sam_refine": True}
        delineate(**kwargs)
        current = RESULTS_VERSIONS["sam_refine"]
        assert read_provenance(kwargs["output_path"])["facts"]["results_versions"] == {
            "sam_refine": current
        }
        monkeypatch.setitem(_results.RESULTS_VERSIONS, "sam_refine", current + 1)
        with pytest.raises(FileExistsError, match=rf"sam_refine {current} -> {current + 1}"):
            delineate(**kwargs)
        assert engine.calls == 1 and len(fake_refine) == 1
        delineate(**kwargs, overwrite=True)
        assert delineate(**kwargs).attrs["reused"] is True
        assert engine.calls == 2

    def test_bump_of_a_component_the_run_does_not_use_is_ignored(
        self, engine, run_kwargs, monkeypatch
    ):
        delineate(**run_kwargs)
        monkeypatch.setitem(_results.RESULTS_VERSIONS, "embedding", 99)
        assert delineate(**run_kwargs).attrs["reused"] is True
        assert engine.calls == 1

    def test_1_0_0_record_is_reused_with_a_warning(self, engine, run_kwargs, caplog):
        delineate(**run_kwargs)
        _as_1_0_0_record(run_kwargs["output_path"])
        with caplog.at_level(logging.WARNING):
            assert delineate(**run_kwargs).attrs["reused"] is True
        assert engine.calls == 1
        (warning,) = _reuse_warnings(caplog)
        assert "written by agribound 1.0.0" in warning and "no study-area fingerprint" in warning
        assert "aoi.geojson" in warning and "overwrite=True" in warning

    def test_1_0_0_record_of_a_changed_component_is_not_reused(
        self, engine, run_kwargs, fake_refine
    ):
        kwargs = {**run_kwargs, "sam_refine": True}
        delineate(**kwargs)
        _as_1_0_0_record(kwargs["output_path"])
        current = RESULTS_VERSIONS["sam_refine"]
        with pytest.raises(FileExistsError) as info:
            delineate(**kwargs)
        message = str(info.value)
        assert "produced by agribound 1.0.0" in message
        assert f"sam_refine 1 -> {current}" in message and "overwrite=True" in message
        assert engine.calls == 1

    def test_1_0_0_record_with_a_bbox_study_area_is_reused_silently(
        self, engine, run_kwargs, sample_rgb_tif, caplog
    ):
        with rasterio.open(sample_rgb_tif) as src:
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        kwargs = {**run_kwargs, "study_area": "bbox:" + ",".join(str(v) for v in bounds)}
        delineate(**kwargs)
        _as_1_0_0_record(kwargs["output_path"])
        with caplog.at_level(logging.WARNING):
            assert delineate(**kwargs).attrs["reused"] is True
        assert _reuse_warnings(caplog) == []  # the configuration hash covers a bbox

    def test_replaced_local_raster_without_study_area_is_not_reused(self, engine, run_kwargs):
        kwargs = {**run_kwargs, "study_area": None}
        delineate(**kwargs)
        with rasterio.open(kwargs["local_tif_path"], "r+") as dst:
            dst.write(np.zeros((3, 64, 64), dtype=np.uint16))
        _bump_mtime(kwargs["local_tif_path"])
        with pytest.raises(FileExistsError, match="different local raster"):
            delineate(**kwargs)
        assert engine.calls == 1

    def test_unreadable_study_area_file_warns_and_reuses(self, engine, run_kwargs, caplog):
        delineate(**run_kwargs)
        os.remove(run_kwargs["study_area"])
        with caplog.at_level(logging.WARNING):
            assert delineate(**run_kwargs).attrs["reused"] is True
        (warning,) = _reuse_warnings(caplog)
        assert warning.startswith("Could not read the study-area file")


class TestReuseMismatch:
    def _record(self, config, **facts):
        from agribound.provenance import config_hash

        return {"status": "success", "config_hash": config_hash(config), "facts": facts}

    def test_configuration_hash_is_checked_first(self):
        config = AgriboundConfig(source="local", local_tif_path="x.tif", sam_refine=True)
        record = {**self._record(config), "config_hash": "0" * 40}
        assert reuse_mismatch(record, config).startswith("it was produced with a different")

    def test_string_study_areas_are_not_fingerprinted_again(self):
        # bbox, WKT and GEE asset study areas are part of the configuration hash.
        for study_area in (BBOX, "POLYGON ((0 0, 1 0, 1 1, 0 0))", "projects/p/assets/a"):
            config = AgriboundConfig(
                source="sentinel2", study_area=study_area, gee_project="test-project"
            )
            record = self._record(config, aoi_fingerprint="000000000000", results_versions={})
            assert reuse_mismatch(record, config) is None

    def test_malformed_results_versions_count_as_version_1(self):
        config = AgriboundConfig(
            source="sentinel2", study_area=BBOX, sam_refine=True, gee_project="test-project"
        )
        record = self._record(config, results_versions="bad")
        assert "sam_refine 1 ->" in reuse_mismatch(record, config)
