"""Output reuse: study-area file contents and results versions (agribound._results).

Stubbed engine on the local test raster (no GPU, GEE or ML dependencies).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
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
    # An engine without a results version (the stub replaces it), so that only the
    # components a test turns on are versioned.
    return dict(
        source="local",
        local_tif_path=sample_rgb_tif,
        study_area=_write_square_aoi(tmp_path / "aoi.geojson", 1500),  # 3 km square
        engine="geoai",
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
        # The Delineate-Anything fine-tuning recipe changed after 1.0.1 (warmup_bias_lr=0),
        # then again (overlapping chips, no gradient accumulation, slivers, nodata rule).
        assert RESULTS_VERSIONS["delineate_anything_finetune"] >= 3
        # So did the native Delineate-Anything tile stitching.
        assert RESULTS_VERSIONS["delineate_anything"] >= 2
        # And DINOv3: the fine-tuning recipe (class weights, mean-IoU checkpoint, input
        # normalisation, boundary band) and the inference (blended windows).
        assert RESULTS_VERSIONS["dinov3_finetune"] >= 2 and RESULTS_VERSIONS["dinov3"] >= 2
        # And DINOv3 on rasters coarser than 10 m, now read on a 10 m processing grid.
        assert RESULTS_VERSIONS["dinov3_processing_grid"] >= 2

    @pytest.mark.parametrize(
        ("engine", "engine_params", "fine_tune", "expected"),
        [
            ("dinov3", {}, True, {"dinov3_finetune", "dinov3"}),
            # a fine-tuned checkpoint is in the config hash; the inference changed
            ("dinov3", {"checkpoint_path": "m.ckpt"}, False, {"dinov3"}),
            ("ensemble", {"engines": [{"engine": "dinov3"}, "ftw"]}, False, {"dinov3"}),
            ("ensemble", {"engines": [" DINOv3 "]}, False, {"dinov3"}),
            ("ensemble", {}, False, set()),  # default members: delineate-anything, ftw
            ("geoai", {}, True, set()),
        ],
    )
    def test_dinov3(self, engine, engine_params, fine_tune, expected):
        config = SimpleNamespace(
            engine=engine, engine_params=engine_params, fine_tune=fine_tune, sam_refine=False
        )
        versions = results_versions(config)
        assert {k for k in versions if k.startswith("dinov3")} == expected
        assert all(versions[k] == RESULTS_VERSIONS[k] for k in expected)

    @pytest.mark.parametrize(
        ("source", "engine_params", "fine_tune", "expected"),
        [
            # Fine-tuned on pixels coarser than 10 m: the 10 m processing grid ("auto").
            ("landsat", {}, True, True),
            ("hls", {}, True, True),
            ("landsat-pan", {}, True, True),  # 15 m
            ("landsat", {"processing_gsd_m": "auto"}, True, True),
            ("landsat", {"processing_gsd_m": 15}, True, True),
            ("landsat", {"processing_gsd_m": "native"}, True, False),  # as before the grid
            ("landsat", {"processing_gsd_m": 30}, True, False),  # the raster's own grid
            # 10 m or finer: read as they are, so their results versions do not change.
            ("sentinel2", {}, True, False),
            ("sentinel1", {}, True, False),
            ("spot", {}, True, False),
            ("spot-pan", {}, True, False),
            ("naip", {}, True, False),
            ("usgs-naip-plus", {}, True, False),
            ("umbra", {}, True, False),
            ("sentinel2", {"processing_gsd_m": 5}, True, True),  # a finer grid when asked
            ("usgs-naip-plus", {"processing_gsd_m": 0.5}, True, True),  # pixel size unknown
            # A checkpoint reads the grid recorded with it (its path is in the config hash),
            ("landsat", {"checkpoint_path": "m.ckpt"}, False, False),
            # unless engine_params set one.
            ("landsat", {"checkpoint_path": "m.ckpt", "processing_gsd_m": "auto"}, False, True),
            ("landsat", {"checkpoint_path": "m.ckpt", "processing_gsd_m": "native"}, False, False),
            ("sentinel2", {"checkpoint_path": "m.ckpt", "processing_gsd_m": "auto"}, False, False),
            # An invalid setting counts: the run raises it.
            ("landsat", {"processing_gsd_m": "fine"}, True, True),
            ("landsat", {"processing_gsd_m": 60}, True, True),  # coarser than the raster
        ],
    )
    def test_dinov3_processing_grid(self, source, engine_params, fine_tune, expected):
        config = SimpleNamespace(
            source=source,
            engine="dinov3",
            engine_params=engine_params,
            fine_tune=fine_tune,
            sam_refine=False,
            naip_resolution_m=1.0,
            vhr_sar_resolution_m=1.0,
        )
        versions = results_versions(config)
        assert ("dinov3_processing_grid" in versions) is expected
        assert "dinov3" in versions and ("dinov3_finetune" in versions) is fine_tune
        if expected:
            assert versions["dinov3_processing_grid"] == RESULTS_VERSIONS["dinov3_processing_grid"]

    def test_dinov3_processing_grid_naip_resolution(self):
        config = SimpleNamespace(
            source="naip",
            engine="dinov3",
            engine_params={},
            fine_tune=True,
            sam_refine=False,
            naip_resolution_m=30.0,
        )
        assert "dinov3_processing_grid" in results_versions(config)

    def test_dinov3_processing_grid_local_raster(self, tmp_path):
        from rasterio.transform import from_origin

        def raster(name, res):
            path = tmp_path / name
            profile = dict(driver="GTiff", height=4, width=4, count=3, dtype="uint8")
            transform = from_origin(X0, Y0 + 4 * res, res, res)
            with rasterio.open(path, "w", crs="EPSG:32611", transform=transform, **profile) as dst:
                dst.write(np.ones((3, 4, 4), dtype=np.uint8))
            return str(path)

        ref = tmp_path / "ref.geojson"
        ref.write_text('{"type": "FeatureCollection", "features": []}')
        for path, expected in (
            (raster("r30.tif", 30.0), True),
            (raster("r10.tif", 10.0), False),
            (str(tmp_path / "missing.tif"), False),  # not readable: the run reports it
        ):
            config = AgriboundConfig(
                source="local",
                local_tif_path=path,
                engine="dinov3",
                fine_tune=True,
                reference_boundaries=str(ref),
            )
            assert ("dinov3_processing_grid" in results_versions(config)) is expected

    def test_dinov3_processing_grid_local_raster_in_feet(self, tmp_path):
        """EPSG:2263 is in US survey feet: a 10 m raster (32.8 ft) is read as it is."""
        from rasterio.transform import from_origin

        foot = 0.3048006096012192
        ref = tmp_path / "ref.geojson"
        ref.write_text('{"type": "FeatureCollection", "features": []}')
        for metres, expected in ((10.0, False), (30.0, True)):
            path = tmp_path / f"ft{int(metres)}.tif"
            profile = dict(driver="GTiff", height=4, width=4, count=3, dtype="uint8")
            transform = from_origin(1_000_000.0, 200_000.0, metres / foot, metres / foot)
            with rasterio.open(path, "w", crs="EPSG:2263", transform=transform, **profile) as dst:
                dst.write(np.ones((3, 4, 4), dtype=np.uint8))
            config = AgriboundConfig(
                source="local",
                local_tif_path=str(path),
                engine="dinov3",
                fine_tune=True,
                reference_boundaries=str(ref),
            )
            assert ("dinov3_processing_grid" in results_versions(config)) is expected

    def test_dinov3_processing_grid_local_raster_in_degrees(self, tmp_path):
        """A crop is measured at its own latitude, so any latitude of the file can count.

        0.0001 degree pixels are 10.15 m wide at 24.2 N and 10.09 m at 25 N. The file spans
        24-26 N (10.09 m at its centre), and a study area at 24.2 N is read on the grid.
        """
        from rasterio.transform import from_origin

        from agribound.engines.finetune._data import dinov3_processing_gsd

        ref = tmp_path / "ref.geojson"
        ref.write_text('{"type": "FeatureCollection", "features": []}')

        def raster(name, north, south):
            path = tmp_path / name
            rows = int(round((north - south) / 0.0001))
            profile = dict(driver="GTiff", height=rows, width=4, count=3, dtype="uint8", nodata=0)
            transform = from_origin(-100.0, north, 0.0001, 0.0001)
            with rasterio.open(path, "w", crs="EPSG:4326", transform=transform, **profile) as dst:
                dst.write(np.full((3, rows, 4), 100, dtype=np.uint8))
            return str(path)

        spanning = raster("geo_24_26.tif", 26.0, 24.0)
        assert dinov3_processing_gsd(spanning) is None  # 10.09 m at the file's centre
        config = AgriboundConfig(
            source="local",
            local_tif_path=spanning,
            engine="dinov3",
            fine_tune=True,
            reference_boundaries=str(ref),
            study_area="bbox:-99.9999,24.19,-99.9997,24.21",
            output_path=str(tmp_path / "out" / "fields.gpkg"),
        )
        from agribound.composites.local import LocalCompositeBuilder

        crop = LocalCompositeBuilder().build(config)
        assert dinov3_processing_gsd(crop) == 10.0  # the engine reads the 10 m grid
        assert "dinov3_processing_grid" in results_versions(config)
        # A file whose every latitude gives pixels of 10.1 m or less does not count.
        north = raster("geo_30_31.tif", 31.0, 30.0)  # 9.54-9.64 m
        assert "dinov3_processing_grid" not in results_versions(
            config.merged(local_tif_path=north, study_area="")
        )

    @pytest.mark.parametrize(
        ("source", "export_crs", "params", "expected"),
        [
            # An export grid is laid out in the CRS's units: 30 ft (9.14 m) in EPSG:2263.
            ("landsat", "EPSG:2263", {}, False),
            ("landsat", "EPSG:32613", {}, True),
            ("landsat", "utm", {}, True),
            ("landsat", "EPSG:2263", {"processing_gsd_m": 5}, True),
            # In degrees the export resolution counts (documented: errs toward re-running).
            ("landsat", "EPSG:4326", {}, True),
        ],
    )
    def test_dinov3_processing_grid_export_crs_units(self, source, export_crs, params, expected):
        config = SimpleNamespace(
            source=source,
            engine="dinov3",
            engine_params=params,
            fine_tune=True,
            sam_refine=False,
            export_crs=export_crs,
        )
        assert ("dinov3_processing_grid" in results_versions(config)) is expected

    @pytest.mark.parametrize(
        ("processing_gsd_m", "expected"),
        [(3, False), (2, True), ("auto", False)],
    )
    def test_dinov3_processing_grid_vhr_sar_resolution(self, processing_gsd_m, expected):
        """Umbra and Capella composites have ``vhr_sar_resolution_m`` pixels (3 m here)."""
        config = SimpleNamespace(
            source="umbra",
            engine="dinov3",
            engine_params={"processing_gsd_m": processing_gsd_m},
            fine_tune=True,
            sam_refine=False,
            vhr_sar_resolution_m=3.0,
        )
        assert ("dinov3_processing_grid" in results_versions(config)) is expected

    @pytest.mark.parametrize(
        ("member_params", "expected"),
        [
            ({"checkpoint_path": "m.ckpt"}, False),
            ({"checkpoint_path": "m.ckpt", "processing_gsd_m": "auto"}, True),
        ],
    )
    def test_dinov3_processing_grid_ensemble_member(self, member_params, expected):
        config = SimpleNamespace(
            source="landsat",
            engine="ensemble",
            engine_params={"engines": [{"engine": "dinov3", "engine_params": member_params}]},
            fine_tune=False,
            sam_refine=False,
        )
        assert ("dinov3_processing_grid" in results_versions(config)) is expected

    @pytest.mark.parametrize(
        ("engine", "engine_params", "fine_tune", "expected"),
        [
            ("delineate-anything", {}, False, True),  # native is the default backend
            ("delineate-anything", {"backend": " Native "}, False, True),
            ("delineate-anything", {"checkpoint_path": "best.pt"}, False, True),
            ("delineate-anything", {}, True, True),  # the fine-tuned model runs natively
            ("delineate-anything", {"backend": "reference"}, False, False),
            ("delineate-anything", {"backend": "ftw"}, False, False),
            ("geoai", {}, False, False),
            ("ensemble", {}, False, True),  # default members include delineate-anything
            ("ensemble", {"engines": ["ftw", {"engine": "delineate-anything"}]}, False, True),
            (
                "ensemble",
                {
                    "engines": [
                        {"engine": "delineate-anything", "engine_params": {"backend": "ftw"}},
                        "ftw",
                    ]
                },
                False,
                False,
            ),
        ],
    )
    def test_native_delineate_anything(self, engine, engine_params, fine_tune, expected):
        # Not validated: only the fields results_versions reads matter.
        config = SimpleNamespace(
            engine=engine, engine_params=engine_params, fine_tune=fine_tune, sam_refine=False
        )
        versions = results_versions(config)
        assert ("delineate_anything" in versions) is expected
        if expected:
            assert versions["delineate_anything"] == RESULTS_VERSIONS["delineate_anything"]

    @pytest.mark.parametrize(
        ("engine", "fine_tune", "expected"),
        [
            ("delineate-anything", True, True),
            ("delineate-anything", False, False),  # a checkpoint_path is in the config hash
            ("geoai", True, False),  # other trainers are unchanged
        ],
    )
    def test_delineate_anything_fine_tuning(self, tmp_path, engine, fine_tune, expected):
        ref = tmp_path / "ref.geojson"
        ref.write_text('{"type": "FeatureCollection", "features": []}')
        config = AgriboundConfig(
            source="local",
            local_tif_path="x.tif",
            engine=engine,
            fine_tune=fine_tune,
            reference_boundaries=str(ref),
        )
        versions = results_versions(config)
        assert ("delineate_anything_finetune" in versions) is expected
        if expected:
            assert (
                versions["delineate_anything_finetune"]
                == RESULTS_VERSIONS["delineate_anything_finetune"]
            )

    @pytest.mark.parametrize(
        ("fields", "components"),
        [
            # The default engine is Delineate-Anything with its native backend.
            ({}, ["delineate_anything"]),
            ({"sam_refine": True}, ["delineate_anything", "sam_refine"]),
            (  # legacy switch
                {"engine_params": {"sam_refine": True}},
                ["delineate_anything", "sam_refine"],
            ),
            ({"engine": "geoai"}, []),
            ({"engine": "geoai", "sam_refine": True}, ["sam_refine"]),
            ({"engine": "ensemble"}, ["delineate_anything"]),
            ({"engine": "ensemble", "sam_refine": True}, ["delineate_anything", "sam_refine"]),
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

    def test_native_delineate_anything_output_of_1_0_1_is_not_reused(self, engine, run_kwargs):
        """The native backend stitches its tiles differently after 1.0.1 (a new results version)."""
        kwargs = {**run_kwargs, "engine": "delineate-anything"}
        delineate(**kwargs)
        current = RESULTS_VERSIONS["delineate_anything"]
        facts = read_provenance(kwargs["output_path"])["facts"]
        assert facts["results_versions"] == {"delineate_anything": current}
        # Rewrite the record as 1.0.1 wrote it: the fact without this component.
        path = provenance_path(kwargs["output_path"])
        record = json.loads(path.read_text())
        record["agribound_version"] = "1.0.1"
        record["facts"]["results_versions"] = {}
        path.write_text(json.dumps(record))
        with pytest.raises(FileExistsError, match=rf"delineate_anything 1 -> {current}"):
            delineate(**kwargs)
        assert engine.calls == 1
        # The reference backend's results did not change.
        reference = {**kwargs, "engine_params": {"backend": "reference"}}
        reference["output_path"] = str(Path(kwargs["output_path"]).with_name("reference.gpkg"))
        delineate(**reference)
        assert read_provenance(reference["output_path"])["facts"]["results_versions"] == {}

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
            record = self._record(
                config,
                aoi_fingerprint="000000000000",
                results_versions=results_versions(config),
            )
            assert reuse_mismatch(record, config) is None

    def test_malformed_results_versions_count_as_version_1(self):
        config = AgriboundConfig(
            source="sentinel2", study_area=BBOX, sam_refine=True, gee_project="test-project"
        )
        record = self._record(config, results_versions="bad")
        assert "sam_refine 1 ->" in reuse_mismatch(record, config)

    @pytest.mark.parametrize(("source", "reused"), [("landsat", False), ("sentinel2", True)])
    def test_dinov3_outputs_before_the_processing_grid(self, tmp_path, source, reused):
        """A DINOv3 fine-tuning made before the 10 m grid is refused on 30 m imagery only."""
        ref = tmp_path / "ref.geojson"
        ref.write_text('{"type": "FeatureCollection", "features": []}')
        config = AgriboundConfig(
            source=source,
            year=2024,
            study_area=BBOX,
            gee_project="test-project",
            engine="dinov3",
            fine_tune=True,
            reference_boundaries=str(ref),
        )
        before = {"dinov3_finetune": 2, "dinov3": 2}  # recorded before the grid
        mismatch = reuse_mismatch(self._record(config, results_versions=before), config)
        if reused:
            assert mismatch is None and results_versions(config) == before
        else:
            assert "dinov3_processing_grid 1 -> 2" in mismatch


# ---------------------------------------------------------------------------
# TESSERA: the state of the store (agribound.composites.tessera_stamp)
# ---------------------------------------------------------------------------


class StubTesseraBuilder:
    """Writes the test raster with the tags of a TESSERA raster, as the builder records them."""

    def __init__(self, raster):
        self.raster = raster
        self.calls = 0

    def build(self, config):
        import shutil

        from agribound.composites import tessera_stamp
        from agribound.composites.local import (
            current_tessera_stamp,
            format_read_bbox,
            tessera_read_bbox,
        )

        self.calls += 1
        bbox = tessera_read_bbox(config)
        try:
            tags = current_tessera_stamp(config, bbox).tags()
        except tessera_stamp.StoreCheckError:
            tags = {}
        path = Path(config.get_working_dir()) / f"tessera_embedding_{self.calls}.tif"
        shutil.copy(self.raster, path)
        with rasterio.open(path, "r+") as dst:
            dst.update_tags(
                AGRIBOUND_SOURCE="tessera-embedding",
                TESSERA_READ_BBOX=format_read_bbox(bbox),
                **tags,
            )
        return str(path)


@pytest.fixture
def tessera_run(engine, sample_rgb_tif, tmp_path, monkeypatch):
    """Keyword arguments of a TESSERA run with a stub builder (``kwargs, builder``)."""
    pytest.importorskip("geotessera")  # resolve_tessera_store reads geotessera's registry
    builder = StubTesseraBuilder(sample_rgb_tif)
    monkeypatch.setattr("agribound.composites.get_composite_builder", lambda source: builder)
    kwargs = dict(
        source="tessera-embedding",
        engine="embedding",
        year=2024,
        study_area=_write_square_aoi(tmp_path / "aoi.geojson", 1500),
        output_path=str(tmp_path / "out" / "fields.gpkg"),
        device="cpu",
        lulc_filter=False,
    )
    return kwargs, builder


def _store_stamps(monkeypatch, factory, fingerprint=None, error=None):
    """Make tessera_stamp.store_stamp return *fingerprint* or raise *error*; record its calls.

    Without *fingerprint*, the stamp's fingerprint follows the store, box and year
    (conftest ``fake_tessera_fingerprint``), as the unit tests' default stamp does.
    """
    from agribound.composites import tessera_stamp

    calls = []

    def store_stamp(store_url, bbox, year, *, client=None, **kwargs):
        calls.append(tuple(bbox))
        if error is not None:
            raise error
        return factory(store_url, bbox, year, fingerprint=fingerprint)

    monkeypatch.setattr(tessera_stamp, "store_stamp", store_stamp)
    return calls


def _edit_record(output_path, edit) -> dict:
    """Apply *edit* to the provenance record of *output_path* and write it back."""
    path = provenance_path(output_path)
    record = json.loads(path.read_text())
    edit(record)
    path.write_text(json.dumps(record))
    return record


class TestTesseraStore:
    def test_output_records_the_store_stamp_and_is_reused(
        self, tessera_run, engine, caplog, tessera_fingerprint
    ):
        kwargs, builder = tessera_run
        delineate(**kwargs)
        composite = read_provenance(kwargs["output_path"])["facts"]["composite"]
        from agribound.composites.local import parse_read_bbox, resolve_tessera_store

        box = parse_read_bbox(composite["TESSERA_READ_BBOX"])
        store_url, _ = resolve_tessera_store("v1")
        assert composite["TESSERA_STORE_FINGERPRINT"] == tessera_fingerprint(store_url, box, 2024)
        assert composite["TESSERA_STORE_LAST_MODIFIED"] == "2026-10-07T12:40:56Z"
        with caplog.at_level(logging.WARNING):
            assert delineate(**kwargs).attrs["reused"] is True
        assert engine.calls == 1 and builder.calls == 1
        assert _reuse_warnings(caplog) == []

    def test_output_of_a_changed_store_is_not_reused(
        self, tessera_run, engine, monkeypatch, tessera_stamp_factory
    ):
        kwargs, builder = tessera_run
        delineate(**kwargs)
        recorded = read_provenance(kwargs["output_path"])["facts"]["composite"]
        calls = _store_stamps(monkeypatch, tessera_stamp_factory, fingerprint="f" * 64)
        with pytest.raises(FileExistsError) as info:
            delineate(**kwargs)
        message = str(info.value)
        assert "the TESSERA store has changed since it was made" in message
        fingerprint = recorded["TESSERA_STORE_FINGERPRINT"]
        assert f"fingerprint {fingerprint[:12]}" in message and "ffffffffffff" in message
        assert "overwrite=True" in message
        assert engine.calls == 1 and builder.calls == 1 and len(calls) == 1
        delineate(**kwargs, overwrite=True)
        assert engine.calls == 2 and builder.calls == 2
        assert delineate(**kwargs).attrs["reused"] is True
        assert engine.calls == 2

    def test_stamp_is_computed_for_the_recorded_box(
        self, tessera_run, engine, monkeypatch, tessera_stamp_factory
    ):
        """The check reads the box from the record (TESSERA_READ_BBOX), not the config."""
        from agribound.composites.local import format_read_bbox, parse_read_bbox

        kwargs, _ = tessera_run
        delineate(**kwargs)
        box = parse_read_bbox(
            read_provenance(kwargs["output_path"])["facts"]["composite"]["TESSERA_READ_BBOX"]
        )
        other = (box[0] - 0.001, box[1], box[2], box[3] + 0.001)

        def move(record):
            record["facts"]["composite"]["TESSERA_READ_BBOX"] = format_read_bbox(other)

        _edit_record(kwargs["output_path"], move)
        calls = _store_stamps(monkeypatch, tessera_stamp_factory)
        with pytest.raises(FileExistsError, match="the TESSERA store has changed"):
            delineate(**kwargs)  # the stamp of the other box differs from the one recorded
        assert calls == [other]

    def test_record_without_a_stamp_is_not_reused(self, tessera_run, engine):
        kwargs, _ = tessera_run
        delineate(**kwargs)
        _edit_record(kwargs["output_path"], _drop_stamp)
        with pytest.raises(FileExistsError, match="does not hold the state of the TESSERA store"):
            delineate(**kwargs)
        delineate(**kwargs, overwrite=True)
        assert delineate(**kwargs).attrs["reused"] is True
        assert engine.calls == 2

    def test_record_of_1_0_0_is_refused_without_a_reuse_warning(self, tessera_run, engine, caplog):
        """No study-area fingerprint and no stamp: refused, and nothing says "reusing it"."""
        kwargs, _ = tessera_run
        delineate(**kwargs)

        def as_1_0_0(record):
            _drop_stamp(record)
            record["facts"].pop("aoi_fingerprint")

        _edit_record(kwargs["output_path"], as_1_0_0)
        with caplog.at_level(logging.WARNING), pytest.raises(FileExistsError) as info:
            delineate(**kwargs)
        assert "does not hold the state of the TESSERA store" in str(info.value)
        assert _reuse_warnings(caplog) == []
        # With the store stamp, the study-area WARNING is logged and the output reused.
        delineate(**kwargs, overwrite=True)
        _edit_record(kwargs["output_path"], lambda record: record["facts"].pop("aoi_fingerprint"))
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert delineate(**kwargs).attrs["reused"] is True
        (warning,) = _reuse_warnings(caplog)
        assert "has no study-area fingerprint" in warning and "reusing it" in warning

    def test_store_check_failure_reuses_with_a_warning(
        self, tessera_run, engine, monkeypatch, tessera_stamp_factory, caplog
    ):
        from agribound.composites.tessera_stamp import StoreCheckError

        kwargs, _ = tessera_run
        delineate(**kwargs)
        error = StoreCheckError("HEAD https://data.source.coop/... failed: gaierror")
        _store_stamps(monkeypatch, tessera_stamp_factory, error=error)
        with caplog.at_level(logging.WARNING):
            assert delineate(**kwargs).attrs["reused"] is True
        (warning,) = _reuse_warnings(caplog)
        assert warning.startswith("Could not check whether the TESSERA store has changed")
        assert "gaierror" in warning and "fields.gpkg" in warning
        assert "no longer has" in warning and "overwrite=True" in warning
        assert engine.calls == 1

    def test_store_is_not_checked_when_something_else_differs(
        self, tessera_run, engine, monkeypatch, tessera_stamp_factory
    ):
        """Same configuration hash, another results version: the store is not asked."""
        kwargs, _ = tessera_run
        delineate(**kwargs)
        calls = _store_stamps(monkeypatch, tessera_stamp_factory)
        monkeypatch.setitem(_results.RESULTS_VERSIONS, "embedding", 99)
        with pytest.raises(FileExistsError, match="results versions"):
            delineate(**kwargs)
        assert calls == []


def _drop_stamp(record):
    """Remove the store stamp from a record, as written before the store was recorded."""
    for key in ("TESSERA_STORE_FINGERPRINT", "TESSERA_STORE_LAST_MODIFIED", "TESSERA_READ_BBOX"):
        record["facts"]["composite"].pop(key, None)


def _retag(path, **tags):
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(**tags)


class TestTesseraStoreOffline:
    """Without network access the record is compared with the cached raster's stamp."""

    @pytest.fixture
    def made_online(self, tessera_run, monkeypatch, tessera_stamp_factory):
        """An output made with network access; then the store cannot be reached."""
        from agribound.composites.tessera_stamp import StoreCheckError

        kwargs, _ = tessera_run
        delineate(**kwargs)
        record = read_provenance(kwargs["output_path"])
        assert record["facts"]["composite"]["TESSERA_STORE_FINGERPRINT"]
        _store_stamps(
            monkeypatch,
            tessera_stamp_factory,
            error=StoreCheckError("GET https://data.source.coop/... failed: gaierror"),
        )
        return kwargs, record, record["facts"]["raster_path"]

    def test_raster_read_again_since_refuses_the_output(self, made_online, engine):
        kwargs, record, raster = made_online
        _retag(raster, TESSERA_STORE_FINGERPRINT="e" * 64)  # read again online (composite stage)
        with pytest.raises(FileExistsError) as info:
            delineate(**kwargs)
        message = str(info.value)
        assert "it was made from another state of the TESSERA store" in message
        assert raster in message and "eeeeeeeeeeee" in message
        recorded = record["facts"]["composite"]["TESSERA_STORE_FINGERPRINT"]
        assert recorded[:12] in message
        assert "could not be checked now" in message and "gaierror" in message
        assert engine.calls == 1

    def test_record_without_a_stamp_is_refused_when_the_raster_has_one(self, made_online, engine):
        kwargs, _, _ = made_online
        _edit_record(kwargs["output_path"], _drop_stamp)
        with pytest.raises(FileExistsError, match="does not hold the state of the TESSERA store"):
            delineate(**kwargs)

    def test_raster_without_a_stamp_or_of_another_box_shows_nothing(
        self, made_online, engine, caplog
    ):
        from agribound.composites.local import format_read_bbox, parse_read_bbox

        kwargs, record, raster = made_online
        _retag(raster, TESSERA_STORE_FINGERPRINT="")  # cached when the store was unreachable
        with caplog.at_level(logging.WARNING):
            assert delineate(**kwargs).attrs["reused"] is True
        box = parse_read_bbox(record["facts"]["composite"]["TESSERA_READ_BBOX"])
        other = format_read_bbox((box[0], box[1], box[2] + 0.01, box[3]))
        _retag(raster, TESSERA_STORE_FINGERPRINT="e" * 64, TESSERA_READ_BBOX=other)
        with caplog.at_level(logging.WARNING):
            assert delineate(**kwargs).attrs["reused"] is True  # stamps of other reads
        warnings = _reuse_warnings(caplog)
        assert len(warnings) == 2
        assert all(w.startswith("Could not check whether the TESSERA store") for w in warnings)
        assert engine.calls == 1

    def test_the_raster_the_run_uses_comes_first(self, made_online, engine):
        """The cached raster of the configuration (tessera_raster_path) is compared first."""
        import shutil

        from agribound.composites.local import tessera_raster_path

        kwargs, _, raster = made_online
        cached = tessera_raster_path(AgriboundConfig(**kwargs))
        shutil.copy(raster, cached)
        _retag(cached, TESSERA_STORE_FINGERPRINT="d" * 64)
        with pytest.raises(FileExistsError) as info:
            delineate(**kwargs)
        assert str(cached) in str(info.value) and "dddddddddddd" in str(info.value)

    def test_unverified_outputs_can_be_collected_instead_of_logged(
        self, made_online, engine, caplog
    ):
        kwargs, record, _ = made_online
        config = AgriboundConfig(**kwargs)
        notes = []
        with caplog.at_level(logging.WARNING):
            assert reuse_mismatch(record, config, "out.gpkg", unverified=notes) is None
        assert _reuse_warnings(caplog) == []
        (note,) = notes
        assert note.startswith("the existing output 'out.gpkg': StoreCheckError")


class TestOtherSourcesUnchanged:
    """Outside TESSERA, the store is never checked and hashes and cache keys are unchanged."""

    @pytest.fixture
    def store_checks(self, monkeypatch):
        """Calls of the store check (a list, which must stay empty)."""
        from agribound.composites import tessera_stamp

        calls = []

        def checked(*args, **kwargs):
            calls.append(args)
            raise tessera_stamp.StoreCheckError("the TESSERA store was checked")

        monkeypatch.setattr(tessera_stamp, "store_stamp", checked)
        monkeypatch.setattr(tessera_stamp, "refresh_local_cache", checked)
        return calls

    def test_local_source_is_reused_without_a_store_check(
        self, engine, run_kwargs, caplog, store_checks
    ):
        delineate(**run_kwargs)
        with caplog.at_level(logging.WARNING):
            assert delineate(**run_kwargs).attrs["reused"] is True
        assert engine.calls == 1 and _reuse_warnings(caplog) == []
        assert store_checks == []

    @pytest.mark.parametrize(
        ("fields", "config_hash_value", "key"),
        [
            # Values of the code before the TESSERA store check (identical hashes).
            (
                {"source": "google-embedding", "engine": "embedding", "gee_project": "p"},
                "ba1dae93a322a76aa9119ba2686a86fc09db258c",
                "5b9ae323f30a",
            ),
            (
                {
                    "source": "google-embedding",
                    "engine": "embedding",
                    "google_embedding_backend": "source_coop",
                },
                "f1b3411655710552861ad3282087469862d995c2",
                "14a88a10e8e6",
            ),
            (
                {"source": "sentinel2", "gee_project": "p"},
                "179d2dab439db30b19d596124b5dd8fd7fadfdf8",
                "6c85881afb2d",
            ),
            # TESSERA's configuration hash and cache key are unchanged too: the stamp is
            # in the raster's tags and the provenance facts, not in either.
            (
                {"source": "tessera-embedding", "engine": "embedding"},
                "a5aa2358b1a2f2b5f0e829bf3c8f847b98dd5a7f",
                "6314cf8ffad3",
            ),
        ],
    )
    def test_hashes_cache_keys_and_reuse(self, fields, config_hash_value, key, store_checks):
        from agribound._cache import cache_key
        from agribound.provenance import config_hash

        config = AgriboundConfig(study_area=BBOX, year=2024, output_path="o.gpkg", **fields)
        assert config_hash(config) == config_hash_value
        assert cache_key(config) == key
        if config.source == "tessera-embedding":
            return
        record = {
            "status": "success",
            "config_hash": config_hash(config),
            "facts": {"results_versions": results_versions(config), "composite": {}},
        }
        assert reuse_mismatch(record, config) is None
        assert store_checks == []

    def test_raster_file_names_are_unchanged(self, tmp_path):
        pytest.importorskip("geotessera")
        from agribound._cache import cache_path
        from agribound.composites.local import EMBEDDING_RECIPE_VERSION, resolve_tessera_store

        base = dict(study_area=BBOX, year=2024, output_path=str(tmp_path / "o.gpkg"))
        tessera = AgriboundConfig(source="tessera-embedding", engine="embedding", **base)
        url, variant = resolve_tessera_store("v1")
        path = cache_path(
            tessera, "tessera_embedding", ".tif", EMBEDDING_RECIPE_VERSION, url, variant
        )
        assert path.name == "tessera_embedding_65a38eda818f.tif"
        google = AgriboundConfig(
            source="google-embedding", engine="embedding", gee_project="p", **base
        )
        path = cache_path(google, "google_embedding", ".tif", EMBEDDING_RECIPE_VERSION, "gee")
        assert path.name == "google_embedding_d3ac5334bf46.tif"


# ---------------------------------------------------------------------------
# HLS: outputs made before HLS recipe 2 at the MGRS tiles at 180 degrees
# ---------------------------------------------------------------------------

#: Inside MGRS tile T01KAA (Koro, Fiji), whose HLSS30 images had reversed footprints in
#: Earth Engine, and away from 180 degrees (example 03's Murray-Darling box).
KORO = "bbox:179.38,-17.36,179.44,-17.30"
MURRAY = "bbox:149.7,-30.3,149.85,-30.15"
#: Composite tags of an HLS output made before HLS recipe 2 (agribound 1.0.1 at Koro).
PRE_RECIPE_2_TAGS = {"AGRIBOUND_N_IMAGES": "112", "AGRIBOUND_EXPORT_CRS": "EPSG:32760"}


class TestHlsAt180:
    """``reuse_mismatch`` for ``source="hls"``: AGRIBOUND_HLS_RECIPE_VERSION and the study area."""

    @staticmethod
    def _config(study_area, **fields):
        return AgriboundConfig(
            source="hls", year=2022, study_area=study_area, gee_project="p", **fields
        )

    @staticmethod
    def _record(config, composite=None, **facts):
        from agribound.provenance import config_hash

        facts.setdefault("results_versions", results_versions(config))
        if composite is not None:
            facts["composite"] = composite
        return {"status": "success", "config_hash": config_hash(config), "facts": facts}

    @pytest.mark.parametrize(
        ("study_area", "refused"),
        [
            (KORO, True),
            ("bbox:179.95,-16.85,180.0,-16.80", True),  # ends at 180 degrees
            ("bbox:-179.76,-17.62,-179.71,-17.58", True),  # the east edge of T01KAA
            ("bbox:179.23,51.48,179.27,51.52", True),  # Amchitka, Aleutian Islands
            ("bbox:179.48,71.28,179.52,71.32", True),  # Wrangel Island, Chukotka
            (
                "POLYGON ((179.4 -17.4, 179.45 -17.4, 179.45 -17.35, 179.4 -17.35, 179.4 -17.4))",
                True,
            ),
            (MURRAY, False),
            (BBOX, False),
            # Within 5 degrees of 180, outside the latitudes of the affected tiles.
            ("bbox:175.0,-41.6,178.6,-37.5", False),  # New Zealand's North Island
        ],
    )
    def test_record_without_the_recipe_version(self, study_area, refused):
        config = self._config(study_area)
        reason = reuse_mismatch(self._record(config, composite=PRE_RECIPE_2_TAGS), config)
        if refused:
            assert reason == (
                "it was made from an HLS composite selected before HLS recipe 2 (its record "
                "has no AGRIBOUND_HLS_RECIPE_VERSION), and its study area lies at the MGRS "
                "tiles at 180 degrees whose HLSS30 images Earth Engine selected everywhere "
                "except over their own tile, so that composite can lack images of those tiles "
                "(see the HLS section of the satellite-sources guide)"
            )
        else:
            assert reason is None
        # A record without composite tags is treated alike.
        assert (reuse_mismatch(self._record(config), config) is None) is not refused
        # Made with HLS recipe 2: reused.
        tags = {**PRE_RECIPE_2_TAGS, "AGRIBOUND_HLS_RECIPE_VERSION": "2"}
        assert reuse_mismatch(self._record(config, composite=tags), config) is None

    @pytest.mark.parametrize("source", ["sentinel2", "landsat", "sentinel1"])
    def test_other_sources_are_not_checked(self, source):
        config = AgriboundConfig(source=source, year=2022, study_area=KORO, gee_project="p")
        assert reuse_mismatch(self._record(config, composite={}), config) is None

    def test_checked_only_when_nothing_else_differs(self, monkeypatch):
        import agribound.provenance as provenance

        reads = []
        monkeypatch.setattr(
            provenance, "_study_area_4326_offline", lambda config: reads.append(config) or None
        )
        config = self._config(KORO)
        record = self._record(config, composite={}, results_versions={"delineate_anything": 0})
        reason = reuse_mismatch(record, config)
        assert "results versions" in reason and "HLS recipe 2" not in reason
        assert reads == []

    def test_study_area_file(self, tmp_path, caplog):
        path = tmp_path / "koro.geojson"
        koro = box(179.38, -17.36, 179.44, -17.30)
        gpd.GeoDataFrame(geometry=[koro], crs="EPSG:4326").to_file(path)
        config = self._config(str(path))
        record = self._record(config, composite={}, aoi_fingerprint=aoi_fingerprint(config))
        assert reuse_mismatch(record, config).startswith("it was made from an HLS composite")
        # In a projected CRS too.
        gpd.GeoDataFrame(geometry=[koro], crs="EPSG:4326").to_crs("EPSG:32760").to_file(path)
        record = self._record(config, composite={}, aoi_fingerprint=aoi_fingerprint(config))
        assert reuse_mismatch(record, config).startswith("it was made from an HLS composite")

    def test_gee_asset_is_read_from_its_local_copy_only(self, tmp_path, monkeypatch, caplog):
        from shapely.geometry import mapping

        import agribound.io.vector as vector

        def no_request(*args, **kwargs):
            raise AssertionError("Earth Engine must not be contacted")

        monkeypatch.setattr(vector, "_read_gee_asset", no_request)
        monkeypatch.setattr("agribound.auth.ensure_gee", no_request)
        config = self._config("projects/p/assets/koro", cache_dir=str(tmp_path / "cache"))
        record = self._record(config, composite={})
        # Without a local copy, or with one that cannot be read: reused, with a WARNING.
        copy = vector.study_area_cache_file(config)
        for content, why in (
            (None, "FileNotFoundError: the GEE asset has no local copy"),
            ("{not json", "ValueError: the local copy of the GEE asset"),
            (json.dumps({"features": "x"}), "ValueError: the local copy of the GEE asset"),
        ):
            if content is not None:
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_text(content)
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                assert reuse_mismatch(record, config, "out.gpkg") is None
            (warning,) = _reuse_warnings(caplog)
            assert warning.startswith(
                "Could not read the study area without Earth Engine to check whether the "
                "existing output 'out.gpkg', made from an HLS composite selected before HLS "
                "recipe 2, lies at the MGRS tiles at 180 degrees"
            )
            assert f"({why}" in warning
        # With the copy (written when the asset was read for the composite): refused at Koro.
        feature = {"type": "Feature", "geometry": mapping(box(179.38, -17.36, 179.44, -17.30))}
        copy.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
        assert reuse_mismatch(record, config).startswith("it was made from an HLS composite")

    def test_delineate_refuses_the_output(self, tmp_path, engine):
        from agribound.provenance import write_provenance

        config = self._config(KORO, output_path=str(tmp_path / "out.gpkg"))
        gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_file(config.output_path)
        record = self._record(config, composite=PRE_RECIPE_2_TAGS)
        write_provenance(config.output_path, {**record, "agribound_version": "1.0.1"})
        with pytest.raises(FileExistsError, match="selected before HLS recipe 2") as info:
            delineate(config=config)
        assert "Pass overwrite=True (CLI: --overwrite)" in str(info.value)
        assert engine.calls == 0
