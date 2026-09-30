"""Tests for agribound.hpc (tiling, manifest, idempotent tile runs, merge rule, CLI, scripts)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import pytest
import shapely
import yaml
from click.testing import CliRunner
from shapely.geometry import box

from agribound.config import AgriboundConfig
from agribound.hpc import regions as hpc_regions
from agribound.hpc import tiles as hpc_tiles
from agribound.io.crs import utm_zone_for_lon
from agribound.provenance import config_hash, read_provenance, write_provenance

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"

# A box straddling the UTM 55/56 seam (150 E) in the Namoi region.
SEAM_BBOX = "bbox:149.80,-30.50,150.30,-30.20"

#: The HPC/region scripts are bash for Linux clusters (and macOS). On Windows,
#: ``bash`` may resolve to the WSL launcher, the fake ``sbatch`` shims cannot run
#: and a CRLF checkout breaks the scripts, so these tests are skipped there.
needs_posix_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="needs a POSIX bash (HPC scripts are not supported on Windows)",
)


def _utm_box_aoi(epsg: int, x0: float, y0: float, width: float, height: float):
    """A rectangle defined in UTM, returned as a densified EPSG:4326 polygon."""
    rect = shapely.segmentize(box(x0, y0, x0 + width, y0 + height), 500.0)
    return gpd.GeoSeries([rect], crs=f"EPSG:{epsg}").to_crs("EPSG:4326").iloc[0]


def _base_config(tmp_path: Path, **overrides) -> AgriboundConfig:
    params = {
        "source": "local",
        "local_tif_path": str(tmp_path / "dummy.tif"),
        "engine": "delineate-anything",
        "study_area": SEAM_BBOX,
        "output_path": str(tmp_path / "base" / "fields.gpkg"),
        "lulc_filter": False,
    }
    params.update(overrides)
    return AgriboundConfig(**params)


def _make_manifest(tmp_path: Path, study_area=SEAM_BBOX, **tile_kwargs) -> dict:
    tiles = hpc_tiles.make_tiles(study_area, **tile_kwargs)
    path = hpc_tiles.write_tile_manifest(tiles, _base_config(tmp_path), tmp_path / "run")
    return hpc_tiles.load_manifest(path)


# ---------------------------------------------------------------------------
# make_tiles
# ---------------------------------------------------------------------------


class TestMakeTiles:
    def test_utm_seam_split(self):
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        assert set(tiles["system"]) == {"55S", "56S"}
        epsg = dict(zip(tiles["system"], tiles["utm_epsg"], strict=True))
        assert epsg == {"55S": 32755, "56S": 32756}
        # Every core lies in its own zone band (west/east of 150 E).
        for row in tiles.itertuples():
            minx, _, maxx, _ = row.geometry.bounds
            if row.system == "55S":
                assert maxx <= 150.0 + 1e-9
            else:
                assert minx >= 150.0 - 1e-9

    def test_cores_partition_the_study_area(self):
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        ea = tiles.geometry.to_crs("EPSG:6933")
        aoi = gpd.GeoSeries([box(149.80, -30.50, 150.30, -30.20)], crs="EPSG:4326")
        aoi_area = aoi.to_crs("EPSG:6933").area.iloc[0]
        assert ea.area.sum() == pytest.approx(aoi_area, rel=1e-4)
        # No two cores overlap (beyond numerical noise).
        union_area = shapely.union_all(ea.values).area
        assert union_area == pytest.approx(ea.area.sum(), rel=1e-6)

    def test_counts_for_aligned_utm_box(self):
        # 40 km x 60 km rectangle whose lower-left corner is on whole metres.
        aoi = _utm_box_aoi(32613, 600000.0, 3600000.0, 40000.0, 60000.0)
        tiles = hpc_tiles.make_tiles(aoi, tile_size_m=20000, halo_m=1500)
        assert len(tiles) == 6
        assert sorted(zip(tiles["col"], tiles["row"], strict=True)) == [
            (c, r) for c in range(2) for r in range(3)
        ]
        assert tiles["core_area_km2"].to_numpy() == pytest.approx(400.0, rel=1e-4)
        assert list(tiles["index"]) == list(range(6))

    def test_halo_geometry(self):
        aoi = _utm_box_aoi(32613, 600000.0, 3600000.0, 40000.0, 40000.0)
        tiles = hpc_tiles.make_tiles(aoi, tile_size_m=20000, halo_m=1500)
        grid = tiles.attrs["grid"]
        origin = grid["systems"]["13N"]["origin"]
        for row in tiles.itertuples():
            cx0 = origin[0] + row.col * 20000
            cy0 = origin[1] + row.row * 20000
            assert row.halo_minx == pytest.approx(cx0 - 1500, abs=1e-3)
            assert row.halo_miny == pytest.approx(cy0 - 1500, abs=1e-3)
            assert row.halo_maxx == pytest.approx(cx0 + 21500, abs=1e-3)
            assert row.halo_maxy == pytest.approx(cy0 + 21500, abs=1e-3)
            assert row.halo.contains(row.geometry)
            # The tile's study area (bbox string) covers the whole halo.
            bbox = box(*[float(v) for v in row.study_area[5:].split(",")])
            assert bbox.covers(row.halo)

    def test_no_clip_keeps_whole_cells(self):
        aoi = _utm_box_aoi(32613, 600500.0, 3600500.0, 30000.0, 10000.0)
        clipped = hpc_tiles.make_tiles(aoi, tile_size_m=20000, halo_m=1000, clip=True)
        whole = hpc_tiles.make_tiles(aoi, tile_size_m=20000, halo_m=1000, clip=False)
        assert len(clipped) == len(whole) == 2
        assert whole["core_area_km2"].to_numpy() == pytest.approx(400.0, rel=1e-4)
        assert clipped["core_area_km2"].sum() == pytest.approx(300.0, rel=1e-4)

    def test_equal_area_grid(self):
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000, crs="equal-area")
        assert set(tiles["system"]) == {"ea"}
        assert tiles.attrs["grid"]["systems"]["ea"]["crs"].startswith("+proj=laea")
        # Export CRS follows each core's UTM zone.
        assert set(tiles["utm_epsg"]) == {32755, 32756}

    def test_equator_split(self):
        tiles = hpc_tiles.make_tiles("bbox:36.9,-0.2,37.1,0.2", tile_size_m=10000, halo_m=500)
        assert set(tiles["system"]) == {"37N", "37S"}

    @pytest.mark.parametrize("kwargs", [{"tile_size_m": 0}, {"halo_m": -1}, {"crs": "mercator"}])
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            hpc_tiles.make_tiles(SEAM_BBOX, **kwargs)

    def test_antimeridian_like_bounds_rejected(self):
        with pytest.raises(ValueError):
            hpc_tiles.make_tiles(box(170, 0, 190, 1))


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------


class TestOwnership:
    def test_zone_formula_matches_io_crs(self):
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        grid = tiles.attrs["grid"]
        lons = np.linspace(149.81, 150.29, 200)
        points = shapely.points(lons, np.full_like(lons, -30.3))
        ids = hpc_tiles.assign_tile_ids(points, grid)
        for lon, tid in zip(lons, ids, strict=True):
            assert tid.startswith(f"{utm_zone_for_lon(lon):02d}S_")

    def test_every_core_owns_its_representative_point(self):
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        ids = hpc_tiles.assign_tile_ids(
            hpc_tiles.representative_points(tiles.geometry), tiles.attrs["grid"]
        )
        assert list(ids) == list(tiles["tile_id"])

    def test_shared_edge_belongs_to_east_cell(self):
        grid = {
            "kind": "equal-area",
            "tile_size_m": 1000.0,
            "systems": {"ea": {"crs": "EPSG:32613", "origin": [600000.0, 3600000.0]}},
        }
        to_ll = pyproj.Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
        lon, lat = to_ll.transform(601000.0, 3600500.0)
        # Round-trip through lon/lat can move the point by ~1e-9 m, so compare both sides.
        east = hpc_tiles.assign_tile_ids([shapely.Point(lon, lat)], grid)[0]
        assert east in ("ea_001_000", "ea_000_000")
        exact = hpc_tiles.assign_tile_ids(
            [shapely.Point(*to_ll.transform(601000.0 + 1e-3, 3600500.0))], grid
        )[0]
        assert exact == "ea_001_000"

    def test_empty_points(self):
        grid = {"kind": "utm", "tile_size_m": 1000.0, "systems": {}}
        ids = hpc_tiles.assign_tile_ids([shapely.Point(), shapely.Point(10, 10)], grid)
        assert list(ids) == [None, None]

    def test_format_index_ranges(self):
        assert hpc_tiles.format_index_ranges([5, 0, 1, 2, 7, 8, 2]) == "0-2,5,7-8"
        assert hpc_tiles.format_index_ranges([]) == ""
        assert hpc_tiles.format_index_ranges([3]) == "3"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_round_trip(self, tmp_path):
        m = _make_manifest(tmp_path)
        root = Path(m["_root"])
        assert m["n_tiles"] == len(m["tiles"]) > 1
        lines = (root / "tiles.txt").read_text().splitlines()
        assert lines == [e["tile_id"] for e in m["tiles"]]
        layers = set(gpd.list_layers(root / "tiles.gpkg")["name"])
        assert layers == {"tiles", "halos", "study_area"}
        for entry in m["tiles"]:
            cfg = AgriboundConfig.from_yaml(root / entry["config"])
            assert cfg.study_area == entry["study_area"]
            assert cfg.output_path == str(root / entry["output"])
            assert cfg.cache_dir == str(root / entry["dir"] / "cache")
            assert cfg.export_crs == f"EPSG:{entry['utm_epsg']}"
            assert cfg.overwrite is False and cfg.provenance is True
            assert config_hash(cfg) == entry["config_hash"]

    def test_rewrite_is_idempotent_and_changes_are_refused(self, tmp_path):
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        base = _base_config(tmp_path)
        path = hpc_tiles.write_tile_manifest(tiles, base, tmp_path / "run")
        first = json.loads(path.read_text())
        hpc_tiles.write_tile_manifest(tiles, base, tmp_path / "run")
        assert json.loads(path.read_text())["created_utc"] == first["created_utc"]

        other = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=10000, halo_m=1000)
        with pytest.raises(FileExistsError):
            hpc_tiles.write_tile_manifest(other, base, tmp_path / "run")
        hpc_tiles.write_tile_manifest(other, base, tmp_path / "run", overwrite=True)
        assert json.loads(path.read_text())["n_tiles"] == len(other)

    def test_shared_cache_root(self, tmp_path):
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        base = _base_config(tmp_path, cache_dir=str(tmp_path / "shared"))
        m = hpc_tiles.load_manifest(hpc_tiles.write_tile_manifest(tiles, base, tmp_path / "run"))
        cfg = hpc_tiles.load_tile_config(m, 0)
        assert cfg.cache_dir == str((tmp_path / "shared" / m["tiles"][0]["tile_id"]).resolve())

    def test_fine_tune_refused_and_reference_dropped(self, tmp_path):
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        ref = tmp_path / "ref.gpkg"
        gpd.GeoDataFrame(geometry=[box(149.9, -30.4, 149.91, -30.39)], crs=4326).to_file(ref)
        tuned = _base_config(tmp_path, reference_boundaries=str(ref), fine_tune=True)
        with pytest.raises(ValueError, match="fine-tune"):
            hpc_tiles.write_tile_manifest(tiles, tuned, tmp_path / "a")
        evaluated = _base_config(tmp_path, reference_boundaries=str(ref))
        m = hpc_tiles.load_manifest(hpc_tiles.write_tile_manifest(tiles, evaluated, tmp_path / "b"))
        assert hpc_tiles.load_tile_config(m, 0).reference_boundaries is None
        assert m["reference_boundaries"] == str(ref)
        kept = hpc_tiles.write_tile_manifest(tiles, evaluated, tmp_path / "c", keep_reference=True)
        assert hpc_tiles.load_tile_config(kept, 0).reference_boundaries == str(ref)

    def test_explicit_export_crs_is_kept(self, tmp_path):
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        base = _base_config(tmp_path, export_crs="EPSG:3577")
        m = hpc_tiles.load_manifest(hpc_tiles.write_tile_manifest(tiles, base, tmp_path / "run"))
        assert {hpc_tiles.load_tile_config(m, e["index"]).export_crs for e in m["tiles"]} == {
            "EPSG:3577"
        }

    def test_bad_manifest(self, tmp_path):
        (tmp_path / "manifest.json").write_text("{}")
        with pytest.raises(ValueError):
            hpc_tiles.load_manifest(tmp_path)
        with pytest.raises(FileNotFoundError):
            hpc_tiles.load_manifest(tmp_path / "missing")


# ---------------------------------------------------------------------------
# run_tile with a mocked pipeline
# ---------------------------------------------------------------------------


class FakePipeline:
    """Stand-ins for build_composite / delineate that write real files."""

    def __init__(self, polygons_4326=None):
        self.composite_calls = 0
        self.delineate_calls = 0
        self.fail_delineate = False
        self.polygons = polygons_4326

    def build_composite(self, config):
        self.composite_calls += 1
        path = Path(config.get_working_dir()) / "composite.tif"
        path.write_bytes(b"fake")
        return str(path)

    def delineate(self, config=None, **_):
        self.delineate_calls += 1
        if self.fail_delineate:
            raise RuntimeError("engine exploded")
        geoms = self.polygons if self.polygons is not None else []
        gdf = gpd.GeoDataFrame({"score": np.arange(len(geoms))}, geometry=geoms, crs=4326)
        out = Path(config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out, layer="fields", driver="GPKG")
        write_provenance(
            out,
            {
                "status": "success",
                "config_hash": config_hash(config),
                "run_id": f"run{self.delineate_calls}",
                "wall_s": 1.5,
                "facts": {"n_output": len(gdf), "raster_path": str(out)},
                "steps": [{"name": "delineate", "wall_s": 1.0}],
            },
        )
        return gdf


@pytest.fixture
def fake_pipeline(monkeypatch):
    fake = FakePipeline()
    import agribound.pipeline as pipeline

    monkeypatch.setattr(pipeline, "build_composite", fake.build_composite)
    monkeypatch.setattr(pipeline, "delineate", fake.delineate)
    return fake


class TestRunTile:
    def test_all_stage_is_idempotent(self, tmp_path, fake_pipeline):
        m = _make_manifest(tmp_path)
        first = hpc_tiles.run_tile(m, 0, stage="all")
        assert first["status"] == "done" and fake_pipeline.delineate_calls == 1
        again = hpc_tiles.run_tile(m, 0, stage="all")
        assert again["status"] == "skipped" and fake_pipeline.delineate_calls == 1
        forced = hpc_tiles.run_tile(m, 0, stage="all", overwrite=True)
        assert forced["status"] == "done" and fake_pipeline.delineate_calls == 2
        tile_dir = Path(m["_root"]) / m["tiles"][0]["dir"]
        assert (tile_dir / "delineate.done.json").exists()

    def test_overwritten_tile_requires_merge_overwrite(self, tmp_path, fake_pipeline):
        """As the run_tile docstring says: a re-run tile invalidates the merged output."""
        m = _make_manifest(tmp_path)
        for i in range(len(m["tiles"])):
            hpc_tiles.run_tile(m, i, stage="all")
        hpc_tiles.merge_tiles(m)
        assert hpc_tiles.merge_tiles(m).attrs["reused"] is True  # up to date
        hpc_tiles.run_tile(m, 0, stage="all", overwrite=True)  # new run ID for tile 0
        with pytest.raises(FileExistsError, match="not made from the current tile outputs"):
            hpc_tiles.merge_tiles(m)
        merged = hpc_tiles.merge_tiles(m, overwrite=True)
        assert not merged.attrs.get("reused")
        doc = " ".join((hpc_tiles.run_tile.__doc__ or "").split())
        assert ":func:`merge_tiles` then raises :class:`FileExistsError`" in doc
        assert "also given ``overwrite=True``" in doc

    def test_two_phase(self, tmp_path, fake_pipeline):
        m = _make_manifest(tmp_path)
        with pytest.raises(RuntimeError, match="has not been staged"):
            hpc_tiles.run_tile(m, 1, stage="delineate")
        status = hpc_tiles.tile_status(m).set_index("index")
        assert status.loc[1, "delineate"] == "failed"
        assert "not been staged" in status.loc[1, "error"]

        staged = hpc_tiles.run_tile(m, 1, stage="composite")
        assert staged["status"] == "done" and fake_pipeline.composite_calls == 1
        assert hpc_tiles.run_tile(m, 1, stage="composite")["status"] == "skipped"
        assert fake_pipeline.composite_calls == 1
        assert hpc_tiles.run_tile(m, 1, stage="delineate")["status"] == "done"
        status = hpc_tiles.tile_status(m).set_index("index")
        assert status.loc[1, "composite"] == "done"
        assert status.loc[1, "delineate"] == "done"
        assert status.loc[1, "error"] is None

    def test_stage_marker_is_shared_through_cache(self, tmp_path, fake_pipeline):
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        shared = tmp_path / "cache"
        a = hpc_tiles.write_tile_manifest(
            tiles, _base_config(tmp_path, cache_dir=str(shared)), tmp_path / "a"
        )
        b_cfg = _base_config(tmp_path, cache_dir=str(shared), engine="geoai")
        b = hpc_tiles.write_tile_manifest(tiles, b_cfg, tmp_path / "b")
        hpc_tiles.run_tile(a, 0, stage="composite")
        # Another engine over the same tiles sees the composite as staged.
        assert hpc_tiles.run_tile(b, 0, stage="delineate")["status"] == "done"

    def test_failure_marker_then_recovery(self, tmp_path, fake_pipeline):
        m = _make_manifest(tmp_path)
        fake_pipeline.fail_delineate = True
        with pytest.raises(RuntimeError, match="engine exploded"):
            hpc_tiles.run_tile(m, 0)
        marker = Path(m["_root"]) / m["tiles"][0]["dir"] / "delineate.failed.json"
        record = json.loads(marker.read_text())
        assert record["error"] == "RuntimeError: engine exploded"
        assert "Traceback" in record["traceback"]
        fake_pipeline.fail_delineate = False
        hpc_tiles.run_tile(m, 0)
        assert not marker.exists()

    def test_stale_output_detected(self, tmp_path, fake_pipeline):
        m = _make_manifest(tmp_path)
        hpc_tiles.run_tile(m, 0)
        cfg_path = Path(m["_root"]) / m["tiles"][0]["config"]
        data = yaml.safe_load(cfg_path.read_text())
        data["min_field_area_m2"] = 9999.0
        cfg_path.write_text(yaml.safe_dump(data))
        assert hpc_tiles.tile_status(m).set_index("index").loc[0, "delineate"] == "stale"

    def test_output_with_other_results_versions_is_stale(self, tmp_path, fake_pipeline):
        """The pipeline's reuse test (agribound._results): a 1.0.0 SAM-refined tile is stale."""
        from agribound._results import results_versions

        tiles = hpc_tiles.make_tiles(SEAM_BBOX)
        config = _base_config(tmp_path, sam_refine=True)
        m = hpc_tiles.load_manifest(hpc_tiles.write_tile_manifest(tiles, config, tmp_path / "run"))
        hpc_tiles.run_tile(m, 0)
        # The fake pipeline writes the record without results_versions, as 1.0.0 did.
        assert hpc_tiles.tile_status(m).set_index("index").loc[0, "delineate"] == "stale"
        # Not skipped (the real pipeline then asks for --overwrite, as for a changed config).
        assert hpc_tiles.run_tile(m, 0)["status"] == "done"
        assert fake_pipeline.delineate_calls == 2

        output = Path(m["_root"]) / m["tiles"][0]["output"]
        record = read_provenance(output)
        record["facts"]["results_versions"] = results_versions(config)
        write_provenance(output, record)
        assert hpc_tiles.tile_status(m).set_index("index").loc[0, "delineate"] == "done"
        assert hpc_tiles.run_tile(m, 0)["status"] == "skipped"
        assert fake_pipeline.delineate_calls == 2

    def test_tile_selection(self, tmp_path, fake_pipeline):
        m = _make_manifest(tmp_path)
        tid = m["tiles"][2]["tile_id"]
        assert hpc_tiles.run_tile(m, tid)["index"] == 2
        with pytest.raises(ValueError, match="out of range"):
            hpc_tiles.run_tile(m, len(m["tiles"]))
        with pytest.raises(ValueError):
            hpc_tiles.run_tile(m, 0, stage="bogus")


class FakeFTW:
    """Stands in for ftw.resolve_ftw_model and FTWEngine._two_windows."""

    def __init__(self, n_windows: int = 2):
        self.n_windows = n_windows
        self.window_calls: list[tuple[str, dict]] = []

    def resolve(self, engine_params):
        from types import SimpleNamespace

        return SimpleNamespace(n_windows=self.n_windows)

    def two_windows(self, config, raster_path, params, rgbn):
        self.window_calls.append((raster_path, dict(params)))
        record = {}
        windows = {"a": ("2023-09-01", "2023-10-31"), "b": ("2024-02-01", "2024-03-31")}
        for label, (start, end) in windows.items():
            path = Path(config.get_working_dir()) / f"window_{label}.tif"
            path.write_bytes(b"window")
            record[label] = {"start": start, "end": end, "raster": str(path), "status": "composite"}
        return [(record["a"]["raster"], rgbn), (record["b"]["raster"], rgbn)], record


@pytest.fixture
def fake_ftw(monkeypatch):
    import agribound.engines.ftw as ftw

    fake = FakeFTW()
    monkeypatch.setattr(ftw, "resolve_ftw_model", fake.resolve)
    monkeypatch.setattr(ftw.FTWEngine, "_two_windows", staticmethod(fake.two_windows))
    hpc_tiles._ftw_n_windows.cache_clear()
    yield fake
    hpc_tiles._ftw_n_windows.cache_clear()


def _ftw_manifest(tmp_path: Path, name: str = "ftw", **overrides) -> dict:
    tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
    params = {
        "source": "sentinel2",
        "local_tif_path": None,
        "engine": "ftw",
        "gee_project": "test-project",
    }
    params.update(overrides)
    base = _base_config(tmp_path, **params)
    return hpc_tiles.load_manifest(hpc_tiles.write_tile_manifest(tiles, base, tmp_path / name))


class TestEngineInputStaging:
    def test_two_window_ftw_windows_built_in_composite_stage(
        self, tmp_path, fake_pipeline, fake_ftw
    ):
        m = _ftw_manifest(tmp_path, engine_params={"window_days": 20})
        result = hpc_tiles.run_tile(m, 0, stage="composite")
        assert result["status"] == "done"
        assert len(fake_ftw.window_calls) == 1
        raster, params = fake_ftw.window_calls[0]
        # The engine's window builder gets the composite and the tile's engine_params.
        assert raster == result["raster_path"] and params == {"window_days": 20}
        status = hpc_tiles.tile_status(m).set_index("index")
        assert status.loc[0, "composite"] == "done"
        # Staged tiles are skipped, and delineation proceeds offline.
        assert hpc_tiles.run_tile(m, 0, stage="composite")["status"] == "skipped"
        assert len(fake_ftw.window_calls) == 1
        assert hpc_tiles.run_tile(m, 0, stage="delineate")["status"] == "done"

    def test_missing_window_raster_means_not_staged(self, tmp_path, fake_pipeline, fake_ftw):
        m = _ftw_manifest(tmp_path)
        hpc_tiles.run_tile(m, 0, stage="composite")
        cfg = hpc_tiles.load_tile_config(m, 0)
        (Path(cfg.get_working_dir()) / "window_b.tif").unlink()
        assert hpc_tiles.tile_status(m).set_index("index").loc[0, "composite"] == "pending"
        with pytest.raises(RuntimeError, match="not been staged"):
            hpc_tiles.run_tile(m, 0, stage="delineate")
        # Re-staging rebuilds the windows.
        assert hpc_tiles.run_tile(m, 0, stage="composite")["status"] == "done"
        assert len(fake_ftw.window_calls) == 2

    def test_single_window_model_needs_no_windows(self, tmp_path, fake_pipeline, fake_ftw):
        fake_ftw.n_windows = 1
        m = _ftw_manifest(tmp_path)
        hpc_tiles.run_tile(m, 0, stage="composite")
        assert fake_ftw.window_calls == []
        assert hpc_tiles.run_tile(m, 0, stage="delineate")["status"] == "done"

    def test_shared_composite_does_not_stage_ftw_windows(self, tmp_path, fake_pipeline, fake_ftw):
        shared = str(tmp_path / "cache")
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        common = {"source": "sentinel2", "local_tif_path": None, "gee_project": "test-project"}
        da = _base_config(tmp_path, cache_dir=shared, **common)
        ftw = _base_config(tmp_path, cache_dir=shared, engine="ftw", **common)
        m_da = hpc_tiles.write_tile_manifest(tiles, da, tmp_path / "da")
        m_ftw = hpc_tiles.load_manifest(hpc_tiles.write_tile_manifest(tiles, ftw, tmp_path / "f"))
        hpc_tiles.run_tile(m_da, 0, stage="composite")
        # The composite is shared, but FTW's windows were never built.
        assert hpc_tiles.tile_status(m_ftw).set_index("index").loc[0, "composite"] == "pending"
        hpc_tiles.run_tile(m_ftw, 0, stage="composite")
        assert len(fake_ftw.window_calls) == 1
        assert hpc_tiles.tile_status(m_ftw).set_index("index").loc[0, "composite"] == "done"

    def test_ensemble_ftw_member_windows_are_staged(self, tmp_path, fake_pipeline, fake_ftw):
        """FTW members of an ensemble get their windows staged in their own cache."""
        engines = [
            "delineate-anything",
            {"engine": "ftw", "engine_params": {"window_days": 15}, "label": "ftw a"},
        ]
        m = _ftw_manifest(
            tmp_path, name="ens", engine="ensemble", engine_params={"engines": engines}
        )
        cfg = hpc_tiles.load_tile_config(m, 0)
        assert hpc_tiles._engine_staging(cfg) == "ensemble-inputs"
        assert hpc_tiles.tile_status(m).set_index("index").loc[0, "composite"] == "pending"
        result = hpc_tiles.run_tile(m, 0, stage="composite")
        assert result["status"] == "done"
        assert len(fake_ftw.window_calls) == 1
        raster, params = fake_ftw.window_calls[0]
        assert raster == result["raster_path"] and params == {"window_days": 15}
        marker = hpc_tiles._staged_marker(cfg)["engine_inputs"]
        assert marker["kind"] == "ensemble-inputs"
        member_dir = Path(cfg.get_working_dir()) / "ensemble" / "ftw_a"
        assert marker["rasters"] == [
            str(member_dir / "window_a.tif"),
            str(member_dir / "window_b.tif"),
        ]
        assert set(marker["record"]["members"]) == {"ftw a"}
        assert hpc_tiles.run_tile(m, 0, stage="delineate")["status"] == "done"
        # Removing a member's window un-stages the tile.
        (member_dir / "window_b.tif").unlink()
        assert hpc_tiles.tile_status(m).set_index("index").loc[0, "composite"] == "pending"

    def test_ensemble_member_staging_failure_with_skip(
        self, tmp_path, fake_pipeline, fake_ftw, monkeypatch
    ):
        """All members failing to stage with on_member_error='skip' still stages the tile."""
        import agribound.engines.ftw as ftw

        def failing_windows(config, raster_path, params, rgbn):
            fake_ftw.window_calls.append((raster_path, dict(params)))
            raise RuntimeError("Earth Engine quota exceeded")

        monkeypatch.setattr(ftw.FTWEngine, "_two_windows", staticmethod(failing_windows))
        engines = ["delineate-anything", {"engine": "ftw", "label": "ftw a"}]
        m = _ftw_manifest(
            tmp_path,
            name="ens_skip",
            engine="ensemble",
            engine_params={"engines": engines, "on_member_error": "skip"},
        )
        result = hpc_tiles.run_tile(m, 0, stage="composite")
        assert result["status"] == "done"
        cfg = hpc_tiles.load_tile_config(m, 0)
        marker = hpc_tiles._staged_marker(cfg)["engine_inputs"]
        assert marker["kind"] == "ensemble-inputs" and marker["rasters"] == []
        failed = marker["record"]["failed_members"]
        assert [f["label"] for f in failed] == ["ftw a"]
        assert "quota exceeded" in failed[0]["error"]
        assert hpc_tiles.tile_status(m).set_index("index").loc[0, "composite"] == "done"
        # Staged: a second composite run is skipped and delineation proceeds.
        assert hpc_tiles.run_tile(m, 0, stage="composite")["status"] == "skipped"
        assert len(fake_ftw.window_calls) == 1
        assert hpc_tiles.run_tile(m, 0, stage="delineate")["status"] == "done"

    def test_ensemble_member_staging_failure_raises_by_default(
        self, tmp_path, fake_pipeline, fake_ftw, monkeypatch
    ):
        import agribound.engines.ftw as ftw

        def failing_windows(config, raster_path, params, rgbn):
            raise RuntimeError("Earth Engine quota exceeded")

        monkeypatch.setattr(ftw.FTWEngine, "_two_windows", staticmethod(failing_windows))
        engines = ["delineate-anything", {"engine": "ftw", "label": "ftw a"}]
        m = _ftw_manifest(
            tmp_path, name="ens_raise", engine="ensemble", engine_params={"engines": engines}
        )
        with pytest.raises(RuntimeError, match="quota exceeded"):
            hpc_tiles.run_tile(m, 0, stage="composite")
        assert hpc_tiles.tile_status(m).set_index("index").loc[0, "composite"] != "done"

    def test_ensemble_without_two_window_ftw_needs_no_staging(self, tmp_path, fake_ftw):
        cfg = _base_config(
            tmp_path,
            source="sentinel2",
            local_tif_path=None,
            gee_project="test-project",
            engine="ensemble",
            engine_params={"engines": ["delineate-anything", "geoai"]},
        )
        assert hpc_tiles._engine_staging(cfg) is None
        fake_ftw.n_windows = 1
        assert hpc_tiles._engine_staging(cfg.merged(engine_params={})) is None
        fake_ftw.n_windows = 2
        hpc_tiles._ftw_n_windows.cache_clear()
        assert hpc_tiles._engine_staging(cfg.merged(engine_params={})) == "ensemble-inputs"

    def test_all_stage_builds_windows_in_composite_part(self, tmp_path, fake_pipeline, fake_ftw):
        # Stage "all" = the composite stage (composite + FTW windows) then delineate().
        m = _ftw_manifest(tmp_path)
        result = hpc_tiles.run_tile(m, 0, stage="all")
        assert result["status"] == "done"
        assert len(fake_ftw.window_calls) == 1 and fake_pipeline.composite_calls == 1
        status = hpc_tiles.tile_status(m).set_index("index")
        assert status.loc[0, "composite"] == "done" and status.loc[0, "delineate"] == "done"
        # A later offline --stage delineate finds the output current.
        assert hpc_tiles.run_tile(m, 0, stage="delineate")["status"] == "skipped"
        assert hpc_tiles.run_tile(m, 0, stage="all")["status"] == "skipped"
        assert len(fake_ftw.window_calls) == 1 and fake_pipeline.delineate_calls == 1

    def test_non_gee_source_needs_no_windows(self, tmp_path):
        cfg = _base_config(tmp_path, engine="ftw")
        assert cfg.source == "local"
        assert hpc_tiles._engine_staging(cfg) is None


NO_DATA_MESSAGE = (
    "No sentinel2 images (scene cloud cover <= 20%) intersect the study-area extent for "
    "2023-01-01 to 2024-01-01 (end exclusive). Years with images over the study-area extent: none."
)


class TestNoDataPatterns:
    def test_real_builder_messages_match(self, tmp_path):
        from agribound.composites import gee, local
        from agribound.composites.base import NoDataError

        cfg = _base_config(tmp_path, source="naip", year=2022, local_tif_path=None, gee_project="p")
        error = gee._years_error(cfg, [2019], ("2021-01-01", "2024-01-01"))
        assert isinstance(error, NoDataError) and isinstance(error, ValueError)
        assert hpc_tiles.no_data_reason(error) == f"NoDataError: {error}"
        with pytest.raises(NoDataError) as info:
            local._check_coverage(0.0, "The TESSERA v1 2023 embedding", "advice")
        assert hpc_tiles.no_data_reason(info.value) is not None

    def test_no_data_error_type_is_checked_before_patterns(self):
        """A NoDataError counts whatever its message; patterns are only a fallback."""
        from agribound.composites.base import NoDataError

        assert hpc_tiles.no_data_reason(NoDataError("reworded message")) == (
            "NoDataError: reworded message"
        )
        try:
            try:
                raise NoDataError("window B: nothing here")
            except NoDataError as inner:
                raise RuntimeError("FTW window B (2024-02-01 to 2024-03-31): none") from inner
        except RuntimeError as outer:
            assert hpc_tiles.no_data_reason(outer) == "NoDataError: window B: nothing here"
        # The type wins over an earlier pattern match in the chain.
        try:
            try:
                raise NoDataError("typed")
            except NoDataError as inner:
                raise ValueError(NO_DATA_MESSAGE) from inner
        except ValueError as outer:
            assert hpc_tiles.no_data_reason(outer) == "NoDataError: typed"

    @pytest.mark.parametrize(
        "message",
        [
            NO_DATA_MESSAGE,
            "No GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL image intersects the study-area extent for "
            "2023. Years with images over the study-area extent: none.",
            "The sentinel2 2023 composite has no valid pixel inside the study area although 3 "
            "image(s) matched the filters. advice",
            "TESSERA v1 (vultr) returned no data for 2023. advice",
            "USGS NAIP Plus has no imagery for 2022 in state 'CA' over the study-area extent. "
            "Years available for this area: none.",
            "The USGS NAIP Plus export for 2022 has no imagery inside the study area (LockRaster "
            "IDs [1]). Use source='naip' (Earth Engine).",
            "The Source Cooperative mirror has no Google Satellite Embedding tile for 2023 over "
            "the study-area extent. Years with tiles there: none.",
            "The Source Cooperative Google Satellite Embedding tiles for 2023 do not overlap the "
            "study-area extent.",
            "Could not crop a.tif to the study area 'bbox:1,2,3,4': the study-area extent does "
            "not overlap a.tif. Check that they overlap.",
        ],
    )
    def test_no_data_messages(self, message):
        assert hpc_tiles.no_data_reason(ValueError(message)) == f"ValueError: {message}"

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("local_tif_path must be set when source='local'"),
            ValueError("TESSERA v1 (vultr) has no 2016 layer; available years: [2017]."),
            RuntimeError(NO_DATA_MESSAGE),  # only ValueErrors count
        ],
    )
    def test_other_errors_do_not_match(self, exc):
        assert hpc_tiles.no_data_reason(exc) is None

    def test_chained_value_error_matches(self):
        try:
            try:
                raise ValueError(NO_DATA_MESSAGE)
            except ValueError as inner:
                raise RuntimeError("FTW window A: no sentinel2 composite could be built") from inner
        except RuntimeError as outer:
            assert hpc_tiles.no_data_reason(outer) == f"ValueError: {NO_DATA_MESSAGE}"


class TestNoDataTiles:
    def _patch_no_data(self, monkeypatch, fake, tile_ids, exc_factory=None):
        original = fake.build_composite

        def build(config):
            if any(tid in str(config.cache_dir) for tid in tile_ids):
                fake.composite_calls += 1
                raise (exc_factory or (lambda: ValueError(NO_DATA_MESSAGE)))()
            return original(config)

        import agribound.pipeline as pipeline

        monkeypatch.setattr(pipeline, "build_composite", build)

    def test_composite_stage_records_no_data(self, tmp_path, monkeypatch, fake_pipeline):
        m = _make_manifest(tmp_path)
        empty = m["tiles"][1]["tile_id"]
        self._patch_no_data(monkeypatch, fake_pipeline, [empty])
        result = hpc_tiles.run_tile(m, 1, stage="composite")
        assert (
            result["status"] == "no-data" and "intersect the study-area extent" in result["reason"]
        )
        tile_dir = Path(m["_root"]) / m["tiles"][1]["dir"]
        assert (tile_dir / "no_data.json").exists()
        assert not (tile_dir / "composite.failed.json").exists()
        # Recorded once: later runs do not call the builder again.
        calls = fake_pipeline.composite_calls
        assert hpc_tiles.run_tile(m, 1, stage="composite")["status"] == "no-data"
        assert hpc_tiles.run_tile(m, 1, stage="delineate")["status"] == "no-data"
        assert fake_pipeline.composite_calls == calls and fake_pipeline.delineate_calls == 0
        status = hpc_tiles.tile_status(m).set_index("index")
        assert status.loc[1, "composite"] == "no-data" and status.loc[1, "delineate"] == "no-data"
        assert "intersect" in status.loc[1, "no_data_reason"] and status.loc[1, "error"] is None
        # --overwrite retries the download.
        assert hpc_tiles.run_tile(m, 1, stage="composite", overwrite=True)["status"] == "no-data"
        assert fake_pipeline.composite_calls == calls + 1

    def test_other_value_error_fails_the_tile(self, tmp_path, monkeypatch, fake_pipeline):
        m = _make_manifest(tmp_path)
        tid = m["tiles"][0]["tile_id"]
        self._patch_no_data(
            monkeypatch, fake_pipeline, [tid], lambda: ValueError("export_crs is invalid")
        )
        with pytest.raises(ValueError, match="export_crs"):
            hpc_tiles.run_tile(m, 0, stage="composite")
        status = hpc_tiles.tile_status(m).set_index("index")
        assert status.loc[0, "composite"] == "failed" and status.loc[0, "delineate"] == "pending"

    def test_stage_all_no_data_and_merge(self, tmp_path, monkeypatch, fake_pipeline):
        from agribound.cli import main

        m = _make_manifest(tmp_path)
        n = len(m["tiles"])
        empty = [m["tiles"][0]["tile_id"], m["tiles"][2]["tile_id"]]
        self._patch_no_data(monkeypatch, fake_pipeline, empty)
        fake_pipeline.polygons = []
        results = [hpc_tiles.run_tile(m, i, stage="all") for i in range(n)]
        assert [r["status"] for r in results].count("no-data") == 2
        runner = CliRunner()
        res = runner.invoke(
            main, ["tiles", "status", "--manifest", m["_root"], "--list", "no-data"]
        )
        assert res.stdout.strip() == "0,2"
        res = runner.invoke(
            main, ["tiles", "status", "--manifest", m["_root"], "--list", "not-done"]
        )
        assert res.stdout.strip() == ""
        merged = hpc_tiles.merge_tiles(m)
        summary = merged.attrs["merge_summary"]
        assert summary["n_no_data_tiles"] == 2 and summary["missing_tiles"] == []
        assert [t["tile_id"] for t in summary["no_data_tiles"]] == empty
        assert all("intersect" in t["reason"] for t in summary["no_data_tiles"])
        cores = {e["tile_id"]: e["core_area_km2"] for e in m["tiles"]}
        assert summary["no_data_core_area_km2"] == pytest.approx(
            sum(cores[t] for t in empty), abs=1e-3
        )
        assert any("no input data" in w for w in summary["warnings"])
        res = runner.invoke(main, ["tiles", "merge", "--manifest", m["_root"], "--dry-run"])
        info = json.loads(res.stdout)
        assert info["n_no_data"] == 2 and info["not_done_indices"] == ""
        assert info["output_exists"] is True and info["output_missing_tiles"] == 0

    def test_all_tiles_no_data_merge_raises(self, tmp_path, monkeypatch, fake_pipeline):
        m = _make_manifest(tmp_path)
        self._patch_no_data(monkeypatch, fake_pipeline, [e["tile_id"] for e in m["tiles"]])
        for i in range(len(m["tiles"])):
            assert hpc_tiles.run_tile(m, i, stage="composite")["status"] == "no-data"
        with pytest.raises(RuntimeError, match="No tile produced output"):
            hpc_tiles.merge_tiles(m)

    def test_cli_run_exits_zero_for_no_data(self, tmp_path, monkeypatch, fake_pipeline):
        from agribound.cli import main

        m = _make_manifest(tmp_path)
        self._patch_no_data(monkeypatch, fake_pipeline, [m["tiles"][0]["tile_id"]])
        res = CliRunner().invoke(
            main, ["tiles", "run", "--manifest", m["_root"], "--index", "0", "--stage", "composite"]
        )
        assert res.exit_code == 0, res.output
        assert json.loads(res.stdout.strip().splitlines()[-1])["status"] == "no-data"

    def test_ftw_window_no_data_is_engine_specific(
        self, tmp_path, monkeypatch, fake_pipeline, fake_ftw
    ):
        shared = str(tmp_path / "cache")
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        common = {"source": "sentinel2", "local_tif_path": None, "gee_project": "test-project"}
        da = _base_config(tmp_path, cache_dir=shared, **common)
        ftw = _base_config(tmp_path, cache_dir=shared, engine="ftw", **common)
        m_da = hpc_tiles.load_manifest(hpc_tiles.write_tile_manifest(tiles, da, tmp_path / "da"))
        m_ftw = hpc_tiles.load_manifest(hpc_tiles.write_tile_manifest(tiles, ftw, tmp_path / "f"))

        def windows_without_imagery(config, raster_path, params, rgbn):
            try:
                raise ValueError(NO_DATA_MESSAGE)
            except ValueError as exc:
                raise RuntimeError("FTW window A (2023-09-01 to 2023-10-31): none") from exc

        import agribound.engines.ftw as ftw_module

        monkeypatch.setattr(
            ftw_module.FTWEngine, "_two_windows", staticmethod(windows_without_imagery)
        )
        assert hpc_tiles.run_tile(m_ftw, 0, stage="composite")["status"] == "no-data"
        record = json.loads(
            (Path(m_ftw["_root"]) / m_ftw["tiles"][0]["dir"] / "no_data.json").read_text()
        )
        assert record["level"] == "engine"
        # The composite itself was built and is shared: Delineate-Anything is not affected.
        assert hpc_tiles.tile_status(m_da).set_index("index").loc[0, "composite"] == "done"
        assert hpc_tiles.run_tile(m_da, 0, stage="delineate")["status"] == "done"


class TestLulcStaging:
    @pytest.fixture
    def fake_lulc(self, monkeypatch):
        import agribound.postprocess.lulc_filter as lulc

        state = {"calls": 0, "fail": False}

        def prefetch(config):
            state["calls"] += 1
            if state["fail"]:
                raise RuntimeError("LULC raster prefetch failed: HttpError 503")
            path = Path(config.get_working_dir()) / f"lulc_{config.lulc_dataset}.tif"
            path.write_bytes(b"lulc")
            return str(path)

        monkeypatch.setattr(lulc, "prefetch_lulc_raster", prefetch)
        return state

    def _manifest(self, tmp_path, name="run", **overrides):
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        params = {"lulc_filter": True, "lulc_mode": "raster", "lulc_on_error": "warn"}
        params.update(overrides)
        base = _base_config(tmp_path, **params)
        return hpc_tiles.load_manifest(hpc_tiles.write_tile_manifest(tiles, base, tmp_path / name))

    def test_lulc_raster_staged(self, tmp_path, monkeypatch, fake_pipeline, fake_lulc):
        seen = []
        original = fake_pipeline.build_composite

        def build(config):
            seen.append(config.lulc_filter)
            return original(config)

        import agribound.pipeline as pipeline

        monkeypatch.setattr(pipeline, "build_composite", build)
        m = self._manifest(tmp_path)
        assert hpc_tiles.run_tile(m, 0, stage="composite")["status"] == "done"
        # The composite is built without the LULC step; the stage downloads the raster itself.
        assert seen == [False] and fake_lulc["calls"] == 1
        done = json.loads(
            (Path(m["_root"]) / m["tiles"][0]["dir"] / "composite.done.json").read_text()
        )
        assert Path(done["lulc_raster_path"]).exists()
        assert hpc_tiles.tile_status(m).set_index("index").loc[0, "composite"] == "done"
        assert hpc_tiles.run_tile(m, 0, stage="delineate")["status"] == "done"

    def test_failed_lulc_prefetch_fails_the_stage_despite_warn(
        self, tmp_path, fake_pipeline, fake_lulc
    ):
        fake_lulc["fail"] = True
        m = self._manifest(tmp_path)
        with pytest.raises(RuntimeError, match="LULC raster prefetch failed"):
            hpc_tiles.run_tile(m, 0, stage="composite")
        status = hpc_tiles.tile_status(m).set_index("index")
        assert status.loc[0, "composite"] == "failed"
        assert "whatever lulc_on_error" in status.loc[0, "error"]
        with pytest.raises(RuntimeError, match="not been staged"):
            hpc_tiles.run_tile(m, 0, stage="delineate")
        # Re-staging retries the LULC download; the composite comes from the cache.
        fake_lulc["fail"] = False
        assert hpc_tiles.run_tile(m, 0, stage="composite")["status"] == "done"
        assert hpc_tiles.tile_status(m).set_index("index").loc[0, "composite"] == "done"

    def test_lulc_marker_depends_on_dataset(self, tmp_path, fake_pipeline, fake_lulc):
        shared = str(tmp_path / "cache")
        a = self._manifest(tmp_path, "a", cache_dir=shared, lulc_dataset="dynamic_world")
        b = self._manifest(tmp_path, "b", cache_dir=shared, lulc_dataset="c3s")
        hpc_tiles.run_tile(a, 0, stage="composite")
        # Same composite, other LULC dataset: not staged until its own raster exists.
        assert hpc_tiles.tile_status(b).set_index("index").loc[0, "composite"] == "pending"
        assert hpc_tiles.run_tile(b, 0, stage="composite")["status"] == "done"
        assert fake_pipeline.composite_calls == 2 and fake_lulc["calls"] == 2

    def test_stage_all_records_lulc_raster_from_facts(self, tmp_path, monkeypatch, fake_pipeline):
        m = self._manifest(tmp_path)
        cfg = hpc_tiles.load_tile_config(m, 0)
        lulc_path = Path(cfg.get_working_dir()) / "lulc_from_pipeline.tif"
        original = fake_pipeline.delineate

        def delineate_with_lulc(config=None, **kwargs):
            gdf = original(config=config)
            lulc_path.write_bytes(b"lulc")
            out = Path(config.output_path)
            record = read_provenance(out)
            record["facts"]["lulc_raster_path"] = str(lulc_path)
            write_provenance(out, record)
            return gdf

        import agribound.pipeline as pipeline

        monkeypatch.setattr(pipeline, "delineate", delineate_with_lulc)
        assert hpc_tiles.run_tile(m, 0, stage="all")["status"] == "done"
        marker = hpc_tiles._lulc_marker(cfg)
        assert marker is not None and marker["lulc_raster_path"] == str(lulc_path)
        assert hpc_tiles.tile_status(m).set_index("index").loc[0, "composite"] == "done"


class TestManifestPaths:
    def test_cache_root_change_is_refused(self, tmp_path):
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        base = _base_config(tmp_path)
        hpc_tiles.write_tile_manifest(tiles, base, tmp_path / "run", cache_root=tmp_path / "A")
        hpc_tiles.write_tile_manifest(tiles, base, tmp_path / "run", cache_root=tmp_path / "A")
        with pytest.raises(FileExistsError, match="cache root"):
            hpc_tiles.write_tile_manifest(tiles, base, tmp_path / "run", cache_root=tmp_path / "B")
        path = hpc_tiles.write_tile_manifest(
            tiles, base, tmp_path / "run", cache_root=tmp_path / "B", overwrite=True
        )
        m = hpc_tiles.load_manifest(path)
        assert m["cache_root"] == str((tmp_path / "B").resolve())
        assert hpc_tiles.load_tile_config(m, 0).cache_dir.startswith(
            str((tmp_path / "B").resolve())
        )

    def test_relative_paths_become_absolute(self, tmp_path, monkeypatch):
        work = tmp_path / "work"
        (work / "data").mkdir(parents=True)
        (work / "data" / "s2.tif").write_bytes(b"x")
        (work / "ckpt.pt").write_bytes(b"x")
        monkeypatch.chdir(work)
        base = _base_config(
            tmp_path,
            local_tif_path="data/s2.tif",
            embedding_cache_dir="emb_cache",
            engine_params={"checkpoint_path": "ckpt.pt", "weights_path": "org/model-name"},
        )
        tiles = hpc_tiles.make_tiles(SEAM_BBOX, tile_size_m=20000, halo_m=1000)
        m = hpc_tiles.load_manifest(hpc_tiles.write_tile_manifest(tiles, base, tmp_path / "run"))
        monkeypatch.chdir(tmp_path)  # tile jobs start elsewhere
        cfg = hpc_tiles.load_tile_config(m, 0)
        assert cfg.local_tif_path == str((work / "data" / "s2.tif").resolve())
        assert Path(cfg.local_tif_path).exists()
        assert cfg.embedding_cache_dir == str((work / "emb_cache").resolve())
        assert cfg.engine_params["checkpoint_path"] == str((work / "ckpt.pt").resolve())
        # Not an existing file: a model name, kept as given.
        assert cfg.engine_params["weights_path"] == "org/model-name"


class TestMergeReference:
    def test_reference_restricted_to_study_area_without_clip(self, tmp_path, monkeypatch):
        import importlib

        aoi = _utm_box_aoi(32613, 600000.0, 3600000.0, 30000.0, 10000.0)
        tiles = hpc_tiles.make_tiles(aoi, tile_size_m=20000, halo_m=1000, clip=False)
        m = hpc_tiles.load_manifest(
            hpc_tiles.write_tile_manifest(tiles, _base_config(tmp_path), tmp_path / "run")
        )
        inside = TestMerge._poly(605000, 3605000, 605500, 3605500)
        outside = TestMerge._poly(635000, 3605000, 635500, 3605500)  # in a core, not the AOI
        for entry in m["tiles"]:
            _write_tile_output(m, entry["index"], [inside] if entry["index"] == 0 else [])
        ref = tmp_path / "ref.gpkg"
        gpd.GeoDataFrame(geometry=[inside, outside], crs=4326).to_file(ref)
        seen = {}

        def fake_evaluate(predicted, reference, *args, **kwargs):
            seen["n_ref"] = len(reference)
            return {"precision": 1.0}

        monkeypatch.setattr(
            importlib.import_module("agribound.evaluate"), "evaluate", fake_evaluate
        )
        merged = hpc_tiles.merge_tiles(m, reference=ref)
        assert seen["n_ref"] == 1
        assert merged.attrs["evaluation_metrics"] == {"precision": 1.0}
        assert merged.attrs["merge_summary"]["evaluation_reference"] == {
            "n_reference_total": 2,
            "selection": "intersects study area",
            "n_reference_used": 1,
        }

    def test_reference_selected_by_representative_point_with_clip(self, tmp_path, monkeypatch):
        """With clip=True references follow the predictions' rule: representative point."""
        import importlib

        aoi = _utm_box_aoi(32613, 600000.0, 3600000.0, 30000.0, 10000.0)
        tiles = hpc_tiles.make_tiles(aoi, tile_size_m=20000, halo_m=1000, clip=True)
        m = hpc_tiles.load_manifest(
            hpc_tiles.write_tile_manifest(tiles, _base_config(tmp_path), tmp_path / "run")
        )
        inside = TestMerge._poly(605000, 3605000, 605500, 3605500)
        crossing_in = TestMerge._poly(610000, 3599900, 610400, 3601000)  # point inside
        crossing_out = TestMerge._poly(612000, 3599000, 612400, 3600100)  # point outside
        fields = [inside, crossing_in, crossing_out]
        for entry in m["tiles"]:
            _write_tile_output(m, entry["index"], fields if entry["index"] == 0 else [])
        ref = tmp_path / "ref.gpkg"
        gpd.GeoDataFrame(
            {"name": ["inside", "crossing_in", "crossing_out"]},
            geometry=[inside, crossing_in, crossing_out],
            crs=4326,
        ).to_file(ref)
        seen = {}

        def fake_evaluate(predicted, reference, *args, **kwargs):
            seen["pred"] = len(predicted)
            seen["ref"] = sorted(reference["name"])
            return {"recall": 1.0}

        monkeypatch.setattr(
            importlib.import_module("agribound.evaluate"), "evaluate", fake_evaluate
        )
        merged = hpc_tiles.merge_tiles(m, reference=ref)
        # The same two polygons are kept on both sides.
        assert seen == {"pred": 2, "ref": ["crossing_in", "inside"]}
        assert merged.attrs["merge_summary"]["evaluation_reference"] == {
            "n_reference_total": 3,
            "selection": "representative point in study area",
            "n_reference_used": 2,
        }


class TileStub:
    """Engine stub: one 200 m square at the centre of each tile raster, plus a shared field."""

    def __init__(self, shared_utm):
        self.shared = shared_utm
        self.rasters: list[str] = []

    def delineate(self, raster_path, config):
        import rasterio

        self.rasters.append(raster_path)
        with rasterio.open(raster_path) as src:
            b, crs, width = src.bounds, src.crs, src.width
        cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
        geoms = [box(cx - 100, cy - 100, cx + 100, cy + 100)]
        if box(*b).contains(self.shared):
            geoms.append(self.shared)
        gdf = gpd.GeoDataFrame({"score": [0.9] * len(geoms)}, geometry=geoms, crs=crs)
        gdf.attrs["engine_meta"] = {"backend": "tile-stub", "raster_width": width}
        return gdf


class TestRealPipelineTiles:
    """run_tile and merge_tiles with the real pipeline and local builder (stub engine)."""

    X0, Y0 = 500000.0, 4000000.0  # EPSG:32611

    @pytest.fixture
    def local_raster(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds

        path = tmp_path / "local_rgb.tif"
        width, height = 300, 200  # 6 km x 4 km at 20 m
        data = np.random.default_rng(0).integers(0, 10000, (3, height, width), dtype=np.uint16)
        transform = from_bounds(self.X0, self.Y0, self.X0 + 6000, self.Y0 + 4000, width, height)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype="uint16",
            crs="EPSG:32611",
            transform=transform,
        ) as dst:
            dst.write(data)
        return str(path)

    def _manifest(self, tmp_path, raster, aoi_utm, name="run"):
        aoi = _utm_box_aoi(32611, *aoi_utm)
        tiles = hpc_tiles.make_tiles(aoi, tile_size_m=1500, halo_m=300)
        base = AgriboundConfig(
            source="local",
            local_tif_path=raster,
            engine="delineate-anything",
            output_path=str(tmp_path / "base" / "fields.gpkg"),
            device="cpu",
            lulc_filter=False,
            simplify_tolerance=0,
            engine_params={"smooth_iterations": 0},
        )
        path = hpc_tiles.write_tile_manifest(tiles, base, tmp_path / name)
        return hpc_tiles.load_manifest(path)

    def test_two_phase_tiles_and_merge(self, tmp_path, monkeypatch, local_raster):
        # Study area 3 x 1.5 km inside the raster: 2 x 1 tiles of 1.5 km, 300 m halos.
        m = self._manifest(tmp_path, local_raster, (self.X0 + 1000, self.Y0 + 1000, 3000, 1500))
        grid = m["grid"]["systems"]["11N"]
        boundary_x = grid["origin"][0] + 1500  # core boundary between the two tiles
        mid_y = grid["origin"][1] + 750
        shared = box(boundary_x - 100, mid_y - 50, boundary_x + 150, mid_y + 50)
        stub = TileStub(shared)
        monkeypatch.setattr("agribound.engines.get_engine", lambda name: stub)
        assert len(m["tiles"]) == 2

        staged = [hpc_tiles.run_tile(m, i, stage="composite") for i in range(2)]
        assert [s["status"] for s in staged] == ["done", "done"]
        for s in staged:
            assert Path(s["raster_path"]).exists()
            assert Path(s["raster_path"]).name.startswith("local_crop_")  # cropped per tile
        runs = [hpc_tiles.run_tile(m, i, stage="delineate") for i in range(2)]
        assert [r["status"] for r in runs] == ["done", "done"]
        # The delineation read the staged composite (the facts HPC relies on).
        for i, s in enumerate(staged):
            record = read_provenance(Path(m["_root"]) / m["tiles"][i]["output"])
            assert record["facts"]["raster_path"] == s["raster_path"]
            assert record["facts"]["n_output"] == 2
            assert record["engine_meta"]["backend"] == "tile-stub"
        assert stub.rasters == [s["raster_path"] for s in staged]
        status = hpc_tiles.tile_status(m)
        assert set(status["composite"]) == {"done"} and set(status["delineate"]) == {"done"}

        merged = hpc_tiles.merge_tiles(m)
        # Two tile-centre squares plus the shared field, kept once (east tile owns it).
        assert len(merged) == 3
        summary = merged.attrs["merge_summary"]
        assert summary["n_polygons_in"] == 4 and summary["n_reaching_halo_edge"] == 0
        assert summary["engine_meta"]["backend"] == "tile-stub"
        east = m["tiles"][1]["tile_id"]
        kept_shared = merged.to_crs(32611).geometry.apply(
            lambda g: g.symmetric_difference(shared).area < 1.0
        )
        assert merged.loc[kept_shared, "agribound:tile_id"].tolist() == [east]
        assert hpc_tiles.run_tile(m, 0, stage="all")["status"] == "skipped"

    def test_tile_outside_local_raster_is_no_data(self, tmp_path, monkeypatch, local_raster):
        # 12 km wide study area: tiles east of the 6 km raster have no data.
        m = self._manifest(tmp_path, local_raster, (self.X0 + 1000, self.Y0 + 1000, 10500, 1500))
        stub = TileStub(box(0, 0, 1, 1))
        monkeypatch.setattr("agribound.engines.get_engine", lambda name: stub)
        results = [hpc_tiles.run_tile(m, i, stage="all") for i in range(len(m["tiles"]))]
        statuses = [r["status"] for r in results]
        assert "no-data" in statuses and "done" in statuses
        for r in results:
            if r["status"] == "no-data":
                assert "does not overlap" in r["reason"]
        merged = hpc_tiles.merge_tiles(m)
        summary = merged.attrs["merge_summary"]
        assert summary["n_no_data_tiles"] == statuses.count("no-data")
        assert len(merged) == statuses.count("done")


class TestPrefetchCommand:
    def test_dry_run_and_run(self, tmp_path, monkeypatch):
        from agribound.cli import main

        base = tmp_path / "base.yaml"
        _base_config(tmp_path, sam_refine=True, sam_backend="sam2.1").to_yaml(base)
        runner = CliRunner()
        res = runner.invoke(main, ["tiles", "prefetch", "--config", str(base), "--dry-run"])
        assert res.exit_code == 0, res.output
        plan = json.loads(res.stdout)
        assert plan["engine"] == "delineate-anything"
        assert plan["sam_refine"] is True and plan["sam_backend"] == "sam2.1"

        calls = []

        class FakeEngine:
            def prefetch(self, config):
                calls.append(("engine", config.engine))
                return [str(tmp_path / "da.pt")]

        import agribound.engines as engines
        import agribound.engines.samgeo_engine as samgeo_engine

        monkeypatch.setattr(engines, "get_engine", lambda name: FakeEngine())

        def fake_sam(config):
            calls.append(("sam", config.sam_backend))
            return [str(tmp_path / "sam.pt")]

        monkeypatch.setattr(samgeo_engine, "prefetch", fake_sam)
        res = runner.invoke(main, ["tiles", "prefetch", "--config", str(base)])
        assert res.exit_code == 0, res.output
        assert calls == [("engine", "delineate-anything"), ("sam", "sam2.1")]
        assert "Prefetched 2 file(s)." in res.stdout

    def test_requires_one_source(self, tmp_path):
        from agribound.cli import main

        res = CliRunner().invoke(main, ["tiles", "prefetch"])
        assert res.exit_code != 0 and "exactly one" in res.output


# ---------------------------------------------------------------------------
# merge_tiles
# ---------------------------------------------------------------------------


def _write_tile_output(m: dict, index: int, geoms_4326: list, run_id: str = "r") -> None:
    entry = m["tiles"][index]
    cfg = hpc_tiles.load_tile_config(m, index)
    out = Path(m["_root"]) / entry["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {"src": [entry["tile_id"]] * len(geoms_4326)}, geometry=geoms_4326, crs=4326
    )
    # Store in the tile's UTM CRS, as the pipeline does for UTM composites.
    gdf.to_crs(entry["utm_epsg"]).to_file(out, layer="fields", driver="GPKG")
    write_provenance(
        out,
        {
            "status": "success",
            "config_hash": config_hash(cfg),
            "run_id": f"{run_id}{index}",
            "wall_s": 2.0,
            "facts": {"n_output": len(gdf)},
        },
    )


class TestMerge:
    def _aligned_manifest(self, tmp_path):
        # Two 20 km tiles side by side in UTM 13N, 2 km halo.
        aoi = _utm_box_aoi(32613, 600000.0, 3600000.0, 40000.0, 20000.0)
        tiles = hpc_tiles.make_tiles(aoi, tile_size_m=20000, halo_m=2000)
        path = hpc_tiles.write_tile_manifest(tiles, _base_config(tmp_path), tmp_path / "run")
        return hpc_tiles.load_manifest(path)

    @staticmethod
    def _poly(x0, y0, x1, y1):
        return gpd.GeoSeries([box(x0, y0, x1, y1)], crs=32613).to_crs(4326).iloc[0]

    def test_polygon_in_halo_overlap_kept_once(self, tmp_path):
        m = self._aligned_manifest(tmp_path)
        x = 620000.0  # core boundary between the two tiles
        straddling = self._poly(x - 300, 3605000, x + 500, 3605400)  # rep point east of x
        west_only = self._poly(605000, 3605000, 605500, 3605500)
        east_only = self._poly(630000, 3605000, 630500, 3605500)
        _write_tile_output(m, 0, [straddling, west_only])
        _write_tile_output(m, 1, [straddling, east_only])
        merged = hpc_tiles.merge_tiles(m)
        assert len(merged) == 3
        tile_ids = [e["tile_id"] for e in m["tiles"]]
        # The straddling polygon is kept from the east tile's output only.
        assert sorted(merged["src"]) == sorted([tile_ids[0], tile_ids[1], tile_ids[1]])
        assert (merged["src"] == merged["agribound:tile_id"]).all()
        summary = merged.attrs["merge_summary"]
        assert summary["n_polygons_in"] == 4 and summary["n_polygons_out"] == 3
        assert summary["n_cross_tile_overlap_pairs"] == 0
        assert read_provenance(hpc_tiles.default_merge_output(m))["kind"] == "agribound-tile-merge"

    def test_utm_seam_polygon_kept_once(self, tmp_path):
        m = _make_manifest(tmp_path, tile_size_m=20000, halo_m=2000)
        # Crosses 150 E; its representative point (box centre, 150.001 E) is in zone 56.
        seam = box(149.996, -30.35, 150.006, -30.345)
        owner = hpc_tiles.assign_tile_ids(
            hpc_tiles.representative_points(gpd.GeoSeries([seam], crs=4326)), m["grid"]
        )[0]
        assert owner.startswith("56S_")

        def halo_bbox(entry):
            return box(*[float(v) for v in entry["study_area"][5:].split(",")])

        holders = [e for e in m["tiles"] if halo_bbox(e).contains(seam)]
        assert {e["system"] for e in holders} == {"55S", "56S"}
        for entry in m["tiles"]:
            _write_tile_output(m, entry["index"], [seam] if entry in holders else [])
        merged = hpc_tiles.merge_tiles(m)
        assert len(merged) == 1
        assert merged["agribound:tile_id"].iloc[0] == owner

    def test_rep_point_outside_study_area_dropped_only_with_clip(self, tmp_path):
        # L-shaped study area: 40 x 20 km minus its 10 x 10 km north-east corner.
        full = box(600000.0, 3600000.0, 640000.0, 3620000.0)
        notch = box(630000.0, 3610000.0, 640000.0, 3620000.0)
        shape = shapely.segmentize(full.difference(notch), 500.0)
        aoi = gpd.GeoSeries([shape], crs=32613).to_crs(4326).iloc[0]
        in_notch = self._poly(634000, 3614000, 635000, 3615000)
        results = {}
        for clip in (True, False):
            tiles = hpc_tiles.make_tiles(aoi, tile_size_m=20000, halo_m=2000, clip=clip)
            assert len(tiles) == 2
            path = hpc_tiles.write_tile_manifest(
                tiles, _base_config(tmp_path), tmp_path / f"c{clip}"
            )
            m = hpc_tiles.load_manifest(path)
            _write_tile_output(m, 0, [])
            _write_tile_output(m, 1, [in_notch])
            results[clip] = len(hpc_tiles.merge_tiles(m))
        assert results == {True: 0, False: 1}

    def test_rep_point_outside_grid_dropped(self, tmp_path):
        m = self._aligned_manifest(tmp_path)
        outside = self._poly(600000 - 900, 3605000, 600000 - 100, 3605400)  # in the halo only
        _write_tile_output(m, 0, [outside])
        _write_tile_output(m, 1, [])
        assert len(hpc_tiles.merge_tiles(m)) == 0

    def test_halo_edge_and_overlap_diagnostics(self, tmp_path):
        m = self._aligned_manifest(tmp_path)
        # Tile 0 halo: x 598-622 km; tile 1 halo: x 618-642 km.
        near_dup = self._poly(617700, 3605000, 622100, 3606000)  # centre 619.9 km -> tile 0
        big = self._poly(617800, 3605000, 622500, 3606000)  # centre 620.15 km -> tile 1
        _write_tile_output(m, 0, [near_dup])
        _write_tile_output(m, 1, [big])
        merged = hpc_tiles.merge_tiles(m)
        summary = merged.attrs["merge_summary"]
        # Both detections are kept (different representative points) ...
        assert len(merged) == 2
        # ... both extend past their own halo box ...
        assert summary["n_reaching_halo_edge"] == 2
        # ... and the pair is reported as a cross-tile overlap.
        assert summary["n_cross_tile_overlap_pairs"] == 1
        assert any("halo" in w for w in summary["warnings"])

    def test_missing_tiles_and_reuse(self, tmp_path):
        m = self._aligned_manifest(tmp_path)
        _write_tile_output(m, 0, [self._poly(605000, 3605000, 605500, 3605500)])
        with pytest.raises(RuntimeError, match="not done"):
            hpc_tiles.merge_tiles(m)
        out = tmp_path / "merged.gpkg"
        merged = hpc_tiles.merge_tiles(m, out, allow_missing=True)
        assert len(merged) == 1
        assert merged.attrs["merge_summary"]["missing_tiles"][0]["index"] == 1
        again = hpc_tiles.merge_tiles(m, out, allow_missing=True)
        assert again.attrs.get("reused") is True
        _write_tile_output(m, 1, [], run_id="new")
        with pytest.raises(FileExistsError):
            hpc_tiles.merge_tiles(m, out)
        assert len(hpc_tiles.merge_tiles(m, out, overwrite=True)) == 1

    def test_engine_meta_common_and_varying(self, tmp_path):
        m = self._aligned_manifest(tmp_path)
        _write_tile_output(m, 0, [self._poly(605000, 3605000, 605500, 3605500)])
        _write_tile_output(m, 1, [])
        metas = [
            {"weights_sha256": "abc", "n_raw": 12, "stretch_lows_bgr": [1, 2, 3]},
            {"weights_sha256": "abc", "n_raw": 7, "stretch_lows_bgr": [1, 2, 4], "extra": 1},
        ]
        for entry, meta in zip(m["tiles"], metas, strict=True):
            out = Path(m["_root"]) / entry["output"]
            record = read_provenance(out)
            record["engine_meta"] = meta
            write_provenance(out, record)
        summary = hpc_tiles.merge_tiles(m).attrs["merge_summary"]
        assert summary["engine_meta"] == {"weights_sha256": "abc"}
        assert summary["engine_meta_varying_keys"] == ["extra", "n_raw", "stretch_lows_bgr"]
        assert summary["engine_meta_consistent"] is False

    def test_output_crs(self, tmp_path):
        m = self._aligned_manifest(tmp_path)
        _write_tile_output(m, 0, [self._poly(605000, 3605000, 605500, 3605500)])
        _write_tile_output(m, 1, [])
        merged = hpc_tiles.merge_tiles(m, tmp_path / "m.gpkg", crs="EPSG:6933")
        assert merged.crs.equals("EPSG:6933")
        assert gpd.read_file(tmp_path / "m.gpkg").crs.equals("EPSG:6933")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def _base_yaml(self, tmp_path) -> Path:
        path = tmp_path / "base.yaml"
        _base_config(tmp_path).to_yaml(path)
        return path

    def test_registered_on_main(self):
        from agribound.cli import main

        assert "tiles" in main.commands

    def test_make_status_run_merge(self, tmp_path, fake_pipeline):
        from agribound.cli import main

        runner = CliRunner()
        base = self._base_yaml(tmp_path)
        out = tmp_path / "run"
        res = runner.invoke(
            main,
            [
                "tiles",
                "make",
                "--config",
                str(base),
                "--out-dir",
                str(out),
                "--tile-size-m",
                "20000",
                "--dry-run",
            ],
        )
        assert res.exit_code == 0, res.output
        n = json.loads(res.stdout)["n_tiles"]
        assert not out.exists()
        res = runner.invoke(
            main,
            [
                "tiles",
                "make",
                "--config",
                str(base),
                "--out-dir",
                str(out),
                "--tile-size-m",
                "20000",
            ],
        )
        assert res.exit_code == 0, res.output
        res = runner.invoke(main, ["tiles", "status", "--manifest", str(out), "--list", "not-done"])
        assert res.stdout.strip() == f"0-{n - 1}"
        res = runner.invoke(
            main, ["tiles", "run", "--manifest", str(out), "--index", "0", "--dry-run"]
        )
        assert json.loads(res.stdout)["delineate"] == "pending"
        env = {"SLURM_ARRAY_TASK_ID": "0", "AGB_INDEX_OFFSET": "1"}
        res = runner.invoke(main, ["tiles", "run", "--manifest", str(out)], env=env)
        assert res.exit_code == 0, res.output
        assert json.loads(res.stdout.strip().splitlines()[-1])["index"] == 1
        res = runner.invoke(main, ["tiles", "status", "--manifest", str(out), "--list", "done"])
        assert res.stdout.strip() == "1"
        res = runner.invoke(main, ["tiles", "merge", "--manifest", str(out), "--dry-run"])
        assert json.loads(res.stdout)["n_done"] == 1
        res = runner.invoke(main, ["tiles", "merge", "--manifest", str(out)])
        assert res.exit_code != 0 and "not done" in res.output

    def test_run_index_out_of_range(self, tmp_path):
        from agribound.cli import main

        runner = CliRunner()
        m = _make_manifest(tmp_path)
        res = runner.invoke(main, ["tiles", "run", "--manifest", m["_root"], "--index", "999"])
        assert res.exit_code != 0 and "out of range" in res.output

    @needs_posix_bash
    def test_region_shell_output(self, tmp_path):
        from agribound.cli import main

        region = tmp_path / "demo.yaml"
        region.write_text(
            yaml.safe_dump(
                {
                    "name": "demo",
                    "title": "Demo region's title",
                    "bbox": [1.4, 48.1, 1.55, 48.2],
                    "test_bbox": [1.4, 48.1, 1.45, 48.12],
                    "run": {
                        "years": [2023],
                        "sources": ["sentinel2"],
                        "engines": ["ftw"],
                        "tile_size_km": 20,
                        "halo_m": 1500,
                    },
                }
            )
        )
        res = CliRunner().invoke(main, ["tiles", "region", "--region", str(region)])
        assert res.exit_code == 0, res.output
        env = subprocess.run(
            [
                "bash",
                "-c",
                f'{res.stdout}\necho "$AGB_REGION_TITLE|'
                '$AGB_REGION_STUDY_AREA|$AGB_REGION_ENGINES"',
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert env == "Demo region's title|bbox:1.4,48.1,1.55,48.2|ftw"
        res = CliRunner().invoke(main, ["tiles", "region", "--region", str(region), "--test"])
        assert "AGB_REGION_STUDY_AREA=bbox:1.4,48.1,1.45,48.12" in res.stdout


# ---------------------------------------------------------------------------
# Region files and example scripts
# ---------------------------------------------------------------------------

REGION_FILES = sorted((EXAMPLES / "regions").glob("*.yaml"))
SHELL_SCRIPTS = sorted(
    [
        *(EXAMPLES / "hpc").glob("*.sh"),
        *(EXAMPLES / "hpc").glob("*.sbatch"),
        *(EXAMPLES / "regions").glob("*.sh"),
        *(EXAMPLES / "hpc" / "profiles").glob("*.env"),
        EXAMPLES / "run_region_delineation.sh",
        EXAMPLES / "run_namoi_delineation.sh",
    ]
)


@pytest.mark.parametrize("path", REGION_FILES, ids=lambda p: p.stem)
def test_region_files_are_consistent(path):
    from agribound.registry import ENGINE_REGISTRY, engine_supports_source, source_year_range

    region = hpc_regions.load_region(path)
    assert region["name"] == path.stem
    run = region["run"]
    for engine in run["engines"]:
        assert engine in ENGINE_REGISTRY
        assert any(engine_supports_source(engine, s) for s in run["sources"]), engine
    # Every listed year, source and engine takes part in at least one run (the driver skips
    # the other combinations, e.g. NAIP after 2023).
    plan = hpc_regions.plan_runs(
        run["years"], run["sources"], run["engines"], tessera_version=run.get("tessera_version")
    )
    runs = [p for p in plan if p["action"] == "run"]
    for key in ("year", "source", "engine"):
        listed = run[f"{key}s"]
        assert {p[key] for p in runs} == {int(v) if key == "year" else v for v in listed}, key
    for source in run["sources"]:
        yr = source_year_range(source, tessera_version=run.get("tessera_version"))
        assert yr is None or any(yr[0] <= int(y) <= (yr[1] or 9999) for y in run["years"])
    frac = region["verification"]["worldcover_v200_cropland_fraction"]
    assert 0.0 < float(frac) <= 1.0
    assert len(region["test_bbox"]) == 4
    # Test boxes taken from example scripts may stick out slightly (documented in
    # test_bbox_source); most of each test box lies inside the region.
    tb = box(*region["test_bbox"])
    assert box(*region["bbox"]).intersection(tb).area >= 0.5 * tb.area
    assert region["tiling"]["n_tiles"] > 0
    ref = run.get("reference")
    assert ref is None or Path(ref).is_absolute()


def test_namoi_region_bbox_is_polygon_extent():
    ref = EXAMPLES / "namoi_polygons.geojson"
    if not ref.exists():
        pytest.skip("examples/namoi_polygons.geojson is a local (gitignored) file")
    region = hpc_regions.load_region(EXAMPLES / "regions" / "namoi_catchment_au.yaml")
    minx, miny, maxx, maxy = gpd.read_file(ref).to_crs(4326).total_bounds
    bx = region["bbox"]
    # Rounded outwards to 1e-4 degrees.
    assert bx[0] <= minx < bx[0] + 1e-4 and bx[1] <= miny < bx[1] + 1e-4
    assert bx[2] - 1e-4 < maxx <= bx[2] and bx[3] - 1e-4 < maxy <= bx[3]
    assert region["run"]["reference"] == str(ref.resolve())


def test_every_region_has_a_wrapper_script():
    wrappers = {p.name for p in (EXAMPLES / "regions").glob("run_*.sh")}
    assert REGION_FILES
    for path in REGION_FILES:
        assert f"run_{path.stem}.sh" in wrappers, path.stem


class TestPlanRuns:
    def test_skip_reasons(self):
        plan = hpc_regions.plan_runs(
            [2016, 2024],
            ["sentinel2", "naip", "tessera-embedding", "spot"],
            ["delineate-anything", "embedding", "dinov3", "prithvi"],
            tessera_version="v1.1",
        )
        by_key = {(p["year"], p["source"], p["engine"]): p for p in plan}
        assert by_key[(2024, "sentinel2", "delineate-anything")]["action"] == "run"
        assert "does not support" in by_key[(2024, "sentinel2", "embedding")]["reason"]
        assert "2017-present" in by_key[(2016, "sentinel2", "delineate-anything")]["reason"]
        assert "2002-2023" in by_key[(2024, "naip", "delineate-anything")]["reason"]
        # TESSERA v1.1 starts in 2015, v1 in 2017.
        assert by_key[(2016, "tessera-embedding", "embedding")]["action"] == "run"
        assert "restricted" in by_key[(2016, "spot", "delineate-anything")]["reason"]
        assert "label-free" in by_key[(2024, "sentinel2", "dinov3")]["reason"]
        assert by_key[(2024, "sentinel2", "prithvi")]["note"] == "gfm-env"

    def test_fine_tune_and_checkpoint(self):
        ft = hpc_regions.plan_runs(
            [2024], ["sentinel2"], ["dinov3", "delineate-anything"], fine_tune=True
        )
        assert [(p["engine"], p["action"], p["fine_tune"]) for p in ft] == [
            ("dinov3", "run", True),
            ("delineate-anything", "run", False),
        ]
        ck = hpc_regions.plan_runs([2024], ["sentinel2"], ["geoai"], has_checkpoint=True)
        assert ck[0]["action"] == "run" and ck[0]["fine_tune"] is False
        v1 = hpc_regions.plan_runs([2016], ["tessera-embedding"], ["embedding"])
        assert v1[0]["action"] == "skip"

    def test_unknown_names(self):
        with pytest.raises(ValueError, match="Unknown"):
            hpc_regions.plan_runs([2024], ["sentinel3"], ["delineate-anything"])

    def test_matrix_cli(self):
        from agribound.cli import main

        res = CliRunner().invoke(
            main,
            [
                "tiles",
                "matrix",
                "--years",
                "2023,2024",
                "--sources",
                "sentinel2 naip",
                "--engines",
                "ftw,delineate-anything",
            ],
        )
        assert res.exit_code == 0, res.output
        rows = [line.split("\t") for line in res.stdout.strip().splitlines()]
        assert len(rows) == 8 and all(len(r) == 7 for r in rows)
        runs = {(r[1], r[2], r[3]) for r in rows if r[0] == "run"}
        assert ("2024", "naip", "delineate-anything") not in runs
        assert ("2023", "naip", "delineate-anything") in runs
        assert ("2023", "naip", "ftw") not in runs


@needs_posix_bash
@pytest.mark.parametrize("path", [p for p in SHELL_SCRIPTS if p.exists()], ids=lambda p: p.name)
def test_shell_syntax(path):
    subprocess.run(["bash", "-n", str(path)], check=True)
    shellcheck = shutil.which("shellcheck")
    if shellcheck and path.suffix != ".env":
        subprocess.run([shellcheck, "-S", "warning", "-x", str(path)], check=True)


def _agribound_bin() -> str:
    candidate = Path(sys.executable).with_name("agribound")
    if candidate.exists():
        return str(candidate.parent)
    found = shutil.which("agribound")
    if not found:
        pytest.skip("agribound console script not found")
    return str(Path(found).parent)


@needs_posix_bash
def test_submit_region_dry_run(tmp_path):
    script = EXAMPLES / "hpc" / "submit_region.sh"
    if not script.exists():
        pytest.skip("submit_region.sh not present")
    base = tmp_path / "base.yaml"
    _base_config(tmp_path).to_yaml(base)
    env = dict(os.environ)
    env["PATH"] = _agribound_bin() + os.pathsep + env.get("PATH", "")
    env["AGB_GPU_ACCOUNT"] = "abcd-delta-gpu"
    env["AGB_CPU_ACCOUNT"] = "abcd-delta-cpu"
    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--profile",
            "delta",
            "--config",
            str(base),
            "--out-dir",
            str(tmp_path / "out"),
            "--tile-size-km",
            "20",
            "--halo-m",
            "1000",
            "--mode",
            "stage",
            "--gee-max-requests",
            "8",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.startswith("sbatch ")]
    assert len(lines) == 3, proc.stdout
    stage, gpu, merge = lines
    assert "agribound_stage.sbatch" in stage and "%5" in stage  # 40 // 8 concurrent stage tasks
    assert "--partition=gpuA100x4" in gpu and "--dependency=afterok:" in gpu
    assert "--account=abcd-delta-gpu" in gpu
    assert "agribound_merge.sbatch" in merge and "--dependency=afterok:" in merge
    assert all("--export=ALL,AGB_HPC_DIR=" in line for line in lines)
    assert not (tmp_path / "out" / "manifest.json").exists()


def _script_env() -> dict:
    env = dict(os.environ)
    env["PATH"] = _agribound_bin() + os.pathsep + env.get("PATH", "")
    return env


@needs_posix_bash
def test_submit_region_dry_run_tacc_env_export_and_cpu_compute(tmp_path):
    base = tmp_path / "base.yaml"
    _base_config(tmp_path).to_yaml(base)
    cmd = [
        "bash",
        str(EXAMPLES / "hpc" / "submit_region.sh"),
        "--profile",
        "stampede3",
        "--config",
        str(base),
        "--out-dir",
        str(tmp_path / "out"),
        "--tile-size-km",
        "10",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=_script_env())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    jobs = [line for line in proc.stdout.splitlines() if " sbatch --parsable " in line]
    assert len(jobs) == 3
    # TACC guides say to avoid --export: variables go into sbatch's environment instead.
    assert all(line.startswith("env AGB_HPC_DIR=") and "--export" not in line for line in jobs)
    gpu = jobs[1]
    assert "--partition=h100" in gpu and "AGB_PAR=4" in gpu and "--gres" not in gpu
    assert "--array=0-1%2" in gpu  # at most 2 h100 jobs per user

    proc = subprocess.run(
        [*cmd[:3], "delta", *cmd[4:], "--compute", "cpu"],
        capture_output=True,
        text=True,
        env={**_script_env(), "AGB_GPU_ACCOUNT": "g", "AGB_CPU_ACCOUNT": "c"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    compute = [line for line in proc.stdout.splitlines() if "agribound_array.sbatch" in line][0]
    assert "--partition=cpu" in compute and "--account=c" in compute
    assert "AGB_COMPUTE=cpu" in compute and "gpuA100" not in compute


@needs_posix_bash
def test_region_driver_dry_run(tmp_path):
    region = tmp_path / "demo.yaml"
    region.write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "title": "Demo",
                "bbox": [149.8, -30.5, 150.3, -30.2],
                "test_bbox": [149.9, -30.4, 149.95, -30.35],
                "run": {
                    "years": [2023, 2024],
                    "sources": ["local"],
                    "engines": ["delineate-anything", "embedding", "dinov3"],
                    "tile_size_km": 20,
                    "halo_m": 1000,
                },
            }
        )
    )
    driver = str(EXAMPLES / "run_region_delineation.sh")
    common = ["--region-file", str(region), "--out-root", str(tmp_path / "out"), "--dry-run"]
    env = {**_script_env(), "AGB_GPU_ACCOUNT": "g", "AGB_CPU_ACCOUNT": "c"}
    # 'local' needs --local-tif, which the driver does not pass: every run fails validation.
    proc = subprocess.run(
        ["bash", driver, *common, "--no-lulc-filter"], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 1
    assert proc.stdout.count("FAIL  ") == 2 and "local_tif_path is required" in proc.stdout
    assert "SKIP  2023/local__dinov3: dinov3 has no label-free weights" in proc.stdout
    assert "(2 engine/source pairs skipped" in proc.stdout
    assert not (tmp_path / "out").exists()


@needs_posix_bash
def test_region_driver_slurm_dry_run_chains_gee(tmp_path):
    driver = str(EXAMPLES / "run_region_delineation.sh")
    key = tmp_path / "key.json"
    key.write_text("{}")
    proc = subprocess.run(
        [
            "bash",
            driver,
            "--region",
            "beauce_fr",
            "--mode",
            "slurm",
            "--profile",
            "delta",
            "--years",
            "2024",
            "--sources",
            "sentinel2,tessera-embedding",
            "--engines",
            "delineate-anything,embedding",
            "--gee-project",
            "test-project",
            "--gee-service-account-key",
            str(key),
            "--out-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env={**_script_env(), "AGB_GPU_ACCOUNT": "g", "AGB_CPU_ACCOUNT": "c"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    stages = [line for line in proc.stdout.splitlines() if "agribound_stage.sbatch" in line]
    computes = [line for line in proc.stdout.splitlines() if "agribound_array.sbatch" in line]
    assert len(stages) == len(computes) == 2
    # The second submission's stage array waits for the first one's (Earth Engine budget).
    assert "--dependency" not in stages[0]
    assert "--dependency=afterany:DRYRUN-c1-1" in stages[1]
    # The embedding engine runs on the CPU partition.
    assert "--partition=gpuA100x4" in computes[0] and "--partition=cpu" in computes[1]
    assert "using --lulc-mode raster" in proc.stderr
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# submit_region.sh: planning, submit limits, failures (fake Slurm commands)
# ---------------------------------------------------------------------------

SUBMIT = str(EXAMPLES / "hpc" / "submit_region.sh")
DRIVER = str(EXAMPLES / "run_region_delineation.sh")


def _submit(tmp_path, *args, env=None, out="out"):
    base = tmp_path / "base.yaml"
    if not base.exists():
        _base_config(tmp_path).to_yaml(base)
    cmd = ["bash", SUBMIT, "--config", str(base), "--out-dir", str(tmp_path / out), *args]
    full_env = {**_script_env(), "AGB_GPU_ACCOUNT": "g", "AGB_CPU_ACCOUNT": "c", **(env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, env=full_env)


def _sbatch_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if " sbatch --parsable " in f" {line}"]


@pytest.fixture
def fake_slurm(tmp_path):
    """Fake sbatch / scancel / squeue on PATH that log their arguments."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    log = tmp_path / "slurm.log"
    (bindir / "sbatch").write_text(
        "#!/usr/bin/env bash\n"
        'n=$(( $(grep -c "^sbatch" "$SLURM_LOG" 2>/dev/null || true) + 1 ))\n'
        'echo "sbatch $*" >>"$SLURM_LOG"\n'
        'if [[ -n "${FAIL_AT:-}" && "$n" == "$FAIL_AT" ]]; then echo "sbatch: error: '
        'QOSMaxSubmitJobPerUserLimit" >&2; exit 1; fi\n'
        'echo "$((1000 + n));cluster"\n'
    )
    (bindir / "scancel").write_text('#!/usr/bin/env bash\necho "scancel $*" >>"$SLURM_LOG"\n')
    (bindir / "squeue").write_text(
        "#!/usr/bin/env bash\n"
        'echo "squeue $*" >>"$SLURM_LOG"\n'
        'for ((i = 0; i < ${FAKE_QUEUED:-0}; i++)); do echo "$((500 + i))"; done\n'
    )
    for name in ("sbatch", "scancel", "squeue"):
        (bindir / name).chmod(0o755)
    env = {"PATH": f"{bindir}{os.pathsep}{_script_env()['PATH']}", "SLURM_LOG": str(log)}
    return env, log


@needs_posix_bash
def test_submit_limit_refuses_before_any_sbatch(tmp_path):
    ok = _submit(tmp_path, "--profile", "stampede3", "--tile-size-km", "10", "--dry-run")
    assert ok.returncode == 0, ok.stderr
    assert "AGB_PLANNED_GPU_TASKS=2" in ok.stdout
    busy = _submit(
        tmp_path,
        "--profile",
        "stampede3",
        "--tile-size-km",
        "10",
        "--dry-run",
        env={"AGB_ASSUME_QUEUED_GPU": "3"},
    )
    assert busy.returncode == 3
    assert _sbatch_lines(busy.stdout) == []
    assert "limit is 4 per user" in busy.stderr and "Nothing was submitted" in busy.stderr


@needs_posix_bash
def test_pipelined_mismatch_detected_before_any_sbatch(tmp_path):
    # 5 km tiles: more tiles than Expanse's 24 GPU array tasks, so each GPU task runs several
    # tiles while each stage task runs one.
    proc = _submit(
        tmp_path, "--profile", "expanse", "--tile-size-km", "5", "--pipelined", "--dry-run"
    )
    assert proc.returncode != 0
    assert "--pipelined needs the same tiles per task" in proc.stderr
    assert _sbatch_lines(proc.stdout) == []


@needs_posix_bash
def test_gee_budget_exceeded_by_one_task(tmp_path):
    proc = _submit(tmp_path, "--profile", "stampede3", "--gee-max-requests", "16", "--dry-run")
    assert proc.returncode == 1
    assert "5 process(es) x 16 = 80 concurrent Earth Engine requests" in proc.stderr
    assert _sbatch_lines(proc.stdout) == []


@needs_posix_bash
def test_keep_going_and_allow_missing(tmp_path):
    proc = _submit(tmp_path, "--profile", "delta", "--keep-going", "--allow-missing", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    stage, compute, merge = _sbatch_lines(proc.stdout)
    assert "--dependency=afterany:DRYRUN-1" in compute
    assert "--dependency=afterany:DRYRUN-2" in merge
    assert "AGB_MERGE_ALLOW_MISSING=1" in merge
    bad = _submit(tmp_path, "--profile", "delta", "--keep-going", "--pipelined", "--dry-run")
    assert bad.returncode == 1 and "cannot be combined" in bad.stderr


@needs_posix_bash
def test_resume_dry_run_writes_nothing(tmp_path, fake_pipeline):
    from agribound.cli import main

    base = tmp_path / "base.yaml"
    _base_config(tmp_path).to_yaml(base)
    out = tmp_path / "out"
    res = CliRunner().invoke(
        main,
        ["tiles", "make", "--config", str(base), "--out-dir", str(out), "--tile-size-m", "20000"],
    )
    assert res.exit_code == 0, res.output
    hpc_tiles.run_tile(out, 0, stage="all")
    before = sorted(p.name for p in out.iterdir())
    proc = _submit(tmp_path, "--profile", "delta", "--resume", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in out.iterdir()) == before
    stage, compute, merge = _sbatch_lines(proc.stdout)
    n = len(hpc_tiles.load_manifest(out)["tiles"])
    assert f"AGB_N_TILES={n - 1}" in compute and "AGB_TILE_LIST=" in compute
    assert "AGB_MERGE_OVERWRITE=1" in merge


@needs_posix_bash
def test_real_submission_with_fake_sbatch(tmp_path, fake_slurm):
    env, log = fake_slurm
    proc = _submit(tmp_path, "--profile", "delta", env=env)
    assert proc.returncode == 0, proc.stderr
    assert "AGB_STAGE_JOBS=1001" in proc.stdout and "AGB_GPU_JOBS=1002" in proc.stdout
    assert "AGB_MERGE_JOB=1003" in proc.stdout
    calls = log.read_text().splitlines()
    sbatch = [c for c in calls if c.startswith("sbatch")]
    assert len(sbatch) == 3
    assert "--dependency=afterok:1001" in sbatch[1] and "--dependency=afterok:1002" in sbatch[2]
    assert (tmp_path / "out" / "manifest.json").exists()


@needs_posix_bash
def test_failed_sbatch_cancels_earlier_jobs(tmp_path, fake_slurm):
    env, log = fake_slurm
    proc = _submit(tmp_path, "--profile", "delta", env={**env, "FAIL_AT": "2"})
    assert proc.returncode == 1
    assert "sbatch failed" in proc.stderr
    calls = log.read_text().splitlines()
    assert [c for c in calls if c.startswith("scancel")] == ["scancel 1001"]


@needs_posix_bash
def test_squeue_counts_toward_submit_limit(tmp_path, fake_slurm):
    env, log = fake_slurm
    proc = _submit(tmp_path, "--profile", "stampede3", env={**env, "FAKE_QUEUED": "3"})
    assert proc.returncode == 3
    calls = log.read_text().splitlines()
    assert not [c for c in calls if c.startswith("sbatch")]
    assert any(c.startswith("squeue -h -r -u") and "-p h100" in c for c in calls)


# ---------------------------------------------------------------------------
# common.sh helpers
# ---------------------------------------------------------------------------


def _bash(script: str, env: dict | None = None) -> subprocess.CompletedProcess:
    common = EXAMPLES / "hpc" / "common.sh"
    return subprocess.run(
        ["bash", "-c", f"source {common}\n{script}"],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


@needs_posix_bash
def test_gee_concurrency_helper():
    assert _bash("agb_gee_concurrency 40 8 1").stdout.strip() == "5"
    assert _bash("agb_gee_concurrency 40 8 5").stdout.strip() == "1"
    over = _bash("agb_gee_concurrency 40 16 5")
    assert over.returncode == 1 and "more than the budget of 40" in over.stderr
    assert _bash("agb_gee_concurrency 40 0 1").returncode == 1


@needs_posix_bash
def test_gpu_mapping_uses_inherited_devices(tmp_path):
    script = """
# args: tiles run --manifest M --index I ...
agribound() { echo "$6 ${CUDA_VISIBLE_DEVICES:-unset}" >>"$OUT"; }
AGB_MANIFEST=m AGB_N_TILES=2 AGB_LOG_DIR="$LOGS"
agb_run_task_tiles 0 2 1 "$STAGE"
"""
    for devices, stage, expected in [
        ("2,3", "delineate", {"0 2", "1 3"}),
        ("", "delineate", {"0 0", "1 1"}),
        ("2,3", "composite", {"0 2,3", "1 2,3"}),
    ]:
        out = tmp_path / f"out_{stage}_{devices or 'none'}.txt"
        env = {"OUT": str(out), "LOGS": str(tmp_path), "STAGE": stage}
        if devices:
            env["CUDA_VISIBLE_DEVICES"] = devices
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        proc = _bash(script, env)
        assert proc.returncode == 0, proc.stderr
        assert set(out.read_text().splitlines()) == expected
    short = _bash(
        script,
        {
            "OUT": str(tmp_path / "x"),
            "LOGS": str(tmp_path),
            "STAGE": "all",
            "CUDA_VISIBLE_DEVICES": "5",
        },
    )
    assert short.returncode == 1 and "lists 1" in short.stderr


# ---------------------------------------------------------------------------
# Region driver and Namoi script
# ---------------------------------------------------------------------------


@needs_posix_bash
def test_region_driver_stops_at_submit_limit(tmp_path):
    key = tmp_path / "key.json"
    key.write_text("{}")
    proc = subprocess.run(
        [
            "bash",
            DRIVER,
            "--region",
            "beauce_fr",
            "--mode",
            "slurm",
            "--profile",
            "stampede3",
            "--years",
            "2024",
            "--sources",
            "sentinel2,landsat",
            "--engines",
            "delineate-anything,ftw",
            "--gee-project",
            "p",
            "--gee-service-account-key",
            str(key),
            "--out-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    # Stampede3 h100: 4 submitted jobs per user, 2 GPU tasks per run -> 2 runs fit.
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert proc.stdout.count("DRY  2024/sentinel2__") == 2
    assert "WAIT  2024/landsat__delineate-anything: the profile's submit limit" in proc.stdout
    assert "WAIT  2024/landsat__ftw: not submitted" in proc.stdout
    assert not (tmp_path / "out").exists()


def _demo_region(tmp_path, sources, engines, years=(2023,)) -> Path:
    region = tmp_path / "demo.yaml"
    region.write_text(
        yaml.safe_dump(
            {
                "name": "demo",
                "title": "Demo",
                "bbox": [149.8, -30.5, 150.3, -30.2],
                "test_bbox": [149.9, -30.4, 149.95, -30.35],
                "run": {
                    "years": list(years),
                    "sources": list(sources),
                    "engines": list(engines),
                    "tile_size_km": 20,
                    "halo_m": 1000,
                },
            }
        )
    )
    return region


@needs_posix_bash
def test_region_driver_local_tiles_overwrite(tmp_path):
    region = _demo_region(tmp_path, ["sentinel2"], ["delineate-anything"])
    proc = subprocess.run(
        [
            "bash",
            DRIVER,
            "--region-file",
            str(region),
            "--mode",
            "local-tiles",
            "--gee-project",
            "p",
            "--no-lulc-filter",
            "--overwrite",
            "--out-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env=_script_env(),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()
    assert any(line.startswith("agribound tiles make") and "--overwrite" in line for line in lines)
    assert any(line.startswith("agribound tiles merge") and "--overwrite" in line for line in lines)
    assert any("tiles run" in line and "--overwrite" in line for line in lines)


@needs_posix_bash
def _fake_gfm_env(tmp_path):
    """A fake GFM env prefix whose bin/agribound runs the real one, and a fake conda.

    The fake ``conda env list`` lists the env under the name ``agribound-gfm``; other
    ``conda run --no-capture-output -p PREFIX CMD...`` calls are logged and run CMD.
    """
    prefix = tmp_path / "envs" / "agribound-gfm"
    (prefix / "bin").mkdir(parents=True)
    real = shutil.which("agribound", path=_script_env()["PATH"])
    (prefix / "bin" / "agribound").write_text(f'#!/usr/bin/env bash\nexec "{real}" "$@"\n')
    (prefix / "bin" / "agribound").chmod(0o755)
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    (bindir / "conda").write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1 $2" == "env list" ]]; then echo "# conda environments:"; '
        f'echo "agribound-gfm           {prefix}"; exit 0; fi\n'
        'echo "$*" >>"$CONDA_LOG"\nshift 4\nexec "$@"\n'
    )
    (bindir / "conda").chmod(0o755)
    return prefix, bindir


@needs_posix_bash
def test_region_driver_gfm_env_prefix(tmp_path):
    """The GFM env's own bin/agribound runs (never an agribound found on PATH)."""
    prefix, bindir = _fake_gfm_env(tmp_path)
    log = tmp_path / "conda.log"
    region = _demo_region(tmp_path, ["hls"], ["prithvi"])
    env = {**_script_env(), "CONDA_LOG": str(log)}
    env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
    for gfm in (str(prefix), f"{prefix}/", "agribound-gfm"):
        proc = subprocess.run(
            [
                "bash",
                DRIVER,
                "--region-file",
                str(region),
                "--gfm-env",
                gfm,
                "--gee-project",
                "p",
                "--no-lulc-filter",
                "--out-root",
                str(tmp_path / "out"),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (
            log.read_text()
            .splitlines()[-1]
            .startswith(
                f"run --no-capture-output -p {prefix} {prefix}/bin/agribound delineate --dry-run"
            )
        )
    missing = subprocess.run(
        [
            "bash",
            DRIVER,
            "--region-file",
            str(region),
            "--gfm-env",
            "no-such-env",
            "--gee-project",
            "p",
            "--no-lulc-filter",
            "--out-root",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert "no bin/agribound in the GFM env 'no-such-env'" in missing.stdout + missing.stderr


@needs_posix_bash
def test_namoi_script_hpc_dry_run_validates_and_writes_nothing(tmp_path):
    script = EXAMPLES / "run_namoi_delineation.sh"
    if not script.exists():
        pytest.skip("examples/run_namoi_delineation.sh is a local (gitignored) file")
    out = tmp_path / "namoi"
    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--gee-project",
            "p",
            "--only",
            "extras",
            "--hpc-profile",
            "delta",
            "--dry-run",
            "--out-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
        env={**_script_env(), "AGB_GPU_ACCOUNT": "g", "AGB_CPU_ACCOUNT": "c"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DRY   ext_sam_refine_da" in proc.stdout
    # submit_region.sh --dry-run ran: its sbatch commands are shown, with the 2 km halo.
    assert len(_sbatch_lines(proc.stdout)) == 3
    assert "--halo-m 2000" in proc.stdout
    assert not out.exists()


@needs_posix_bash
def test_online_mode_single_phase(tmp_path):
    proc = _submit(
        tmp_path, "--profile", "delta", "--mode", "online", "--gee-after", "77", "--dry-run"
    )
    assert proc.returncode == 0, proc.stderr
    compute, merge = _sbatch_lines(proc.stdout)
    assert "agribound_stage.sbatch" not in proc.stdout
    # One phase: the GPU tasks download, so they get the GEE throttle (40 // 8 = 5) and
    # wait for the earlier Earth Engine jobs.
    assert "AGB_STAGE=all" in compute and "%5 " in compute
    assert "--dependency=afterany:77" in compute
    assert "--dependency=afterok:DRYRUN-1" in merge


@needs_posix_bash
@pytest.mark.parametrize("profile", ["delta", "expanse"])
def test_missing_gpu_account_is_refused_before_any_sbatch(tmp_path, fake_slurm, profile):
    """Regression: with only AGB_CPU_ACCOUNT set, the stage array was submitted, then the
    script died at the GPU array and left the stage array running."""
    env, log = fake_slurm
    proc = _submit(tmp_path, "--profile", profile, env={**env, "AGB_GPU_ACCOUNT": ""})
    assert proc.returncode == 1
    assert "set AGB_GPU_ACCOUNT" in proc.stderr and "nothing was submitted" in proc.stderr
    calls = log.read_text().splitlines() if log.exists() else []
    assert not [c for c in calls if c.startswith(("sbatch", "scancel"))]


@needs_posix_bash
def test_missing_cpu_account_is_refused_before_any_sbatch(tmp_path, fake_slurm):
    env, log = fake_slurm
    # No stage array (--skip-stage) and a GPU account: the merge job still needs the CPU one.
    proc = _submit(
        tmp_path, "--profile", "delta", "--skip-stage", env={**env, "AGB_CPU_ACCOUNT": ""}
    )
    assert proc.returncode == 1 and "set AGB_CPU_ACCOUNT" in proc.stderr
    calls = log.read_text().splitlines() if log.exists() else []
    assert not [c for c in calls if c.startswith("sbatch")]
    # --dry-run still prints placeholders instead of failing.
    dry = _submit(
        tmp_path, "--profile", "delta", "--dry-run", env={"AGB_CPU_ACCOUNT": ""}, out="dry"
    )
    assert dry.returncode == 0, dry.stderr
    stage, compute, merge = _sbatch_lines(dry.stdout)
    assert "--account=\\<AGB_CPU_ACCOUNT\\>" in stage and "--account=\\<AGB_CPU_ACCOUNT\\>" in merge
    assert "--account=g" in compute


@needs_posix_bash
@pytest.mark.parametrize("option", ["--reference", "--conda-env"])
def test_comma_in_exported_paths_is_refused_before_any_sbatch(tmp_path, fake_slurm, option):
    """Regression: a comma in --reference split the merge job's --export list after the
    stage and GPU arrays had been submitted."""
    env, log = fake_slurm
    ref_dir = tmp_path / "ref,dir"
    ref_dir.mkdir()
    value = str(ref_dir / "ref.geojson") if option == "--reference" else str(ref_dir)
    proc = _submit(tmp_path, "--profile", "delta", option, value, env=env)
    assert proc.returncode == 1 and "must not contain commas" in proc.stderr
    calls = log.read_text().splitlines() if log.exists() else []
    assert not [c for c in calls if c.startswith("sbatch")]
    assert not (tmp_path / "out" / "manifest.json").exists()  # died before tiles make


def _namoi_script() -> Path:
    script = EXAMPLES / "run_namoi_delineation.sh"
    if not script.exists():
        pytest.skip("examples/run_namoi_delineation.sh is a local (gitignored) file")
    return script


@needs_posix_bash
def test_namoi_script_appends_run_logs_and_flags_smoke_fine_tunes(tmp_path):
    """Regression: a re-run truncated run.log (the computing run's log was lost), and the
    --test fine-tuning cases were reported as a plain PASS."""
    script = _namoi_script()
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    (bindir / "agribound").write_text('#!/usr/bin/env bash\necho "fake agribound $*"\n')
    (bindir / "agribound").chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}
    out = tmp_path / "namoi"
    args = ["--gee-project", "p", "--test", "--no-lulc-filter", "--out-dir", str(out)]
    for _ in range(2):
        proc = subprocess.run(
            ["bash", str(script), *args, "--only", "extras"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
    for case in ("ext_sam_refine_da", "ext_ftw_published_2024"):
        log = (out / case / "run.log").read_text()
        assert log.count("===== ") == 2 and log.count("fake agribound") == 2, log
    engines = subprocess.run(
        ["bash", str(script), *args, "--only", "engines"], capture_output=True, text=True, env=env
    )
    assert engines.returncode == 0, engines.stdout + engines.stderr
    if (EXAMPLES / "namoi_polygons.geojson").exists():  # fine-tuning needs the reference
        assert "PASS  eng_geoai (smoke test only" in engines.stdout
        assert "PASS  eng_dinov3 (smoke test only" in engines.stdout
    assert "PASS  eng_delineate_anything\n" in engines.stdout


@needs_posix_bash
def test_namoi_script_runs_the_gfm_envs_own_agribound(tmp_path):
    """Regression: `conda run -p GFM agribound` ran the core env's agribound from PATH."""
    script = _namoi_script()
    prefix, bindir = _fake_gfm_env(tmp_path)
    (prefix / "bin" / "python").write_text(
        '#!/usr/bin/env bash\necho "fake-gfm-python terratorch 1.2.13"\n'
    )
    (prefix / "bin" / "python").chmod(0o755)
    log = tmp_path / "conda.log"
    env = {**_script_env(), "CONDA_LOG": str(log)}
    env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
    proc = subprocess.run(
        [
            "bash",
            str(script),
            "--gee-project",
            "p",
            "--test",
            "--no-lulc-filter",
            "--only",
            "engines",
            "--gfm-env",
            "agribound-gfm",
            "--dry-run",
            "--out-dir",
            str(tmp_path / "namoi"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert "DRY   eng_prithvi" in proc.stdout, proc.stdout + proc.stderr
    assert f"GFM env: {prefix} (fake-gfm-python terratorch 1.2.13)" in proc.stdout
    calls = log.read_text().splitlines()
    assert any(
        c.startswith(f"run --no-capture-output -p {prefix} {prefix}/bin/agribound") for c in calls
    )
    assert not (tmp_path / "namoi").exists()
