"""Google Satellite Embedding builder tests (Earth Engine and geoai stubbed; no network)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import Affine

from agribound.composites import gee, local
from agribound.config import AgriboundConfig

BBOX = (-104.20, 35.00, -104.19, 35.01)


def _config(tmp_path, backend="gee", bbox=BBOX, **kwargs):
    return AgriboundConfig(
        source="google-embedding",
        engine="embedding",
        year=2023,
        study_area="bbox:" + ",".join(str(v) for v in bbox),
        output_path=str(tmp_path / "fields.gpkg"),
        lulc_filter=False,
        gee_project="test-project",
        google_embedding_backend=backend,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Earth Engine backend
# ---------------------------------------------------------------------------


class _Chain:
    """Chainable stand-in for ee collections/images (records method names)."""

    def __init__(self, log):
        self.log = log

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.log.append((name, args))
            return self

        return call


@pytest.fixture
def stub_gee(monkeypatch):
    log: list = []
    state = {"summary": {"n_images": 4, "years": [2023]}, "fill": 1.0, "exports": []}
    fake_ee = types.ModuleType("ee")
    fake_ee.ImageCollection = lambda cid: (log.append(("ImageCollection", (cid,))), _Chain(log))[1]
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    monkeypatch.setattr("agribound.auth.ensure_gee", lambda config: None)
    monkeypatch.setattr(gee, "ee_geometry", lambda geom: "REGION")
    monkeypatch.setattr(gee, "collection_summary", lambda col, context="": state["summary"])
    monkeypatch.setattr(gee, "available_years", lambda col, context="": [2017, 2018])

    def fake_export(image, out_path, *, grid, dtype, band_names, tags=None, **kwargs):
        state["exports"].append({"grid": grid, "bands": list(band_names), "dtype": dtype})
        data = np.full((len(band_names), grid.height, grid.width), 0.1, dtype=np.float32)
        if state["fill"] < 1.0:
            data[:, :, : int(grid.width * (1 - state["fill"]))] = np.nan
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            count=len(band_names),
            height=grid.height,
            width=grid.width,
            dtype="float32",
            crs=grid.crs,
            transform=grid.transform,
            nodata=float("nan"),
        ) as dst:
            dst.write(data)
            dst.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
        return str(out_path)

    monkeypatch.setattr(gee, "export_ee_image", fake_export)
    state["log"] = log
    return state


class TestGoogleEmbeddingGEE:
    def test_exports_64_bands_float32_10m_utm(self, tmp_path, stub_gee):
        path, tags = local.build_google_embedding_gee(_config(tmp_path))
        (export,) = stub_gee["exports"]
        assert export["bands"] == [f"A{i:02d}" for i in range(64)]
        assert export["dtype"] == "float32"
        assert export["grid"].crs == "EPSG:32613"
        assert export["grid"].transform.a == 10.0
        assert ("ImageCollection", (local.GOOGLE_EMBEDDING_COLLECTION,)) in stub_gee["log"]
        assert ("filterDate", ("2023-01-01", "2024-01-01")) in stub_gee["log"]
        assert tags["AGRIBOUND_BACKEND"] == "gee"
        assert tags["AGRIBOUND_VALID_FRACTION"] == pytest.approx(1.0)

    def test_no_images_lists_years(self, tmp_path, stub_gee):
        stub_gee["summary"] = {"n_images": 0, "years": []}
        from agribound.composites import NoDataError

        with pytest.raises(NoDataError, match=r"2017, 2018"):
            local.build_google_embedding_gee(_config(tmp_path))

    def test_empty_download_is_rejected_and_removed(self, tmp_path, stub_gee):
        stub_gee["fill"] = 0.0
        from agribound.composites import NoDataError

        with pytest.raises(NoDataError, match="no valid pixels"):
            local.build_google_embedding_gee(_config(tmp_path))
        assert not list((tmp_path / ".agribound_cache").glob("google_embedding_*.tif"))

    def test_backend_changes_cache_key(self, tmp_path, stub_gee):
        from agribound._cache import cache_key

        a = _config(tmp_path)
        assert cache_key(a) != cache_key(a.merged(google_embedding_backend="source_coop"))


# ---------------------------------------------------------------------------
# Source Cooperative backend (fake tile index and local int8 tiles; no network)
# ---------------------------------------------------------------------------

AEF_BANDS = [f"A{i:02d}" for i in range(64)]


def _write_aef_tile(path, crs, lonlat_box, footprint, code, overlap_deg=0.0, bottom_up=True):
    """Write an AEF-like tile: int8, 64 bands, nodata -128 outside *footprint* (+overlap)."""
    import shapely
    from shapely.geometry import box
    from shapely.ops import transform as shp_transform

    fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    dense = shapely.segmentize(box(*lonlat_box), 0.001)
    x0, y0, x1, y1 = shp_transform(fwd.transform, dense).bounds
    x0, y0 = np.floor(x0 / 10) * 10, np.floor(y0 / 10) * 10
    x1, y1 = np.ceil(x1 / 10) * 10, np.ceil(y1 / 10) * 10
    w, h = int(round((x1 - x0) / 10)), int(round((y1 - y0) / 10))
    xs = x0 + 5 + 10 * np.arange(w)
    ys = y0 + 5 + 10 * np.arange(h)  # south to north
    gx, gy = np.meshgrid(xs, ys)
    lon, lat = inv.transform(gx, gy)
    region = footprint.buffer(overlap_deg) if overlap_deg else footprint
    inside = shapely.contains_xy(region, lon, lat)
    data = np.full((64, h, w), -128, dtype=np.int8)
    data[:, inside] = np.int8(code)
    if bottom_up:
        transform = Affine(10, 0, x0, 0, 10, y0)  # row 0 = southern edge, as the mirror's COGs
    else:
        data = data[:, ::-1, :]
        transform = Affine(10, 0, x0, 0, -10, y1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=64,
        height=h,
        width=w,
        dtype="int8",
        crs=crs,
        transform=transform,
        nodata=-128,
    ) as dst:
        dst.write(data)
        for i, name in enumerate(AEF_BANDS, start=1):
            dst.set_band_description(i, name)
    return str(path)


def _write_index(cache_dir, rows):
    import geopandas as gpd

    cache_dir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {
            "year": [r["year"] for r in rows],
            "crs": [r["crs"] for r in rows],
            "path": [r["path"] for r in rows],
        },
        geometry=[r["geometry"] for r in rows],
        crs="OGC:CRS84",
    )
    gdf.to_parquet(cache_dir / "aef_index.parquet")


def _sc_config(tmp_path, bbox, **kwargs):
    return _config(
        tmp_path, "source_coop", bbox=bbox, embedding_cache_dir=str(tmp_path / "aef"), **kwargs
    )


def _lonlat_of_pixels(path):
    with rasterio.open(path) as src:
        rows, cols = np.mgrid[0 : src.height, 0 : src.width]
        xs, ys = rasterio.transform.xy(src.transform, rows.ravel(), cols.ravel())
        lon, lat = pyproj.Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(
            xs, ys
        )
        return (
            src.read(),
            np.asarray(lon).reshape(src.height, src.width),
            np.asarray(lat).reshape(src.height, src.width),
        )


SEAM_BBOX = (149.99, -30.58, 150.01, -30.57)  # crosses the UTM 55/56 boundary at 150 E


class TestGoogleEmbeddingSourceCoop:
    def test_dequantize_matches_formula(self):
        codes = np.arange(-128, 128, dtype=np.int8)
        x = codes.astype(np.float64)
        expected = (np.sign(x) * (x / 127.5) ** 2).astype(np.float32)
        expected[0] = np.nan  # -128 is nodata
        out = local.dequantize_aef(codes.reshape(16, 16)).ravel()
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, expected)
        with pytest.raises(TypeError):
            local.dequantize_aef(codes.astype(np.int16))

    def test_tile_url(self):
        s3 = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2023/56S/x.tiff"
        assert local.aef_tile_url(s3) == (
            "/vsicurl/https://data.source.coop/tge-labs/aef/v1/annual/2023/56S/x.tiff"
        )
        assert local.aef_tile_url("https://h/x.tif") == "/vsicurl/https://h/x.tif"
        assert local.aef_tile_url("/data/x.tif") == "/data/x.tif"

    def test_seam_crossing_study_area_is_complete(self, tmp_path):
        # Regression: geoai's corner-based request failed here ("All tiles must share the
        # same CRS") or left unread wedges. Tiles hold nodata beyond their zone, except a
        # small overlap strip (as the mirror's tiles do).
        from shapely.geometry import box

        tiles = [
            ("EPSG:32755", box(149.9, -30.62, 150.0, -30.53), 10),
            ("EPSG:32756", box(150.0, -30.62, 150.1, -30.53), 20),
        ]
        rows = []
        for i, (crs, fp, code) in enumerate(tiles):
            path = _write_aef_tile(
                tmp_path / f"t{i}.tif", crs, fp.bounds, fp, code, overlap_deg=0.003
            )
            rows.append({"year": 2023, "crs": crs, "path": path, "geometry": fp})
        _write_index(tmp_path / "aef", rows)
        path, tags = local.build_google_embedding_source_coop(_sc_config(tmp_path, SEAM_BBOX))
        data, lon, _lat = _lonlat_of_pixels(path)
        assert tags["AGRIBOUND_N_TILES"] == 2
        assert tags["AGRIBOUND_VALID_FRACTION"] == pytest.approx(1.0)
        assert np.isfinite(data).all()
        v10, v20 = local.dequantize_aef(np.array([10, 20], dtype=np.int8))
        # Every pixel comes from the tile of its own zone, although each tile also
        # holds valid data ~300 m beyond the seam; only within the 10 m margin (plus
        # nearest-neighbour rounding, < 19 m) may the export CRS's zone (56) supply it.
        assert (data[5][lon > 150.0] == v20).all()
        assert (data[5][lon < 150.0 - 0.0002] == v10).all()
        assert (data[5][lon < 150.0] == v20).mean() < 0.05
        with rasterio.open(path) as src:
            assert src.crs.to_epsg() == 32756 and src.res == (10.0, 10.0)

    def test_seam_grid_matches_earth_engine_grid(self, tmp_path):
        from shapely.geometry import box

        from agribound.composites.gee import compute_export_grid

        fp = box(149.9, -30.62, 150.1, -30.53)
        rows = []
        for i, (crs, sub) in enumerate(
            [
                ("EPSG:32755", box(149.9, -30.62, 150.0, -30.53)),
                ("EPSG:32756", box(150.0, -30.62, 150.1, -30.53)),
            ]
        ):
            path = _write_aef_tile(tmp_path / f"t{i}.tif", crs, fp.bounds, sub, 5)
            rows.append({"year": 2023, "crs": crs, "path": path, "geometry": sub})
        _write_index(tmp_path / "aef", rows)
        path, _ = local.build_google_embedding_source_coop(_sc_config(tmp_path, SEAM_BBOX))
        grid = compute_export_grid(box(*SEAM_BBOX), "EPSG:32756", 10.0)
        with rasterio.open(path) as src:
            assert src.transform == grid.transform
            assert (src.width, src.height) == (grid.width, grid.height)

    def test_equator_crossing_uses_both_hemispheres(self, tmp_path):
        from shapely.geometry import LineString, box

        north = box(31.9, 0.0, 32.1, 0.05)
        south = box(31.9, -0.05, 32.1, 0.0)
        rows = [
            {
                "year": 2023,
                "crs": "EPSG:32636",
                "path": _write_aef_tile(tmp_path / "n.tif", "EPSG:32636", north.bounds, north, 30),
                "geometry": north,
            },
            {
                "year": 2023,
                "crs": "EPSG:32736",
                "path": _write_aef_tile(tmp_path / "s.tif", "EPSG:32736", south.bounds, south, 40),
                "geometry": south,
            },
            # zero-height tile on the equator (as in the mirror's index): must be skipped
            {
                "year": 2023,
                "crs": "EPSG:32736",
                "path": str(tmp_path / "does_not_exist.tif"),
                "geometry": LineString([(31.9, 0.0), (32.1, 0.0)]),
            },
        ]
        _write_index(tmp_path / "aef", rows)
        cfg = _sc_config(tmp_path, (31.99, -0.007, 32.01, 0.009))
        path, tags = local.build_google_embedding_source_coop(cfg)
        data, _lon, lat = _lonlat_of_pixels(path)
        assert tags["AGRIBOUND_N_TILES"] == 2
        assert np.isfinite(data).all()
        v30, v40 = local.dequantize_aef(np.array([30, 40], dtype=np.int8))
        assert (data[0][lat > 0.0003] == v30).all()
        assert (data[0][lat < -0.0003] == v40).all()

    @pytest.mark.parametrize("bottom_up", [True, False])
    def test_high_latitude_zone_edge_is_read_completely(self, tmp_path, bottom_up):
        # A tall box next to a zone edge at 60 N: the UTM grid is strongly rotated
        # against longitude/latitude, so two box corners do not bound the needed window.
        from shapely.geometry import box

        tile_box = (6.0, 59.95, 6.2, 60.25)
        fp = box(*tile_box)
        path = _write_aef_tile(
            tmp_path / "t.tif", "EPSG:32632", tile_box, fp, 50, bottom_up=bottom_up
        )
        _write_index(
            tmp_path / "aef",
            [{"year": 2023, "crs": "EPSG:32632", "path": path, "geometry": box(*tile_box)}],
        )
        cfg = _sc_config(tmp_path, (6.005, 60.0, 6.03, 60.2), export_crs="EPSG:32632")
        out, tags = local.build_google_embedding_source_coop(cfg)
        data, lon, lat = _lonlat_of_pixels(out)
        assert tags["AGRIBOUND_VALID_FRACTION"] == pytest.approx(1.0)
        # every grid pixel inside the tile footprint has data (the grid's rotated
        # corners west of 6 E are outside this fake tile and stay NaN)
        inside = (lon > 6.0 + 0.0002) & (lat > 59.95) & (lat < 60.25)
        assert inside.mean() > 0.8
        assert (data[:, inside] == local.dequantize_aef(np.array([50], dtype=np.int8))[0]).all()

    def test_no_tile_for_year_lists_years(self, tmp_path):
        from shapely.geometry import box

        fp = box(-104.3, 34.9, -104.1, 35.1)
        rows = [
            {
                "year": y,
                "crs": "EPSG:32613",
                "path": _write_aef_tile(tmp_path / f"{y}.tif", "EPSG:32613", fp.bounds, fp, 1),
                "geometry": fp,
            }
            for y in (2021, 2022)
        ]
        _write_index(tmp_path / "aef", rows)
        from agribound.composites import NoDataError

        with pytest.raises(NoDataError, match=r"Years with tiles there: \[2021, 2022\]"):
            local.build_google_embedding_source_coop(_sc_config(tmp_path, BBOX))

    def test_unexpected_tile_is_rejected(self, tmp_path):
        from shapely.geometry import box

        fp = box(-104.3, 34.9, -104.1, 35.1)
        bad = tmp_path / "bad.tif"
        _write_utm_float(bad, fp.bounds)
        _write_index(
            tmp_path / "aef",
            [{"year": 2023, "crs": "EPSG:32613", "path": str(bad), "geometry": fp}],
        )
        with pytest.raises(ValueError, match="not a Google Satellite Embedding tile"):
            local.build_google_embedding_source_coop(_sc_config(tmp_path, BBOX))

    def test_cached_and_dispatched_by_builder(self, tmp_path):
        from shapely.geometry import box

        fp = box(-104.3, 34.9, -104.1, 35.1)
        path = _write_aef_tile(tmp_path / "t.tif", "EPSG:32613", fp.bounds, fp, 7)
        _write_index(
            tmp_path / "aef", [{"year": 2023, "crs": "EPSG:32613", "path": path, "geometry": fp}]
        )
        builder = local.EmbeddingCompositeBuilder()
        out = builder.build(_sc_config(tmp_path, BBOX))
        assert builder.last_metadata["AGRIBOUND_BACKEND"] == "source_coop"
        Path(path).unlink()  # the second build must not read tiles again
        assert local.EmbeddingCompositeBuilder().build(_sc_config(tmp_path, BBOX)) == out


def _write_utm_float(path, lonlat_box):
    t = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32613", always_xy=True)
    xs, ys = t.transform([lonlat_box[0], lonlat_box[2]], [lonlat_box[1], lonlat_box[3]])
    w, h = 50, 50
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=64,
        height=h,
        width=w,
        dtype="float32",
        crs="EPSG:32613",
        transform=Affine(10, 0, min(xs), 0, 10, min(ys)),
    ) as dst:
        dst.write(np.zeros((64, h, w), np.float32))


class TestAefIndexDownload:
    def test_download_sends_user_agent_and_renames(self, tmp_path, monkeypatch):
        import io
        import urllib.request

        seen = {}

        class _Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        def fake_urlopen(request, timeout=None):
            seen["ua"] = request.get_header("User-agent")
            seen["url"] = request.full_url
            seen["timeout"] = timeout
            return _Response(b"PAR1-data")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        dest = local.aef_index_path(tmp_path)
        local._download_aef_index(dest)
        assert dest.read_bytes() == b"PAR1-data"
        assert seen["url"] == local.AEF_INDEX_URL
        assert seen["ua"].startswith("agribound/") and seen["timeout"] > 0
        assert not list(tmp_path.glob("*.part"))

    def test_failed_download_leaves_no_file(self, tmp_path, monkeypatch):
        import urllib.request

        def boom(request, timeout=None):
            raise TimeoutError("stalled")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        dest = local.aef_index_path(tmp_path)
        with pytest.raises(TimeoutError):
            local._download_aef_index(dest)
        assert not dest.exists() and not list(tmp_path.glob("*.part"))

    def test_default_location(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        assert local.aef_index_path(None) == tmp_path / ".cache" / "agribound" / "aef_index.parquet"
