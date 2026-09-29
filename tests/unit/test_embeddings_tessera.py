"""TESSERA embedding builder tests with a mocked ``geotessera.GeoTesseraZarr`` (no network)."""

from __future__ import annotations

import logging
import math

import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import Affine

from agribound.composites import local
from agribound.config import AgriboundConfig

N_BANDS = 128
# Box across the UTM 55/56 seam (150 E) in the southern hemisphere (Namoi region).
SEAM_BBOX = (149.99, -30.58, 150.01, -30.57)


class FakeZarr:
    """Mimics GeoTesseraZarr.read_region: native UTM (EPSG:326zz, also in the south)."""

    instances: list[FakeZarr] = []
    offset_m = 5.0  # lattice offset, to check that the primary zone is not resampled
    valid = True

    def __init__(self, store_url, cache_dir=None, cache_max_size=None):
        self.url = store_url
        self.cache_dir = cache_dir
        self.years = list(range(2015, 2026))
        self.model_version = "https://geotessera.org/model/1.1"
        self.calls: list[tuple] = []
        FakeZarr.instances.append(self)

    def read_region(self, bbox, year, *, depth=None, progress=False):
        self.calls.append((tuple(bbox), year))
        centre = (bbox[0] + bbox[2]) / 2
        zone = int(math.floor((centre + 180) / 6)) + 1
        crs = f"EPSG:{32600 + zone}"
        t = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = t.transform(
            [bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]]
        )
        o = self.offset_m
        x0 = math.floor((min(xs) - o) / 10) * 10 + o
        y1 = math.ceil((max(ys) - o) / 10) * 10 + o
        w = math.ceil((max(xs) - x0) / 10)
        h = math.ceil((y1 - min(ys)) / 10)
        arr = np.empty((h, w, N_BANDS), dtype=np.float32)
        arr[:] = float(zone)
        # band 1 encodes the native pixel position, band 2 the zone
        rows, cols = np.mgrid[0:h, 0:w]
        arr[:, :, 0] = rows * 10000 + cols
        if not FakeZarr.valid:
            arr[:] = np.nan
        return arr, Affine(10, 0, x0, 0, -10, y1), crs


@pytest.fixture
def fake_zarr(monkeypatch):
    # The builder resolves stores through geotessera (the ``tessera`` extra).
    geotessera = pytest.importorskip("geotessera")

    FakeZarr.instances = []
    FakeZarr.valid = True
    monkeypatch.setattr(geotessera, "GeoTesseraZarr", FakeZarr, raising=False)
    return FakeZarr


def _config(tmp_path, bbox=SEAM_BBOX, **kwargs):
    params = {
        "source": "tessera-embedding",
        "engine": "embedding",
        "year": 2024,
        "study_area": "bbox:" + ",".join(str(v) for v in bbox),
        "output_path": str(tmp_path / "fields.gpkg"),
        "lulc_filter": False,
        "tessera_version": "v1.1",
    }
    params.update(kwargs)
    return AgriboundConfig(**params)


@pytest.fixture
def needs_geotessera():
    """Skip unless geotessera (the ``tessera`` extra) is installed."""
    return pytest.importorskip("geotessera")


@pytest.mark.usefixtures("needs_geotessera")
class TestStoreResolution:
    def test_defaults(self):
        url, variant = local.resolve_tessera_store("v1")
        assert url.endswith("/zarr/v1") and variant == "vultr"
        url, variant = local.resolve_tessera_store("v1.1")
        assert url.endswith("/zarr/v1.1") and variant == "cambridge"
        url, variant = local.resolve_tessera_store("v2")
        assert url.endswith("/zarr/v2-2B-L~beta1") and variant == "2B-L~beta1"

    def test_explicit_v2_variant(self):
        url, variant = local.resolve_tessera_store("v2", "2B-L~beta2")
        assert url.endswith("/zarr/v2-2B-L~beta2") and variant == "2B-L~beta2"

    def test_unpublished_variant_raises(self):
        with pytest.raises(ValueError, match="No TESSERA Zarr store"):
            local.resolve_tessera_store("v1.1", "dclimate")
        with pytest.raises(ValueError, match="Unknown tessera_version"):
            local.resolve_tessera_store("v3")


class TestZoneSplit:
    def test_sub_bboxes_do_not_cross_the_seam(self):
        west = local.zone_sub_bbox(SEAM_BBOX, 55)
        east = local.zone_sub_bbox(SEAM_BBOX, 56)
        assert west[0] == SEAM_BBOX[0] and west[2] < 150.0 and 150.0 - west[2] < 1e-6
        assert east[0] == 150.0 and east[2] == SEAM_BBOX[2]
        assert local.zone_sub_bbox(SEAM_BBOX, 54) is None

    def test_one_read_per_zone(self, tmp_path, fake_zarr):
        local.build_tessera_embedding(_config(tmp_path))
        (store,) = fake_zarr.instances
        assert len(store.calls) == 2
        centres = [(b[0] + b[2]) / 2 for b, _ in store.calls]
        assert centres[0] < 150.0 < centres[1]
        assert all(year == 2024 for _, year in store.calls)


class TestSeamMosaic:
    def test_output_grid_values_and_tags(self, tmp_path, fake_zarr):
        path, tags = local.build_tessera_embedding(_config(tmp_path))
        with rasterio.open(path) as src:
            assert src.crs.to_epsg() == 32756  # centroid at 150.0 E, south
            assert src.count == N_BANDS and src.dtypes[0] == "float32"
            assert np.isnan(src.nodata)
            assert src.descriptions[0] == "T000" and src.descriptions[-1] == "T127"
            assert src.tags()["TESSERA_DATASET_VERSION"] == "v1.1"
            assert src.tags()["TESSERA_DATASET_VARIANT"] == "cambridge"
            assert src.tags()["TESSERA_YEAR"] == "2024"
            t = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            west = t.transform(149.995, -30.575)
            east = t.transform(150.005, -30.575)
            zone_band = src.read(2)
            r, c = src.index(*west)
            assert zone_band[r, c] == 55.0
            r, c = src.index(*east)
            assert zone_band[r, c] == 56.0
        assert tags["AGRIBOUND_VALID_FRACTION"] == pytest.approx(1.0)

    def test_reads_cover_the_whole_export_grid(self, tmp_path, fake_zarr):
        # The export grid is the study-area bounding box in UTM, whose corners reach
        # beyond the lon/lat box; the zone reads must cover them (no NaN corners).
        path, _ = local.build_tessera_embedding(_config(tmp_path))
        with rasterio.open(path) as src:
            band = src.read(1)
        assert np.isfinite(band).all()

    def test_not_masked_to_study_area_polygons(self, tmp_path, fake_zarr):
        # A triangle study area: pixels inside its bounding box but outside the
        # triangle keep their embeddings (no polygon mask on the engine input).
        wkt = "POLYGON ((149.99 -30.58, 150.01 -30.58, 149.99 -30.57, 149.99 -30.58))"
        cfg = _config(tmp_path).merged(study_area=wkt)
        path, tags = local.build_tessera_embedding(cfg)
        with rasterio.open(path) as src:
            t = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            r, c = src.index(*t.transform(150.008, -30.5705))  # NE corner, outside
            assert np.isfinite(src.read(1)[r, c])
        assert tags["AGRIBOUND_VALID_FRACTION"] == pytest.approx(1.0)

    def test_primary_zone_pixels_are_copied_not_resampled(self, tmp_path, fake_zarr):
        path, _ = local.build_tessera_embedding(_config(tmp_path))
        (store,) = fake_zarr.instances
        with rasterio.open(path) as src:
            # lattice follows the native grid (5 m offset), shifted by 10 000 km (326 -> 327)
            assert (src.transform.c - 5.0) % 10 == 0
            assert (src.transform.f - 5.0) % 10 == 0
            band1 = src.read(1)
            transform = src.transform
        # rebuild the primary (zone 56) native read and compare a pixel exactly
        (sub,) = [b for b, _ in store.calls if (b[0] + b[2]) / 2 > 150.0]
        arr, native_t, _crs = FakeZarr("x").read_region(sub, 2024)
        r0, c0 = arr.shape[0] // 2, arr.shape[1] // 2  # a pixel well inside the AOI
        x, y = native_t @ (c0 + 0.5, r0 + 0.5)
        col = int((x - transform.c) / transform.a)
        row = int((y + 10_000_000 - transform.f) / transform.e)
        assert band1[row, col] == arr[r0, c0, 0]
        assert band1[row, col + 1] == arr[r0, c0 + 1, 0]
        assert band1[row + 1, col] == arr[r0 + 1, c0, 0]

    def test_overlap_prefers_primary_zone(self, tmp_path, fake_zarr):
        # Zone 55's fake grid runs past 150 E; the primary zone (56) must win there.
        path, _ = local.build_tessera_embedding(_config(tmp_path))
        with rasterio.open(path) as src:
            t = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            r, c = src.index(*t.transform(150.0005, -30.575))
            assert src.read(2)[r, c] == 56.0

    def test_northern_single_zone_crs(self, tmp_path, fake_zarr):
        path, _ = local.build_tessera_embedding(
            _config(tmp_path, bbox=(-104.2, 35.0, -104.19, 35.01))
        )
        with rasterio.open(path) as src:
            assert src.crs.to_epsg() == 32613
        assert len(fake_zarr.instances[0].calls) == 1


class TestCoverageChecks:
    def test_no_valid_pixels_raises_with_advice(self, tmp_path, fake_zarr):
        fake_zarr.valid = False
        from agribound.composites import NoDataError

        with pytest.raises(NoDataError, match=r"v1\.1.*2024|2024.*v1\.1"):
            local.build_tessera_embedding(_config(tmp_path, tessera_version="v1"))
        assert not list(tmp_path.glob(".agribound_cache/tessera_embedding_*.tif"))

    def test_v11_advice_does_not_suggest_v11_or_a_global_v11_layer(self, tmp_path, fake_zarr):
        fake_zarr.valid = False
        with pytest.raises(ValueError) as info:
            local.build_tessera_embedding(_config(tmp_path, tessera_version="v1.1"))
        msg = str(info.value)
        assert "tessera_version='v1.1'" not in msg
        assert "no near-global year" in msg and "tessera_version='v1' with year=2024" in msg

    def test_v2_advice_says_sparse_beta(self, tmp_path, fake_zarr):
        fake_zarr.valid = False
        with pytest.raises(ValueError, match="sparse beta"):
            local.build_tessera_embedding(_config(tmp_path, tessera_version="v2"))

    def test_v1_2024_advice_does_not_suggest_2024(self):
        advice = local.tessera_coverage_advice("v1", 2024)
        assert "year=2024" not in advice and "water" in advice
        assert "year=2024" in local.tessera_coverage_advice("v1", 2023)

    @pytest.mark.parametrize("version", ["v1", "v1.1", "v2"])
    def test_advice_is_version_specific(self, version):
        advice = local.tessera_coverage_advice(version)
        assert advice.startswith(f"TESSERA {version} ")
        assert "tessera_coverage()" in advice
        other = {"v1": "v1.1", "v1.1": "v1", "v2": "v1"}[version]
        assert f"tessera_version='{other}'" in advice

    def test_low_coverage_warns(self, tmp_path, fake_zarr, monkeypatch, caplog):
        orig = FakeZarr.read_region

        def half(self, bbox, year, **kw):
            arr, t, crs = orig(self, bbox, year, **kw)
            arr[:, : int(arr.shape[1] * 0.8)] = np.nan
            return arr, t, crs

        monkeypatch.setattr(FakeZarr, "read_region", half)
        with caplog.at_level(logging.WARNING):
            _, tags = local.build_tessera_embedding(_config(tmp_path))
        assert tags["AGRIBOUND_VALID_FRACTION"] < 0.5
        assert any("covers only" in r.message for r in caplog.records)

    def test_missing_year_raises(self, tmp_path, fake_zarr, monkeypatch):
        monkeypatch.setattr(FakeZarr, "__init__", _init_years([2024]))
        from agribound.composites import NoDataError

        with pytest.raises(ValueError, match="no 2023 layer") as info:
            local.build_tessera_embedding(_config(tmp_path, year=2023))
        # A year missing from the whole dataset is not a study-area no-data condition.
        assert not isinstance(info.value, NoDataError)


def _init_years(years):
    def __init__(self, store_url, cache_dir=None, cache_max_size=None):  # noqa: N807
        self.url = store_url
        self.years = list(years)
        self.model_version = ""
        self.calls = []
        FakeZarr.instances.append(self)

    return __init__


class TestCaching:
    def test_second_build_reads_nothing(self, tmp_path, fake_zarr):
        cfg = _config(tmp_path)
        p1, _ = local.build_tessera_embedding(cfg)
        p2, tags = local.build_tessera_embedding(cfg)
        assert p1 == p2
        assert len(fake_zarr.instances) == 1
        assert tags["TESSERA_DATASET_VERSION"] == "v1.1"

    def test_version_and_year_change_the_file(self, tmp_path, fake_zarr):
        cfg = _config(tmp_path)
        paths = {
            local.build_tessera_embedding(cfg)[0],
            local.build_tessera_embedding(cfg.merged(tessera_version="v1"))[0],
            local.build_tessera_embedding(cfg.merged(year=2023))[0],
        }
        assert len(paths) == 3

    def test_builder_dispatch(self, tmp_path, fake_zarr):
        builder = local.EmbeddingCompositeBuilder()
        path = builder.build(_config(tmp_path))
        assert builder.last_metadata["AGRIBOUND_SOURCE"] == "tessera-embedding"
        assert path.endswith(".tif")


class FakeRegistry:
    def __init__(self, tiles):
        self.tiles = tiles

    def load_blocks_for_region(self, bounds, year):
        return [t for t in self.tiles if t[0] == year]

    def get_available_years(self):
        return sorted({t[0] for t in self.tiles})


class TestTesseraCoverage:
    def test_counts_only_tiles_intersecting_the_box(self, monkeypatch):
        geotessera = pytest.importorskip("geotessera")

        tiles = [
            (2024, 149.95, -30.55),
            (2024, 150.05, -30.55),
            (2024, 150.15, -30.55),  # touches the box edge only (registry expansion)
            (2023, 149.95, -30.55),
        ]

        class FakeGT:
            def __init__(self, dataset_version, dataset_variant=None, cache_dir=None):
                self.registry = FakeRegistry(tiles)
                self.dataset_variant = dataset_variant or "cambridge"

        monkeypatch.setattr(geotessera, "GeoTessera", FakeGT, raising=False)
        report = local.tessera_coverage((149.9, -30.6, 150.1, -30.5), 2024, "v1.1")
        assert report["tiles_expected"] == 2
        assert report["tiles_available"] == 2
        assert report["fraction"] == 1.0
        assert report["tiles_by_year"] == {2023: 1, 2024: 2}
        assert report["variant"] == "cambridge"
