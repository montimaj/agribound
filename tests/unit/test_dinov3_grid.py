"""DINOv3's processing grid: rasters coarser than 10 m are read on a bilinear 10 m copy.

Covers the rule (``dinov3_processing_gsd``), the copy (``dinov3_processing_raster``),
the band rule on that grid (``widest_boundary_erosion(..., grid_gsd_m=...)``), the
training chips and the fine-tuning cache key, and inference with a checkpoint that
records the grid. Rasters of 10 m or finer must be read exactly as before.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from agribound.config import AgriboundConfig
from agribound.engines import finetune
from agribound.engines.finetune import _data

UTM = "EPSG:32613"
X0, Y0 = 663_570.0, 3_647_460.0


def _raster(path, res, shape=(6, 40, 50), nodata=float("nan"), fill=None):
    """Landsat-like float32 raster: a ramp in every band, a bright block in R."""
    count, h, w = shape
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float32)
    data = np.stack([1000.0 + 10.0 * cols + 5.0 * rows + 100.0 * b for b in range(count)])
    data = data.astype(np.float32)
    data[2, h // 4 : h // 2, w // 4 : w // 2] += 3000.0
    if fill is not None:
        data[:, :2, :2] = fill
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=count,
        dtype="float32",
        crs=UTM,
        transform=from_origin(X0, Y0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data)
        dst.update_tags(AGRIBOUND_SOURCE="landsat")
        for b in range(count):
            dst.set_band_description(b + 1, f"B{b + 1}")
    return str(path)


def _config(tmp_path, **kwargs):
    values = {
        "source": "landsat",
        "year": 2024,
        "study_area": "bbox:-103.25,32.75,-103.05,32.95",
        "gee_project": "test-project",
        "engine": "dinov3",
        "output_path": str(tmp_path / "out" / "fields.gpkg"),
        "lulc_filter": False,
        "device": "cpu",
    }
    values.update(kwargs)
    return AgriboundConfig(**values)


class TestRule:
    @pytest.mark.parametrize(
        ("res", "expected"),
        [
            (30.0, 10.0),  # Landsat, HLS
            (20.0, 10.0),
            (15.0, 10.0),  # Landsat PAN
            (10.2, 10.0),
            (10.05, None),
            (10.0, None),
            (6.0, None),
        ],
    )
    def test_auto(self, tmp_path, res, expected):
        raster = _raster(tmp_path / "r.tif", res, shape=(3, 8, 8))
        assert _data.dinov3_processing_gsd(raster) == expected
        assert _data.dinov3_processing_gsd(raster, {"processing_gsd_m": "auto"}) == expected
        assert _data.dinov3_processing_gsd(raster, {"processing_gsd_m": None}) == expected
        # The same rule without the raster (results_versions applies it to a source).
        assert _data.dinov3_grid_gsd(res) == expected

    @pytest.mark.parametrize(("metres", "expected"), [(10.0, None), (30.0, 10.0)])
    def test_projected_crs_in_feet(self, tmp_path, metres, expected):
        """Pixel sizes are measured in metres: EPSG:2263 is in US survey feet."""
        foot = 0.3048006096012192
        path = tmp_path / "ft.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=12,
            width=12,
            count=3,
            dtype="float32",
            crs="EPSG:2263",
            transform=from_origin(1_000_000.0, 200_000.0, metres / foot, metres / foot),
            nodata=float("nan"),
        ) as dst:
            dst.write(np.ones((3, 12, 12), np.float32))
        with rasterio.open(path) as src:
            assert _data._grid_pixel_size_m(src) == pytest.approx(metres)
            assert _data._gsd_m(src) == pytest.approx(metres / foot)  # other engines: as before
        assert _data.dinov3_processing_gsd(str(path)) == expected
        if expected is None:
            return
        copy, _, info = _data.dinov3_processing_raster(str(path), _config(tmp_path), 10.0, [1])
        assert info["scale"] == pytest.approx(3.0) and info["size_px"] == [36, 36]
        with rasterio.open(path) as src, rasterio.open(copy) as out:
            assert out.res == pytest.approx((10.0 / foot, 10.0 / foot))  # 10 m, 32.8 ft
            assert out.bounds == pytest.approx(src.bounds)

    def test_native_number_and_errors(self, tmp_path):
        raster = _raster(tmp_path / "r.tif", 30.0, shape=(3, 8, 8))
        assert _data.dinov3_processing_gsd(raster, {"processing_gsd_m": "native"}) is None
        assert _data.dinov3_processing_gsd(raster, {"processing_gsd_m": 15}) == 15.0
        assert _data.dinov3_processing_gsd(raster, {"processing_gsd_m": 30.1}) is None
        with pytest.raises(ValueError, match="coarser than the raster"):
            _data.dinov3_processing_gsd(raster, {"processing_gsd_m": 60})
        for bad in ("fine", 0, -10, float("nan"), True, [10]):
            with pytest.raises(ValueError, match="processing_gsd_m"):
                _data.dinov3_processing_gsd(raster, {"processing_gsd_m": bad})


class TestCopy:
    def test_grid_values_tags_and_reuse(self, tmp_path):
        raster = _raster(tmp_path / "r.tif", 30.0, fill=np.nan)
        cfg = _config(tmp_path)
        path, bands, info = _data.dinov3_processing_raster(raster, cfg, 10.0, [3, 2, 1])
        assert bands == [3, 2, 1] and info["gsd_m"] == 10.0 and info["scale"] == 3.0
        assert info["resampling"] == "bilinear" and info["size_px"] == [120, 150]
        with rasterio.open(raster) as src, rasterio.open(path) as out:
            assert out.count == 3 and out.crs == src.crs and out.bounds == src.bounds
            assert out.res == (10.0, 10.0) and out.dtypes[0] == "float32"
            assert np.isnan(out.nodata)
            assert out.tags()["AGRIBOUND_SOURCE"] == "landsat"
            assert "bilinear" in out.tags()["AGRIBOUND_DINOV3_GRID"]
            assert out.descriptions == ("B1", "B2", "B3")
            got = out.read(1)
            ramp = src.read(1)
        # Bilinear on a linear ramp is exact away from the edges and the nodata corner:
        # the centre of copy pixel (r, c) is source position ((r + 0.5) / 3 - 0.5, ...).
        r, c = np.mgrid[30:90, 30:120]
        expected = 1000.0 + 10.0 * ((c + 0.5) / 3 - 0.5) + 5.0 * ((r + 0.5) / 3 - 0.5)
        np.testing.assert_allclose(got[30:90, 30:120], expected, rtol=0, atol=1e-2)
        assert np.isnan(got[0, 0]) and np.isnan(ramp[0, 0])
        # Written once, then reused.
        mtime = Path(path).stat().st_mtime_ns
        again, _, _ = _data.dinov3_processing_raster(raster, cfg, 10.0, [1, 2, 3])
        assert again == path and Path(path).stat().st_mtime_ns == mtime

    def test_repeated_band_is_copied_once(self, tmp_path):
        """Landsat PAN (15 m) gives B8 as R, G and B: the copy holds it once."""
        raster = _raster(tmp_path / "pan.tif", 15.0)
        path, bands, info = _data.dinov3_processing_raster(
            raster, _config(tmp_path), 10.0, [1, 1, 1]
        )
        assert bands == [1, 1, 1] and info["scale"] == 1.5 and info["size_px"] == [60, 75]
        with rasterio.open(raster) as src, rasterio.open(path) as out:
            assert out.count == 1 and out.res == (10.0, 10.0) and out.bounds == src.bounds

    def test_odd_size_without_nodata_stays_inside_the_raster(self, tmp_path):
        """15 m (Landsat PAN) with an odd size: the copy holds no pixel beyond the raster.

        A copy that covered the whole raster (rounded up) reached 5 m beyond it on the east
        and south, where the warp left 0, which counts as valid without nodata.
        """
        path = tmp_path / "pan.tif"
        values = (np.arange(41 * 51).reshape(41, 51) % 251 + 10).astype(np.uint16)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=41,
            width=51,
            count=1,
            dtype="uint16",
            crs=UTM,
            transform=from_origin(X0, Y0, 15.0, 15.0),
        ) as dst:
            dst.write(values[np.newaxis])
        copy, bands, info = _data.dinov3_processing_raster(
            str(path), _config(tmp_path), 10.0, [1, 1, 1]
        )
        assert bands == [1, 1, 1] and info["size_px"] == [61, 76]  # floor(41 x 1.5), ...
        with rasterio.open(path) as src, rasterio.open(copy) as out:
            assert out.nodata is None and out.res == (10.0, 10.0)
            assert (out.bounds.left, out.bounds.top) == (src.bounds.left, src.bounds.top)
            assert src.bounds.right - 10.0 < out.bounds.right <= src.bounds.right
            assert src.bounds.bottom <= out.bounds.bottom < src.bounds.bottom + 10.0
            data = out.read([1, 1, 1])
        assert _data.valid_pixels(data, None).all() and (data >= 10).all()

    def test_copy_key_holds_the_bands(self, tmp_path):
        """A one-band (PAN) copy is not reused for an R, G, B read of the same raster."""
        raster = _raster(tmp_path / "r.tif", 30.0, shape=(3, 8, 8))
        cfg = _config(tmp_path)
        pan, pan_bands, _ = _data.dinov3_processing_raster(raster, cfg, 10.0, [1, 1, 1])
        rgb, rgb_bands, _ = _data.dinov3_processing_raster(raster, cfg, 10.0, [3, 2, 1])
        assert pan != rgb and pan_bands == [1, 1, 1] and rgb_bands == [3, 2, 1]
        with rasterio.open(pan) as a, rasterio.open(rgb) as b:
            assert (a.count, b.count) == (1, 3)

    def test_local_rgbn_crop_keeps_rgb_where_nir_is_zero(self, tmp_path):
        """The crop of a 4-band uint8 local raster is not tagged R, G, B, alpha.

        The copy is warped by GDAL, which treated an alpha band 4 as transparency: R, G
        and B became 0 wherever NIR was 0 (and, without nodata, counted as valid).
        """
        from pyproj import Transformer

        from agribound.composites.local import LocalCompositeBuilder

        data = np.full((4, 64, 64), 120, np.uint8)
        data[3, 20:30, 20:30] = 0
        path = tmp_path / "rgbn30.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=64,
            width=64,
            count=4,
            dtype="uint8",
            crs=UTM,
            transform=from_origin(X0, Y0, 30.0, 30.0),
            ALPHA="UNSPECIFIED",  # R, G, B, undefined
        ) as dst:
            dst.write(data)
        to_ll = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
        (lon0, lon1), (lat0, lat1) = to_ll.transform([X0 + 60, X0 + 1800], [Y0 - 1800, Y0 - 60])
        cfg = AgriboundConfig(
            source="local",
            local_tif_path=str(path),
            engine="dinov3",
            study_area=f"bbox:{lon0},{lat0},{lon1},{lat1}",
            bands={"R": 1, "G": 2, "B": 3},
            output_path=str(tmp_path / "out" / "fields.gpkg"),
            lulc_filter=False,
        )
        crop = LocalCompositeBuilder().build(cfg)
        assert crop != str(path)
        copy, _, info = _data.dinov3_processing_raster(
            crop, cfg, _data.dinov3_processing_gsd(crop), [1, 2, 3]
        )
        assert info["scale"] == 3.0
        with rasterio.open(copy) as out:
            rgb = out.read()
        assert (rgb == 120).all()  # every pixel of the copy, the NIR-0 block included

    def test_no_grid_is_the_raster(self, tmp_path):
        raster = _raster(tmp_path / "r.tif", 10.0, shape=(3, 8, 8))
        out = _data.dinov3_processing_raster(raster, _config(tmp_path), None, [3, 2, 1])
        assert out == (raster, [3, 2, 1], None)
        assert not list(Path(_config(tmp_path).get_working_dir()).glob("dinov3_grid_*"))

    def test_band_rule_on_the_grid_equals_the_copy(self, tmp_path):
        raster = _raster(tmp_path / "r.tif", 30.0)
        fields = [
            box(X0 + 90, Y0 - 600, X0 + 400, Y0 - 300),
            box(X0 + 400, Y0 - 600, X0 + 520, Y0 - 300),  # 120 m wide: 4 px at 30 m
            box(X0 + 600, Y0 - 1100, X0 + 1400, Y0 - 700),
        ]
        ref = gpd.GeoDataFrame(geometry=fields, crs=UTM)
        path, _, _ = _data.dinov3_processing_raster(raster, _config(tmp_path), 10.0, [1])
        on_grid = _data.field_interior_depths(ref, raster, grid_gsd_m=10.0)
        np.testing.assert_array_equal(on_grid, _data.field_interior_depths(ref, path))
        assert (on_grid > _data.field_interior_depths(ref, raster)).all()
        band, shares = _data.widest_boundary_erosion(ref, raster, grid_gsd_m=10.0)
        assert (band, shares) == _data.widest_boundary_erosion(ref, path)
        assert finetune.dinov3_processing_gsd is _data.dinov3_processing_gsd


class TestChipsAndKeys:
    @pytest.fixture
    def scene(self, tmp_path):
        ref = tmp_path / "ref.gpkg"
        fields = [
            box(X0 + 90, Y0 - 600, X0 + 700, Y0 - 300),
            box(X0 + 90, Y0 - 1100, X0 + 1400, Y0 - 700),
            box(X0 + 900, Y0 - 600, X0 + 1400, Y0 - 150),
        ]
        gpd.GeoDataFrame(geometry=fields, crs=UTM).to_file(ref)
        return ref

    def _prepare(self, tmp_path, raster, ref, **params):
        cfg = _config(
            tmp_path,
            reference_boundaries=str(ref),
            fine_tune_split="random",
            engine_params={"chip_size": 64, "min_label_fraction": 0.0, **params},
        )
        return cfg, _data._prepare_training_data(raster, cfg, "dinov3")

    def test_coarse_raster_chips_come_from_the_copy(self, tmp_path, scene):
        raster = _raster(tmp_path / "r30.tif", 30.0, shape=(6, 50, 50))
        cfg, td = self._prepare(tmp_path, raster, scene)
        meta = _data.read_chip_meta(td)
        assert meta["gsd_m"] == 10.0 and meta["processing_gsd_m"] == 10.0
        assert meta["processing"]["scale"] == 3.0
        assert meta["processing"]["version"] == _data.DINOV3_GRID_VERSION
        # 150 x 150 px on the 10 m grid: 2 x 2 chips of 64 px, against none at 30 m
        assert meta["n_chips"] == 4
        chip = _data.split_files(td, "train", "images")[0]
        with rasterio.open(chip) as src:
            assert src.res == (10.0, 10.0) and src.dtypes[0] == "float32"
        # 'native' reads the 30 m raster as it is (no 64 px chip fits in 40 x 50 px)
        with pytest.raises(ValueError, match="training chip"):
            self._prepare(tmp_path, raster, scene, processing_gsd_m="native")

    def test_fine_raster_chips_and_keys_are_unchanged(self, tmp_path, scene):
        raster = _raster(tmp_path / "r10.tif", 10.0, shape=(6, 140, 160))
        cfg, td = self._prepare(tmp_path, raster, scene)
        meta = _data.read_chip_meta(td)
        assert meta["gsd_m"] == 10.0 and meta["processing"] is None
        assert meta["processing_gsd_m"] is None
        # The chips' key does not hold the grid ('auto' and unset give the same chips).
        _, same = self._prepare(tmp_path, raster, scene, processing_gsd_m="auto")
        assert same.name == td.name
        # The fine-tuning key of a raster read as it is does not change.
        assert finetune._run_dir(cfg, "dinov3", "large", raster) == finetune._run_dir(
            cfg, "dinov3", "large"
        )

    def test_coarse_raster_has_its_own_fine_tuning_key(self, tmp_path, scene):
        raster = _raster(tmp_path / "r30.tif", 30.0)
        cfg = _config(tmp_path, reference_boundaries=str(scene))
        with_grid = finetune._run_dir(cfg, "dinov3", "large", raster)
        assert with_grid != finetune._run_dir(cfg, "dinov3", "large")
        native = cfg.merged(engine_params={"processing_gsd_m": "native"})
        assert finetune._run_dir(native, "dinov3", "large", raster) != with_grid
        # Other engines' keys do not read the raster.
        geo = cfg.merged(engine="geoai")
        assert finetune._run_dir(geo, "geoai", "maskrcnn", raster) == finetune._run_dir(
            geo, "geoai", "maskrcnn"
        )


class TestInference:
    @pytest.fixture
    def fake_segmenter(self, monkeypatch):
        torch = pytest.importorskip("torch")
        pytest.importorskip("geoai.dinov3_finetune")
        from agribound.engines import dinov3

        calls = {"inputs": []}

        class FakeModel(torch.nn.Module):
            def forward(self, x):
                calls["inputs"].append(x.detach().clone())
                logits = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
                logits[:, 1] = torch.where(x[:, 0] > 0.5, 5.0, -5.0)
                logits[:, 2] = 1.0
                return logits

        class FakeSegmenter:
            @staticmethod
            def load_from_checkpoint(path, map_location=None, **kwargs):
                return FakeModel()

        monkeypatch.setattr(dinov3, "dinov3_segmenter_class", lambda: FakeSegmenter)
        return calls

    @staticmethod
    def _ckpt(path, meta=None, **hparams):
        import torch

        hyper = {"model_name": "dinov3_vitl16", "num_classes": 3, **hparams}
        torch.save({"hyper_parameters": hyper}, path)
        if meta is not None:
            _data.write_training_meta(path, meta)
        return str(path)

    def test_checkpoint_trained_on_the_grid_reads_it(self, tmp_path, fake_segmenter):
        from agribound.engines.dinov3 import DINOv3Engine, inference_processing_gsd

        raster = _raster(tmp_path / "r30.tif", 30.0)
        ckpt = self._ckpt(
            tmp_path / "grid.ckpt",
            {"chip_size": 64, "boundary_erosion": 1, "processing_gsd_m": 10.0},
        )
        cfg = _config(tmp_path, min_field_area_m2=0, engine_params={"checkpoint_path": ckpt})
        assert inference_processing_gsd(raster, {}, {"processing_gsd_m": 10.0}) == 10.0
        gdf = DINOv3Engine().delineate(raster, cfg)
        meta = gdf.attrs["engine_meta"]
        assert meta["processing"]["gsd_m"] == 10.0 and meta["processing"]["size_px"] == [120, 150]
        assert meta["window_size"] == 64
        assert all(x.shape[-2:] == (64, 64) for x in fake_segmenter["inputs"])
        # The bright block of R (rows 10-19, columns 12-24 at 30 m) is one field, in the
        # raster's CRS, on 10 m pixel edges.
        assert len(gdf) == 1
        minx, miny, maxx, maxy = gdf.total_bounds
        assert minx == pytest.approx(X0 + 12 * 30.0, abs=20) and maxy == pytest.approx(
            Y0 - 10 * 30.0, abs=20
        )
        assert all(round(v / 10.0, 6).is_integer() for v in (minx, miny, maxx, maxy))
        json.dumps(meta)

    @pytest.mark.parametrize("handling", ["copied without its file", "renamed with its file"])
    def test_checkpoint_without_its_metadata_file_reads_its_grid(
        self, tmp_path, fake_segmenter, handling
    ):
        """The grid is also a hyper-parameter of the checkpoint.

        ``<checkpoint>.agribound.json`` is not read when it is missing or names another
        checkpoint (a renamed one); the checkpoint then still reads its training grid.
        """
        import shutil

        from agribound.engines.dinov3 import DINOv3Engine, inference_processing_gsd

        raster = _raster(tmp_path / "r30.tif", 30.0)
        trained = self._ckpt(
            tmp_path / "epoch=22-val_miou=0.60.ckpt",
            {"chip_size": 64, "boundary_erosion": 1, "processing_gsd_m": 10.0},
            chip_size=64,
            boundary_erosion=1,
            processing_gsd_m=10.0,
        )
        moved = tmp_path / "models" / "landsat_dinov3.ckpt"
        moved.parent.mkdir()
        shutil.copy(trained, moved)
        if handling == "renamed with its file":
            shutil.copy(_data.training_meta_path(trained), _data.training_meta_path(moved))
        assert _data.read_training_meta(moved) == {}
        hparams = _data.read_checkpoint_hparams(moved)
        assert inference_processing_gsd(raster, {}, {}, hparams) == 10.0
        cfg = _config(tmp_path, min_field_area_m2=0, engine_params={"checkpoint_path": str(moved)})
        meta = DINOv3Engine().delineate(raster, cfg).attrs["engine_meta"]
        assert meta["processing"]["gsd_m"] == 10.0 and meta["window_size"] == 64
        # A checkpoint whose hyper-parameter says it was trained on its raster's own grid.
        own = self._ckpt(tmp_path / "own.ckpt", processing_gsd_m=None)
        own_hparams = _data.read_checkpoint_hparams(own)
        assert inference_processing_gsd(raster, {}, {}, own_hparams) is None

    def test_other_checkpoints_read_the_raster_as_it_is(self, tmp_path, fake_segmenter):
        from agribound.engines.dinov3 import DINOv3Engine, inference_processing_gsd

        raster = _raster(tmp_path / "r30.tif", 30.0, shape=(6, 64, 64))
        for name, meta in (("none.ckpt", None), ("native.ckpt", {"chip_size": 64})):
            ckpt = self._ckpt(tmp_path / name, meta)
            cfg = _config(tmp_path, engine_params={"checkpoint_path": ckpt})
            out = DINOv3Engine().delineate(raster, cfg).attrs["engine_meta"]
            assert "processing" not in out
        assert inference_processing_gsd(raster, {}, {}) is None
        # A raster finer than the training grid is read as it is.
        fine = _raster(tmp_path / "r6.tif", 6.0, shape=(3, 16, 16))
        assert inference_processing_gsd(fine, {}, {"processing_gsd_m": 10.0}) is None
        # engine_params decide when given.
        assert inference_processing_gsd(raster, {"processing_gsd_m": "auto"}, {}) == 10.0
        native = {"processing_gsd_m": "native"}
        assert inference_processing_gsd(raster, native, {"processing_gsd_m": 10.0}) is None
