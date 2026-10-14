"""Unit tests for the Delineate-Anything engine and its YOLO fine-tuning (no network, no GPU)."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin
from shapely.geometry import Polygon, box

from agribound.config import AgriboundConfig
from agribound.engines import delineate_anything as da
from agribound.registry import ENGINE_REGISTRY

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path, data, crs="EPSG:32611", transform=None, nodata=None):
    count, height, width = data.shape
    transform = transform or from_origin(500000, 4000000, 1.0, 1.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return str(path)


def _square_raster(tmp_path, size=512, square=(200, 300), pixel=1.0, name="rgb.tif"):
    """uint8 R, G, B raster: R=120 with a 250 square, G=100, B=50."""
    data = np.zeros((3, size, size), dtype=np.uint8)
    data[0] = 120
    data[1] = 100
    data[2] = 50
    lo, hi = square
    data[0, lo:hi, lo:hi] = 250
    return _write(tmp_path / name, data, transform=from_origin(500000, 4000000, pixel, pixel))


def _config(tmp_path, tif, **engine_params):
    return AgriboundConfig(
        source="local",
        engine="delineate-anything",
        local_tif_path=tif,
        output_path=str(tmp_path / "out" / "fields.gpkg"),
        device="cpu",
        lulc_filter=False,
        min_field_area_m2=100.0,
        engine_params=engine_params,
    )


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4)
        self.conf = torch.tensor(conf, dtype=torch.float32)


class _Masks:
    def __init__(self, data):
        self.data = data


class _Result:
    def __init__(self, masks, boxes):
        self.masks = masks
        self.boxes = boxes


class FakeYOLO:
    """Detects pixels whose R value (BGR channel 2) exceeds 200 as one field."""

    instances: list = []

    def __init__(self, weights):
        self.weights = weights
        self.calls: list = []
        FakeYOLO.instances.append(self)

    def predict(self, images, **kwargs):
        self.calls.append(([img.copy() for img in images], kwargs))
        out = []
        for img in images:
            mask = img[..., 2] > 200
            if not mask.any():
                out.append(_Result(None, _Boxes(np.zeros((0, 4)), [])))
                continue
            rows, cols = np.nonzero(mask)
            xyxy = [cols.min(), rows.min(), cols.max() + 1, rows.max() + 1]
            data = torch.from_numpy(mask[None].astype(np.uint8))  # uint8, like ultralytics
            out.append(_Result(_Masks(data), _Boxes(xyxy, [0.9])))
        return out

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        (save_dir / "weights" / "best.pt").write_bytes(b"best")
        return types.SimpleNamespace(save_dir=save_dir)


@pytest.fixture
def fake_ultralytics(monkeypatch):
    FakeYOLO.instances = []
    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    cfg = types.ModuleType("ultralytics.cfg")
    cfg.DEFAULT_CFG_DICT = {"quantize": None}
    module.cfg = cfg
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setitem(sys.modules, "ultralytics.cfg", cfg)
    return FakeYOLO


@pytest.fixture
def fake_weights(monkeypatch, tmp_path):
    requested: list[str] = []
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")

    def download(key, verify=True):
        requested.append(key)
        return str(weights)

    monkeypatch.setattr(da, "download_da_weights", download)
    return requested


# ---------------------------------------------------------------------------
# Model registry and options
# ---------------------------------------------------------------------------


def test_model_registry_pins_and_default_confidences():
    assert da.DEFAULT_DA_MODEL == "large_v2"
    v2 = da.DA_MODELS["large_v2"]
    assert v2.filename == "DelineateAnythingv2.pt"
    assert v2.revision == "369d0b4c44cf9bec2bd3a27bc81810cadd2c963e"
    assert v2.sha256.startswith("46700b8a279b") and len(v2.sha256) == 64
    assert (v2.default_conf, v2.ftw_name) == (0.15, "DelineateAnythingV2")
    assert da.DA_MODELS["large"].default_conf == 0.005
    assert da.DA_MODELS["small"].ftw_name == "DelineateAnything-S"
    assert da.DA_MODELS["large"].revision == da.DA_MODELS["small"].revision


@pytest.mark.parametrize(
    ("params", "key"),
    [
        ({}, "large_v2"),
        ({"da_model": "DelineateAnythingV2"}, "large_v2"),
        ({"da_model": "DelineateAnything"}, "large"),
        ({"da_model": "delineateanything-s"}, "small"),
        ({"da_model": "small"}, "small"),
        ({"model_size": "large"}, "large"),
        ({"model_size": "small"}, "small"),
        ({"da_model": "small", "model_size": "small"}, "small"),
    ],
)
def test_resolve_da_model_key(params, key):
    assert da.resolve_da_model_key(params) == key


def test_resolve_da_model_key_rejects_unknown_and_conflicts():
    with pytest.raises(ValueError, match="Unknown da_model"):
        da.resolve_da_model_key({"da_model": "huge"})
    with pytest.raises(ValueError, match="model_size"):
        da.resolve_da_model_key({"model_size": "medium"})
    with pytest.raises(ValueError, match="different models"):
        da.resolve_da_model_key({"da_model": "large_v2", "model_size": "small"})


def test_options_confidence_per_model_and_backend():
    assert da.DAOptions.from_engine_params({}).conf_threshold == 0.15
    assert da.DAOptions.from_engine_params({"da_model": "large"}).conf_threshold == 0.005
    ftw_v1 = da.DAOptions.from_engine_params({"backend": "ftw", "da_model": "large"})
    assert ftw_v1.conf_threshold == 0.05 and ftw_v1.conf_source == "ftw-tools default"
    assert da.DAOptions.from_engine_params({"backend": "ftw"}).conf_threshold == 0.15
    explicit = da.DAOptions.from_engine_params({"conf_threshold": 0.3})
    assert (explicit.conf_threshold, explicit.conf_source) == (0.3, "engine_params")


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"backend": "reference", "iou_threshold": 0.5}, "iou_threshold"),
        ({"backend": "native", "da_repo": "/x"}, "da_repo"),
        ({"backend": "ftw", "checkpoint_path": "/x.pt"}, "custom weights"),
        ({"backend": "ftw", "super_resolution": 2}, "super_resolution"),
        ({"backend": "onnx"}, "Unknown Delineate-Anything backend"),
        ({"super_resolution": 3}, "super_resolution"),
        ({"conf_threshold": 1.5}, "conf_threshold"),
        ({"tile_step": 0}, "tile_step"),
        ({"batch_size": 0}, "batch_size"),
        ({"half": "yes"}, "half"),
        # Legacy names for the confidence must not be silently ignored.
        ({"confidence": 0.2}, "conf_threshold"),
        ({"backend": "reference", "minimal_confidence": 0.2}, "conf_threshold"),
        ({"backend": "ftw", "patch_size": 100}, "multiple of 32"),
        ({"merge_tile_pieces": "yes"}, "merge_tile_pieces"),
        ({"backend": "reference", "merge_tile_pieces": False}, "merge_tile_pieces"),
        ({"backend": "native", "min_hole_area_m2": 100}, "min_hole_area_m2"),
        ({"backend": "reference", "min_hole_area_m2": -1}, "min_hole_area_m2"),
        ({"merge_tile_seams": 1}, "merge_tile_seams must be True or False"),
        ({"backend": "ftw", "merge_tile_seams": True}, "merge_tile_seams"),
        ({"merge_tile_seams": True, "resolve_overlaps": False}, "needs resolve_overlaps=True"),
        ({"merge_tile_seams": True, "tile_step": 0.25}, "needs tile_step=0.5"),
        # processing_gsd_m: a number within the 0.25-10 m training range; native backend only.
        ({"processing_gsd_m": 0.2}, "processing_gsd_m must be a pixel size"),
        ({"processing_gsd_m": 10.5}, "processing_gsd_m must be a pixel size"),
        ({"processing_gsd_m": float("nan")}, "processing_gsd_m must be a pixel size"),
        ({"processing_gsd_m": True}, "processing_gsd_m must be a pixel size"),
        ({"processing_gsd_m": "3"}, "processing_gsd_m must be a pixel size"),
        ({"backend": "reference", "processing_gsd_m": 3.0}, "processing_gsd_m"),
        ({"backend": "ftw", "processing_gsd_m": 3.0}, "processing_gsd_m"),
        # stretch_percentiles: (low, high) with 0 <= low < high <= 100; native backend only.
        ({"stretch_percentiles": (99, 1)}, "stretch_percentiles must be two percentiles"),
        ({"stretch_percentiles": (5, 5)}, "stretch_percentiles must be two percentiles"),
        ({"stretch_percentiles": (-1, 99)}, "stretch_percentiles must be two percentiles"),
        ({"stretch_percentiles": (1, 100.5)}, "stretch_percentiles must be two percentiles"),
        ({"stretch_percentiles": (1, float("nan"))}, "stretch_percentiles must be two percentiles"),
        ({"stretch_percentiles": [1]}, "stretch_percentiles must be two percentiles"),
        ({"stretch_percentiles": [1, 50, 99]}, "stretch_percentiles must be two percentiles"),
        ({"stretch_percentiles": (True, 99)}, "stretch_percentiles must be two percentiles"),
        ({"stretch_percentiles": "1,99"}, "stretch_percentiles must be two percentiles"),
        ({"stretch_percentiles": 99}, "stretch_percentiles must be two percentiles"),
        ({"backend": "reference", "stretch_percentiles": (1, 99)}, "stretch_percentiles"),
        ({"backend": "ftw", "stretch_percentiles": (1, 99)}, "stretch_percentiles"),
    ],
)
def test_options_reject_unsupported_or_invalid(params, match):
    with pytest.raises(ValueError, match=match):
        da.DAOptions.from_engine_params(params)


@pytest.mark.parametrize(
    ("params", "gsd", "percentiles"),
    [
        ({}, None, (1.0, 99.0)),
        ({"processing_gsd_m": None, "stretch_percentiles": None}, None, (1.0, 99.0)),
        ({"processing_gsd_m": 3}, 3.0, (1.0, 99.0)),
        ({"processing_gsd_m": 0.25}, 0.25, (1.0, 99.0)),
        ({"processing_gsd_m": np.float64(10.0)}, 10.0, (1.0, 99.0)),
        ({"stretch_percentiles": [0.1, 99.9]}, None, (0.1, 99.9)),
        ({"stretch_percentiles": (0, 100)}, None, (0.0, 100.0)),
        ({"stretch_percentiles": (np.float32(2), 98)}, None, (2.0, 98.0)),
    ],
)
def test_options_large_field_parameters(params, gsd, percentiles):
    opts = da.DAOptions.from_engine_params(params)
    assert opts.processing_gsd_m == gsd and opts.stretch_percentiles == percentiles
    assert all(type(v) is float for v in opts.stretch_percentiles)


@pytest.mark.parametrize(
    ("params", "seams"),
    [
        ({}, True),
        ({"merge_tile_seams": False}, False),
        ({"merge_tile_seams": True}, True),
        # It needs non-overlapping polygons and the half-tile step: off without them.
        ({"resolve_overlaps": False}, False),
        ({"tile_step": 0.25}, False),
    ],
)
def test_options_merge_tile_seams_default(params, seams):
    assert da.DAOptions.from_engine_params(params).merge_tile_seams is seams


def test_options_ignore_non_da_pipeline_params():
    opts = da.DAOptions.from_engine_params({"smooth_iterations": 2, "sam_refine": True})
    assert opts.backend == "native"


def test_engine_attributes_come_from_registry():
    entry = ENGINE_REGISTRY["delineate-anything"]
    engine = da.DelineateAnythingEngine()
    assert engine.supported_sources == entry["supported_sources"]
    assert "spot-pan" in engine.supported_sources
    assert "usgs-naip-plus" in engine.supported_sources
    assert engine.requires_bands == entry["requires_bands"]


# ---------------------------------------------------------------------------
# Tiling, GSD, morphology, stretch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("gsd", "sr"), [(0.6, 1), (3.99, 1), (4.0, 2), (10.0, 2), (30.0, 2)])
def test_super_resolution_rule(gsd, sr):
    assert da.select_super_resolution(gsd) == sr
    assert da.MODEL_INPUT_PX // sr in (512, 256)


def test_super_resolution_override():
    assert da.select_super_resolution(10.0, 4) == 4
    assert da.select_super_resolution(1.0, 2) == 2
    with pytest.raises(ValueError):
        da.select_super_resolution(10.0, 3)


def test_tile_origins_match_upstream_planner_and_cover_centres():
    assert da.tile_origins(1000, 512, 256) == [-256, 0, 256, 512, 768]
    assert da.tile_origins(100, 256, 128) == [-128, 0]
    for size, tile in ((1000, 512), (777, 256)):
        step = tile // 2
        origins = da.tile_origins(size, tile, step)
        for x in range(size):
            # every pixel is in the central half of some tile
            assert any(o + tile // 4 <= x < o + 3 * tile // 4 for o in origins)


def test_pixel_size_projected_and_geographic():
    tf = from_origin(500000, 4000000, 10.0, 10.0)
    assert da.pixel_size_m("EPSG:32611", tf, 100, 100) == pytest.approx(10.0)
    deg = 8.983152841195215e-05
    tf_geo = from_origin(36.0, 0.5 * 100 * deg, deg, deg)  # centred on the equator
    expected = deg * 6371000.0 * np.pi / 180.0
    assert da.pixel_size_m("EPSG:4326", tf_geo, 100, 100) == pytest.approx(expected, rel=1e-6)


def test_refine_masks_casts_uint8_before_morphology():
    mask = torch.zeros((1, 9, 9), dtype=torch.uint8)
    mask[0, 3:6, 3:6] = 1  # 3x3 square survives opening + closing
    mask[0, 0, 8] = 1  # isolated pixel is removed by the opening
    out = da.refine_masks(mask)
    assert out.dtype == torch.float32
    assert int(out.sum()) == 9
    assert float(out[0, 0, 8]) == 0.0
    # The uint8 negation trick (upstream before 34eddf7) dilates instead of eroding.
    buggy = -torch.nn.functional.max_pool2d(-mask.clone(), 3, 1, 1)
    assert int((buggy > 0).sum()) > 9


def test_stretch_with_bounds_formula_and_invalid():
    data = np.array([[[0.0, 500.0, 1000.0, 2000.0]]], dtype=np.float32)
    valid = np.array([[True, True, True, False]])
    out = da.stretch_with_bounds(data, [500.0], [1500.0], valid)
    assert out.dtype == np.uint8
    assert out.tolist() == [[[0, 0, 127, 0]]]


def test_scene_stretch_bounds(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.uniform(100, 3000, (3, 64, 64)).astype(np.float32)
    data[:, :4, :] = 0.0  # all-zero pixels are nodata
    data[:, 4:6, :] = np.nan
    path = _write(tmp_path / "f.tif", data, nodata=float("nan"))
    with rasterio.open(path) as src:
        lows, highs = da.scene_stretch_bounds(src, [3, 2, 1])
    valid = data[:, 6:, :]
    for i, band in enumerate((2, 1, 0)):
        lo, hi = np.percentile(valid[band], (1, 99))
        assert lows[i] == pytest.approx(lo, rel=1e-6)
        assert highs[i] == pytest.approx(hi, rel=1e-6)
    u8 = _write(tmp_path / "u.tif", np.full((3, 8, 8), 7, dtype=np.uint8))
    with rasterio.open(u8) as src:
        assert da.scene_stretch_bounds(src, [3, 2, 1]) == ([0.0] * 3, [255.0] * 3)


def test_sentinel1_composite_reaches_the_model_unstretched(tmp_path):
    """A sentinel1 composite (uint8 VV, VH, VV_VH; 0 = nodata) is used as is, in BGR order."""
    from agribound.engines.base import get_canonical_band_indices

    assert "sentinel1" in da.DelineateAnythingEngine().supported_sources
    data = np.zeros((3, 8, 8), dtype=np.uint8)
    data[0], data[1], data[2] = 128, 140, 61  # VV, VH, VV_VH on the 1-255 scale
    data[:, :, :2] = 0  # nodata (outside the swath)
    path = _write(tmp_path / "s1.tif", data, nodata=0)
    rgb = get_canonical_band_indices("sentinel1", ["R", "G", "B"])
    bgr = list(reversed(rgb))  # as _run_native does
    assert rgb == [1, 2, 3]
    with rasterio.open(path) as src:
        lows, highs = da.scene_stretch_bounds(src, bgr)
        tile, valid = da._read_tile(src, bgr, 0, 0, 8)
    assert (lows, highs) == ([0.0] * 3, [255.0] * 3)
    assert not valid[:, :2].any() and valid[:, 2:].all()
    out = da.stretch_with_bounds(tile, lows, highs, valid)
    np.testing.assert_array_equal(out[:, :, 2:], data[::-1, :, 2:])
    assert not out[:, :, :2].any()


def test_scene_stretch_bounds_never_samples_overviews(tmp_path, monkeypatch):
    """Averaged overviews would narrow the percentile range (upstream opens OVERVIEW_LEVEL=NONE)."""
    from rasterio.enums import Resampling

    import agribound.io.raster as io_raster

    monkeypatch.setattr(io_raster, "MAX_SAMPLE_SIDE", 64)  # 256 px raster -> decimated sample
    data = np.ones((3, 256, 256), dtype=np.float32)
    data[:, ::2, ::2] = 1000.0
    data[:, 1::2, 1::2] = 1000.0  # checkerboard of 1 and 1000
    path = _write(tmp_path / "ovr.tif", data)
    with rasterio.open(path, "r+") as dst:
        dst.build_overviews([2, 4], Resampling.average)
    with rasterio.open(path) as src:
        averaged = src.read(1, out_shape=(64, 64), resampling=Resampling.nearest)
        assert np.allclose(averaged, 500.5)  # what a plain decimated read returns
        lows, highs = da.scene_stretch_bounds(src, [3, 2, 1])
    assert lows == [1000.0] * 3 and highs == [1000.0] * 3  # full-resolution nearest sample


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------


def _gdf(geoms, conf, edge):
    return gpd.GeoDataFrame(
        {"confidence": conf, "_tile_edge": edge}, geometry=geoms, crs="EPSG:32611"
    )


def test_dedup_keeps_higher_confidence_duplicate():
    g = _gdf([box(0, 0, 10, 10), box(0.5, 0, 10.5, 10)], [0.4, 0.8], [False, False])
    out = da.deduplicate_detections(g)
    assert out["confidence"].tolist() == [0.8]


def test_dedup_prefers_complete_over_tile_cut_piece():
    full = box(0, 0, 10, 10)
    piece = box(0, 0, 4, 10)  # cut by a tile edge, contained in the full field
    g = _gdf([piece, full], [0.95, 0.6], [True, False])
    out = da.deduplicate_detections(g)
    assert len(out) == 1 and out.geometry.iloc[0].equals(full)


def test_dedup_keeps_distinct_and_weakly_overlapping_fields():
    a, b = box(0, 0, 10, 10), box(9, 0, 19, 10)  # IoU 0.053, containment 0.1
    c = box(50, 50, 60, 60)
    out = da.deduplicate_detections(_gdf([a, b, c], [0.5, 0.6, 0.7], [False] * 3))
    assert len(out) == 3


def test_dedup_containment_threshold():
    big, small = box(0, 0, 10, 10), box(1, 1, 5, 5)  # small fully inside big
    out = da.deduplicate_detections(_gdf([big, small], [0.5, 0.9], [False, False]))
    assert out["confidence"].tolist() == [0.9]
    kept = da.deduplicate_detections(
        _gdf([big, small], [0.5, 0.9], [False, False]), containment_threshold=1.01
    )
    assert len(kept) == 2


def test_resolve_overlaps_gives_contested_area_to_priority_polygon():
    strong = box(0, 0, 10, 10)
    weak = box(8, 0, 18, 10)  # 20 % overlap: not a duplicate, but overlapping
    far = box(50, 0, 60, 10)
    out = da.resolve_overlaps(_gdf([weak, strong, far], [0.4, 0.9, 0.5], [False] * 3))
    geoms = dict(zip(out["confidence"], out.geometry, strict=True))
    assert geoms[0.9].equals(strong) and geoms[0.5].equals(far)
    assert geoms[0.4].area == pytest.approx(80.0)  # lost the 2 x 10 m overlap
    assert geoms[0.4].intersection(strong).area == pytest.approx(0.0)


def test_resolve_overlaps_keeps_largest_part_and_drops_swallowed():
    strong = box(4, -5, 6, 15)  # cuts the weak polygon in two
    weak = box(0, 0, 12, 10)
    inner = box(4.5, 1, 5.5, 2)  # entirely inside the strong polygon
    out = da.resolve_overlaps(_gdf([strong, weak, inner], [0.9, 0.5, 0.3], [False] * 3))
    assert sorted(out["confidence"]) == [0.5, 0.9]
    weak_out = out.loc[out["confidence"] == 0.5].geometry.iloc[0]
    assert weak_out.geom_type == "Polygon" and weak_out.area == pytest.approx(60.0)


def test_dedup_keeps_large_tile_cut_field_over_nested_small_complete_detection():
    """A small complete detection inside a large tile-cut field must not suppress it."""
    big = box(0, 0, 600, 600)  # cut by a tile edge in every tile (e.g. 1 m NAIP)
    small = box(100, 100, 160, 160)  # complete, contained in the big field
    out = da.deduplicate_detections(_gdf([big, small], [0.9, 0.2], [True, False]))
    assert len(out) == 1 and out.geometry.iloc[0].equals(big)
    # With the higher score the small detection is kept (score first).
    out = da.deduplicate_detections(_gdf([big, small], [0.3, 0.8], [True, False]))
    assert out["confidence"].tolist() == [0.8]


def test_dedup_tile_cut_piece_survives_when_its_complete_view_is_suppressed():
    full = box(0, 0, 10, 10)  # complete view of the field
    piece = box(0, 0, 4, 10)  # tile-cut piece inside it
    rival = box(5, 0, 15, 10)  # complete, IoU 0.33 with `full`, disjoint from `piece`
    g = _gdf([full, piece, rival], [0.5, 0.9, 0.95], [False, True, False])
    out = da.deduplicate_detections(g)
    kept = sorted(out["confidence"].tolist())
    assert kept == [0.9, 0.95]  # full suppressed by rival; piece no longer duplicated


def test_merge_tile_pieces_rebuilds_a_field_larger_than_a_tile():
    # Pieces of one 0-1000 m field seen by three overlapping tiles, all tile-cut.
    pieces = [box(0, 0, 512, 400), box(256, 0, 768, 400), box(512, 0, 1000, 400)]
    other = box(2000, 0, 2100, 100)  # an unrelated complete field
    g = _gdf([*pieces, other], [0.6, 0.8, 0.7, 0.9], [True, True, True, False])
    out = da.merge_tile_pieces(g)
    assert len(out) == 2
    merged = out.loc[out["_n_pieces"] == 3].iloc[0]
    assert merged.geometry.equals(box(0, 0, 1000, 400))
    assert merged["confidence"] == 0.8 and bool(merged["_tile_edge"]) is True
    assert out.loc[out["_n_pieces"] == 1].geometry.iloc[0].equals(other)


def test_merge_tile_pieces_leaves_pieces_that_have_a_complete_view():
    full = box(0, 0, 300, 300)  # complete detection of the field
    pieces = [box(0, 0, 200, 300), box(100, 0, 300, 300)]  # tile-cut views of it
    far = [box(1000, 0, 1100, 50), box(1200, 0, 1300, 50)]  # distinct tile-cut fields
    g = _gdf([full, *pieces, *far], [0.5, 0.9, 0.9, 0.4, 0.4], [False, True, True, True, True])
    out = da.merge_tile_pieces(g)
    assert len(out) == 5 and (out["_n_pieces"] == 1).all()  # nothing merged
    dedup = da.deduplicate_detections(out)
    assert sorted(round(a) for a in dedup.geometry.area) == [5000, 5000, 90000]


def test_merge_tile_pieces_without_edge_column_is_a_no_op():
    g = gpd.GeoDataFrame({"confidence": [0.5, 0.6]}, geometry=[box(0, 0, 2, 2), box(1, 0, 3, 2)])
    out = da.merge_tile_pieces(g, edge_column="_tile_edge")
    assert len(out) == 2 and out["_n_pieces"].tolist() == [1, 1]


def test_area_m2_is_equal_area_for_web_mercator_and_feet():
    square = gpd.GeoDataFrame(geometry=[box(500000, 4980000, 501000, 4981000)], crs="EPSG:32611")
    for crs in ("EPSG:3857", "EPSG:2227", "EPSG:4326", "EPSG:32611"):
        area = da._area_m2(square.to_crs(crs))[0]
        assert area == pytest.approx(1e6, rel=2e-3), crs
    assert square.to_crs("EPSG:3857").area.iloc[0] > 1.9e6  # 45 N: sec^2 inflation


# ---------------------------------------------------------------------------
# Tile-cut flag, overlap order and tile seams
# ---------------------------------------------------------------------------


def test_resolve_overlaps_visits_complete_polygons_before_tile_cut_ones():
    """A polygon cut by its tile edge must not cut that edge into a field seen whole."""
    piece = box(800, 0, 1024, 100)  # tile-cut at x = 1024, higher confidence
    field = box(980, 0, 1150, 100)  # complete; IoU 0.13 and containment 0.26: not duplicates
    out = da.resolve_overlaps(_gdf([piece, field], [0.9, 0.5], [True, False]))
    geoms = dict(zip(out["confidence"], out.geometry, strict=True))
    assert geoms[0.5].equals(field)
    assert geoms[0.9].equals(box(800, 0, 980, 100))
    # Within each group the higher confidence still comes first.
    weak, strong = box(0, 0, 10, 10), box(8, 0, 18, 10)
    out = da.resolve_overlaps(_gdf([weak, strong], [0.4, 0.9], [True, True]))
    assert out.loc[out["confidence"] == 0.9].geometry.iloc[0].equals(strong)
    assert out.loc[out["confidence"] == 0.4].geometry.iloc[0].equals(box(0, 0, 8, 10))


def test_merged_union_stays_tile_cut_through_dedup_and_gives_way_in_resolve():
    pieces = [box(0, 0, 512, 100), box(256, 0, 700, 100)]  # IoU 0.37: one field
    field = box(650, 0, 800, 100)  # complete neighbour, overlapping the union a little
    merged = da.merge_tile_pieces(_gdf([*pieces, field], [0.9, 0.8, 0.3], [True, True, False]))
    union = merged.loc[merged["_n_pieces"] == 2].iloc[0]
    assert union.geometry.equals(box(0, 0, 700, 100)) and bool(union["_tile_edge"]) is True
    kept = da.deduplicate_detections(merged)
    assert kept["_tile_edge"].tolist() == [False, True]  # the flag reaches resolve_overlaps
    out = da.resolve_overlaps(kept)
    geoms = dict(zip(out["confidence"], out.geometry, strict=True))
    assert geoms[0.3].equals(field)  # 1.0.1 gave the overlap to the union (higher score)
    assert geoms[0.9].equals(box(0, 0, 650, 100))


def _field_raster(tmp_path, fields, shape=(512, 1024), name="fields.tif", green=None):
    """uint8 R, G, B raster (120, 100, 50) with R = 250 in the (row0, row1, col0, col1) fields."""
    data = np.zeros((3, *shape), dtype=np.uint8)
    data[0], data[1], data[2] = 120, 100, 50
    for r0, r1, c0, c1 in fields:
        data[0, r0:r1, c0:c1] = 250
    if green is not None:
        (r0, r1, c0, c1), value = green
        data[1, r0:r1, c0:c1] = value
    return _write(tmp_path / name, data)


def _raw_detections(tif, model=None, sr=1, **engine_params):
    """Rows of the native backend's _native_detections with the fake model on CPU."""
    model = model or FakeYOLO("weights")
    opts = da.DAOptions.from_engine_params(engine_params)
    with rasterio.open(tif) as src:
        lows, highs = da.scene_stretch_bounds(src, [3, 2, 1])
        rows, _ = da._native_detections(model, src, [3, 2, 1], lows, highs, sr, opts, "cpu")
    return {(r["_tile_x0"], r["_tile_y0"]): r for r in rows}


@pytest.mark.parametrize(
    ("start", "flagged"), [(256, True), (258, True), (259, True), (260, False)]
)
def test_tile_cut_flag_uses_the_4_px_edge_strip(tmp_path, start, flagged):
    """Upstream marks merging_edge_width (4 px) along the tile edges; masks stop 2-3 px short."""
    tif = _field_raster(tmp_path, [(100, 200, start, 400)])
    rows = _raw_detections(tif)
    # The tile starting at x = 256 sees the field from its column start - 256 (an interior side).
    assert rows[(256, 0)]["_tile_edge"] is flagged
    assert rows[(0, 0)]["_tile_edge"] is False  # complete in the tile starting at x = 0


@pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
@pytest.mark.parametrize(("gap", "flagged"), [(0, True), (2, True), (3, True), (4, False)])
def test_tile_cut_flag_uses_the_4_px_strip_on_every_interior_side(tmp_path, side, gap, flagged):
    """The tile at (256, 256) of a 1024 px raster has four interior sides (512 px tiles)."""
    near, far = 256 + gap, 768 - gap  # *gap* px inside the tile's left/top or right/bottom side
    field = {
        "left": (400, 500, near, 600),
        "right": (400, 500, 400, far),
        "top": (near, 600, 400, 500),
        "bottom": (400, far, 400, 500),
    }[side]  # (row0, row1, col0, col1), at least 144 px from the tile's other sides
    tif = _field_raster(tmp_path, [field], shape=(1024, 1024))
    assert _raw_detections(tif)[(256, 256)]["_tile_edge"] is flagged


def test_tile_cut_flag_follows_the_kept_polygon(tmp_path):
    """A minor part of the mask at a tile edge does not make the detection tile-cut."""
    tif = _field_raster(tmp_path, [(100, 200, 300, 450), (150, 158, 256, 264)])  # field + blob
    row = _raw_detections(tif)[(256, 0)]  # one detection: the field and the blob at its edge
    assert row["geometry"].area == pytest.approx(150 * 100)  # the largest polygon is kept
    assert row["_tile_edge"] is False  # 1.0.1 flagged it through the blob


def test_raw_detections_record_their_tile_origin_in_model_input_pixels(tmp_path):
    tif = _square_raster(tmp_path, size=256, square=(100, 120), pixel=10.0)  # 256 px tiles
    rows = _raw_detections(tif, sr=2)
    # Native origins -128 and 0 on both axes, upsampled 2x.
    assert set(rows) == {(-256, -256), (-256, 0), (0, -256), (0, 0)}


def test_compose_tile_view_writes_smaller_fields_over_larger_ones():
    tf = from_origin(0, 512, 1.0, 1.0)

    def px(c0, r0, c1, r1):
        return box(c0, 512 - r1, c1, 512 - r0)

    big = px(0, 0, 300, 300)
    small = px(100, 100, 160, 160)  # 3600 px, inside big
    twin = px(0, 0, 300, 290)  # IoU 0.97 with big: the same field, it takes big's id
    tiny = px(400, 400, 420, 420)  # 400 px < 512: dropped
    view = da._compose_tile_view([big, small, twin, tiny], tf)
    assert view[120, 120] != view[200, 200] and view[120, 120] > 0
    assert view[295, 10] == view[200, 200] > 0  # big's id, also where only big reaches
    assert view[410, 410] == 0
    assert set(np.unique(view).tolist()) == {0, int(view[120, 120]), int(view[200, 200])}


def test_compose_tile_view_leaves_out_detections_under_512_px_even_when_they_match():
    """A 500 px detection is left out although its IoU with the visible part of a larger one is
    0.81 (> compose_merge_iou): it would otherwise take that one's id, also outside it."""
    tf = from_origin(0, 512, 1.0, 1.0)

    def px(c0, r0, c1, r1):
        return box(c0, 512 - r1, c1, 512 - r0)

    big = px(0, 0, 40, 28)  # 1120 px; 560 px stay visible under `over`
    over = px(0, 0, 20, 34)  # 680 px, IoU 0.45 with big: its own id
    small = px(21, 3, 41, 28)  # 500 px: 475 on big's visible part, column 40 outside it
    view = da._compose_tile_view([big, over, small], tf)
    assert view[10, 30] > 0 and view[10, 10] > 0 and view[10, 30] != view[10, 10]
    assert view[10, 40] == 0


def test_model_input_reader_matches_the_tiles_inside_them(tmp_path):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    data = rng.uniform(100, 3000, (3, 48, 64)).astype(np.float32)
    tif = _write(tmp_path / "f.tif", data, transform=from_origin(500000, 4000000, 10.0, 10.0))
    with rasterio.open(tif) as src:
        lows, highs = da.scene_stretch_bounds(src, [3, 2, 1])
        read = da._model_input_reader(src, [3, 2, 1], lows, highs, 2)
        tile, valid = da._read_tile(src, [3, 2, 1], 8, 4, 32)  # a 32 px tile at (8, 4)
        img = np.ascontiguousarray(
            np.transpose(da.stretch_with_bounds(tile, lows, highs, valid), (1, 2, 0))
        )
        expected = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        # Model-input pixels 24-39 of the tile, away from its own edges.
        window = read(16 + 24, 8 + 20, 16 + 40, 8 + 36)
        np.testing.assert_array_equal(window, expected[20:36, 24:40])
        same = da._model_input_reader(src, [3, 2, 1], lows, highs, 1)(5, 6, 9, 8)
        tile1, valid1 = da._read_tile(src, [3, 2, 1], 5, 6, 4, 2)
        np.testing.assert_array_equal(
            same, np.transpose(da.stretch_with_bounds(tile1, lows, highs, valid1), (1, 2, 0))
        )


TF = from_origin(500000, 4000000, 1.0, 1.0)  # model-input pixels of 1 m


def _px(c0, r0, c1, r1, tf=TF):
    """Box over model-input pixels [c0, c1) x [r0, r1) of *tf*."""
    return box(tf.c + c0 * tf.a, tf.f + r1 * tf.e, tf.c + c1 * tf.a, tf.f + r0 * tf.e)


def _seams(polygons, edges, raw=None, image=None, shape=(1024, 1024), tf=TF, conf=None):
    gdf = _gdf(polygons, conf or [0.5 + 0.01 * i for i in range(len(polygons))], edges)
    return da.merge_tile_seams(gdf, raw, tf, shape, image)


@pytest.mark.parametrize("pixel", [1.0, 1.5, 0.5])
def test_merge_tile_seams_joins_pieces_cut_by_a_tile_edge(pixel):
    tf = from_origin(500000, 4000000, pixel, pixel)
    left, right = _px(300, 100, 512, 300, tf), _px(512, 100, 700, 300, tf)  # x = 512 is an edge
    other = _px(800, 800, 900, 900, tf)
    out, n = _seams([left, right, other], [True, True, True], tf=tf, conf=[0.6, 0.8, 0.7])
    assert n == 1 and len(out) == 2
    joined = out.iloc[-1]  # unjoined rows first, then the joined group
    assert joined.geometry.equals(_px(300, 100, 700, 300, tf))
    assert joined["confidence"] == 0.8 and bool(joined["_tile_edge"]) is True
    assert out.iloc[0].geometry.equals(other)


def test_merge_tile_seams_joins_across_horizontal_edges_and_transitively():
    top, middle = _px(100, 300, 300, 512), _px(100, 512, 300, 768)  # y = 512, then y = 768
    bottom = _px(100, 768, 300, 900)
    out, n = _seams([top, middle, bottom], [True, True, True])
    assert n == 2 and len(out) == 1 and out.geometry.iloc[0].equals(_px(100, 300, 300, 900))


@pytest.mark.parametrize(
    ("left", "right", "edges", "joined"),
    [
        ((300, 100, 512, 300), (512, 100, 700, 300), (True, False), False),  # complete
        ((100, 100, 300, 300), (300, 100, 450, 300), (True, True), False),  # x = 300: no edge
        ((300, 100, 512, 106), (512, 100, 700, 106), (True, True), False),  # contact 6 < 8 px
        ((300, 0, 512, 400), (512, 0, 700, 100), (True, True), False),  # coverage 0.25 < 0.3
        ((300, 100, 512, 300), (512, 100, 700, 220), (True, True), False),  # ends 80 px apart
        ((300, 100, 512, 300), (512, 100, 524, 220), (True, True), True),  # ... but a 12 px strip
        ((300, 100, 512, 300), (512, 104, 700, 300), (True, True), True),  # ends 4 px apart
        # ends 20 px apart: more than 8 px, but within 0.25 x the 180 px contact
        ((300, 100, 512, 300), (512, 120, 700, 300), (True, True), True),
        # a 12 px strip along 100 px: 33 % of the longer cut edge (300 px), but 25 % of 400 px
        ((300, 0, 512, 300), (512, 0, 524, 100), (True, True), True),
        ((300, 0, 512, 400), (512, 0, 524, 100), (True, True), False),
    ],
)
def test_merge_tile_seams_geometry_tests(left, right, edges, joined):
    _, n = _seams([_px(*left), _px(*right)], list(edges))
    assert n == int(joined)


@pytest.mark.parametrize(
    ("block", "crossing", "split"),
    [
        ([[2, 2, 2, 2]], True, False),  # one field across the window
        ([[2, 2, 4, 4]], False, True),  # two fields
        ([[2, 0, 0, 2]], False, True),  # a gap within one field: a boundary
        ([[2, 2, 0, 0]], False, False),  # a field on one side only
        ([[0, 0, 0, 0]], False, False),  # no field
    ],
)
def test_seam_evidence(block, crossing, split):
    """A four-column window (seam_split_window_px = 2; the default 1 px gives two columns)."""
    got = da._seam_evidence(np.asarray(block, dtype=np.int32))
    assert (bool(got[0][0]), bool(got[1][0])) == (crossing, split)


def test_merge_tile_seams_joins_pieces_whose_shared_edge_differs_by_one_ulp():
    """Pieces from two tiles get their shared edge from different affine compositions."""
    x = TF.c + 512 * TF.a
    left = box(TF.c + 300, TF.f - 300, x, TF.f - 100)
    right = box(np.nextafter(x, np.inf), TF.f - 300, TF.c + 700, TF.f - 100)
    assert left.union(right).geom_type == "MultiPolygon"  # one ulp apart
    out, n = _seams([left, right], [True, True])
    assert n == 1 and len(out) == 1
    assert out.geometry.iloc[0].geom_type == "Polygon"
    assert out.geometry.iloc[0].area == pytest.approx(400 * 200)


def test_merge_tile_seams_never_drops_a_piece_that_the_union_leaves_apart():
    """A piece 0.3 px away from the edge still touches it on the pixel grid of the seam test;
    the union cannot connect it, so it is kept as it was and no join is counted."""
    left, right = _px(300, 100, 512, 300), _px(512.3, 100, 700, 300)
    other = _px(800, 800, 900, 900)
    out, n = _seams([left, right, other], [True, True, False])
    assert n == 0 and len(out) == 3  # unchanged, in the original order
    assert [g.equals(h) for g, h in zip(out.geometry, [left, right, other], strict=True)] == [
        True
    ] * 3
    # In a group of three, the two pieces that the union connects are still joined.
    top, middle = _px(100, 300, 300, 512), _px(100, 512, 300, 768)
    bottom = _px(100, 768.3, 300, 900)
    out, n = _seams([top, middle, bottom], [True, True, True])
    assert n == 1 and len(out) == 2
    assert out.geometry.iloc[0].equals(bottom)
    assert out.geometry.iloc[1].equals(_px(100, 300, 300, 768))


def test_merge_tile_seams_never_joins_a_complete_polygon():
    """A tile saw a complete polygon whole: its outline is the model's, not the tiling's."""
    pieces = [_px(300, 100, 512, 300), _px(512, 100, 700, 300)]  # cut by x = 512
    complete, piece = _px(300, 500, 512, 700), _px(512, 500, 700, 700)  # on the same edge
    out, n = _seams([*pieces, complete, piece], [True, True, False, True])
    assert n == 1 and len(out) == 3
    assert out.iloc[0].geometry.equals(complete) and out.iloc[1].geometry.equals(piece)


def test_merge_tile_seams_keeps_each_piece_to_its_longest_contact():
    strip = _px(500, 100, 512, 200)  # 12 px deep: its ends need not match
    long_, short = _px(512, 100, 700, 160), _px(512, 160, 700, 200)  # contacts 60 and 40 px
    out, n = _seams([strip, long_, short], [True, True, True])
    assert n == 1
    assert out.iloc[0].geometry.equals(short) and out.iloc[1].geometry.equals(strip.union(long_))


def _raw(*detections):
    """Raw detections ((col0, row0, col1, row1), (tile_x0, tile_y0)) in model-input pixels."""
    geoms = [_px(*b) for b, _ in detections]
    return gpd.GeoDataFrame(
        {
            "_tile_x0": [t[0] for _, t in detections],
            "_tile_y0": [t[1] for _, t in detections],
        },
        geometry=geoms,
        crs="EPSG:32611",
    )


@pytest.mark.parametrize(
    ("raw", "joined"),
    [
        (None, True),  # no view: the geometry decides
        # The tile centred on x = 512 (it starts at x = 256) saw two fields meeting there.
        (_raw(((300, 140, 512, 360), (256, 0)), ((512, 140, 720, 360), (256, 0))), False),
        # ... or one field across it.
        (
            _raw(
                ((300, 140, 720, 360), (256, 0)),
            ),
            True,
        ),
        # Only that tile counts: the tile starting at x = 0 has the edge at its side.
        (_raw(((300, 140, 512, 360), (0, 0)), ((512, 140, 720, 360), (0, 0))), True),
    ],
)
def test_merge_tile_seams_asks_the_tile_centred_on_the_edge(raw, joined):
    pieces = [_px(300, 150, 512, 350), _px(512, 150, 700, 350)]
    _, n = _seams(pieces, [True, True], raw=raw)
    assert n == int(joined)


@pytest.mark.parametrize(("right_value", "joined"), [(60, True), (200, False)])
def test_merge_tile_seams_compares_the_colour_on_both_sides(right_value, joined):
    img = np.full((1024, 1024, 3), 50, dtype=np.uint8)
    img[:, 512:] = right_value  # Euclidean distance of the means / 255 vs 0.2

    def image(col0, row0, col1, row1):
        return img[row0:row1, col0:col1]

    pieces = [_px(300, 100, 512, 300), _px(512, 100, 700, 300)]
    _, n = _seams(pieces, [True, True], image=image)
    assert n == int(joined)


@pytest.mark.parametrize(
    ("raw", "joined"),
    [
        # The tile centred on y = 512 (it starts at y = 256) saw two fields meeting there.
        (_raw(((140, 300, 360, 512), (0, 256)), ((140, 512, 360, 720), (0, 256))), False),
        (
            _raw(
                ((140, 300, 360, 720), (0, 256)),
            ),
            True,
        ),  # ... or one field across it
        # The tile starting at x = 256, y = 0 is not centred on this edge.
        (_raw(((140, 300, 360, 512), (256, 0)), ((140, 512, 360, 720), (256, 0))), True),
    ],
)
def test_merge_tile_seams_asks_the_tile_centred_on_a_horizontal_edge(raw, joined):
    pieces = [_px(150, 300, 350, 512), _px(150, 512, 350, 700)]
    _, n = _seams(pieces, [True, True], raw=raw)
    assert n == int(joined)


@pytest.mark.parametrize(("split_from", "joined"), [(250, True), (230, False)])
def test_merge_tile_seams_needs_one_field_in_half_of_the_centre_tile_view(split_from, joined):
    """Rows 150-349 along x = 512: one field across up to *split_from*, two fields below."""
    raw = _raw(
        ((300, 150, 720, split_from), (256, 0)),
        ((300, split_from, 512, 350), (256, 0)),
        ((512, split_from, 720, 350), (256, 0)),
    )
    pieces = [_px(300, 150, 512, 350), _px(512, 150, 720, 350)]
    _, n = _seams(pieces, [True, True], raw=raw)
    assert n == int(joined)  # one field on 100 of 200 rows: joined; on 80 of 200: not


def test_merge_tile_seams_asks_each_position_the_tile_whose_central_half_holds_it():
    """Rows 100-127 of x = 512 are in the central half of the tile at y = -256 (two fields),
    rows 128-383 in that of the tile at y = 0 (one field): one field on 256 of 284 rows."""
    raw = _raw(
        ((300, 100, 720, 400), (256, 0)),
        ((300, 100, 512, 250), (256, -256)),
        ((512, 100, 720, 250), (256, -256)),
    )
    pieces = [_px(300, 100, 512, 400), _px(512, 100, 720, 400)]
    _, n = _seams(pieces, [True, True], raw=raw)
    assert n == 1


def test_merge_tile_seams_compares_the_colour_on_the_contact_rows_only():
    img = np.full((1024, 1024, 3), 50, dtype=np.uint8)
    img[:100, 512:] = img[300:, 512:] = 200  # other colours beyond the contact (rows 100-299)

    def image(col0, row0, col1, row1):
        return img[row0:row1, col0:col1]

    pieces = [_px(300, 100, 512, 300), _px(512, 100, 700, 300)]
    _, n = _seams(pieces, [True, True], image=image)
    assert n == 1


@pytest.mark.parametrize(("below_value", "joined"), [(60, True), (200, False)])
def test_merge_tile_seams_compares_the_colour_across_a_horizontal_edge(below_value, joined):
    img = np.full((1024, 1024, 3), 50, dtype=np.uint8)
    img[512:] = below_value

    def image(col0, row0, col1, row1):
        return img[row0:row1, col0:col1]

    pieces = [_px(100, 300, 300, 512), _px(100, 512, 300, 768)]
    _, n = _seams(pieces, [True, True], image=image)
    assert n == int(joined)


# ---------------------------------------------------------------------------
# Native backend (fake ultralytics model)
# ---------------------------------------------------------------------------


def test_native_backend_bgr_order_tiling_confidence_and_dedup(
    tmp_path, fake_ultralytics, fake_weights
):
    tif = _square_raster(tmp_path)
    config = _config(tmp_path, tif)
    gdf = da.DelineateAnythingEngine().delineate(tif, config)

    model = fake_ultralytics.instances[0]
    images = [img for call in model.calls for img in call[0]]
    kwargs = model.calls[0][1]
    assert fake_weights == ["large_v2"]
    assert len(images) == 9  # origins -256, 0, 256 on both axes (1 m GSD: 512 px tiles)
    assert all(img.shape == (512, 512, 3) and img.dtype == np.uint8 for img in images)
    # BGR: channel 0 = blue (50), channel 2 = red (120 / 250)
    full = [img for img in images if (img[..., 2] == 250).sum() == 100 * 100]
    assert full and full[0][..., 0].max() == 50 and full[0][..., 1].max() == 100
    assert kwargs["conf"] == 0.15 and kwargs["iou"] == 0.3 and kwargs["max_det"] == 300
    assert kwargs["retina_masks"] is True and kwargs["imgsz"] == 512
    assert "quantize" not in kwargs and "half" not in kwargs  # FP32 on CPU

    assert len(gdf) == 1  # duplicates from overlapping tiles removed
    geom = gdf.geometry.iloc[0]
    assert geom.area == pytest.approx(100 * 100)
    assert geom.bounds == pytest.approx((500200.0, 4000000 - 300.0, 500300.0, 4000000 - 200.0))
    meta = gdf.attrs["engine_meta"]
    assert meta["backend"] == "native" and meta["model_key"] == "large_v2"
    assert meta["super_resolution"] == 1 and meta["tile_size_native_px"] == 512
    assert meta["band_indices_bgr"] == [3, 2, 1]
    assert meta["n_raw"] > meta["n_after_dedup"] == meta["n_output"] == 1
    assert meta["resolve_overlaps"] is True
    json.dumps(meta)  # JSON-serialisable


def test_native_backend_rebuilds_field_larger_than_a_tile(tmp_path, fake_ultralytics, fake_weights):
    # 700 x 700 px field at 1 m GSD: every 512 px tile cuts it, so no single
    # detection is complete; the tile-cut pieces must be merged back together.
    tif = _square_raster(tmp_path, size=1024, square=(100, 800))
    gdf = da.DelineateAnythingEngine().delineate(tif, _config(tmp_path, tif))
    meta = gdf.attrs["engine_meta"]
    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].area == pytest.approx(700 * 700)
    assert meta["n_merged_groups"] == 1 and meta["n_pieces_merged"] >= 4
    assert meta["area_crs"] == "EPSG:6933"
    unmerged = da.DelineateAnythingEngine().delineate(
        tif, _config(tmp_path, tif, merge_tile_pieces=False)
    )
    # Without merging the field is split into tile-sized pieces.
    assert len(unmerged) > 1 and unmerged.geometry.area.max() <= 512 * 512


class WideBlindYOLO(FakeYOLO):
    """FakeYOLO that misses a field running across its whole tile (side to side)."""

    def predict(self, images, **kwargs):
        out = super().predict(images, **kwargs)
        for i, img in enumerate(images):
            mask = img[..., 2] > 200
            if mask[:, 0].any() and mask[:, -1].any():
                out[i] = _Result(None, _Boxes(np.zeros((0, 4)), []))
        return out


@pytest.mark.parametrize(("green", "n_out"), [(None, 1), (((0, 512, 512, 1024), 230), 2)])
def test_native_backend_joins_pieces_that_meet_at_a_tile_edge(
    tmp_path, fake_ultralytics, fake_weights, monkeypatch, green, n_out
):
    """The tile centred on x = 512 misses the field: its pieces meet there without overlapping."""
    monkeypatch.setattr(sys.modules["ultralytics"], "YOLO", WideBlindYOLO)
    # A 770 x 100 m field; with green, the image differs on the two sides of x = 512.
    tif = _field_raster(tmp_path, [(100, 200, 130, 900)], green=green)
    gdf = da.DelineateAnythingEngine().delineate(tif, _config(tmp_path, tif))
    meta = gdf.attrs["engine_meta"]
    assert meta["n_after_dedup"] == 2 and len(gdf) == n_out
    assert meta["n_seam_joins"] == 2 - n_out
    assert meta["implementation"] == "agribound-native-4" and meta["merge_tile_seams"] is True
    assert meta["tile_edge_strip_px"] == 4 and meta["tile_seam_params"]["seam_min_contact_px"] == 8
    assert set(gdf.columns) == {"confidence", "geometry"}  # no internal columns
    if n_out == 1:
        assert gdf.geometry.iloc[0].area == pytest.approx(770 * 100)
        off = da.DelineateAnythingEngine().delineate(
            tif, _config(tmp_path, tif, merge_tile_seams=False)
        )
        assert len(off) == 2 and off.attrs["engine_meta"]["n_seam_joins"] == 0
        assert off.attrs["engine_meta"]["tile_seam_params"] is None
        assert len(FakeYOLO.instances) == 2  # merge_tile_seams is part of the cache key


class CentreBlindYOLO(FakeYOLO):
    """FakeYOLO that misses the field in a tile whose central half holds field pixels marked
    green 77, along its columns (``axis = 1``) or rows (``axis = 0``): the tile centred on the
    edge the marker straddles. Its neighbours see the field cut by that edge."""

    axis = 1

    def predict(self, images, **kwargs):
        out = super().predict(images, **kwargs)
        for i, img in enumerate(images):
            marked = (np.abs(img[..., 1].astype(int) - 77) <= 2) & (img[..., 2] > 200)
            idx = np.nonzero(marked)[self.axis]
            if ((idx >= 128) & (idx < 384)).any():
                out[i] = _Result(None, _Boxes(np.zeros((0, 4)), []))
        return out


class CentreBlindRowsYOLO(CentreBlindYOLO):
    axis = 0


def _marker_raster(path, shape, field, marker, transform, crs="EPSG:32611"):
    """uint8 R, G, B raster: R = 250 in *field*, G = 77 in *marker* (row0, row1, col0, col1)."""
    data = np.zeros((3, *shape), dtype=np.uint8)
    data[0], data[1], data[2] = 120, 100, 50
    r0, r1, c0, c1 = field
    data[0, r0:r1, c0:c1] = 250
    r0, r1, c0, c1 = marker
    data[1, r0:r1, c0:c1] = 77
    return _write(path, data, crs=crs, transform=transform)


_NAIP_PLUS_TF = Affine(
    0.5999433618607072, 0, -13341640.971573837, 0, -0.5999433618607072, 4408485.827829681
)


@pytest.mark.parametrize(
    ("transform", "crs", "seam", "params", "scale"),
    [
        (from_origin(512345.67, 4000000.0, 1.2, 1.2), "EPSG:32611", 256, {}, 1.0),
        (_NAIP_PLUS_TF, "EPSG:3857", 256, {}, 1.0),  # usgs-naip-plus grid
        (_NAIP_PLUS_TF, "EPSG:3857", 1536, {}, 1.0),
        # processing_gsd_m: a 2.4 m copy of a 1.5 m raster (1.6 raster pixels per copy pixel)
        (
            from_origin(661660.5, 614878.5, 1.5, 1.5),
            "EPSG:32630",
            256,
            {"processing_gsd_m": 2.4},
            1.6,
        ),
    ],
)
def test_native_backend_joins_seam_pieces_on_grids_not_exact_in_binary(
    tmp_path, fake_ultralytics, fake_weights, monkeypatch, transform, crs, seam, params, scale
):
    """The tiles next to the seam compute its coordinates with different affine compositions;
    agribound 1.0.1's union then lost the smaller piece (52 % of the field kept)."""
    monkeypatch.setattr(sys.modules["ultralytics"], "YOLO", CentreBlindYOLO)

    def s(v):  # model-input (copy) pixels -> raster pixels
        return int(round(v * scale))

    shape = (s(512), s(seam + 512))
    field = (s(150), s(350), s(seam - 156), s(seam + 144))  # 300 x 200 px across the seam
    marker = (s(150), s(350), s(seam - 8), s(seam + 8))
    tif = _marker_raster(tmp_path / "f.tif", shape, field, marker, transform, crs)
    config = _config(tmp_path, tif, **params).merged(min_field_area_m2=1.0)
    gdf = da.DelineateAnythingEngine().delineate(tif, config)
    meta = gdf.attrs["engine_meta"]
    assert meta["n_after_dedup"] == 2 and meta["n_seam_joins"] == 1 and len(gdf) == 1
    pixel = abs(transform.a) * scale
    assert gdf.geometry.iloc[0].area == pytest.approx(300 * 200 * pixel**2)


@pytest.mark.parametrize("vertical", [True, False])
def test_native_backend_joins_seam_pieces_at_super_resolution_2(
    tmp_path, fake_ultralytics, fake_weights, monkeypatch, vertical
):
    """10 m pixels: 256 px native tiles every 128 px, upsampled 2x. The pieces meet at native
    128 (model-input 256), and the field lies beyond native 256 across the edge, so the seam
    test must work on the model-input grid (transform and shape) along both axes."""
    model = CentreBlindYOLO if vertical else CentreBlindRowsYOLO
    monkeypatch.setattr(sys.modules["ultralytics"], "YOLO", model)
    shape, field, marker = (512, 768), (300, 400, 40, 230), (300, 400, 120, 136)
    if not vertical:  # the same layout transposed
        shape, field, marker = shape[::-1], (40, 230, 300, 400), (120, 136, 300, 400)
    tif = _marker_raster(
        tmp_path / "f.tif", shape, field, marker, from_origin(500000, 4000000, 10.0, 10.0)
    )
    gdf = da.DelineateAnythingEngine().delineate(tif, _config(tmp_path, tif))
    meta = gdf.attrs["engine_meta"]
    assert meta["super_resolution"] == 2
    assert meta["n_after_dedup"] == 2 and meta["n_seam_joins"] == 1 and len(gdf) == 1
    # 190 x 100 native px; the bicubic upsampling blurs the outline by about a model pixel
    assert gdf.geometry.iloc[0].area == pytest.approx(190 * 100 * 100.0, rel=0.03)


class ComponentYOLO(FakeYOLO):
    """FakeYOLO with one detection per 4-connected component of R > 200."""

    def predict(self, images, **kwargs):
        from scipy import ndimage

        self.calls.append(([img.copy() for img in images], kwargs))
        out = []
        for img in images:
            labels, n = ndimage.label(img[..., 2] > 200)
            if n == 0:
                out.append(_Result(None, _Boxes(np.zeros((0, 4)), [])))
                continue
            masks = np.stack([labels == i for i in range(1, n + 1)])
            boxes = []
            for mask in masks:
                rows, cols = np.nonzero(mask)
                boxes.append([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1])
            data = torch.from_numpy(masks.astype(np.uint8))
            out.append(_Result(_Masks(data), _Boxes(boxes, [0.9] * n)))
        return out


def test_native_backend_gives_the_seam_test_every_raw_detection(
    tmp_path, fake_ultralytics, fake_weights, monkeypatch
):
    """The views of the tiles centred on the edges are composed from every raw detection,
    including those the area filter removes (upstream composes before its area filter)."""
    monkeypatch.setattr(sys.modules["ultralytics"], "YOLO", ComponentYOLO)
    seen = {}
    real = da.merge_tile_seams

    def spy(gdf, raw, *args, **kwargs):
        seen["raw"] = raw
        return real(gdf, raw, *args, **kwargs)

    monkeypatch.setattr(da, "merge_tile_seams", spy)
    # Two 100 x 100 m fields and a 30 x 30 m one (900 m2 < 1000 m2).
    tif = _field_raster(tmp_path, [(100, 200, 100, 200), (300, 400, 300, 400), (20, 50, 400, 430)])
    gdf = da.DelineateAnythingEngine().delineate(
        tif, _config(tmp_path, tif).merged(min_field_area_m2=1000.0)
    )
    meta = gdf.attrs["engine_meta"]
    assert len(gdf) == 2 and meta["n_after_min_area"] < meta["n_raw"]
    assert len(seen["raw"]) == meta["n_raw"]
    assert (seen["raw"].geometry.area < 1000).any()


def test_native_backend_upsamples_coarse_imagery(tmp_path, fake_ultralytics, fake_weights):
    tif = _square_raster(tmp_path, size=256, square=(100, 120), pixel=10.0)
    config = _config(tmp_path, tif, da_model="small")
    gdf = da.DelineateAnythingEngine().delineate(tif, config)
    model = fake_ultralytics.instances[0]
    assert fake_weights == ["small"]
    assert model.calls[0][1]["conf"] == 0.005  # per-model default
    assert all(img.shape == (512, 512, 3) for call in model.calls for img in call[0])
    meta = gdf.attrs["engine_meta"]
    assert (meta["super_resolution"], meta["tile_size_native_px"]) == (2, 256)
    assert len(gdf) == 1
    # 20 x 20 px at 10 m; bicubic upsampling blurs the edge by about one output pixel
    assert gdf.geometry.iloc[0].area == pytest.approx(200 * 200, rel=0.1)
    assert meta["gsd_outside_training_range"] is False  # 10 m is inside 0.25-10 m


@pytest.mark.parametrize(
    ("gsd", "outside"),
    [(0.2, True), (0.25, False), (0.6, False), (9.99, False), (10.4, False), (10.6, True)],
)
def test_training_gsd_range(gsd, outside):
    assert da.gsd_outside_training_range(gsd) is outside


def test_native_backend_flags_imagery_outside_training_gsd(
    tmp_path, fake_ultralytics, fake_weights, caplog
):
    tif = _square_raster(tmp_path, size=256, square=(100, 120), pixel=30.0)  # Landsat-like
    with caplog.at_level("WARNING"):
        gdf = da.DelineateAnythingEngine().delineate(tif, _config(tmp_path, tif))
    assert gdf.attrs["engine_meta"]["gsd_outside_training_range"] is True
    assert "trained on 0.25-10 m imagery" in caplog.text


def test_native_backend_caches_result(tmp_path, fake_ultralytics, fake_weights):
    tif = _square_raster(tmp_path)
    config = _config(tmp_path, tif)
    first = da.DelineateAnythingEngine().delineate(tif, config)
    second = da.DelineateAnythingEngine().delineate(tif, config)
    assert len(fake_ultralytics.instances) == 1  # second call did not load the model
    assert second.attrs["engine_meta"]["cached_result"]
    assert second.geometry.iloc[0].equals(first.geometry.iloc[0])
    other = _config(tmp_path, tif, conf_threshold=0.5)
    da.DelineateAnythingEngine().delineate(tif, other)
    assert len(fake_ultralytics.instances) == 2  # a different conf is a different cache key


def test_native_backend_uses_checkpoint(tmp_path, fake_ultralytics, fake_weights):
    tif = _square_raster(tmp_path)
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"finetuned")
    config = _config(tmp_path, tif, checkpoint_path=str(ckpt))
    gdf = da.DelineateAnythingEngine().delineate(tif, config)
    assert fake_ultralytics.instances[0].weights == str(ckpt.resolve())
    assert fake_weights == []
    meta = gdf.attrs["engine_meta"]
    assert meta["weights"] == "checkpoint"
    assert meta["checkpoint_sha256"] == hashlib.sha256(b"finetuned").hexdigest()
    assert "checkpoint_input_mismatch" not in meta  # no training metadata: not checked


# ---------------------------------------------------------------------------
# Large fields: processing_gsd_m and stretch_percentiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raster", "processing", "scale"),
    [(1.5, 3.0, 2.0), (1.5, 1.5, 1.0), (1.5, 1.51, 1.0), (1.5, 1.49, 1.0), (0.6, 1.5, 2.5)],
)
def test_processing_scale(raster, processing, scale):
    assert da.processing_scale(raster, processing) == pytest.approx(scale)


@pytest.mark.parametrize(("raster", "processing"), [(1.5, 1.4), (10.0, 3.0)])
def test_processing_scale_refuses_finer_pixels(raster, processing):
    with pytest.raises(ValueError, match="finer than the raster's pixels"):
        da.processing_scale(raster, processing)


@pytest.mark.parametrize(("n_src", "scale"), [(10, 2.0), (9, 2.0), (17, 1.6), (13, 3.0)])
def test_area_weights_spread_every_raster_pixel_once(n_src, scale):
    n_dst = int(np.ceil(n_src / scale - 1e-9))
    w = da._area_weights(n_src, scale, n_dst).toarray()
    assert w.shape == (n_dst, n_src)
    np.testing.assert_allclose(w.sum(axis=0), 1.0)  # each pixel is shared out in full
    expected = np.minimum(scale, n_src - np.arange(n_dst) * scale)  # the last one is clipped
    np.testing.assert_allclose(w.sum(axis=1), expected)


def _pan_raster(tmp_path, data, pixel=1.5, name="pan.tif", nodata=float("nan")):
    return _write(
        tmp_path / name, data, transform=from_origin(500000, 4000000, pixel, pixel), nodata=nodata
    )


def test_processing_raster_averages_valid_pixels_on_a_coarser_grid(tmp_path):
    rng = np.random.default_rng(3)
    data = rng.uniform(100, 3000, (1, 9, 7)).astype(np.float32)
    data[0, 0, 0] = np.nan  # invalid: left out of the mean of its 2 x 2 block
    data[0, 2:4, 2:4] = np.nan  # a whole block without valid pixels: nodata
    data[0, 4, 6] = 0.0  # all bands 0: nodata for Delineate-Anything (upstream [0, 0, 0])
    tif = _pan_raster(tmp_path, data)
    config = _config(tmp_path, tif)
    path, bands, info = da.processing_raster(tif, config, 3.0, [1, 1, 1])
    assert path != tif and bands == [1, 1, 1]
    assert info["scale"] == 2.0 and info["gsd_m"] == 3.0 and info["size_px"] == [5, 4]
    assert info["resampling"] == "average" and info["raster"] == path
    with rasterio.open(path) as src:
        assert (src.height, src.width, src.count, src.dtypes[0]) == (5, 4, 1, "float32")
        assert src.crs == "EPSG:32611" and np.isnan(src.nodata)
        assert src.transform == from_origin(500000, 4000000, 3.0, 3.0)  # same origin
        out = src.read(1)
    valid = np.isfinite(data[0]) & (data[0] != 0)
    for r in range(5):
        for c in range(4):
            block = data[0, 2 * r : 2 * r + 2, 2 * c : 2 * c + 2]
            ok = valid[2 * r : 2 * r + 2, 2 * c : 2 * c + 2]
            if ok.any():
                assert out[r, c] == pytest.approx(block[ok].mean(), rel=1e-6)
            else:
                assert np.isnan(out[r, c])
    assert np.isnan(out[1, 1])
    # The last row and column are half outside the raster: the mean of what they cover.
    assert out[4, 0] == pytest.approx(data[0, 8, 0:2].mean(), rel=1e-6)
    stamp = Path(path).stat().st_mtime_ns
    again = da.processing_raster(tif, config, 3.0, [1, 1, 1])
    assert again == (path, bands, info) and Path(path).stat().st_mtime_ns == stamp  # reused


def test_processing_raster_matches_gdal_average_inside_the_raster(tmp_path):
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    rng = np.random.default_rng(4)
    data = rng.uniform(100, 3000, (1, 40, 48)).astype(np.float32)
    data[0, 5:9, 3:6] = np.nan
    tif = _pan_raster(tmp_path, data, pixel=1.0)
    path, _, info = da.processing_raster(tif, _config(tmp_path, tif), 1.6, [1, 1, 1])
    assert info["scale"] == pytest.approx(1.6)
    with rasterio.open(path) as src:
        out, tf = src.read(1), src.transform
    gdal = np.full(out.shape, np.nan, dtype=np.float32)
    reproject(
        data[0],
        gdal,
        src_transform=from_origin(500000, 4000000, 1.0, 1.0),
        src_crs="EPSG:32611",
        dst_transform=tf,
        dst_crs="EPSG:32611",
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    inside = (slice(0, 40 * 10 // 16), slice(0, 48 * 10 // 16))  # pixels inside the raster
    np.testing.assert_allclose(out[inside], gdal[inside], rtol=1e-5)
    assert np.array_equal(np.isnan(out[inside]), np.isnan(gdal[inside]))


def test_processing_raster_keeps_uint8_unstretched_and_maps_bands(tmp_path):
    data = np.zeros((4, 6, 6), dtype=np.uint8)
    data[0], data[1], data[2], data[3] = 10, 101, 200, 77
    data[0, 0, 1] = 13  # block (0, 0) of band 1: mean 10.75, rounded to 11
    data[:3, 4:, 4:] = 0  # all three bands read are 0: nodata (band 4 is not read)
    tif = _write(tmp_path / "rgbn.tif", data, transform=from_origin(500000, 4000000, 1.0, 1.0))
    path, bands, info = da.processing_raster(tif, _config(tmp_path, tif), 2.0, [3, 2, 1])
    assert bands == [3, 2, 1] and info["size_px"] == [3, 3]  # distinct bands 1, 2, 3 in order
    with rasterio.open(path) as src:
        assert src.count == 3 and src.dtypes[0] == "uint8" and src.nodata == 0
        out = src.read()
        lows, highs = da.scene_stretch_bounds(src, bands, (0.1, 99.9))
    assert (lows, highs) == ([0.0] * 3, [255.0] * 3)  # still used unstretched
    assert out[0, 0, 0] == 11 and out[1, 1, 1] == 101 and out[2, 1, 1] == 200
    assert out[:, 2, 2].tolist() == [0, 0, 0]


def test_processing_raster_reads_the_raster_as_it_is_when_not_coarser(tmp_path):
    tif = _pan_raster(tmp_path, np.ones((1, 8, 8), dtype=np.float32))
    config = _config(tmp_path, tif)
    assert da.processing_raster(tif, config, None, [1, 1, 1]) == (tif, [1, 1, 1], None)
    assert da.processing_raster(tif, config, 1.505, [1, 1, 1]) == (tif, [1, 1, 1], None)
    with pytest.raises(ValueError, match="finer than the raster's pixels"):
        da.processing_raster(tif, config, 1.0, [1, 1, 1])
    assert not list(Path(config.get_working_dir()).glob("da_processing_*"))


def test_processing_raster_leaves_no_partial_copy_when_interrupted(tmp_path, monkeypatch):
    tif = _pan_raster(tmp_path, np.full((1, 8, 8), 500.0, dtype=np.float32))
    config = _config(tmp_path, tif)
    real = da._valid_pixels

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted")  # e.g. a full disk while the copy is written

    monkeypatch.setattr(da, "_valid_pixels", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        da.processing_raster(tif, config, 3.0, [1, 1, 1])
    workdir = Path(config.get_working_dir())
    assert not [p.name for p in workdir.iterdir() if "da_processing_" in p.name]  # nor a .tmp
    monkeypatch.setattr(da, "_valid_pixels", real)
    path, _, info = da.processing_raster(tif, config, 3.0, [1, 1, 1])  # the next run writes it
    assert Path(path).is_file() and info["size_px"] == [4, 4]
    assert [p.name for p in workdir.iterdir() if "da_processing_" in p.name] == [Path(path).name]


def test_native_backend_sees_a_field_larger_than_a_tile_whole_at_a_coarser_gsd(
    tmp_path, fake_ultralytics, fake_weights
):
    # A 700 x 700 m field at 1 m is cut by every 512 m tile; at 2 m a 1024 m tile holds it.
    tif = _square_raster(tmp_path, size=1024, square=(100, 800))
    plain = da.DelineateAnythingEngine().delineate(tif, _config(tmp_path, tif))
    assert plain.attrs["engine_meta"]["n_merged_groups"] == 1  # rebuilt from tile-cut pieces
    coarse = da.DelineateAnythingEngine().delineate(
        tif, _config(tmp_path, tif, processing_gsd_m=2.0)
    )
    model = fake_ultralytics.instances[-1]
    images = [img for call in model.calls for img in call[0]]
    assert len(images) == 9  # 512 px at 2 m: origins -256, 0, 256 on both axes (25 at 1 m)
    assert all(img.shape == (512, 512, 3) for img in images)
    meta = coarse.attrs["engine_meta"]
    assert meta["processing_gsd_m"] == 2.0 and meta["gsd_m"] == 1.0
    assert meta["processing"]["scale"] == 2.0 and meta["processing"]["size_px"] == [512, 512]
    assert meta["processing"]["gsd_m"] == 2.0 and meta["super_resolution"] == 1
    assert meta["tile_size_native_px"] == 512 and meta["band_indices_bgr"] == [3, 2, 1]
    assert meta["n_merged_groups"] == 0  # one tile saw the field whole
    assert meta["gsd_outside_training_range"] is False
    assert len(coarse) == 1 and coarse.crs == plain.crs
    assert coarse.geometry.iloc[0].equals(plain.geometry.iloc[0])  # 700 x 700 m, same place
    json.dumps(meta)
    assert plain.attrs["engine_meta"]["processing_gsd_m"] is None
    assert plain.attrs["engine_meta"]["processing"] is None


def test_native_backend_resamples_to_bring_fine_imagery_into_the_training_range(
    tmp_path, fake_ultralytics, fake_weights, caplog
):
    tif = _square_raster(tmp_path, size=512, square=(100, 300), pixel=0.1)  # 0.1 m pixels
    with caplog.at_level("WARNING"):
        gdf = da.DelineateAnythingEngine().delineate(
            tif, _config(tmp_path, tif, processing_gsd_m=0.3)
        )
    meta = gdf.attrs["engine_meta"]
    assert meta["gsd_m"] == 0.1 and meta["processing"]["gsd_m"] == pytest.approx(0.3)
    assert meta["gsd_outside_training_range"] is False  # the model sees 0.3 m pixels
    assert "trained on 0.25-10 m imagery" not in caplog.text


def test_native_backend_stretch_percentiles(tmp_path, fake_ultralytics, fake_weights):
    rng = np.random.default_rng(5)
    data = rng.uniform(100, 3000, (3, 256, 256)).astype(np.float32)
    tif = _write(tmp_path / "f.tif", data, transform=from_origin(500000, 4000000, 1.0, 1.0))
    gdf = da.DelineateAnythingEngine().delineate(
        tif, _config(tmp_path, tif, stretch_percentiles=[0.1, 99.9])
    )
    meta = gdf.attrs["engine_meta"]
    assert meta["stretch_percentiles"] == [0.1, 99.9]
    for i, band in enumerate((2, 1, 0)):  # BGR
        lo, hi = np.percentile(data[band], (0.1, 99.9))
        assert meta["stretch_lows_bgr"][i] == pytest.approx(lo, rel=1e-6)
        assert meta["stretch_highs_bgr"][i] == pytest.approx(hi, rel=1e-6)
    image = fake_ultralytics.instances[0].calls[0][0][0]  # the tile at (-256, -256)
    with rasterio.open(tif) as src:
        tile, valid = da._read_tile(src, [3, 2, 1], -256, -256, 512)
    lows, highs = meta["stretch_lows_bgr"], meta["stretch_highs_bgr"]
    expected = da.stretch_with_bounds(tile, lows, highs, valid)
    np.testing.assert_array_equal(image, np.transpose(expected, (1, 2, 0)))
    default = da.DelineateAnythingEngine().delineate(tif, _config(tmp_path, tif))
    assert default.attrs["engine_meta"]["stretch_percentiles"] == [1.0, 99.0]
    assert default.attrs["engine_meta"]["stretch_lows_bgr"] != meta["stretch_lows_bgr"]


def test_native_backend_warns_that_uint8_rasters_are_not_stretched(
    tmp_path, fake_ultralytics, fake_weights, caplog
):
    tif = _square_raster(tmp_path)
    with caplog.at_level("WARNING"):
        gdf = da.DelineateAnythingEngine().delineate(
            tif, _config(tmp_path, tif, stretch_percentiles=[0.1, 99.9])
        )
    assert "has no effect: uint8 rasters" in caplog.text
    assert gdf.attrs["engine_meta"]["stretch_lows_bgr"] == [0.0] * 3


def test_native_cache_key_gets_the_options_only_when_set(
    tmp_path, fake_ultralytics, fake_weights, monkeypatch
):
    """Runs without the options keep the cache key (and cached results) of earlier runs."""
    import agribound._cache as cache

    keys: list[tuple] = []
    real = cache.cache_path

    def spy(config, stem, suffix, *parts, **kwargs):
        if stem == "da_native":
            keys.append(parts)
        return real(config, stem, suffix, *parts, **kwargs)

    monkeypatch.setattr(cache, "cache_path", spy)
    tif = _square_raster(tmp_path)
    for params in (
        {},
        {"stretch_percentiles": [1, 99], "processing_gsd_m": None},
        {"processing_gsd_m": 1.0},
        {"stretch_percentiles": (0.5, 99.5)},
        {"processing_gsd_m": 2, "stretch_percentiles": [0.5, 99.5]},
    ):
        da.DelineateAnythingEngine().delineate(tif, _config(tmp_path, tif, **params))
    default = keys[0]
    assert not any(str(p).startswith(("processing_gsd_m=", "stretch_")) for p in default)
    assert keys[1] == default  # the defaults, given explicitly: same key, cached result
    assert keys[2] == (*default, "processing_gsd_m=1.0")
    assert keys[3] == (*default, "stretch_percentiles=[0.5, 99.5]")
    # A resampled copy is read: its version is part of the key too.
    assert keys[4] == (
        *default,
        "processing_gsd_m=2.0",
        "processing_raster=1",
        "stretch_percentiles=[0.5, 99.5]",
    )
    assert len(fake_ultralytics.instances) == 4  # the second run used the first one's cache
    # A new version of the copy is not served the results made from the old one.
    monkeypatch.setattr(da, "_PROCESSING_RASTER_VERSION", "2")
    da.DelineateAnythingEngine().delineate(tif, _config(tmp_path, tif, processing_gsd_m=2))
    assert keys[5] == (*default, "processing_gsd_m=2.0", "processing_raster=2")
    assert len(fake_ultralytics.instances) == 5


def test_large_field_options_leave_config_hash_unchanged_when_not_set(
    tmp_path, fake_ultralytics, fake_weights
):
    from agribound.provenance import config_hash

    tif = _square_raster(tmp_path)
    config = _config(tmp_path, tif)
    before = config_hash(config)
    da.DelineateAnythingEngine().delineate(tif, config)
    assert config.engine_params == {} and config_hash(config) == before
    for params in ({"processing_gsd_m": 2.0}, {"stretch_percentiles": [0.1, 99.9]}):
        other = _config(tmp_path, tif, **params)
        da.DelineateAnythingEngine().delineate(tif, other)
        assert other.engine_params == params  # nothing is added or normalised
        assert config_hash(other) != before  # a different configuration: never reused


@pytest.mark.filterwarnings("ignore:SPOT 6/7 imagery:UserWarning")
def test_config_hash_without_the_large_field_options_is_that_of_agribound_1_0_1():
    """Example 23's Twifo Praso SPOT-Pan run: the config_hash its 1.0.1 provenance recorded."""
    from agribound.provenance import config_hash

    config = AgriboundConfig(
        source="spot-pan",
        year=2020,
        study_area="outputs/tree_crops/twifo/study_area.geojson",
        engine="delineate-anything",
        cloud_cover_max=15,
        lulc_filter=False,
        gee_project="p",  # not hashed
    )
    assert config_hash(config) == "ac15e961840ed7887150e5863a515c105d06b86c"
    hashes = {
        config_hash(config.merged(engine_params=params))
        for params in (
            {"processing_gsd_m": 3},
            {"stretch_percentiles": [0.1, 99.9]},
            {"processing_gsd_m": 3, "stretch_percentiles": [0.1, 99.9]},
        )
    }
    assert len(hashes) == 3 and config_hash(config) not in hashes


def _checkpoint_with_training_meta(tmp_path, **recorded):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"finetuned")
    from agribound.engines.finetune._data import write_training_meta

    write_training_meta(ckpt, {"engine": "delineate-anything", **recorded})
    return str(ckpt)


_UINT8 = {"stretch_lows": [0.0] * 3, "stretch_highs": [255.0] * 3}  # chips of a uint8 raster


@pytest.mark.parametrize(
    ("pixel", "recorded", "params", "mismatch"),
    [
        (1.0, {}, {}, False),  # nothing recorded: nothing compared
        # agribound 1.0.1 records (no raster_dtype): uint8 chips, read from their 0/255 bounds
        (1.0, {"gsd_m": 1.0, "super_resolution": 1, "chips_stretch": _UINT8}, {}, False),
        # stretch_percentiles has no effect on a uint8 raster, in training or here
        (
            1.0,
            {"gsd_m": 1.0, "raster_dtype": "uint8", "stretch_percentiles": [1.0, 99.0]},
            {"stretch_percentiles": [0.1, 99.9]},
            False,
        ),
        # processing_gsd_m equal to the raster's pixel size: no copy, the same input
        (1.0, {"gsd_m": 1.0, "processing_gsd_m": 1.0, "raster_dtype": "uint8"}, {}, False),
        (1.0, {"gsd_m": 1.005, "raster_dtype": "uint8"}, {}, False),  # within 1 %
        # a 1.5 m checkpoint on a 3 m raster, without any option
        (3.0, {"gsd_m": 1.5, "super_resolution": 1, "raster_dtype": "uint8"}, {}, True),
        (1.0, {"gsd_m": 1.0, "super_resolution": 2, "raster_dtype": "uint8"}, {}, True),
        # chips of a stretched raster against this uint8 raster (used without a stretch)
        (1.0, {"gsd_m": 1.0, "raster_dtype": "uint16"}, {}, True),
    ],
)
def test_native_backend_warns_when_the_checkpoint_saw_other_input(
    tmp_path, fake_ultralytics, fake_weights, caplog, pixel, recorded, params, mismatch
):
    """The input the model sees is compared (pixel size, factor, stretch), not the options."""
    tif = _square_raster(tmp_path, pixel=pixel)  # uint8
    ckpt = _checkpoint_with_training_meta(tmp_path, **recorded)
    with caplog.at_level("WARNING"):
        gdf = da.DelineateAnythingEngine().delineate(
            tif, _config(tmp_path, tif, checkpoint_path=ckpt, **params)
        )
    meta = gdf.attrs["engine_meta"]
    assert ("model sees a different input than in training" in caplog.text) is mismatch
    assert ("checkpoint_input_mismatch" in meta) is mismatch
    if mismatch:
        run = {"gsd_m": pixel, "super_resolution": 1, "stretch_percentiles": None}
        assert meta["checkpoint_input_mismatch"]["run"] == run
        assert meta["checkpoint_input_mismatch"]["trained"] == da._trained_input(
            {"engine": "delineate-anything", **recorded}
        )


@pytest.mark.parametrize(("tolerance", "warns"), [(None, True), (0.0101, False), (0.02, False)])
def test_pixel_size_just_outside_the_tolerance_warns(
    tmp_path, caplog, monkeypatch, tolerance, warns
):
    """Chips of 1.0102 m against a 1 m raster are 1.02 % apart, just outside the 1 % tolerance.

    The case pins PROCESSING_GSD_TOLERANCE: with a looser tolerance (1.01 % or more) it no
    longer warns, so loosening the tolerance fails the first case.
    """
    assert da.PROCESSING_GSD_TOLERANCE == 0.01
    if tolerance is not None:
        monkeypatch.setattr(da, "PROCESSING_GSD_TOLERANCE", tolerance)
    ckpt = _checkpoint_with_training_meta(tmp_path, gsd_m=1.0102, raster_dtype="uint8")
    meta = {}
    run = {"gsd_m": 1.0, "super_resolution": 1, "stretch_percentiles": None}
    with caplog.at_level("WARNING"):
        da._check_checkpoint_input({"weights": "checkpoint", "checkpoint_path": ckpt}, meta, run)
    assert ("model sees a different input than in training" in caplog.text) is warns
    assert ("checkpoint_input_mismatch" in meta) is warns
    if warns:
        assert meta["checkpoint_input_mismatch"]["trained"]["gsd_m"] == 1.0102


def test_trained_input_of_a_stretched_raster():
    record = {
        "gsd_m": 1.49999,
        "super_resolution": 1,
        "chips_stretch": {"stretch_lows": [443.0] * 3, "stretch_highs": [1167.0] * 3},
    }
    # agribound 1.0.1: no percentiles recorded, i.e. 1-99
    assert da._trained_input(record) == {
        "gsd_m": 1.5,
        "super_resolution": 1,
        "stretch_percentiles": [1.0, 99.0],
    }
    record.update(raster_dtype="float32", stretch_percentiles=[0.1, 99.9])
    assert da._trained_input(record)["stretch_percentiles"] == [0.1, 99.9]


# ---------------------------------------------------------------------------
# Weights download and prefetch
# ---------------------------------------------------------------------------


def _fake_hf(monkeypatch, path):
    calls = []
    module = types.ModuleType("huggingface_hub")

    def hf_hub_download(repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        return str(path)

    module.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


def test_download_verifies_pinned_revision_and_sha(monkeypatch, tmp_path):
    blob = tmp_path / "DelineateAnythingv2.pt"
    blob.write_bytes(b"not the real weights")
    calls = _fake_hf(monkeypatch, blob)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        da.download_da_weights("large_v2")
    assert calls == [
        ("MykolaL/DelineateAnything", "DelineateAnythingv2.pt", da.DA_MODELS["large_v2"].revision)
    ]
    spec = da.DA_MODELS["large_v2"]
    patched = da.DAModel(
        **{**spec.__dict__, "sha256": hashlib.sha256(blob.read_bytes()).hexdigest()}
    )
    monkeypatch.setitem(da.DA_MODELS, "large_v2", patched)
    assert da.download_da_weights("large_v2") == str(blob)


def test_prefetch_downloads_selected_model(monkeypatch, tmp_path, fake_weights):
    tif = _square_raster(tmp_path)
    paths = da.DelineateAnythingEngine.prefetch(_config(tmp_path, tif, da_model="large"))
    assert fake_weights == ["large"] and len(paths) == 1


# ---------------------------------------------------------------------------
# Reference backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["large_v2", "large", "small"])
def test_reference_config_aligns_model_keys(key):
    opts = da.DAOptions.from_engine_params({"backend": "reference", "da_model": key})
    cfg = da.build_reference_config(opts, [3, 2, 1], 2500.0, "/in", "/tmp", "/out.gpkg")
    assert cfg["model"] == [key]
    model_args = cfg["passes"][0]["model_args"][0]
    assert model_args["name"] == key
    assert model_args["minimal_confidence"] == da.DA_MODELS[key].default_conf
    assert cfg["execution_planner"]["pixel_offset"] == [0, 0]
    assert cfg["mask_info"]["range"] == 40
    assert cfg["filtering_args"]["automatic_area_scale"] is False
    assert cfg["filtering_args"]["minimum_area_m2"] == 2500.0
    assert cfg["data_loader"]["bands"] == [3, 2, 1]
    assert cfg["passes"][0]["batch_size"] == 4 and cfg["passes"][0]["tile_step"] == 0.5
    assert cfg["simplification_args"]["simplify"] is False
    assert cfg["execution_args"]["src_folder"] == "/in"
    # the template itself is not modified
    assert da._REFERENCE_CONFIG["model"] == ["large_v2"]


def test_reference_config_hole_threshold_is_independent_of_min_area():
    opts = da.DAOptions.from_engine_params({"backend": "reference"})
    cfg = da.build_reference_config(opts, [3, 2, 1], 100.0, "/in", "/tmp", "/out.gpkg")
    assert cfg["filtering_args"]["minimum_area_m2"] == 100.0
    assert cfg["filtering_args"]["minimum_hole_area_m2"] == 2500.0  # upstream default
    opts = da.DAOptions.from_engine_params({"backend": "reference", "min_hole_area_m2": 400})
    cfg = da.build_reference_config(opts, [3, 2, 1], 100.0, "/in", "/tmp", "/out.gpkg")
    assert cfg["filtering_args"]["minimum_hole_area_m2"] == 400.0


def test_reference_repo_checks(tmp_path):
    with pytest.raises(FileNotFoundError, match="methods/main/inference.py"):
        da.check_reference_repo(tmp_path)
    inference = tmp_path / "methods" / "main" / "inference.py"
    inference.parent.mkdir(parents=True)
    inference.write_text("result.masks.data = -F.max_pool2d(-result.masks.data, 3)\n")
    with pytest.raises(RuntimeError, match="34eddf7"):
        da.check_reference_repo(tmp_path)
    inference.write_text("x = -F.max_pool2d(-result.masks.data.float(), 3)\n")
    assert da.check_reference_repo(tmp_path) == tmp_path.resolve()


def test_reference_backend_requires_repo(tmp_path, monkeypatch):
    monkeypatch.delenv("AGRIBOUND_DA_REPO", raising=False)
    tif = _square_raster(tmp_path)
    with pytest.raises(RuntimeError, match="AGRIBOUND_DA_REPO"):
        da.DelineateAnythingEngine().delineate(tif, _config(tmp_path, tif, backend="reference"))


# ---------------------------------------------------------------------------
# FTW backend
# ---------------------------------------------------------------------------


def _fake_ftw(monkeypatch, registry, calls):
    pkg = types.ModuleType("ftw_tools")
    inference_pkg = types.ModuleType("ftw_tools.inference")
    inference = types.ModuleType("ftw_tools.inference.inference")
    model_registry = types.ModuleType("ftw_tools.inference.model_registry")
    model_registry.MODEL_REGISTRY = registry
    models = types.ModuleType("ftw_tools.inference.models")
    models.DelineateAnything = types.SimpleNamespace(checkpoints={"DelineateAnything": "https://x"})

    def run_instance_segmentation(input, model, out, **kwargs):  # noqa: A002
        calls.append({"input": input, "model": model, "out": out, **kwargs})
        with rasterio.open(input) as src:
            crs = src.crs
        gpd.GeoDataFrame(geometry=[box(500000, 3999900, 500100, 4000000)], crs=crs).to_file(out)

    inference.run_instance_segmentation = run_instance_segmentation
    for name, module in {
        "ftw_tools": pkg,
        "ftw_tools.inference": inference_pkg,
        "ftw_tools.inference.inference": inference,
        "ftw_tools.inference.model_registry": model_registry,
        "ftw_tools.inference.models": models,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_ftw_backend_rejects_v2_when_registry_lacks_it(tmp_path, monkeypatch):
    _fake_ftw(monkeypatch, {"DelineateAnything": object()}, [])
    tif = _square_raster(tmp_path)
    config = _config(tmp_path, tif, backend="ftw", value_scale="reflectance_x10000")
    with pytest.raises(RuntimeError, match="fa86d4a"):
        da.DelineateAnythingEngine().delineate(tif, config)


def test_ftw_backend_rejects_non_reflectance(tmp_path, monkeypatch):
    _fake_ftw(monkeypatch, {"DelineateAnything": object()}, [])
    tif = _square_raster(tmp_path)
    config = _config(tmp_path, tif, backend="ftw", da_model="large")
    with pytest.raises(ValueError, match="reflectance x 10000"):
        da.DelineateAnythingEngine().delineate(tif, config)


def test_ftw_backend_arguments_and_input(tmp_path, monkeypatch):
    calls: list = []
    _fake_ftw(monkeypatch, {"DelineateAnything": object()}, calls)
    data = np.full((3, 64, 64), 1500.0, dtype=np.float32)
    data[:, 0, 0] = np.nan
    tif = _write(tmp_path / "s2.tif", data, nodata=float("nan"))
    config = _config(
        tmp_path,
        tif,
        backend="ftw",
        da_model="large",
        value_scale="reflectance_x10000",
        patch_size=32,  # must be smaller than the 64 px raster
    )
    gdf = da.DelineateAnythingEngine().delineate(tif, config)
    call = calls[0]
    assert call["model"] == "DelineateAnything"
    assert call["patch_size"] == 32 and call["resize_factor"] == 2
    assert call["conf_threshold"] == 0.05 and call["max_detections"] == 300
    assert call["iou_threshold"] == 0.3 and call["min_size"] == 100.0
    assert "nan_fill_value" not in call  # fake signature has no such parameter
    with rasterio.open(call["input"]) as src:
        written = src.read()
    assert written.dtype == np.float32 and np.isfinite(written).all()
    assert written[0, 0, 0] == 0.0 and written[0, 1, 1] == 1500.0
    assert len(gdf) == 1 and gdf.attrs["engine_meta"]["backend"] == "ftw"


def test_ftw_backend_writes_atomically_and_keys_cache_by_version(tmp_path, monkeypatch):
    calls: list = []
    _fake_ftw(monkeypatch, {"DelineateAnything": object()}, calls)
    data = np.full((3, 64, 64), 1500.0, dtype=np.float32)
    data[:, :2, :2] = -9999.0  # declared nodata
    tif = _write(tmp_path / "s2.tif", data, nodata=-9999.0)
    params = {
        "backend": "ftw",
        "da_model": "large",
        "value_scale": "reflectance_x10000",
        "patch_size": 32,
    }
    config = _config(tmp_path, tif, **params)
    versions = iter(["2.0.0b5", "2.0.0b5", "2.1.0"])
    real_version = da._package_version
    monkeypatch.setattr(
        da, "_package_version", lambda d: next(versions) if d == "ftw-tools" else real_version(d)
    )
    first = da.DelineateAnythingEngine().delineate(tif, config)
    assert Path(calls[0]["out"]).name.endswith(".partial.gpkg")
    with rasterio.open(calls[0]["input"]) as src:
        assert src.read(1)[0, 0] == 0.0  # nodata -> 0, not -9999 / 3000
    assert "nodata -> 0" in first.attrs["engine_meta"]["input_units"]
    second = da.DelineateAnythingEngine().delineate(tif, config)
    assert len(calls) == 1 and second.attrs["engine_meta"]["cached_result"]
    da.DelineateAnythingEngine().delineate(tif, config)
    assert len(calls) == 2  # a new ftw-tools version is a new cache key


def test_ftw_backend_rejects_patch_spanning_the_raster(tmp_path, monkeypatch):
    calls: list = []
    _fake_ftw(monkeypatch, {"DelineateAnything": object()}, calls)
    data = np.full((3, 64, 64), 1500.0, dtype=np.float32)
    tif = _write(tmp_path / "s2.tif", data)
    for patch in (None, 64):  # default 256, and a patch equal to the raster side
        params = {"backend": "ftw", "da_model": "large", "value_scale": "reflectance_x10000"}
        if patch is not None:
            params["patch_size"] = patch
        config = _config(tmp_path, tif, **params)
        with pytest.raises(ValueError, match="smaller side"):
            da.DelineateAnythingEngine().delineate(tif, config)
    assert calls == []  # ftw-tools is never called


# ---------------------------------------------------------------------------
# YOLO fine-tuning labels and dataset
# ---------------------------------------------------------------------------


def test_chip_labels_one_instance_per_reference_polygon():
    from agribound.engines.finetune._yolo import chip_labels

    tf = from_origin(0.0, 100.0, 1.0, 1.0)  # 100 x 100 px chip, 1 m pixels
    touching_a = box(10, 50, 30, 80)
    touching_b = box(30, 50, 50, 80)  # shares an edge with touching_a
    isolated = box(70, 10, 90, 30)
    lines = chip_labels([touching_a, touching_b, isolated], tf, 100, 100)
    assert len(lines) == 3
    for line, geom in zip(lines, (touching_a, touching_b, isolated), strict=True):
        values = np.array(line.split()[1:], dtype=float).reshape(-1, 2)
        assert line.startswith("0 ")
        assert values.min() >= 0 and values.max() <= 1
        poly = Polygon(values * 100)
        assert poly.area == pytest.approx(geom.area)  # 1 m pixels: px area == m2


def test_chip_labels_bridge_holes_so_filled_mask_excludes_them():
    import cv2

    from agribound.engines.finetune._yolo import chip_labels

    tf = from_origin(0.0, 64.0, 1.0, 1.0)
    field = Polygon(
        [(8, 8), (56, 8), (56, 56), (8, 56)], holes=[[(24, 24), (40, 24), (40, 40), (24, 40)]]
    )
    lines = chip_labels([field], tf, 64, 64)
    assert len(lines) == 1
    ring = np.array(lines[0].split()[1:], dtype=float).reshape(-1, 2) * 64
    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(ring).astype(np.int32)], 1)  # as ultralytics polygon2mask
    assert mask[32, 32] == 0  # hole centre stays empty
    assert mask[12, 12] == 1
    assert abs(int(mask.sum()) - (48 * 48 - 16 * 16)) < 0.1 * 48 * 48


def test_chip_labels_clip_to_chip_and_valid_area():
    from agribound.engines.finetune._yolo import chip_labels

    tf = from_origin(0.0, 100.0, 1.0, 1.0)
    crossing = box(80, 40, 140, 60)  # half outside the chip
    u_shape = Polygon(
        [(90, 10), (130, 10), (130, 30), (90, 30), (90, 25), (125, 25), (125, 15), (90, 15)]
    )  # both arms reach x < 100
    lines = chip_labels([crossing], tf, 100, 100)
    xy = np.array(lines[0].split()[1:], dtype=float).reshape(-1, 2)
    assert xy[:, 0].max() == pytest.approx(1.0)
    assert Polygon(xy * 100).area == pytest.approx(20 * 20)
    assert len(chip_labels([u_shape], tf, 100, 100)) == 2  # two parts inside the chip
    valid = box(0, 0, 90, 100)  # right 10 px invalid
    clipped = chip_labels([crossing], tf, 100, 100, valid_area_px=valid)
    xy = np.array(clipped[0].split()[1:], dtype=float).reshape(-1, 2)
    assert Polygon(xy * 100).area == pytest.approx(10 * 20)
    assert chip_labels([box(10, 10, 11, 11)], tf, 100, 100, min_area_px=4.0) == []


def test_frame_cut_depth_measures_from_the_sides_that_cut_the_part():
    from agribound.engines.finetune._yolo import frame_cut_depth

    frame = box(0, 0, 100, 100)  # chip pixels, y down

    def depth(polygon):
        return frame_cut_depth(polygon.intersection(frame), polygon, 100, 100)

    assert depth(box(-50, 20, 10, 80)) == pytest.approx(10)  # left side
    assert depth(box(70, 20, 150, 80)) == pytest.approx(30)  # right side
    assert depth(box(20, -40, 80, 12)) == pytest.approx(12)  # top side (row 0)
    assert depth(box(20, 95, 80, 140)) == pytest.approx(5)  # bottom side
    assert depth(box(-10, -10, 60, 8)) == pytest.approx(8)  # corner: the shallower side
    assert depth(box(-10, 40, 110, 50)) == pytest.approx(100)  # crosses the chip
    assert depth(box(10, 10, 30, 30)) is None  # inside the chip
    assert depth(box(0, 10, 8, 90)) is None  # ends on the frame but is not cut by it
    # Multi-part: only the part's own component counts (the other one lies left of the chip).
    from shapely.geometry import MultiPolygon

    two = MultiPolygon([box(0, 10, 8, 90), box(-40, 10, -20, 90)])
    assert frame_cut_depth(box(0, 10, 8, 90), two, 100, 100) is None
    two = MultiPolygon([box(-50, 20, 10, 80), box(40, 40, 60, 60)])
    assert frame_cut_depth(box(0, 20, 10, 80), two, 100, 100) == pytest.approx(10)


def test_chip_labels_drop_slivers_cut_by_the_frame():
    from agribound.engines.finetune._yolo import chip_labels

    tf = from_origin(0.0, 100.0, 1.0, 1.0)  # 100 x 100 px chip, 1 m pixels
    sliver = box(-50, 20, 10, 80)  # 10 px into the chip from the left side
    deep = box(-50, 20, 30, 80)  # 30 px
    complete_thin = box(0, 20, 8, 80)  # a whole 8 px field along the frame
    narrow_cut = box(-20, 45, 60, 50)  # 5 px wide, cut by the left side, 60 px deep
    inside = box(40, 40, 60, 60)
    fields = [sliver, deep, complete_thin, narrow_cut, inside]
    assert len(chip_labels(fields, tf, 100, 100)) == 5  # default: no sliver test
    counts: dict = {}
    lines = chip_labels(fields, tf, 100, 100, sliver_depth_px=16, counts=counts)
    assert len(lines) == 4 and counts == {"n_slivers": 1}
    areas = sorted(
        Polygon(np.array(line.split()[1:], dtype=float).reshape(-1, 2) * 100).area for line in lines
    )
    assert areas == pytest.approx(sorted([30 * 60, 8 * 60, 60 * 5, 20 * 20]))
    assert len(chip_labels(fields, tf, 100, 100, sliver_depth_px=30)) == 3  # "at most"
    # A part that ends at invalid pixels, not at the frame, is kept however thin.
    valid = box(20, 0, 100, 100)  # the left 20 px of the chip are nodata
    assert len(chip_labels([deep], tf, 100, 100, valid_area_px=valid, sliver_depth_px=16)) == 1


def test_prepare_yolo_dataset_drops_slivers_in_model_input_pixels(tmp_path):
    from agribound.engines.finetune._yolo import prepare_yolo_dataset

    # 64 px chips of 10 m; train chip 0 covers x 500000-500640, y 3999360-4000000.
    train_dir, _ = _training_dir(tmp_path)
    fields = [
        box(499000, 3999600, 500060, 3999700),  # 6 px into train chip 0 from its left side
        box(499000, 3999450, 500100, 3999550),  # 10 px
        box(500200, 3999450, 500300, 3999550),  # inside
    ]
    ref_path = tmp_path / "frame_fields.gpkg"
    gpd.GeoDataFrame(geometry=fields, crs="EPSG:32611").to_file(ref_path)
    config = AgriboundConfig(
        source="local",
        engine="delineate-anything",
        local_tif_path=str(tmp_path / "x.tif"),
        output_path=str(tmp_path / "o.gpkg"),
        reference_boundaries=str(ref_path),
        lulc_filter=False,
    )
    out = tmp_path / "yolo"
    expected = {(1, 16.0): (1, 2), (2, 16.0): (2, 1), (1, 0.0): (3, 0)}  # (instances, slivers)
    for (sr, depth), (n_instances, n_slivers) in expected.items():
        info = prepare_yolo_dataset(train_dir, config, out, sr, sliver_depth_px=depth)
        assert (info["n_train_instances"], info["n_train_slivers"]) == (n_instances, n_slivers)
        assert info["n_val_slivers"] == 0
    # The default is DEFAULT_SLIVER_DEPTH_PX model-input pixels (16).
    info = prepare_yolo_dataset(train_dir, config, out, 1)
    assert (info["n_train_instances"], info["n_train_slivers"]) == (1, 2)


def _training_dir(tmp_path, chip=64, pixel=10.0, n_train=2, n_val=1):
    root = tmp_path / "train_dir"
    fields = []
    for split, n in (("images", n_train), ("val_images", n_val)):
        (root / split).mkdir(parents=True)
        for i in range(n):
            x0 = 500000 + (i + (10 if split == "val_images" else 0)) * chip * pixel
            tf = from_origin(x0, 4000000, pixel, pixel)
            data = np.full((3, chip, chip), 90, dtype=np.uint8)
            name = f"chip_{len(fields):05d}.tif"
            _write(root / split / name, data, transform=tf)
            fields.append(box(x0 + 100, 4000000 - 300, x0 + 300, 4000000 - 100))
    ref = gpd.GeoDataFrame({"fid": range(len(fields))}, geometry=fields, crs="EPSG:32611")
    ref_path = tmp_path / "reference.gpkg"
    ref.to_file(ref_path)
    return root, ref_path


def test_chip_valid_pixels_reads_the_data_mask(tmp_path):
    from agribound.engines.finetune._data import IGNORE_INDEX
    from agribound.engines.finetune._yolo import _mask_path, chip_valid_pixels

    assert _mask_path(Path("d/images/chip_00001.tif")) == Path("d/masks/chip_00001.tif")
    assert _mask_path(Path("d/val_images/chip_00002.tif")) == Path("d/val_masks/chip_00002.tif")
    image = np.full((3, 8, 8), 50, dtype=np.uint8)
    image[:, :2, :] = 0  # dark but valid pixels (below the 1st percentile in every band)
    mask = np.ones((1, 8, 8), dtype=np.uint8)
    mask[0, :, 6:] = IGNORE_INDEX  # last two columns are nodata
    mask_path = Path(_write(tmp_path / "mask.tif", mask))
    valid = chip_valid_pixels(image, mask_path)
    assert valid[:2, :6].all()  # the mask, not the zero heuristic, decides
    assert not valid[:, 6:].any()
    # Without a mask the zero heuristic is used.
    assert not chip_valid_pixels(image, tmp_path / "missing.tif")[:2].any()


def test_prepare_yolo_dataset_clips_labels_to_mask_valid_area(tmp_path):
    from agribound.engines.finetune._data import IGNORE_INDEX
    from agribound.engines.finetune._yolo import prepare_yolo_dataset

    train_dir, ref_path = _training_dir(tmp_path)
    # Train chip 0: the field (x 10-30 px, y 10-30 px) has dark-but-valid pixels on its
    # left half and the right half of the chip (x >= 20) is nodata in the mask.
    image_path = train_dir / "images" / "chip_00000.tif"
    with rasterio.open(image_path) as src:
        data, tf = src.read(), src.transform
    data[:, :, :20] = 0
    _write(image_path, data, transform=tf)
    for split in ("masks", "val_masks"):
        (train_dir / split).mkdir()
    mask = np.ones((1, 64, 64), dtype=np.uint8)
    mask[0, :, 20:] = IGNORE_INDEX
    _write(train_dir / "masks" / "chip_00000.tif", mask, transform=tf)
    config = AgriboundConfig(
        source="local",
        engine="delineate-anything",
        local_tif_path=str(tmp_path / "x.tif"),
        output_path=str(tmp_path / "o.gpkg"),
        reference_boundaries=str(ref_path),
        lulc_filter=False,
    )
    prepare_yolo_dataset(train_dir, config, tmp_path / "yolo", super_resolution=1)
    line = (tmp_path / "yolo" / "labels" / "train" / "chip_00000.txt").read_text()
    xy = np.array(line.split()[1:], dtype=float).reshape(-1, 2) * 64
    # Field kept on the dark-but-valid side (x 10-20), cut at the nodata edge (x = 20).
    assert Polygon(xy).area == pytest.approx(10 * 20)
    assert xy[:, 0].min() == pytest.approx(10) and xy[:, 0].max() == pytest.approx(20)


def test_prepare_yolo_dataset_uses_split_and_upsampling(tmp_path):
    from agribound.engines.finetune._yolo import prepare_yolo_dataset

    train_dir, ref_path = _training_dir(tmp_path)
    config = AgriboundConfig(
        source="local",
        engine="delineate-anything",
        local_tif_path=str(tmp_path / "x.tif"),
        output_path=str(tmp_path / "o.gpkg"),
        reference_boundaries=str(ref_path),
        lulc_filter=False,
    )
    info = prepare_yolo_dataset(train_dir, config, tmp_path / "yolo", super_resolution=2)
    import yaml

    data = yaml.safe_load(Path(info["data_yaml"]).read_text())
    assert data["train"] == "images/train" and data["val"] == "images/val"
    assert data["names"] == {0: "field"}
    assert (info["chip_size"], info["imgsz"]) == (64, 128)
    assert (info["n_train_images"], info["n_val_images"]) == (2, 1)
    assert (info["n_train_instances"], info["n_val_instances"]) == (2, 1)
    from PIL import Image

    png = sorted((tmp_path / "yolo" / "images" / "val").glob("*.png"))
    assert len(png) == 1 and Image.open(png[0]).size == (128, 128)


def test_finetune_yolo_training_arguments(tmp_path, fake_ultralytics, fake_weights):
    from agribound.engines.finetune._yolo import _finetune_yolo

    train_dir, ref_path = _training_dir(tmp_path, chip=256)
    config = AgriboundConfig(
        source="local",
        engine="delineate-anything",
        local_tif_path=str(tmp_path / "x.tif"),
        output_path=str(tmp_path / "o.gpkg"),
        reference_boundaries=str(ref_path),
        lulc_filter=False,
        device="cpu",
        seed=7,
        fine_tune_epochs=3,
        cache_dir=str(tmp_path / "run"),
    )
    best = _finetune_yolo(train_dir, config, "DA-large_v2")
    model = fake_ultralytics.instances[0]
    kwargs = model.train_kwargs
    assert fake_weights == ["large_v2"]
    assert kwargs["seed"] == 7 and kwargs["deterministic"] is True
    assert kwargs["mosaic"] == 0.0 and kwargs["optimizer"] == "AdamW"
    assert kwargs["fliplr"] == 0.5 and kwargs["flipud"] == 0.5
    assert kwargs["imgsz"] == 512 and kwargs["epochs"] == 3 and kwargs["lr0"] == 0.002
    # Ultralytics' optimizer="auto" sets warmup_bias_lr=0 (its 0.1 is for a named optimizer).
    assert kwargs["warmup_bias_lr"] == 0.0
    # One optimizer step per batch: no gradient accumulation (Ultralytics' default nbs is 64).
    assert kwargs["batch"] == 16 and kwargs["nbs"] == 16
    assert Path(best).is_file() and Path(best).name == "best.pt"
    assert str(Path(best)).startswith(str(tmp_path / "run"))
    meta = json.loads(Path(f"{best}.agribound.json").read_text())
    assert meta["base_model_key"] == "large_v2" and meta["super_resolution"] == 2
    assert meta["imgsz_matches_model_input"] is True
    assert meta["recipe_version"] == 3 and meta["train_kwargs"]["warmup_bias_lr"] == 0.0
    assert meta["train_kwargs"]["nbs"] == 16 and meta["sliver_depth_px"] == 16.0
    # No chips_meta.json here: the options come from engine_params (none set).
    assert meta["processing_gsd_m"] is None and meta["stretch_percentiles"] == [1.0, 99.0]


def test_finetune_yolo_records_the_input_options_of_its_chips(
    tmp_path, fake_ultralytics, fake_weights
):
    from agribound.engines.finetune._yolo import _finetune_yolo

    train_dir, ref_path = _training_dir(tmp_path, chip=256)
    chips = {"processing_gsd_m": 3.0, "image": {"stretch_percentiles": [0.1, 99.9]}}
    (train_dir / "chips_meta.json").write_text(json.dumps(chips))
    config = AgriboundConfig(
        source="local",
        engine="delineate-anything",
        local_tif_path=str(tmp_path / "x.tif"),
        output_path=str(tmp_path / "o.gpkg"),
        reference_boundaries=str(ref_path),
        lulc_filter=False,
        device="cpu",
        cache_dir=str(tmp_path / "run"),
        engine_params={"processing_gsd_m": 3.0, "stretch_percentiles": [0.1, 99.9]},
    )
    best = _finetune_yolo(train_dir, config, "DA-large_v2")
    meta = json.loads(Path(f"{best}.agribound.json").read_text())
    assert meta["processing_gsd_m"] == 3.0 and meta["stretch_percentiles"] == [0.1, 99.9]
    assert meta["chips_stretch"] == {"stretch_percentiles": [0.1, 99.9]}


def test_fine_tuning_and_inference_with_its_checkpoint_read_the_same_input(
    tmp_path, fake_ultralytics, fake_weights, caplog
):
    """fine_tune() with both options, then the native backend with its checkpoint."""
    from agribound.engines.finetune import fine_tune
    from agribound.engines.finetune._data import read_training_meta

    rng = np.random.default_rng(8)
    data = rng.uniform(300, 600, (1, 260, 260)).astype(np.float32)
    data[:, 60:200, 20:240] += 900.0
    tif = _pan_raster(tmp_path, data, pixel=1.0)
    fields = [box(500020, 3999800, 500130, 3999940), box(500130, 3999800, 500240, 3999940)]
    ref = tmp_path / "ref.gpkg"
    gpd.GeoDataFrame({"k": [1, 2]}, geometry=fields, crs="EPSG:32611").to_file(ref)
    options = {"processing_gsd_m": 2.0, "stretch_percentiles": [0.5, 99.5]}
    config = AgriboundConfig(
        source="local",
        local_tif_path=tif,
        bands={"R": 1, "G": 1, "B": 1},
        engine="delineate-anything",
        output_path=str(tmp_path / "out" / "fields.gpkg"),
        device="cpu",
        lulc_filter=False,
        min_field_area_m2=100.0,
        reference_boundaries=str(ref),
        fine_tune_split="random",
        fine_tune_epochs=1,
        seed=3,
        engine_params={**options, "chip_size": 64, "min_label_fraction": 0.0},
    )
    ckpt = fine_tune(tif, config)
    record = read_training_meta(ckpt)
    assert record["processing_gsd_m"] == 2.0 and record["stretch_percentiles"] == [0.5, 99.5]
    assert record["gsd_m"] == pytest.approx(2.0)  # the chips' pixels: the resampled raster
    run = config.merged(engine_params={**config.engine_params, "checkpoint_path": ckpt})
    with caplog.at_level("WARNING"):
        meta = da.DelineateAnythingEngine().delineate(tif, run).attrs["engine_meta"]
    assert "checkpoint_input_mismatch" not in meta
    assert "different input than in training" not in caplog.text
    assert meta["processing"]["size_px"] == record["processing"]["size_px"] == [130, 130]
    chips = record["chips_stretch"]
    assert meta["stretch_lows_bgr"] == chips["stretch_lows"]  # one band, read as R, G and B
    assert meta["stretch_highs_bgr"] == chips["stretch_highs"]
    # Without the options the same checkpoint sees another input: recorded and logged.
    plain = config.merged(engine_params={"checkpoint_path": ckpt})
    meta = da.DelineateAnythingEngine().delineate(tif, plain).attrs["engine_meta"]
    assert meta["checkpoint_input_mismatch"] == {
        "trained": {"gsd_m": 2.0, "super_resolution": 1, "stretch_percentiles": [0.5, 99.5]},
        "run": {"gsd_m": 1.0, "super_resolution": 1, "stretch_percentiles": [1.0, 99.0]},
    }
    assert "different input than in training" in caplog.text
    assert record["raster_dtype"] == "float32"


def test_finetune_yolo_warmup_and_learning_rate_overrides(tmp_path, fake_ultralytics, fake_weights):
    from agribound.engines.finetune._yolo import _finetune_yolo

    train_dir, ref_path = _training_dir(tmp_path, chip=256)
    roots = []
    for params in (
        {},
        {"yolo_warmup_bias_lr": 0.1},
        {"yolo_warmup_bias_lr": 0.1, "yolo_lr0": 1e-4},
    ):
        config = AgriboundConfig(
            source="local",
            engine="delineate-anything",
            local_tif_path=str(tmp_path / "x.tif"),
            output_path=str(tmp_path / "o.gpkg"),
            reference_boundaries=str(ref_path),
            lulc_filter=False,
            device="cpu",
            fine_tune_epochs=3,
            cache_dir=str(tmp_path / "run"),
            engine_params=params,
        )
        best = _finetune_yolo(train_dir, config, "DA-large_v2")
        roots.append(Path(best).parents[2])
    warmup_only = fake_ultralytics.instances[1].train_kwargs
    assert warmup_only["warmup_bias_lr"] == 0.1 and warmup_only["lr0"] == 0.002
    kwargs = fake_ultralytics.instances[2].train_kwargs
    assert kwargs["warmup_bias_lr"] == 0.1 and kwargs["lr0"] == 1e-4
    # warmup_bias_lr and lr0 are each part of the run directory key (the first two runs
    # differ only in warmup_bias_lr, the last two only in lr0): no run directory is shared.
    assert len(set(roots)) == 3


def _yolo_config(tmp_path, ref_path, **engine_params):
    return AgriboundConfig(
        source="local",
        engine="delineate-anything",
        local_tif_path=str(tmp_path / "x.tif"),
        output_path=str(tmp_path / "o.gpkg"),
        reference_boundaries=str(ref_path),
        lulc_filter=False,
        device="cpu",
        fine_tune_epochs=3,
        cache_dir=str(tmp_path / "run"),
        engine_params=engine_params,
    )


def test_finetune_yolo_nominal_batch_size(tmp_path, fake_ultralytics, fake_weights):
    """nbs defaults to the batch size (no accumulation); yolo_nbs sets it and keys the run."""
    from agribound.engines.finetune._yolo import _finetune_yolo

    train_dir, ref_path = _training_dir(tmp_path, chip=256)
    cases = [
        ({}, 16, 16),
        ({"yolo_batch": 8}, 8, 8),
        ({"yolo_nbs": 64}, 16, 64),  # Ultralytics' default: 4 batches per optimizer step
        ({"yolo_nbs": 32.0}, 16, 32),
        ({"yolo_batch": -1}, -1, 64),  # AutoBatch: the batch size is chosen later
    ]
    roots = []
    for params, batch, nbs in cases:
        best = _finetune_yolo(train_dir, _yolo_config(tmp_path, ref_path, **params), "x")
        kwargs = fake_ultralytics.instances[-1].train_kwargs
        assert (kwargs["batch"], kwargs["nbs"]) == (batch, nbs)
        meta = json.loads(Path(f"{best}.agribound.json").read_text())
        assert meta["train_kwargs"]["nbs"] == nbs
        roots.append(Path(best).parents[2])
    assert len(set(roots[:4])) == 4  # nbs is part of the run directory key
    for bad in (0, -64, 1.5, "64", True, float("nan")):
        with pytest.raises(ValueError, match="yolo_nbs"):
            _finetune_yolo(train_dir, _yolo_config(tmp_path, ref_path, yolo_nbs=bad), "x")


def test_finetune_yolo_sliver_depth(tmp_path, fake_ultralytics, fake_weights):
    from agribound.engines.finetune._yolo import _finetune_yolo

    train_dir, ref_path = _training_dir(tmp_path, chip=256)
    default = _finetune_yolo(train_dir, _yolo_config(tmp_path, ref_path), "x")
    keep_all = _finetune_yolo(
        train_dir, _yolo_config(tmp_path, ref_path, yolo_sliver_depth_px=0), "x"
    )
    meta = json.loads(Path(f"{default}.agribound.json").read_text())
    assert meta["sliver_depth_px"] == 16.0 and meta["dataset"]["n_train_slivers"] == 0
    assert json.loads(Path(f"{keep_all}.agribound.json").read_text())["sliver_depth_px"] == 0.0
    assert Path(default).parents[2] != Path(keep_all).parents[2]  # part of the key
    for bad in (-1, float("nan"), float("inf"), "16", True):
        with pytest.raises(ValueError, match="yolo_sliver_depth_px"):
            _finetune_yolo(
                train_dir, _yolo_config(tmp_path, ref_path, yolo_sliver_depth_px=bad), "x"
            )


@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan"), float("inf")])
def test_finetune_yolo_rejects_invalid_warmup_bias_lr(
    tmp_path, fake_ultralytics, fake_weights, value
):
    from agribound.engines.finetune._yolo import _finetune_yolo

    train_dir, ref_path = _training_dir(tmp_path, chip=256)
    config = AgriboundConfig(
        source="local",
        engine="delineate-anything",
        local_tif_path=str(tmp_path / "x.tif"),
        output_path=str(tmp_path / "o.gpkg"),
        reference_boundaries=str(ref_path),
        lulc_filter=False,
        device="cpu",
        cache_dir=str(tmp_path / "run"),
        engine_params={"yolo_warmup_bias_lr": value},
    )
    with pytest.raises(ValueError, match="yolo_warmup_bias_lr"):
        _finetune_yolo(train_dir, config, "DA-large_v2")


def test_finetune_yolo_warns_when_imgsz_is_not_the_model_input(
    tmp_path, fake_ultralytics, fake_weights, caplog
):
    from agribound.engines.finetune._yolo import _finetune_yolo

    train_dir, ref_path = _training_dir(tmp_path, chip=64)  # 64 px x SR 2 = 128 px
    config = AgriboundConfig(
        source="local",
        engine="delineate-anything",
        local_tif_path=str(tmp_path / "x.tif"),
        output_path=str(tmp_path / "o.gpkg"),
        reference_boundaries=str(ref_path),
        lulc_filter=False,
        device="cpu",
        cache_dir=str(tmp_path / "run"),
    )
    with caplog.at_level("WARNING"):
        best = _finetune_yolo(train_dir, config, "DA-large_v2")
    assert "imgsz=128" in caplog.text and "chip_size'] to 256" in caplog.text
    meta = json.loads(Path(f"{best}.agribound.json").read_text())
    assert meta["imgsz"] == 128 and meta["imgsz_matches_model_input"] is False


def test_finetune_yolo_rejects_ftw_backend(tmp_path, fake_ultralytics, fake_weights):
    from agribound.engines.finetune._yolo import _finetune_yolo

    train_dir, ref_path = _training_dir(tmp_path)
    config = AgriboundConfig(
        source="local",
        engine="delineate-anything",
        local_tif_path=str(tmp_path / "x.tif"),
        output_path=str(tmp_path / "o.gpkg"),
        reference_boundaries=str(ref_path),
        lulc_filter=False,
        engine_params={"backend": "ftw"},
    )
    with pytest.raises(ValueError, match="backend"):
        _finetune_yolo(train_dir, config, "x")


def test_finetune_ftw_is_not_supported():
    from agribound.engines.finetune._ftw import _finetune_ftw

    with pytest.raises(NotImplementedError, match="ftw model fit"):
        _finetune_ftw(Path("."), None, "FTW_PRUE_EFNET_B5")


# ---------------------------------------------------------------------------
# Ultralytics' NMS time limit
# ---------------------------------------------------------------------------


def _slow_clock():
    """A stand-in for ``time``: every reading is 100 s after the previous one."""
    import itertools
    import types

    ticks = itertools.count(0, 100)
    return types.SimpleNamespace(time=lambda: next(ticks))


def _one_box_per_image(n_images: int):
    torch = pytest.importorskip("torch")
    pred = torch.zeros((n_images, 5, 1))  # (batch, xywh + 1 class score, boxes)
    pred[:, :4, 0] = torch.tensor([10.0, 10.0, 4.0, 4.0])
    pred[:, 4, 0] = 0.9
    return pred


def test_nms_time_limit_is_lifted_for_predictions_and_validation(monkeypatch):
    """Ultralytics stops NMS after 2 s + 0.05 s per image and drops the batch's later images.

    Example 23's 1.1 run logged "NMS time limit 3.150s exceeded" in 4 validation passes
    of a fine-tuning (12n: 1), on which best.pt is chosen.
    """
    nms = pytest.importorskip("ultralytics.utils.nms")
    from agribound.engines.delineate_anything import no_nms_time_limit

    pred = _one_box_per_image(3)
    original = nms.non_max_suppression
    monkeypatch.setattr(nms, "time", _slow_clock())
    assert [len(o) for o in nms.non_max_suppression(pred, 0.25, 0.45)] == [1, 0, 0]
    with no_nms_time_limit():
        assert [len(o) for o in nms.non_max_suppression(pred, 0.25, 0.45)] == [1, 1, 1]
        # The validator and the predictor call it through the patched module.
        from ultralytics.models.yolo.detect import predict, val

        assert val.nms.non_max_suppression is nms.non_max_suppression is not original
        assert predict.nms.non_max_suppression is nms.non_max_suppression
        with no_nms_time_limit():  # nested: still patched, restored by the outer context
            pass
        assert nms.non_max_suppression is not original
    assert nms.non_max_suppression is original
    with pytest.raises(RuntimeError), no_nms_time_limit():
        raise RuntimeError("training failed")
    assert nms.non_max_suppression is original


def test_no_nms_time_limit_changes_nothing_else():
    torch = pytest.importorskip("torch")
    nms = pytest.importorskip("ultralytics.utils.nms")
    from agribound.engines.delineate_anything import no_nms_time_limit

    gen = torch.Generator().manual_seed(0)
    pred = torch.rand((4, 4 + 1 + 32, 400), generator=gen)  # boxes, 1 class, 32 mask coefficients
    pred[:, :4] *= 512
    plain = nms.non_max_suppression(pred, 0.25, 0.7, nc=1)
    with no_nms_time_limit():
        lifted = nms.non_max_suppression(pred, 0.25, 0.7, nc=1)
    assert [len(o) for o in plain] == [len(o) for o in lifted] and sum(map(len, plain)) > 0
    assert all(torch.equal(a, b) for a, b in zip(plain, lifted, strict=True))


class _NmsProbeYOLO(FakeYOLO):
    """FakeYOLO that records, per predict call, whether the NMS time limit is lifted."""

    lifted: list = []

    def predict(self, images, **kwargs):
        from ultralytics.utils import nms

        _NmsProbeYOLO.lifted.append(
            getattr(nms.non_max_suppression, "_agribound_no_time_limit", False)
        )
        return super().predict(images, **kwargs)


def test_native_predictions_run_without_the_nms_time_limit(tmp_path):
    nms = pytest.importorskip("ultralytics.utils.nms")
    original = nms.non_max_suppression
    _NmsProbeYOLO.lifted = []
    _raw_detections(_field_raster(tmp_path, [(100, 200, 100, 400)]), model=_NmsProbeYOLO("weights"))
    assert _NmsProbeYOLO.lifted and all(_NmsProbeYOLO.lifted)
    assert nms.non_max_suppression is original


def test_finetune_yolo_trains_without_the_nms_time_limit(
    tmp_path, fake_ultralytics, fake_weights, monkeypatch
):
    """best.pt is chosen on the validation passes run inside train()."""
    from agribound.engines.finetune._yolo import _finetune_yolo

    utils, nms = types.ModuleType("ultralytics.utils"), types.ModuleType("ultralytics.utils.nms")

    def non_max_suppression(*args, **kwargs):
        return []

    nms.non_max_suppression, utils.nms = non_max_suppression, nms
    monkeypatch.setattr(sys.modules["ultralytics"], "utils", utils, raising=False)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.nms", nms)
    seen = []

    class ProbeYOLO(fake_ultralytics):
        def train(self, **kwargs):
            seen.append(getattr(nms.non_max_suppression, "_agribound_no_time_limit", False))
            return super().train(**kwargs)

    monkeypatch.setattr(sys.modules["ultralytics"], "YOLO", ProbeYOLO)
    train_dir, ref_path = _training_dir(tmp_path, chip=256)
    config = AgriboundConfig(
        source="local",
        engine="delineate-anything",
        local_tif_path=str(tmp_path / "x.tif"),
        output_path=str(tmp_path / "o.gpkg"),
        reference_boundaries=str(ref_path),
        lulc_filter=False,
        device="cpu",
        fine_tune_epochs=1,
        cache_dir=str(tmp_path / "run"),
    )
    _finetune_yolo(train_dir, config, "DA-large_v2")
    assert seen == [True]
    assert nms.non_max_suppression is non_max_suppression


def test_ftw_backend_predicts_without_the_nms_time_limit(tmp_path, monkeypatch):
    """ftw-tools' instance segmentation predicts with Ultralytics in this process too."""
    nms = pytest.importorskip("ultralytics.utils.nms")
    original = nms.non_max_suppression
    calls: list = []
    _fake_ftw(monkeypatch, {"DelineateAnything": object()}, calls)
    inference = sys.modules["ftw_tools.inference.inference"]
    fake = inference.run_instance_segmentation
    lifted = []

    def probe(*args, **kwargs):
        lifted.append(getattr(nms.non_max_suppression, "_agribound_no_time_limit", False))
        return fake(*args, **kwargs)

    monkeypatch.setattr(inference, "run_instance_segmentation", probe)
    data = np.full((3, 64, 64), 1500.0, dtype=np.float32)
    tif = _write(tmp_path / "s2.tif", data, nodata=float("nan"))
    config = _config(
        tmp_path,
        tif,
        backend="ftw",
        da_model="large",
        value_scale="reflectance_x10000",
        patch_size=32,
    )
    da.DelineateAnythingEngine().delineate(tif, config)
    assert lifted == [True] and len(calls) == 1
    assert nms.non_max_suppression is original
