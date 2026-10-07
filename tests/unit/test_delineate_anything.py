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
from rasterio.transform import from_origin
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
    ],
)
def test_options_reject_unsupported_or_invalid(params, match):
    with pytest.raises(ValueError, match=match):
        da.DAOptions.from_engine_params(params)


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
    assert Path(best).is_file() and Path(best).name == "best.pt"
    assert str(Path(best)).startswith(str(tmp_path / "run"))
    meta = json.loads(Path(f"{best}.agribound.json").read_text())
    assert meta["base_model_key"] == "large_v2" and meta["super_resolution"] == 2
    assert meta["imgsz_matches_model_input"] is True
    assert meta["recipe_version"] == 2 and meta["train_kwargs"]["warmup_bias_lr"] == 0.0


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
