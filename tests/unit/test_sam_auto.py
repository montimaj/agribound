"""Tests of engine ``sam-auto`` (direct SAM 2.1 automatic masks) with a fake generator.

No SAM weights, torch, sam2 or OpenCV are needed, except in the tests that skip
without them. The fake generator stands in for SAM: the test rasters are uint8
(used without a stretch) and hold object ids in band 1, and every distinct id
in a tile gives one mask. Most tests make SAM's input as large as the tile
(``SAM_INPUT_PX`` patched to 128), so that no tile is resampled.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from agribound.config import AgriboundConfig
from agribound.engines import sam_auto
from agribound.engines.sam_auto import (
    MaskCandidate,
    SAMAutoEngine,
    TileCandidates,
    box_nms,
    check_params,
    cut_by_tile,
    interior_edges,
    labels_to_polygons,
    largest_whole_object_px,
    masks_to_boxes,
    partition_masks,
    resolve_model,
    resolve_settings,
    settings_warning,
    tile_origins,
    tile_stride,
)
from agribound.registry import ENGINE_REGISTRY

UTM = "EPSG:32611"
ORIGIN = (500000.0, 4002000.0)
BBOX = "bbox:-116.05,39.55,-115.95,39.62"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path, data, *, res=10.0, crs=UTM, nodata=None, transform=None, tags=None):
    data = np.asarray(data)
    count, height, width = data.shape
    transform = transform or Affine(res, 0.0, ORIGIN[0], 0.0, -res, ORIGIN[1])
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(data.dtype),
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if tags:
            dst.update_tags(**tags)
    return str(path)


def _objects(path, shape, objects, *, res=10.0):
    """uint8 R, G, B: object ids in R (0 = background); G and B are 50, so every pixel is valid."""
    h, w = shape
    data = np.zeros((3, h, w), dtype=np.uint8)
    data[1:] = 50
    for oid, (r0, r1, c0, c1) in objects.items():
        data[0, r0:r1, c0:c1] = oid
    return _write(path, data, res=res)


class FakeGenerator:
    """One mask per nonzero id in the image's red channel; *groups* add unions of ids.

    *scores* maps an id to its (predicted IoU, stability); others get *default*.
    The device floors are applied as the real generator does.
    """

    model_id = "fake/sam"
    device = "cpu"

    def __init__(self, scores=None, groups=(), default=(0.9, 0.97)):
        self.scores = dict(scores or {})
        self.groups = list(groups)
        self.default = default
        self.images = []

    def candidates(
        self, image, *, points_per_side, points_per_batch, iou_floor, stability_floor, timing
    ):
        self.images.append(image.shape)
        timing["decode_s"] = timing.get("decode_s", 0.0) + 0.001
        ids = image[..., 0]
        masks, ious, stabs = [], [], []
        for v in np.unique(ids):
            if v == 0:
                continue
            iou, stab = self.scores.get(int(v), self.default)
            masks.append(ids == v)
            ious.append(iou)
            stabs.append(stab)
        for members, (iou, stab) in self.groups:
            m = np.isin(ids, members)
            if m.any():
                masks.append(m)
                ious.append(iou)
                stabs.append(stab)
        keep = [k for k in range(len(ious)) if ious[k] > iou_floor and stabs[k] >= stability_floor]
        if not keep:
            return TileCandidates.from_masks(np.zeros((0, *ids.shape), bool), [], [])
        return TileCandidates.from_masks(
            np.stack([masks[k] for k in keep]),
            [ious[k] for k in keep],
            [stabs[k] for k in keep],
        )

    def describe(self):
        return {
            "model": self.model_id,
            "checkpoint": {"file": "fake.pt", "revision": "0", "sha256": "0" * 64},
            "device": self.device,
            "torch": None,
            "torchvision": None,
            "sam2": {},
        }


class FixedGenerator(FakeGenerator):
    """Returns the same masks (given in SAM input pixels) for every image."""

    def __init__(self, masks, scores=None):
        super().__init__()
        self.fixed = np.asarray(masks, dtype=bool)
        self.fixed_scores = scores or [(0.9, 0.97)] * len(self.fixed)

    def candidates(
        self, image, *, points_per_side, points_per_batch, iou_floor, stability_floor, timing
    ):
        self.images.append(image.shape)
        return TileCandidates.from_masks(
            self.fixed, [s[0] for s in self.fixed_scores], [s[1] for s in self.fixed_scores]
        )


@pytest.fixture
def small_input(monkeypatch):
    """SAM's input as large as the test tiles (128 px), so that no tile is resampled."""
    monkeypatch.setattr(sam_auto, "SAM_INPUT_PX", 128)


def _config(tmp_path, raster, **params):
    min_area = params.pop("min_field_area_m2", 2500.0)
    return AgriboundConfig(
        source="local",
        engine="sam-auto",
        local_tif_path=str(raster),
        output_path=str(tmp_path / "out" / "fields.gpkg"),
        cache_dir=str(tmp_path / "cache"),
        lulc_filter=False,
        device="cpu",
        min_field_area_m2=min_area,
        engine_params={"sam_tile_m": 1280.0, **params},
    )


def _px_box(r0, r1, c0, c1, res=10.0):
    """Map bounds of pixel rows r0:r1 and columns c0:c1 of the test grid."""
    return (
        ORIGIN[0] + c0 * res,
        ORIGIN[1] - r1 * res,
        ORIGIN[0] + c1 * res,
        ORIGIN[1] - r0 * res,
    )


# ---------------------------------------------------------------------------
# Tiles and the cut-edge rule
# ---------------------------------------------------------------------------


class TestTiles:
    def test_full_tiles_and_the_last_shifted_inwards(self):
        assert tile_origins(5001, 2048, 1024) == [0, 1024, 2048, 2953]
        assert tile_origins(4096, 2048, 1024) == [0, 1024, 2048]
        assert tile_origins(192, 128, 64) == [0, 64]
        assert tile_origins(200, 128, 64) == [0, 64, 72]

    def test_one_padded_tile_when_the_raster_is_smaller(self):
        assert tile_origins(834, 1024, 512) == [0]
        assert tile_origins(1024, 1024, 512) == [0]

    def test_stride(self):
        assert tile_stride(835, 0.5) == 417 == 835 // 2
        assert tile_stride(2048, 0.5) == 1024
        assert tile_stride(1024, 0.0) == 1024
        assert tile_stride(1024, 0.75) == 256
        assert tile_stride(3, 0.9) == 1

    def test_interior_edges(self):
        # A raster of 192 px with 128 px tiles at origins 0 and 64.
        assert interior_edges(0, 0, 128, 192, 192) == (False, False, True, True)
        assert interior_edges(64, 64, 128, 192, 192) == (True, True, False, False)
        # One padded tile: no edge is interior.
        assert interior_edges(0, 0, 1024, 834, 834) == (False, False, False, False)

    @pytest.mark.parametrize(
        ("box", "cut"),
        [
            ((3, 10, 50, 50), False),  # 3 px from the left edge: kept
            ((2, 10, 50, 50), True),  # within 2 px: cut
            ((10, 10, 124, 50), False),
            ((10, 10, 125, 50), True),  # 127 - 2 = 125
            ((10, 2, 50, 50), True),
            ((10, 10, 50, 125), True),
        ],
    )
    def test_cut_by_an_interior_edge(self, box, cut):
        assert cut_by_tile(box, (True, True, True, True), 128, 2) is cut

    @pytest.mark.parametrize("tile_px", [16, 21, 64])
    @pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5, 0.75])
    @pytest.mark.parametrize("margin", [0, 2])
    def test_largest_whole_object_matches_a_brute_force_search(self, tile_px, overlap, margin):
        """The bound holds for every position away from the raster's borders, and no larger
        object is whole in some tile wherever it lies (review L1: it depends on the overlap)."""
        stride = tile_stride(tile_px, overlap)
        size = 12 * tile_px
        origins = tile_origins(size, tile_px, stride)

        def whole_somewhere(a, length):
            for o in origins:
                x0, x1 = a - o, a + length - 1 - o
                if x0 < 0 or x1 > tile_px - 1:
                    continue  # not inside this tile
                left, _, right, _ = interior_edges(o, 0, tile_px, size, tile_px)
                box = (x0, margin + 1, x1, margin + 1)
                if not cut_by_tile(box, (left, False, right, False), tile_px, margin):
                    return True
            return False

        positions = range(2 * tile_px, size - 3 * tile_px)
        whole = [n for n in range(1, tile_px + 1) if all(whole_somewhere(a, n) for a in positions)]
        assert max(whole, default=0) == largest_whole_object_px(tile_px, stride, margin)
        if overlap == 0.5 and margin == 2:
            assert largest_whole_object_px(tile_px, stride, margin) == tile_px - stride - 5

    def test_engine_records_the_bound_of_its_overlap(self, tmp_path, small_input):
        raster = _objects(tmp_path / "o.tif", (100, 100), {1: (10, 30, 10, 30)})
        cfg = _config(tmp_path, raster, sam_tile_overlap=0.75)
        meta = SAMAutoEngine(generator=FakeGenerator()).delineate(raster, cfg).attrs["engine_meta"]
        # 128 px tiles every 32 px: 128 - 32 - 4 - 1 = 91 px of 10 m
        assert meta["tiles"]["stride_px"] == 32
        assert meta["tiles"]["largest_whole_object_m"] == 910.0

    def test_raster_edges_never_cut(self):
        assert not cut_by_tile((0, 0, 127, 127), (False, False, False, False), 128, 2)
        assert cut_by_tile((0, 10, 50, 50), (True, False, False, False), 128, 0)
        assert not cut_by_tile((1, 10, 50, 50), (True, False, False, False), 128, 0)


class TestEndToEndTiling:
    """192 x 192 px at 10 m, 128 px tiles, origins 0 and 64 on both axes: 4 tiles."""

    OBJECTS = {
        1: (10, 30, 10, 30),  # A: inside tile (0, 0) only
        2: (20, 40, 110, 140),  # B: cut by tile (0, 0)'s right edge, whole in tile (64, 0)
        3: (70, 90, 0, 16),  # C: on the raster's left edge, whole in tiles (0, 0), (0, 64)
        4: (150, 160, 20, 170),  # E: wider than a tile, cut in every tile
        5: (2, 5, 60, 63),  # F: 9 px, too small
    }

    def _run(self, tmp_path, generator=None, **params):
        tmp_path.mkdir(parents=True, exist_ok=True)
        raster = _objects(tmp_path / "objects.tif", (192, 192), self.OBJECTS)
        gen = generator or FakeGenerator()
        gdf = SAMAutoEngine(generator=gen).delineate(raster, _config(tmp_path, raster, **params))
        return gdf, gen

    def test_cut_duplicate_and_kept_masks(self, tmp_path, small_input):
        gdf, gen = self._run(tmp_path)
        assert gen.images == [(128, 128, 3)] * 4
        meta = gdf.attrs["engine_meta"]
        assert meta["tiles"]["n_tiles"] == 4 and meta["tiles"]["tile_px"] == 128
        assert meta["tiles"]["stride_px"] == 64
        counts = meta["counts"]
        # Tile (0,0): A, B (cut), C, F (small); (64,0): B; (0,64): C, E (cut); (64,64): E (cut).
        assert counts["n_survivors"] == 8
        assert counts["n_cut"] == 3
        assert counts["n_small"] == 1
        assert counts["n_large"] == 0
        assert counts["n_overlap_dropped"] == 1  # C's second copy
        assert counts["n_kept"] == 3 and counts["n_polygons"] == 3
        # Equal scores: smaller first, so C (320 px), A (400 px), B (600 px).
        assert list(gdf["sam_label"]) == [1, 2, 3]
        expected = [_px_box(70, 90, 0, 16), _px_box(10, 30, 10, 30), _px_box(20, 40, 110, 140)]
        for geom, bounds in zip(gdf.geometry, expected, strict=True):
            assert geom.bounds == pytest.approx(bounds)
        assert list(gdf.geometry.area) == pytest.approx([32_000.0, 40_000.0, 60_000.0])
        assert gdf.crs == UTM

    def test_scores_order_the_partition(self, tmp_path, small_input):
        gdf, _ = self._run(tmp_path, FakeGenerator(scores={2: (0.99, 0.99), 1: (0.95, 0.99)}))
        assert list(gdf.geometry.area) == pytest.approx([60_000.0, 40_000.0, 32_000.0])
        assert list(gdf["sam_iou"]) == pytest.approx([0.99, 0.95, 0.9])

    def test_thresholds_and_floors(self, tmp_path, small_input):
        # B below the predicted-IoU threshold (0.8 for a local raster); A below the stability one.
        gen = FakeGenerator(scores={2: (0.79, 0.99), 1: (0.9, 0.94)})
        gdf, _ = self._run(tmp_path, gen)
        assert list(gdf.geometry.area) == pytest.approx([32_000.0])
        # Scores are float32: 0.79 is 0.7900000214..., above a threshold of 0.79 in float64.
        gdf, _ = self._run(tmp_path, gen, pred_iou_thresh=0.79, stability_score_thresh=0.9)
        assert sorted(gdf.geometry.area) == pytest.approx([32_000.0, 40_000.0, 60_000.0])

    def test_area_asc_order(self, tmp_path, small_input):
        gen = FakeGenerator(scores={2: (0.99, 0.99)})
        gdf, _ = self._run(tmp_path, gen, partition_order="area_asc")
        assert list(gdf.geometry.area) == pytest.approx([32_000.0, 40_000.0, 60_000.0])

    def test_same_output_twice(self, tmp_path, small_input):
        a, _ = self._run(tmp_path / "a")
        b, _ = self._run(tmp_path / "b")
        assert [g.wkb for g in a.geometry] == [g.wkb for g in b.geometry]
        assert a.drop(columns="geometry").equals(b.drop(columns="geometry"))


# ---------------------------------------------------------------------------
# Filters, validity, partition and polygons
# ---------------------------------------------------------------------------


class TestFilters:
    """One 100 x 100 px raster at 10 m in one padded 128 px tile."""

    def _run(self, tmp_path, objects, **params):
        raster = _objects(tmp_path / "f.tif", (100, 100), objects)
        gdf = SAMAutoEngine(generator=FakeGenerator()).delineate(
            raster, _config(tmp_path, raster, **params)
        )
        return gdf, gdf.attrs["engine_meta"]["counts"]

    def test_small_and_large(self, tmp_path, small_input):
        objects = {
            1: (0, 3, 0, 3),  # 9 px < min_mask_px 16
            2: (5, 9, 5, 10),  # 20 px = 2,000 m2 < min_field_area_m2
            3: (20, 90, 20, 90),  # 4,900 px > 0.25 x 128^2
            4: (92, 98, 0, 100),  # 600 px: kept
        }
        gdf, counts = self._run(tmp_path, objects, max_tile_fraction=0.25)
        assert counts["n_small"] == 2 and counts["n_large"] == 1 and len(gdf) == 1
        gdf, counts = self._run(tmp_path, objects)  # 0.5 x 128^2 = 8,192 px: object 3 is kept
        assert counts["n_large"] == 0 and len(gdf) == 2
        gdf, counts = self._run(tmp_path, objects, max_field_area_m2=100_000.0)
        assert counts["n_large"] == 1 and len(gdf) == 1  # 490,000 m2 > 100,000 m2
        gdf, counts = self._run(tmp_path, objects, min_mask_px=10, min_field_area_m2=0.0)
        assert counts["n_small"] == 1 and len(gdf) == 3

    def test_nested_masks_end_to_end(self, tmp_path, small_input):
        """A group of two fields: the fields win when they score higher, the group otherwise."""
        objects = {1: (10, 30, 10, 40), 2: (10, 30, 40, 70)}  # two touching fields, 600 px each
        raster = _objects(tmp_path / "g.tif", (100, 100), objects)
        lower = FakeGenerator(
            scores={1: (0.95, 0.97), 2: (0.95, 0.97)}, groups=[([1, 2], (0.9, 0.97))]
        )
        gdf = SAMAutoEngine(generator=lower).delineate(raster, _config(tmp_path, raster))
        counts = gdf.attrs["engine_meta"]["counts"]
        assert len(gdf) == 2 and counts["n_overlap_dropped"] == 1
        higher = FakeGenerator(groups=[([1, 2], (0.99, 0.99))])
        gdf = SAMAutoEngine(generator=higher).delineate(raster, _config(tmp_path, raster))
        assert len(gdf) == 1 and gdf.geometry.iloc[0].bounds == pytest.approx(
            _px_box(10, 30, 10, 70)
        )
        assert gdf.attrs["engine_meta"]["counts"]["n_overlap_dropped"] == 2

    def test_no_masks(self, tmp_path, small_input):
        gdf, counts = self._run(tmp_path, {})
        assert len(gdf) == 0 and counts["n_candidates"] == 0
        assert list(gdf.columns) == ["sam_label", "sam_iou", "sam_stability", "geometry"]
        assert gdf.crs == UTM


class TestValidity:
    def test_validity_is_read_from_the_raw_data(self, tmp_path, small_input):
        """Regression: dark valid pixels (below the 1st percentile) stay in the mask."""
        rng = np.random.default_rng(0)
        data = rng.uniform(1000, 3000, (3, 100, 100)).astype(np.float32)
        data[:, 20:40, 20:60] = 5.0  # dark but valid: stretched to 0 in all three bands
        data[:, 40:50, 20:60] = 0.0  # 0 in every band: nodata
        data[0, 50:60, 20:60] = np.nan  # not finite
        raster = _write(tmp_path / "dark.tif", data, nodata=float("nan"))
        mask = np.zeros((1, 128, 128), dtype=bool)
        mask[0, 20:60, 20:60] = True
        engine = SAMAutoEngine(generator=FixedGenerator(mask))
        gdf = engine.delineate(raster, _config(tmp_path, raster))
        assert len(gdf) == 1
        assert gdf.geometry.iloc[0].area == pytest.approx(20 * 40 * 100.0)
        assert gdf.geometry.iloc[0].bounds == pytest.approx(_px_box(20, 40, 20, 60))
        stretch = gdf.attrs["engine_meta"]["stretch"]
        assert stretch["method"] == "percentile" and stretch["percentiles"] == [1.0, 99.0]
        assert all(low <= 5.0 for low in stretch["lows"])  # the dark field sets the 1st percentile

    def test_nodata_value_and_padding(self, tmp_path, small_input):
        data = np.full((3, 50, 50), 7, dtype=np.uint8)
        data[:, :10, :] = 255  # nodata in all bands
        raster = _write(tmp_path / "nd.tif", data, nodata=255)
        mask = np.zeros((1, 128, 128), dtype=bool)
        mask[0, 0:30, 0:90] = True  # reaches into the padding beyond column 49
        gdf = SAMAutoEngine(generator=FixedGenerator(mask)).delineate(
            raster, _config(tmp_path, raster)
        )
        assert gdf.geometry.iloc[0].bounds == pytest.approx(_px_box(10, 30, 0, 50))


def _candidate(r0, r1, c0, c1, iou=0.9, stab=0.97, mask=None):
    m = np.ones((r1 - r0, c1 - c0), dtype=bool) if mask is None else mask
    return MaskCandidate(
        col=c0,
        row=r0,
        packed=np.packbits(m, axis=-1),
        shape=m.shape,
        area_px=int(m.sum()),
        iou=iou,
        stability=stab,
    )


class TestPartition:
    def test_nested_masks(self):
        inner = _candidate(10, 20, 10, 20, iou=0.95)  # 100 px
        outer = _candidate(5, 25, 5, 25, iou=0.9)  # 400 px, 300 left: kept as a ring
        labels, scores, dropped = partition_masks([outer, inner], 30, 30)
        assert dropped == 0 and scores == {1: (0.95, 0.97), 2: (0.9, 0.97)}
        assert (labels == 1).sum() == 100 and (labels == 2).sum() == 300
        group = _candidate(8, 22, 8, 22, iou=0.9)  # 196 px, 96 left (< half): dropped
        labels, scores, dropped = partition_masks([group, inner], 30, 30)
        assert dropped == 1 and list(scores) == [1] and (labels == 1).sum() == 100

    def test_duplicates_from_overlapping_tiles(self):
        a = _candidate(0, 10, 0, 10)
        b = _candidate(0, 10, 0, 10)
        labels, scores, dropped = partition_masks([a, b], 10, 10)
        assert dropped == 1 and len(scores) == 1 and (labels == 1).all()

    def test_equal_scores_smaller_first_then_input_order(self):
        big = _candidate(0, 10, 0, 10, iou=0.9, stab=0.97)
        small = _candidate(0, 5, 0, 5, iou=0.9, stab=0.97)
        labels, scores, _ = partition_masks([big, small], 10, 10)
        assert (labels == 1).sum() == 25 and (labels == 2).sum() == 75
        first = _candidate(0, 4, 0, 4)
        second = _candidate(0, 4, 0, 4)
        second.iou = first.iou = 0.8
        labels, scores, dropped = partition_masks([first, second], 4, 4)
        assert dropped == 1 and (labels == 1).all()
        # Stability breaks a tie of predicted IoU.
        steady = _candidate(0, 10, 0, 10, iou=0.9, stab=0.99)
        labels, _, _ = partition_masks([small, steady], 10, 10)
        assert (labels == 1).sum() == 100

    def test_min_keep(self):
        first = _candidate(0, 10, 0, 10, iou=0.99)
        half = _candidate(0, 10, 5, 15, iou=0.9)  # 100 px, 50 free: exactly half, kept
        _, scores, dropped = partition_masks([first, half], 10, 20, min_keep=0.5)
        assert dropped == 0 and len(scores) == 2
        _, scores, dropped = partition_masks([first, half], 10, 20, min_keep=0.51)
        assert dropped == 1 and len(scores) == 1
        _, scores, dropped = partition_masks([first, half], 10, 20, min_keep=0.0)
        assert dropped == 0

    def test_a_mask_that_claims_no_pixel_is_dropped_also_at_min_keep_0(self):
        """Every kept mask holds pixels, so the counts equal the polygons (review L2)."""
        first = _candidate(0, 10, 0, 10, iou=0.99)
        covered = _candidate(2, 8, 2, 8, iou=0.9)  # all of its pixels already claimed
        transform = Affine(10.0, 0.0, ORIGIN[0], 0.0, -10.0, ORIGIN[1])
        for min_keep in (0.0, 0.5):
            labels, scores, dropped = partition_masks([first, covered], 10, 10, min_keep=min_keep)
            assert dropped == 1 and list(scores) == [1] and (labels == 1).all()
            assert len(labels_to_polygons(labels, scores, transform, UTM)) == len(scores)

    def test_area_asc(self):
        small = _candidate(0, 5, 0, 5, iou=0.8)
        big = _candidate(0, 10, 0, 10, iou=0.99)
        labels, scores, _ = partition_masks([big, small], 10, 10, order="area_asc")
        assert scores[1] == (0.8, 0.97) and (labels == 1).sum() == 25
        with pytest.raises(ValueError, match="partition order"):
            partition_masks([big], 10, 10, order="random")


class TestPolygons:
    def test_one_polygon_per_label_its_largest_part(self):
        divider = _candidate(0, 41, 40, 51, iou=0.99)  # columns 40-50
        split = _candidate(0, 41, 10, 91, iou=0.9)  # its free pixels: 10-39 and 51-90
        labels, scores, _ = partition_masks([split, divider], 41, 100)
        transform = Affine(10.0, 0.0, ORIGIN[0], 0.0, -10.0, ORIGIN[1])
        gdf = labels_to_polygons(labels, scores, transform, UTM)
        assert list(gdf["sam_label"]) == [1, 2]
        assert gdf.geometry.iloc[1].bounds == pytest.approx(_px_box(0, 41, 51, 91))
        assert gdf.geometry.iloc[1].area == pytest.approx(41 * 40 * 100.0)
        assert list(gdf["sam_iou"]) == [0.99, 0.9]

    def test_empty_labels(self):
        transform = Affine(10.0, 0.0, ORIGIN[0], 0.0, -10.0, ORIGIN[1])
        gdf = labels_to_polygons(np.zeros((5, 5), np.int32), {}, transform, UTM)
        assert len(gdf) == 0 and isinstance(gdf, gpd.GeoDataFrame)


# ---------------------------------------------------------------------------
# NMS and candidates
# ---------------------------------------------------------------------------


class TestNMS:
    def test_suppression_and_order(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [0, 0, 10, 10]])
        scores = np.array([0.8, 0.9, 0.7, 0.9], dtype=np.float32)
        # Visited 1, 3 (equal scores: input order), 0, 2. IoU(1, 3) = IoU(1, 0) = 81 / 119 =
        # 0.68, IoU(3, 0) = 1: at 0.7, 3 suppresses 0; at 0.5, 1 suppresses 3 and 0.
        assert list(box_nms(boxes, scores, 0.7)) == [1, 2, 3]
        assert list(box_nms(boxes, scores, 0.5)) == [1, 2]
        assert list(box_nms(np.zeros((0, 4)), np.zeros(0), 0.7)) == []

    def test_zero_area_boxes(self):
        boxes = np.array([[5, 5, 5, 20], [5, 5, 5, 20]], dtype=np.float32)  # one column wide
        assert list(box_nms(boxes, np.array([0.9, 0.8]), 0.7)) == [0, 1]

    def test_equals_torchvision(self):
        torch = pytest.importorskip("torch")
        ops = pytest.importorskip("torchvision.ops")
        rng = np.random.default_rng(0)
        for trial in range(60):
            n = int(rng.integers(1, 300))
            x1, y1 = rng.integers(0, 1000, n), rng.integers(0, 1000, n)
            w, h = rng.integers(0, 300, n), rng.integers(0, 300, n)
            boxes = np.stack([x1, y1, np.minimum(x1 + w, 1023), np.minimum(y1 + h, 1023)], 1)
            boxes = boxes.astype(np.float32)
            scores = rng.choice(np.float32([0.81, 0.9, 0.9, 0.95]), n).astype(np.float32)
            threshold = (0.7, 0.5, 0.3)[trial % 3]
            ref = ops.batched_nms(
                torch.as_tensor(boxes), torch.as_tensor(scores), torch.zeros(n), threshold
            )
            assert np.array_equal(np.sort(ref.numpy()), box_nms(boxes, scores, threshold))


class TestCandidates:
    def test_boxes_as_sam2(self):
        masks = np.zeros((2, 6, 9), dtype=bool)
        masks[0, 1:4, 2:7] = True
        boxes = masks_to_boxes(masks)
        assert boxes.dtype == np.float32
        assert boxes.tolist() == [[2, 1, 6, 3], [0, 0, 0, 0]]

    def test_pack_round_trip(self):
        masks = np.random.default_rng(0).random((3, 7, 13)) > 0.5
        cand = TileCandidates.from_masks(masks, [0.9, 0.8, 0.7], [0.95, 0.9, 0.85])
        assert cand.masks.shape == (3, 7, 2) and cand.width == 13
        for i in range(3):
            assert np.array_equal(cand.mask(i), masks[i])
        assert cand.iou.dtype == np.float64 and cand.iou[1] == np.float64(np.float32(0.8))
        with pytest.raises(ValueError, match=r"\(n, h, w\)"):
            TileCandidates.from_masks(np.zeros((4, 4), bool), [], [])

    def test_boxes_match_sam2(self):
        torch = pytest.importorskip("torch")
        amg = pytest.importorskip("sam2.utils.amg")
        masks = np.random.default_rng(1).random((5, 32, 40)) > 0.97
        masks[4] = False
        expected = amg.batched_mask_to_box(torch.as_tensor(masks)).float().numpy()
        assert np.array_equal(masks_to_boxes(masks), expected)


# ---------------------------------------------------------------------------
# Settings per source and their warnings
# ---------------------------------------------------------------------------


class TestSettings:
    @pytest.mark.parametrize(
        ("source", "gsd", "expected", "basis"),
        [
            ("naip", 1.0, (2048.0, 32, 0.8, 0.95), "default"),
            ("usgs-naip-plus", 0.6, (614.4, 32, 0.8, 0.95), "native scale, not piloted"),
            ("spot", 6.0, (6144.0, 32, 0.6, 0.95), "default"),
            ("sentinel2", 10.0, (5120.0, 32, 0.8, 0.9), "default, costly at scale"),
            ("landsat", 30.0, (7680.0, 32, 0.6, 0.8), "default, costly at scale"),
            ("hls", 30.0, (30720.0, 32, 0.8, 0.95), "native scale, not piloted"),
            ("spot-pan", 1.5, (1536.0, 32, 0.8, 0.95), "native scale, not piloted"),
            ("local", 0.5, (512.0, 32, 0.8, 0.95), "native scale, not piloted"),
        ],
    )
    def test_per_source_settings(self, source, gsd, expected, basis):
        s = resolve_settings(source, {}, gsd)
        p = s.params
        got = (p["sam_tile_m"], p["points_per_side"], p["pred_iou_thresh"])
        assert (*got, p["stability_score_thresh"]) == expected
        assert s.basis == basis and s.user_keys == []
        assert p["sam_model"] == "facebook/sam2.1-hiera-large"
        assert p["sam_tile_overlap"] == 0.5 and p["box_nms_thresh"] == 0.7
        assert p["tile_edge_margin_px"] == 2 and p["min_mask_px"] == 16
        assert p["max_field_area_m2"] == 2.59e6 and p["max_tile_fraction"] == 0.5
        assert p["partition_order"] == "score" and p["min_keep"] == 0.5
        assert p["points_per_batch"] == 64

    def test_the_frozen_selection(self):
        """The defaults are the pilot's selections (sam-direct runs/selection.json, 2026-10-12)."""
        selection = {
            "naip": (2048.0, 32, 0.8, 0.95),
            "spot": (6144.0, 32, 0.6, 0.95),
            "sentinel2": (5120.0, 32, 0.8, 0.9),
            "landsat": (7680.0, 32, 0.6, 0.8),
        }
        for source, values in selection.items():
            assert tuple(sam_auto.SOURCE_SETTINGS[source].values()) == values
        assert "sentinel1" not in sam_auto.SOURCE_SETTINGS
        # usgs-naip-plus shows NAIP's imagery but was not piloted (round-2 plan, decision S7)
        assert "usgs-naip-plus" not in sam_auto.SOURCE_SETTINGS

    def test_user_values_win(self):
        s = resolve_settings("naip", {"sam_tile_m": 1024, "points_per_side": 64.0}, 1.0)
        assert s.params["sam_tile_m"] == 1024.0 and s.params["points_per_side"] == 64
        assert s.user_keys == ["points_per_side", "sam_tile_m"] and s.basis == "default"
        s = resolve_settings("naip", {"sam_model": "tiny", "smooth_iterations": 0}, 1.0)
        assert s.params["sam_model"] == "facebook/sam2.1-hiera-tiny"

    def test_warnings(self):
        assert settings_warning("naip", resolve_settings("naip", {}, 1.0)) is None
        assert settings_warning("spot", resolve_settings("spot", {}, 6.0)) is None
        # The cost measured back to back in the pre-registered evaluation (PREREG-SAM 5), and on
        # Sentinel-2 its result there: it did not transfer (PREREG-SAM 4).
        s2 = settings_warning("sentinel2", resolve_settings("sentinel2", {}, 10.0))
        assert "17.0 times" in s2 and "0.035 s" in s2 and "0.59 s" in s2 and "in-sample" in s2
        assert "back to back on the same composites" in s2 and "5,120 m tiles" in s2
        # The boxes are 415 and 496 km2; the seconds per km2 divide by their export grids.
        assert "boxes of 415 and 496 km2" in s2 and "about 430 and 526 km2" in s2
        assert "did not transfer" in s2 and "0.15 below" in s2 and "-0.18 to -0.13" in s2
        assert "a quarter as many polygons (mean recall 0.09 against 0.26)" in s2
        ls = settings_warning("landsat", resolve_settings("landsat", {}, 30.0))
        assert "20.7 times" in ls and "0.011 s" in ls and "0.23 s" in ls
        assert "did not transfer" not in ls
        assert "approximate" not in s2 + ls  # the pilot's ratios on other composites are gone
        assert sam_auto.COSTLY_SOURCES == {
            "sentinel2": {"sam_s_per_km2": 0.59, "da_s_per_km2": 0.035, "ratio": 17.0},
            "landsat": {"sam_s_per_km2": 0.23, "da_s_per_km2": 0.011, "ratio": 20.7},
        }
        # "Did not transfer" changes no setting (PREREG-SAM 4): Sentinel-2 keeps its default.
        assert set(sam_auto.DID_NOT_TRANSFER) == {"sentinel2"}
        assert sam_auto.SOURCE_SETTINGS["sentinel2"]["sam_tile_m"] == 5120.0
        hls = settings_warning("hls", resolve_settings("hls", {}, 30.0))
        assert "no setting was piloted for source 'hls'" in hls
        assert "native scale and sam2's own thresholds" in hls and "30,720 m tiles" in hls
        given = settings_warning("hls", resolve_settings("hls", {"sam_tile_m": 9000}, 30.0))
        assert "engine_params given: sam_tile_m" in given
        plus = settings_warning("usgs-naip-plus", resolve_settings("usgs-naip-plus", {}, 0.6))
        assert "no setting was piloted for source 'usgs-naip-plus'" in plus

    def test_sentinel1_needs_every_setting(self):
        with pytest.raises(ValueError, match="no default setting for source 'sentinel1'") as err:
            check_params("sentinel1", {"sam_tile_m": 2560})
        assert "0.437 against 0.441" in str(err.value) and "22" in str(err.value)
        # The selection rule chose that setting; the highest of the 72 had 0.443.
        assert "selection rule chose" in str(err.value) and "0.443" in str(err.value)
        assert "missing: ['points_per_side', 'pred_iou_thresh'" in str(err.value)
        full = {
            "sam_tile_m": 2560,
            "points_per_side": 32,
            "pred_iou_thresh": 0.6,
            "stability_score_thresh": 0.8,
        }
        s = resolve_settings("sentinel1", full, 10.0)
        assert s.basis == "user" and s.params["sam_tile_m"] == 2560.0
        assert settings_warning("sentinel1", s) is None

    def test_engine_logs_the_cost_warning(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(sam_auto, "SAM_INPUT_PX", 512)
        data = np.random.default_rng(0).uniform(500, 3000, (12, 40, 40)).astype(np.float32)
        raster = _write(tmp_path / "s2.tif", data)
        cfg = AgriboundConfig(
            source="sentinel2",
            engine="sam-auto",
            year=2024,
            study_area=BBOX,
            gee_project="p",
            output_path=str(tmp_path / "s2.gpkg"),
            lulc_filter=False,
            device="cpu",
        )
        generator = FixedGenerator(np.zeros((0, 512, 512), dtype=bool))
        with caplog.at_level(logging.WARNING, logger="agribound.engines.sam_auto"):
            gdf = SAMAutoEngine(generator=generator).delineate(raster, cfg)
        assert any("17.0 times" in r.getMessage() for r in caplog.records)
        meta = gdf.attrs["engine_meta"]
        assert meta["settings_basis"] == "default, costly at scale"
        assert "17.0 times" in meta["settings_warning"]
        assert "did not transfer" in meta["settings_warning"]
        assert meta["rgb_bands"] == [4, 3, 2] and meta["tiles"]["tile_px"] == 512


# ---------------------------------------------------------------------------
# Configuration refusals
# ---------------------------------------------------------------------------


def _cfg(**kwargs):
    base = dict(source="naip", engine="sam-auto", year=2022, study_area=BBOX, gee_project="p")
    return AgriboundConfig(**{**base, **kwargs})


class TestConfig:
    def test_valid(self):
        cfg = _cfg(engine_params={"points_per_side": 4.0, "sam_model": "facebook/sam2-hiera-small"})
        assert cfg.engine == "sam-auto"
        assert resolve_model("sam2.1-hiera-base-plus") == "facebook/sam2.1-hiera-base-plus"
        assert resolve_model("Base-Plus") == "facebook/sam2.1-hiera-base-plus"
        assert resolve_model(None) == "facebook/sam2.1-hiera-large"

    def test_refines_nothing(self):
        with pytest.raises(ValueError, match="sam_refine does not apply to engine 'sam-auto'"):
            _cfg(sam_refine=True)
        with pytest.raises(ValueError, match="sam_refine does not apply"):
            _cfg(engine_params={"sam_refine": True})
        _cfg(engine_params={"sam_refine": False, "smooth_iterations": 0, "regularize": "none"})
        _cfg(engine_params={"pivots": "snap"})  # the pipeline's centre-pivot post-process
        with pytest.raises(ValueError, match="set engine_params\\['sam_model'\\] instead"):
            _cfg(sam_model="facebook/sam2.1-hiera-tiny")

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("sam_tile_m", 0),
            ("sam_tile_m", -5.0),
            ("sam_tile_m", "2048"),
            ("sam_tile_m", float("nan")),
            ("sam_tile_overlap", 0.9),
            ("sam_tile_overlap", -0.1),
            ("points_per_side", 3),
            ("points_per_side", 4.5),
            ("points_per_side", True),
            ("points_per_batch", 0),
            ("pred_iou_thresh", 1.5),
            ("stability_score_thresh", -0.1),
            ("box_nms_thresh", 2),
            ("tile_edge_margin_px", -1),
            ("min_mask_px", 0),
            ("max_field_area_m2", 0),
            ("max_tile_fraction", 0),
            ("max_tile_fraction", 1.5),
            ("partition_order", "random"),
            ("min_keep", 1.5),
            ("sam_rgb_bands", [1, 2]),
            ("sam_rgb_bands", [0, 1, 2]),
            ("sam_rgb_bands", "123"),
            ("sam_model", "facebook/sam3"),
            ("sam_model", "huge"),
        ],
    )
    def test_invalid_values(self, key, value):
        with pytest.raises(ValueError, match=key):
            _cfg(engine_params={key: value})

    def test_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown engine_params for engine 'sam-auto'"):
            _cfg(engine_params={"points_per_sides": 32})

    def test_sources(self):
        for source in ("google-embedding", "tessera-embedding", "cdl-sequence", "aci-sequence"):
            with pytest.raises(ValueError, match="does not support source"):
                _cfg(source=source, year=2023)
        assert (
            ENGINE_REGISTRY["sam-auto"]["supported_sources"]
            == ENGINE_REGISTRY["delineate-anything"]["supported_sources"]
        )
        with pytest.raises(ValueError, match="no default setting"):
            _cfg(source="sentinel1", year=2024)

    def test_not_fine_tunable(self):
        with pytest.raises(ValueError, match="does not support fine-tuning"):
            _cfg(fine_tune=True, reference_boundaries="ref.gpkg")

    def test_ensemble_member(self):
        cfg = _cfg(
            engine="ensemble",
            engine_params={"engines": ["delineate-anything", {"engine": "sam-auto"}]},
        )
        from agribound._results import results_versions

        assert results_versions(cfg) == {"delineate_anything": 2, "sam_auto": 1}

    def test_ensemble_member_settings_are_checked_by_the_configuration(self):
        """A sam-auto member is checked when the configuration is built, not after the
        composite (its engine_params, and sentinel1 without every per-source setting)."""

        def ensemble(member, source="naip", year=2022):
            members = ["delineate-anything", member]
            return _cfg(
                engine="ensemble", source=source, year=year, engine_params={"engines": members}
            )

        ensemble({"engine": "sam-auto", "engine_params": {"points_per_side": 16}})
        with pytest.raises(ValueError, match="Unknown engine_params for engine 'sam-auto'"):
            ensemble({"engine": "sam-auto", "engine_params": {"points_per_sides": 16}})
        with pytest.raises(ValueError, match="sam_tile_overlap"):
            ensemble({"engine": "sam-auto", "engine_params": {"sam_tile_overlap": 0.95}})
        for member in ("sam-auto", {"engine": "sam-auto"}):
            with pytest.raises(ValueError, match="no default setting for source 'sentinel1'"):
                ensemble(member, source="sentinel1", year=2024)


# ---------------------------------------------------------------------------
# Engine refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_class_codes(self, tmp_path):
        codes = np.ones((3, 20, 20), dtype=np.uint8)
        raster = _write(
            tmp_path / "codes.tif", codes, tags={"AGRIBOUND_VALUE_SCALE": "categorical"}
        )
        with pytest.raises(ValueError, match="class codes"):
            SAMAutoEngine(generator=FakeGenerator()).delineate(raster, _config(tmp_path, raster))

    def test_rotated_geographic_and_non_square(self, tmp_path):
        data = np.ones((3, 20, 20), dtype=np.uint8)
        rotated = _write(
            tmp_path / "rot.tif",
            data,
            transform=Affine(10.0, 1.0, ORIGIN[0], 0.0, -10.0, ORIGIN[1]),
        )
        geographic = _write(
            tmp_path / "geo.tif",
            data,
            crs="EPSG:4326",
            transform=Affine(1e-4, 0.0, -116.0, 0.0, -1e-4, 39.6),
        )
        oblong = _write(
            tmp_path / "oblong.tif",
            data,
            transform=Affine(10.0, 0.0, ORIGIN[0], 0.0, -20.0, ORIGIN[1]),
        )
        engine = SAMAutoEngine(generator=FakeGenerator())
        for raster, match in (
            (rotated, "rotated or sheared"),
            (geographic, "projected CRS"),
            (oblong, "square pixels"),
        ):
            with pytest.raises(ValueError, match=match):
                engine.delineate(raster, _config(tmp_path, raster))

    def test_band_indices(self, tmp_path):
        raster = _objects(tmp_path / "o.tif", (20, 20), {})
        with pytest.raises(ValueError, match="out of range"):
            SAMAutoEngine(generator=FakeGenerator()).delineate(
                raster, _config(tmp_path, raster, sam_rgb_bands=[1, 2, 5])
            )

    def test_hpc_tiling_is_refused(self):
        from agribound.engines.sam_auto import hpc_tiling_problem

        assert "does not support HPC tiling" in hpc_tiling_problem(_cfg())
        member = _cfg(engine="ensemble", engine_params={"engines": ["sam-auto"]})
        assert "does not support HPC tiling" in hpc_tiling_problem(member)
        assert hpc_tiling_problem(_cfg(engine="delineate-anything")) is None


# ---------------------------------------------------------------------------
# Provenance, pipeline, labels raster
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_engine_meta(self, tmp_path, small_input):
        raster = _objects(tmp_path / "o.tif", (100, 100), {1: (10, 30, 10, 30)})
        gdf = SAMAutoEngine(generator=FakeGenerator()).delineate(raster, _config(tmp_path, raster))
        meta = gdf.attrs["engine_meta"]
        for key in (
            "engine",
            "implementation",
            "model",
            "checkpoint",
            "device",
            "torch",
            "torchvision",
            "sam2",
            "opencv",
            "settings",
            "settings_basis",
            "settings_given",
            "settings_warning",
            "tiles",
            "rgb_bands",
            "stretch",
            "validity",
            "counts",
            "seconds",
            "label_raster_mb",
            "determinism",
        ):
            assert key in meta, key
        assert meta["engine"] == "sam-auto" and meta["checkpoint"]["file"] == "fake.pt"
        assert meta["settings"]["sam_tile_m"] == 1280.0
        assert meta["settings"]["min_field_area_m2"] == 2500.0
        assert meta["settings"]["device_floor_iou"] == 0.6
        assert meta["settings"]["device_floor_stability"] == 0.8
        assert meta["settings_given"] == ["sam_tile_m"]
        # 128 px tiles every 64 px with the 2 px cut margin: 128 - 64 - 4 - 1 = 59 px of 10 m
        assert meta["tiles"]["largest_whole_object_m"] == 590.0
        assert meta["tiles"]["point_spacing_m"] == 40.0
        assert meta["stretch"]["method"] == "none (uint8)"
        assert "raw data" in meta["validity"]
        assert set(meta["seconds"]) >= {"load_s", "encode_s", "decode_s", "post_s", "total_s"}
        json.dumps(meta, default=str)

    def test_pipeline_end_to_end(self, tmp_path, small_input, monkeypatch):
        from agribound.pipeline import delineate
        from agribound.provenance import read_provenance

        raster = _objects(tmp_path / "o.tif", (192, 192), TestEndToEndTiling.OBJECTS)
        loaded = []

        def fake_load(model_id, device):
            loaded.append((model_id, device))
            return FakeGenerator()

        monkeypatch.setattr(sam_auto, "load_generator", fake_load)
        cfg = _config(tmp_path, raster)
        gdf = delineate(config=cfg)
        assert loaded == [("facebook/sam2.1-hiera-large", "cpu")]
        assert len(gdf) == 3 and set(gdf["agribound:engine"]) == {"sam-auto"}
        record = read_provenance(cfg.output_path)
        assert record["facts"]["results_versions"] == {"sam_auto": 1}
        assert record["engine_meta"]["counts"]["n_kept"] == 3
        assert record["engine_meta"]["settings_basis"] == "native scale, not piloted"
        again = delineate(config=cfg)  # reused: the generator is not loaded again
        assert len(again) == 3 and len(loaded) == 1

    def test_large_label_raster_warning(self, tmp_path, small_input, monkeypatch, caplog):
        monkeypatch.setattr(sam_auto, "LABEL_RASTER_WARN_BYTES", 100)
        raster = _objects(tmp_path / "o.tif", (20, 20), {})
        with caplog.at_level(logging.WARNING, logger="agribound.engines.sam_auto"):
            SAMAutoEngine(generator=FakeGenerator()).delineate(raster, _config(tmp_path, raster))
        assert any("int32 label raster" in r.getMessage() for r in caplog.records)


class TestResultsVersions:
    def test_sam_auto_component(self):
        from agribound._results import RESULTS_VERSIONS, results_versions

        assert RESULTS_VERSIONS["sam_auto"] == 1
        assert results_versions(_cfg()) == {"sam_auto": 1}

    def test_other_engines_unchanged(self):
        """Configuration hashes and results versions computed before engine sam-auto existed."""
        from agribound._results import results_versions
        from agribound.provenance import config_hash

        golden = {
            "da_naip": (
                dict(source="naip", engine="delineate-anything", year=2022),
                "f4c7bd03223ff23954eec4d4748118aee89b072a",
                {"delineate_anything": 2},
            ),
            "da_s2_params": (
                dict(
                    source="sentinel2",
                    engine="delineate-anything",
                    year=2024,
                    date_range=("2024-04-01", "2024-10-31"),
                    engine_params={"max_detections": 1000},
                ),
                "8293ed7d7ea65d9fbc4b851e561eeb68a2acfb9a",
                {"delineate_anything": 2},
            ),
            "da_naip_sam_refine": (
                dict(
                    source="naip",
                    engine="delineate-anything",
                    year=2022,
                    sam_refine=True,
                    sam_backend="sam2.1",
                ),
                "5dc84a5498f6d68fb182a5fc188f95b18b1d1224",
                {"sam_refine": 2, "delineate_anything": 2},
            ),
            "dinov3_ft_spot": (
                dict(
                    source="spot",
                    engine="dinov3",
                    year=2023,
                    fine_tune=True,
                    reference_boundaries="ref.gpkg",
                    cloud_cover_max=15,
                ),
                "0d3ea935571edca3920f78f3877acdece56b94cb",
                {"dinov3_finetune": 2, "dinov3": 2},
            ),
            "ensemble_default": (
                dict(source="sentinel2", engine="ensemble", year=2024),
                "44e58c85b93cedcc7d3411d78ce1b723bdff4346",
                {"delineate_anything": 2},
            ),
            "embedding_tessera": (
                dict(source="tessera-embedding", engine="embedding", year=2024),
                "9251116b343f5202d9ebedac16a824cab774e617",
                {"embedding": 2},
            ),
            "crop_sequence": (
                dict(source="cdl-sequence", engine="crop-sequence", year=2023),
                "84c45459d698c1861f5835067bf5e68e277cf49b",
                {},
            ),
            "ftw_s2": (
                dict(source="sentinel2", engine="ftw", year=2024, lulc_filter=False),
                "3da6d300800b543bf7d80522b5296228b1fc0520",
                {},
            ),
        }
        import warnings

        for name, (kwargs, digest, versions) in golden.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the SPOT access warning
                cfg = AgriboundConfig(
                    study_area=BBOX, gee_project="p", output_path="out.gpkg", **kwargs
                )
            assert config_hash(cfg) == digest, name
            assert results_versions(cfg) == versions, name


# ---------------------------------------------------------------------------
# Model, checkpoint, sam2 distribution, imports
# ---------------------------------------------------------------------------


class TestModelAndCheckpoint:
    def test_checkpoint_files_match_sam2(self):
        build_sam = pytest.importorskip("sam2.build_sam")
        for model_id, filename in sam_auto.CHECKPOINT_FILES.items():
            assert build_sam.HF_MODEL_ID_TO_FILENAMES[model_id][1] == filename
        assert set(build_sam.HF_MODEL_ID_TO_FILENAMES) == set(sam_auto.CHECKPOINT_FILES)

    def test_prefetch_unpinned_and_pinned(self, tmp_path, monkeypatch):
        import hashlib

        import huggingface_hub

        snapshot = tmp_path / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        weights = snapshot / "sam2.1_hiera_tiny.pt"
        weights.write_bytes(b"weights")
        calls = []

        def fake_download(repo_id, filename, revision=None, **kwargs):
            calls.append((repo_id, filename, revision))
            return str(weights)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        cfg = _cfg(engine_params={"sam_model": "tiny"})
        assert SAMAutoEngine.prefetch(cfg) == [str(weights)]
        assert calls[-1] == ("facebook/sam2.1-hiera-tiny", "sam2.1_hiera_tiny.pt", None)
        path, revision, sha256 = sam_auto.checkpoint_path("facebook/sam2.1-hiera-tiny")
        assert revision == "abc123" and sha256 == hashlib.sha256(b"weights").hexdigest()
        # The default model is pinned: its revision is requested and its SHA-256 checked.
        pin = sam_auto.PINNED_CHECKPOINTS["facebook/sam2.1-hiera-large"]
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            SAMAutoEngine.prefetch(_cfg())
        assert calls[-1] == ("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt", pin[0])
        monkeypatch.setitem(
            sam_auto.PINNED_CHECKPOINTS,
            "facebook/sam2.1-hiera-large",
            (pin[0], hashlib.sha256(b"weights").hexdigest(), 7),
        )
        assert sam_auto.checkpoint_path("facebook/sam2.1-hiera-large")[1] == pin[0]

    def test_pinned_default(self):
        revision, sha256, size = sam_auto.PINNED_CHECKPOINTS[sam_auto.DEFAULT_MODEL]
        assert len(revision) == 40 and len(sha256) == 64 and size == 898_083_611

    @pytest.mark.parametrize(
        ("home", "meta", "files"),
        [
            ("https://github.com/JinsuaFeito-dev/segment-anything-2", False, "verified"),
            ("https://github.com/JinsuaFeito-dev/segment-anything-2", False, "other"),
            ("https://github.com/JinsuaFeito-dev/segment-anything-2", False, None),
            ("https://github.com/facebookresearch/sam2", True, None),
        ],
    )
    def test_sam2_distribution_origin(self, monkeypatch, home, meta, files):
        from email.message import Message
        from importlib import metadata

        md = Message()
        md["Name"] = "sam2"
        md["Version"] = "1.1.0"
        md["Home-page"] = home
        md["Author"] = "someone"
        md["License"] = "Apache 2.0"
        monkeypatch.setattr(metadata, "packages_distributions", lambda: {"sam2": ["sam2"]})
        monkeypatch.setattr(metadata, "metadata", lambda name: md)
        digest = {"verified": sam_auto.SAM2_1_1_0_FILES_SHA256, "other": "0" * 64}.get(files)
        found = (digest, 62) if digest else None
        monkeypatch.setattr(sam_auto, "sam2_files_sha256", lambda name="sam2": found)
        dist = sam_auto.sam2_distribution()
        assert dist["version"] == "1.1.0" and dist["home_page"] == home
        assert dist["files_sha256"] == digest and dist["n_files"] == (62 if digest else None)
        if meta:
            assert dist["origin"].startswith("Meta's release")
            assert dist["meta_commit"] is None
        else:
            assert dist["origin"].startswith("a third-party build of Meta's Apache-2.0")
            assert home in dist["origin"]
            if files == "verified":
                assert dist["meta_commit"] == sam_auto.SAM2_META_COMMIT
                assert (
                    "62 code, configuration and licence files equal Meta's commit 2b90b9f"
                    in (dist["origin"])
                )
            else:
                assert dist["meta_commit"] is None
                assert dist["origin"].endswith("its files were not compared with Meta's")

    def test_sam2_files_sha256(self, tmp_path, monkeypatch):
        """Files read from disk, without bytecode and installer metadata, in path order."""
        import hashlib
        from importlib import metadata
        from pathlib import PurePosixPath

        names = [
            "pkg/b.py",
            "pkg/a.py",
            "pkg/__pycache__/a.cpython-312.pyc",
            "pkg-1.0.dist-info/RECORD",
            "pkg-1.0.dist-info/INSTALLER",
            "pkg-1.0.dist-info/licenses/LICENSE",
        ]
        for i, name in enumerate(names):
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text(f"content {i}")

        class Dist:
            files = [PurePosixPath(n) for n in names]

            def locate_file(self, f):
                return tmp_path / f

        monkeypatch.setattr(metadata, "distribution", lambda name: Dist())

        def expected():
            kept = sorted(["pkg/a.py", "pkg/b.py", "pkg-1.0.dist-info/licenses/LICENSE"])
            lines = [
                f"{n}\t{hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()}\n" for n in kept
            ]
            return hashlib.sha256("".join(lines).encode()).hexdigest(), 3

        first = sam_auto.sam2_files_sha256("pkg")
        assert first == expected()
        (tmp_path / "pkg/__pycache__/a.cpython-312.pyc").write_text("other bytecode")
        (tmp_path / "pkg-1.0.dist-info/INSTALLER").write_text("uv")
        assert sam_auto.sam2_files_sha256("pkg") == first  # neither counts
        (tmp_path / "pkg/a.py").write_text("changed")
        assert sam_auto.sam2_files_sha256("pkg") == expected() != first

        def missing(name):
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(metadata, "distribution", missing)
        assert sam_auto.sam2_files_sha256() is None

    def test_installed_sam2_is_the_build_compared_with_meta(self):
        """Where sam2 1.1.0 is installed, its files are those compared with Meta's commit."""
        from importlib import metadata

        pytest.importorskip("sam2")
        if metadata.version("sam2") != sam_auto.SAM2_VERSION_CHECKED:
            pytest.skip("another sam2 version")
        assert sam_auto.sam2_files_sha256() == (sam_auto.SAM2_1_1_0_FILES_SHA256, 62)
        assert sam_auto.sam2_distribution()["meta_commit"] == sam_auto.SAM2_META_COMMIT

    def test_load_generator_needs_opencv_before_any_checkpoint(self, monkeypatch):
        """sam2 does not require OpenCV: without it the engine stops before reading weights."""
        import types

        sam2 = types.ModuleType("sam2")
        build_sam = types.ModuleType("sam2.build_sam")
        build_sam.HF_MODEL_ID_TO_FILENAMES = {}
        build_sam.build_sam2 = None
        predictor = types.ModuleType("sam2.sam2_image_predictor")
        predictor.SAM2ImagePredictor = None
        for name, module in (
            ("sam2", sam2),
            ("sam2.build_sam", build_sam),
            ("sam2.sam2_image_predictor", predictor),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setitem(sys.modules, "cv2", None)  # import cv2 raises ImportError

        def no_download(model_id):
            raise AssertionError("no checkpoint may be read without OpenCV")

        monkeypatch.setattr(sam_auto, "checkpoint_path", no_download)
        with pytest.raises(ImportError, match=r"needs OpenCV \(cv2\).*agribound\[sam-auto\]"):
            sam_auto.load_generator(sam_auto.DEFAULT_MODEL, "cpu")

    def test_sam2_distribution_absent(self, monkeypatch):
        from importlib import metadata

        def missing(name):
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(metadata, "packages_distributions", lambda: {})
        monkeypatch.setattr(metadata, "metadata", missing)
        assert sam_auto.sam2_distribution() == {}

    def test_import_needs_neither_samgeo_nor_sam2(self):
        code = (
            "import sys\n"
            "sys.modules['samgeo'] = None\n"
            "import agribound.engines.sam_auto as m\n"
            "from agribound.config import AgriboundConfig\n"
            "from agribound.engines import get_engine\n"
            "AgriboundConfig(source='naip', engine='sam-auto', year=2022, "
            "study_area='bbox:-116.05,39.55,-115.95,39.62', gee_project='p')\n"
            "assert type(get_engine('sam-auto')).__name__ == 'SAMAutoEngine'\n"
            "loaded = ('samgeo', 'sam2', 'torch', 'cv2')\n"
            "print(sorted(k for k in loaded if sys.modules.get(k) is not None))\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "[]"

    def test_extra(self):
        """Its own extra: sam2 pinned, no segment-geospatial (GPL-3 patool), in all and all-gfm."""
        import tomllib
        from pathlib import Path

        extras = tomllib.loads(
            (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
        )["project"]["optional-dependencies"]
        names = {r.split(">")[0].split("=")[0].split("[")[0] for r in extras["sam-auto"]}
        assert "sam2==1.1.0" in extras["sam-auto"]
        assert names == {"sam2", "huggingface-hub", "opencv-python"}
        assert not any("segment-geospatial" in r or "pycocotools" in r for r in extras["sam-auto"])
        assert ENGINE_REGISTRY["sam-auto"]["install_extra"] in extras
        for bundle in ("all", "all-gfm"):
            assert "sam-auto" in extras[bundle][0]

    def test_class_attributes_match_registry(self):
        info = ENGINE_REGISTRY["sam-auto"]
        assert SAMAutoEngine.supported_sources == info["supported_sources"]
        assert SAMAutoEngine.requires_bands == info["requires_bands"]

    def test_registry_entry(self):
        info = ENGINE_REGISTRY["sam-auto"]
        assert info["label_free"] is True and info["fine_tunable"] is False
        assert info["requires_bands"] == ["R", "G", "B"] and info["gpu_recommended"] is True
        assert info["install_extra"] == "sam-auto"
        assert "2304.02643" in info["reference"] and "2408.00714" in info["reference"]
        assert "Apache-2.0" in info["notes"] and "third-party build" in info["notes"]
        notes = info["source_notes"]
        # The measured costs and the pre-registered results (PREREG-SAM 4 and 5).
        assert "17.0 times" in notes["sentinel2"] and "20.7 times" in notes["landsat"]
        assert "6.1 times" in notes["naip"] and "6.4 times" in notes["spot"]
        assert notes["sentinel2"].startswith("Did not transfer")
        assert "did not transfer" in info["notes"] and "measured back to back" in info["notes"]
        assert "approximate" not in info["notes"]
        assert "0.017 higher" in notes["naip"] and "0.004 to 0.030" in notes["naip"]
        assert "no 'better' claim" in notes["spot"] and "no 'better' claim" in notes["landsat"]
        assert (
            "No default" in notes["sentinel1"]
            and "no setting was piloted" in notes["usgs-naip-plus"]
        )
        assert "never an engine default" in info["notes"] and "example 28" in info["notes"]
        assert "includes 0" in notes["sentinel2"] and "+0.03 to +0.12" in notes["naip"]
        # The SPOT/Landsat endpoint as pre-registered: filter states, the amendment, the
        # San Juan box's Landsat 7/8/9 composite; the Sentinel-1 setting the rule chose.
        assert "7/8/9 on the San Juan box" in info["notes"] and "amendment" in info["notes"]
        assert "without the crop filter" in notes["spot"] and "+0.21" in notes["spot"]
        assert "7/8/9 on the San Juan box" in notes["landsat"] and "+0.09" in notes["landsat"]
        assert "selection rule chose" in notes["sentinel1"] and "0.443" in notes["sentinel1"]
        assert "ICLR 2025" in info["reference"]
        engines = list(ENGINE_REGISTRY)
        assert engines.index("sam-auto") == engines.index("crop-sequence") - 1


# ---------------------------------------------------------------------------
# Resampling (OpenCV)
# ---------------------------------------------------------------------------


class TestResampling:
    def test_tiles_smaller_than_the_input_are_enlarged(self, tmp_path):
        pytest.importorskip("cv2")
        raster = _objects(tmp_path / "o.tif", (300, 300), {})  # 10 m: one 512 px tile, padded
        mask = np.zeros((1, 1024, 1024), dtype=bool)
        mask[0, 100:300, 200:400] = True  # SAM input pixels: 2x zoom
        gen = FixedGenerator(mask)
        gdf = SAMAutoEngine(generator=gen).delineate(
            raster, _config(tmp_path, raster, sam_tile_m=5120.0)
        )
        assert gen.images == [(1024, 1024, 3)]
        assert gdf.geometry.iloc[0].bounds == pytest.approx(_px_box(50, 150, 100, 200))
        assert gdf.attrs["engine_meta"]["tiles"]["sam_pixel_m"] == 5.0

    def test_tiles_larger_than_the_input_are_reduced(self, tmp_path):
        pytest.importorskip("cv2")
        raster = _objects(tmp_path / "o.tif", (300, 300), {}, res=1.0)  # one 2,048 px tile
        mask = np.zeros((1, 1024, 1024), dtype=bool)
        mask[0, 50:100, 50:100] = True  # 0.5x zoom: native rows and columns 100-199
        gdf = SAMAutoEngine(generator=FixedGenerator(mask)).delineate(
            raster, _config(tmp_path, raster, sam_tile_m=2048.0)
        )
        assert gdf.geometry.iloc[0].bounds == pytest.approx(_px_box(100, 200, 100, 200, res=1.0))
