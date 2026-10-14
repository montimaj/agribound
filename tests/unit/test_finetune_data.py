"""Tests for fine-tuning data preparation: splits, masks, radiometry and chips."""

from __future__ import annotations

import json
import logging

import geopandas as gpd
import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import binary_erosion
from scipy.ndimage import label as nd_label
from shapely.geometry import Polygon, box

from agribound.config import AgriboundConfig
from agribound.engines.base import get_canonical_band_indices
from agribound.engines.finetune import _data
from agribound.engines.finetune._data import (
    IGNORE_INDEX,
    apply_stretch,
    assign_splits,
    interior_polygons,
    mask_invalid_predictions,
    prithvi_band_names,
    prithvi_reflectance,
    read_chip_meta,
    read_training_meta,
    scene_stretch_bounds,
    semantic_from_instances,
    split_files,
    training_output_dir,
    valid_pixels,
    write_rgb_input,
    write_training_meta,
)
from agribound.io.raster import percentile_stretch_uint8

UTM = "EPSG:32755"  # WGS 84 / UTM 55S (Namoi)
X0, Y0 = 700_000.0, 6_600_000.0


def _config(tmp_path, **overrides):
    values = {
        "source": "local",
        "local_tif_path": str(tmp_path / "composite.tif"),
        "engine": "geoai",
        "output_path": str(tmp_path / "out" / "fields.gpkg"),
        "lulc_filter": False,
        "seed": 11,
    }
    values.update(overrides)
    return AgriboundConfig(**values)


def _grid_units(n_x, n_y, size_m, x0=X0, y0=Y0, crs=UTM):
    geoms = [
        box(x0 + i * size_m, y0 + j * size_m, x0 + (i + 1) * size_m, y0 + (j + 1) * size_m)
        for j in range(n_y)
        for i in range(n_x)
    ]
    return gpd.GeoDataFrame(geometry=geoms, crs=crs)


# ---------------------------------------------------------------------------
# assign_splits
# ---------------------------------------------------------------------------


class TestAssignSplitsRandom:
    def test_counts_and_values(self, tmp_path):
        cfg = _config(tmp_path, fine_tune_split="random", fine_tune_val_split=0.25)
        units = _grid_units(10, 4, 1000)
        splits = assign_splits(units, cfg)
        assert len(splits) == 40
        assert set(splits) == {"train", "val"}
        assert (splits == "val").sum() == 10

    def test_deterministic_and_seed_dependent(self, tmp_path):
        units = _grid_units(10, 10, 1000)
        a = assign_splits(units, _config(tmp_path, fine_tune_split="random", seed=1))
        b = assign_splits(units, _config(tmp_path, fine_tune_split="random", seed=1))
        c = assign_splits(units, _config(tmp_path, fine_tune_split="random", seed=2))
        assert (a == b).all()
        assert not (a == c).all()

    def test_two_units_get_one_of_each(self, tmp_path):
        cfg = _config(tmp_path, fine_tune_split="random", fine_tune_val_split=0.05)
        splits = assign_splits(_grid_units(2, 1, 1000), cfg)
        assert sorted(splits) == ["train", "val"]

    def test_fewer_than_two_units_raises(self, tmp_path):
        with pytest.raises(ValueError, match="At least 2 units"):
            assign_splits(_grid_units(1, 1, 1000), _config(tmp_path))

    def test_missing_crs_raises(self, tmp_path):
        units = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)])
        with pytest.raises(ValueError, match="CRS"):
            assign_splits(units, _config(tmp_path))


class TestAssignSplitsBlock:
    @staticmethod
    def _independent_blocks(units, size):
        """Blocks as documented: grid from the SW corner in a LAEA centred on the units."""
        centre = units.to_crs("EPSG:4326").geometry.union_all().centroid
        laea = pyproj.CRS.from_proj4(
            f"+proj=laea +lat_0={round(centre.y, 3)} +lon_0={round(centre.x, 3)} "
            "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        projected = units.to_crs(laea).geometry
        minx, miny, _, _ = projected.total_bounds
        c = projected.centroid
        return list(
            zip(
                np.floor((c.x - minx) / size).astype(int),
                np.floor((c.y - miny) / size).astype(int),
                strict=True,
            )
        )

    def test_units_of_one_block_share_a_split(self, tmp_path):
        cfg = _config(tmp_path, fine_tune_block_size_m=5000, fine_tune_val_split=0.2)
        units = _grid_units(24, 24, 1000)  # 1 km chips over 24 x 24 km
        info = {}
        splits = assign_splits(units, cfg, info=info)
        blocks = self._independent_blocks(units, 5000)
        per_block = {}
        for blk, split in zip(blocks, splits, strict=True):
            per_block.setdefault(blk, set()).add(split)
        assert all(len(v) == 1 for v in per_block.values())
        assert {"train", "val"} <= {next(iter(v)) for v in per_block.values()}
        assert info["n_groups"] == len(per_block)
        assert info["block_size_m"] == 5000
        # Whole blocks approximate the requested fraction of units.
        assert abs((splits == "val").mean() - 0.2) < 0.06

    def test_single_block_is_subdivided_with_warning(self, tmp_path, caplog):
        cfg = _config(tmp_path, fine_tune_block_size_m=100_000)
        units = _grid_units(3, 3, 500)  # 1.5 km x 1.5 km, inside one 100 km block
        info = {}
        with caplog.at_level(logging.WARNING, logger="agribound.engines.finetune._data"):
            splits = assign_splits(units, cfg, info=info)
        assert set(splits) == {"train", "val"}
        assert info["block_size_m"] < info["requested_block_size_m"] == 100_000
        assert "one 100000 m block" in caplog.text

    def test_val_fraction_selects_whole_groups_close_to_target(self):
        groups = np.repeat(np.arange(10), 5)  # 10 groups of 5
        rng = np.random.default_rng(0)
        is_val = _data._groups_to_val(groups, 0.2, rng)
        assert is_val.sum() == 10
        # each group entirely in or out
        for g in range(10):
            assert len(set(is_val[groups == g])) == 1

    def test_all_groups_selected_moves_one_back(self):
        groups = np.array([0, 0, 1])
        is_val = _data._groups_to_val(groups, 0.99, np.random.default_rng(0))
        assert 0 < is_val.sum() < 3


class TestAssignSplitsColumn:
    def _reference(self):
        # Four reference polygons; column "region" = a, a, b, c
        polys = [
            box(X0, Y0, X0 + 900, Y0 + 900),
            box(X0 + 1000, Y0, X0 + 1900, Y0 + 900),
            box(X0 + 2000, Y0, X0 + 2900, Y0 + 900),
            box(X0 + 3000, Y0, X0 + 3900, Y0 + 900),
        ]
        return gpd.GeoDataFrame({"region": ["a", "a", "b", "c"]}, geometry=polys, crs=UTM)

    def test_groups_follow_majority_column_value(self, tmp_path):
        cfg = _config(tmp_path, fine_tune_split="column", fine_tune_split_column="region")
        units = _grid_units(4, 1, 1000)
        splits = assign_splits(units, cfg, reference=self._reference())
        assert splits[0] == splits[1]  # both region "a"
        assert set(splits) == {"train", "val"}

    def test_tie_broken_by_intersection_area(self, tmp_path):
        cfg = _config(tmp_path, fine_tune_split="column", fine_tune_split_column="region")
        ref = gpd.GeoDataFrame(
            {"region": ["a", "b", "b"]},
            geometry=[
                box(X0, Y0, X0 + 800, Y0 + 1000),  # large "a" part of unit 0
                box(X0 + 800, Y0, X0 + 1000, Y0 + 1000),  # small "b" part of unit 0
                box(X0 + 1000, Y0, X0 + 2000, Y0 + 1000),  # unit 1
            ],
            crs=UTM,
        )
        units = _grid_units(2, 1, 1000)
        groups = _data._column_groups(units, cfg, ref)
        # unit 0: one "a" and one "b" polygon -> tie -> larger area "a"; unit 1: "b"
        assert groups[0] != groups[1]

    def test_single_group_raises(self, tmp_path):
        cfg = _config(tmp_path, fine_tune_split="column", fine_tune_split_column="region")
        ref = self._reference().assign(region="same")
        with pytest.raises(ValueError, match="at least 2 groups"):
            assign_splits(_grid_units(4, 1, 1000), cfg, reference=ref)

    def test_missing_column_or_reference_raises(self, tmp_path):
        cfg = _config(tmp_path, fine_tune_split="column", fine_tune_split_column="nope")
        with pytest.raises(ValueError, match="not a column"):
            assign_splits(_grid_units(4, 1, 1000), cfg, reference=self._reference())
        with pytest.raises(ValueError, match="requires the reference"):
            assign_splits(_grid_units(4, 1, 1000), cfg)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


class TestSemanticFromInstances:
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_isolated_polygon_matches_binary_erosion(self, k):
        inst = np.zeros((40, 40), dtype=np.int32)
        inst[10:30, 8:33] = 5
        sem = semantic_from_instances(inst, k)
        eroded = binary_erosion(inst > 0, iterations=k)
        expected = np.zeros_like(sem)
        expected[inst > 0] = 2
        expected[eroded] = 1
        np.testing.assert_array_equal(sem, expected)

    def test_touching_polygons_are_separated(self):
        inst = np.zeros((30, 40), dtype=np.int32)
        inst[5:25, 5:20] = 1
        inst[5:25, 20:35] = 2  # shares the edge x=20 with polygon 1
        sem = semantic_from_instances(inst, 2)
        # boundary on both sides of the shared edge
        assert (sem[10, 18:22] == 2).all()
        _, n_interiors = nd_label(sem == 1)
        assert n_interiors == 2
        # eroding the union (the 0.1.x behaviour) would merge them
        _, n_union = nd_label(binary_erosion(inst > 0, iterations=2))
        assert n_union == 1

    def test_zero_erosion_has_no_boundary(self):
        inst = np.zeros((10, 10), dtype=np.int32)
        inst[2:8, 2:8] = 1
        assert set(np.unique(semantic_from_instances(inst, 0))) == {0, 1}

    def test_array_edge_is_not_a_boundary(self):
        inst = np.ones((10, 10), dtype=np.int32)
        assert (semantic_from_instances(inst, 2) == 1).all()


class TestBoundaryBandRule:
    """field_interior_depths / widest_boundary_erosion: the masks' rule on a raster's grid."""

    @staticmethod
    def _grid(tmp_path, pixel=10.0, size=60):
        path = tmp_path / "grid.tif"
        profile = {"driver": "GTiff", "height": size, "width": size, "count": 1}
        profile.update(dtype="uint8", crs=UTM, transform=from_origin(X0, Y0, pixel, pixel))
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.zeros((1, size, size), dtype=np.uint8))
        return str(path)

    @staticmethod
    def _px(c0, r0, c1, r1, pixel=10.0):
        return box(X0 + c0 * pixel, Y0 - r1 * pixel, X0 + c1 * pixel, Y0 - r0 * pixel)

    def test_depths_match_the_training_masks(self, tmp_path):
        raster = self._grid(tmp_path)
        fields = [
            self._px(2, 2, 11, 30),  # 9 px wide: depth 5
            self._px(14, 2, 17, 30),  # 3 px: depth 2
            self._px(20, 2, 21, 30),  # 1 px: depth 1
            self._px(24, 2, 34.4, 30),  # covers 10 pixel centres across: depth 5
            self._px(40, 2, 50, 12),  # touches the next field: measured on its own
            self._px(50, 2, 52, 12),
            self._px(0, 40, 4, 60),  # 4 px wide at the raster's edge, which is no boundary
            # a diamond 8 px from its centre to each corner: city-block depth 8 (chessboard 4)
            Polygon(
                [
                    (X0 + c * 10.0, Y0 - r * 10.0)
                    for c, r in ((45, 33), (53, 41), (45, 49), (37, 41))
                ]
            ),
            self._px(80, 80, 90, 90),  # outside the raster: left out
        ]
        ref = gpd.GeoDataFrame({"k": range(len(fields))}, geometry=fields, crs=UTM)
        depths = _data.field_interior_depths(ref, raster)
        assert depths.tolist() == [5, 2, 1, 5, 5, 1, 4, 8]
        # The masks of _prepare_training_data: a field keeps an interior pixel at k px
        # exactly when its depth exceeds k.
        from rasterio.features import rasterize

        inst = rasterize(
            [(g, i + 1) for i, g in enumerate(fields[:8])],
            out_shape=(60, 60),
            transform=from_origin(X0, Y0, 10.0, 10.0),
            dtype="int32",
        )
        for k in range(1, 9):
            sem = semantic_from_instances(inst, k)
            with_interior = set(np.unique(inst[sem == 1]).tolist())
            assert with_interior == {i + 1 for i, d in enumerate(depths) if d > k}

    def test_widest_band_with_at_most_10_percent_of_the_fields_without_interior(self, tmp_path):
        raster = self._grid(tmp_path, size=100)
        wide = [self._px(2 + 10 * i, 2, 11 + 10 * i, 40) for i in range(9)]  # depth 5
        thin = [self._px(2, 50, 5, 90), self._px(10, 50, 13, 90)]  # depth 2
        ref = gpd.GeoDataFrame(geometry=[*wide, thin[0]], crs=UTM)
        band, shares = _data.widest_boundary_erosion(ref, raster)
        assert band == 4 and shares == {1: 0.0, 2: 0.1, 3: 0.1, 4: 0.1}  # 10 %: allowed
        ref = gpd.GeoDataFrame(geometry=[*wide, *thin], crs=UTM)
        band, shares = _data.widest_boundary_erosion(ref, raster)
        assert band == 1 and shares[2] == pytest.approx(2 / 11)
        assert _data.widest_boundary_erosion(ref, raster, max_share=0.2)[0] == 4
        assert _data.widest_boundary_erosion(ref, raster, max_px=2)[1].keys() == {1, 2}
        # No band qualifies: the narrowest, 1 px.
        ones = gpd.GeoDataFrame(geometry=[self._px(2, 2, 3, 40)], crs=UTM)
        assert _data.widest_boundary_erosion(ones, raster) == (1, {k: 1.0 for k in (1, 2, 3, 4)})
        with pytest.raises(ValueError, match="max_px"):
            _data.widest_boundary_erosion(ref, raster, max_px=0)

    def test_reference_in_another_crs_and_coarser_pixels(self, tmp_path):
        raster = self._grid(tmp_path, pixel=30.0)
        field = self._px(300, 300, 420, 420, pixel=1.0)  # 120 m: 4 px at 30 m, depth 2
        path = tmp_path / "ref.gpkg"
        gpd.GeoDataFrame(geometry=[field], crs=UTM).to_crs(4326).to_file(path)
        from agribound.engines.finetune import widest_boundary_erosion

        assert _data.field_interior_depths(str(path), raster).tolist() == [2]
        assert widest_boundary_erosion(str(path), raster) == (1, {1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0})

    def test_a_field_covering_the_whole_raster(self, tmp_path):
        """No pixel of the raster lies outside the field, and the raster's edge is no
        boundary: its depth is the raster's longer side, deeper than any band."""
        path = tmp_path / "wide.tif"
        profile = {"driver": "GTiff", "height": 30, "width": 50, "count": 1, "dtype": "uint8"}
        profile.update(crs=UTM, transform=from_origin(X0, Y0, 10.0, 10.0))
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.zeros((1, 30, 50), dtype=np.uint8))
        fields = [
            self._px(-5, -5, 55, 35),  # beyond every edge of the raster
            self._px(0, 0, 50, 30),  # exactly the raster's extent
            self._px(2, 2, 5, 28),  # 3 px wide: depth 2, measured as usual
        ]
        ref = gpd.GeoDataFrame(geometry=fields, crs=UTM)
        assert _data.field_interior_depths(ref, str(path)).tolist() == [50, 50, 2]
        band, shares = _data.widest_boundary_erosion(ref.iloc[:2], str(path))
        assert band == 4 and shares == {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        # As in the masks: one instance over the whole chip keeps every pixel interior.
        assert (semantic_from_instances(np.ones((30, 50), dtype=np.int32), 4) == 1).all()


# ---------------------------------------------------------------------------
# Radiometry and bands
# ---------------------------------------------------------------------------


class TestStretch:
    def test_apply_stretch_matches_percentile_stretch(self):
        rng = np.random.default_rng(0)
        arr = rng.uniform(100, 5000, (3, 64, 80)).astype(np.float32)
        arr[:, :3, :3] = np.nan
        expected, lows, highs = percentile_stretch_uint8(arr, return_bounds=True)
        np.testing.assert_array_equal(apply_stretch(arr, lows, highs), expected)

    def test_scene_bounds_equal_full_array_bounds(self, tmp_path):
        rng = np.random.default_rng(1)
        arr = rng.uniform(100, 5000, (3, 50, 60)).astype(np.float32)
        path = tmp_path / "r.tif"
        _write(path, arr)
        lows, highs = scene_stretch_bounds(path, [1, 2, 3])
        _, lo_ref, hi_ref = percentile_stretch_uint8(arr, return_bounds=True)
        np.testing.assert_allclose(lows, lo_ref)
        np.testing.assert_allclose(highs, hi_ref)

    def test_scene_bounds_never_sample_overviews(self, tmp_path):
        """Same rule as Delineate-Anything's scene_stretch_bounds (OVERVIEW_LEVEL=NONE)."""
        from rasterio.enums import Resampling

        arr = np.ones((3, 256, 256), dtype=np.float32)
        arr[:, ::2, ::2] = 1000.0
        arr[:, 1::2, 1::2] = 1000.0
        path = tmp_path / "ovr.tif"
        _write(path, arr)
        with rasterio.open(path, "r+") as dst:
            dst.build_overviews([2, 4], Resampling.average)
        lows, highs = scene_stretch_bounds(path, [1, 2, 3], max_sample_side=64)
        # An overview-served decimated read would give 500.5 everywhere.
        assert set(lows) | set(highs) <= {1.0, 1000.0}

    def test_uint8_is_not_stretched(self, tmp_path):
        arr = np.arange(3 * 20 * 20, dtype=np.uint8).reshape(3, 20, 20)
        path = tmp_path / "u8.tif"
        _write(path, arr)
        assert scene_stretch_bounds(path, [1, 2, 3]) == ([0.0] * 3, [255.0] * 3)
        np.testing.assert_array_equal(apply_stretch(arr, [0] * 3, [255] * 3), arr)

    def test_write_rgb_input_is_scene_level(self, tmp_path):
        # Left half dark, right half bright: a scene-level stretch keeps them apart.
        arr = np.full((3, 64, 64), 500.0, dtype=np.float32)
        arr[:, :, 32:] = 3000.0
        arr += np.random.default_rng(2).uniform(0, 50, arr.shape).astype(np.float32)
        src = tmp_path / "scene.tif"
        _write(src, arr)
        out = tmp_path / "rgb.tif"
        info = write_rgb_input(src, out, [1, 2, 3], block_rows=16)
        with rasterio.open(out) as ds:
            rgb = ds.read()
            assert ds.dtypes[0] == "uint8" and ds.count == 3 and ds.nodata is None
        np.testing.assert_array_equal(rgb, apply_stretch(arr, info["lows"], info["highs"]))
        assert rgb[:, :, :32].mean() < 40 and rgb[:, :, 32:].mean() > 200
        out_f = tmp_path / "rgb_unit.tif"
        write_rgb_input(src, out_f, [1, 2, 3], unit_float=True)
        with rasterio.open(out_f) as ds:
            unit = ds.read()
        assert unit.dtype == np.float32 and unit.max() <= 1.0
        np.testing.assert_allclose(unit, rgb.astype(np.float32) / 255.0)


class TestPrithviBands:
    @pytest.mark.parametrize(
        ("source", "names", "indices"),
        [
            ("sentinel2", ["B", "G", "R", "NIR_NARROW", "SWIR1", "SWIR2"], [2, 3, 4, 9, 11, 12]),
            ("hls", ["B", "G", "R", "NIR_NARROW", "SWIR1", "SWIR2"], [2, 3, 4, 5, 6, 7]),
            ("landsat", ["B", "G", "R", "NIR", "SWIR1", "SWIR2"], [1, 2, 3, 4, 5, 6]),
            ("local", ["B", "G", "R", "NIR_NARROW", "SWIR1", "SWIR2"], [1, 2, 3, 4, 5, 6]),
        ],
    )
    def test_band_order_per_source(self, source, names, indices):
        assert prithvi_band_names(source) == names
        assert get_canonical_band_indices(source, names) == indices

    def test_local_mapping_with_only_nir(self):
        bands = {"B": 1, "G": 2, "R": 3, "NIR": 7, "SWIR1": 8, "SWIR2": 9}
        names = prithvi_band_names("local", bands)
        assert names[3] == "NIR"
        assert get_canonical_band_indices("local", names, bands=bands) == [1, 2, 3, 7, 8, 9]

    def test_reflectance_x10000_is_unchanged(self):
        arr = np.array([[[1234.0, 4321.5]]], dtype=np.float32)
        out = prithvi_reflectance(arr, "sentinel2")
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, arr)  # exact: no divide/multiply round trip

    def test_unit_scale_is_multiplied(self):
        out = prithvi_reflectance(np.array([[[0.1234]]]), "local", value_scale="unit")
        np.testing.assert_allclose(out, [[[1234.0]]], rtol=1e-6)

    @pytest.mark.parametrize(("source", "scale"), [("naip", None), ("spot", None), ("local", None)])
    def test_non_reflectance_raises(self, source, scale):
        with pytest.raises(ValueError, match="surface reflectance"):
            prithvi_reflectance(np.ones((1, 2, 2)), source, scale)


def test_valid_pixels_rules():
    arr = np.ones((3, 2, 3), dtype=np.float32)
    arr[1, 0, 0] = np.nan  # one band non-finite -> invalid
    arr[:, 1, 1] = -9999  # all bands == nodata -> invalid
    arr[0, 1, 2] = -9999  # only one band == nodata -> valid
    valid = valid_pixels(arr, nodata=-9999)
    assert valid.tolist() == [[False, True, True], [True, False, True]]
    arr[:, 0, 1] = 0.0  # all bands 0: valid, except under the Delineate-Anything rule
    arr[0, 0, 2] = 0.0  # one band 0: valid
    assert valid_pixels(arr, nodata=-9999).tolist() == [[False, True, True], [True, False, True]]
    zero = valid_pixels(arr, nodata=-9999, zero_invalid=True)
    assert zero.tolist() == [[False, False, True], [True, False, True]]


@pytest.mark.parametrize("dtype", ["float32", "uint8", "int16"])
@pytest.mark.parametrize("nodata", [None, 0.0, 7.0, float("nan")])
def test_zero_invalid_rule_is_the_native_backend_rule(dtype, nodata):
    """Delineate-Anything chips and its native backend's tiles agree on every pixel."""
    from agribound.engines.delineate_anything import _valid_pixels

    rng = np.random.default_rng(4)
    arr = rng.integers(0, 3, (3, 40, 40)).astype(dtype)  # many pixels 0 in some or all bands
    arr[:, 5:9, 5:9] = 7
    if np.issubdtype(arr.dtype, np.floating):
        arr[1, 20:23, 20:23] = np.nan
    expected = _valid_pixels(arr.astype(np.float32), nodata)
    np.testing.assert_array_equal(valid_pixels(arr, nodata, zero_invalid=True), expected)
    assert not expected.all() and expected.any()


# ---------------------------------------------------------------------------
# Chip extraction
# ---------------------------------------------------------------------------


def _write(path, data, crs=UTM, res=10.0, nodata=None, x0=X0, y0=Y0):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=str(data.dtype),
        crs=crs,
        transform=from_origin(x0, y0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return str(path)


@pytest.fixture
def s2_scene(tmp_path):
    """12-band S2-like x10000 raster (130 x 130 px at 10 m) and 3 reference fields.

    Fields 1 and 2 touch; field 3 is isolated. The top-left 10 x 10 px are NaN.
    """
    rng = np.random.default_rng(5)
    data = rng.uniform(300, 600, (12, 130, 130)).astype(np.float32)
    data[:, 60:100, 10:120] += 2000.0  # bright fields
    data[:, :10, :10] = np.nan
    raster = _write(tmp_path / "s2.tif", data, nodata=float("nan"))
    fields = [
        box(X0 + 100, Y0 - 1000, X0 + 600, Y0 - 600),
        box(X0 + 600, Y0 - 1000, X0 + 1100, Y0 - 600),
        box(X0 + 200, Y0 - 400, X0 + 500, Y0 - 150),
    ]
    ref = gpd.GeoDataFrame({"name": ["a", "b", "c"]}, geometry=fields, crs=UTM)
    ref_path = tmp_path / "ref.gpkg"
    ref.to_file(ref_path)
    return raster, str(ref_path), data


def _chip_config(tmp_path, raster, ref, engine, **params):
    return AgriboundConfig(
        source="sentinel2",
        year=2023,
        study_area=f"bbox:{150.0},{-30.8},{150.1},{-30.7}",
        gee_project="test-project",
        engine=engine,
        output_path=str(tmp_path / "out" / "fields.gpkg"),
        lulc_filter=False,
        reference_boundaries=ref,
        fine_tune_split="random",
        seed=3,
        engine_params={"chip_size": 64, "min_label_fraction": 0.0, **params},
    )


class TestPrepareTrainingData:
    def test_rgb_chips_masks_instances_and_layout(self, tmp_path, s2_scene):
        raster, ref, data = s2_scene
        cfg = _chip_config(tmp_path, raster, ref, "geoai")
        td = _data._prepare_training_data(raster, cfg, "geoai")
        meta = read_chip_meta(td)
        # 130 px -> two full 64 px chips per axis; partial edge chips are skipped
        assert meta["n_chips"] == 4
        assert meta["band_names"] == ["R", "G", "B"] and meta["band_indices"] == [4, 3, 2]
        n_train = len(split_files(td, "train", "images"))
        n_val = len(split_files(td, "val", "images"))
        assert n_train + n_val == 4 and n_train >= 1 and n_val >= 1
        chips = gpd.read_file(td / "chips.gpkg", layer="chips")
        assert len(chips) == 4 and set(chips["split"]) == {"train", "val"}

        # Recompose the full-raster instance mask from the chips.
        inst_full = np.zeros((128, 128), dtype=np.int32)
        sem_full = np.zeros((128, 128), dtype=np.uint8)
        img_full = np.zeros((3, 128, 128), dtype=np.uint8)
        for split in ("train", "val"):
            for kind in ("images", "masks", "instances"):
                assert [p.name for p in split_files(td, split, kind)] == [
                    p.name for p in split_files(td, split, "images")
                ]
            for img_p, mask_p, inst_p in zip(
                split_files(td, split, "images"),
                split_files(td, split, "masks"),
                split_files(td, split, "instances"),
                strict=True,
            ):
                row = chips.loc[chips["chip_id"] == int(img_p.stem.split("_")[1])].iloc[0]
                r, c = int(row["row_off"]), int(row["col_off"])
                with rasterio.open(img_p) as ds:
                    assert ds.dtypes[0] == "uint8" and ds.count == 3
                    img_full[:, r : r + 64, c : c + 64] = ds.read()
                with rasterio.open(mask_p) as ds:
                    sem_full[r : r + 64, c : c + 64] = ds.read(1)
                with rasterio.open(inst_p) as ds:
                    assert ds.dtypes[0] == "int32"
                    inst_full[r : r + 64, c : c + 64] = ds.read(1)
        # three reference polygons -> exactly three instance ids
        assert sorted(set(np.unique(inst_full)) - {0}) == [1, 2, 3]
        # the touching fields 1 and 2 are separated by boundary pixels
        assert (sem_full[80, 58:62] == 2).all()
        # NaN pixels are ignored in the semantic mask
        assert (sem_full[:10, :10] == IGNORE_INDEX).all()
        # scene-level stretch of canonical R, G, B (bands 4, 3, 2)
        lows, highs = meta["image"]["stretch_lows"], meta["image"]["stretch_highs"]
        expected = apply_stretch(data[[3, 2, 1], :128, :128], lows, highs)
        np.testing.assert_array_equal(img_full, expected)

    def test_prithvi_chips_keep_reflectance_and_fill_invalid(self, tmp_path, s2_scene):
        from agribound.engines.prithvi import PRITHVI_MEAN

        raster, ref, data = s2_scene
        cfg = _chip_config(tmp_path, raster, ref, "prithvi")
        td = _data._prepare_training_data(raster, cfg, "prithvi")
        meta = read_chip_meta(td)
        assert meta["band_indices"] == [2, 3, 4, 9, 11, 12]
        chips = gpd.read_file(td / "chips.gpkg", layer="chips")
        top_left = chips.loc[(chips["row_off"] == 0) & (chips["col_off"] == 0)].iloc[0]
        split = top_left["split"]
        name = f"chip_{int(top_left['chip_id']):05d}.tif"
        prefix = "" if split == "train" else "val_"
        with rasterio.open(td / f"{prefix}images" / name) as ds:
            img = ds.read()
        with rasterio.open(td / f"{prefix}masks" / name) as ds:
            mask = ds.read(1)
        assert img.shape == (6, 64, 64) and img.dtype == np.float32
        np.testing.assert_array_equal(img[:, 20:, 20:], data[[1, 2, 3, 8, 10, 11], 20:64, 20:64])
        np.testing.assert_array_equal(img[:, 0, 0], np.asarray(PRITHVI_MEAN, dtype=np.float32))
        assert (mask[:10, :10] == IGNORE_INDEX).all()

    def test_dinov3_chips_are_unit_float(self, tmp_path, s2_scene):
        raster, ref, _ = s2_scene
        cfg = _chip_config(tmp_path, raster, ref, "dinov3")
        td = _data._prepare_training_data(raster, cfg, "dinov3")
        img_path = (split_files(td, "train", "images") + split_files(td, "val", "images"))[0]
        with rasterio.open(img_path) as ds:
            img = ds.read()
        assert img.dtype == np.float32 and img.min() >= 0.0 and img.max() <= 1.0
        np.testing.assert_allclose(img * 255.0, np.round(img * 255.0), atol=1e-3)

    def test_default_boundary_width_per_engine(self, tmp_path, s2_scene):
        """DINOv3 masks have a 4 px boundary band by default, the other engines 2 px."""
        raster, ref, _ = s2_scene
        assert _data.default_boundary_erosion("dinov3") == 4
        assert _data.default_boundary_erosion("prithvi") == 2
        assert _data.default_boundary_erosion("geoai") == 2

        def full_mask(engine, **params):
            cfg = _chip_config(tmp_path, raster, ref, engine, **params)
            td = _data._prepare_training_data(raster, cfg, engine)
            chips = gpd.read_file(td / "chips.gpkg", layer="chips")
            sem = np.zeros((128, 128), dtype=np.uint8)
            for split in ("train", "val"):
                for path in split_files(td, split, "masks"):
                    row = chips.loc[chips["chip_id"] == int(path.stem.split("_")[1])].iloc[0]
                    r, c = int(row["row_off"]), int(row["col_off"])
                    with rasterio.open(path) as ds:
                        sem[r : r + 64, c : c + 64] = ds.read(1)
            return read_chip_meta(td)["boundary_erosion"], sem

        width, sem = full_mask("dinov3")
        # fields 1 and 2 touch at column 60: 4 px of boundary on each side
        assert width == 4 and (sem[80, 56:64] == 2).all() and (sem[80, [55, 64]] == 1).all()
        width, sem = full_mask("prithvi")
        assert width == 2 and (sem[80, 58:62] == 2).all() and (sem[80, [57, 62]] == 1).all()
        width, _ = full_mask("dinov3", boundary_erosion=2)
        assert width == 2

    def test_fields_without_interior_are_counted_and_warned(self, tmp_path, s2_scene, caplog):
        """Mask engines warn when the boundary band leaves many fields without interior."""
        raster, ref, _ = s2_scene
        with caplog.at_level(logging.WARNING, logger="agribound.engines.finetune._data"):
            td = _data._prepare_training_data(
                raster, _chip_config(tmp_path, raster, ref, "dinov3"), "dinov3"
            )
        meta = read_chip_meta(td)
        assert meta["n_reference_fields"] == 3
        assert meta["n_reference_fields_without_interior"] == 0
        assert "no field-interior pixel" not in caplog.text
        # field 3 is 25 px tall: a 13 px band (27 px) leaves it no interior pixel
        cfg = _chip_config(tmp_path, raster, ref, "dinov3", boundary_erosion=13)
        with caplog.at_level(logging.WARNING, logger="agribound.engines.finetune._data"):
            td = _data._prepare_training_data(raster, cfg, "dinov3")
        assert read_chip_meta(td)["n_reference_fields_without_interior"] == 1
        assert "1 of the 3 reference fields" in caplog.text and "13 px" in caplog.text
        # Delineate-Anything and GeoAI train on the instances: no count, no warning
        caplog.clear()
        cfg = _chip_config(tmp_path, raster, ref, "geoai", boundary_erosion=13)
        with caplog.at_level(logging.WARNING, logger="agribound.engines.finetune._data"):
            td = _data._prepare_training_data(raster, cfg, "geoai")
        assert read_chip_meta(td)["n_reference_fields_without_interior"] is None
        assert "no field-interior pixel" not in caplog.text

    def test_cache_reuse_and_key(self, tmp_path, s2_scene):
        raster, ref, _ = s2_scene
        cfg = _chip_config(tmp_path, raster, ref, "geoai")
        td1 = _data._prepare_training_data(raster, cfg, "geoai")
        stamp = (td1 / "chips_meta.json").stat().st_mtime_ns
        td2 = _data._prepare_training_data(raster, cfg, "geoai")
        assert td1 == td2 and (td2 / "chips_meta.json").stat().st_mtime_ns == stamp
        td3 = _data._prepare_training_data(raster, cfg.merged(seed=99), "geoai")
        assert td3 != td1

    def test_incomplete_directory_is_rebuilt(self, tmp_path, s2_scene):
        raster, ref, _ = s2_scene
        cfg = _chip_config(tmp_path, raster, ref, "geoai")
        td = _data._prepare_training_data(raster, cfg, "geoai")
        (td / "chips_meta.json").unlink()
        td_again = _data._prepare_training_data(raster, cfg, "geoai")
        assert td_again == td and (td / "chips_meta.json").is_file()

    def test_too_few_chips_raises(self, tmp_path, s2_scene):
        raster, ref, _ = s2_scene
        cfg = _chip_config(tmp_path, raster, ref, "geoai", chip_size=128)
        with pytest.raises(ValueError, match="at least 2 are needed"):
            _data._prepare_training_data(raster, cfg, "geoai")

    def test_unknown_engine_raises(self, tmp_path, s2_scene):
        raster, ref, _ = s2_scene
        cfg = _chip_config(tmp_path, raster, ref, "geoai")
        with pytest.raises(ValueError, match="No training-chip definition"):
            _data._prepare_training_data(raster, cfg, "ftw")

    def test_geoai_chip_size_fits_the_reference_fields(self):
        # NMOSE-like centre pivots (~800 m) at NAIP 1 m: 1.25 x 800 px -> clamped to 1024.
        assert _data.geoai_chip_size_for_fields(np.full(50, 800.0), 1.0) == 1024
        # Small fields (60 m) at 1 m: 75 px -> the 256 px minimum.
        assert _data.geoai_chip_size_for_fields(np.full(50, 60.0), 1.0) == 256
        # 400 m fields at 1 m: 1.25 x 400 = 500 -> 512 (multiple of 32).
        assert _data.geoai_chip_size_for_fields(np.full(50, 400.0), 1.0) == 512
        # 10 m imagery: 800 m fields are 80 px -> 256.
        assert _data.geoai_chip_size_for_fields(np.full(50, 800.0), 10.0) == 256
        # The 90th percentile, not the maximum, drives the size.
        sides = np.r_[np.full(95, 200.0), np.full(5, 3000.0)]
        assert _data.geoai_chip_size_for_fields(sides, 1.0) == 256
        assert _data.geoai_chip_size_for_fields(np.zeros(0), 1.0) == 256

    def test_reference_field_sides_are_in_metres(self):
        from shapely.geometry import Polygon

        ref = gpd.GeoDataFrame(geometry=[box(X0, Y0 - 300, X0 + 800, Y0), Polygon()], crs=UTM)
        sides = _data.reference_field_sides_m(ref)
        assert sides.tolist() == [800.0]  # the empty geometry is dropped
        geo = ref.iloc[:1].to_crs(4326)
        assert _data.reference_field_sides_m(geo)[0] == pytest.approx(800.0, rel=0.02)

    def test_geoai_default_chip_comes_from_the_fields(self, tmp_path, s2_scene, caplog):
        raster, ref, _ = s2_scene
        cfg = _chip_config(tmp_path, raster, ref, "geoai")
        cfg = cfg.merged(engine_params={"min_label_fraction": 0.0})  # no chip_size
        # Fields of up to 500 m at 10 m are 50 px: 1.25 x 50 -> 64 -> the 256 px minimum,
        # larger than this 130 px raster, so no chip fits and the call raises.
        with (
            caplog.at_level(logging.INFO, logger="agribound.engines.finetune._data"),
            pytest.raises(ValueError, match="at least 2 are needed"),
        ):
            _data._prepare_training_data(raster, cfg, "geoai")
        assert "GeoAI chip size 256 px (2560 m)" in caplog.text
        assert "larger than a" not in caplog.text  # the fields fit: no warning

    def test_geoai_warns_when_fields_exceed_an_explicit_chip(self, tmp_path, s2_scene, caplog):
        raster, ref, _ = s2_scene
        cfg = _chip_config(tmp_path, raster, ref, "geoai", chip_size=32)  # 320 m chips
        with caplog.at_level(logging.WARNING, logger="agribound.engines.finetune._data"):
            _data._prepare_training_data(raster, cfg, "geoai")
        assert "larger than a 32 px (320 m) GeoAI chip" in caplog.text

    def test_delineate_anything_default_chip_size_follows_gsd(self):
        assert _data._default_chip_size("delineate-anything", 1.0) == 512
        assert _data._default_chip_size("delineate-anything", 10.0) == 256
        assert _data._default_chip_size("prithvi", 30.0) == 224

    def test_chip_gsd_uses_delineate_anything_pixel_size_rule(self, tmp_path):
        """Near the 4 m threshold the two GSD rules disagree on a geographic raster."""
        from agribound.engines.delineate_anything import pixel_size_m

        res, n = 0.00005, 20  # degrees; raster centred on 45 N
        raster = _write(
            tmp_path / "geo.tif",
            np.ones((3, n, n), dtype=np.float32),
            crs="EPSG:4326",
            res=res,
            x0=-100.0,
            y0=45.0 + n / 2 * res,
        )
        with rasterio.open(raster) as src:
            da_gsd = _data._chip_gsd_m(src, "delineate-anything")
            other_gsd = _data._chip_gsd_m(src, "geoai")
            assert da_gsd == pytest.approx(
                pixel_size_m(src.crs, src.transform, src.height, src.width)
            )
            assert other_gsd == pytest.approx(_data._gsd_m(src))
        # Mean of 5.56 m (north-south) and 3.93 m (east-west) vs 3.94 m east-west only.
        assert da_gsd == pytest.approx(4.75, abs=0.01)
        assert other_gsd == pytest.approx(3.94, abs=0.01)
        assert _data._default_chip_size("delineate-anything", da_gsd) == 256
        assert _data._default_chip_size("delineate-anything", other_gsd) == 512


@pytest.fixture
def pan_scene(tmp_path):
    """1-band float32 panchromatic-like raster (260 x 260 px at 1 m) and 2 touching fields."""
    rng = np.random.default_rng(8)
    data = rng.uniform(300, 600, (1, 260, 260)).astype(np.float32)
    data[:, 60:200, 20:240] += 900.0
    data[:, :6, :6] = np.nan
    raster = _write(tmp_path / "pan.tif", data, res=1.0, nodata=float("nan"))
    fields = [
        box(X0 + 20, Y0 - 200, X0 + 130, Y0 - 60),
        box(X0 + 130, Y0 - 200, X0 + 240, Y0 - 60),
    ]
    ref = gpd.GeoDataFrame({"name": ["a", "b"]}, geometry=fields, crs=UTM)
    ref_path = tmp_path / "ref.gpkg"
    ref.to_file(ref_path)
    return raster, str(ref_path)


def _da_chip_config(tmp_path, raster, ref, **params):
    return AgriboundConfig(
        source="local",
        local_tif_path=raster,
        bands={"R": 1, "G": 1, "B": 1},
        engine="delineate-anything",
        output_path=str(tmp_path / "out" / "fields.gpkg"),
        lulc_filter=False,
        reference_boundaries=ref,
        fine_tune_split="random",
        seed=3,
        engine_params={"chip_size": 64, "min_label_fraction": 0.0, **params},
    )


class TestDelineateAnythingChipOptions:
    """Chips follow processing_gsd_m and stretch_percentiles, as the native backend does."""

    def test_chips_are_cut_from_the_raster_the_backend_reads(self, tmp_path, pan_scene):
        from agribound.engines import delineate_anything as da

        raster, ref = pan_scene
        params = {"processing_gsd_m": 2.0, "stretch_percentiles": [0.5, 99.5]}
        cfg = _da_chip_config(tmp_path, raster, ref, **params)
        td = _data._prepare_training_data(raster, cfg, "delineate-anything")
        meta = read_chip_meta(td)
        assert meta["processing_gsd_m"] == 2.0 and meta["gsd_m"] == pytest.approx(2.0)
        assert meta["processing"]["scale"] == 2.0 and meta["processing"]["size_px"] == [130, 130]
        # 130 px: 3 x 3 chips of 64 px every 32 px, less those that cover cells of both splits.
        assert meta["band_indices"] == [1, 1, 1] and meta["chip_stride"] == 32
        assert meta["n_chips"] + meta["split"]["n_dropped_both_splits"] == 9
        assert meta["image"]["stretch_percentiles"] == [0.5, 99.5]
        # The raster and the stretch are those of the native backend at inference.
        path, bands, info = da.processing_raster(raster, cfg, 2.0, [1, 1, 1])
        assert (path, info) == (meta["processing"]["raster"], meta["processing"])
        with rasterio.open(path) as src:
            proc = src.read(bands)
            bounds = da.scene_stretch_bounds(src, bands, (0.5, 99.5))
        lows, highs = meta["image"]["stretch_lows"], meta["image"]["stretch_highs"]
        assert (lows, highs) == bounds
        chips = gpd.read_file(td / "chips.gpkg", layer="chips")
        for split in ("train", "val"):
            for img_p in split_files(td, split, "images"):
                row = chips.loc[chips["chip_id"] == int(img_p.stem.split("_")[1])].iloc[0]
                r, c = int(row["row_off"]), int(row["col_off"])
                with rasterio.open(img_p) as ds:
                    assert ds.res == (2.0, 2.0)
                    window = proc[:, r : r + 64, c : c + 64]
                    np.testing.assert_array_equal(
                        ds.read(), apply_stretch(window, lows, highs, valid_pixels(window))
                    )
        # Another working directory (a fine-tuning run's) gets an identical copy.
        other, _, _ = da.processing_raster(
            raster, cfg.merged(cache_dir=str(tmp_path / "other")), 2.0, [1, 1, 1]
        )
        with rasterio.open(other) as a, rasterio.open(path) as b:
            np.testing.assert_array_equal(a.read(), b.read())
            assert a.transform == b.transform

    def test_options_enter_the_cache_key_only_when_set(self, tmp_path, pan_scene):
        raster, ref = pan_scene
        base = _data._prepare_training_data(
            raster, _da_chip_config(tmp_path, raster, ref), "delineate-anything"
        )
        meta = read_chip_meta(base)
        assert meta["processing_gsd_m"] is None and meta["processing"] is None
        assert meta["gsd_m"] == pytest.approx(1.0) and meta["chip_stride"] == 32
        assert meta["n_chips"] + meta["split"]["n_dropped_both_splits"] == 49  # 7 x 7 chips
        assert meta["image"]["stretch_percentiles"] == [1.0, 99.0]
        explicit = _da_chip_config(tmp_path, raster, ref, stretch_percentiles=[1, 99])
        assert _data._prepare_training_data(raster, explicit, "delineate-anything") == base
        for params in ({"processing_gsd_m": 2.0}, {"stretch_percentiles": [0.5, 99.5]}):
            cfg = _da_chip_config(tmp_path, raster, ref, **params)
            assert _data._prepare_training_data(raster, cfg, "delineate-anything") != base

    def test_a_new_version_of_the_resampled_copy_makes_new_chips(
        self, tmp_path, pan_scene, monkeypatch
    ):
        from agribound.engines import delineate_anything as da

        raster, ref = pan_scene
        cfg = _da_chip_config(tmp_path, raster, ref, processing_gsd_m=2.0)
        first = _data._prepare_training_data(raster, cfg, "delineate-anything")
        assert _data._prepare_training_data(raster, cfg, "delineate-anything") == first
        monkeypatch.setattr(da, "_PROCESSING_RASTER_VERSION", "2")
        second = _data._prepare_training_data(raster, cfg, "delineate-anything")
        assert second != first
        assert (
            read_chip_meta(second)["processing"]["raster"]
            != (read_chip_meta(first)["processing"]["raster"])
        )

    def test_invalid_options_fail_before_chips_are_cut(self, tmp_path, pan_scene):
        raster, ref = pan_scene
        for params, match in (
            ({"processing_gsd_m": 0.5}, "finer than the raster's pixels"),
            ({"stretch_percentiles": [99, 1]}, "stretch_percentiles"),
            ({"backend": "reference", "processing_gsd_m": 2.0}, "processing_gsd_m"),
        ):
            cfg = _da_chip_config(tmp_path, raster, ref, **params)
            with pytest.raises(ValueError, match=match):
                _data._prepare_training_data(raster, cfg, "delineate-anything")
        assert not list(tmp_path.glob("out/.agribound_cache/finetune_data_*"))

    def test_other_engines_ignore_the_options(self, tmp_path, pan_scene):
        raster, ref = pan_scene
        cfg = _da_chip_config(tmp_path, raster, ref, processing_gsd_m=2.0).merged(engine="geoai")
        meta = read_chip_meta(_data._prepare_training_data(raster, cfg, "geoai"))
        assert meta["processing"] is None and meta["gsd_m"] == pytest.approx(1.0)
        assert meta["image"]["stretch_percentiles"] == [1.0, 99.0]


def _chip_windows(td):
    """``(chip_id, row_off, col_off, size, split)`` of every chip in chips.gpkg."""
    chips = gpd.read_file(td / "chips.gpkg", layer="chips")
    return [
        (int(r.chip_id), int(r.row_off), int(r.col_off), int(r.size), r.split)
        for r in chips.itertuples()
    ]


def _share_pixels(a, b):
    """True if two chip windows ``(row, col, size)`` share at least one pixel."""
    (ra, ca, sa), (rb, cb, sb) = a, b
    return ra < rb + sb and rb < ra + sa and ca < cb + sb and cb < ca + sa


class TestOverlappingChips:
    """Delineate-Anything chips are half a chip apart; the split never shares a pixel."""

    @pytest.mark.parametrize(
        "split",
        [
            {"fine_tune_split": "random"},
            {"fine_tune_split": "block", "fine_tune_block_size_m": 100.0},
            {"fine_tune_split": "column", "fine_tune_split_column": "name"},
        ],
    )
    def test_chips_overlap_and_no_pixel_is_in_both_splits(self, tmp_path, pan_scene, split):
        raster, ref = pan_scene
        cfg = _da_chip_config(tmp_path, raster, ref).merged(**split)
        td = _data._prepare_training_data(raster, cfg, "delineate-anything")
        meta = read_chip_meta(td)
        chips = _chip_windows(td)
        assert meta["chip_size"] == 64 and meta["chip_stride"] == 32
        assert all(row % 32 == 0 and col % 32 == 0 for _, row, col, _, _ in chips)
        info = meta["split"]
        assert info["unit"] == "chip-size cell" and info["strategy"] == split["fine_tune_split"]
        n_train = sum(s == "train" for *_, s in chips)
        assert (info["n_train"], info["n_val"]) == (n_train, len(chips) - n_train)
        assert info["n_train"] >= 1 and info["n_val"] >= 1 and meta["n_chips"] == len(chips)
        # 7 x 7 windows of 64 px every 32 px on the 260 px raster. They end at 256 px, so
        # they cover 4 x 4 cells of 64 px.
        assert len(chips) + info["n_dropped_both_splits"] == 49
        assert info["n_train_cells"] + info["n_val_cells"] == 16
        for split_name, prefix in (("train", ""), ("val", "val_")):
            names = {p.name for p in split_files(td, split_name, "images")}
            assert names == {f"chip_{cid:05d}.tif" for cid, *_, s in chips if s == split_name}
            assert not (td / f"{prefix}images" / "missing").exists()
        overlapping = 0
        for a in chips:
            for b in chips:
                if a[0] < b[0] and _share_pixels(a[1:4], b[1:4]):
                    overlapping += 1
                    assert a[4] == b[4], (a, b)  # chips that share a pixel share the split
        assert overlapping > 0

    @pytest.mark.parametrize(
        "split",
        [
            {"fine_tune_split": "random"},
            {"fine_tune_split": "block", "fine_tune_block_size_m": 100.0},
        ],
    )
    def test_chips_that_do_not_overlap_keep_the_chip_split(self, tmp_path, pan_scene, split):
        """With chip_stride = chip_size (other engines' default), each chip is its own cell."""
        raster, ref = pan_scene
        for engine, params in (("geoai", {}), ("delineate-anything", {"chip_stride": 64})):
            cfg = _da_chip_config(tmp_path, raster, ref, **params).merged(engine=engine, **split)
            td = _data._prepare_training_data(raster, cfg, engine)
            meta = read_chip_meta(td)
            chips = gpd.read_file(td / "chips.gpkg", layer="chips")
            assert meta["chip_stride"] == 64 and len(chips) == meta["n_chips"] == 16
            assert "n_dropped_both_splits" not in meta["split"]
            reference = gpd.read_file(ref).to_crs(chips.crs)
            expected = assign_splits(chips[["geometry"]], cfg, reference=reference)
            assert chips["split"].tolist() == expected.tolist()

    def test_chip_stride_default_and_validation(self, tmp_path, pan_scene):
        assert _data.chip_stride("delineate-anything", 512) == 256
        assert _data.chip_stride("delineate-anything", 65) == 32
        for engine in ("geoai", "dinov3", "prithvi"):
            assert _data.chip_stride(engine, 256) == 256
        assert _data.chip_stride("dinov3", 256, {"chip_stride": 128}) == 128
        assert _data.chip_stride("delineate-anything", 512, {"chip_stride": 512.0}) == 512
        assert _data.chip_stride("geoai", 256, {"chip_stride": np.int64(64)}) == 64
        for bad in (0, 513, 1.5, True, "256", float("nan")):
            with pytest.raises(ValueError, match="chip_stride"):
                _data.chip_stride("delineate-anything", 512, {"chip_stride": bad})
        raster, ref = pan_scene
        cfg = _da_chip_config(tmp_path, raster, ref, chip_stride=65)
        with pytest.raises(ValueError, match="chip_stride"):
            _data._prepare_training_data(raster, cfg, "delineate-anything")
        assert not list(tmp_path.glob("out/.agribound_cache/finetune_data_*"))

    def test_a_split_left_without_chips_raises(self, tmp_path):
        # 64 x 100 px: chips at columns 0 and 32. The second covers both cells, which are
        # in different splits, so one split keeps no chip.
        data = np.random.default_rng(1).uniform(300, 600, (1, 64, 100)).astype(np.float32)
        raster = _write(tmp_path / "strip.tif", data, res=1.0)
        ref = tmp_path / "ref.gpkg"
        field = box(X0 + 5, Y0 - 60, X0 + 95, Y0 - 5)
        gpd.GeoDataFrame({"name": ["a"]}, geometry=[field], crs=UTM).to_file(ref)
        cfg = _da_chip_config(tmp_path, raster, str(ref))
        with pytest.raises(ValueError, match=r"chip is left after dropping .* \(1 of 2\)"):
            _data._prepare_training_data(raster, cfg, "delineate-anything")
        assert not list(tmp_path.glob("out/.agribound_cache/finetune_data_*"))
        # Chips that do not overlap: one chip only, the usual error.
        with pytest.raises(ValueError, match="at least 2 are needed"):
            _data._prepare_training_data(
                raster,
                _da_chip_config(tmp_path, raster, str(ref), chip_stride=64),
                "delineate-anything",
            )


@pytest.fixture(params=["float32", "uint8"])
def nodata_scene(request, tmp_path):
    """3-band raster with a nodata block, a block of pixels 0 in every band, and one field."""
    dtype = request.param
    rng = np.random.default_rng(6)
    data = rng.uniform(20, 200, (3, 128, 128)).astype(dtype)
    data[:, 40:60, 40:70] = 0  # 0 in every band
    data[0, 80:90, 10:20] = 0  # 0 in one band only: valid
    nodata = -9999.0 if dtype == "float32" else 255
    data[:, 100:110, 90:120] = nodata
    raster = _write(tmp_path / f"rgb_{dtype}.tif", data, res=1.0, nodata=nodata)
    ref = tmp_path / "ref.gpkg"
    gpd.GeoDataFrame(geometry=[box(X0 + 8, Y0 - 120, X0 + 120, Y0 - 8)], crs=UTM).to_file(ref)
    return raster, str(ref)


def test_delineate_anything_chips_are_the_native_backend_tiles(tmp_path, nodata_scene):
    """Same valid pixels and the same image as the tiles the native backend reads."""
    from agribound.engines import delineate_anything as da

    raster, ref = nodata_scene
    cfg = _da_chip_config(tmp_path, raster, ref).merged(bands={"R": 1, "G": 2, "B": 3})
    td = _data._prepare_training_data(raster, cfg, "delineate-anything")
    meta = read_chip_meta(td)
    assert meta["zero_pixels_invalid"] is True and meta["band_indices"] == [1, 2, 3]
    lows, highs = meta["image"]["stretch_lows"], meta["image"]["stretch_highs"]
    n_zero_checked = 0
    with rasterio.open(raster) as src:
        for cid, row, col, size, split in _chip_windows(td):
            prefix = "" if split == "train" else "val_"
            with rasterio.open(td / f"{prefix}images" / f"chip_{cid:05d}.tif") as ds:
                image = ds.read()
            with rasterio.open(td / f"{prefix}masks" / f"chip_{cid:05d}.tif") as ds:
                mask = ds.read(1)
            data, valid = da._read_tile(src, [1, 2, 3], col, row, size)
            np.testing.assert_array_equal(mask != IGNORE_INDEX, valid)
            np.testing.assert_array_equal(image, da.stretch_with_bounds(data, lows, highs, valid))
            n_zero_checked += int((~valid).sum())
    assert n_zero_checked > 0
    # The other RGB engines keep their rule: pixels 0 in every band are valid there.
    geo = _data._prepare_training_data(raster, cfg.merged(engine="geoai"), "geoai")
    assert read_chip_meta(geo)["zero_pixels_invalid"] is False
    cid, _, _, _, split = next(c for c in _chip_windows(geo) if c[1:3] == (0, 0))
    prefix = "" if split == "train" else "val_"
    with rasterio.open(geo / f"{prefix}masks" / f"chip_{cid:05d}.tif") as ds:
        mask = ds.read(1)
    assert (mask[40:60, 40:64] != IGNORE_INDEX).all()


# ---------------------------------------------------------------------------
# Inference helpers and metadata
# ---------------------------------------------------------------------------


def test_mask_invalid_predictions(tmp_path):
    src = np.ones((3, 20, 20), dtype=np.float32)
    src[:, :5, :] = np.nan
    raster = _write(tmp_path / "src.tif", src, nodata=float("nan"))
    pred = _write(tmp_path / "pred.tif", np.ones((1, 20, 20), dtype=np.uint8))
    n = mask_invalid_predictions(pred, raster, [1, 2, 3], block_rows=7)
    assert n == 100
    with rasterio.open(pred) as ds:
        out = ds.read(1)
    assert (out[:5] == 0).all() and (out[5:] == 1).all()


def test_interior_polygons_restores_extent(tmp_path):
    pred = np.zeros((1, 40, 40), dtype=np.uint8)
    pred[0, 5:35, 5:35] = 2
    pred[0, 7:33, 7:33] = 1  # interior inset by 2 px
    path = _write(tmp_path / "seg.tif", pred, res=10.0)
    gdf = interior_polygons(path, 2, min_area_m2=0)
    assert len(gdf) == 1 and gdf["instance_id"].tolist() == [1]
    # 30 x 30 px field minus 2 * 3 / 2 = 3 px at each of its 4 corners
    assert gdf.geometry.iloc[0].area == pytest.approx((30 * 30 - 4 * 3) * 10.0**2)
    assert len(interior_polygons(path, 0, min_area_m2=0)) == 1
    assert interior_polygons(path, 0, min_area_m2=0).geometry.iloc[0].area == pytest.approx(
        (26 * 10.0) ** 2
    )
    assert len(interior_polygons(path, 2, min_area_m2=10 * 300.0**2)) == 0
    # growth stays inside the predicted boundary class: a 5 px dilation gives the same
    # result as 2 px here plus the corners, never more than the 30 x 30 px field
    assert interior_polygons(path, 5, min_area_m2=0).geometry.iloc[0].area == pytest.approx(
        (30 * 10.0) ** 2
    )


def test_interior_polygons_can_write_the_label_raster_elsewhere(tmp_path):
    """The label raster went next to the prediction only, which wrote into a finished run's
    cache when a script polygonised a cached DINOv3 prediction (example 14, 1.1 review L8)."""
    pred = np.zeros((1, 40, 40), dtype=np.uint8)
    pred[0, 5:35, 5:35] = 2
    pred[0, 7:33, 7:33] = 1
    run_cache = tmp_path / "run_cache"
    run_cache.mkdir()
    path = _write(run_cache / "seg.tif", pred, res=10.0)
    before = sorted(p.name for p in run_cache.iterdir())
    elsewhere = tmp_path / "scratch" / "labels"
    gdf = interior_polygons(path, 2, min_area_m2=0, out_dir=elsewhere)
    assert sorted(p.name for p in run_cache.iterdir()) == before
    assert sorted(p.name for p in elsewhere.iterdir()) == ["seg_fields_d2.tif"]
    beside = interior_polygons(path, 2, min_area_m2=0)  # the default is unchanged
    assert (run_cache / "seg_fields_d2.tif").exists()
    assert gdf.geometry.iloc[0].equals(beside.geometry.iloc[0])
    with (
        rasterio.open(elsewhere / "seg_fields_d2.tif") as a,
        rasterio.open(run_cache / "seg_fields_d2.tif") as b,
    ):
        assert (a.read(1) == b.read(1)).all() and a.transform == b.transform


def _diamond(k):
    from scipy.ndimage import generate_binary_structure, iterate_structure

    return iterate_structure(generate_binary_structure(2, 1), k)


@pytest.mark.parametrize("angle", [45.0, 30.0, 0.0])
def test_interior_polygons_neighbours_do_not_overlap(tmp_path, angle):
    """Fields sharing a slanted edge: no overlap, and each field is its opening."""
    from scipy.ndimage import binary_opening

    n, k = 140, 2
    yy, xx = np.mgrid[0:n, 0:n]
    inside = (yy > 10) & (yy < n - 10) & (xx > 10) & (xx < n - 10)
    t = np.tan(np.radians(angle))
    inst = np.where(inside, np.where(xx - n / 2 > t * (yy - n / 2), 2, 1), 0).astype(np.int32)
    sem = semantic_from_instances(inst, k)
    path = _write(tmp_path / f"seg_{angle}.tif", sem[None], res=10.0)
    gdf = interior_polygons(path, k, min_area_m2=0)
    assert len(gdf) == 2
    a, b = gdf.geometry.iloc[0], gdf.geometry.iloc[1]
    assert a.intersection(b).area == pytest.approx(0.0, abs=1e-6)
    # the grown label raster equals the city-block opening of each reference field
    with rasterio.open(tmp_path / f"seg_{angle}_fields_d{k}.tif") as src:
        labels = src.read(1)
    for field_id in (1, 2):
        grown = np.isin(labels, np.unique(labels[(inst == field_id) & (labels > 0)]))
        np.testing.assert_array_equal(grown, binary_opening(inst == field_id, _diamond(k)))


def test_grow_labels_competition_and_mask():
    from agribound.engines.finetune._data import grow_labels

    labels = np.zeros((1, 7), dtype=np.int32)
    labels[0, 0], labels[0, 6] = 1, 2
    allowed = np.ones((1, 7), dtype=bool)
    allowed[0, 1] = False
    out = grow_labels(labels, allowed, 3)
    # label 1 cannot pass the blocked pixel; label 2 grows 3 steps
    assert out.tolist() == [[1, 0, 0, 2, 2, 2, 2]]
    # equidistant pixel reached by both in the same step takes the larger label
    out = grow_labels(labels, np.ones((1, 7), dtype=bool), 5)
    assert out.tolist() == [[1, 1, 1, 2, 2, 2, 2]]
    assert labels.tolist() == [[1, 0, 0, 0, 0, 0, 2]]  # input not modified


def test_reflect_indices_matches_numpy_pad():
    from agribound.engines.finetune._data import reflect_indices

    for n in range(1, 7):
        for extra in (0, 1, n, 3 * n + 2):
            expected = np.pad(np.arange(n), (0, extra), mode="reflect")
            np.testing.assert_array_equal(reflect_indices(np.arange(n + extra), n), expected)
    assert reflect_indices([-1, -2], 4).tolist() == [1, 2]


def test_write_rgb_input_pad_to_reflects(tmp_path):
    rng = np.random.default_rng(3)
    data = rng.uniform(100, 3000, (3, 21, 13)).astype(np.float32)
    raster = _write(tmp_path / "src.tif", data, nodata=float("nan"))
    plain = tmp_path / "plain.tif"
    padded = tmp_path / "padded.tif"
    info_plain = write_rgb_input(raster, plain, [1, 2, 3])
    info = write_rgb_input(raster, padded, [1, 2, 3], pad_to=(64, 40), block_rows=8)
    assert "padded_to" not in info_plain
    assert info["padded_to"] == [64, 40] and info["padding"] == "reflect"
    with rasterio.open(plain) as a, rasterio.open(padded) as b:
        base = a.read()
        out = b.read()
        assert b.transform == a.transform and (b.height, b.width) == (64, 40)
    np.testing.assert_array_equal(out, np.pad(base, ((0, 0), (0, 43), (0, 27)), mode="reflect"))
    with pytest.raises(ValueError, match="smaller than the raster"):
        write_rgb_input(raster, tmp_path / "bad.tif", [1, 2, 3], pad_to=(20, 13))


def test_hf_cached_file_info_reads_snapshot_and_lfs_blob(tmp_path, monkeypatch):
    import huggingface_hub

    repo = tmp_path / "hub" / "models--org--model"
    commit = "63adbd39c271da4c42f447e69b1a7c91a338cdc9"
    sha = "3629cedfbb350faafcb0dac902ae0d3c927e25ce8d9e0024aa1276ec66956ddb"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / sha).write_bytes(b"weights")
    (repo / "blobs" / ("a" * 40)).write_bytes(b"{}")  # non-LFS blob: git SHA-1 name
    snap = repo / "snapshots" / commit
    (snap / "sub").mkdir(parents=True)
    files = {"sub/w.pt": snap / "sub" / "w.pt", "config.json": snap / "config.json"}
    resolved = {
        str(files["sub/w.pt"]): str(repo / "blobs" / sha),
        str(files["config.json"]): str(repo / "blobs" / ("a" * 40)),
    }
    cache_calls = []
    resolution_calls = []

    def cached_file(repo_id, filename, revision=None):
        cache_calls.append((repo_id, filename, revision))
        return str(files[filename]) if filename in files else None

    realpath = _data.os.path.realpath

    def resolve_cache_path(path, *args, **kwargs):
        # Exercise the resolved LFS/non-LFS paths without requiring Windows
        # symlink privileges. Filesystem link traversal belongs to os.path.
        if path in resolved:
            resolution_calls.append(path)
            return resolved[path]
        return realpath(path, *args, **kwargs)

    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        cached_file,
    )
    monkeypatch.setattr(_data.os.path, "realpath", resolve_cache_path)
    info = _data.hf_cached_file_info("org/model", "sub/w.pt", revision="release")
    assert info == {
        "repo_id": "org/model",
        "filename": "sub/w.pt",
        "revision": commit,
        "sha256": sha,
    }
    assert _data.hf_cached_file_info("org/model", "config.json")["sha256"] is None
    missing = _data.hf_cached_file_info("org/model", "absent.pt")
    assert missing["revision"] is None and missing["sha256"] is None
    assert cache_calls == [
        ("org/model", "sub/w.pt", "release"),
        ("org/model", "config.json", None),
        ("org/model", "absent.pt", None),
    ]
    assert resolution_calls == [str(files["sub/w.pt"]), str(files["config.json"])]


def test_hf_cached_file_info_reads_symlink_free_snapshot(tmp_path, monkeypatch):
    import huggingface_hub

    # Hugging Face can copy files into snapshots where symlinks are unavailable.
    # The snapshot still identifies its revision, but its ordinary filename does
    # not establish a content SHA-256, so do not manufacture one from the bytes.
    commit = "63adbd39c271da4c42f447e69b1a7c91a338cdc9"
    cached = tmp_path / "hub" / "models--org--model" / "snapshots" / commit / "w.pt"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"weights")
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo_id, filename, revision=None: str(cached),
    )
    assert _data.hf_cached_file_info("org/model", "w.pt") == {
        "repo_id": "org/model",
        "filename": "w.pt",
        "revision": commit,
        "sha256": None,
    }


def test_training_meta_round_trip(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"x")
    write_training_meta(ckpt, {"boundary_erosion": 3})
    assert read_training_meta(ckpt)["boundary_erosion"] == 3
    other = tmp_path / "other.ckpt"
    other.write_bytes(b"y")
    (tmp_path / "other.ckpt.agribound.json").write_text(json.dumps({"checkpoint": "best.ckpt"}))
    assert read_training_meta(other) == {}


def test_training_output_dir_depends_on_params(tmp_path):
    cfg = _config(tmp_path, engine_params={"use_lora": True})
    a = training_output_dir(cfg, "dinov3", tmp_path / "chips")
    b = training_output_dir(
        cfg.merged(engine_params={"use_lora": False}), "dinov3", tmp_path / "chips"
    )
    assert a != b and a.parent == b.parent == cfg.get_working_dir()


def test_chip_size_rule_is_part_of_the_fine_tuning_cache_key(tmp_path, monkeypatch):
    """A changed default chip rule must not reuse a checkpoint trained on other chips."""
    from agribound.engines import finetune

    cfg = _config(tmp_path, engine="geoai", reference_boundaries=str(tmp_path / "ref.gpkg"))
    (tmp_path / "ref.gpkg").write_text("x")
    first = finetune._run_dir(cfg, "geoai", "maskrcnn")
    monkeypatch.setattr(_data, "GEOAI_MAX_CHIP", 2048)
    second = finetune._run_dir(cfg, "geoai", "maskrcnn")
    assert first != second
    assert _data.chip_size_rule("geoai") == "fields-q0.9x1.25-256-2048"


def test_dinov3_recipe_version_is_part_of_the_dinov3_cache_key(tmp_path, monkeypatch):
    """A new DINOv3 training recipe must not reuse earlier checkpoints."""
    from agribound.engines import finetune
    from agribound.engines.finetune import _dinov3

    (tmp_path / "ref.gpkg").write_text("x")
    cfg = _config(tmp_path, engine="dinov3", reference_boundaries=str(tmp_path / "ref.gpkg"))
    first = finetune._run_dir(cfg, "dinov3", "large")
    monkeypatch.setattr(_dinov3, "RECIPE_VERSION", _dinov3.RECIPE_VERSION + 1)
    assert finetune._run_dir(cfg, "dinov3", "large") != first
    geo = _config(tmp_path, engine="geoai", reference_boundaries=str(tmp_path / "ref.gpkg"))
    before = finetune._run_dir(geo, "geoai", "maskrcnn")
    monkeypatch.setattr(_dinov3, "RECIPE_VERSION", _dinov3.RECIPE_VERSION + 1)
    assert finetune._run_dir(geo, "geoai", "maskrcnn") == before


def test_yolo_recipe_version_is_part_of_the_delineate_anything_cache_key(tmp_path, monkeypatch):
    """A new Delineate-Anything training recipe must not reuse earlier checkpoints."""
    from agribound.engines import finetune
    from agribound.engines.finetune import _yolo

    (tmp_path / "ref.gpkg").write_text("x")
    cfg = _config(
        tmp_path, engine="delineate-anything", reference_boundaries=str(tmp_path / "ref.gpkg")
    )
    first = finetune._run_dir(cfg, "delineate-anything", "large_v2")
    monkeypatch.setattr(_yolo, "RECIPE_VERSION", _yolo.RECIPE_VERSION + 1)
    assert finetune._run_dir(cfg, "delineate-anything", "large_v2") != first
    # Other engines' keys do not depend on it (their cached checkpoints stay valid).
    geo = _config(tmp_path, engine="geoai", reference_boundaries=str(tmp_path / "ref.gpkg"))
    before = finetune._run_dir(geo, "geoai", "maskrcnn")
    monkeypatch.setattr(_yolo, "RECIPE_VERSION", _yolo.RECIPE_VERSION + 1)
    assert finetune._run_dir(geo, "geoai", "maskrcnn") == before
