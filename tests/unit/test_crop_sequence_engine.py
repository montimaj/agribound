"""Tests of the crop-sequence zones engine on synthetic stacks and on the IA golden fixture.

The fixture (``tests/data/crop_sequence/``) holds the raw CDL 2016-2023 codes of
the crop-sequence pilot's IA square (public domain; pixels outside the 5 km
square are 0), the pilot's zone ids after the merge and the areas of its
polygons. The engine must reproduce them exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

pytest.importorskip("skimage")

from agribound.config import AgriboundConfig  # noqa: E402
from agribound.engines import crop_sequence as eng  # noqa: E402
from agribound.engines.crop_sequence import (  # noqa: E402
    CropSequenceEngine,
    barrier_mask,
    crop_sequence_zones,
    cultivated_mask,
    focal_mode,
    merge_max_px,
    sequence_regions,
    zone_attributes,
)
from agribound.postprocess.regions import merge_small_regions, polygonize_labels  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "crop_sequence"
UTM = "EPSG:32615"
ORIGIN = (600000.0, 4650000.0)


def _write_stack(path, stack, *, crs=UTM, res=30.0, years=None, tags=None, nodata=0, desc=None):
    stack = np.asarray(stack)
    count, height, width = stack.shape
    transform = Affine(res, 0.0, ORIGIN[0], 0.0, -res, ORIGIN[1])
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(stack.dtype),
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)
        names = desc or [f"CDL{y}" for y in (years or range(2016, 2016 + count))]
        for i, name in enumerate(names, start=1):
            dst.set_band_description(i, name)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return str(path)


def _local_config(tmp_path, path, **params):
    min_area = params.pop("min_field_area_m2", 0.0)
    engine_params = {"value_scale": "categorical", "dataset": "cdl", **params}
    return AgriboundConfig(
        source="local",
        engine="crop-sequence",
        local_tif_path=str(path),
        output_path=str(tmp_path / "zones.gpkg"),
        cache_dir=str(tmp_path / "cache"),
        lulc_filter=False,
        min_field_area_m2=min_area,
        engine_params=engine_params,
    )


def _two_blocks(height=20, width=20):
    """8 years: left half corn/soy (1/5) rotation, right half soy/corn; all cultivated."""
    left = np.array([1, 5] * 4, dtype=np.uint8)
    right = np.array([5, 1] * 4, dtype=np.uint8)
    stack = np.zeros((8, height, width), dtype=np.uint8)
    stack[:, :, : width // 2] = left[:, None, None]
    stack[:, :, width // 2 :] = right[:, None, None]
    return stack


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


class TestMasks:
    def test_strict_majority_and_the_4_of_8_tie(self):
        stack = np.full((8, 1, 4), 63, dtype=np.uint8)  # 63 = forest (not agricultural)
        stack[:5, 0, 0] = 1  # 5 of 8 agricultural
        stack[:4, 0, 1] = 1  # 4 of 8: a tie, not cultivated
        stack[:, 0, 2] = 37  # 8 of 8 (hay)
        cult = cultivated_mask(stack, [1, 37], 8 // 2 + 1)
        assert cult[0].tolist() == [True, False, True, False]
        base = np.array([[False, True, True, True]])
        assert cultivated_mask(stack, [1, 37], 5, base=base)[0].tolist() == [
            False,
            False,
            True,
            False,
        ]

    def test_barrier_in_any_year(self):
        stack = np.ones((8, 1, 3), dtype=np.uint8)
        stack[6, 0, 1] = 121
        assert barrier_mask(stack, [121, 122])[0].tolist() == [False, True, False]
        assert not barrier_mask(stack, []).any()

    def test_focal_mode_ties_go_to_the_lowest_code_and_only_counted_pixels_vote(self):
        # disk(1) is the centre and its 4 neighbours.
        band = np.array([[0, 9, 0], [5, 7, 9], [0, 5, 0]], dtype=np.uint8)
        stack = band[None]
        count = np.ones((3, 3), dtype=bool)
        count[1, 1] = False  # the centre does not vote
        out = focal_mode(stack, 1, count)
        assert out[0, 1, 1] == 5  # two 5s against two 9s: the lower code
        count2 = count.copy()
        count2[2, 1] = False  # one 5 no longer votes
        assert focal_mode(stack, 1, count2)[0, 1, 1] == 9
        # A pixel with no counted neighbour gets 0; radius 0 keeps the counted codes.
        assert focal_mode(stack, 1, np.zeros((3, 3), dtype=bool))[0, 1, 1] == 0
        kept = focal_mode(stack, 0, np.ones((3, 3), dtype=bool))
        assert np.array_equal(kept, stack)

    def test_sequence_regions_are_8_connected_and_numbered_in_raster_order(self):
        stack = np.zeros((2, 3, 4), dtype=np.uint8)
        stack[:, :, :] = np.array([1, 2], dtype=np.uint8)[:, None, None]
        stack[:, 0, 3] = [3, 3]  # another sequence in the top-right corner
        valid = np.ones((3, 4), dtype=bool)
        valid[1, :] = False  # splits the first sequence into two rows
        regions, n_sequences = sequence_regions(stack, valid)
        assert n_sequences == 2
        assert regions.tolist() == [[1, 1, 1, 2], [0, 0, 0, 0], [3, 3, 3, 3]]
        # Diagonal neighbours are connected (8-connectivity).
        diag = np.array([[True, False], [False, True]])
        regions, _ = sequence_regions(np.ones((1, 2, 2), dtype=np.uint8), diag)
        assert regions.tolist() == [[1, 0], [0, 1]]

    def test_merge_max_px(self):
        assert merge_max_px(10_000, 900.0) == 11
        assert merge_max_px(10_000, 100.0) == 99
        assert merge_max_px(10_000, 900.0000000001) == 11
        assert merge_max_px(0, 900.0) == 0


class TestMergeSmallRegions:
    def test_smallest_first_into_the_largest_neighbour(self):
        regions = np.array(
            [
                [1, 1, 1, 2, 3, 3],
                [1, 1, 1, 2, 3, 3],
                [1, 1, 1, 4, 3, 3],
            ]
        )
        labels, stats = merge_small_regions(regions, max_px=2)
        # 4 (1 px) goes first, into its largest neighbour: 1 (9 px) over 3 (6 px) and 2.
        # 2 (2 px) then goes into 1 (now 10 px) over 3.
        assert labels.tolist() == [[1, 1, 1, 1, 3, 3]] * 2 + [[1, 1, 1, 1, 3, 3]]
        assert stats["n_merges"] == 2 and stats["n_regions_after_merge"] == 2
        assert stats["zone_px"][1] == 12 and stats["n_merged"][1] == 2
        assert stats["n_isolated_small_regions"] == 0

    def test_ties_go_to_the_lowest_id_and_merged_regions_are_taken_again(self):
        regions = np.array([[1, 1, 2, 3, 3]])
        labels, stats = merge_small_regions(regions, max_px=2)
        # 2 (1 px) merges first; 1 and 3 have 2 px each: the lower id, 1, takes it.
        # 1 (3 px) is no longer small; 3 (2 px) then merges into 1.
        assert labels.tolist() == [[1, 1, 1, 1, 1]]
        assert stats["n_merges"] == 2 and stats["n_merged"][1] == 2

    def test_isolated_small_regions_are_kept(self):
        regions = np.array([[1, 0, 2, 2, 2], [0, 0, 2, 2, 2]])
        labels, stats = merge_small_regions(regions, max_px=1)
        assert labels.tolist() == regions.tolist()
        assert stats["n_isolated_small_regions"] == 1 and stats["n_merges"] == 0

    def test_only_4_adjacency_counts(self):
        regions = np.array([[1, 0], [0, 2], [0, 2]])
        labels, stats = merge_small_regions(regions, max_px=1)
        assert labels.tolist() == regions.tolist()  # 1 touches 2 only at a corner

    def test_rejects_an_unknown_rule(self):
        with pytest.raises(ValueError, match="merge rule"):
            merge_small_regions(np.ones((2, 2), dtype=int), max_px=1, rule="longest_border")


class TestZones:
    def test_two_blocks_with_different_sequences_give_two_zones(self):
        zones = crop_sequence_zones(
            _two_blocks(),
            classes=[1, 5],
            barrier_classes=[121],
            min_crop_years=5,
            smooth_radius_px=3,
            max_merge_px=11,
        )
        assert sorted(np.unique(zones.labels[zones.labels > 0]).tolist()) == [1, 2]
        assert zones.stats["n_sequences"] == 2 and zones.stats["n_regions_after_merge"] == 2

    def test_equal_sequences_split_by_a_barrier_give_two_zones(self):
        stack = np.ones((8, 20, 21), dtype=np.uint8)
        stack[3, :, 10] = 121  # a road in one year
        zones = crop_sequence_zones(
            stack,
            classes=[1],
            barrier_classes=[121],
            min_crop_years=5,
            smooth_radius_px=0,
            max_merge_px=11,
        )
        assert not zones.valid[:, 10].any()
        assert len(np.unique(zones.labels[zones.labels > 0])) == 2
        assert zones.stats["n_sequences"] == 1

    def test_the_smoothing_absorbs_a_single_odd_pixel(self):
        stack = np.ones((8, 15, 15), dtype=np.uint8)
        stack[2, 7, 7] = 5  # one pixel with another crop in one year
        zones = crop_sequence_zones(
            stack,
            classes=[1, 5],
            barrier_classes=[],
            min_crop_years=5,
            smooth_radius_px=3,
            max_merge_px=11,
        )
        assert zones.smoothed[2, 7, 7] == 1
        assert zones.stats["n_regions_after_merge"] == 1

    def test_zone_attributes_take_the_most_frequent_code_per_year(self):
        labels = np.array([[1, 1, 1, 2], [1, 1, 0, 2]])
        stack = np.zeros((2, 2, 4), dtype=np.uint8)
        stack[0] = [[1, 1, 5, 9], [5, 1, 0, 9]]  # zone 1: three 1s, two 5s
        stack[1] = [[5, 1, 5, 9], [1, 0, 0, 3]]  # zone 1: two 5s, two 1s, one 0 -> 1 (tie)
        table = zone_attributes(labels, stack, [2016, 2017])
        assert table.index.tolist() == [1, 2]
        assert table["agribound:zone_px"].tolist() == [5, 2]
        assert table["agribound:crop_2016"].tolist() == [1, 9]
        assert table["agribound:crop_2017"].tolist() == [1, 3]
        assert table["agribound:crop_2016"].dtype == np.uint8


class TestPolygonizeLabels:
    def test_parts_area_filter_and_one_pixel_densification(self):
        labels = np.zeros((6, 8), dtype=np.int32)
        labels[0:4, 0:4] = 1  # 16 px
        labels[4, 4] = 1  # touches zone 1 only at a corner: a separate part
        labels[0:2, 6:8] = 2  # 4 px
        transform = Affine(30.0, 0, 0.0, 0, -30.0, 180.0)
        gdf = polygonize_labels(labels, transform, UTM, densify=30.0, column="zone")
        assert gdf["zone"].tolist() == [1, 1, 2]
        assert sorted(gdf.area.round(6).tolist()) == [900.0, 3600.0, 14400.0]
        big = gdf.geometry.iloc[int(np.argmax(gdf.area))]
        coords = np.asarray(big.exterior.coords)
        steps = np.hypot(*np.diff(coords, axis=0).T)
        assert steps.max() <= 30.0 + 1e-9 and len(coords) == 17  # 16 edges of 30 m
        filtered = polygonize_labels(labels, transform, UTM, min_area_m2=1000.0, column="zone")
        assert sorted(filtered.area.round(3).tolist()) == [3600.0, 14400.0]

    def test_empty_labels(self):
        gdf = polygonize_labels(np.zeros((3, 3), dtype=np.int32), Affine.identity(), UTM)
        assert len(gdf) == 0 and "label" in gdf.columns


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngine:
    def test_delineate_columns_meta_and_densified_outlines(self, tmp_path):
        path = _write_stack(tmp_path / "s.tif", _two_blocks(), tags={"AGRIBOUND_FOO": "x"})
        cfg = _local_config(tmp_path, path, classes=[1, 5], barrier_classes=[121])
        gdf = CropSequenceEngine().delineate(path, cfg)
        assert len(gdf) == 2 and gdf.crs.to_epsg() == 32615
        years = list(range(2016, 2024))
        expected = ["agribound:zone_id", "agribound:zone_px", "agribound:n_zones_merged"]
        assert list(gdf.columns) == expected + [f"agribound:crop_{y}" for y in years] + ["geometry"]
        left = gdf[gdf["agribound:crop_2016"] == 1].iloc[0]
        assert [int(left[f"agribound:crop_{y}"]) for y in years] == [1, 5] * 4
        assert int(left["agribound:zone_px"]) == 200
        meta = gdf.attrs["engine_meta"]
        assert meta["method"] == "zones" and meta["implementation"] == eng.IMPLEMENTATION
        assert meta["field_instances"] is False
        assert meta["years"] == years and meta["min_crop_years"] == 5
        assert meta["classes"] == [1, 5] and meta["barrier_classes"] == [121]
        assert meta["pixel_size_m"] == 30.0 and meta["max_merge_px"] == 11
        for key in (
            "n_valid_px",
            "n_sequences",
            "n_regions_before_merge",
            "n_merges",
            "n_regions_after_merge",
            "n_isolated_small_regions",
        ):
            assert isinstance(meta[key], int), key
        assert set(meta["unique_sequence_share_3x3"]) == {"smoothed", "raw"}
        assert Path(meta["label_raster"]).exists() and meta["cache_hit"] is False
        json.dumps(meta)  # JSON-serialisable for the provenance record
        steps = np.hypot(*np.diff(np.asarray(gdf.geometry.iloc[0].exterior.coords), axis=0).T)
        assert steps.max() <= 30.0 + 1e-6

    def test_crop_columns_hold_the_smoothed_codes_not_the_raw_ones(self, tmp_path):
        # A one-pixel strip of corn (1) in every year between cultivated barrier pixels:
        # soybeans (5) in 2016-2022 and developed land (121) in 2023. The strip is the only
        # valid region, and the focal mode (radius 3) counts the barrier pixels around it,
        # so its smoothed codes are 5 in 2016-2022 and 121 in 2023, while its raw codes
        # are 1 in every year.
        stack = np.full((8, 9, 17), 5, dtype=np.uint8)
        stack[7] = 121
        stack[:, 4, 1:16] = 1
        path = _write_stack(tmp_path / "s.tif", stack)
        cfg = _local_config(tmp_path, path, classes=[1, 5], barrier_classes=[121])
        gdf = CropSequenceEngine().delineate(path, cfg)
        assert len(gdf) == 1 and int(gdf["agribound:zone_px"].iloc[0]) == 15
        years = list(range(2016, 2024))
        crops = [int(gdf[f"agribound:crop_{y}"].iloc[0]) for y in years]
        assert crops == [5] * 7 + [121]
        with rasterio.open(gdf.attrs["engine_meta"]["label_raster"]) as src:
            labels = src.read(1)
        raw = zone_attributes(labels, stack, years)
        assert [int(raw[f"agribound:crop_{y}"].iloc[0]) for y in years] == [1] * 8

    def test_concurrent_writers_of_one_cache_entry_both_finish(self, tmp_path, monkeypatch):
        """Each writer uses its own partial files, so a second writer of the same entry that
        runs while the first is between writing and renaming does not break either."""
        import os

        labels = np.arange(1, 13, dtype=np.int32).reshape(3, 4)
        table = {"zone_id": np.arange(1, 13), "years": np.array([2016, 2017])}
        label_path = tmp_path / "crop_sequence_zones_k.tif"
        paths = (label_path, label_path.with_suffix(".json"), label_path.with_suffix(".npz"))
        transform = Affine(30.0, 0, ORIGIN[0], 0, -30.0, ORIGIN[1])
        real_replace = os.replace
        sources = []

        def replace(src, dst):
            sources.append(Path(src).name)
            if len(sources) == 1:  # the first writer is about to rename its label raster
                eng._write_cache(*paths, labels + 100, transform, UTM, {"w": 2}, table)
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", replace)
        eng._write_cache(*paths, labels, transform, UTM, {"w": 1}, table)
        monkeypatch.setattr(os, "replace", real_replace)
        assert len(sources) == 6 and len(set(sources)) == 6
        assert all(f".{os.getpid()}." in name and ".partial" in name for name in sources)
        assert not [p.name for p in tmp_path.iterdir() if ".partial" in p.name]
        cached = eng._read_cache(*paths, labels.shape)
        assert cached is not None
        assert np.array_equal(cached[0], labels)  # the first writer renamed last
        assert cached[1] == {"w": 1}

    def test_a_failed_cache_write_leaves_no_partial_file(self, tmp_path, monkeypatch):
        label_path = tmp_path / "crop_sequence_zones_k.tif"
        paths = (label_path, label_path.with_suffix(".json"), label_path.with_suffix(".npz"))
        transform = Affine(30.0, 0, ORIGIN[0], 0, -30.0, ORIGIN[1])

        def fail(file, **arrays):
            Path(file).write_bytes(b"half a table")  # a partial file is left behind ...
            raise OSError("disk full")  # ... when the write fails

        monkeypatch.setattr(np, "savez_compressed", fail)
        with pytest.raises(OSError, match="disk full"):
            eng._write_cache(
                *paths, np.ones((2, 2), np.int32), transform, UTM, {}, {"zone_id": np.ones(1)}
            )
        assert not [p.name for p in tmp_path.iterdir() if ".partial" in p.name]
        assert eng._read_cache(*paths, (2, 2)) is None  # incomplete: rebuilt next time

    def test_default_classes_of_the_dataset(self, tmp_path):
        path = _write_stack(tmp_path / "s.tif", _two_blocks())
        meta = (
            CropSequenceEngine().delineate(path, _local_config(tmp_path, path)).attrs["engine_meta"]
        )
        from agribound.composites.crop_sequence import CDL_AGRICULTURAL_CLASSES

        assert meta["classes"] == list(CDL_AGRICULTURAL_CLASSES)
        assert meta["barrier_classes"] == [121, 122, 123, 124] and meta["dataset"] == "cdl"

    def test_cache_reuse(self, tmp_path, monkeypatch):
        path = _write_stack(tmp_path / "s.tif", _two_blocks())
        cfg = _local_config(tmp_path, path)
        first = CropSequenceEngine().delineate(path, cfg)

        def fail(*args, **kwargs):
            raise AssertionError("the zones were computed again")

        monkeypatch.setattr(eng, "crop_sequence_zones", fail)
        again = CropSequenceEngine().delineate(path, cfg)
        assert again.attrs["engine_meta"]["cache_hit"] is True
        assert again.drop(columns="geometry").equals(first.drop(columns="geometry"))
        assert again.geometry.equals(first.geometry)
        # min_field_area_m2 and attributes only change the polygons: the cache is reused.
        bigger = cfg.merged(min_field_area_m2=1e9)
        assert len(CropSequenceEngine().delineate(path, bigger)) == 0
        plain = cfg.merged(engine_params={**cfg.engine_params, "attributes": False})
        assert list(CropSequenceEngine().delineate(path, plain).columns) == [
            "agribound:zone_id",
            "geometry",
        ]
        # A parameter of the zones is another cache entry.
        other = cfg.merged(engine_params={**cfg.engine_params, "smooth_radius_px": 2})
        with pytest.raises(AssertionError, match="computed again"):
            CropSequenceEngine().delineate(path, other)

    def test_deterministic(self, tmp_path):
        path = _write_stack(tmp_path / "s.tif", _ia_like_noise())
        a = CropSequenceEngine().delineate(path, _local_config(tmp_path / "a", path))
        b = CropSequenceEngine().delineate(path, _local_config(tmp_path / "b", path))
        with rasterio.open(a.attrs["engine_meta"]["label_raster"]) as src:
            la = src.read(1)
        with rasterio.open(b.attrs["engine_meta"]["label_raster"]) as src:
            lb = src.read(1)
        assert np.array_equal(la, lb) and a.geometry.equals(b.geometry)

    def test_years_from_tags_descriptions_and_params(self, tmp_path):
        stack = _two_blocks()[:3]
        tagged = _write_stack(
            tmp_path / "t.tif",
            stack,
            desc=["a", "b", "c"],
            tags={"AGRIBOUND_YEARS": "2001,2003,2005"},
        )
        assert eng.read_stack(tagged)[1] == [2001, 2003, 2005]
        described = _write_stack(tmp_path / "d.tif", stack, desc=["LC_2019 x", "LC_2020", "y2021"])
        assert eng.read_stack(described)[1] == [2019, 2020, 2021]
        assert eng.read_stack(described, years=[1, 2, 3])[1] == [1, 2, 3]
        bare = _write_stack(tmp_path / "b.tif", stack, desc=["", "", ""])
        with pytest.raises(ValueError, match="engine_params\\['years'\\]"):
            eng.read_stack(bare)
        with pytest.raises(ValueError, match="give 2 years for the 3 bands"):
            eng.read_stack(described, years=[2019, 2020])
        with pytest.raises(ValueError, match="repeat a year"):
            eng.read_stack(described, years=[2019, 2019, 2020])

    def test_nodata_and_whole_number_floats_are_read_as_codes(self, tmp_path):
        stack = _two_blocks()[:2].astype(np.float32)
        stack[0, 0, 0] = np.nan
        stack[1, 0, 1] = -9999.0
        path = _write_stack(tmp_path / "f.tif", stack, nodata=-9999.0)
        codes = eng.read_stack(path)[0]
        assert codes.dtype == np.uint8 and codes[0, 0, 0] == 0 and codes[1, 0, 1] == 0

    @pytest.mark.parametrize(
        ("values", "dtype", "match"),
        [
            (0.5, "float32", "non-integer"),
            (300, "uint16", "0-255"),
            (-2, "int16", "0-255"),
        ],
    )
    def test_refuses_values_that_are_not_codes(self, tmp_path, values, dtype, match):
        stack = np.ones((2, 4, 4), dtype=dtype)
        stack[0, 0, 0] = values
        path = _write_stack(tmp_path / "x.tif", stack, nodata=None)
        with pytest.raises(ValueError, match=match):
            CropSequenceEngine().delineate(path, _local_config(tmp_path, path))

    def test_refuses_geographic_and_non_square_rasters(self, tmp_path):
        stack = _two_blocks(4, 4)
        geo = tmp_path / "geo.tif"
        with rasterio.open(
            geo, "w", driver="GTiff", height=4, width=4, count=8, dtype="uint8",
            crs="EPSG:4326", transform=Affine(0.0003, 0, -93.6, 0, -0.0003, 42.0), nodata=0,
        ) as dst:  # fmt: skip
            dst.write(stack)
            for i in range(8):
                dst.set_band_description(i + 1, f"CDL{2016 + i}")
        with pytest.raises(ValueError, match="projected CRS"):
            CropSequenceEngine().delineate(str(geo), _local_config(tmp_path, geo))
        rect = tmp_path / "rect.tif"
        with rasterio.open(
            rect, "w", driver="GTiff", height=4, width=4, count=8, dtype="uint8",
            crs=UTM, transform=Affine(30.0, 0, 6e5, 0, -20.0, 4.65e6), nodata=0,
        ) as dst:  # fmt: skip
            dst.write(stack)
            for i in range(8):
                dst.set_band_description(i + 1, f"CDL{2016 + i}")
        with pytest.raises(ValueError, match="square pixels"):
            CropSequenceEngine().delineate(str(rect), _local_config(tmp_path, rect))

    def test_refuses_an_imagery_raster_even_when_declared_categorical(self, tmp_path):
        path = _write_stack(
            tmp_path / "s2.tif", _two_blocks(), tags={"AGRIBOUND_VALUE_SCALE": "reflectance_x10000"}
        )
        with pytest.raises(ValueError, match="'reflectance_x10000'"):
            CropSequenceEngine().delineate(path, _local_config(tmp_path, path))


class TestParams:
    def _cfg(self, tmp_path, source="local", **engine_params):
        if source == "local":
            return AgriboundConfig(
                source="local",
                engine="crop-sequence",
                local_tif_path="/tmp/x.tif",
                lulc_filter=False,
                output_path=str(tmp_path / "o.gpkg"),
                engine_params={"value_scale": "categorical", "dataset": "cdl", **engine_params},
            )
        return AgriboundConfig(
            source=source,
            engine="crop-sequence",
            year=2023,
            study_area="bbox:-93.6,42.0,-93.5,42.1",
            gee_project="p",
            lulc_filter=False,
            output_path=str(tmp_path / "o.gpkg"),
            engine_params=engine_params,
        )

    def test_defaults_resolve_per_dataset(self, tmp_path):
        from agribound.composites.crop_sequence import ACI_AGRICULTURAL_CLASSES

        params = CropSequenceEngine.resolve_params(self._cfg(tmp_path, "aci-sequence"))
        assert params["dataset"] == "aci" and params["classes"] == list(ACI_AGRICULTURAL_CLASSES)
        assert params["barrier_classes"] == [34] and params["min_crop_years"] == "auto"
        assert params["smooth_radius_px"] == 3 and params["min_region_m2"] == 10_000.0
        assert params["method"] == "zones" and params["merge_rule"] == "largest"

    def test_unknown_params_raise_and_pipeline_keys_pass(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown engine_params .*'seed_spacing_px'"):
            CropSequenceEngine.resolve_params(self._cfg(tmp_path, seed_spacing_px=30))
        params = CropSequenceEngine.resolve_params(
            self._cfg(tmp_path, smooth_iterations=0, regularize="none", pivots="snap")
        )
        assert "smooth_iterations" not in params and "pivots" not in params

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            ({"method": "snic"}, "method must be one of"),
            ({"merge_rule": "longest_border"}, "merge_rule must be one of"),
            ({"classes": [0, 1]}, "1-255"),
            ({"classes": "corn"}, "list of class codes"),
            ({"classes": [1, 121]}, "share the codes \\[121\\]"),
            ({"min_crop_years": 0}, ">= 1"),
            ({"smooth_radius_px": -1}, "smooth_radius_px"),
            ({"min_region_m2": -5}, "min_region_m2"),
            ({"dataset": "csb"}, "dataset"),
            ({"attributes": "yes"}, "attributes"),
        ],
    )
    def test_invalid_values(self, tmp_path, params, match):
        with pytest.raises(ValueError, match=match):
            CropSequenceEngine.resolve_params(self._cfg(tmp_path, **params))

    def test_source_specific_rules(self, tmp_path):
        with pytest.raises(ValueError, match="contradicts source 'cdl-sequence'"):
            CropSequenceEngine.resolve_params(self._cfg(tmp_path, "cdl-sequence", dataset="aci"))
        with pytest.raises(ValueError, match="applies to source='local' only"):
            CropSequenceEngine.resolve_params(self._cfg(tmp_path, "cdl-sequence", years=[1, 2]))
        cfg = self._cfg(tmp_path)
        cfg.engine_params = {"value_scale": "categorical"}  # changed after validation
        with pytest.raises(ValueError, match="'dataset'"):
            CropSequenceEngine.resolve_params(cfg)
        cfg.engine_params = {"dataset": "cdl"}
        with pytest.raises(ValueError, match="value_scale"):
            CropSequenceEngine.resolve_params(cfg)
        cfg.source = "sentinel2"
        with pytest.raises(ValueError, match="reads stacks of crop-type maps"):
            CropSequenceEngine.resolve_params(cfg)

    def test_radius_may_not_exceed_the_stack_pad(self, tmp_path):
        cfg = self._cfg(tmp_path, "cdl-sequence", smooth_radius_px=4)
        cfg.cache_dir = str(tmp_path / "cache")
        path = _write_stack(
            tmp_path / "stack.tif",
            _two_blocks(),
            tags={
                "AGRIBOUND_VALUE_SCALE": "categorical",
                "AGRIBOUND_SOURCE": "cdl-sequence",
                "AGRIBOUND_PAD_PX": 3,
            },
        )
        with pytest.raises(ValueError, match="exceeds the stack's pad of 3 px"):
            CropSequenceEngine().delineate(path, cfg)
        aci = _write_stack(
            tmp_path / "aci.tif",
            _two_blocks(),
            tags={"AGRIBOUND_VALUE_SCALE": "categorical", "AGRIBOUND_SOURCE": "aci-sequence"},
        )
        with pytest.raises(ValueError, match="is a 'aci-sequence' stack"):
            CropSequenceEngine().delineate(aci, self._cfg(tmp_path, "cdl-sequence"))

    def test_explicit_classes_missing_from_the_legend_warn(self, tmp_path, caplog):
        cfg = self._cfg(tmp_path, "cdl-sequence", classes=[1, 5, 7], barrier_classes=[121])
        cfg.cache_dir = str(tmp_path / "cache")
        path = _write_stack(
            tmp_path / "stack.tif",
            _two_blocks(),
            tags={
                "AGRIBOUND_VALUE_SCALE": "categorical",
                "AGRIBOUND_SOURCE": "cdl-sequence",
                "AGRIBOUND_LEGEND": json.dumps({"1": "Corn", "5": "Soybeans", "121": "Dev"}),
            },
        )
        meta = CropSequenceEngine().delineate(path, cfg).attrs["engine_meta"]
        assert meta["classes_not_in_legend"] == {"classes": [7]}
        assert any("not in the legend" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Golden fixture: the pilot's IA square
# ---------------------------------------------------------------------------


def _ia_like_noise():
    rng = np.random.default_rng(3)
    stack = _two_blocks(30, 30)
    noise = rng.random(stack.shape) < 0.05
    stack[noise] = 37
    return stack


@pytest.mark.skipif(not (DATA / "ia_cdl_2016_2023.tif").is_file(), reason="fixture missing")
def test_golden_ia_square_reproduces_the_pilot_zones(tmp_path):
    expected = json.loads((DATA / "ia_expected.json").read_text())
    stack_path = DATA / "ia_cdl_2016_2023.tif"
    cfg = _local_config(tmp_path, stack_path, min_field_area_m2=expected["min_area_m2"])
    gdf = CropSequenceEngine().delineate(str(stack_path), cfg)
    meta = gdf.attrs["engine_meta"]
    with rasterio.open(meta["label_raster"]) as src:
        labels = src.read(1)
    with rasterio.open(DATA / "ia_zones_expected.tif") as src:
        expected_labels = src.read(1)
    assert np.array_equal(labels, expected_labels)
    assert meta["n_regions_after_merge"] == expected["n_zones"]
    assert meta["years"] == list(range(2016, 2024)) and meta["min_crop_years"] == 5
    assert meta["max_merge_px"] == 11
    assert len(gdf) == expected["n_polygons"] == 209
    areas = np.sort(gdf.area.to_numpy())
    assert np.allclose(areas, expected["areas_m2"], rtol=0, atol=1e-6)
    assert gdf.union_all().area == pytest.approx(expected["union_area_m2"], abs=1e-6)
    assert f"EPSG:{gdf.crs.to_epsg()}" == expected["crs"]
    # The pilot's counts (its research code), and the per-zone columns: the most frequent
    # code of the pilot's smoothed stack per year (in 48 of the 207 zones it differs from
    # the most frequent raw code in some year), the zone's pixels and the regions merged
    # into it.
    for key, value in expected["pilot_counts"].items():
        assert meta[key] == value, key
    years = list(range(2016, 2024))
    per_zone = gdf.drop_duplicates("agribound:zone_id").set_index("agribound:zone_id")
    assert len(per_zone) == len(expected["zone_n_merged"]) == 182
    for zone, row in per_zone.iterrows():
        crops = "-".join(str(int(row[f"agribound:crop_{y}"])) for y in years)
        assert crops == expected["zone_crops"][str(zone)], zone
        assert int(row["agribound:zone_px"]) == expected["zone_px"][str(zone)], zone
        assert int(row["agribound:n_zones_merged"]) == expected["zone_n_merged"][str(zone)], zone
    # Every merge is counted once, in the zone it ends in (no zone without a polygon here
    # absorbed another region).
    assert int(per_zone["agribound:n_zones_merged"].sum()) == meta["n_merges"] == 1579


@pytest.mark.skipif(not (DATA / "ia_cdl_2016_2023.tif").is_file(), reason="fixture missing")
def test_local_crop_with_a_study_area_gives_the_zones_of_the_padded_stack(tmp_path):
    """source='local' with a study area: the local builder keeps 10 px around the study
    area's box for this engine, so the zones equal those of a stack padded by 10 px (as the
    cdl-sequence builder pads it); the crop without the margin gives other zones."""
    import geopandas as gpd
    from rasterio.windows import Window
    from shapely.geometry import box

    from agribound.composites.local import LocalCompositeBuilder

    fixture = DATA / "ia_cdl_2016_2023.tif"
    with rasterio.open(fixture) as src:
        x0, y1 = src.transform * (45, 45)
        x1, y0 = src.transform * (src.width - 45, src.height - 45)
    aoi = tmp_path / "aoi.gpkg"
    gpd.GeoDataFrame(geometry=[box(x0, y0, x1, y1)], crs="EPSG:5070").to_file(aoi)
    params = {"value_scale": "categorical", "dataset": "cdl"}

    def config(path, engine="crop-sequence"):
        return AgriboundConfig(
            source="local",
            local_tif_path=str(path),
            study_area=str(aoi),
            engine=engine,
            engine_params=params if engine == "crop-sequence" else {},
            output_path=str(tmp_path / "zones.gpkg"),
            cache_dir=str(tmp_path / "cache"),
            lulc_filter=False,
            min_field_area_m2=0.0,
        )

    def labels(path, cfg):
        gdf = CropSequenceEngine().delineate(str(path), cfg)
        with rasterio.open(gdf.attrs["engine_meta"]["label_raster"]) as src:
            return src.read(1), src.transform

    plain = LocalCompositeBuilder().build(config(fixture, engine="delineate-anything"))
    cropped = LocalCompositeBuilder().build(config(fixture))
    # The padded stack: the box's window plus 10 px on every side, as its own file.
    with rasterio.open(plain) as p, rasterio.open(fixture) as src:
        col = round((p.transform.c - src.transform.c) / src.res[0])
        row = round((src.transform.f - p.transform.f) / src.res[1])
        window = Window(col - 10, row - 10, p.width + 20, p.height + 20)
        profile = {**src.profile, "width": p.width + 20, "height": p.height + 20}
        profile["transform"] = src.window_transform(window)
        padded = tmp_path / "padded.tif"
        with rasterio.open(padded, "w", **profile) as dst:
            dst.write(src.read(window=window))
            dst.descriptions = src.descriptions
            dst.update_tags(**src.tags())
    # The padded stack is used as it is (its box plus the margin is the whole file) ...
    assert LocalCompositeBuilder().build(config(padded)) == str(padded)
    zones_cropped, t_cropped = labels(cropped, config(fixture))
    zones_padded, t_padded = labels(padded, config(padded))
    # ... and the crop of the larger file holds the same pixels and gives the same zones.
    assert t_cropped == t_padded
    assert np.array_equal(zones_cropped, zones_padded)
    # Without the margin, the focal windows at the edge are cut: the same pixels are in
    # zones, but they are grouped into other zones.
    zones_plain, _ = labels(plain, config(plain))
    inner = zones_padded[10:-10, 10:-10]
    valid = inner > 0
    assert np.array_equal(zones_plain > 0, valid)
    pairs = set(zip(zones_plain[valid].tolist(), inner[valid].tolist(), strict=True))
    n_plain, n_inner = len(np.unique(zones_plain[valid])), len(np.unique(inner[valid]))
    assert not len(pairs) == n_plain == n_inner
