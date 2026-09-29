"""GeoAI window-seam merge: instances a field was split into at window edges are joined."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from agribound.engines.geoai_field import merge_window_seams, window_edges


def _write(path, arr):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        dtype="int32",
        crs="EPSG:32613",
        transform=from_origin(500000, 4000000, 1, 1),
    ) as dst:
        dst.write(arr.astype("int32"), 1)
    return str(path)


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1)


def test_window_edges_follow_geoai_placement():
    # 100 px, window 40, overlap 20: starts 0, 20, 40, 60 (last clamped to 60); ends +40.
    assert window_edges(100, 40, 20) == [20, 40, 60, 80]
    # Raster smaller than a window: one window at 0, no interior edge.
    assert window_edges(30, 40, 20) == []


def test_a_field_split_at_a_window_edge_is_joined(tmp_path):
    arr = np.zeros((80, 80), dtype=np.int32)
    arr[10:50, 10:40] = 1  # one field, cut at the window edge x = 40
    arr[10:50, 40:70] = 2
    arr[60:75, 5:15] = 3  # an unrelated field
    out = tmp_path / "merged.tif"
    info = merge_window_seams(_write(tmp_path / "inst.tif", arr), str(out), window=40, overlap=20)
    merged = _read(out)
    assert set(np.unique(merged[10:50, 10:70])) == {1}
    assert merged[65, 10] == 3
    assert info["n_instances_merged_at_seams"] == 1


def test_fields_touching_away_from_a_window_edge_stay_separate(tmp_path):
    arr = np.zeros((80, 80), dtype=np.int32)
    arr[10:50, 10:35] = 1  # touch at x = 35, not a window edge (edges 20, 40, 60)
    arr[10:50, 35:70] = 2
    out = tmp_path / "merged.tif"
    info = merge_window_seams(_write(tmp_path / "inst.tif", arr), str(out), window=40, overlap=20)
    assert set(np.unique(_read(out))) == {0, 1, 2}
    assert info["n_instances_merged_at_seams"] == 0


def test_a_short_contact_on_a_window_edge_is_not_joined(tmp_path):
    arr = np.zeros((80, 80), dtype=np.int32)
    arr[10:50, 10:40] = 1
    arr[45:50, 40:70] = 2  # touches the edge x = 40 along 5 px only
    out = tmp_path / "merged.tif"
    merge_window_seams(_write(tmp_path / "inst.tif", arr), str(out), window=40, overlap=20)
    assert set(np.unique(_read(out))) == {0, 1, 2}


def test_a_field_cut_by_two_edges_becomes_one_instance(tmp_path):
    arr = np.zeros((80, 80), dtype=np.int32)
    arr[5:40, 5:40] = 1  # four quadrants meeting at the edges x = 40 and y = 40
    arr[5:40, 40:75] = 2
    arr[40:75, 5:40] = 3
    arr[40:75, 40:75] = 4
    out = tmp_path / "merged.tif"
    info = merge_window_seams(_write(tmp_path / "inst.tif", arr), str(out), window=40, overlap=20)
    assert set(np.unique(_read(out)[5:75, 5:75])) == {1}
    assert info["n_instances_merged_at_seams"] == 3


@pytest.mark.parametrize("min_px", [16, 64])
def test_min_seam_length_is_respected(tmp_path, min_px):
    arr = np.zeros((80, 80), dtype=np.int32)
    arr[10:40, 10:40] = 1  # 30 px of shared edge at x = 40
    arr[10:40, 40:70] = 2
    out = tmp_path / "merged.tif"
    merge_window_seams(
        _write(tmp_path / "inst.tif", arr), str(out), window=40, overlap=20, min_seam_px=min_px
    )
    n_ids = len(set(np.unique(_read(out))) - {0})
    assert n_ids == (1 if min_px <= 30 else 2)


def test_a_thin_gap_across_a_window_edge_is_bridged_and_filled(tmp_path):
    arr = np.zeros((80, 80), dtype=np.int32)
    arr[10:50, 10:39] = 1  # one field, cut at x = 40 with a 1-px background gap (x = 39)
    arr[10:50, 40:70] = 2
    arr[30:35, 38:41] = 0  # a 3-px gap for 5 rows: wider than max_gap_px=2, not filled
    out = tmp_path / "merged.tif"
    info = merge_window_seams(_write(tmp_path / "inst.tif", arr), str(out), window=40, overlap=20)
    merged = _read(out)
    assert set(np.unique(merged[10:50, 10:70])) >= {1}
    assert 2 not in set(np.unique(merged))  # joined
    assert merged[20, 39] == 1  # the 1-px gap is filled
    assert merged[32, 39] == 0  # the 3-px gap is left
    assert info["n_seam_gap_pixels_filled"] == 35  # 35 rows x 1 px


def test_a_wide_gap_is_not_bridged(tmp_path):
    arr = np.zeros((80, 80), dtype=np.int32)
    arr[10:50, 10:37] = 1  # 3-px gap (x = 37..39) before the edge x = 40
    arr[10:50, 40:70] = 2
    out = tmp_path / "merged.tif"
    merge_window_seams(_write(tmp_path / "inst.tif", arr), str(out), window=40, overlap=20)
    assert set(np.unique(_read(out))) == {0, 1, 2}
