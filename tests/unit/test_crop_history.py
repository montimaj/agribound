"""agribound.postprocess.crop_history.attach_crop_history: synthetic and fixture stacks."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from agribound.postprocess.crop_history import attach_crop_history

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "crop_sequence" / "ia_cdl_2016_2023.tif"
EPSG = 32615
X0, Y0 = 500000.0, 4600000.0  # upper-left corner of the synthetic stacks
PX = 30.0


def write_stack(path, bands, *, years=(2021, 2022, 2023), tags=None, descriptions=None,
                dtype="uint8", nodata=0):  # fmt: skip
    """A (years, rows, cols) stack at 30 m with its upper-left corner at (X0, Y0)."""
    bands = np.asarray(bands)
    profile = {
        "driver": "GTiff",
        "height": bands.shape[1],
        "width": bands.shape[2],
        "count": bands.shape[0],
        "dtype": dtype,
        "crs": f"EPSG:{EPSG}",
        "transform": from_origin(X0, Y0, PX, PX),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(bands.astype(dtype))
        default_tags = {"AGRIBOUND_VALUE_SCALE": "categorical"}
        if years is not None:
            default_tags["AGRIBOUND_YEARS"] = ",".join(str(y) for y in years)
        dst.update_tags(**{**default_tags, **(tags or {})})
        for i, text in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, text)
    return path


def cells(col0, row0, col1, row1):
    """The box covering pixel columns col0..col1-1 and rows row0..row1-1."""
    return box(X0 + col0 * PX, Y0 - row1 * PX, X0 + col1 * PX, Y0 - row0 * PX)


def frame(*geoms, crs=EPSG):
    return gpd.GeoDataFrame({"name": [f"f{i}" for i in range(len(geoms))]}, geometry=list(geoms),
                            crs=crs)  # fmt: skip


@pytest.fixture
def stack(tmp_path):
    """3 years x 4 x 6 px: block A (cols 0-1, rows 0-1) and block B (cols 3-5, rows 0-3)."""
    y1 = np.zeros((4, 6), int)
    y2 = np.zeros((4, 6), int)
    y3 = np.zeros((4, 6), int)
    y1[0:2, 0:2] = [[5, 5], [7, 7]]  # a tie: the lowest code wins
    y2[0:2, 0:2] = [[9, 9], [9, 3]]
    y3[0:2, 0:2] = [[0, 0], [0, 4]]  # pixels without a code are not counted
    y1[:, 3:6], y2[:, 3:6], y3[:, 3:6] = 1, 5, 1
    y2[0, 3] = 24
    return write_stack(tmp_path / "stack.tif", [y1, y2, y3])


def test_majority_per_year_ties_to_the_lowest_code_and_zeros_not_counted(stack):
    out = attach_crop_history(frame(cells(0, 0, 2, 2), cells(3, 0, 6, 4)), stack, shares=True)
    assert out["agribound:crop_2021"].tolist() == [5, 1]
    assert out["agribound:crop_2022"].tolist() == [9, 5]
    assert out["agribound:crop_2023"].tolist() == [4, 1]
    assert out["agribound:crop_px"].tolist() == [4, 12]
    assert out["agribound:crop_share_2021"].tolist() == [0.5, 1.0]
    assert out["agribound:crop_share_2022"].tolist() == pytest.approx([0.75, 11 / 12])
    assert out["agribound:crop_share_2023"].tolist() == [1.0, 1.0]  # 1 of the 1 coded pixel
    assert out["agribound:crop_2021"].dtype == np.uint8
    # the input's columns come first and the geometry stays last
    assert list(out.columns)[:1] == ["name"] and out.columns[-1] == "geometry"


def test_the_outlines_rows_and_crs_are_unchanged(stack):
    fields = frame(cells(0, 0, 2, 2), cells(3, 0, 6, 4)).to_crs(4326)
    fields.index = [10, 20]
    out = attach_crop_history(fields, stack)
    assert out.crs == fields.crs and list(out.index) == [10, 20]
    assert out.geometry.equals(fields.geometry)
    assert out["agribound:crop_2022"].tolist() == [9, 5]  # looked up in the stack's CRS
    assert "agribound:crop_2022" not in fields.columns  # the input is not modified


def test_a_polygon_without_a_pixel_centre_takes_the_pixel_under_its_point(stack):
    tiny = box(X0 + 3 * PX + 2, Y0 - PX + 2, X0 + 3 * PX + 12, Y0 - PX + 12)  # inside pixel (0, 3)
    out = attach_crop_history(frame(tiny), stack)
    assert out[["agribound:crop_2021", "agribound:crop_2022", "agribound:crop_2023"]].iloc[
        0
    ].tolist() == [1, 24, 1]
    assert out["agribound:crop_px"].tolist() == [0]
    assert out.attrs["crop_history"]["n_from_point"] == 1


def test_polygons_outside_the_stack_or_without_geometry_get_no_code(stack, caplog):
    outside = box(X0 - 10_000, Y0 - 10_000, X0 - 9_000, Y0 - 9_000)
    fields = frame(cells(3, 0, 6, 4), outside, Polygon())
    out = attach_crop_history(fields, stack)
    assert out["agribound:crop_2021"].tolist() == [1, 0, 0]
    assert out["agribound:crop_px"].tolist() == [12, 0, 0]
    assert out.attrs["crop_history"]["n_without_code"] == 2
    with caplog.at_level(logging.WARNING):
        attach_crop_history(frame(outside), stack)
    assert "No polygon got a crop class" in caplog.text


def test_overlapping_polygons_each_count_their_own_pixels(stack):
    # The two boxes share column 3: each counts it.
    left, right = cells(2, 0, 4, 4), cells(3, 0, 6, 4)
    out = attach_crop_history(frame(left, right), stack)
    assert out["agribound:crop_px"].tolist() == [8, 12]
    assert out["agribound:crop_2021"].tolist() == [1, 1]  # left: 4 zeros, 4 ones -> 1


def test_one_label_raster_equals_polygon_by_polygon(tmp_path):
    rng = np.random.default_rng(0)
    bands = rng.integers(1, 6, size=(4, 40, 40))  # many ties among 5 codes
    bands[:, 10:14, 10:30] = 0  # some pixels without a code
    path = write_stack(tmp_path / "random.tif", bands, years=(2020, 2021, 2022, 2023))
    # Non-overlapping polygons of uneven shapes that share edges, plus thin and tilted ones.
    boxes = [(0, 0, 7, 5), (7, 0, 3, 5), (0, 5, 10, 9), (12, 3, 11, 13), (25, 25, 1, 9)]
    geoms = [cells(c, r, c + w, r + h) for c, r, w, h in boxes]
    geoms.append(Polygon([(X0 + 900, Y0 - 900), (X0 + 1100, Y0 - 850), (X0 + 1050, Y0 - 1150)]))
    fields = frame(*geoms)
    fast = attach_crop_history(fields, path, shares=True)
    # The same polygons one by one: a second, far polygon overlapping each forces that path.
    slow = []
    for g in geoms:
        decoy = frame(g, g)  # two identical polygons overlap: both are read one by one
        slow.append(attach_crop_history(decoy, path, shares=True).iloc[0])
    columns = [c for c in fast.columns if c.startswith("agribound:crop_")]
    for k, row in enumerate(slow):
        assert row[columns].tolist() == pytest.approx(fast.iloc[k][columns].tolist(), nan_ok=True)


def test_all_touched_counts_the_edge_pixels(stack):
    # A box from the middle of pixel column 3 to the middle of column 5, rows 0-3.
    half = box(X0 + 3.5 * PX, Y0 - 4 * PX, X0 + 5.5 * PX, Y0)
    centres = attach_crop_history(frame(half), stack)
    touched = attach_crop_history(frame(half), stack, all_touched=True)
    assert centres["agribound:crop_px"].tolist() == [8]  # centres of columns 4 and 5
    assert touched["agribound:crop_px"].tolist() == [12]
    assert touched.attrs["crop_history"]["all_touched"] is True


def test_years_from_the_argument_the_tag_or_the_band_descriptions(tmp_path):
    bands = np.ones((2, 3, 3), int)
    tagged = write_stack(tmp_path / "tagged.tif", bands, years=(2018, 2019))
    described = write_stack(tmp_path / "described.tif", bands, years=None,
                            descriptions=["ACI2016", "ACI2017"])  # fmt: skip
    bare = write_stack(tmp_path / "bare.tif", bands, years=None)
    field = frame(cells(0, 0, 3, 3))
    assert "agribound:crop_2019" in attach_crop_history(field, tagged).columns
    assert "agribound:crop_2017" in attach_crop_history(field, described).columns
    out = attach_crop_history(field, bare, years=[2001, 2002])
    assert out.attrs["crop_history"]["years"] == [2001, 2002]
    with pytest.raises(ValueError, match="no single year"):
        attach_crop_history(field, bare)
    with pytest.raises(ValueError, match="give 3 years for the 2 bands"):
        attach_crop_history(field, tagged, years=[1, 2, 3])


def test_refuses_imagery_fractions_and_codes_out_of_range(tmp_path):
    field = frame(cells(0, 0, 2, 2))
    imagery = write_stack(tmp_path / "s2.tif", np.ones((2, 3, 3)), years=(2022, 2023),
                          tags={"AGRIBOUND_VALUE_SCALE": "reflectance_x10000"})  # fmt: skip
    with pytest.raises(ValueError, match="not a stack of crop-type map codes"):
        attach_crop_history(field, imagery)
    fractions = write_stack(tmp_path / "frac.tif", np.full((2, 3, 3), 0.5), years=(2022, 2023),
                            dtype="float32", tags={"AGRIBOUND_VALUE_SCALE": "unknown"})  # fmt: skip
    with pytest.raises(ValueError, match="non-integer values"):
        attach_crop_history(field, fractions)
    wide = write_stack(tmp_path / "wide.tif", np.full((2, 3, 3), 300), years=(2022, 2023),
                       dtype="uint16")  # fmt: skip
    with pytest.raises(ValueError, match="class codes must be 0-255"):
        attach_crop_history(field, wide)


def test_whole_number_floats_and_the_nodata_value_are_read(tmp_path):
    bands = np.full((2, 3, 3), 7.0)
    bands[:, 0, :] = -9999.0  # nodata: not counted
    path = write_stack(tmp_path / "float.tif", bands, years=(2022, 2023), dtype="float32",
                       nodata=-9999.0)  # fmt: skip
    out = attach_crop_history(frame(cells(0, 0, 3, 3)), path, shares=True)
    assert out["agribound:crop_2023"].tolist() == [7]
    assert out["agribound:crop_share_2023"].tolist() == [1.0]
    assert out["agribound:crop_px"].tolist() == [9]


def test_existing_columns_inputs_and_crs_checks(stack):
    field = frame(cells(0, 0, 2, 2))
    once = attach_crop_history(field, stack)
    with pytest.raises(ValueError, match="already has the columns"):
        attach_crop_history(once, stack)
    twice = attach_crop_history(once, stack, prefix="raw_")
    assert twice["raw_2021"].tolist() == once["agribound:crop_2021"].tolist()
    with pytest.raises(ValueError, match="prefix"):
        attach_crop_history(field, stack, prefix="")
    with pytest.raises(TypeError):
        attach_crop_history(field.drop(columns="geometry"), stack)
    with pytest.raises(FileNotFoundError):
        attach_crop_history(field, stack.with_name("missing.tif"))
    with pytest.raises(ValueError, match="no CRS"):
        attach_crop_history(gpd.GeoDataFrame(geometry=[cells(0, 0, 2, 2)]), stack)
    series = attach_crop_history(field.geometry, stack)  # a GeoSeries is accepted
    assert series["agribound:crop_2022"].tolist() == [9]
    empty = attach_crop_history(field.iloc[:0], stack)
    assert len(empty) == 0 and "agribound:crop_2023" in empty.columns


def test_attrs_record_the_stack_legend_and_attribution(tmp_path):
    legend = {"1": "Corn", "5": "Soybeans"}
    path = write_stack(
        tmp_path / "cdl.tif",
        np.ones((3, 3, 3), int),
        tags={
            "AGRIBOUND_SOURCE": "cdl-sequence",
            "AGRIBOUND_LEGEND": json.dumps(legend),
            "AGRIBOUND_LEGEND_YEAR": "2023",
            "AGRIBOUND_ATTRIBUTION": "USDA NASS CDL (public domain)",
        },
    )
    field = frame(cells(0, 0, 3, 3))
    field.attrs["engine_meta"] = {"engine": "delineate-anything"}
    info = attach_crop_history(field, path, shares=True).attrs
    assert info["engine_meta"] == {"engine": "delineate-anything"}  # earlier attrs are kept
    record = info["crop_history"]
    assert record["source"] == "cdl-sequence" and record["years"] == [2021, 2022, 2023]
    assert record["legend"] == {1: "Corn", 5: "Soybeans"} and record["legend_year"] == "2023"
    assert record["attribution"] == "USDA NASS CDL (public domain)"
    assert record["columns"][-1] == "agribound:crop_px" and record["n_fields"] == 1


def test_fixture_square_majorities_equal_numpy_modes():
    """The pilot's Iowa square: one polygon over the whole stack gives each year's mode."""
    with rasterio.open(FIXTURE) as src:
        data, bounds, crs = src.read(), src.bounds, src.crs
    out = attach_crop_history(gpd.GeoDataFrame(geometry=[box(*bounds)], crs=crs), FIXTURE,
                              shares=True)  # fmt: skip
    for b, year in enumerate(range(2016, 2024)):
        counts = np.bincount(data[b][data[b] > 0], minlength=256)
        assert out[f"agribound:crop_{year}"].iloc[0] == counts.argmax()
        assert out[f"agribound:crop_share_{year}"].iloc[0] == pytest.approx(
            counts.max() / counts.sum()
        )
    assert out["agribound:crop_px"].iloc[0] == data.shape[1] * data.shape[2]


def test_top_level_attribute():
    import agribound

    assert agribound.attach_crop_history is attach_crop_history
    assert "attach_crop_history" in agribound.__all__
