"""agribound.csb_query: release names, local reads of a synthetic release zip, and one live read."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import pytest
from shapely.geometry import box

from agribound.csb_query import (
    CSB_GDAL_OPTIONS,
    CSB_YEARS,
    csb_dataset,
    csb_release,
    query_csb,
)

#: The CRS of the national releases as read from the geodatabase (EPSG:5070's parameters).
ALBERS = "EPSG:5070"


def test_release_names_follow_the_published_pattern():
    assert tuple(range(2015, 2026)) == CSB_YEARS
    first, pilot, last = csb_release(2015), csb_release(2023), csb_release(2025)
    assert first["window"] == (2008, 2015) and first["layer"] == "national0815"
    assert first["gdb"] == "NationalCSB_2008-2015_rev23/CSB0815.gdb"
    assert pilot["name"] == "NationalCSB_2016-2023_rev23"
    assert pilot["url"] == (
        "https://www.nass.usda.gov/Research_and_Science/Crop-Sequence-Boundaries/datasets/"
        "NationalCSB_2016-2023_rev23.zip"
    )
    assert last["gdb"] == "NationalCSB_2018-2025_rev23/CSB1825.gdb"
    assert last["layer"] == "national1825"
    for bad in (2014, 2026, "2023", 2023.0, True):
        with pytest.raises(ValueError):
            csb_release(bad)


def test_dataset_paths_remote_local_zip_and_as_given(tmp_path):
    path, layer, remote = csb_dataset(2023)
    assert remote and layer == "national1623"
    assert path == (
        "/vsizip//vsicurl/https://www.nass.usda.gov/Research_and_Science/Crop-Sequence-"
        "Boundaries/datasets/NationalCSB_2016-2023_rev23.zip/NationalCSB_2016-2023_rev23/"
        "CSB1623.gdb"
    )
    local = tmp_path / "NationalCSB_2016-2023_rev23.zip"
    path, _, remote = csb_dataset(2023, local)
    assert not remote
    assert path == f"/vsizip/{local.resolve()}/NationalCSB_2016-2023_rev23/CSB1623.gdb"
    mirror = "https://example.org/csb/NationalCSB_2016-2023_rev23.zip"
    path, _, remote = csb_dataset(2023, mirror)
    assert remote and path.startswith(f"/vsizip//vsicurl/{mirror}/")
    gdb = tmp_path / "CSB1623.gdb"
    assert csb_dataset(2023, gdb) == (str(gdb), "national1623", False)


def _square(x0, y0, side=1000.0):
    return box(x0, y0, x0 + side, y0 + side)


@pytest.fixture
def release_zip(tmp_path):
    """A synthetic 2016-2023 release: a File Geodatabase of five polygons inside a zip."""
    polygons = [_square(0, 0), _square(1000, 0), _square(5000, 5000), _square(500, 500, 200)]
    polygons.append(box(900, 900, 1500, 1300))
    data = {
        "CSBID": [f"19162300000{i}" for i in range(5)],
        "CSBYEARS": ["1623"] * 5,
        "CSBACRES": [round(p.area / 4046.8564224, 3) for p in polygons],
        **{f"CDL{y}": np.array([1, 5, 1, 176, 5], dtype="int16") for y in range(2016, 2024)},
        "STATEFIPS": ["19"] * 5,
    }
    frame = gpd.GeoDataFrame(data, geometry=polygons, crs=ALBERS)
    gdb = tmp_path / "build" / "CSB1623.gdb"
    gdb.parent.mkdir()
    pyogrio.write_dataframe(frame, gdb, layer="national1623", driver="OpenFileGDB")
    archive = tmp_path / "NationalCSB_2016-2023_rev23.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for member in sorted(gdb.iterdir()):
            z.write(member, f"NationalCSB_2016-2023_rev23/CSB1623.gdb/{member.name}")
    return archive


def test_query_reads_the_polygons_that_intersect_the_study_area(release_zip):
    aoi = gpd.GeoDataFrame(geometry=[box(100, 100, 1200, 800)], crs=ALBERS)
    out = query_csb(aoi, 2023, source=release_zip)
    # Whole polygons: the two big squares and the small one inside; not the far one, nor
    # the one north of the study area.
    assert sorted(out["CSBID"]) == ["191623000000", "191623000001", "191623000003"]
    assert out.crs.equals(ALBERS)
    assert {"CDL2016", "CDL2023", "CSBACRES"} <= set(out.columns)
    query = out.attrs["csb_query"]
    assert query["release"] == "NationalCSB_2016-2023_rev23" and query["window"] == [2016, 2023]
    assert query["layer"] == "national1623" and query["remote"] is False
    assert query["n_returned"] == 3 and query["n_bbox"] >= 3


def test_query_skips_polygons_that_only_reach_the_bounding_box(release_zip):
    # An L-shaped study area whose bounding box holds the polygon at (900-1500, 900-1300).
    aoi = box(0, 0, 2000, 300).union(box(1700, 0, 2000, 2000))
    out = query_csb(gpd.GeoSeries([aoi], crs=ALBERS), 2023, source=release_zip)
    assert "191623000004" not in set(out["CSBID"])
    assert out.attrs["csb_query"]["n_bbox"] > out.attrs["csb_query"]["n_returned"]


def test_query_clips_reprojects_and_selects_columns(release_zip):
    aoi = gpd.GeoDataFrame(geometry=[box(100, 100, 1200, 800)], crs=ALBERS)
    out = query_csb(aoi, 2023, source=release_zip, clip=True, columns=["CSBID", "CDL2023"],
                    dst_crs=4326)  # fmt: skip
    assert out.crs.to_epsg() == 4326
    assert set(out.columns) == {"CSBID", "CDL2023", "agribound:clipped", "geometry"}
    clipped = out.set_index("CSBID")["agribound:clipped"]
    assert clipped.to_dict() == {
        "191623000000": True,
        "191623000001": True,
        "191623000003": False,
    }
    inside = out.to_crs(ALBERS).union_all()
    assert inside.within(box(100, 100, 1200, 800).buffer(0.01))


def test_clip_keeps_the_area_of_a_polygon_that_also_touches_the_edge(tmp_path):
    """A polygon that overlaps the study area and touches its edge from outside elsewhere
    is cut to a GeometryCollection (an area and a line): its area is kept. A polygon that
    only touches the study area leaves nothing to keep."""
    from shapely.geometry import Polygon

    # A hook around the north-east corner of the study area (0-1000 m square): it covers
    # 900-1000 x 900-1000 inside it and touches its north edge between x=200 and 400.
    hook = Polygon(
        [(900, 900), (1500, 900), (1500, 1500), (200, 1500), (200, 1000), (400, 1000),
         (400, 1200), (900, 1200)]
    )  # fmt: skip
    inside, touching = box(100, 100, 300, 300), box(-500, 400, 0, 600)
    layer = tmp_path / "csb.gpkg"
    gpd.GeoDataFrame(
        {"CSBID": ["hook", "inside", "touching"]}, geometry=[hook, inside, touching], crs=ALBERS
    ).to_file(layer, layer="national1623")
    aoi = gpd.GeoSeries([box(0, 0, 1000, 1000)], crs=ALBERS)
    assert hook.intersection(aoi.iloc[0]).geom_type == "GeometryCollection"
    whole = query_csb(aoi, 2023, source=layer, layer="national1623")
    assert sorted(whole["CSBID"]) == ["hook", "inside", "touching"]
    out = query_csb(aoi, 2023, source=layer, layer="national1623", clip=True).set_index("CSBID")
    assert sorted(out.index) == ["hook", "inside"]
    assert out.geometry["hook"].geom_type == "Polygon"
    assert out.geometry["hook"].area == pytest.approx(100 * 100)
    assert out["agribound:clipped"].to_dict() == {"hook": True, "inside": False}
    assert out.attrs["csb_query"]["n_returned"] == 2


def test_query_accepts_a_wgs84_bbox_and_writes_provenance(release_zip, tmp_path):
    bbox = gpd.GeoSeries([box(100, 100, 1200, 800)], crs=ALBERS).to_crs(4326).total_bounds
    target = tmp_path / "csb.gpkg"
    out = query_csb(tuple(bbox), 2023, source=release_zip, output_path=target)
    assert len(out) >= 3 and target.exists()
    record = json.loads(Path(out.attrs["provenance_path"]).read_text())
    assert record["kind"] == "query_csb" and record["parameters"]["year"] == 2023
    assert record["query"]["release"] == "NationalCSB_2016-2023_rev23"
    assert "not field outlines" in record["data"]


def test_query_outside_the_release_returns_nothing(release_zip):
    aoi = gpd.GeoDataFrame(geometry=[box(90000, 90000, 91000, 91000)], crs=ALBERS)
    out = query_csb(aoi, 2023, source=release_zip)
    assert len(out) == 0 and out.attrs["csb_query"]["n_returned"] == 0
    with pytest.raises(ValueError, match="no geometry"):
        query_csb(gpd.GeoDataFrame(geometry=[], crs=ALBERS), 2023, source=release_zip)


def test_remote_reads_set_the_gdal_options_only_for_the_read(monkeypatch):
    import agribound.csb_query as module

    seen = {}

    def fake_info(path, layer=None):
        seen["info"] = {k: pyogrio.get_gdal_config_option(k) for k in CSB_GDAL_OPTIONS}
        seen["path"], seen["layer"] = path, layer
        return {"crs": ALBERS}

    def fake_read(path, layer=None, bbox=None, columns=None):
        seen["bbox"] = bbox
        return gpd.GeoDataFrame({"CSBID": ["x"]}, geometry=[box(0, 0, 10, 10)], crs=ALBERS)

    before = {k: pyogrio.get_gdal_config_option(k) for k in CSB_GDAL_OPTIONS}
    monkeypatch.setattr(pyogrio, "read_info", fake_info)
    monkeypatch.setattr(pyogrio, "read_dataframe", fake_read)
    out = module.query_csb(gpd.GeoSeries([box(1, 1, 5, 5)], crs=ALBERS), 2023)
    assert len(out) == 1 and out.attrs["csb_query"]["remote"] is True
    assert seen["path"].startswith("/vsizip//vsicurl/https://www.nass.usda.gov/")
    assert seen["layer"] == "national1623"
    assert seen["info"]["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
    assert seen["info"]["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] == ".zip"
    after = {k: pyogrio.get_gdal_config_option(k) for k in CSB_GDAL_OPTIONS}
    assert after == before


def test_top_level_attribute():
    import agribound

    assert agribound.query_csb is query_csb
    assert "query_csb" in agribound.__all__


@pytest.mark.network
@pytest.mark.slow
def test_live_query_of_the_2016_2023_release():
    """A 1 km box in Yuma County, Arizona (records near the start of the national file)."""
    aoi = gpd.GeoSeries([box(721000, 3606000, 722000, 3607000)], crs=32611)
    out = query_csb(aoi, 2023)
    assert len(out) > 0
    assert {"CSBID", "CSBACRES", "CDL2016", "CDL2023"} <= set(out.columns)
    assert set(out["STATEFIPS"]) == {"04"}
