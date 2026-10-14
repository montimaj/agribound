"""Live checks of the HLS selection near and away from 180 degrees (Earth Engine).

In ``NASA/HLS/HLSS30/v002``, 10,324 images of 31 MGRS tiles that straddle 180
degrees have footprint rings that run clockwise; Earth Engine reads them as the
globe except the tile, so ``filterBounds`` used to return them everywhere except
over their own tile. These tests count what :func:`agribound.composites.gee._build_hls`
and the agent's live check select now, at the areas of the investigation of
2026-10-09 (``outputs/hls_antimeridian`` in the development tree). The expected
counts are those of the Earth Engine catalogue on that day; a reprocessing of HLS
can change them.

They need Earth Engine credentials (``gee`` marker) and are excluded from the
offline suite. Run them with::

    GEE_PROJECT=my-project pytest -m gee tests/unit/test_hls_footprints_live.py -s

``AGRIBOUND_TEST_WORKLOAD_TAG`` sets the Earth Engine workload tag of the
requests. A restricted-mode report stops the session. Ten small requests; no
download.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from agribound.config import AgriboundConfig

pytestmark = pytest.mark.gee

#: Example 23's Twifo Praso square (Ghana), MGRS tile T30NXM.
TWIFO = (
    "POLYGON ((-1.4955032599320377 5.515674790288927, -1.495388865913433 5.560891430400164, "
    "-1.5405184541386092 5.56100479685173, -1.540629420016046 5.515787229421631, "
    "-1.4955032599320377 5.515674790288927))"
)
#: Example 03's Murray-Darling box (New South Wales), MGRS tile T55JGG.
MURRAY_BBOX = (149.7, -30.3, 149.85, -30.15)
MURRAY = "bbox:" + ",".join(str(v) for v in MURRAY_BBOX)
#: Open ocean at (0, 0).
OCEAN = "bbox:-0.01,-0.01,0.01,0.01"
#: A 5 km square that ends at 180 degrees (Fiji, MGRS tile T01KAB): its UTM grid reaches
#: across 180 degrees.
ENDS_AT_180 = "bbox:179.95,-16.85,180.0,-16.80"
#: A 5 km UTM square inside MGRS tile T01KAA (Vanua Levu, Fiji), 0.3 degrees west of 180.
FIJI_WEST = (
    "POLYGON ((179.72387519652057 -17.62224855615063, 179.7231981170115 -17.577104845118164, "
    "179.67613063757983 -17.5777477100631, 179.67679604888772 -17.6228931770348, "
    "179.72387519652057 -17.62224855615063))"
)


def _config(study_area, year, cloud_cover_max=20, output_path="out.gpkg"):
    return AgriboundConfig(
        source="hls",
        year=year,
        study_area=study_area,
        cloud_cover_max=cloud_cover_max,
        gee_project=os.environ.get("GEE_PROJECT") or None,
        gee_workload_tag=os.environ.get("AGRIBOUND_TEST_WORKLOAD_TAG") or None,
        output_path=output_path,
        lulc_filter=False,
    )


def _stop_if_restricted(text: str) -> None:
    if "restricted mode" in text.lower():
        pytest.exit(f"Earth Engine reported restricted mode; stopping: {text}", returncode=1)


@pytest.fixture
def live(caplog):
    """``ee`` and a check of the logs (agribound logs restricted-mode warnings)."""
    ee = pytest.importorskip("ee")
    caplog.set_level("WARNING")
    state = SimpleNamespace(ee=ee, check=lambda: _stop_if_restricted(caplog.text))
    yield state
    state.check()


def _info(obj, what="hls live check"):
    """``obj.getInfo()``; a restricted-mode warning or error stops the session."""
    from agribound.composites import gee

    try:
        with gee.ee_warning_monitor(what) as state:
            value = obj.getInfo()
    except Exception as exc:
        _stop_if_restricted(str(exc))
        raise
    if state.restricted_mode:
        _stop_if_restricted(" ".join(state.messages) or "restricted mode")
    return value


def _grid_and_region(config):
    """The export grid and the selection region, as ``GEECompositeBuilder.build`` makes them."""
    from agribound.auth import ensure_gee
    from agribound.composites import gee

    ensure_gee(config)
    geom = gee.study_area_geometry_4326(config)
    crs = gee.resolve_export_crs(config.export_crs, geom)
    grid = gee.compute_export_grid(geom, crs, gee.export_resolution_m(config))
    return grid, gee.ee_geometry(gee.grid_footprint_4326(grid))


def _spec(live, config):
    from agribound.composites import gee

    grid, region = _grid_and_region(config)
    spec = gee._build_hls(config, region)  # near 180 degrees: one listing request
    live.check()
    return spec, grid, region


def test_twifo_2022_selects_the_images_of_its_own_tile(live):
    # Before the fix: 713 (45 of T30NXM and 668 of 26 tiles at 180 degrees).
    spec, _, _ = _spec(live, _config(TWIFO, 2022, 80))
    ids = _info(spec.raw.aggregate_array("system:index"))
    assert len(ids) == 45
    assert sum(i.startswith("1_") for i in ids) == 13  # HLSL30
    assert sum(i.startswith("2_") for i in ids) == 32  # HLSS30
    assert all("T30NXM" in i for i in ids)
    assert "hlss30_images_by_id" not in spec.notes


def test_murray_2022_selects_77_images(live):
    # Example 03: AGRIBOUND_N_IMAGES 298 before the fix (221 from the tiles at 180 degrees).
    spec, _, _ = _spec(live, _config(MURRAY, 2022))
    ids = _info(spec.raw.aggregate_array("system:index"))
    assert len(ids) == 77
    assert sum(i.startswith("1_") for i in ids) == 20
    assert sum(i.startswith("2_") for i in ids) == 57
    assert all("T55JGG" in i for i in ids)


def test_open_ocean_has_no_hls_image(live):
    # Before the fix: 1,094 HLSS30 images in 2022 here, and history years 2016-2020, 2022-2025.
    from agribound.composites import gee

    spec, _, _ = _spec(live, _config(OCEAN, 2022, 100))
    query = live.ee.Dictionary({"n": spec.raw.size(), "years": gee._years_of(spec.history)})
    assert _info(query) == {"n": 0, "years": []}


def test_fiji_square_adds_the_t01kaa_images_by_id(live, monkeypatch):
    # Inside T01KAA, 29 of its 45 HLSS30 images of 2022 (cloud <= 80) have reversed
    # footprints. Before the fix the selection held 816 images: 177 real ones and 639 from
    # the other tiles at 180 degrees, without those 29.
    from agribound.composites import gee

    config = _config(FIJI_WEST, 2022, 80)
    spec, grid, region = _spec(live, config)
    assert spec.notes["hlss30_images_by_id"] == 29
    assert spec.notes["hlsl30_images_by_id"] == 0
    ids = _info(spec.raw.aggregate_array("system:index"))
    assert len(ids) == 206  # 177 + 29
    by_id = [i for i in ids if i.startswith("2_2_")]  # the second part of the HLSS30 merge
    assert len(by_id) == 29 and all(i.startswith("2_2_T01KAA_") for i in by_id)
    # The images loaded by ID supply pixels: mean observations per pixel of the grid,
    # with them and without them.
    monkeypatch.setattr(gee, "near_antimeridian", lambda *args, **kwargs: False)
    spec_a = gee._build_hls(config, region)

    def mean_observations(collection):
        counts = collection.select("B4").count()
        reducer = live.ee.Reducer.mean()
        stats = counts.reduceRegion(reducer, region, crs=grid.crs, crsTransform=grid.crs_transform)
        return stats.get("B4")

    query = {"ab": mean_observations(spec.prepared), "a": mean_observations(spec_a.prepared)}
    obs = _info(live.ee.Dictionary(query))
    assert obs["a"] == pytest.approx(90.19, abs=0.05)
    assert obs["ab"] == pytest.approx(106.47, abs=0.05)


def test_area_that_ends_at_180_selects_the_images_of_its_own_tile(live):
    # The export grid reaches about 100 m across 180 degrees. As one polygon its outline
    # spanned -180 to 180, and the selection held 20,166 images of tiles all around the
    # globe at that latitude (20,806 before the HLS fix); split at 180 degrees, it holds
    # the images of tile T01KAB, 28 of its HLSS30 images loaded by ID.
    from shapely.geometry import shape

    spec, _, region = _spec(live, _config(ENDS_AT_180, 2022, 80))
    assert shape(region.toGeoJSON()).geom_type == "MultiPolygon"
    assert spec.notes["hlss30_images_by_id"] == 28
    assert spec.notes["hlsl30_images_by_id"] == 0
    ids = _info(spec.raw.aggregate_array("system:index"))
    assert len(ids) == 76
    assert sum(i.startswith("1_") for i in ids) == 25  # HLSL30
    assert sum(i.startswith("2_") for i in ids) == 51  # HLSS30
    by_id = [i for i in ids if i.startswith("2_2_")]
    assert len(by_id) == 28 and all(i.startswith("2_2_T01KAB_") for i in by_id)
    assert all("T01KAB" in i for i in ids)


def test_agent_live_check_murray_2022(live):
    # Before the fix the live check reported 1,254 (1,094 from the tiles at 180 degrees).
    pytest.importorskip("pydantic")
    from agribound.agent.tools import _hls_live_check
    from agribound.auth import ensure_gee
    from agribound.composites import gee

    ensure_gee(_config(MURRAY, 2022))
    region = live.ee.Geometry.Rectangle(list(MURRAY_BBOX), "EPSG:4326", False)
    try:
        with gee.ee_warning_monitor("agent live check") as state:
            check = _hls_live_check(region, list(MURRAY_BBOX), 2022)
    except Exception as exc:
        _stop_if_restricted(str(exc))
        raise
    if state.restricted_mode:
        _stop_if_restricted(" ".join(state.messages) or "restricted mode")
    assert check.status == "ok"
    assert check.image_count == 160
