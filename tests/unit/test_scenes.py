"""Unit tests for the scene survey and the date layers (no network, no Earth Engine).

The ranking, the reduction scale and the assembly of the survey table are pure Python.
The Earth Engine expressions are checked with recording stand-ins for ``ee``, and the date
layers numerically with a small numpy stand-in for the parts of ``ee.Image`` and
``ee.ImageCollection`` they use. The expressions themselves were run against Earth Engine
by the ``gee`` tests in ``tests/unit/test_scenes_live.py``.
"""

from __future__ import annotations

import math
import sys
import types
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, box

from agribound.composites import NoDataError, gee, scenes
from agribound.config import VALID_SOURCES, AgriboundConfig
from agribound.registry import SCENE_RANKED_SOURCES, SCENE_SURVEY_SOURCES

# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _row(date, clear, extent=1.0, **kwargs):
    return {
        "date": date,
        "acquisition": [f"Sentinel-2A_{date}_R065"],
        "extent_share": extent,
        "clear_share": clear,
        **kwargs,
    }


class TestRankAcquisitions:
    def test_sentinel2_clear_share_then_mean_cloud_score(self):
        rows = [
            _row("2020-01-07", 1.0, cs_mean=0.672),
            _row("2020-08-04", 1.0, cs_mean=0.903),
            _row("2020-04-21", 0.982, cs_mean=0.882),
            _row("2020-01-27", 1.0, cs_mean=0.679),
            _row("2020-01-12", 1.0, cs_mean=0.851),
        ]
        ranked = scenes.rank_acquisitions(rows, "sentinel2")
        # The pilot's Twifo Praso order of 2020 (the four fully clear dates by mean cs_cdf).
        assert [r["date"] for r in ranked] == [
            "2020-08-04",
            "2020-01-12",
            "2020-01-27",
            "2020-01-07",
            "2020-04-21",
        ]
        assert [r["rank"] for r in ranked] == [1, 2, 3, 4, 5]

    def test_other_sources_clear_share_then_scene_cloud(self):
        rows = [
            _row("2022-01-22", 1.0, scene_cloud_pct=0.98, cs_mean=0.1),
            _row("2022-03-03", 1.0, scene_cloud_pct=0.91, cs_mean=0.9),
            _row("2022-12-24", 0.9997, scene_cloud_pct=49.94),
        ]
        for source in ("landsat", "landsat-pan", "hls", "spot", "spot-pan"):
            ranked = scenes.rank_acquisitions(rows, source)
            assert [r["date"] for r in ranked] == ["2022-03-03", "2022-01-22", "2022-12-24"]

    def test_rows_without_coverage_are_dropped(self):
        rows = [
            _row("2020-01-01", 0.0, extent=0.0),
            _row("2020-01-02", None, extent=None),
            _row("2020-01-03", 0.0, extent=0.25),  # covered but fully cloudy: kept
        ]
        ranked = scenes.rank_acquisitions(rows, "sentinel2")
        assert [r["date"] for r in ranked] == ["2020-01-03"]
        assert ranked[0]["rank"] == 1

    def test_missing_values_sort_last(self):
        rows = [
            _row("2020-01-01", None, cs_mean=0.9),
            _row("2020-01-02", 0.5, cs_mean=None),
            _row("2020-01-03", 0.5, cs_mean=0.2),
            _row("2020-01-04", float("nan"), cs_mean=0.9),
        ]
        ranked = scenes.rank_acquisitions(rows, "sentinel2")
        assert [r["date"] for r in ranked] == [
            "2020-01-03",
            "2020-01-02",
            "2020-01-01",
            "2020-01-04",
        ]
        rows = [
            _row("2020-01-01", 1.0, scene_cloud_pct=None),
            _row("2020-01-02", 1.0, scene_cloud_pct=100.0),
        ]
        assert [r["date"] for r in scenes.rank_acquisitions(rows, "landsat")] == [
            "2020-01-02",
            "2020-01-01",
        ]

    def test_ties_by_date_then_acquisition(self):
        rows = [
            {**_row("2020-02-01", 1.0, cs_mean=0.5), "acquisition": ["b"]},
            {**_row("2020-01-01", 1.0, cs_mean=0.5), "acquisition": ["z"]},
            {**_row("2020-01-01", 1.0, cs_mean=0.5), "acquisition": ["a"]},
        ]
        ranked = scenes.rank_acquisitions(rows, "sentinel2")
        assert [(r["date"], r["acquisition"]) for r in ranked] == [
            ("2020-01-01", "a"),
            ("2020-01-01", "z"),
            ("2020-02-01", "b"),
        ]

    def test_acquisition_lists_are_joined_and_rows_copied(self):
        rows = [
            {
                **_row("2022-01-22", 1.0),
                "acquisition": ["LANDSAT_7_2022-01-22_P194", "LANDSAT_9_2022-01-22_P194"],
            }
        ]
        (ranked,) = scenes.rank_acquisitions(rows, "landsat")
        assert ranked["acquisition"] == "LANDSAT_7_2022-01-22_P194+LANDSAT_9_2022-01-22_P194"
        assert "rank" not in rows[0] and isinstance(rows[0]["acquisition"], list)

    def test_rank_rules_name_the_order(self):
        assert scenes.RANK_RULES["sentinel2"].startswith("clear_share desc, cs_mean desc")
        assert scenes.RANK_RULES["default"].startswith("clear_share desc, scene_cloud_pct asc")


class TestSurveyScale:
    def test_scales(self):
        assert scenes.survey_scale_m(5_000.0**2) == 60.0
        assert scenes.survey_scale_m(50_000.0**2) == 100.0
        assert scenes.survey_scale_m(0.0) == 60.0
        assert scenes.survey_scale_m(1.0) == 60.0

    @pytest.mark.parametrize("side_km", [1, 5, 31, 32, 50, 77, 120, 500])
    def test_at_most_the_pixel_cap_and_a_multiple_of_10_m(self, side_km):
        area = (side_km * 1000.0) ** 2
        scale = scenes.survey_scale_m(area)
        assert area / scale**2 <= scenes.SURVEY_MAX_PIXELS + 1e-6
        assert scale % 10 == 0 and scale >= scenes.SURVEY_MIN_SCALE_M
        # the next finer multiple of 10 m (if not below the minimum) would exceed the cap
        if scale > scenes.SURVEY_MIN_SCALE_M:
            assert area / (scale - 10.0) ** 2 > scenes.SURVEY_MAX_PIXELS


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def test_registry_sources():
    assert SCENE_SURVEY_SOURCES == (
        "sentinel2",
        "landsat",
        "landsat-pan",
        "hls",
        "spot",
        "spot-pan",
    )
    assert SCENE_RANKED_SOURCES == ("sentinel2", "landsat", "landsat-pan", "hls")
    assert set(SCENE_SURVEY_SOURCES) <= set(gee._COLLECTION_BUILDERS)
    for name in ("SCENE_SURVEY_SOURCES", "SCENE_RANKED_SOURCES"):
        from agribound import registry

        assert name in registry.__all__


REFUSED = sorted(set(VALID_SOURCES) - set(SCENE_SURVEY_SOURCES))


def _cfg(source="sentinel2", year=2020, **kwargs):
    kwargs.setdefault("study_area", "bbox:-1.5406,5.5157,-1.4954,5.5610")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SPOT is a restricted source
        return AgriboundConfig(
            source=source,
            year=year,
            gee_project="p",
            lulc_filter=False,
            **kwargs,
        )


def _as_source(config, source):
    """*config* with its source changed after validation (sources with other requirements)."""
    config.source = source
    return config


def test_refused_sources_cover_the_other_sources():
    assert {"naip", "sentinel1", "umbra", "capella", "local", "usgs-naip-plus"} <= set(REFUSED)
    assert {"google-embedding", "tessera-embedding"} <= set(REFUSED)


@pytest.mark.parametrize("source", REFUSED)
def test_survey_refuses_other_sources_before_earth_engine(monkeypatch, source):
    def no_gee(config):
        raise AssertionError("Earth Engine must not be initialised")

    monkeypatch.setattr("agribound.auth.ensure_gee", no_gee)
    config = _as_source(_cfg(), source)
    with pytest.raises(
        ValueError, match="The scene survey supports the sources sentinel2,"
    ) as info:
        scenes.survey_scenes(config)
    assert f"source={source!r}" in str(info.value)
    if source == "naip":
        assert "mosaicked" in str(info.value)
    if source == "sentinel1":
        assert "no scene cloud filter or pixel cloud mask" in str(info.value)
    if source in ("umbra", "capella"):
        assert "select their collects themselves" in str(info.value)


@pytest.mark.parametrize("source", REFUSED)
def test_date_layers_refuse_other_sources_before_earth_engine(monkeypatch, source):
    monkeypatch.setattr(
        "agribound.auth.ensure_gee", lambda config: pytest.fail("Earth Engine was initialised")
    )
    with pytest.raises(ValueError, match="Date layers are available for the sources sentinel2,"):
        scenes.build_date_layers(_as_source(_cfg(), source))


# ---------------------------------------------------------------------------
# Recording stand-in for the expressions
# ---------------------------------------------------------------------------


class Rec:
    """Records every call made on an Earth Engine object chain."""

    def __init__(self, name, log):
        self.name = name
        self.log = log

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)

        def call(*args, **kwargs):
            self.log.append((self.name, attr, args, kwargs))
            return Rec(f"{self.name}.{attr}", self.log)

        return call


@pytest.fixture
def rec_ee(monkeypatch):
    log = []
    module = types.ModuleType("ee")

    def wrap(kind):
        def make(*args):
            log.append((kind, "new", args, {}))
            return args[0] if args and isinstance(args[0], Rec) else Rec(kind, log)

        return make

    module.Image = wrap("Image")
    module.Image.constant = lambda value: Rec("Image.constant", log)
    module.Image.cat = lambda images: (
        log.append(("Image", "cat", (images,), {})) or Rec("Image.cat", log)
    )
    module.Number = wrap("Number")
    module.String = wrap("String")
    module.Date = wrap("Date")
    module.List = wrap("List")
    module.Dictionary = wrap("Dictionary")
    module.ImageCollection = lambda cid: Rec(f"IC({cid})", log)
    module.Algorithms = types.SimpleNamespace(
        If=lambda *args: log.append(("Algorithms", "If", args, {})) or Rec("If", log)
    )
    module.Reducer = types.SimpleNamespace(
        mean=lambda: "MEAN", min=lambda: "MIN", max=lambda: "MAX"
    )
    module.Filter = types.SimpleNamespace(
        eq=lambda prop, value: ("eq", prop, value),
        inList=lambda prop, values: ("inList", prop, values),
    )
    monkeypatch.setitem(sys.modules, "ee", module)
    return log


def _gets(log):
    return [args[0] for _, attr, args, _ in log if attr == "get"]


@pytest.mark.parametrize(
    ("source", "properties", "band"),
    [
        ("sentinel2", ["SENSING_ORBIT_NUMBER", "SPACECRAFT_NAME"], None),
        ("landsat", ["WRS_PATH", "SPACECRAFT_ID"], None),
        ("landsat-pan", ["WRS_PATH", "SPACECRAFT_ID"], None),
        ("hls", [], "B8A"),
        ("spot", ["system:index"], None),
        ("spot-pan", ["system:index"], None),
    ],
)
def test_acquisition_key_properties(rec_ee, source, properties, band):
    scenes.acquisition_key(source, Rec("img", rec_ee))
    gets = _gets(rec_ee)
    assert "system:time_start" in gets  # the UTC date of every key
    assert [g for g in gets if g != "system:time_start"] == properties
    contains = [args for _, attr, args, _ in rec_ee if attr == "contains"]
    assert contains == ([(band,)] if band else [])
    formats = [args[0] for _, attr, args, _ in rec_ee if attr == "format"]
    assert "YYYY-MM-dd" in formats
    if source in ("sentinel2", "landsat", "landsat-pan"):
        assert "%03d" in formats


def _fake_spec(source="sentinel2", bands=("B4", "B8"), cloud="CLOUDY_PIXEL_PERCENTAGE", log=None):
    log = [] if log is None else log
    return gee.CollectionSpec(
        raw=Rec("RAW", log),
        prepared=Rec("PREPARED", log),
        history="HISTORY",
        bands=list(bands),
        resolution_m=10.0,
        dtype="float32",
        collections=["X"],
        notes={"cloud_mask": "unit-test mask"},
        scene_cloud_property=cloud,
    )


@pytest.mark.parametrize("source", ["sentinel2", "landsat"])
def test_survey_expression_structure(rec_ee, source):
    spec = _fake_spec(source, log=rec_ee)
    scenes.survey_expression(spec, source, "AREA", "EPSG:32630", 60.0, 0.6)
    links = [args for name, attr, args, _ in rec_ee if attr == "linkCollection"]
    assert len(links) == (source == "sentinel2")
    # One mapped function per date: run it on a recorded date.
    ((_, _, (per_date,), _),) = [
        e for e in rec_ee if e[1] == "map" and e[0].endswith("distinct.sort")
    ]
    per_date(Rec("DATE", rec_ee))
    filters = [args[0] for name, attr, args, _ in rec_ee if attr == "filter"]
    assert ("eq", "agribound_date", filters[0][2]) == filters[0]
    assert filters[1][:2] == ("inList", "system:index")  # prepared images by raw index
    (reduce,) = [kw for _, attr, _, kw in rec_ee if attr == "reduceRegion"]
    assert reduce == {
        "reducer": "MEAN",
        "geometry": "AREA",
        "crs": "EPSG:32630",
        "scale": 60.0,
        "maxPixels": 1e9,
    }
    renamed = [args[0] for _, attr, args, _ in rec_ee if attr == "rename"]
    expected = {"clear_share", "extent_share"}
    if source == "sentinel2":
        expected |= {"scl_clear_share", "cs_clear_share", "cs_mean", "aot_mean"}
        assert ("gte", (0.6,)) in [(attr, args) for _, attr, args, _ in rec_ee]
    assert set(renamed) == expected
    assert ("CLOUDY_PIXEL_PERCENTAGE",) in [
        args for _, attr, args, _ in rec_ee if attr == "aggregate_max"
    ]


# ---------------------------------------------------------------------------
# survey_scenes with the Earth Engine layer stubbed
# ---------------------------------------------------------------------------

#: Rows as the survey expression returns them (Twifo Praso 2020, abridged).
S2_ROWS = [
    {
        "date": "2020-01-07",
        "acquisition": ["Sentinel-2B_2020-01-07_R065"],
        "image_ids": ["COPERNICUS/S2_SR_HARMONIZED/20200107T102319_20200107T103621_T30NXM"],
        "n_images": 1,
        "scene_cloud_pct": 0.016758,
        "extent_share": 1,
        "clear_share": 1.0,
        "cs_clear_share": 1.0,
        "scl_clear_share": 1.0,
        "cs_mean": 0.6719,
        "aot_mean": 1.0804,
    },
    {
        "date": "2020-08-04",
        "acquisition": ["Sentinel-2B_2020-08-04_R065"],
        "image_ids": ["COPERNICUS/S2_SR_HARMONIZED/20200804T101559_20200804T103905_T30NXM"],
        "n_images": 1,
        "scene_cloud_pct": 10.626962,
        "extent_share": 1,
        "clear_share": 1.0,
        "cs_clear_share": 1.0,
        "scl_clear_share": 1.0,
        "cs_mean": 0.9030,
        "aot_mean": 0.0556,
    },
    {
        "date": "2020-03-01",
        "acquisition": ["Sentinel-2A_2020-03-01_R065"],
        "image_ids": ["COPERNICUS/S2_SR_HARMONIZED/a"],
        "n_images": 1,
        "scene_cloud_pct": 75.0,
        "extent_share": 1,
        "clear_share": 0.0,
        "cs_clear_share": 0.0,
        "scl_clear_share": 0.4,
        "cs_mean": 0.21,
    },  # aot_mean missing: no clear pixel
    {
        "date": "2020-05-05",
        "acquisition": ["Sentinel-2A_2020-05-05_R108"],
        "image_ids": ["COPERNICUS/S2_SR_HARMONIZED/b"],
        "n_images": 1,
        "scene_cloud_pct": 5.0,
        "extent_share": 0,
        "clear_share": 0,
    },  # the image misses the study area
]


@pytest.fixture
def stub_survey(monkeypatch):
    state = {"gee_calls": 0, "rows": [dict(r) for r in S2_ROWS], "expressions": []}

    def fake_ensure(config):
        state["gee_calls"] += 1

    monkeypatch.setattr("agribound.auth.ensure_gee", fake_ensure)
    monkeypatch.setattr(gee, "ee_geometry", lambda geom: ("EE", geom))

    def fake_builder(config, region):
        state["region"] = region
        return _fake_spec(config.source)

    for source in SCENE_SURVEY_SOURCES:
        monkeypatch.setitem(gee._COLLECTION_BUILDERS, source, fake_builder)

    class Expression:
        def getInfo(self):  # noqa: N802 - ee API name
            if isinstance(state["rows"], Exception):
                raise state["rows"]
            return state["rows"]

    def fake_expression(spec, source, area, crs, scale_m, cs_threshold):
        state["expressions"].append(
            {"source": source, "area": area, "crs": crs, "scale_m": scale_m, "cs": cs_threshold}
        )
        return Expression()

    monkeypatch.setattr(scenes, "survey_expression", fake_expression)
    return state


class TestSurveyScenes:
    def test_table_and_attrs(self, stub_survey):
        config = _cfg(cloud_cover_max=80, s2_cloud_mask="cloud_score_plus")
        table = scenes.survey_scenes(config)
        assert list(table.columns) == [
            *scenes.SURVEY_COLUMNS,
            *scenes.S2_SURVEY_COLUMNS,
            "date_range",
        ]
        assert table["date"].tolist() == ["2020-08-04", "2020-01-07", "2020-03-01"]
        assert table["rank"].tolist() == [1, 2, 3]
        assert table["date_range"].tolist() == [
            ("2020-08-04", "2020-08-04"),
            ("2020-01-07", "2020-01-07"),
            ("2020-03-01", "2020-03-01"),
        ]
        assert table.loc[0, "acquisition"] == "Sentinel-2B_2020-08-04_R065"
        assert table.loc[0, "image_ids"] == [
            "COPERNICUS/S2_SR_HARMONIZED/20200804T101559_20200804T103905_T30NXM"
        ]
        assert table.loc[0, "aot_mean"] == pytest.approx(0.0556)
        assert table.loc[2, "aot_mean"] is None or math.isnan(table.loc[2, "aot_mean"])
        attrs = table.attrs
        assert attrs["source"] == "sentinel2" and attrs["year"] == 2020
        assert attrs["window"] == ("2020-01-01", "2021-01-01")
        assert attrs["cloud_cover_max"] == 80
        assert attrs["cloud_mask"] == "unit-test mask"
        assert attrs["n_acquisitions"] == 4 and attrs["n_without_coverage"] == 1
        assert attrs["rank_rule"] == scenes.RANK_RULES["sentinel2"]
        assert attrs["survey_version"] == scenes.SCENE_SURVEY_VERSION
        assert attrs["scale_m"] == 60.0 and attrs["crs"] == "EPSG:32630"
        assert attrs["reduced_over"] == "the study area"
        assert "then mean Cloud Score+ cs_cdf" in attrs["ranked_by"]
        (expression,) = stub_survey["expressions"]
        assert expression["cs"] == 0.6 and expression["scale_m"] == 60.0
        assert expression["crs"] == "EPSG:32630"
        # The images are selected over the export grid's outline, as for the composite,
        # and reduced over the study area.
        geom = gee.study_area_geometry_4326(config)
        grid = gee.compute_export_grid(geom, gee.resolve_export_crs("utm", geom), 10.0)
        assert stub_survey["region"][0] == "EE"
        assert stub_survey["region"][1].equals(gee.grid_footprint_4326(grid))
        assert not stub_survey["region"][1].equals(geom)
        # A box: simplifying it (in UTM and back) changes it by rounding only.
        assert expression["area"][1].symmetric_difference(geom).area < 1e-12

    def test_one_day_window_of_a_row_is_an_ordinary_configuration(self, stub_survey):
        config = _cfg(cloud_cover_max=80, s2_cloud_mask="cloud_score_plus")
        top = scenes.survey_scenes(config).iloc[0]
        one = config.merged(date_range=top["date_range"])
        assert gee.date_window(one) == ("2020-08-04", "2020-08-05")
        assert one.composite_method == "median"

    def test_other_source_columns_and_ranking(self, stub_survey):
        stub_survey["rows"] = [
            {
                "date": "2022-01-22",
                "acquisition": ["LANDSAT_7_2022-01-22_P194", "LANDSAT_9_2022-01-22_P194"],
                "image_ids": ["LANDSAT/LC09/C02/T1_L2/x", "LANDSAT/LE07/C02/T1_L2/y"],
                "n_images": 2,
                "scene_cloud_pct": 0.98,
                "extent_share": 1.0,
                "clear_share": 1.0,
            },
            {
                "date": "2022-03-03",
                "acquisition": ["LANDSAT_8_2022-03-03_P194"],
                "image_ids": ["LANDSAT/LC08/C02/T1_L2/z"],
                "n_images": 1,
                "scene_cloud_pct": 0.91,
                "extent_share": 1.0,
                "clear_share": 1.0,
            },
        ]
        table = scenes.survey_scenes(_cfg("landsat", year=2022))
        assert list(table.columns) == [*scenes.SURVEY_COLUMNS, "date_range"]
        assert table["date"].tolist() == ["2022-03-03", "2022-01-22"]
        assert table.loc[1, "acquisition"] == (
            "LANDSAT_7_2022-01-22_P194+LANDSAT_9_2022-01-22_P194"
        )
        assert table.loc[1, "n_images"] == 2
        assert table.attrs["rank_rule"] == scenes.RANK_RULES["default"]
        assert "then scene cloud cover" in table.attrs["ranked_by"]

    def test_spot_is_ranked_by_footprint(self, stub_survey):
        stub_survey["rows"] = []
        table = scenes.survey_scenes(_cfg("spot", cloud_cover_max=15))
        assert table.empty and list(table.columns)[-1] == "date_range"
        assert "no pixel mask" in table.attrs["ranked_by"]
        assert table.attrs["cloud_mask"] == "unit-test mask"
        assert table.attrs["n_acquisitions"] == 0

    def test_scale_m(self, stub_survey):
        table = scenes.survey_scenes(_cfg(), scale_m=250)
        assert table.attrs["scale_m"] == 250.0
        assert stub_survey["expressions"][0]["scale_m"] == 250.0

    @pytest.mark.parametrize("bad", [0, 0.0, -5, float("nan"), float("inf")])
    def test_bad_scale_m_is_refused_before_earth_engine(self, monkeypatch, bad):
        monkeypatch.setattr(
            "agribound.auth.ensure_gee", lambda config: pytest.fail("Earth Engine initialised")
        )
        with pytest.raises(ValueError, match="scale_m must be > 0"):
            scenes.survey_scenes(_cfg(), scale_m=bad)

    def test_default_scale_keeps_the_grid_outline_under_the_pixel_cap(self, stub_survey, tmp_path):
        # 1,600 fields of about 100 m spread over a 0.5-degree box: more than 5,000
        # vertices once simplified, so the survey reduces over the export grid's outline
        # (about 2,500 km²), which needs 110 m to stay at 250,000 pixels per date.
        import geopandas as gpd

        polys = [
            box(-1.6 + i * 0.0125, 5.3 + j * 0.0125, -1.6 + i * 0.0125 + 0.001,
                5.3 + j * 0.0125 + 0.001)
            for i in range(40) for j in range(40)
        ]  # fmt: skip
        path = tmp_path / "fields.geojson"
        gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326").to_file(path, driver="GeoJSON")
        config = _cfg(study_area=str(path))
        table = scenes.survey_scenes(config)
        (expression,) = stub_survey["expressions"]
        outline = expression["area"][1]
        assert table.attrs["reduced_over"].startswith("the export grid's outline")
        geom = gee.study_area_geometry_4326(config)
        grid = gee.compute_export_grid(geom, gee.resolve_export_crs("utm", geom), 10.0)
        assert outline.equals(gee.grid_footprint_4326(grid))
        outline_m2 = scenes._area_m2(outline)
        assert scenes.survey_scale_m(scenes._area_m2(geom)) == 60.0  # the fields alone
        assert expression["scale_m"] == table.attrs["scale_m"] == 110.0
        assert outline_m2 / 110.0**2 <= scenes.SURVEY_MAX_PIXELS < outline_m2 / 100.0**2
        # A scale given by the caller is used as it is.
        stub_survey["expressions"].clear()
        assert scenes.survey_scenes(config, scale_m=60).attrs["scale_m"] == 60.0
        assert stub_survey["expressions"][0]["scale_m"] == 60.0

    @pytest.mark.parametrize(
        "message", ["Computation timed out.", "User memory limit exceeded.", "Too many pixels"]
    )
    def test_time_and_memory_errors_get_advice(self, stub_survey, message):
        stub_survey["rows"] = RuntimeError(message)
        with pytest.raises(RuntimeError, match="Use a shorter date window") as info:
            scenes.survey_scenes(_cfg())
        assert "a larger scale_m (this survey used 60 m)" in str(info.value)
        assert message in str(info.value)

    def test_other_errors_are_raised_unchanged(self, stub_survey):
        stub_survey["rows"] = KeyError("boom")
        with pytest.raises(KeyError, match="boom"):
            scenes.survey_scenes(_cfg())


class WindowCollection:
    """Stand-in for a collection: the date window (start, end exclusive) of its images."""

    def __init__(self, window):
        self.window = tuple(window)

    def filterDate(self, start, end):  # noqa: N802 - ee API name
        return WindowCollection((max(self.window[0], start), min(self.window[1], end)))


@pytest.fixture
def stub_window_survey(stub_survey, monkeypatch):
    """stub_survey whose request covers its own window: S2_ROWS of that window, or an error.

    The builder's collections are :class:`WindowCollection` objects. ``state["fail"]``
    maps a window (start, end exclusive) to the message its request raises;
    ``state["windows"]`` records the window of every request and ``state["builds"]``
    the window of every collection built.
    """
    state = stub_survey
    state.update({"fail": {}, "windows": [], "builds": []})

    def fake_builder(config, region):
        window = gee.date_window(config)
        state["builds"].append(window)
        spec = _fake_spec(config.source)
        spec.raw, spec.prepared = WindowCollection(window), WindowCollection(window)
        return spec

    for source in SCENE_SURVEY_SOURCES:
        monkeypatch.setitem(gee._COLLECTION_BUILDERS, source, fake_builder)

    class Expression:
        def __init__(self, spec):
            assert spec.prepared.window == spec.raw.window  # both narrowed alike
            self.window = spec.raw.window

        def getInfo(self):  # noqa: N802 - ee API name
            state["windows"].append(self.window)
            if self.window in state["fail"]:
                raise RuntimeError(state["fail"][self.window])
            start, end = self.window
            return [dict(r) for r in S2_ROWS if start <= r["date"] < end]

    monkeypatch.setattr(scenes, "survey_expression", lambda spec, *args: Expression(spec))
    return state


YEAR_2020 = ("2020-01-01", "2021-01-01")


class TestSurveyByMonth:
    def test_month_windows(self):
        months = scenes.month_windows(YEAR_2020)
        assert len(months) == 12
        assert months[0] == ("2020-01-01", "2020-01-31")
        assert months[1] == ("2020-02-01", "2020-02-29")
        assert months[-1] == ("2020-12-01", "2020-12-31")
        assert scenes.month_windows(("2020-03-15", "2020-05-10")) == [
            ("2020-03-15", "2020-03-31"),
            ("2020-04-01", "2020-04-30"),
            ("2020-05-01", "2020-05-09"),
        ]
        assert scenes.month_windows(("2020-12-20", "2021-01-10")) == [
            ("2020-12-20", "2020-12-31"),
            ("2021-01-01", "2021-01-09"),
        ]
        assert scenes.month_windows(("2020-08-04", "2020-08-05")) == [("2020-08-04", "2020-08-04")]

    def test_one_request_when_the_window_fits(self, stub_window_survey):
        table = scenes.survey_scenes(_cfg(cloud_cover_max=80, s2_cloud_mask="cloud_score_plus"))
        assert stub_window_survey["windows"] == [YEAR_2020]
        assert table.attrs["n_requests"] == 1
        assert table["date"].tolist() == ["2020-08-04", "2020-01-07", "2020-03-01"]

    @pytest.mark.parametrize("message", ["User memory limit exceeded.", "Computation timed out."])
    def test_months_give_the_rows_of_the_whole_window(self, stub_window_survey, message):
        config = _cfg(cloud_cover_max=80, s2_cloud_mask="cloud_score_plus")
        whole = scenes.survey_scenes(config)
        stub_window_survey["windows"].clear()
        stub_window_survey["fail"][YEAR_2020] = message
        by_month = scenes.survey_scenes(config)
        months = [
            gee.date_window(config.merged(date_range=month))
            for month in scenes.month_windows(YEAR_2020)
        ]
        assert months[0] == ("2020-01-01", "2020-02-01") and len(months) == 12
        assert stub_window_survey["windows"] == [YEAR_2020, *months]
        # One collection per survey, of the whole window; the months narrow it by date.
        assert stub_window_survey["builds"] == [YEAR_2020, YEAR_2020]
        assert by_month.attrs["n_requests"] == 13
        assert by_month.drop(columns=["image_ids"]).equals(whole.drop(columns=["image_ids"]))
        assert by_month["image_ids"].tolist() == whole["image_ids"].tolist()
        for key in ("n_acquisitions", "n_without_coverage", "window", "cloud_mask"):
            assert by_month.attrs[key] == whole.attrs[key], key

    def test_too_many_pixels_is_not_split(self, stub_window_survey):
        stub_window_survey["fail"][YEAR_2020] = "Too many pixels in the region."
        with pytest.raises(RuntimeError, match="Use a shorter date window"):
            scenes.survey_scenes(_cfg())
        assert stub_window_survey["windows"] == [YEAR_2020]

    def test_a_month_that_fails_raises_with_advice(self, stub_window_survey):
        stub_window_survey["fail"][YEAR_2020] = "User memory limit exceeded."
        stub_window_survey["fail"][("2020-03-01", "2020-04-01")] = "User memory limit exceeded."
        with pytest.raises(RuntimeError, match=r"sentinel2 2020 \(2020-03-01\.\.2020-03-31\)"):
            scenes.survey_scenes(_cfg())
        assert stub_window_survey["windows"][-1] == ("2020-03-01", "2020-04-01")
        assert len(stub_window_survey["windows"]) == 4  # the year, January to March

    def test_a_window_of_one_month_is_not_split(self, stub_window_survey):
        window = ("2020-03-01", "2020-04-01")
        stub_window_survey["fail"][window] = "User memory limit exceeded."
        with pytest.raises(RuntimeError, match="a larger scale_m"):
            scenes.survey_scenes(_cfg(date_range=("2020-03-01", "2020-03-31")))
        assert stub_window_survey["windows"] == [window]


# ---------------------------------------------------------------------------
# The month-by-month survey with the real Landsat collection builders
# ---------------------------------------------------------------------------


def _utc_ms(day: str) -> int:
    import datetime

    when = datetime.datetime.fromisoformat(day).replace(tzinfo=datetime.UTC)
    return int(when.timestamp() * 1000)


class ArchiveImage:
    """An image of the in-memory archive: properties only; image methods return it."""

    def __init__(self, props):
        self.props = dict(props)

    def get(self, name):
        return self.props.get(name)

    def __getattr__(self, name):  # select, bitwiseAnd, eq, updateMask, multiply, ...
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self


class ArchiveCollection:
    def __init__(self, images):
        self.images = list(images)

    def filterBounds(self, region):  # noqa: N802 - ee API name
        return self

    def filter(self, predicate):
        return ArchiveCollection([i for i in self.images if predicate(i)])

    def filterDate(self, start, end):  # noqa: N802 - ee API name
        lo, hi = _utc_ms(start), _utc_ms(end)
        return ArchiveCollection(
            [i for i in self.images if lo <= i.props["system:time_start"] < hi]
        )

    def map(self, fn):
        return ArchiveCollection([fn(i) for i in self.images])

    def merge(self, other):
        return ArchiveCollection(self.images + other.images)


def _landsat_archive(years=(1984, 1999, 2013, 2021, 2024)):
    """Images on the 10th and 25th of each month of *years*, within each mission's record.

    Keyed by the collection IDs of ``landsat`` (C2 L2) and ``landsat-pan`` (TOA).
    """
    import datetime

    spacecraft = {"LT05": "LANDSAT_5", "LE07": "LANDSAT_7", "LC08": "LANDSAT_8"}
    spacecraft["LC09"] = "LANDSAT_9"
    archive = {}
    for mission, (cid, first, last) in gee.LANDSAT_COLLECTIONS.items():
        props = []
        for year in years:
            for month in range(1, 13):
                for day in (10, 25):
                    date = datetime.date(year, month, day).isoformat()
                    if first <= date and (last is None or date <= last):
                        props.append(
                            {
                                "system:time_start": _utc_ms(date) + 10 * 3_600_000,
                                "system:index": f"{mission}_{date}",
                                "SPACECRAFT_ID": spacecraft[mission],
                                "CLOUD_COVER": 5.0,
                            }
                        )
        archive[cid] = [ArchiveImage(p) for p in props]
        archive[cid.replace("T1_L2", "T1_TOA")] = [ArchiveImage(p) for p in props]
    return archive


@pytest.fixture
def archive_ee(monkeypatch):
    """The real ``landsat`` and ``landsat-pan`` builders on an in-memory archive.

    The survey request returns one row per UTC date of the raw images of the spec it
    is given (the acquisition label ``<SPACECRAFT_ID>_<date>``); with
    ``state["fail_first"]`` the first request raises "User memory limit exceeded.".
    ``state["builds"]`` records the window of every collection built.
    """
    import datetime

    archive = _landsat_archive()
    module = types.ModuleType("ee")
    module.ImageCollection = lambda cid: ArchiveCollection(archive.get(cid, []))
    module.Image = lambda image: image
    module.Filter = types.SimpleNamespace(
        lte=lambda prop, value: lambda image: (image.get(prop) or 0) <= value
    )
    monkeypatch.setitem(sys.modules, "ee", module)
    monkeypatch.setattr("agribound.auth.ensure_gee", lambda config: None)
    monkeypatch.setattr(gee, "ee_geometry", lambda geom: ("EE", geom))
    state = {"builds": [], "requests": 0, "fail_first": False}

    def counted(build):
        def builder(config, region):
            state["builds"].append(gee.date_window(config))
            return build(config, region)

        return builder

    for source in ("landsat", "landsat-pan"):
        monkeypatch.setitem(
            gee._COLLECTION_BUILDERS, source, counted(gee._COLLECTION_BUILDERS[source])
        )

    class Expression:
        def __init__(self, spec):
            assert len(spec.prepared.images) == len(spec.raw.images)
            self.images = spec.raw.images

        def getInfo(self):  # noqa: N802 - ee API name
            state["requests"] += 1
            if state["fail_first"] and state["requests"] == 1:
                raise RuntimeError("User memory limit exceeded.")
            by_date = {}
            for image in self.images:
                seconds = image.props["system:time_start"] / 1000
                day = datetime.datetime.fromtimestamp(seconds, datetime.UTC).date().isoformat()
                by_date.setdefault(day, []).append(image)
            return [
                {
                    "date": day,
                    "acquisition": sorted({f"{i.props['SPACECRAFT_ID']}_{day}" for i in images}),
                    "image_ids": sorted(i.props["system:index"] for i in images),
                    "n_images": len(images),
                    "scene_cloud_pct": max(i.props["CLOUD_COVER"] for i in images),
                    "extent_share": 1.0,
                    "clear_share": 1.0,
                }
                for day, images in sorted(by_date.items())
            ]

    monkeypatch.setattr(scenes, "survey_expression", lambda spec, *args: Expression(spec))
    return state


class TestSurveyByMonthRealBuilders:
    """The month-by-month survey keeps the images the whole window's composite uses.

    Rebuilding each month's collection would apply the rules that depend on the window
    to each month: for ``landsat-pan``, a valid configuration would fail where its
    missions' records miss the month, and with ``"auto"`` in 2013 the months before
    Landsat 8 (2013-03-18) would select Landsat 7, which the year's composite does not
    use.
    """

    @pytest.mark.parametrize(
        ("source", "year", "missions", "spacecraft", "first_date"),
        [
            ("landsat-pan", 2021, ("LC09",), {"LANDSAT_9"}, "2021-11-10"),
            ("landsat-pan", 2024, ("LE07",), {"LANDSAT_7"}, "2024-01-10"),
            ("landsat-pan", 2013, "auto", {"LANDSAT_8"}, "2013-03-25"),
            ("landsat-pan", 1999, "auto", {"LANDSAT_7"}, "1999-06-10"),
            ("landsat", 1984, None, {"LANDSAT_5"}, "1984-03-25"),
        ],
    )
    def test_months_survey_the_images_of_the_whole_window(
        self, archive_ee, source, year, missions, spacecraft, first_date
    ):
        kwargs = {} if missions is None else {"landsat_pan_missions": missions}
        config = _cfg(source, year=year, **kwargs)
        whole = scenes.survey_scenes(config)
        archive_ee.update({"requests": 0, "fail_first": True})
        by_month = scenes.survey_scenes(config)
        year_window = (f"{year}-01-01", f"{year + 1}-01-01")
        assert archive_ee["builds"] == [year_window, year_window]  # once per survey
        assert (whole.attrs["n_requests"], by_month.attrs["n_requests"]) == (1, 13)
        assert len(whole) > 0
        assert by_month.drop(columns=["image_ids"]).equals(whole.drop(columns=["image_ids"]))
        assert by_month["image_ids"].tolist() == whole["image_ids"].tolist()
        labels = {a.rsplit("_", 1)[0] for a in by_month["acquisition"]}
        assert labels == spacecraft
        assert by_month["date"].min() == first_date


def test_survey_area_falls_back_to_the_grid_outline_for_complex_study_areas():
    small = box(-1.54, 5.51, -1.53, 5.52)
    crs = gee.resolve_export_crs("utm", small)
    grid = gee.compute_export_grid(small, crs, 10.0)
    area, what = scenes._survey_area(small, 60.0, grid)
    assert what == "the study area" and area.area == pytest.approx(small.area, rel=0.05)
    # 1,500 squares of 200 m, 300 m apart: 7,500 vertices that a 30 m tolerance keeps.
    step, side = 0.0027, 0.0018
    parts = [
        box(-1.6 + i * step, 5.4 + j * step, -1.6 + i * step + side, 5.4 + j * step + side)
        for i in range(50)
        for j in range(30)
    ]
    many = MultiPolygon(parts)
    grid = gee.compute_export_grid(many, crs, 10.0)
    area, what = scenes._survey_area(many, 60.0, grid)
    assert what.startswith("the export grid's outline")
    assert area.equals(gee.grid_footprint_4326(grid))


# ---------------------------------------------------------------------------
# Date layers: numpy stand-in for ee.Image / ee.ImageCollection
# ---------------------------------------------------------------------------

SHAPE = (1, 4)


def _arr(value, dtype=float):
    return np.broadcast_to(np.asarray(value, dtype=dtype), SHAPE).copy()


class NNumber:
    def __init__(self, value):
        self.value = float(value.value if isinstance(value, NNumber) else value)

    def divide(self, d):
        return NNumber(self.value / d)

    def floor(self):
        return NNumber(math.floor(self.value))

    def format(self, fmt):
        return fmt % self.value


#: The reduceRegion calls of NImage (cleared by the survey_ee fixture).
REDUCTIONS: list = []


class NImage:
    """Bands of equal-shape arrays with per-band masks (True = valid) and properties."""

    def __init__(self, bands, masks=None, props=None):
        self.bands = {k: _arr(v) for k, v in bands.items()}
        self.masks = {
            k: (_arr(True, bool) if masks is None or k not in masks else _arr(masks[k], bool))
            for k in self.bands
        }
        self.props = dict(props or {})

    def _new(self, bands, masks):
        return NImage(bands, masks, self.props)

    def get(self, name):
        return self.props.get(name)

    def set(self, *pairs):
        out = self._new(self.bands, self.masks)
        out.props.update(dict(zip(pairs[::2], pairs[1::2], strict=True)))
        return out

    def bandNames(self):  # noqa: N802 - ee API name
        return NList(self.bands)

    def select(self, names):
        names = [names] if isinstance(names, str | int) else list(names)
        names = [list(self.bands)[n] if isinstance(n, int) else n for n in names]
        return self._new({n: self.bands[n] for n in names}, {n: self.masks[n] for n in names})

    def mask(self):
        return self._new({k: m.astype(float) for k, m in self.masks.items()}, None)

    def reduce(self, reducer):
        values = np.stack(list(self.bands.values()))
        masks = np.stack(list(self.masks.values()))
        name = {"MIN": "min", "MAX": "max"}[reducer]
        return self._new({name: getattr(values, name)(axis=0)}, {name: masks.all(axis=0)})

    def _compare(self, fn):
        return self._new({k: fn(a).astype(float) for k, a in self.bands.items()}, self.masks)

    def neq(self, value):
        return self._compare(lambda a: a != value)

    def eq(self, value):
        return self._compare(lambda a: a == value)

    def gte(self, value):
        return self._compare(lambda a: a >= value)

    def And(self, other):  # noqa: N802 - ee API name
        (b,) = other.bands.values()
        (bm,) = other.masks.values()
        return self._new(
            {k: ((a != 0) & (b != 0)).astype(float) for k, a in self.bands.items()},
            {k: m & bm for k, m in self.masks.items()},
        )

    def multiply(self, factor):
        return self._new({k: a * factor for k, a in self.bands.items()}, self.masks)

    def reduceRegion(self, reducer, geometry, crs, scale, maxPixels):  # noqa: N802, N803
        assert reducer == "MEAN"
        REDUCTIONS.append({"geometry": geometry, "crs": crs, "scale": scale})
        return {
            k: (float(a[self.masks[k]].mean()) if self.masks[k].any() else None)
            for k, a in self.bands.items()
        }

    def gt(self, value):
        return self._new({k: (a > value).astype(float) for k, a in self.bands.items()}, self.masks)

    def unmask(self, value, same_footprint=True):
        assert same_footprint is False
        filled = {k: np.where(self.masks[k], a, value) for k, a in self.bands.items()}
        return self._new(filled, None)

    def rename(self, *names):
        names = list(names[0]) if len(names) == 1 and isinstance(names[0], list) else list(names)
        old = list(self.bands)
        return self._new(
            {n: self.bands[o] for o, n in zip(old, names, strict=True)},
            {n: self.masks[o] for o, n in zip(old, names, strict=True)},
        )

    def toInt32(self):  # noqa: N802 - ee API name
        return self

    def toUint16(self):  # noqa: N802 - ee API name
        out = self._new(self.bands, self.masks)
        out.dtype = "uint16"
        return out

    def updateMask(self, other):  # noqa: N802 - ee API name
        (m,) = other.bands.values()
        (om,) = other.masks.values()
        valid = (m != 0) & om
        return self._new(self.bands, {k: v & valid for k, v in self.masks.items()})

    def normalizedDifference(self, names):  # noqa: N802 - ee API name
        a, b = self.bands[names[0]], self.bands[names[1]]
        mask = self.masks[names[0]] & self.masks[names[1]]
        return self._new({"nd": (a - b) / (a + b)}, {"nd": mask})

    def addBands(self, other):  # noqa: N802 - ee API name
        bands, masks = dict(self.bands), dict(self.masks)
        bands.update(other.bands)
        masks.update(other.masks)
        return self._new(bands, masks)

    def value(self, band):
        return np.where(self.masks[band], self.bands[band], np.nan)


def _cat(images):
    bands, masks = {}, {}
    for image in images:
        for k in image.bands:
            name = k
            while name in bands:  # Earth Engine renames duplicate bands
                name = f"{name}_1"
            bands[name], masks[name] = image.bands[k], image.masks[k]
    return NImage(bands, masks)


class NList(list):
    def distinct(self):
        return NList(dict.fromkeys(self))

    def sort(self):
        return NList(sorted(self))

    def map(self, fn):
        return NList(fn(x) for x in self)

    def contains(self, value):
        return value in self


class NCollection:
    def __init__(self, images):
        self.images = list(images)

    def map(self, fn):
        return NCollection([fn(i) for i in self.images])

    def select(self, name):
        return NCollection([i.select(name) for i in self.images])

    def filter(self, predicate):
        return NCollection([i for i in self.images if predicate(i)])

    def linkCollection(self, collection, bands):  # noqa: N802 - ee API name
        return self  # the test images carry their cs_cdf band already

    def aggregate_array(self, name):
        return NList(i.get(name) for i in self.images)

    def aggregate_max(self, name):
        return max(i.get(name) for i in self.images)

    def size(self):
        return len(self.images)

    def mosaic(self):
        """The last image on top, masked pixels transparent."""
        names = list(self.images[0].bands)
        bands = {n: np.zeros(SHAPE) for n in names}
        masks = {n: np.zeros(SHAPE, bool) for n in names}
        for image in self.images:
            for n in names:
                bands[n] = np.where(image.masks[n], image.bands[n], bands[n])
                masks[n] = masks[n] | image.masks[n]
        return NImage(bands, masks)

    def _stack(self):
        (name,) = self.images[0].bands
        values = np.stack([i.bands[name] for i in self.images])
        masks = np.stack([i.masks[name] for i in self.images])
        return name, values, masks

    def sum(self):
        name, values, masks = self._stack()
        return NImage({name: np.where(masks, values, 0).sum(axis=0)}, {name: masks.any(axis=0)})

    def min(self):
        name, values, masks = self._stack()
        out = np.where(masks, values, np.inf).min(axis=0)
        return NImage({name: np.where(np.isfinite(out), out, 0)}, {name: masks.any(axis=0)})

    def max(self):
        name, values, masks = self._stack()
        out = np.where(masks, values, -np.inf).max(axis=0)
        return NImage({name: np.where(np.isfinite(out), out, 0)}, {name: masks.any(axis=0)})

    def qualityMosaic(self, band):  # noqa: N802 - ee API name
        q = np.stack([np.where(i.masks[band], i.bands[band], -np.inf) for i in self.images])
        best = np.argmax(q, axis=0)
        any_valid = np.isfinite(q).any(axis=0)
        names = list(self.images[0].bands)
        bands = {n: np.choose(best, [i.bands[n] for i in self.images]) for n in names}
        masks = {n: np.choose(best, [i.masks[n] for i in self.images]) & any_valid for n in names}
        return NImage(bands, masks)


@pytest.fixture
def numeric_ee(monkeypatch):
    module = types.ModuleType("ee")

    def image(x):
        return x

    image.constant = lambda value: NImage({"constant": NNumber(value).value})
    image.cat = _cat
    module.Image = image
    module.Number = NNumber
    module.Reducer = types.SimpleNamespace(min=lambda: "MIN")
    monkeypatch.setitem(sys.modules, "ee", module)
    return module


DAY = scenes.DAY_MS


def _image(day, bands, masks=None, hour=10):
    return NImage(bands, masks, {"system:time_start": day * DAY + hour * 3_600_000})


class TestDateLayersImage:
    # Four pixels; days 18263 (2020-01-02), 18478 (2020-08-04), 18623 (2020-12-27).
    IMAGES = (
        (18478, {"B4": 500, "B8": 3000}, {"B4": [1, 1, 0, 1], "B8": [1, 1, 1, 0]}),
        (18263, {"B4": 600, "B8": 2000}, {"B4": [1, 0, 0, 1], "B8": [1, 0, 0, 1]}),
        (18623, {"B4": 400, "B8": [1000, 1000, 1000, 4000]}, {"B4": [1, 1, 0, 0]}),
    )

    def _collection(self):
        return NCollection([_image(d, b, m) for d, b, m in self.IMAGES])

    def test_median_counts_and_first_and_last_day(self, numeric_ee):
        out = scenes.date_layers_image(self._collection(), ["B4", "B8"], "median", "sentinel2")
        assert list(out.bands) == list(scenes.DATE_LAYER_BANDS) and out.dtype == "uint16"
        # Valid observations need every band unmasked (pixel 4: only the 2020-01-02 image).
        np.testing.assert_array_equal(out.bands["n_valid"], [[3, 2, 0, 1]])
        assert out.masks["n_valid"].all()  # 0, not masked, where there is no observation
        np.testing.assert_array_equal(out.value("date_first"), [[18263, 18478, np.nan, 18263]])
        np.testing.assert_array_equal(out.value("date_last"), [[18623, 18623, np.nan, 18263]])
        assert scenes.day_to_date(18478) == "2020-08-04"

    def test_greenest_gives_the_day_of_the_greenest_observation(self, numeric_ee):
        out = scenes.date_layers_image(self._collection(), ["B4", "B8"], "greenest", "sentinel2")
        np.testing.assert_array_equal(out.bands["n_valid"], [[3, 2, 0, 1]])
        # NDVI: 2020-08-04 0.714, 2020-01-02 0.538, 2020-12-27 0.429 (0.818 at pixel 4,
        # where its B4 is masked and its NDVI therefore masked).
        np.testing.assert_array_equal(out.value("date_first"), [[18478, 18478, np.nan, 18263]])
        np.testing.assert_array_equal(out.value("date_last"), out.value("date_first"))
        alias = scenes.date_layers_image(self._collection(), ["B4", "B8"], "max_ndvi", "sentinel2")
        np.testing.assert_array_equal(alias.value("date_first"), out.value("date_first"))

    def test_greenest_without_nir_and_red_raises(self, numeric_ee):
        with pytest.raises(ValueError, match="needs NIR and red bands"):
            scenes.date_layers_image(self._collection(), ["B8"], "greenest", "landsat-pan")

    def test_day_to_date(self):
        assert scenes.day_to_date(0) is None
        assert scenes.day_to_date(1) == "1970-01-02"
        assert scenes.day_to_date(18263) == "2020-01-02"
        assert scenes.DATE_ORIGIN == "1970-01-01"
        days_2149 = (np.datetime64("2149-01-01") - np.datetime64("1970-01-01")).astype(int)
        assert days_2149 < 65535  # uint16 holds the dates until 2149


# ---------------------------------------------------------------------------
# The survey expression, computed with the numpy stand-in
# ---------------------------------------------------------------------------


class NString(str):
    def cat(self, other):
        return NString(f"{self}{other}")


class NDate:
    def __init__(self, time_ms):
        self.time_ms = time_ms

    def format(self, fmt):
        import datetime

        assert fmt == "YYYY-MM-dd"
        when = datetime.datetime.fromtimestamp(self.time_ms / 1000, datetime.UTC)
        return NString(when.date().isoformat())


class NDict(dict):
    def combine(self, other):
        return NDict({**self, **other})


@pytest.fixture
def survey_ee(numeric_ee):
    """numeric_ee with the rest of the ``ee`` API that survey_expression uses."""
    module = numeric_ee
    module.String = NString
    module.Date = NDate
    module.List = NList
    module.Dictionary = NDict
    module.ImageCollection = lambda cid: NCollection([])
    module.Algorithms = types.SimpleNamespace(If=lambda cond, a, b: a if cond else b)
    module.Reducer = types.SimpleNamespace(
        min=lambda: "MIN", max=lambda: "MAX", mean=lambda: "MEAN"
    )
    module.Filter = types.SimpleNamespace(
        eq=lambda prop, value: lambda image: image.get(prop) == value,
        inList=lambda prop, values: lambda image: image.get(prop) in values,
    )
    REDUCTIONS.clear()
    return module


def _survey_images():
    """Raw and prepared images of two dates over four pixels (B4 first: the footprint).

    2020-01-07: two granules of one pass whose clear pixels complement each other
    (the first lacks pixels 2 and 3, the second pixels 0 and 1); the first has an
    asset ID, the second only ``system:index``. 2020-08-04: one image whose B4 is
    masked at pixel 3 (B8 is not), with haze (AOT 0.9) and SCL cloud there.
    """
    common = {"SPACECRAFT_NAME": "Sentinel-2B", "SENSING_ORBIT_NUMBER": 65}
    common.update({"SPACECRAFT_ID": "LANDSAT_8", "WRS_PATH": 194})
    t0107, t0804 = 18268 * DAY + 10 * 3_600_000, 18478 * DAY + 10 * 3_600_000
    specs = [
        ("20200107_T30NXM", True, t0107, 1.5, [1, 1, 1, 0], [4, 4, 8, 4], 100,
         [0.9, 0.9, 0.3, 0.9], {"B4": [1, 1, 0, 0], "B8": [1, 1, 0, 0]}),
        ("20200107_T30NYM", False, t0107 + 5000, 7.25, [0, 1, 1, 1], [4, 4, 4, 4], 200,
         [0.8, 0.8, 0.8, 0.8], {"B4": [0, 0, 1, 1], "B8": [0, 0, 1, 1]}),
        ("20200804_T30NXM", True, t0804, 10.6, [1, 1, 1, 1], [4, 4, 4, 9],
         [100, 100, 100, 900], [0.9, 0.9, 0.9, 0.2], {"B4": [1, 1, 1, 0], "B8": [1, 1, 1, 1]}),
    ]  # fmt: skip
    raw, prepared = [], []
    for index, has_id, time_ms, cloud, footprint, scl, aot, cs, clear in specs:
        props = {**common, "system:index": index, "system:time_start": time_ms}
        props.update({"CLOUDY_PIXEL_PERCENTAGE": cloud, "CLOUD_COVER": cloud})
        if has_id:
            props["system:id"] = f"COPERNICUS/S2_SR_HARMONIZED/{index}"
        bands = {"B4": 500, "B8": 3000, "SCL": scl, "AOT": aot, "cs_cdf": cs}
        raw.append(NImage(bands, {name: footprint for name in bands}, props))
        prepared.append(NImage({"B4": 500, "B8": 3000}, clear, props))
    return NCollection(raw), NCollection(prepared)


class TestSurveyExpressionValues:
    @pytest.mark.parametrize(
        ("source", "cloud"), [("sentinel2", "CLOUDY_PIXEL_PERCENTAGE"), ("landsat", "CLOUD_COVER")]
    )
    def test_shares_per_date(self, survey_ee, source, cloud):
        raw, prepared = _survey_images()
        spec = _fake_spec(source, cloud=cloud)
        spec.raw, spec.prepared = raw, prepared
        rows = scenes.survey_expression(spec, source, "AREA", "EPSG:32630", 60.0, 0.6)
        first, second = rows
        assert REDUCTIONS == [{"geometry": "AREA", "crs": "EPSG:32630", "scale": 60.0}] * 2
        assert (first["date"], second["date"]) == ("2020-01-07", "2020-08-04")
        # The pass of two granules is one row; its image IDs are the asset ID, else the
        # system:index.
        assert first["n_images"] == 2 and second["n_images"] == 1
        assert first["image_ids"] == [
            "20200107_T30NYM",
            "COPERNICUS/S2_SR_HARMONIZED/20200107_T30NXM",
        ]
        labels = {
            "sentinel2": "Sentinel-2B_2020-01-07_R065",
            "landsat": "LANDSAT_8_2020-01-07_P194",
        }
        label = labels[source]
        assert first["acquisition"] == [label]
        assert first["scene_cloud_pct"] == 7.25  # the largest of the date's images
        # Clear share: a pixel counts when any image of the date has every band
        # unmasked there. The two granules together cover every pixel; at the second
        # date, pixel 3 has B8 but not B4.
        assert first["clear_share"] == pytest.approx(1.0)
        assert second["clear_share"] == pytest.approx(0.75)
        assert first["extent_share"] == second["extent_share"] == pytest.approx(1.0)
        s2_keys = {"scl_clear_share", "cs_clear_share", "cs_mean", "aot_mean"}
        if source != "sentinel2":
            assert not s2_keys & set(first) and not s2_keys & set(second)
            return
        # The second granule lies on top of the first in the mosaics of SCL, cs_cdf, AOT.
        assert first["scl_clear_share"] == first["cs_clear_share"] == pytest.approx(1.0)
        assert first["cs_mean"] == pytest.approx((0.9 + 0.8 * 3) / 4)
        assert first["aot_mean"] == pytest.approx((0.1 + 0.2 * 3) / 4)
        assert second["scl_clear_share"] == second["cs_clear_share"] == pytest.approx(0.75)
        assert second["cs_mean"] == pytest.approx(2.9 / 4)
        # Aerosol optical thickness over the clear pixels only (not the hazy pixel 3).
        assert second["aot_mean"] == pytest.approx(0.1)
        ranked = scenes.rank_acquisitions(rows, source)
        assert [r["date"] for r in ranked] == ["2020-01-07", "2020-08-04"]

    def test_partial_footprint_and_fully_masked_date(self, survey_ee):
        raw, prepared = _survey_images()
        # Only the first granule of 2020-01-07, clouded out everywhere.
        raw.images, prepared.images = raw.images[:1], prepared.images[:1]
        prepared.images[0].masks = {k: np.zeros(SHAPE, bool) for k in ("B4", "B8")}
        spec = _fake_spec("sentinel2")
        spec.raw, spec.prepared = raw, prepared
        (row,) = scenes.survey_expression(spec, "sentinel2", "AREA", "EPSG:32630", 60.0, 0.6)
        assert row["clear_share"] == 0.0 and row["extent_share"] == pytest.approx(0.75)
        assert row["aot_mean"] is None  # no clear pixel


# ---------------------------------------------------------------------------
# build_date_layers with the Earth Engine layer stubbed
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_layers(monkeypatch, tmp_path):
    state = {"exports": [], "summary": {"n_images": 52, "years": [2020]}, "years": [2019, 2021]}
    monkeypatch.setattr("agribound.auth.ensure_gee", lambda config: None)
    monkeypatch.setattr(gee, "ee_geometry", lambda geom: "REGION")

    def fake_builder(config, region):
        return _fake_spec(config.source, log=[])

    for source in SCENE_SURVEY_SOURCES:
        monkeypatch.setitem(gee._COLLECTION_BUILDERS, source, fake_builder)

    def fake_summary(collection, context="", count_property=None, **kwargs):
        state["summary_collection"] = collection
        state["summary_context"] = context
        state["summary_kwargs"] = kwargs
        return state["summary"]

    monkeypatch.setattr(gee, "collection_summary", fake_summary)
    monkeypatch.setattr(gee, "available_years", lambda col, context="": state["years"])

    def fake_image(collection, bands, method, source):
        state["image_collection"] = collection
        state["image_args"] = (bands, method, source)
        return "DATE_LAYERS_IMAGE"

    monkeypatch.setattr(scenes, "date_layers_image", fake_image)

    def fake_export(image, out_path, *, grid, dtype, band_names, tags=None, **kwargs):
        state["exports"].append(
            {"image": image, "path": str(out_path), "grid": grid, "dtype": dtype,
             "band_names": list(band_names), "tags": dict(tags or {}), **kwargs}
        )  # fmt: skip
        data = np.ones((len(band_names), grid.height, grid.width), dtype=np.uint16)
        profile = {
            "driver": "GTiff", "dtype": "uint16", "count": len(band_names),
            "width": grid.width, "height": grid.height, "crs": grid.crs,
            "transform": grid.transform, "nodata": 0,
        }  # fmt: skip
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
            dst.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
        return str(out_path)

    monkeypatch.setattr(gee, "export_ee_image", fake_export)
    return state


def _layers_cfg(tmp_path, aoi, **kwargs):
    return AgriboundConfig(
        source=kwargs.pop("source", "sentinel2"),
        year=2020,
        study_area=aoi,
        gee_project="p",
        output_path=str(tmp_path / "out.gpkg"),
        lulc_filter=False,
        **kwargs,
    )


class TestBuildDateLayers:
    def test_export_tags_and_cache(self, tmp_path, sample_aoi_geojson, stub_layers):
        from agribound._cache import cache_key

        config = _layers_cfg(tmp_path, sample_aoi_geojson, cloud_cover_max=80)
        path = scenes.build_date_layers(config)
        (call,) = stub_layers["exports"]
        assert call["path"] == path and call["image"] == "DATE_LAYERS_IMAGE"
        assert call["dtype"] == "uint16"
        assert call["band_names"] == ["n_valid", "date_first", "date_last"]
        assert call["max_requests"] == config.gee_max_requests
        assert call["tile_size"] == config.tile_size
        # The composite's grid: the builder's own (UTM, 10 m for Sentinel-2).
        study = gee.study_area_geometry_4326(config)
        crs = gee.resolve_export_crs("utm", study)
        assert call["grid"] == gee.compute_export_grid(study, crs, 10.0)
        recipe = gee._recipe_parts("sentinel2")
        key = cache_key(config, *recipe, f"date-layers-{scenes.DATE_LAYERS_VERSION}")
        assert path.endswith(f"sentinel2_date_layers_{key}.tif")
        assert key != cache_key(config, *recipe)
        tags = call["tags"]
        assert tags["AGRIBOUND_COMPOSITE_CACHE_KEY"] == cache_key(config, *recipe)
        assert tags["AGRIBOUND_DATE_ORIGIN"] == "1970-01-01"
        assert tags["AGRIBOUND_DATE_UNIT"] == "days"
        assert tags["AGRIBOUND_DATE_LAYERS_VERSION"] == scenes.DATE_LAYERS_VERSION
        assert tags["AGRIBOUND_DATE_LAYERS_RULE"].startswith("n_valid: observations with every")
        assert tags["AGRIBOUND_COMPOSITE_METHOD"] == "median"
        assert tags["AGRIBOUND_N_IMAGES"] == 52
        assert tags["AGRIBOUND_DATE_START"] == "2020-01-01"
        assert tags["AGRIBOUND_DATE_END_EXCLUSIVE"] == "2021-01-01"
        assert tags["AGRIBOUND_CLOUD_MASK"] == "unit-test mask"
        assert stub_layers["image_args"] == (["B4", "B8"], "median", "sentinel2")
        # The layers come from the masked images the composite reduces (spec.prepared);
        # the images are counted on spec.raw, as the composite counts them.
        assert stub_layers["image_collection"].name == "PREPARED"
        assert stub_layers["summary_collection"].name == "RAW"
        assert stub_layers["summary_kwargs"] == {}  # a plain count, no image list
        with rasterio.open(path) as src:
            assert src.tags()["AGRIBOUND_COMPOSITE_CACHE_KEY"] == cache_key(config, *recipe)
        # Cached: no second request or download.
        stub_layers["summary"] = None
        assert scenes.build_date_layers(config) == path
        assert len(stub_layers["exports"]) == 1

    def test_greenest_rule_and_alias(self, tmp_path, sample_aoi_geojson, stub_layers):
        config = _layers_cfg(tmp_path, sample_aoi_geojson, composite_method="max_ndvi")
        scenes.build_date_layers(config)
        tags = stub_layers["exports"][0]["tags"]
        assert tags["AGRIBOUND_COMPOSITE_METHOD"] == "greenest"
        assert "NDVI quality mosaic" in tags["AGRIBOUND_DATE_LAYERS_RULE"]
        assert stub_layers["image_args"][1] == "max_ndvi"

    def test_methods_get_separate_files(self, tmp_path, sample_aoi_geojson, stub_layers):
        median = scenes.build_date_layers(_layers_cfg(tmp_path, sample_aoi_geojson))
        greenest = scenes.build_date_layers(
            _layers_cfg(tmp_path, sample_aoi_geojson, composite_method="greenest")
        )
        assert median != greenest and len(stub_layers["exports"]) == 2

    @pytest.mark.parametrize("method", ["gdrive", "gcs"])
    def test_batch_export_methods_are_refused(
        self, tmp_path, sample_aoi_geojson, monkeypatch, method
    ):
        monkeypatch.setattr(
            "agribound.auth.ensure_gee", lambda config: pytest.fail("Earth Engine initialised")
        )
        config = _layers_cfg(tmp_path, sample_aoi_geojson, export_method=method, gcs_bucket="b")
        with pytest.raises(ValueError, match="need export_method='local'"):
            scenes.build_date_layers(config)

    def test_no_image_raises_the_composite_error(self, tmp_path, sample_aoi_geojson, stub_layers):
        stub_layers["summary"] = {"n_images": 0, "years": []}
        with pytest.raises(NoDataError, match="Years with images over the study-area extent: "):
            scenes.build_date_layers(_layers_cfg(tmp_path, sample_aoi_geojson))
        assert stub_layers["exports"] == []

    def test_check_date_layers(self):
        scenes.check_date_layers(_cfg())
        scenes.check_date_layers(_cfg("spot-pan"))
        cfg = _cfg("landsat-pan")
        cfg.composite_method = "greenest"  # rejected by the configuration; guarded again here
        with pytest.raises(ValueError, match="needs NIR and red bands"):
            scenes.check_date_layers(cfg)


def test_date_layers_path_is_not_a_composite_name(tmp_path, sample_aoi_geojson):
    """``<source>_composite_*.tif`` globs (the gallery, the tests) never match the layers."""
    import fnmatch
    from pathlib import Path

    path = Path(scenes.date_layers_path(_layers_cfg(tmp_path, sample_aoi_geojson)))
    assert path.name.startswith("sentinel2_date_layers_")
    assert not fnmatch.fnmatch(path.name, "sentinel2_composite_*.tif")
    assert path.parent == Path(_layers_cfg(tmp_path, sample_aoi_geojson).get_working_dir())


def test_public_api():
    import agribound

    assert agribound.survey_scenes is scenes.survey_scenes
    assert agribound.build_date_layers is scenes.build_date_layers
    assert {"survey_scenes", "build_date_layers"} <= set(agribound.__all__)
    for name in (
        "survey_scenes",
        "build_date_layers",
        "rank_acquisitions",
        "date_layers_image",
        "survey_scale_m",
    ):
        assert name in scenes.__all__


def _api_members(page_text: str, module: str) -> list[str]:
    """The ``members`` list of the ``::: <module>`` block of an API page."""
    import re

    block = page_text.split(f"::: {module}\n", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^ {8}- (\w+)$", block, flags=re.MULTILINE)


def test_api_page_lists_the_public_names():
    """docs/api/composites.md lists every name of scenes.__all__ (the docstrings cite them)."""
    from pathlib import Path

    page = Path(__file__).resolve().parents[2] / "docs" / "api" / "composites.md"
    if not page.is_file():
        pytest.skip("docs/api/composites.md is not available")
    text = page.read_text(encoding="utf-8")
    listed = _api_members(text, "agribound.composites.scenes")
    assert sorted(listed) == sorted(scenes.__all__)
    for name in listed:
        assert hasattr(scenes, name), name
    gee_listed = _api_members(text, "agribound.composites.gee")
    assert {"IMAGE_LIST_MAX", "collection_summary"} <= set(gee_listed)
    assert set(gee_listed) <= set(gee.__all__)


def test_geotiff_holds_a_long_image_list(tmp_path):
    """1,000 rows of long asset IDs (about 94 KB) round-trip, also after an r+ update.

    The IDs are Sentinel-2 asset IDs with 4 more characters; 1,000 real rows take 88-90 KB.
    """
    import json

    prefix = "COPERNICUS/S2_SR_HARMONIZED/20200102T102421_20200102T103652_T30NXM"
    rows = [[f"{prefix}{i:04d}", "2020-01-02", 99.99] for i in range(gee.IMAGE_LIST_MAX)]
    text = json.dumps(rows, separators=(",", ":"))
    assert len(text) > 90_000
    path = tmp_path / "t.tif"
    profile = {
        "driver": "GTiff", "dtype": "uint16", "count": 1, "width": 4, "height": 4,
        "crs": "EPSG:32630", "transform": Affine(10, 0, 0, 0, -10, 40), "nodata": 0,
    }  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.ones((1, 4, 4), np.uint16))
        dst.update_tags(AGRIBOUND_IMAGE_LIST=text)
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(AGRIBOUND_VALID_FRACTION="1.0")
    assert gee.read_composite_tags(path)["AGRIBOUND_IMAGE_LIST"] == text
