"""Tests of the crop-type map stack builder (cdl-sequence, aci-sequence) without Earth Engine.

A small eager stand-in for the parts of the ``ee`` API that the builder uses
(``ImageCollection``, ``Filter.calendarRange``, ``Feature``, ``Image``,
``Dictionary``) evaluates the builder's one request in Python, so the tests
check the image IDs, scales and legend that it reads, the number of requests,
and the stack it asks to download.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import types

import numpy as np
import pytest
import rasterio

from agribound.composites import crop_sequence as cs
from agribound.composites import gee
from agribound.composites.base import NoDataError, get_composite_builder
from agribound.config import AgriboundConfig
from agribound.registry import CROP_SEQUENCE_SOURCES, SOURCE_REGISTRY

BBOX = "bbox:-93.62,41.98,-93.58,42.01"  # about 3.3 x 3.3 km in Iowa (UTM 15N)


# ---------------------------------------------------------------------------
# Eager stand-in for ee
# ---------------------------------------------------------------------------


def _ms(year: int) -> int:
    return int(dt.datetime(year, 1, 1, tzinfo=dt.UTC).timestamp() * 1000)


class FakeImage:
    """An image of a collection (properties, one band and its scale) or a band selection."""

    def __init__(self, props=None, band=None, scale=30.0, ref=None, rename=None):
        self.props = dict(props or {})
        self.band = band
        self.scale = scale
        self.ref = ref  # asset ID of an image loaded by ID
        self.rename = rename
        self.dtype = None

    def select(self, bands, new_names=None):
        bands = [bands] if isinstance(bands, str) else list(bands)
        names = list(new_names) if new_names is not None else bands
        return FakeImage(self.props, bands[0], self.scale, self.ref, names[0])

    def projection(self):
        return types.SimpleNamespace(nominalScale=lambda: self.scale)


class FakeStack:
    def __init__(self, images):
        self.images = list(images)
        self.dtype = None

    def toUint8(self):  # noqa: N802 - ee API name
        self.dtype = "uint8"
        return self


class FakeCollection:
    def __init__(self, images, ee_module):
        self.images = list(images)
        self.ee = ee_module

    def filterBounds(self, region):  # noqa: N802 - ee API name
        self.ee.calls["regions"].append(region)
        return FakeCollection(self.images, self.ee)

    def filter(self, predicate):
        return FakeCollection([i for i in self.images if predicate(i)], self.ee)

    def sort(self, prop, ascending=True):
        return FakeCollection(
            sorted(self.images, key=lambda i: i.props[prop], reverse=not ascending), self.ee
        )

    def limit(self, n):
        return FakeCollection(self.images[:n], self.ee)

    def map(self, fn):
        return FakeCollection([fn(i) for i in self.images], self.ee)

    def aggregate_array(self, prop):
        return [i.props[prop] for i in self.images if prop in i.props]


def make_fake_ee(collections):
    """An ``ee`` module whose collections are *collections* (asset -> list of FakeImage)."""
    module = types.ModuleType("ee")
    module.calls = {"getInfo": 0, "regions": [], "image_refs": []}

    def image_collection(asset):
        return FakeCollection(collections.get(asset, []), module)

    def image(value):
        if isinstance(value, str):
            module.calls["image_refs"].append(value)
            return FakeImage(ref=value)
        return value

    image.cat = lambda images: FakeStack(images)

    def calendar_range(start, end, field):
        assert field == "year"
        return lambda i: (
            start
            <= dt.datetime.fromtimestamp(i.props["system:time_start"] / 1000, tz=dt.UTC).year
            <= end
        )

    class Feature:
        def __init__(self, geometry, props):
            self.props = dict(props)

    class Dictionary:
        def __init__(self, value):
            self.value = value

        def getInfo(self):  # noqa: N802 - ee API name
            module.calls["getInfo"] += 1
            return _resolve(self.value)

    def _resolve(value):
        if isinstance(value, Dictionary):
            return _resolve(value.value)
        if isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve(v) for v in value]
        return value

    module.ImageCollection = image_collection
    module.Image = image
    module.Filter = types.SimpleNamespace(calendarRange=calendar_range)
    module.Feature = Feature
    module.Dictionary = Dictionary
    return module


CDL_LEGEND_NAMES = ["Background", "Corn", "Soybeans", "Developed/Open Space", "Grass/Pasture"]
CDL_LEGEND_VALUES = [0, 1, 5, 121, 176]


def cdl_images(years, scales=None, extra=()):
    images = []
    for y in years:
        props = {
            "system:index": str(y),
            "system:time_start": _ms(y),
            "cropland_class_names": CDL_LEGEND_NAMES,
            "cropland_class_values": CDL_LEGEND_VALUES,
        }
        images.append(FakeImage(props, "cropland", (scales or {}).get(y, 30.0)))
    for index, y in extra:
        images.append(
            FakeImage({"system:index": index, "system:time_start": _ms(y)}, "cropland", 56.0)
        )
    return images


@pytest.fixture
def fake_ee(monkeypatch):
    def install(collections):
        module = make_fake_ee(collections)
        monkeypatch.setitem(sys.modules, "ee", module)
        return module

    return install


@pytest.fixture
def stub_download(monkeypatch):
    """Stub Earth Engine initialisation, geometry and download; record the exports."""
    calls = {"export": [], "ensure_gee": 0, "zero_columns": 0.0, "zero_band": None}

    def ensure(config):
        calls["ensure_gee"] += 1

    monkeypatch.setattr("agribound.auth.ensure_gee", ensure)
    monkeypatch.setattr(gee, "ee_geometry", lambda geom: "REGION")

    def fake_export(image, out_path, *, grid, dtype, band_names, tags=None, **kwargs):
        calls["export"].append(
            {
                "image": image,
                "grid": grid,
                "dtype": dtype,
                "band_names": list(band_names),
                "tags": dict(tags or {}),
                "kwargs": kwargs,
            }
        )
        data = np.ones((len(band_names), grid.height, grid.width), dtype=np.uint8)
        data[:, :, : int(round(grid.width * calls["zero_columns"]))] = 0
        if calls["zero_band"] is not None:
            data[calls["zero_band"], : grid.height // 2] = 0
        profile = {
            "driver": "GTiff",
            "height": grid.height,
            "width": grid.width,
            "count": len(band_names),
            "dtype": "uint8",
            "crs": grid.crs,
            "transform": grid.transform,
            "nodata": 0,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)
            dst.update_tags(**{k: str(v) for k, v in (tags or {}).items()})
        return str(out_path)

    monkeypatch.setattr(gee, "export_ee_image", fake_export)
    return calls


def _config(tmp_path, source="cdl-sequence", **kwargs):
    kwargs.setdefault("year", 2023)
    return AgriboundConfig(
        source=source,
        engine="crop-sequence",
        study_area=BBOX,
        gee_project="test-project",
        output_path=str(tmp_path / "out.gpkg"),
        cache_dir=str(tmp_path / "cache"),
        lulc_filter=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Datasets, window, grid
# ---------------------------------------------------------------------------


class TestDatasets:
    def test_series_match_the_registry(self):
        assert set(cs.CROP_MAP_DATASETS) == set(CROP_SEQUENCE_SOURCES)
        for source, ds in cs.CROP_MAP_DATASETS.items():
            info = SOURCE_REGISTRY[source]
            assert ds.source == source
            assert info["collection"] == ds.asset
            assert info["year_range"] == (ds.first_year, None)
            assert info["value_scale"] == "categorical"
            assert info["resolution_m"] == cs.CROP_SEQUENCE_RESOLUTION_M
            assert info["all_bands"] is None and info["requires_gee"] is True
        assert cs.DATASET_SOURCES == {"cdl": "cdl-sequence", "aci": "aci-sequence"}

    def test_cdl_defaults_are_example_25s_farmed_classes(self):
        farmed = sorted({*range(1, 62), *range(66, 78), *range(204, 255), 176})
        assert list(cs.CDL_AGRICULTURAL_CLASSES) == farmed
        assert 37 in farmed and 176 in farmed and 61 in farmed  # hay, pasture, fallow
        assert not {62, 63, 64, 65, 81, 82, 83, 111, 121, 131, 141, 152, 190} & set(farmed)
        assert cs.CDL_BARRIER_CLASSES == (121, 122, 123, 124)

    def test_aci_defaults_are_the_legend_codes_120_to_199(self):
        codes = cs.ACI_AGRICULTURAL_CLASSES
        assert len(codes) == 61 and list(codes) == sorted(set(codes))
        assert all(120 <= c <= 199 for c in codes)
        # 2009-2023 legend: 59 codes; 121 (Cropland) and 159 (Other Oilseeds) from 2024.
        assert len(set(codes) - {121, 159}) == 59
        assert cs.ACI_BARRIER_CLASSES == (34,)
        aci = cs.CROP_MAP_DATASETS["aci-sequence"]
        assert (aci.asset, aci.band, aci.band_prefix, aci.first_year) == (
            "AAFC/ACI",
            "landcover",
            "ACI",
            2011,
        )
        assert "Open Government Licence - Canada" in aci.attribution
        assert "public domain" in cs.CROP_MAP_DATASETS["cdl-sequence"].attribution

    def test_window_years_end_in_year(self, tmp_path):
        assert cs.window_years(_config(tmp_path)) == list(range(2016, 2024))
        assert cs.window_years(_config(tmp_path, year=2020, crop_sequence_years=3)) == [
            2018,
            2019,
            2020,
        ]
        assert cs.band_names(_config(tmp_path, crop_sequence_years=2)) == ["CDL2022", "CDL2023"]

    def test_pad_grid(self):
        from rasterio.transform import Affine

        grid = gee.ExportGrid("EPSG:32615", Affine(30.0, 0, 600000.0, 0, -30.0, 4650000.0), 10, 7)
        padded = cs.pad_grid(grid, 10)
        assert (padded.width, padded.height) == (30, 27)
        assert padded.transform.c == 600000.0 - 300.0
        assert padded.transform.f == 4650000.0 + 300.0
        assert padded.transform.a == 30.0 and padded.transform.e == -30.0
        assert padded.crs == grid.crs
        assert cs.pad_grid(grid, 0) == grid
        with pytest.raises(ValueError, match=">= 0"):
            cs.pad_grid(grid, -1)

    def test_cache_path_carries_the_recipe_and_the_window(self, tmp_path):
        cfg = _config(tmp_path)
        assert cs.recipe_parts() == ("crop-sequence-recipe-1", "pad=10")
        path = cs.stack_cache_path(cfg)
        assert path.name.startswith("cdl-sequence_stack_") and path.suffix == ".tif"
        assert cs.stack_cache_path(cfg.merged(crop_sequence_years=7)) != path
        assert cs.stack_cache_path(cfg.merged(year=2022)) != path
        assert cs.stack_cache_path(cfg.merged(engine_params={"smooth_radius_px": 2})) == path

    def test_parse_legend(self):
        assert cs.parse_legend(["A", "B"], [1, 2]) == {1: "A", 2: "B"}
        assert cs.parse_legend("A,B", "1,2") == {1: "A", 2: "B"}
        assert cs.parse_legend(["A"], [1, 2]) == {}
        assert cs.parse_legend(None, [1]) == {}

    def test_builder_factory(self):
        for source in CROP_SEQUENCE_SOURCES:
            builder = get_composite_builder(source)
            assert isinstance(builder, cs.CropSequenceCompositeBuilder)
            assert builder.get_band_mapping(source) == {}
            assert builder.get_resolution(source) == 30


# ---------------------------------------------------------------------------
# Building the stack (Earth Engine mocked)
# ---------------------------------------------------------------------------


class TestBuild:
    def test_one_request_one_image_per_year_and_the_padded_uint8_download(
        self, tmp_path, fake_ee, stub_download
    ):
        ee = fake_ee({"USDA/NASS/CDL": cdl_images(range(2008, 2026), scales={2024: 10.0})})
        cfg = _config(tmp_path)
        path = cs.CropSequenceCompositeBuilder().build(cfg)
        assert ee.calls["getInfo"] == 1
        assert ee.calls["regions"] and set(ee.calls["regions"]) == {"REGION"}
        (export,) = stub_download["export"]
        names = [f"CDL{y}" for y in range(2016, 2024)]
        assert export["dtype"] == "uint8" and export["band_names"] == names
        # One image per window year, its band renamed to CDL<year>, cast to uint8.
        stack = export["image"]
        assert stack.dtype == "uint8"
        assert [i.ref for i in stack.images] == [f"USDA/NASS/CDL/{y}" for y in range(2016, 2024)]
        assert [(i.band, i.rename) for i in stack.images] == [("cropland", n) for n in names]
        # The grid is the study area's UTM grid at 30 m, padded by 10 px on every side.
        geom = gee.study_area_geometry_4326(cfg)
        base = gee.compute_export_grid(geom, gee.resolve_export_crs("utm", geom), 30.0)
        grid = export["grid"]
        assert grid.crs == base.crs == "EPSG:32615"
        assert (grid.width, grid.height) == (base.width + 20, base.height + 20)
        assert grid.transform.c == base.transform.c - 300.0
        assert grid.transform.f == base.transform.f + 300.0
        with rasterio.open(path) as src:
            tags = src.tags()
            assert src.dtypes == ("uint8",) * 8 and src.nodata == 0
            assert list(src.descriptions) == names
        assert tags["AGRIBOUND_SOURCE"] == "cdl-sequence"
        assert tags["AGRIBOUND_VALUE_SCALE"] == "categorical"
        assert tags["AGRIBOUND_YEAR"] == "2023" and tags["AGRIBOUND_CROP_SEQUENCE_YEARS"] == "8"
        assert tags["AGRIBOUND_YEARS"] == ",".join(str(y) for y in range(2016, 2024))
        assert tags["AGRIBOUND_COLLECTIONS"] == "USDA/NASS/CDL"
        assert json.loads(tags["AGRIBOUND_IMAGE_IDS"])["2016"] == "USDA/NASS/CDL/2016"
        assert json.loads(tags["AGRIBOUND_NATIVE_SCALES_M"])["2023"] == 30.0
        assert tags["AGRIBOUND_EXPORT_CRS"] == "EPSG:32615"
        assert tags["AGRIBOUND_RESOLUTION_M"] == "30.0"
        assert tags["AGRIBOUND_RESAMPLING"] == "nearest"
        assert tags["AGRIBOUND_PAD_PX"] == "10"
        assert tags["AGRIBOUND_LEGEND_YEAR"] == "2023"
        assert json.loads(tags["AGRIBOUND_LEGEND"]) == {
            "0": "Background",
            "1": "Corn",
            "5": "Soybeans",
            "121": "Developed/Open Space",
            "176": "Grass/Pasture",
        }
        assert tags["AGRIBOUND_RECIPE_VERSION"] == cs.CROP_SEQUENCE_RECIPE_VERSION == "1"
        from agribound._cache import cache_key

        assert tags["AGRIBOUND_CACHE_KEY"] == cache_key(cfg, *cs.recipe_parts())
        assert float(tags["AGRIBOUND_VALID_FRACTION"]) == 1.0
        by_year = json.loads(tags["AGRIBOUND_VALID_FRACTION_BY_YEAR"])
        assert set(by_year) == {str(y) for y in range(2016, 2024)}
        assert "public domain" in tags["AGRIBOUND_ATTRIBUTION"]
        assert tags["AGRIBOUND_ATTRIBUTION"].endswith("2016-2023")

    def test_10_m_years_are_recorded_with_the_pyramid_level_they_are_read_from(
        self, tmp_path, fake_ee, stub_download, caplog
    ):
        fake_ee({"USDA/NASS/CDL": cdl_images(range(2008, 2026), scales={2024: 10.0})})
        cfg = _config(tmp_path, year=2024)
        with caplog.at_level("INFO", logger="agribound.composites.crop_sequence"):
            path = cs.build_crop_sequence(cfg)[0]
        with rasterio.open(path) as src:
            tags = src.tags()
        scales = json.loads(tags["AGRIBOUND_NATIVE_SCALES_M"])
        assert scales["2024"] == 10.0 and scales["2017"] == 30.0
        assert tags["AGRIBOUND_RESAMPLING"] == "nearest; 2024 (10 m maps): 20 m MODE pyramid level"
        assert stub_download["export"][0]["band_names"][-1] == "CDL2024"
        assert "2024: 20 m level" in caplog.text and "MODE pyramid" in caplog.text

    def test_pyramid_level_and_resampling_tag(self):
        # Earth Engine reads the level with the closest scale <= the requested 30 m.
        assert cs.pyramid_level_m(10.0) == 20.0
        assert cs.pyramid_level_m(15.0) == 30.0
        assert cs.pyramid_level_m(5.0) == 20.0
        assert cs.pyramid_level_m(30.0) == 30.0 and cs.pyramid_level_m(56.0) == 56.0
        with pytest.raises(ValueError):
            cs.pyramid_level_m(0.0)
        assert cs.resampling_tag({"2016": 30.0, "2017": 30.0}) == "nearest"
        assert cs.resampling_tag({"2023": 30.0, "2024": None}) == "nearest"
        assert cs.resampling_tag({"2023": 30.0, "2024": 10.0, "2025": 10.0}) == (
            "nearest; 2024, 2025 (10 m maps): 20 m MODE pyramid level"
        )

    def test_an_export_crs_in_feet_is_refused_before_any_request(
        self, tmp_path, fake_ee, stub_download
    ):
        ee = fake_ee({"USDA/NASS/CDL": cdl_images(range(2008, 2024))})
        cfg = _config(tmp_path)
        cfg.export_crs = "EPSG:2263"  # changed after validation, which refuses it
        with pytest.raises(ValueError, match="needs a projected CRS in metres"):
            cs.build_crop_sequence(cfg)
        assert ee.calls["getInfo"] == 0 and not stub_download["export"]

    def test_cache_hit_makes_no_earth_engine_call(self, tmp_path, fake_ee, stub_download):
        ee = fake_ee({"USDA/NASS/CDL": cdl_images(range(2008, 2026))})
        cfg = _config(tmp_path)
        first = cs.CropSequenceCompositeBuilder().build(cfg)
        assert ee.calls["getInfo"] == 1 and stub_download["ensure_gee"] == 1
        builder = cs.CropSequenceCompositeBuilder()
        assert builder.build(cfg) == first
        assert ee.calls["getInfo"] == 1 and stub_download["ensure_gee"] == 1
        assert len(stub_download["export"]) == 1
        assert builder.last_metadata["AGRIBOUND_VALUE_SCALE"] == "categorical"
        # Another window is another stack.
        cs.CropSequenceCompositeBuilder().build(cfg.merged(crop_sequence_years=5))
        assert ee.calls["getInfo"] == 2 and len(stub_download["export"]) == 2

    def test_missing_year_raises_no_data_error_listing_the_years(
        self, tmp_path, fake_ee, stub_download
    ):
        years = [y for y in range(2008, 2024) if y != 2019]
        fake_ee({"USDA/NASS/CDL": cdl_images(years)})
        with pytest.raises(NoDataError, match="for 2019") as info:
            cs.build_crop_sequence(_config(tmp_path))
        message = str(info.value)
        assert "2018, 2020" in message and "2016-2023" in message
        assert not stub_download["export"]

    def test_no_image_at_all_lists_none(self, tmp_path, fake_ee, stub_download):
        fake_ee({})
        with pytest.raises(NoDataError, match="Years with images over the study-area extent: none"):
            cs.build_crop_sequence(_config(tmp_path))

    def test_duplicate_year_raises(self, tmp_path, fake_ee, stub_download):
        fake_ee({"USDA/NASS/CDL": cdl_images(range(2008, 2024), extra=[("2020b", 2020)])})
        with pytest.raises(ValueError, match="more than one image") as info:
            cs.build_crop_sequence(_config(tmp_path))
        assert not isinstance(info.value, NoDataError)
        assert "2020: 2020, 2020b" in str(info.value)
        assert not stub_download["export"]

    def test_zero_valid_fraction_raises_and_deletes_the_stack(
        self, tmp_path, fake_ee, stub_download
    ):
        fake_ee({"USDA/NASS/CDL": cdl_images(range(2008, 2024))})
        stub_download["zero_columns"] = 1.0
        cfg = _config(tmp_path)
        with pytest.raises(NoDataError, match="no pixel inside the study area"):
            cs.build_crop_sequence(cfg)
        assert not cs.stack_cache_path(cfg).exists()

    def test_low_valid_fraction_warns_with_the_fraction_per_year(
        self, tmp_path, fake_ee, stub_download, caplog
    ):
        fake_ee({"USDA/NASS/CDL": cdl_images(range(2008, 2024))})
        stub_download["zero_band"] = 3  # 2019: half of the rows without a class
        path, tags = cs.build_crop_sequence(_config(tmp_path))
        by_year = json.loads(tags["AGRIBOUND_VALID_FRACTION_BY_YEAR"])
        assert by_year["2016"] == 1.0 and by_year["2019"] < 0.6
        assert tags["AGRIBOUND_VALID_FRACTION"] == pytest.approx(by_year["2019"])
        assert any("in only" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")

    def test_aci_sequence(self, tmp_path, fake_ee, stub_download):
        images = []
        for y in range(2009, 2026):
            props = {
                "system:index": str(y),
                "system:time_start": _ms(y),
                "landcover_class_names": "Urban,Corn",
                "landcover_class_values": "34,147",
            }
            images.append(FakeImage(props, "landcover", 56.0 if y < 2011 else 30.0))
        ee = fake_ee({"AAFC/ACI": images})
        bbox = "bbox:-71.30,46.50,-71.26,46.53"  # Quebec
        cfg = AgriboundConfig(
            source="aci-sequence",
            engine="crop-sequence",
            year=2023,
            study_area=bbox,
            gee_project="p",
            output_path=str(tmp_path / "o.gpkg"),
            cache_dir=str(tmp_path / "c"),
            lulc_filter=False,
        )
        path, tags = cs.build_crop_sequence(cfg)
        (export,) = stub_download["export"]
        assert export["band_names"] == [f"ACI{y}" for y in range(2016, 2024)]
        assert [i.ref for i in export["image"].images][0] == "AAFC/ACI/2016"
        assert ee.calls["getInfo"] == 1
        assert json.loads(tags["AGRIBOUND_LEGEND"]) == {"34": "Urban", "147": "Corn"}
        assert "Open Government Licence - Canada" in tags["AGRIBOUND_ATTRIBUTION"]

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("date_range", ("2023-01-01", "2023-12-31"), "date_range"),
            ("export_method", "gdrive", "export_method"),
            ("composite_method", "greenest", "composite_method"),
        ],
    )
    def test_settings_changed_after_validation_are_refused_before_earth_engine(
        self, tmp_path, fake_ee, stub_download, field, value, match
    ):
        ee = fake_ee({"USDA/NASS/CDL": cdl_images(range(2008, 2024))})
        cfg = _config(tmp_path)
        setattr(cfg, field, value)
        with pytest.raises(ValueError, match=match):
            cs.build_crop_sequence(cfg)
        assert ee.calls["getInfo"] == 0 and stub_download["ensure_gee"] == 0

    def test_builder_needs_a_study_area_and_a_crop_map_source(self, tmp_path):
        cfg = _config(tmp_path)
        cfg.study_area = ""
        with pytest.raises(ValueError, match="study_area is required"):
            cs.CropSequenceCompositeBuilder().build(cfg)
        other = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", lulc_filter=False)
        with pytest.raises(ValueError, match="No crop-type map series"):
            cs.CropSequenceCompositeBuilder().build(other)


def test_valid_fraction_by_band(tmp_path):
    from rasterio.transform import from_bounds
    from shapely.geometry import box

    path = tmp_path / "s.tif"
    data = np.ones((2, 10, 10), dtype=np.uint8)
    data[1, :5] = 0
    transform = from_bounds(500000, 4000000, 500300, 4000300, 10, 10)
    with rasterio.open(
        path, "w", driver="GTiff", height=10, width=10, count=2, dtype="uint8",
        crs="EPSG:32615", transform=transform, nodata=0,
    ) as dst:  # fmt: skip
        dst.write(data)
    import geopandas as gpd

    whole = gpd.GeoSeries([box(500000, 4000000, 500300, 4000300)], crs="EPSG:32615")
    geom = whole.to_crs("EPSG:4326").iloc[0].buffer(-1e-6)
    fractions = cs.valid_fraction_by_band(path, geom)
    assert fractions[0] == pytest.approx(1.0)
    assert fractions[1] == pytest.approx(0.5, abs=0.06)
