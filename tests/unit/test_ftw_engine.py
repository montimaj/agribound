"""Unit tests for the FTW semantic-segmentation engine (no network, no GPU)."""

from __future__ import annotations

import datetime as dt
import inspect
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from agribound.config import AgriboundConfig
from agribound.engines import ftw
from agribound.registry import ENGINE_REGISTRY


@dataclass
class _Spec:
    url: str
    title: str = "t"
    description: str = "d"
    license: str = "CC-BY-4.0"
    version: str = "v3"
    requires_window: bool = True
    requires_polygonize: bool = True
    instance_segmentation: bool = False
    default: bool = False
    legacy: bool = False


_REGISTRY = {
    "FTW_PRUE_EFNET_B5": _Spec(url="https://example/b5.ckpt", default=True),
    "FTW_v2_3_Class_FULL_singleWindow": _Spec(
        url="https://example/single.ckpt", requires_window=False, legacy=True, version="v2"
    ),
    "DelineateAnything": _Spec(
        url="https://example/da.pt",
        requires_window=False,
        requires_polygonize=False,
        instance_segmentation=True,
        license="AGPL-3",
    ),
}


def _write(path, data, crs="EPSG:32611", transform=None, nodata=None):
    count, height, width = data.shape
    transform = transform or from_origin(500000, 4000000, 10.0, 10.0)
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


class FakeFTW:
    """Fake ftw_tools modules recording calls."""

    def __init__(self, monkeypatch, with_nan_fill=False, sos_eos=(279, 161)):
        self.run_calls: list[dict] = []
        self.poly_calls: list[dict] = []
        self.sos_eos = sos_eos
        pkg = types.ModuleType("ftw_tools")
        inference_pkg = types.ModuleType("ftw_tools.inference")
        inference = types.ModuleType("ftw_tools.inference.inference")
        registry = types.ModuleType("ftw_tools.inference.model_registry")
        registry.MODEL_REGISTRY = dict(_REGISTRY)
        postprocess_pkg = types.ModuleType("ftw_tools.postprocess")
        polygonize_mod = types.ModuleType("ftw_tools.postprocess.polygonize")
        utils = types.ModuleType("ftw_tools.utils")

        def run(
            input,  # noqa: A002
            model,
            out,
            resize_factor,
            gpu,
            patch_size,
            batch_size,
            num_workers,
            padding,
            overwrite,
            mps_mode,
            save_scores,
            compute_consensus=False,
        ):
            self.run_calls.append(dict(locals()))
            with rasterio.open(input) as src:
                profile = src.profile.copy()
                data = src.read(1)
            pred = np.zeros(data.shape, dtype=np.uint8)
            pred[5:15, 5:15] = 1  # one 10 x 10 px field
            pred[4, 5:15] = 2
            profile.update(count=1, dtype="uint8", nodata=0)
            with rasterio.open(out, "w", **profile) as dst:
                dst.write(pred, 1)

        if with_nan_fill:

            def run_nan(*args, nan_fill_value=0.0, **kwargs):
                self.run_calls.append({"nan_fill_value": nan_fill_value})
                return run(*args, **kwargs)

            run_nan.__signature__ = inspect.Signature(
                [*inspect.signature(run).parameters.values()]
                + [inspect.Parameter("nan_fill_value", inspect.Parameter.KEYWORD_ONLY, default=0.0)]
            )
            inference.run = run_nan
        else:
            inference.run = run

        def polygonize(
            input,  # noqa: A002
            out,
            simplify=True,
            min_size=500,
            max_size=None,
            overwrite=False,
            close_interiors=False,
            polygonization_stride=2048,
            softmax_threshold=None,
            merge_adjacent=None,
            erode_dilate=0,
            dilate_erode=0,
            erode_dilate_raster=0,
            dilate_erode_raster=0,
            thin_boundaries=False,
        ):
            self.poly_calls.append(dict(locals()))
            with rasterio.open(input) as src:
                crs = src.crs
                tf = src.transform
            geom = box(*rasterio.transform.array_bounds(10, 10, tf @ tf.translation(5, 5)))
            gpd.GeoDataFrame({"id": ["1"]}, geometry=[geom], crs=crs).to_file(out)

        polygonize_mod.polygonize = polygonize

        def get_harvest_integer_from_bbox(
            bbox, start_year_raster_path=None, end_year_raster_path=None
        ):
            self.bbox = bbox
            return list(self.sos_eos)

        import pandas as pd

        utils.get_harvest_integer_from_bbox = get_harvest_integer_from_bbox
        utils.harvest_to_datetime = lambda harvest_day, year: pd.to_datetime(
            f"{year}-{harvest_day}", format="%Y-%j"
        )
        for name, module in {
            "ftw_tools": pkg,
            "ftw_tools.inference": inference_pkg,
            "ftw_tools.inference.inference": inference,
            "ftw_tools.inference.model_registry": registry,
            "ftw_tools.postprocess": postprocess_pkg,
            "ftw_tools.postprocess.polygonize": polygonize_mod,
            "ftw_tools.utils": utils,
        }.items():
            monkeypatch.setitem(sys.modules, name, module)


def _s2_raster(tmp_path, name="s2.tif", value=1000.0, crs="EPSG:32611", transform=None, size=160):
    data = np.full((12, size, size), value, dtype=np.float32)
    data[:, 0, 0] = np.nan
    return _write(tmp_path / name, data, crs=crs, transform=transform, nodata=float("nan"))


def _config(tmp_path, source="sentinel2", **engine_params):
    kwargs = {}
    if source == "local":
        kwargs["local_tif_path"] = str(tmp_path / "local.tif")
    return AgriboundConfig(
        source=source,
        engine="ftw",
        year=2024,
        study_area="bbox:149.70,-30.40,149.72,-30.38",
        gee_project="test-project",
        output_path=str(tmp_path / "out" / "fields.gpkg"),
        device="cpu",
        lulc_filter=False,
        min_field_area_m2=100.0,
        engine_params=engine_params,
        **kwargs,
    )


class FakeBuilder:
    def __init__(self, tmp_path, fail=()):
        self.tmp_path = tmp_path
        self.fail = set(fail)
        self.calls: list[tuple] = []

    def build(self, config):
        self.calls.append((config.date_range, config.composite_method))
        if config.date_range in self.fail:
            from agribound.composites import NoDataError

            raise NoDataError("No sentinel2 images for this window")
        start = config.date_range[0]
        return _s2_raster(self.tmp_path, f"window_{start}.tif", value=2000.0 + len(self.calls))


# ---------------------------------------------------------------------------
# Registry and model resolution
# ---------------------------------------------------------------------------


def test_engine_attributes_come_from_registry():
    entry = ENGINE_REGISTRY["ftw"]
    engine = ftw.FTWEngine()
    assert engine.supported_sources == entry["supported_sources"]
    assert engine.requires_bands == ["R", "G", "B", "NIR"]


def test_list_ftw_models_fields_and_legacy_filter(monkeypatch):
    FakeFTW(monkeypatch)
    models = ftw.list_ftw_models()
    assert "FTW_v2_3_Class_FULL_singleWindow" not in models
    info = models["FTW_PRUE_EFNET_B5"]
    assert set(info) == {
        "url",
        "title",
        "description",
        "license",
        "version",
        "requires_window",
        "requires_polygonize",
        "instance_segmentation",
        "default",
        "legacy",
    }
    assert "FTW_v2_3_Class_FULL_singleWindow" in ftw.list_ftw_models(include_legacy=True)


def test_default_model_is_the_registry_default(monkeypatch):
    FakeFTW(monkeypatch)
    assert ftw.default_ftw_model() == "FTW_PRUE_EFNET_B5"
    choice = ftw.resolve_ftw_model({})
    assert (choice.registry_key, choice.n_windows, choice.in_channels) == (
        "FTW_PRUE_EFNET_B5",
        2,
        8,
    )
    single = ftw.resolve_ftw_model({"model": "FTW_v2_3_Class_FULL_singleWindow"})
    assert single.n_windows == 1


def test_instance_segmentation_and_unknown_models_are_rejected(monkeypatch):
    FakeFTW(monkeypatch)
    with pytest.raises(ValueError, match="delineate-anything"):
        ftw.resolve_ftw_model({"model": "DelineateAnything"})
    with pytest.raises(ValueError, match="Unknown FTW model"):
        ftw.resolve_ftw_model({"model": "FTW_NOPE"})


@pytest.mark.parametrize(("channels", "windows"), [(4, 1), (8, 2)])
def test_checkpoint_in_channels_sets_window_count(monkeypatch, tmp_path, channels, windows):
    torch = pytest.importorskip("torch")
    FakeFTW(monkeypatch)
    ckpt = tmp_path / "sub dir" / "my_model.ckpt"
    ckpt.parent.mkdir()
    torch.save({"hyper_parameters": {"in_channels": channels}, "state_dict": {}}, ckpt)
    choice = ftw.resolve_ftw_model({"checkpoint_path": str(ckpt)})
    assert (choice.in_channels, choice.n_windows) == (channels, windows)
    assert choice.run_model == str(ckpt.resolve())
    assert "/" not in choice.cache_id and choice.cache_id.startswith("ckpt-my_model-")


def test_checkpoint_with_unsupported_channels(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    FakeFTW(monkeypatch)
    ckpt = tmp_path / "m.ckpt"
    torch.save({"hyper_parameters": {"in_channels": 6}}, ckpt)
    with pytest.raises(ValueError, match="in_channels=6"):
        ftw.resolve_ftw_model({"checkpoint_path": str(ckpt)})
    torch.save({"state_dict": {}}, ckpt)
    with pytest.raises(ValueError, match="hyper_parameters"):
        ftw.resolve_ftw_model({"checkpoint_path": str(ckpt)})


# ---------------------------------------------------------------------------
# Crop-calendar windows
# ---------------------------------------------------------------------------


def test_crop_calendar_rollover_southern_hemisphere(monkeypatch):
    fake = FakeFTW(monkeypatch, sos_eos=(279, 161))  # Namoi: SOS DOY 279, EOS DOY 161
    info = ftw.crop_calendar_centres((149.70, -30.40, 149.72, -30.38), 2024)
    assert info["rollover"] is True and info["year_b"] == 2025
    assert info["centre_a"] == "2024-10-05"
    assert info["centre_b"] == "2025-06-10"
    assert fake.bbox == [149.70, -30.40, 149.72, -30.38]


def test_crop_calendar_northern_hemisphere(monkeypatch):
    FakeFTW(monkeypatch, sos_eos=(86, 249))
    info = ftw.crop_calendar_centres((1.0, 48.0, 1.1, 48.1), 2023)
    assert info["rollover"] is False and info["year_b"] == 2023
    assert (info["centre_a"], info["centre_b"]) == ("2023-03-27", "2023-09-06")


@pytest.mark.parametrize("error", ["nan", "out_of_bounds"])
def test_crop_calendar_without_data_raises_actionable_error(monkeypatch, error):
    FakeFTW(monkeypatch)
    utils = sys.modules["ftw_tools.utils"]

    def no_data(bbox, start_year_raster_path=None, end_year_raster_path=None):
        if error == "nan":
            return [int(float("nan")), 1]  # what upstream does with a NaN calendar cell
        from rioxarray.exceptions import NoDataInBounds

        raise NoDataInBounds("No data found in bounds.")

    if error == "out_of_bounds":
        pytest.importorskip("rioxarray")
    monkeypatch.setattr(utils, "get_harvest_integer_from_bbox", no_data)
    with pytest.raises(ValueError, match="window_dates"):
        ftw.crop_calendar_centres((-140.0, -30.0, -139.99, -29.99), 2024)


def test_window_range():
    assert ftw.window_range("2024-10-05", 30) == ("2024-09-05", "2024-11-04")
    assert ftw.window_range(dt.date(2025, 1, 10), 15) == ("2024-12-26", "2025-01-25")


def test_two_windows_are_distinct_median_composites(monkeypatch, tmp_path):
    FakeFTW(monkeypatch)
    builder = FakeBuilder(tmp_path)
    import agribound.composites as composites

    monkeypatch.setattr(composites, "get_composite_builder", lambda source: builder)
    config = _config(tmp_path, window_days=10)
    raster = _s2_raster(tmp_path)
    sources, record = ftw.FTWEngine._two_windows(config, raster, config.engine_params, [4, 3, 2, 8])
    assert builder.calls == [
        (("2024-09-25", "2024-10-15"), "median"),
        (("2025-05-31", "2025-06-20"), "median"),
    ]
    assert sources[0][0] != sources[1][0]
    assert record["a"]["status"] == record["b"]["status"] == "composite"
    assert record["centres"]["method"] == "ftw_crop_calendar"


def test_missing_window_raises_without_fallback(monkeypatch, tmp_path):
    FakeFTW(monkeypatch)
    builder = FakeBuilder(tmp_path, fail={("2025-05-11", "2025-07-10")})
    import agribound.composites as composites

    monkeypatch.setattr(composites, "get_composite_builder", lambda source: builder)
    config = _config(tmp_path)
    raster = _s2_raster(tmp_path)
    with pytest.raises(RuntimeError, match="allow_annual_fallback"):
        ftw.FTWEngine._two_windows(config, raster, config.engine_params, [4, 3, 2, 8])
    config = _config(tmp_path, allow_annual_fallback=True)
    sources, record = ftw.FTWEngine._two_windows(config, raster, config.engine_params, [4, 3, 2, 8])
    assert record["b"]["status"] == "annual_fallback" and sources[1][0] == raster
    assert "No sentinel2 images" in record["b"]["error"]


def test_window_no_data_is_chained_and_other_value_errors_propagate(monkeypatch, tmp_path):
    """Only NoDataError means 'no imagery': other ValueErrors never trigger the fallback."""
    from agribound.composites import NoDataError

    FakeFTW(monkeypatch)
    builder = FakeBuilder(tmp_path, fail={("2025-05-11", "2025-07-10")})
    import agribound.composites as composites

    monkeypatch.setattr(composites, "get_composite_builder", lambda source: builder)
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="FTW window B") as info:
        ftw.FTWEngine._two_windows(config, _s2_raster(tmp_path), config.engine_params, [4, 3, 2, 8])
    assert isinstance(info.value.__cause__, NoDataError)

    class BadConfigBuilder(FakeBuilder):
        def build(self, config):
            if config.date_range[0].startswith("2025"):
                raise ValueError("Unknown export method: 'ftp'")
            return super().build(config)

    bad = BadConfigBuilder(tmp_path)
    monkeypatch.setattr(composites, "get_composite_builder", lambda source: bad)
    config = _config(tmp_path, allow_annual_fallback=True)
    with pytest.raises(ValueError, match="Unknown export method"):
        ftw.FTWEngine._two_windows(config, _s2_raster(tmp_path), config.engine_params, [4, 3, 2, 8])


def test_stage_inputs_builds_the_windows_delineate_uses(monkeypatch, tmp_path):
    FakeFTW(monkeypatch)
    builder = FakeBuilder(tmp_path)
    import agribound.composites as composites

    monkeypatch.setattr(composites, "get_composite_builder", lambda source: builder)
    config = _config(tmp_path, window_days=10)
    raster = _s2_raster(tmp_path)
    staged = ftw.FTWEngine.stage_inputs(config, raster)
    sources, record = ftw.FTWEngine._two_windows(config, raster, config.engine_params, [4, 3, 2, 8])
    assert staged["n_windows"] == 2
    assert staged["band_indices_rgbn"] == [4, 3, 2, 8]
    assert staged["rasters"] == [path for path, _ in sources]
    assert staged["windows"]["a"]["start"] == record["a"]["start"] == "2024-09-25"
    assert staged["windows"]["b"]["raster"] == record["b"]["raster"]
    assert builder.calls[:2] == builder.calls[2:]  # same windows, same builder arguments


def test_stage_inputs_single_window_and_local(monkeypatch, tmp_path):
    FakeFTW(monkeypatch)
    import agribound.composites as composites

    def no_builder(source):  # pragma: no cover - must not be called
        raise AssertionError("single-window and local inputs need no composite builder")

    monkeypatch.setattr(composites, "get_composite_builder", no_builder)
    raster = _s2_raster(tmp_path)
    single = _config(tmp_path, model="FTW_v2_3_Class_FULL_singleWindow")
    staged = ftw.FTWEngine.stage_inputs(single, raster)
    assert staged["n_windows"] == 1 and staged["rasters"] == [raster]
    assert staged["windows"] == {"single": {"raster": raster, "status": "input raster"}}
    stacked = _config(tmp_path, source="local", stacked_windows=True)
    staged = ftw.FTWEngine.stage_inputs(stacked, raster)
    assert staged["n_windows"] == 2 and staged["rasters"] == [raster, raster]
    assert staged["windows"]["b"]["bands"] == [5, 6, 7, 8]


def test_annual_fallback_only_replaces_windows_without_imagery(monkeypatch, tmp_path):
    """Transient builder errors (quota, auth, network) must never swap in the annual composite."""
    FakeFTW(monkeypatch)

    class FlakyBuilder(FakeBuilder):
        def build(self, config):
            if config.date_range[0].startswith("2025"):
                raise RuntimeError("Earth Engine quota exceeded")
            return super().build(config)

    builder = FlakyBuilder(tmp_path)
    import agribound.composites as composites

    monkeypatch.setattr(composites, "get_composite_builder", lambda source: builder)
    config = _config(tmp_path, allow_annual_fallback=True)
    with pytest.raises(RuntimeError, match="quota exceeded"):
        ftw.FTWEngine._two_windows(config, _s2_raster(tmp_path), config.engine_params, [4, 3, 2, 8])


def test_explicit_window_dates(monkeypatch, tmp_path):
    FakeFTW(monkeypatch)
    builder = FakeBuilder(tmp_path)
    import agribound.composites as composites

    monkeypatch.setattr(composites, "get_composite_builder", lambda source: builder)
    config = _config(tmp_path, window_dates=["2024-11-01", "2025-04-01"], window_days=5)
    ftw.FTWEngine._two_windows(config, _s2_raster(tmp_path), config.engine_params, [4, 3, 2, 8])
    assert [c[0] for c in builder.calls] == [
        ("2024-10-27", "2024-11-06"),
        ("2025-03-27", "2025-04-06"),
    ]
    with pytest.raises(ValueError, match="season order"):
        ftw._window_centres(config, "x", {"window_dates": ["2025-04-01", "2024-11-01"]})


def test_local_two_window_models_need_explicit_choice(tmp_path):
    raster = _s2_raster(tmp_path)
    with pytest.raises(ValueError, match="stacked_windows"):
        ftw.FTWEngine._two_windows_local(raster, {}, [1, 2, 3, 4])
    sources, _ = ftw.FTWEngine._two_windows_local(raster, {"stacked_windows": True}, [1, 2, 3, 4])
    assert sources == [(raster, [1, 2, 3, 4]), (raster, [5, 6, 7, 8])]
    sources, record = ftw.FTWEngine._two_windows_local(
        raster, {"allow_annual_fallback": True}, [1, 2, 3, 4]
    )
    assert record["a"]["status"] == "annual_fallback"


# ---------------------------------------------------------------------------
# Input raster
# ---------------------------------------------------------------------------


def test_write_ftw_input_order_units_and_nan(tmp_path):
    a = np.stack([np.full((8, 8), v, dtype=np.float32) for v in range(1, 13)])
    a[:, 0, 0] = np.nan
    b = a * 10
    pa = _write(tmp_path / "a.tif", a, nodata=float("nan"))
    pb = _write(tmp_path / "b.tif", b, nodata=float("nan"))
    out = ftw.write_ftw_input(
        tmp_path / "in.tif", [(pa, [4, 3, 2, 8]), (pb, [4, 3, 2, 8])], "sentinel2"
    )
    with rasterio.open(out) as src:
        data = src.read()
        assert src.nodata is None and src.dtypes[0] == "float32"
    assert data[:, 1, 1].tolist() == [4, 3, 2, 8, 40, 30, 20, 80]
    assert data[:, 0, 0].tolist() == [0.0] * 8


@pytest.mark.parametrize(("dtype", "nodata"), [("float32", -9999.0), ("uint16", 65535)])
def test_write_ftw_input_sets_declared_nodata_to_zero(tmp_path, dtype, nodata):
    data = np.full((4, 8, 8), 1500, dtype=dtype)
    data[:, :2, :2] = nodata
    path = _write(tmp_path / "nd.tif", data, nodata=nodata)
    out = ftw.write_ftw_input(
        tmp_path / "in.tif", [(path, [1, 2, 3, 4])], "local", "reflectance_x10000"
    )
    with rasterio.open(out) as src:
        written = src.read()
        assert src.nodata is None
    assert (written[:, :2, :2] == 0.0).all() and (written[:, 3:, 3:] == 1500.0).all()


def test_write_ftw_input_scales_unit_reflectance_and_regrids(tmp_path):
    unit = np.full((4, 8, 8), 0.25, dtype=np.float32)
    pa = _write(tmp_path / "a.tif", unit)
    shifted = _write(tmp_path / "b.tif", unit, transform=from_origin(500010, 4000000, 10.0, 10.0))
    out = ftw.write_ftw_input(
        tmp_path / "in.tif", [(pa, [1, 2, 3, 4]), (shifted, [1, 2, 3, 4])], "local", "unit"
    )
    with rasterio.open(out) as src:
        data = src.read()
    assert data[0, 3, 3] == pytest.approx(2500.0)
    assert data.shape == (8, 8, 8)
    assert data[4, 3, 0] == 0.0  # outside window B's footprint after regridding


# ---------------------------------------------------------------------------
# End-to-end with fake ftw-tools
# ---------------------------------------------------------------------------


def test_delineate_single_window_calls_and_meta(monkeypatch, tmp_path):
    fake = FakeFTW(monkeypatch)
    raster = _s2_raster(tmp_path)
    config = _config(tmp_path, model="FTW_v2_3_Class_FULL_singleWindow")
    gdf = ftw.FTWEngine().delineate(raster, config)
    call = fake.run_calls[0]
    assert call["model"] == "FTW_v2_3_Class_FULL_singleWindow"
    assert (call["gpu"], call["mps_mode"], call["save_scores"]) == (-1, False, False)
    assert "nan_fill_value" not in call
    assert call["patch_size"] == 128  # strictly smaller than the 160 px raster
    with rasterio.open(call["input"]) as src:
        data = src.read()
    assert data.shape[0] == 4 and np.isfinite(data).all() and data[0, 1, 1] == 1000.0
    poly = fake.poly_calls[0]
    assert poly["simplify"] == 0.0 and poly["min_size"] == 100.0
    assert poly["close_interiors"] is True
    meta = gdf.attrs["engine_meta"]
    assert meta["backend"] == "ftw-tools" and meta["n_windows"] == 1
    assert "copied unchanged" in meta["input_units"]
    assert "declared nodata -> 0" in meta["input_units"]
    assert "prediction_reprojected_to" not in meta
    assert "/" not in Path(call["out"]).name.replace(".tif", "")
    json.dumps(meta)
    assert len(gdf) == 1


def test_prediction_cache_is_keyed_by_registry_checkpoint_url(monkeypatch, tmp_path):
    fake = FakeFTW(monkeypatch)
    raster = _s2_raster(tmp_path)
    config = _config(tmp_path, model="FTW_v2_3_Class_FULL_singleWindow")
    ftw.FTWEngine().delineate(raster, config)
    gdf = ftw.FTWEngine().delineate(raster, config)
    assert len(fake.run_calls) == 1 and gdf.attrs["engine_meta"]["cached_prediction"] is True
    registry = sys.modules["ftw_tools.inference.model_registry"].MODEL_REGISTRY
    registry["FTW_v2_3_Class_FULL_singleWindow"] = _Spec(
        url="https://example/single-v2.ckpt", requires_window=False, legacy=True
    )
    gdf = ftw.FTWEngine().delineate(raster, config)
    assert len(fake.run_calls) == 2  # new weights URL -> new prediction
    assert gdf.attrs["engine_meta"]["checkpoint_url"] == "https://example/single-v2.ckpt"


def test_delineate_passes_nan_fill_when_supported(monkeypatch, tmp_path):
    fake = FakeFTW(monkeypatch, with_nan_fill=True)
    raster = _s2_raster(tmp_path)
    ftw.FTWEngine().delineate(raster, _config(tmp_path, model="FTW_v2_3_Class_FULL_singleWindow"))
    assert {"nan_fill_value": 0.0} in fake.run_calls


def test_delineate_geographic_prediction_is_polygonised_in_utm(monkeypatch, tmp_path):
    fake = FakeFTW(monkeypatch)
    deg = 8.983152841195215e-05
    raster = _s2_raster(tmp_path, crs="EPSG:4326", transform=from_origin(149.70, -30.38, deg, deg))
    config = _config(tmp_path, model="FTW_v2_3_Class_FULL_singleWindow")
    gdf = ftw.FTWEngine().delineate(raster, config)
    with rasterio.open(fake.poly_calls[0]["input"]) as src:
        assert src.crs.to_epsg() == 32755  # UTM 55S for 149.7E, 30.4S
    assert gdf.attrs["engine_meta"]["prediction_reprojected_to"] == "EPSG:32755"
    assert gdf.crs.to_epsg() == 32755


@pytest.mark.parametrize(
    ("crs", "reason"),
    [
        ("EPSG:32611", None),  # UTM
        ("EPSG:5070", None),  # CONUS Albers, metres
        ("EPSG:4326", "geographic CRS"),
        ("EPSG:3857", "Mercator projection"),
        ("EPSG:3395", "Mercator projection"),
        ("EPSG:2227", "linear unit US survey foot"),
    ],
)
def test_metric_reprojection_reason(crs, reason):
    assert ftw.metric_reprojection_reason(crs) == reason


def test_delineate_web_mercator_prediction_is_polygonised_in_utm(monkeypatch, tmp_path):
    fake = FakeFTW(monkeypatch)
    # ~10 m Web Mercator pixels near 149.7E, 30.4S
    raster = _s2_raster(
        tmp_path, crs="EPSG:3857", transform=from_origin(16664300.0, -3553700.0, 11.5, 11.5)
    )
    config = _config(tmp_path, model="FTW_v2_3_Class_FULL_singleWindow")
    gdf = ftw.FTWEngine().delineate(raster, config)
    with rasterio.open(fake.poly_calls[0]["input"]) as src:
        assert src.crs.to_epsg() == 32755
    meta = gdf.attrs["engine_meta"]
    assert meta["prediction_reprojected_to"] == "EPSG:32755"
    assert meta["prediction_reprojection_reason"] == "Mercator projection"


def test_close_interiors_with_vector_morphology_is_rejected_before_inference(monkeypatch, tmp_path):
    fake = FakeFTW(monkeypatch)  # fake polygonize, like 2.0.0b5, lacks the MultiPolygon fix
    raster = _s2_raster(tmp_path)
    config = _config(tmp_path, model="FTW_v2_3_Class_FULL_singleWindow", erode_dilate=20)
    with pytest.raises(ValueError, match="close_interiors"):
        ftw.FTWEngine().delineate(raster, config)
    assert fake.run_calls == []
    assert not list((tmp_path / "out").rglob("ftw_input_*.tif"))  # failed before any input
    config = _config(
        tmp_path, model="FTW_v2_3_Class_FULL_singleWindow", erode_dilate=20, close_interiors=False
    )
    ftw.FTWEngine().delineate(raster, config)
    assert fake.poly_calls[0]["erode_dilate"] == 20


def test_close_interiors_check_accepts_polygonize_with_multipolygon_fix():
    def fixed_polygonize(input, out, close_interiors=False, erode_dilate=0):  # noqa: A002
        # ftw-baselines main: MultiPolygon([shapely.geometry.Polygon(g.exterior) for g in ...])
        return None

    ftw._check_close_interiors(fixed_polygonize, {"close_interiors": True, "erode_dilate": 5})
    ftw._check_close_interiors(lambda **k: None, {"close_interiors": True, "erode_dilate": 0})


def test_interrupted_inference_is_not_reused_as_a_cached_prediction(monkeypatch, tmp_path):
    fake = FakeFTW(monkeypatch)
    inference = sys.modules["ftw_tools.inference.inference"]
    real_run = inference.run

    def crashing_run(**kwargs):
        real_run(**kwargs)  # the output file exists...
        raise KeyboardInterrupt  # ...but the run did not finish

    monkeypatch.setattr(inference, "run", crashing_run)
    raster = _s2_raster(tmp_path)
    config = _config(tmp_path, model="FTW_v2_3_Class_FULL_singleWindow")
    with pytest.raises(KeyboardInterrupt):
        ftw.FTWEngine().delineate(raster, config)
    assert Path(fake.run_calls[0]["out"]).name.endswith(".partial.tif")
    monkeypatch.setattr(inference, "run", real_run)
    gdf = ftw.FTWEngine().delineate(raster, config)
    assert len(fake.run_calls) == 2 and "cached_prediction" not in gdf.attrs["engine_meta"]


def test_prediction_cache_is_keyed_by_ftw_tools_version(monkeypatch, tmp_path):
    fake = FakeFTW(monkeypatch)
    raster = _s2_raster(tmp_path)
    config = _config(tmp_path, model="FTW_v2_3_Class_FULL_singleWindow")
    monkeypatch.setattr(ftw, "_ftw_version", lambda: "2.0.0b5")
    ftw.FTWEngine().delineate(raster, config)
    ftw.FTWEngine().delineate(raster, config)
    assert len(fake.run_calls) == 1
    monkeypatch.setattr(ftw, "_ftw_version", lambda: "2.1.0")
    ftw.FTWEngine().delineate(raster, config)
    assert len(fake.run_calls) == 2


def test_delineate_two_windows_end_to_end(monkeypatch, tmp_path):
    fake = FakeFTW(monkeypatch)
    builder = FakeBuilder(tmp_path)
    import agribound.composites as composites

    monkeypatch.setattr(composites, "get_composite_builder", lambda source: builder)
    raster = _s2_raster(tmp_path)
    gdf = ftw.FTWEngine().delineate(raster, _config(tmp_path))
    with rasterio.open(fake.run_calls[0]["input"]) as src:
        data = src.read()
    assert data.shape[0] == 8
    assert data[0, 1, 1] == 2001.0 and data[4, 1, 1] == 2002.0  # window A then window B
    windows = gdf.attrs["engine_meta"]["windows"]
    assert windows["a"]["start"] == "2024-09-05" and windows["b"]["end"] == "2025-07-10"


def test_softmax_threshold_implies_scores(monkeypatch, tmp_path):
    FakeFTW(monkeypatch)
    raster = _s2_raster(tmp_path)
    config = _config(tmp_path, model="FTW_v2_3_Class_FULL_singleWindow", save_scores=True)
    with pytest.raises(ValueError, match="softmax_threshold"):
        ftw.FTWEngine().delineate(raster, config)


def test_local_raster_needs_value_scale(monkeypatch, tmp_path):
    FakeFTW(monkeypatch)
    raster = _s2_raster(tmp_path, name="local.tif")
    config = _config(tmp_path, source="local", model="FTW_v2_3_Class_FULL_singleWindow")
    with pytest.raises(ValueError, match="value_scale"):
        ftw.FTWEngine().delineate(raster, config)


@pytest.mark.parametrize(("source", "flagged"), [("landsat", True), ("sentinel2", False)])
def test_non_sentinel2_sources_are_flagged_out_of_distribution(
    monkeypatch, tmp_path, caplog, source, flagged
):
    FakeFTW(monkeypatch)
    raster = _s2_raster(tmp_path)
    config = _config(tmp_path, source=source, model="FTW_v2_3_Class_FULL_singleWindow")
    with caplog.at_level("WARNING"):
        gdf = ftw.FTWEngine().delineate(raster, config)
    assert gdf.attrs["engine_meta"]["out_of_distribution_source"] is flagged
    assert ("out of distribution" in caplog.text) is flagged


def test_local_unit_reflectance_is_converted_to_s2_units(monkeypatch, tmp_path):
    fake = FakeFTW(monkeypatch)
    raster = _s2_raster(tmp_path, name="local.tif", value=0.1)  # 0-1 reflectance
    config = _config(
        tmp_path, source="local", model="FTW_v2_3_Class_FULL_singleWindow", value_scale="unit"
    )
    gdf = ftw.FTWEngine().delineate(raster, config)
    with rasterio.open(fake.run_calls[0]["input"]) as src:
        assert src.read(1)[1, 1] == pytest.approx(1000.0)  # 0.1 x 10000
    assert "converted from unit" in gdf.attrs["engine_meta"]["input_units"]


def test_prefetch_downloads_to_torch_hub_and_crop_calendar(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    FakeFTW(monkeypatch)
    downloads = []
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))

    def download(url, dst, progress=True):
        downloads.append(url)
        Path(dst).write_bytes(b"ckpt")

    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    calendar = types.ModuleType("ftw_tools.download.crop_calendar")
    calendar.ensure_crop_calendar_exists = lambda: tmp_path / "cal"
    settings = types.ModuleType("ftw_tools.settings")
    settings.CROP_CAL_SUMMER_START = "sc-sos-3x3-v2-cog.tiff"
    settings.CROP_CAL_SUMMER_END = "sc-eos-3x3-v2-cog.tiff"
    monkeypatch.setitem(sys.modules, "ftw_tools.download", types.ModuleType("ftw_tools.download"))
    monkeypatch.setitem(sys.modules, "ftw_tools.download.crop_calendar", calendar)
    monkeypatch.setitem(sys.modules, "ftw_tools.settings", settings)
    paths = ftw.FTWEngine.prefetch(_config(tmp_path))
    assert paths[0] == str(tmp_path / "hub" / "checkpoints" / "FTW_PRUE_EFNET_B5.ckpt")
    assert downloads == ["https://example/b5.ckpt"]
    assert paths[1:] == [
        str(tmp_path / "cal" / "sc-sos-3x3-v2-cog.tiff"),
        str(tmp_path / "cal" / "sc-eos-3x3-v2-cog.tiff"),
    ]
    ftw.FTWEngine.prefetch(_config(tmp_path))
    assert len(downloads) == 1  # cached


def test_real_registry_default_is_prue_b5():
    pytest.importorskip("ftw_tools.inference.model_registry")
    assert ftw.default_ftw_model() == "FTW_PRUE_EFNET_B5"
    models = ftw.list_ftw_models(include_legacy=True)
    assert models["FTW_PRUE_EFNET_B5"]["requires_window"] is True
    assert models["DelineateAnything"]["instance_segmentation"] is True


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((1024, 1024), 512), ((1025, 2000), 1024), ((600, 5000), 512), ((129, 129), 128)],
)
def test_patch_size_is_strictly_smaller_than_raster(shape, expected):
    assert ftw.select_patch_size(*shape) == expected


def test_patch_size_validation():
    with pytest.raises(ValueError, match="too small"):
        ftw.select_patch_size(128, 500)
    with pytest.raises(ValueError, match="smaller than"):
        ftw.select_patch_size(512, 512, 512)
    with pytest.raises(ValueError, match="multiple of 32"):
        ftw.select_patch_size(1000, 1000, 100)
    assert ftw.select_patch_size(1000, 1000, 256) == 256


def test_delineate_passes_explicit_patch_size(monkeypatch, tmp_path):
    fake = FakeFTW(monkeypatch)
    raster = _s2_raster(tmp_path, size=32)  # too small for any FTW patch
    config = _config(tmp_path, model="FTW_v2_3_Class_FULL_singleWindow")
    with pytest.raises(ValueError, match="too small"):
        ftw.FTWEngine().delineate(raster, config)
    assert fake.run_calls == []


# ---------------------------------------------------------------------------
# Provenance of the windows and of registry checkpoints
# ---------------------------------------------------------------------------


class TaggingBuilder(FakeBuilder):
    """Window composites carrying the tags the GEE builder writes."""

    def build(self, config):
        path = super().build(config)
        with rasterio.open(path, "r+") as dst:
            dst.update_tags(
                AGRIBOUND_N_IMAGES="11" if config.date_range[0] > "2025" else "17",
                AGRIBOUND_VALID_FRACTION="1.0",
                AGRIBOUND_CLOUD_MASK="SCL classes 3, 8, 9, 10 masked",
                AGRIBOUND_DATE_START=config.date_range[0],
            )
        return path


def test_window_records_carry_image_counts(monkeypatch, tmp_path):
    """Regression: per-window image counts were only in the window composites' tags."""
    FakeFTW(monkeypatch)
    import agribound.composites as composites

    monkeypatch.setattr(
        composites, "get_composite_builder", lambda source: TaggingBuilder(tmp_path)
    )
    config = _config(tmp_path, window_days=10)
    _, record = ftw.FTWEngine._two_windows(config, _s2_raster(tmp_path), {}, [4, 3, 2, 8])
    assert (record["a"]["n_images"], record["b"]["n_images"]) == (17, 11)
    assert record["b"]["valid_fraction"] == 1.0
    assert record["b"]["cloud_mask"].startswith("SCL classes")
    assert record["b"]["composite_date_start"] == record["b"]["start"]
    # A raster without tags adds nothing (and never fails).
    assert ftw._window_composite_facts(_s2_raster(tmp_path, "plain.tif")) == {}
    assert ftw._window_composite_facts(str(tmp_path / "missing.tif")) == {}


def test_registry_checkpoint_is_hashed(monkeypatch, tmp_path):
    """Regression: registry FTW checkpoints had checkpoint_sha256 = null."""
    torch = pytest.importorskip("torch")
    import hashlib

    ckpt_dir = tmp_path / "hub" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "FTW_PRUE_EFNET_B5.ckpt").write_bytes(b"weights v1")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    facts = ftw.registry_checkpoint_facts("FTW_PRUE_EFNET_B5")
    assert facts == {
        "checkpoint_path": str(ckpt_dir / "FTW_PRUE_EFNET_B5.ckpt"),
        "checkpoint_sha256": hashlib.sha256(b"weights v1").hexdigest(),
    }
    assert ftw.registry_checkpoint_facts("FTW_NOT_DOWNLOADED") == {}
