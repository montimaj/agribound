"""Class-code rasters (value scale "categorical") are read by engine crop-sequence only.

The refusal matrix of the crop-sequence design: every engine against
``cdl-sequence``, ``aci-sequence`` and a local raster tagged
``AGRIBOUND_VALUE_SCALE=categorical``; SAM refinement; fine-tuning; the
ensemble member check; and the reflectance conversions. Nothing here loads a
model or contacts Earth Engine.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine, get_engine_class
from agribound.io.raster import (
    raster_value_scale,
    refuse_categorical,
    to_s2_dn,
    to_unit_reflectance,
)
from agribound.registry import (
    CATEGORICAL_ENGINES,
    CROP_SEQUENCE_SOURCES,
    ENGINE_REGISTRY,
    VALUE_SCALES,
    engine_supports_source,
)

BBOX = "bbox:-93.62,41.98,-93.58,42.01"
#: Engines that read imagery (or embeddings): they all refuse class codes.
IMAGERY_ENGINES = [e for e in ENGINE_REGISTRY if e not in CATEGORICAL_ENGINES]


def _codes_raster(path, tag="categorical", count=8):
    data = np.ones((count, 16, 16), dtype=np.uint8)
    data[:, :, 8:] = 5
    profile = {
        "driver": "GTiff",
        "height": 16,
        "width": 16,
        "count": count,
        "dtype": "uint8",
        "crs": "EPSG:32615",
        "transform": Affine(30.0, 0, 600000.0, 0, -30.0, 4650000.0),
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i in range(count):
            dst.set_band_description(i + 1, f"CDL{2016 + i}")
        if tag:
            dst.update_tags(AGRIBOUND_VALUE_SCALE=tag)
    return str(path)


@pytest.fixture
def codes(tmp_path):
    return _codes_raster(tmp_path / "codes.tif")


def _local(tmp_path, raster, engine="delineate-anything", **kwargs):
    return AgriboundConfig(
        source="local",
        engine=engine,
        local_tif_path=raster,
        output_path=str(tmp_path / "out.gpkg"),
        cache_dir=str(tmp_path / "cache"),
        lulc_filter=False,
        **kwargs,
    )


def test_registry_constants():
    assert "categorical" in VALUE_SCALES
    assert CROP_SEQUENCE_SOURCES == ("cdl-sequence", "aci-sequence")
    assert CATEGORICAL_ENGINES == ("crop-sequence",)
    info = ENGINE_REGISTRY["crop-sequence"]
    assert info["supported_sources"] == ["cdl-sequence", "aci-sequence", "local"]
    assert info["label_free"] is True and info["fine_tunable"] is False
    assert info["gpu_recommended"] is False and info["requires_bands"] == []
    assert "not field instances" in info["notes"]
    assert "doi:10.3233/SJI-230078" in info["reference"]


# ---------------------------------------------------------------------------
# Engines x sources
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", CROP_SEQUENCE_SOURCES)
@pytest.mark.parametrize("engine", IMAGERY_ENGINES)
def test_imagery_engines_do_not_support_the_crop_map_sources(engine, source):
    assert not engine_supports_source(engine, source)
    with pytest.raises(ValueError, match=f"does not support source '{source}'"):
        AgriboundConfig(source=source, engine=engine, year=2023, study_area=BBOX, gee_project="p")


@pytest.mark.parametrize("source", CROP_SEQUENCE_SOURCES)
def test_the_crop_map_sources_accept_engine_crop_sequence(source):
    cfg = AgriboundConfig(
        source=source, engine="crop-sequence", year=2023, study_area=BBOX, gee_project="p"
    )
    assert cfg.engine == "crop-sequence" and cfg.requires_gee()


@pytest.mark.parametrize("engine", ["delineate-anything", "ftw", "geoai", "dinov3", "prithvi"])
def test_imagery_engines_refuse_a_local_categorical_raster(tmp_path, codes, engine):
    cfg = _local(tmp_path, codes, engine=engine)
    engine_obj = get_engine_class(engine)()
    with pytest.raises(ValueError, match="class codes of annual crop-type maps"):
        engine_obj.validate_input(codes, cfg)
    try:
        engine_obj.delineate(codes, cfg)
    except ValueError as exc:
        assert f"Engine '{engine}' cannot read" in str(exc)
    except ImportError:
        pass  # FTW, GeoAI and DINOv3 import their package first (not in the core install)
    else:  # pragma: no cover - a failure
        pytest.fail(f"engine {engine!r} delineated a raster of class codes")


@pytest.mark.parametrize("source", CROP_SEQUENCE_SOURCES)
@pytest.mark.parametrize("engine", ["delineate-anything", "ftw", "geoai", "dinov3", "prithvi"])
def test_validate_input_refuses_a_crop_map_source_without_a_tag(tmp_path, engine, source):
    """A configuration changed after validation is still refused at the engine."""
    untagged = _codes_raster(tmp_path / "untagged.tif", tag=None)
    cfg = _local(tmp_path, untagged, engine=engine)
    cfg.source = source
    with pytest.raises(ValueError, match=f"source '{source}' is a stack of crop-type maps"):
        get_engine_class(engine)().validate_input(untagged, cfg)


def test_validate_input_of_any_imagery_engine_name(tmp_path, codes):
    class Dummy(DelineationEngine):
        def delineate(self, raster_path, config):  # pragma: no cover - not called
            raise AssertionError

    for name in IMAGERY_ENGINES:
        engine = Dummy()
        engine.name = name
        with pytest.raises(ValueError, match="class codes"):
            engine.validate_input(codes, _local(tmp_path, codes))
    categorical = Dummy()
    categorical.name = "crop-sequence"
    categorical.validate_input(codes, _local(tmp_path, codes))  # not refused


def test_untagged_local_rasters_are_not_refused(tmp_path):
    untagged = _codes_raster(tmp_path / "untagged.tif", tag=None)
    refuse_categorical(untagged, _local(tmp_path, untagged), "Engine 'x'")
    unknown = _codes_raster(tmp_path / "unknown.tif", tag="unknown")
    refuse_categorical(unknown, _local(tmp_path, unknown), "Engine 'x'")
    assert raster_value_scale(unknown) == "unknown"
    assert raster_value_scale(untagged) is None
    assert raster_value_scale(tmp_path / "missing.tif") is None


def test_embedding_engine_refuses_class_codes(tmp_path, codes):
    from agribound.engines.embedding import EmbeddingEngine

    cfg = _local(tmp_path, codes, engine="embedding")
    with pytest.raises(ValueError, match="The embedding engine cannot read"):
        EmbeddingEngine().delineate(codes, cfg)
    for source in CROP_SEQUENCE_SOURCES:
        assert not engine_supports_source("embedding", source)


def test_ensemble_refuses_class_codes_and_crop_sequence_members(tmp_path, codes):
    from agribound.engines.ensemble import EnsembleEngine

    cfg = _local(
        tmp_path, codes, engine="ensemble", engine_params={"engines": ["delineate-anything"]}
    )
    with pytest.raises(ValueError, match="cannot read"):
        EnsembleEngine().delineate(codes, cfg)
    with pytest.raises(ValueError, match="cannot be an ensemble member"):
        _local(tmp_path, codes, engine="ensemble", engine_params={"engines": ["crop-sequence"]})
    with pytest.raises(ValueError, match="cannot be an ensemble member"):
        _local(
            tmp_path,
            codes,
            engine="ensemble",
            engine_params={"engines": [{"engine": "crop-sequence"}, "delineate-anything"]},
        )


# ---------------------------------------------------------------------------
# SAM refinement, fine-tuning, reflectance conversions
# ---------------------------------------------------------------------------


def test_sam_refinement_refuses_class_codes(tmp_path, codes):
    import geopandas as gpd
    from shapely.geometry import box

    from agribound.engines.samgeo_engine import refine_boundaries

    gdf = gpd.GeoDataFrame(geometry=[box(600000, 4649700, 600200, 4649900)], crs="EPSG:32615")
    with pytest.raises(ValueError, match="SAM refinement cannot read"):
        refine_boundaries(gdf, codes, _local(tmp_path, codes))


def test_sam_refine_is_refused_by_the_configuration(tmp_path, codes):
    with pytest.raises(ValueError, match="sam_refine does not apply to source 'cdl-sequence'"):
        AgriboundConfig(
            source="cdl-sequence",
            engine="crop-sequence",
            year=2023,
            study_area=BBOX,
            gee_project="p",
            sam_refine=True,
        )
    with pytest.raises(ValueError, match="sam_refine does not apply to engine 'crop-sequence'"):
        _local(
            tmp_path,
            codes,
            engine="crop-sequence",
            sam_refine=True,
            engine_params={"value_scale": "categorical", "dataset": "cdl"},
        )
    # The legacy engine_params switch is refused too.
    with pytest.raises(ValueError, match="sam_refine does not apply"):
        _local(
            tmp_path,
            codes,
            engine="crop-sequence",
            engine_params={"value_scale": "categorical", "dataset": "cdl", "sam_refine": True},
        )
    from agribound.pipeline import _NO_SAM_STAGE_ENGINES

    assert "crop-sequence" in _NO_SAM_STAGE_ENGINES


def test_fine_tuning_refuses_class_codes(tmp_path, codes, monkeypatch):
    import geopandas as gpd
    from shapely.geometry import box

    import agribound.engines.finetune as finetune
    from agribound.engines.finetune import _data, fine_tune
    from agribound.engines.finetune._data import _prepare_training_data

    ref = tmp_path / "ref.gpkg"
    gpd.GeoDataFrame(geometry=[box(600000, 4649700, 600200, 4649900)], crs="EPSG:32615").to_file(
        ref
    )
    cfg = _local(tmp_path, codes, fine_tune=True, reference_boundaries=str(ref))

    # fine_tune refuses on its own, before it looks for a cached checkpoint or prepares
    # the training data (whose own refusal is checked below).
    def went_past_the_refusal(*args, **kwargs):
        raise AssertionError("fine_tune went past its refusal")

    with monkeypatch.context() as patch:
        patch.setattr(finetune, "_run_dir", went_past_the_refusal)
        patch.setattr(_data, "_prepare_training_data", went_past_the_refusal)
        with pytest.raises(ValueError, match="Fine-tuning cannot read"):
            fine_tune(codes, cfg)
    with pytest.raises(ValueError, match="Fine-tuning cannot read"):
        _prepare_training_data(codes, cfg, "delineate-anything")
    with pytest.raises(ValueError, match="does not support fine-tuning"):
        _local(
            tmp_path,
            codes,
            engine="crop-sequence",
            fine_tune=True,
            reference_boundaries=str(ref),
            engine_params={"value_scale": "categorical", "dataset": "cdl"},
        )


@pytest.mark.parametrize("convert", [to_unit_reflectance, to_s2_dn])
def test_reflectance_conversions_refuse_categorical(convert):
    arr = np.array([[1, 5], [121, 176]], dtype=np.uint8)
    for source in CROP_SEQUENCE_SOURCES:
        with pytest.raises(ValueError, match="holds class codes"):
            convert(arr, source)
    with pytest.raises(ValueError, match="holds class codes"):
        convert(arr, "local", value_scale="categorical")
    with pytest.raises(ValueError, match="holds class codes"):
        convert(arr, "local", allow_unknown=True, value_scale="categorical")


# ---------------------------------------------------------------------------
# The local builder keeps the categorical tag
# ---------------------------------------------------------------------------


def test_local_builder_propagates_the_categorical_value_scale(tmp_path):
    from agribound.composites.local import LocalCompositeBuilder

    codes = _codes_raster(tmp_path / "codes.tif")
    with rasterio.open(codes) as src:
        bounds = src.bounds
    import geopandas as gpd
    from shapely.geometry import box

    inner = gpd.GeoSeries(
        [box(bounds.left + 60, bounds.bottom + 60, bounds.right - 90, bounds.top - 90)],
        crs="EPSG:32615",
    ).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = inner.total_bounds
    cfg = AgriboundConfig(
        source="local",
        local_tif_path=codes,
        study_area=f"bbox:{minx},{miny},{maxx},{maxy}",
        output_path=str(tmp_path / "o.gpkg"),
        cache_dir=str(tmp_path / "cache"),
        lulc_filter=False,
    )
    builder = LocalCompositeBuilder()
    cropped = builder.build(cfg)
    assert cropped != codes
    assert raster_value_scale(cropped) == "categorical"
    assert builder.last_metadata["AGRIBOUND_VALUE_SCALE"] == "categorical"
    with pytest.raises(ValueError, match="class codes"):
        get_engine_class("delineate-anything")().validate_input(cropped, cfg)
    # A file without the tag keeps the value scale "unknown".
    plain = _codes_raster(tmp_path / "plain.tif", tag=None)
    builder.build(cfg.merged(local_tif_path=plain))
    assert builder.last_metadata["AGRIBOUND_VALUE_SCALE"] == "unknown"
    # Another value scale of the file is not carried over (only "categorical" is).
    imagery = _codes_raster(tmp_path / "img.tif", tag="reflectance_x10000")
    builder.build(cfg.merged(local_tif_path=imagery))
    assert builder.last_metadata["AGRIBOUND_VALUE_SCALE"] == "unknown"
