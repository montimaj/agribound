"""Tests for the source/engine registries and engine discovery."""

from __future__ import annotations

import importlib

import pytest

from agribound import registry
from agribound.composites.base import SOURCE_REGISTRY as COMPOSITES_SOURCE_REGISTRY
from agribound.composites.base import list_sources as composites_list_sources
from agribound.engines.base import (
    ENGINE_REGISTRY,
    DelineationEngine,
    get_canonical_band_indices,
    get_engine,
    get_engine_class,
    list_engines,
)


def test_landsat_pan_rgb_and_scale():
    assert get_canonical_band_indices("landsat-pan", ["R", "G", "B"]) == [1, 1, 1]
    assert registry.source_value_scale("landsat-pan") == "unit"
    assert registry.source_year_range("landsat-pan") == (1999, None)
    assert registry.engine_supports_source("delineate-anything", "landsat-pan")
    assert not registry.engine_supports_source("ftw", "landsat-pan")
    assert not registry.engine_supports_source("prithvi", "landsat-pan")


def test_landsat_pan_is_listed_right_after_landsat():
    sources = list(registry.SOURCE_REGISTRY)
    assert sources.index("landsat-pan") == sources.index("landsat") + 1
    assert list(registry.list_sources()) == sources


def test_delineate_anything_notes_landsat_pan_resolution():
    notes = registry.engine_notes("delineate-anything", "landsat-pan")
    (pan,) = [n for n in notes if "Landsat PAN" in n]
    assert pan.startswith("15 m Landsat PAN composites are outside the 0.25-10 m training range")
    assert "grey R, G, B" in pan
    assert not any(n.startswith("30 m Landsat") for n in notes)  # the landsat note only
    assert not any(
        "Landsat PAN" in n for n in registry.engine_notes("delineate-anything", "landsat")
    )


EXPECTED_ENGINES = {
    "delineate-anything",
    "ftw",
    "geoai",
    "dinov3",
    "prithvi",
    "embedding",
    "ensemble",
}
EXPECTED_SOURCES = {
    "landsat",
    "landsat-pan",
    "sentinel2",
    "hls",
    "naip",
    "usgs-naip-plus",
    "spot",
    "spot-pan",
    "local",
    "google-embedding",
    "tessera-embedding",
}
ENGINE_KEYS = {
    "name",
    "approach",
    "strengths",
    "gpu_recommended",
    "requires_bands",
    "supported_sources",
    "label_free",
    "fine_tunable",
    "reference",
    "install_extra",
}
SOURCE_KEYS = {
    "name",
    "collection",
    "resolution_m",
    "native_resolution_m",
    "all_bands",
    "canonical_bands",
    "value_scale",
    "year_range",
    "coverage",
    "requires_gee",
    "restricted",
}


class TestReExports:
    def test_engine_registry_is_shared(self):
        assert ENGINE_REGISTRY is registry.ENGINE_REGISTRY

    def test_source_registry_is_shared(self):
        assert COMPOSITES_SOURCE_REGISTRY is registry.SOURCE_REGISTRY

    def test_list_functions(self):
        assert composites_list_sources() == registry.list_sources()
        assert list_engines() == registry.list_engines()


class TestListEngines:
    """Test the list_engines function."""

    def test_expected_engine_names(self):
        assert set(list_engines()) == EXPECTED_ENGINES

    def test_each_engine_has_required_metadata(self):
        for name, info in list_engines().items():
            missing = ENGINE_KEYS - set(info)
            assert not missing, f"Engine {name!r} missing metadata keys {missing}"
            assert isinstance(info["gpu_recommended"], bool)
            assert isinstance(info["label_free"], bool)
            assert isinstance(info["fine_tunable"], bool)
            assert set(info["supported_sources"]) <= EXPECTED_SOURCES
            assert set(info["requires_bands"]) <= set(registry.CANONICAL_BAND_NAMES)

    def test_returns_copy(self):
        """list_engines returns a deep copy, not the mutable registry."""
        engines = list_engines()
        engines["fake"] = {}
        engines["ftw"]["supported_sources"].append("naip")
        assert "fake" not in ENGINE_REGISTRY
        assert "naip" not in ENGINE_REGISTRY["ftw"]["supported_sources"]

    def test_verified_facts(self):
        ftw = ENGINE_REGISTRY["ftw"]
        assert ftw["requires_bands"] == ["R", "G", "B", "NIR"]
        assert set(ftw["supported_sources"]) == {"sentinel2", "hls", "landsat", "local"}
        assert ftw["label_free"] is True and ftw["fine_tunable"] is False
        assert "10.1609/aaai.v39i27.35034" in ftw["reference"]
        assert "24 countries" in ftw["strengths"]

        da = ENGINE_REGISTRY["delineate-anything"]
        assert {"spot-pan", "usgs-naip-plus"} <= set(da["supported_sources"])
        assert da["label_free"] is True and da["fine_tunable"] is True
        assert "2504.02534" in da["reference"] and "2607.19069" in da["reference"]

        assert ENGINE_REGISTRY["geoai"]["label_free"] is False
        assert "10.21105/joss.09605" in ENGINE_REGISTRY["geoai"]["reference"]
        assert ENGINE_REGISTRY["dinov3"]["label_free"] is False
        assert "2508.10104" in ENGINE_REGISTRY["dinov3"]["reference"]
        prithvi = ENGINE_REGISTRY["prithvi"]
        assert "10.1109/TGRS.2025.3642610" in prithvi["reference"]
        assert {"SWIR1", "SWIR2"} <= set(prithvi["requires_bands"])
        emb = ENGINE_REGISTRY["embedding"]
        assert emb["gpu_recommended"] is False and emb["label_free"] is True
        assert "2507.22291" in emb["reference"] and "CVPR" in emb["reference"]

    def test_engine_notes_are_source_specific(self):
        for name, info in ENGINE_REGISTRY.items():
            for source in info.get("source_notes") or {}:
                assert source in info["supported_sources"], (name, source)
        da_s2 = " ".join(registry.engine_notes("delineate-anything", "sentinel2"))
        assert "Landsat" not in da_s2 and "HLS" not in da_s2 and "AGPL-3.0" in da_s2
        assert any("Landsat" in n for n in registry.engine_notes("delineate-anything", "landsat"))
        assert any("HLS" in n for n in registry.engine_notes("ftw", "hls"))
        assert not any(
            "out of distribution" in n for n in registry.engine_notes("ftw", "sentinel2")
        )
        prithvi = registry.engine_notes("prithvi", "landsat")
        assert "NIR_NARROW" in prithvi[0] and "Landsat 5/7" in prithvi[1]
        assert registry.engine_notes("embedding") == [ENGINE_REGISTRY["embedding"]["notes"]]
        with pytest.raises(ValueError, match="Unknown engine"):
            registry.engine_notes("nope")

    def test_ensemble_default_members(self):
        from agribound.engines.ensemble import DEFAULT_MEMBERS

        assert DEFAULT_MEMBERS == registry.ENSEMBLE_DEFAULT_MEMBERS == ("delineate-anything", "ftw")

    def test_sam_backends(self):
        assert registry.SAM_REFINE_BACKENDS == ("sam2", "sam2.1", "sam3", "sam3-hf")
        assert registry.sam_refine_backends == ["sam2", "sam2.1", "sam3", "sam3-hf"]
        assert registry.list_sam_backends() == list(registry.SAM_REFINE_BACKENDS)


class TestSourceRegistry:
    def test_expected_sources(self):
        assert set(registry.list_sources()) == EXPECTED_SOURCES

    def test_each_source_has_required_keys(self):
        for name, info in registry.SOURCE_REGISTRY.items():
            missing = SOURCE_KEYS - set(info)
            assert not missing, f"Source {name!r} missing keys {missing}"
            assert info["value_scale"] in registry.VALUE_SCALES
            assert isinstance(info["restricted"], bool)
            canonical = info["canonical_bands"] or {}
            for native in canonical.values():
                assert native in info["all_bands"], (name, native)

    def test_value_scales(self):
        assert registry.source_value_scale("sentinel2") == "reflectance_x10000"
        assert registry.source_value_scale("landsat") == "reflectance_x10000"
        assert registry.source_value_scale("hls") == "reflectance_x10000"
        assert registry.source_value_scale("naip") == "uint8"
        assert registry.source_value_scale("usgs-naip-plus") == "uint8"
        assert registry.source_value_scale("spot") == "dn"
        assert registry.source_value_scale("tessera-embedding") == "embedding"
        assert registry.source_value_scale("local") == "unknown"
        with pytest.raises(ValueError, match="Unknown source"):
            registry.source_value_scale("modis")

    def test_dn_and_coverage_texts(self):
        doc = registry.__doc__
        assert "medians" in doc and "float32" in doc and "half-integer" in doc
        for source in ("spot", "spot-pan"):
            assert "medians of raw DN" in registry.SOURCE_REGISTRY[source]["coverage"]
        naip_plus = registry.SOURCE_REGISTRY["usgs-naip-plus"]["coverage"]
        assert "Alaska" in naip_plus and "latest NAIP/HRO vintage per state" in naip_plus

    def test_year_ranges(self):
        assert registry.source_year_range("landsat") == (1984, None)
        assert registry.source_year_range("sentinel2") == (2017, None)
        assert registry.source_year_range("hls") == (2013, None)
        assert registry.source_year_range("naip") == (2002, 2023)
        assert registry.source_year_range("usgs-naip-plus") == (2012, 2023)
        assert registry.source_year_range("spot") == (2012, 2023)
        assert registry.source_year_range("google-embedding") == (2017, 2025)
        assert registry.source_year_range("tessera-embedding") == (2017, 2025)
        assert registry.source_year_range("tessera-embedding", tessera_version="v1.1") == (
            2015,
            2025,
        )
        assert registry.source_year_range("local") is None
        with pytest.raises(ValueError, match="tessera_version"):
            registry.source_year_range("tessera-embedding", tessera_version="v9")

    def test_spot_has_nir(self):
        spot = registry.SOURCE_REGISTRY["spot"]
        assert spot["all_bands"] == ["R", "G", "B", "N"]
        assert spot["canonical_bands"]["NIR"] == "N"
        assert registry.SOURCE_REGISTRY["spot-pan"]["resolution_m"] == 1.5
        assert spot["restricted"] is True

    def test_landsat_lists_all_missions(self):
        landsat = registry.SOURCE_REGISTRY["landsat"]
        for mission in ("LT05", "LE07", "LC08", "LC09"):
            assert mission in landsat["collection"]

    def test_engine_supports_source(self):
        assert registry.engine_supports_source("ftw", "sentinel2") is True
        assert registry.engine_supports_source("FTW", "naip") is False
        assert registry.engine_supports_source("embedding", "tessera-embedding") is True
        assert registry.engine_supports_source("nope", "sentinel2") is False
        assert registry.supported_sources("prithvi") == ["landsat", "sentinel2", "hls", "local"]
        with pytest.raises(ValueError):
            registry.supported_sources("nope")

    def test_every_engine_source_pair_consistent_with_bands(self):
        """Engines only list sources that provide their required bands."""
        for info in ENGINE_REGISTRY.values():
            for source in info["supported_sources"]:
                if source == "local" or not info["requires_bands"]:
                    continue
                get_canonical_band_indices(source, info["requires_bands"])  # must not raise


class TestCanonicalBandIndices:
    def test_sentinel2(self):
        assert get_canonical_band_indices("sentinel2", ["R", "G", "B", "NIR"]) == [4, 3, 2, 8]
        assert get_canonical_band_indices("sentinel2", ["NIR_NARROW", "SWIR1", "SWIR2"]) == [
            9,
            11,
            12,
        ]

    def test_landsat_and_hls(self):
        assert get_canonical_band_indices("landsat", ["R", "NIR", "SWIR1", "SWIR2"]) == [3, 4, 5, 6]
        assert get_canonical_band_indices("hls", ["B", "NIR", "NIR_NARROW", "SWIR2"]) == [
            2,
            5,
            5,
            7,
        ]

    def test_naip_and_spot(self):
        assert get_canonical_band_indices("naip", ["R", "G", "B", "NIR"]) == [1, 2, 3, 4]
        assert get_canonical_band_indices("spot", ["NIR"]) == [4]
        assert get_canonical_band_indices("spot-pan", ["R", "G", "B"]) == [1, 1, 1]

    def test_local_positional(self):
        assert get_canonical_band_indices("local", ["R", "G", "B", "NIR"]) == [1, 2, 3, 4]

    def test_explicit_mapping(self):
        assert get_canonical_band_indices("local", ["R", "G", "B"], bands={"R": 3, "B": 1}) == [
            3,
            2,
            1,
        ]
        assert get_canonical_band_indices("naip", ["R"], bands={"R": 2}) == [2]
        with pytest.raises(ValueError, match=">= 1"):
            get_canonical_band_indices("local", ["R"], bands={"R": 0})

    def test_missing_band(self):
        with pytest.raises(ValueError, match="not defined for source 'naip'"):
            get_canonical_band_indices("naip", ["SWIR1"])
        with pytest.raises(ValueError, match="Unknown source"):
            get_canonical_band_indices("modis", ["R"])


class _Dummy(DelineationEngine):
    name = "dummy"
    requires_bands = ["R", "G", "B", "NIR"]

    def delineate(self, raster_path, config):  # pragma: no cover - not called
        raise NotImplementedError


class TestDelineationEngineBase:
    def test_prefetch_default_returns_empty(self):
        assert _Dummy.prefetch(object()) == []
        assert DelineationEngine.prefetch(object()) == []

    def test_validate_input_band_count(self, sample_rgb_tif, sample_rgbn_tif):
        from agribound.config import AgriboundConfig

        cfg = AgriboundConfig(source="local", local_tif_path=sample_rgb_tif, lulc_filter=False)
        with pytest.raises(ValueError, match="at least 4 bands"):
            _Dummy().validate_input(sample_rgb_tif, cfg)
        _Dummy().validate_input(sample_rgbn_tif, cfg)

    def test_validate_input_uses_source_indices(self, sample_rgbn_tif):
        from agribound.config import AgriboundConfig

        cfg = AgriboundConfig(source="sentinel2", gee_project="p")
        # S2 NIR is band 8 of the composite, so a 4-band raster is too small.
        with pytest.raises(ValueError, match="at least 8 bands"):
            _Dummy().validate_input(sample_rgbn_tif, cfg)

    def test_validate_input_honours_band_override_for_local(self, sample_rgbn_tif):
        """bands={'NIR': 8} on a 4-band local raster fails validation, not the later read."""
        from agribound.config import AgriboundConfig

        cfg = AgriboundConfig(
            source="local", local_tif_path=sample_rgbn_tif, lulc_filter=False, bands={"NIR": 8}
        )
        with pytest.raises(ValueError, match=r"bands=\{'NIR': 8\}, i\.e\. at least 8 bands"):
            _Dummy().validate_input(sample_rgbn_tif, cfg)
        ok = cfg.merged(bands={"R": 1, "G": 2, "B": 3, "NIR": 4})
        _Dummy().validate_input(sample_rgbn_tif, ok)

    def test_validate_input_honours_band_override_for_gee_sources(self, sample_rgbn_tif):
        """A band mapping replaces the registry indices (S2 NIR = 8) in the count check."""
        from agribound.config import AgriboundConfig

        cfg = AgriboundConfig(
            source="sentinel2", gee_project="p", bands={"R": 1, "G": 2, "B": 3, "NIR": 4}
        )
        _Dummy().validate_input(sample_rgbn_tif, cfg)


class TestGetEngine:
    """Test the get_engine factory function."""

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine("nonexistent_engine")

    def test_unknown_engine_empty_string(self):
        with pytest.raises(ValueError):
            get_engine("")

    def test_class_map_covers_registry(self):
        assert set(registry.ENGINE_CLASSES) == set(ENGINE_REGISTRY)
        for target in registry.ENGINE_CLASSES.values():
            module, _, cls = target.partition(":")
            assert module.startswith("agribound.engines.") and cls

    def test_get_engine_resolves_class_path(self, monkeypatch):
        import types

        fake = types.ModuleType("agribound.engines._fake_engine")

        class FakeEngine(_Dummy):
            pass

        fake.FakeEngine = FakeEngine
        monkeypatch.setitem(importlib.sys.modules, "agribound.engines._fake_engine", fake)
        monkeypatch.setitem(
            registry.ENGINE_CLASSES, "ftw", "agribound.engines._fake_engine:FakeEngine"
        )
        assert get_engine_class("FTW") is FakeEngine
        assert isinstance(get_engine("ftw"), FakeEngine)
