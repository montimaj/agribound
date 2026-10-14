"""Tests for agribound.provenance (run records and config hashing)."""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pytest
import yaml

from agribound import provenance
from agribound._version import __version__
from agribound.config import AgriboundConfig
from agribound.provenance import (
    HASH_CONDITIONAL_FIELDS,
    HASH_EXCLUDED_FIELDS,
    RunRecorder,
    canonical_config,
    config_hash,
    drop_inapplicable_fields,
    provenance_path,
    read_provenance,
    reuse_mismatch,
    to_jsonable,
    write_provenance,
)

#: Configuration fields added after agribound 1.0.1.
NEW_FIELDS = (
    "landsat_pan_missions",
    "lulc_tree_crops",
    "vhr_sar_collect_ids",
    "vhr_sar_max_collects",
    "vhr_sar_polarization",
    "vhr_sar_resolution_m",
    "vhr_sar_coregister",
    "vhr_sar_coregister_base",
    "vhr_sar_bands",
    "vhr_sar_calibration",
    "vhr_sar_max_download_gb",
    "vhr_sar_cache_dir",
    "crop_sequence_years",
)
#: The AgriboundConfig fields of agribound 1.0.1 (the same as in 1.0.0).
FIELDS_1_0_1 = frozenset(
    {
        "source",
        "engine",
        "year",
        "study_area",
        "output_path",
        "output_format",
        "gee_project",
        "export_method",
        "gcs_bucket",
        "gee_service_account_key",
        "gee_high_volume",
        "gee_max_requests",
        "gee_workload_tag",
        "usgs_service_url",
        "usgs_state",
        "usgs_allow_year_fallback",
        "usgs_timeout_s",
        "usgs_retries",
        "composite_method",
        "date_range",
        "cloud_cover_max",
        "export_crs",
        "s2_cloud_mask",
        "cloud_score_threshold",
        "naip_resolution_m",
        "tessera_version",
        "tessera_variant",
        "embedding_cache_dir",
        "google_embedding_backend",
        "local_tif_path",
        "bands",
        "aoi_selection",
        "min_field_area_m2",
        "simplify_tolerance",
        "lulc_filter",
        "lulc_crop_threshold",
        "lulc_batch_size",
        "lulc_dataset",
        "lulc_on_error",
        "lulc_mode",
        "lulc_nodata_policy",
        "sam_refine",
        "sam_backend",
        "sam_model",
        "sam_min_crop_px",
        "sam_crop_padding",
        "device",
        "tile_size",
        "n_workers",
        "seed",
        "reference_boundaries",
        "fine_tune",
        "fine_tune_epochs",
        "fine_tune_val_split",
        "fine_tune_split",
        "fine_tune_block_size_m",
        "fine_tune_split_column",
        "cache_dir",
        "overwrite",
        "provenance",
        "engine_params",
    }
)
BBOX = "bbox:-117.0,36.0,-116.99,36.01"


def _hash_without_new_fields(config) -> str:
    """The configuration hash as agribound 1.0.1 computed it (no conditional fields)."""
    data = config.to_dict()
    canonical = {
        k: to_jsonable(v)
        for k, v in sorted(data.items())
        if k not in HASH_EXCLUDED_FIELDS and k not in NEW_FIELDS
    }
    text = yaml.safe_dump(canonical, sort_keys=True, default_flow_style=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _gee(source, **kwargs):
    kwargs.setdefault("year", 2023)
    return AgriboundConfig(source=source, study_area=BBOX, gee_project="p", **kwargs)


@pytest.fixture
def cfg(tmp_path):
    return AgriboundConfig(
        source="local",
        local_tif_path="/tmp/x.tif",
        output_path=str(tmp_path / "fields.gpkg"),
        lulc_filter=False,
        device="cpu",
        gee_workload_tag="agribound-test",
    )


class TestConfigHash:
    def test_stable(self, cfg):
        h = config_hash(cfg)
        assert len(h) == 40 and int(h, 16) >= 0
        assert config_hash(cfg.merged()) == h
        assert config_hash(AgriboundConfig.from_dict(cfg.to_dict())) == h

    @pytest.mark.parametrize(
        "override",
        [
            {"output_path": "other/place.gpkg"},
            {"output_format": "parquet", "output_path": "x.parquet"},
            {"overwrite": True},
            {"provenance": False},
            {"cache_dir": "/scratch/c"},
            {"device": "cuda"},
            {"n_workers": 16},
            {"gee_max_requests": 2},
            {"gee_workload_tag": "other-tag"},
            {"lulc_batch_size": 50},
        ],
    )
    def test_excluded_fields_do_not_change_hash(self, cfg, override):
        assert config_hash(cfg.merged(**override)) == config_hash(cfg)

    @pytest.mark.parametrize(
        "override",
        [
            {"engine": "ftw"},
            {"year": 2020},
            {"min_field_area_m2": 1.0},
            {"lulc_filter": True},
            {"seed": 1},
            {"engine_params": {"model": "x"}},
            {"sam_refine": True},
            {"study_area": "bbox:0,0,1,1"},
            {"lulc_tree_crops": True},
        ],
    )
    def test_result_fields_change_hash(self, cfg, override):
        assert config_hash(cfg.merged(**override)) != config_hash(cfg)

    def test_excluded_fields_are_config_fields(self):
        assert set(AgriboundConfig.field_names()) >= HASH_EXCLUDED_FIELDS


class TestConditionalHashFields:
    """Fields added after 1.0.1 enter the hash only where they apply."""

    def test_exported_and_consistent(self):
        assert "HASH_CONDITIONAL_FIELDS" in provenance.__all__
        assert set(HASH_CONDITIONAL_FIELDS) == set(NEW_FIELDS)
        assert set(HASH_CONDITIONAL_FIELDS) <= set(AgriboundConfig.field_names())
        assert not set(HASH_CONDITIONAL_FIELDS) & HASH_EXCLUDED_FIELDS
        assert all(callable(rule) for rule in HASH_CONDITIONAL_FIELDS.values())

    @pytest.mark.parametrize(
        "make",
        [
            lambda tmp: AgriboundConfig(
                source="local", local_tif_path="/tmp/x.tif", output_path=str(tmp / "f.gpkg")
            ),
            lambda tmp: _gee("sentinel2"),
            lambda tmp: _gee("landsat", year=2015, composite_method="greenest"),
            lambda tmp: _gee("naip", year=2022, lulc_dataset="dynamic_world"),
            lambda tmp: _gee("sentinel2", engine="ftw", lulc_dataset="c3s", lulc_mode="raster"),
            lambda tmp: _gee("sentinel2", lulc_tree_crops=False),
            # landsat_pan_missions is ignored (and not hashed) for other sources
            lambda tmp: _gee("sentinel2", landsat_pan_missions=("LE07", "LC08")),
            lambda tmp: _gee("landsat", year=2015, landsat_pan_missions="LC09"),
        ],
    )
    def test_hash_unchanged_where_the_new_fields_do_not_apply(self, tmp_path, make):
        config = make(tmp_path)
        assert config_hash(config) == _hash_without_new_fields(config)
        assert not set(NEW_FIELDS) & set(canonical_config(config))

    def test_landsat_pan_always_hashes_its_missions(self):
        auto = _gee("landsat-pan")
        assert canonical_config(auto)["landsat_pan_missions"] == "auto"
        # Even at the default: outputs of the code that mixed Landsat 7 and 8/9 PAN
        # (hashed without the field) are not reused.
        assert config_hash(auto) != _hash_without_new_fields(auto)
        oli = auto.merged(landsat_pan_missions="LC09,LC08")
        assert canonical_config(oli)["landsat_pan_missions"] == ["LC08", "LC09"]
        assert config_hash(oli) == config_hash(auto.merged(landsat_pan_missions=["LC08", "LC09"]))
        hashes = {
            config_hash(auto.merged(landsat_pan_missions=value))
            for value in ("auto", "LC08", "LE07", "LC08,LC09", "LE07,LC08")
        }
        assert len(hashes) == 5
        assert "lulc_tree_crops" not in canonical_config(auto)

    def test_lulc_tree_crops_true_changes_the_hash(self, cfg):
        tree = cfg.merged(lulc_tree_crops=True)
        assert canonical_config(tree)["lulc_tree_crops"] is True
        assert config_hash(tree) != config_hash(cfg)
        assert config_hash(tree.merged(lulc_tree_crops=False)) == config_hash(cfg)
        assert config_hash(cfg) == _hash_without_new_fields(cfg)

    def test_dictionaries_hash_like_configurations(self):
        for config in (
            _gee("landsat-pan", landsat_pan_missions="LE07", lulc_tree_crops=True),
            _gee("sentinel2", landsat_pan_missions="LE07"),
        ):
            assert canonical_config(config.to_dict()) == canonical_config(config)
            assert config_hash(config.to_dict()) == config_hash(config)

    def test_record_hashed_without_the_missions_is_not_reused_for_landsat_pan(self):
        from agribound._results import results_versions

        def record(config):
            return {
                "status": "success",
                "config_hash": _hash_without_new_fields(config),
                "facts": {"results_versions": results_versions(config)},
            }

        landsat_pan = _gee("landsat-pan")
        reason = reuse_mismatch(record(landsat_pan), landsat_pan)
        assert reason is not None and reason.startswith("it was produced with a different")
        sentinel2 = _gee("sentinel2")
        assert reuse_mismatch(record(sentinel2), sentinel2) is None

    def test_fields_added_after_1_0_1_are_conditional(self):
        """Every field added after 1.0.1 must be listed in HASH_CONDITIONAL_FIELDS.

        Otherwise it enters every tile-manifest signature and agent plan directory
        (and, unless it is excluded, every config_hash), so the manifests, plan
        directories and outputs of earlier configurations stop being current.
        """
        fields = set(AgriboundConfig.field_names())
        assert fields >= FIELDS_1_0_1
        assert fields - FIELDS_1_0_1 == set(HASH_CONDITIONAL_FIELDS)

    def test_defaults_of_the_conditional_fields(self):
        assert provenance._conditional_field_defaults() == {
            "landsat_pan_missions": "auto",
            "lulc_tree_crops": False,
            "vhr_sar_collect_ids": None,
            "vhr_sar_max_collects": 6,
            "vhr_sar_polarization": "auto",
            "vhr_sar_resolution_m": 1.0,
            "vhr_sar_coregister": "auto",
            "vhr_sar_coregister_base": None,
            "vhr_sar_bands": "median",
            "vhr_sar_calibration": "any",
            "vhr_sar_max_download_gb": 5.0,
            "vhr_sar_cache_dir": None,
            "crop_sequence_years": 8,
        }

    def test_crop_sequence_years_applies_only_to_the_crop_map_sources(self):
        """Hashed at any value for cdl-sequence/aci-sequence; never for another source."""
        for source in ("sentinel2", "landsat-pan", "hls"):
            config = _gee(source)
            assert "crop_sequence_years" not in canonical_config(config)
            for n in (2, 5, 20):
                assert config_hash(config.merged(crop_sequence_years=n)) == config_hash(config)
            if source != "landsat-pan":  # landsat-pan always hashes its missions
                assert config_hash(config) == _hash_without_new_fields(config)
        local = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", lulc_filter=False)
        assert config_hash(local.merged(crop_sequence_years=3)) == config_hash(local)
        for source in ("cdl-sequence", "aci-sequence"):
            stack = _gee(source, engine="crop-sequence", lulc_filter=False)
            assert canonical_config(stack)["crop_sequence_years"] == 8
            hashes = {config_hash(stack.merged(crop_sequence_years=n)) for n in (2, 5, 8)}
            assert len(hashes) == 3
        # Tile manifests and plan directories: unchanged while the field holds its default.
        s2 = _gee("sentinel2").to_dict()
        assert "crop_sequence_years" not in drop_inapplicable_fields(s2, keep_non_default=True)
        cdl = _gee("cdl-sequence", engine="crop-sequence").to_dict()
        assert drop_inapplicable_fields(cdl, keep_non_default=True)["crop_sequence_years"] == 8

    def test_vhr_sar_fields_apply_only_to_their_sources(self):
        """vhr_sar_* fields enter the hash for umbra/capella only (calibration: capella only);
        the download cap and the cache directory never do, but keep plan identities when set."""
        umbra = AgriboundConfig(
            source="umbra", study_area=BBOX, year=2024, lulc_filter=False
        ).to_dict()
        kept = drop_inapplicable_fields(umbra)
        assert {k for k in NEW_FIELDS if k in kept} == {
            "vhr_sar_collect_ids",
            "vhr_sar_max_collects",
            "vhr_sar_polarization",
            "vhr_sar_resolution_m",
            "vhr_sar_coregister",
            "vhr_sar_coregister_base",
            "vhr_sar_bands",
        }
        capella = {**umbra, "source": "capella"}
        assert "vhr_sar_calibration" in drop_inapplicable_fields(capella)
        cap = {**umbra, "vhr_sar_max_download_gb": 20.0, "vhr_sar_cache_dir": "/x"}
        assert "vhr_sar_max_download_gb" not in drop_inapplicable_fields(cap)
        both = drop_inapplicable_fields(cap, keep_non_default=True)
        assert both["vhr_sar_max_download_gb"] == 20.0 and both["vhr_sar_cache_dir"] == "/x"
        assert config_hash(cap) == config_hash(umbra)

    def test_drop_inapplicable_fields(self):
        assert "drop_inapplicable_fields" in provenance.__all__
        sentinel2 = _gee("sentinel2").to_dict()
        for keep_non_default in (False, True):
            kept = drop_inapplicable_fields(sentinel2, keep_non_default=keep_non_default)
            assert set(sentinel2) - set(kept) == set(NEW_FIELDS)
        assert set(NEW_FIELDS) <= set(sentinel2)  # the input is not modified
        pan = drop_inapplicable_fields(_gee("landsat-pan").to_dict(), keep_non_default=True)
        assert pan["landsat_pan_missions"] == "auto" and "lulc_tree_crops" not in pan
        tree = _gee("sentinel2", lulc_tree_crops=True).to_dict()
        assert drop_inapplicable_fields(tree)["lulc_tree_crops"] is True
        assert drop_inapplicable_fields(tree, keep_non_default=True)["lulc_tree_crops"] is True
        # A dictionary written before the fields existed passes through unchanged (1.0.1
        # wrote its default usgs_retries, 3).
        old = {k: v for k, v in sentinel2.items() if k not in NEW_FIELDS} | {"usgs_retries": 3}
        assert drop_inapplicable_fields(old) == old
        assert drop_inapplicable_fields(old, keep_non_default=True) == old

    def test_keep_non_default_keeps_explicit_values_that_do_not_apply(self):
        le07 = _gee("sentinel2", landsat_pan_missions="LE07").to_dict()
        assert "landsat_pan_missions" not in drop_inapplicable_fields(le07)  # as config_hash
        kept = drop_inapplicable_fields(le07, keep_non_default=True)
        assert kept["landsat_pan_missions"] == ["LE07"] and "lulc_tree_crops" not in kept
        auto = _gee("sentinel2", landsat_pan_missions="AUTO").to_dict()
        assert "landsat_pan_missions" not in drop_inapplicable_fields(auto, keep_non_default=True)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: _gee("sentinel2"),
            lambda: _gee("landsat-pan", landsat_pan_missions="LC08"),
            lambda: _gee("naip", year=2022, lulc_tree_crops=True),
            lambda: _gee("sentinel2", landsat_pan_missions="LE07"),
        ],
    )
    def test_canonical_config_drops_the_inapplicable_fields(self, make):
        config = make()
        data = drop_inapplicable_fields(config.to_dict())
        expected = {k: to_jsonable(v) for k, v in data.items() if k not in HASH_EXCLUDED_FIELDS}
        assert canonical_config(config) == expected
        assert list(canonical_config(config)) == sorted(expected)


#: Configurations recorded by agribound 1.0.1 for the tree-crop example (provenance records of
#: outputs/examples_1.0/outputs/tree_crops/), with their config_hash, study-area fingerprint and
#: composite cache key (AGRIBOUND_CACHE_KEY). They must stay reproducible: fields added since
#: (crop_sequence_years among them) may not change them.
_TWIFO_S2_DA_2020 = {
    "aoi_selection": "representative_point", "bands": None, "cache_dir": None,
    "cloud_cover_max": 80, "cloud_score_threshold": 0.6, "composite_method": "median",
    "date_range": None, "device": "auto", "embedding_cache_dir": None,
    "engine": "delineate-anything", "engine_params": {"max_detections": 1000},
    "export_crs": "utm", "export_method": "local", "fine_tune": False,
    "fine_tune_block_size_m": 5000.0, "fine_tune_epochs": 20, "fine_tune_split": "block",
    "fine_tune_split_column": None, "fine_tune_val_split": 0.2, "gcs_bucket": None,
    "gee_high_volume": False, "gee_max_requests": 8, "gee_project": "irrigation-status-474718",
    "gee_service_account_key": None, "gee_workload_tag": None, "google_embedding_backend": "gee",
    "landsat_pan_missions": "auto", "local_tif_path": None, "lulc_batch_size": 200,
    "lulc_crop_threshold": 0.3, "lulc_dataset": "auto", "lulc_filter": False,
    "lulc_mode": "server", "lulc_nodata_policy": "keep", "lulc_on_error": "raise",
    "lulc_tree_crops": False, "min_field_area_m2": 2500.0, "n_workers": 4,
    "naip_resolution_m": 1.0, "output_format": "gpkg",
    "output_path": "outputs/tree_crops/twifo/fields_sentinel2_da_2020.gpkg", "overwrite": False,
    "provenance": True, "reference_boundaries": None, "s2_cloud_mask": "cloud_score_plus",
    "sam_backend": "sam2", "sam_crop_padding": 0.15, "sam_min_crop_px": 64, "sam_model": None,
    "sam_refine": False, "seed": 42, "simplify_tolerance": 2.0, "source": "sentinel2",
    "study_area": "outputs/tree_crops/twifo/study_area.geojson", "tessera_variant": None,
    "tessera_version": "v1", "tile_size": 10000, "usgs_allow_year_fallback": False,
    "usgs_retries": 3,
    "usgs_service_url": (
        "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer"
    ),
    "usgs_state": None, "usgs_timeout_s": 120, "year": 2020,
}  # fmt: skip
GOLDEN_1_0_1 = {
    "twifo sentinel2 da 2020": (
        _TWIFO_S2_DA_2020,
        "0a98f4a46a39",
        "f78e64763c12ede972d3111e8183a99e51e4d0f1",
        "e2c329c1855c",
    ),
    "oro sentinel2 da 2021": (
        {
            **_TWIFO_S2_DA_2020,
            "year": 2021,
            "output_path": "outputs/tree_crops/oro/fields_sentinel2_da_2021.gpkg",
            "study_area": "outputs/tree_crops/oro/study_area.geojson",
        },
        "022062e63d78",
        "d767a270d6580cc35c8bf48092bedb7f67de9345",
        "24daf529b64a",
    ),
    "twifo sentinel2 ftw 2020": (
        {
            **_TWIFO_S2_DA_2020,
            "engine": "ftw",
            "engine_params": {"window_dates": ["2020-01-15", "2020-08-15"]},
            "output_path": "outputs/tree_crops/twifo/fields_sentinel2_ftw_2020.gpkg",
        },
        "0a98f4a46a39",
        "3b689611b3eace06682ebe1f49705bcded9acd62",
        "e2c329c1855c",
    ),
    "twifo landsat-pan da 2020": (
        {
            **_TWIFO_S2_DA_2020,
            "source": "landsat-pan",
            "cloud_cover_max": 20,
            "s2_cloud_mask": "scl",
            "output_path": "outputs/tree_crops/twifo/fields_landsat-pan_da_2020.gpkg",
        },
        "0a98f4a46a39",
        "54fe63ae1a496a959e4e4840fb8c8320bf0f5985",
        "cb79d2e12c3e",
    ),
}


#: Defaults of every AgriboundConfig field of agribound 1.0.1 (tag v1.0.1), in JSON form.
DEFAULTS_1_0_1 = {
    "aoi_selection": "representative_point", "bands": None, "cache_dir": None,
    "cloud_cover_max": 20, "cloud_score_threshold": 0.6, "composite_method": "median",
    "date_range": None, "device": "auto", "embedding_cache_dir": None,
    "engine": "delineate-anything", "engine_params": {}, "export_crs": "utm",
    "export_method": "local", "fine_tune": False, "fine_tune_block_size_m": 5000.0,
    "fine_tune_epochs": 20, "fine_tune_split": "block", "fine_tune_split_column": None,
    "fine_tune_val_split": 0.2, "gcs_bucket": None, "gee_high_volume": False,
    "gee_max_requests": 8, "gee_project": None, "gee_service_account_key": None,
    "gee_workload_tag": None, "google_embedding_backend": "gee", "local_tif_path": None,
    "lulc_batch_size": 200, "lulc_crop_threshold": 0.3, "lulc_dataset": "auto",
    "lulc_filter": True, "lulc_mode": "server", "lulc_nodata_policy": "keep",
    "lulc_on_error": "raise", "min_field_area_m2": 2500.0, "n_workers": 4,
    "naip_resolution_m": 1.0, "output_format": "gpkg", "output_path": "fields.gpkg",
    "overwrite": False, "provenance": True, "reference_boundaries": None,
    "s2_cloud_mask": "scl", "sam_backend": "sam2", "sam_crop_padding": 0.15,
    "sam_min_crop_px": 64, "sam_model": None, "sam_refine": False, "seed": 42,
    "simplify_tolerance": 2.0, "source": "sentinel2", "study_area": "",
    "tessera_variant": None, "tessera_version": "v1", "tile_size": 10000,
    "usgs_allow_year_fallback": False, "usgs_retries": 3,
    "usgs_service_url": (
        "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer"
    ),
    "usgs_state": None, "usgs_timeout_s": 120, "year": 2024,
}  # fmt: skip


def test_defaults_changed_since_1_0_1_are_listed():
    """A field whose default differs from 1.0.1's is in DEFAULTS_OF_1_0_1, with 1.0.1's value.

    Otherwise the tile-manifest signatures and agent plan directories of every 1.0.1
    configuration would change (they sign the whole configuration dictionary).
    """
    import dataclasses

    from agribound.provenance import DEFAULTS_OF_1_0_1, HASH_EXCLUDED_FIELDS

    current = {
        f.name: to_jsonable(
            f.default if f.default is not dataclasses.MISSING else f.default_factory()
        )
        for f in dataclasses.fields(AgriboundConfig)
        if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
    }
    changed = {k: v for k, v in DEFAULTS_1_0_1.items() if current.get(k, v) != v}
    assert changed == DEFAULTS_OF_1_0_1
    assert set(DEFAULTS_OF_1_0_1) <= HASH_EXCLUDED_FIELDS  # they do not change the result


def test_identities_of_a_1_0_1_record_are_kept():
    """A configuration that 1.0.1 recorded, made again now, keeps 1.0.1's identity dictionary."""
    data = dict(_TWIFO_S2_DA_2020)  # written by 1.0.1: usgs_retries 3, its default then
    now = AgriboundConfig.from_dict({k: v for k, v in data.items() if k != "usgs_retries"})
    assert now.usgs_retries == 5  # today's default
    kept = drop_inapplicable_fields(now.to_dict(), keep_non_default=True)
    assert kept == drop_inapplicable_fields(data, keep_non_default=True)
    assert config_hash(now) == GOLDEN_1_0_1["twifo sentinel2 da 2020"][2]
    # An explicit value other than both defaults is still signed.
    other = drop_inapplicable_fields(now.merged(usgs_retries=7).to_dict(), keep_non_default=True)
    assert other["usgs_retries"] == 7
    # The configuration hash form is not affected (usgs_retries is not hashed).
    assert drop_inapplicable_fields(now.to_dict())["usgs_retries"] == 5


@pytest.mark.parametrize("name", sorted(GOLDEN_1_0_1))
def test_golden_hashes_and_cache_keys_of_1_0_1_records(name, monkeypatch):
    """The config hashes and composite cache keys recorded by agribound 1.0.1 are unchanged."""
    from agribound import _cache
    from agribound.composites.gee import _recipe_parts

    data, aoi, recorded_hash, recorded_key = GOLDEN_1_0_1[name]
    config = AgriboundConfig.from_dict(data)
    assert config_hash(config) == recorded_hash
    assert config_hash(config.merged(crop_sequence_years=3)) == recorded_hash
    # The study-area files are not in the repository: use the recorded fingerprint.
    monkeypatch.setattr(_cache, "aoi_fingerprint", lambda c: aoi)
    assert _cache.cache_key(config, *_recipe_parts(config.source)) == recorded_key
    key = _cache.cache_key(config.merged(crop_sequence_years=3), *_recipe_parts(config.source))
    assert key == recorded_key


class TestJsonable:
    def test_conversions(self, tmp_path):
        value = {
            "a": np.float32(1.5),
            "b": np.arange(3),
            "c": (1, 2),
            "d": {3, 1},
            "e": tmp_path,
            "f": float("nan"),
            "g": object(),
        }
        out = to_jsonable(value)
        json.dumps(out)
        assert out["a"] == 1.5 and out["b"] == [0, 1, 2] and out["c"] == [1, 2]
        assert out["d"] == [1, 3] and out["e"] == str(tmp_path) and out["f"] is None
        assert isinstance(out["g"], str)


class TestRunRecorder:
    def test_success_record(self, cfg):
        with RunRecorder(cfg, run_id="20260101T000000Z-abcdef") as rec:
            with rec.step("composite") as step:
                time.sleep(0.01)
                step["detail"] = "x"
            rec.set("n", np.int64(3))
            rec.add_warning("careful")
            rec.record_engine_meta({"backend": "stub", "conf": np.float64(0.2)})
        d = rec.to_dict()
        json.dumps(d)
        assert d["status"] == "success" and d["error"] is None
        assert d["run_id"] == "20260101T000000Z-abcdef"
        assert d["agribound_version"] == __version__
        assert d["config_hash"] == config_hash(cfg)
        assert d["seed"] == 42
        assert d["config"]["source"] == "local"
        assert d["facts"]["n"] == 3
        assert d["warnings"] == ["careful"]
        assert d["engine_meta"] == {"backend": "stub", "conf": 0.2}
        assert d["gee_workload_tag"] == "agribound-test"
        assert d["device"] == "cpu"
        assert d["versions"]["agribound"] == __version__
        step = d["steps"][0]
        assert step["name"] == "composite" and step["status"] == "success"
        assert step["wall_s"] >= 0.009 and step["detail"] == "x"
        assert d["wall_s"] >= step["wall_s"]
        assert d["started_utc"].endswith("Z") and d["finished_utc"].endswith("Z")
        assert d["elapsed_utc_s"] >= step["wall_s"]
        assert d["peak_rss_mb"] is None or d["peak_rss_mb"] > 0
        for key in ("platform", "python", "hostname", "git", "environment", "torch_max_memory_mb"):
            assert key in d

    def test_elapsed_utc_s_counts_the_time_the_computer_slept(self, cfg, monkeypatch):
        """wall_s comes from perf_counter, which stops while macOS sleeps: example 12's
        GeoAI run of 2026-10-09 recorded 9,175 s from started_utc to finished_utc and a
        wall_s 4,680 s shorter. elapsed_utc_s records the clock time beside it."""
        import datetime as dt
        import types

        from agribound import provenance

        start = dt.datetime(2026, 10, 9, 21, 0, 0, 250000, tzinfo=dt.UTC)
        clock = iter([start, start + dt.timedelta(seconds=9175.375)])

        class FakeDateTime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return next(clock)

        counter = iter([100.0, 4595.5])  # perf_counter at the start and the end
        monkeypatch.setattr(
            provenance,
            "_dt",
            types.SimpleNamespace(datetime=FakeDateTime, date=dt.date, UTC=dt.UTC),
        )
        monkeypatch.setattr(
            provenance, "time", types.SimpleNamespace(perf_counter=lambda: next(counter))
        )
        with RunRecorder(cfg) as rec:
            pass
        d = rec.to_dict()
        assert (d["started_utc"], d["finished_utc"]) == (
            "2026-10-09T21:00:00Z",
            "2026-10-09T23:32:55Z",
        )
        assert d["wall_s"] == 4495.5
        assert d["elapsed_utc_s"] == 9175.375  # not rounded to the whole seconds of the timestamps

    def test_failed_step_and_run(self, cfg):
        rec = RunRecorder(cfg)
        with pytest.raises(RuntimeError, match="boom"), rec, rec.step("delineate"):
            raise RuntimeError("boom")
        d = rec.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "RuntimeError: boom"
        assert d["steps"][0]["status"] == "failed"
        assert "boom" in d["steps"][0]["error"]

    def test_hash_captured_at_creation(self, cfg):
        rec = RunRecorder(cfg)
        cfg.engine_params["checkpoint_path"] = "/tmp/model.pt"
        assert rec.to_dict()["config_hash"] != config_hash(cfg)
        assert "checkpoint_path" not in rec.to_dict()["config"]["engine_params"]

    def test_run_id_generated(self, cfg):
        assert RunRecorder(cfg).run_id != RunRecorder(cfg).run_id


class TestProvenanceFiles:
    def test_path(self, tmp_path):
        assert provenance_path(tmp_path / "f.gpkg") == tmp_path / "f.gpkg.provenance.json"

    def test_round_trip(self, cfg):
        with RunRecorder(cfg) as rec:
            rec.set("metrics", {"f1": np.float64(0.5)})
        record = rec.to_dict()
        path = write_provenance(cfg.output_path, record)
        assert path == provenance_path(cfg.output_path)
        loaded = read_provenance(cfg.output_path)
        assert loaded == json.loads(json.dumps(record))
        assert rec.to_dict() == record  # frozen after the run finished
        assert loaded["facts"]["metrics"]["f1"] == 0.5
        assert loaded["versions"]["agribound"] == __version__

    def test_missing_and_corrupt(self, tmp_path):
        out = tmp_path / "x.gpkg"
        assert read_provenance(out) is None
        provenance_path(out).write_text("{not json")
        assert read_provenance(out) is None

    def test_write_is_atomic(self, cfg, tmp_path):
        write_provenance(cfg.output_path, {"a": 1})
        write_provenance(cfg.output_path, {"a": 2})
        assert read_provenance(cfg.output_path) == {"a": 2}
        assert not list(tmp_path.glob(".*.tmp"))
