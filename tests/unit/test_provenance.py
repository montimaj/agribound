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
NEW_FIELDS = ("landsat_pan_missions", "lulc_tree_crops")
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
        def record(config):
            return {"status": "success", "config_hash": _hash_without_new_fields(config)}

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
        }

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
        # A dictionary written before the fields existed passes through unchanged.
        old = {k: v for k, v in sentinel2.items() if k not in NEW_FIELDS}
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
        assert d["peak_rss_mb"] is None or d["peak_rss_mb"] > 0
        for key in ("platform", "python", "hostname", "git", "environment", "torch_max_memory_mb"):
            assert key in d

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
