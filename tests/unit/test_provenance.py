"""Tests for agribound.provenance (run records and config hashing)."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from agribound._version import __version__
from agribound.config import AgriboundConfig
from agribound.provenance import (
    HASH_EXCLUDED_FIELDS,
    RunRecorder,
    config_hash,
    provenance_path,
    read_provenance,
    to_jsonable,
    write_provenance,
)


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
        ],
    )
    def test_result_fields_change_hash(self, cfg, override):
        assert config_hash(cfg.merged(**override)) != config_hash(cfg)

    def test_excluded_fields_are_config_fields(self):
        assert set(AgriboundConfig.field_names()) >= HASH_EXCLUDED_FIELDS


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
