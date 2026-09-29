"""Pipeline tests on local rasters with stubbed engines (no GPU, GEE or ML dependencies)."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import box

import agribound
from agribound.config import AgriboundConfig
from agribound.pipeline import build_composite, delineate
from agribound.provenance import config_hash, provenance_path, read_provenance

METADATA_COLUMNS = {
    "id",
    "metrics:area",
    "metrics:perimeter",
    "agribound:compactness",
    "determination:method",
    "determination:datetime",
    "agribound:engine",
    "agribound:source",
    "agribound:year",
    "agribound:version",
    "agribound:run_id",
}


class StubEngine:
    """Returns two 200 m squares inside the 640 m test raster (EPSG:32611)."""

    def __init__(self, n=2):
        self.calls = []
        self.n = n

    def delineate(self, raster_path, config):
        assert Path(raster_path).exists()
        self.calls.append((raster_path, copy.deepcopy(config.engine_params)))
        boxes = [box(500020, 4000020, 500220, 4000220), box(500300, 4000300, 500500, 4000500)]
        gdf = gpd.GeoDataFrame(
            {"score": [0.9, 0.8][: self.n]}, geometry=boxes[: self.n], crs="EPSG:32611"
        )
        gdf.attrs["engine_meta"] = {"backend": "stub", "conf": np.float32(0.25)}
        return gdf


@pytest.fixture
def engine(monkeypatch):
    stub = StubEngine()
    monkeypatch.setattr("agribound.engines.get_engine", lambda name: stub)
    return stub


@pytest.fixture
def run_kwargs(sample_rgb_tif, tmp_path):
    return dict(
        source="local",
        local_tif_path=sample_rgb_tif,
        engine="delineate-anything",
        output_path=str(tmp_path / "out" / "fields.gpkg"),
        device="cpu",
        lulc_filter=False,
        simplify_tolerance=0,
        engine_params={"smooth_iterations": 0},
    )


def _evaluate_module():
    # ``agribound.evaluate`` (attribute) is the re-exported function; patch the module.
    import importlib

    return importlib.import_module("agribound.evaluate")


def _aoi_bbox_for(raster_path, pad=0.0):
    with rasterio.open(raster_path) as src:
        bounds = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    minx, miny, maxx, maxy = bounds
    return f"bbox:{minx - pad},{miny - pad},{maxx + pad},{maxy + pad}"


class TestLocalPipeline:
    def test_writes_output_metadata_and_provenance(self, engine, run_kwargs):
        gdf = delineate(**run_kwargs)
        out = Path(run_kwargs["output_path"])
        assert out.exists()
        assert len(gdf) == 2
        assert set(gdf.columns) >= METADATA_COLUMNS

        run_id = gdf.attrs["run_id"]
        assert gdf["id"].is_unique
        assert all(i.startswith(f"{run_id}-") for i in gdf["id"])
        assert (gdf["agribound:run_id"] == run_id).all()
        assert (gdf["agribound:version"] == agribound.__version__).all()
        assert (gdf["agribound:year"] == 2024).all()
        assert (gdf["determination:method"] == "auto-imagery").all()
        assert gdf["determination:datetime"].iloc[0] == pd.Timestamp("2024-12-31T23:59:59Z")

        # 200 m UTM squares near the central meridian (scale factor 0.9996).
        np.testing.assert_allclose(gdf["metrics:area"], 40_032, rtol=5e-3)
        np.testing.assert_allclose(gdf["metrics:perimeter"], 800.3, rtol=5e-3)
        np.testing.assert_allclose(gdf["agribound:compactness"], math.pi / 4, rtol=5e-3)

        record = read_provenance(out)
        assert record["status"] == "success"
        assert record["run_id"] == run_id
        assert record["config_hash"] == config_hash(AgriboundConfig(**run_kwargs))
        assert record["engine_meta"] == {"backend": "stub", "conf": 0.25}
        assert record["facts"]["n_detected"] == 2
        assert record["facts"]["n_output"] == 2
        assert record["facts"]["lulc_status"] == "disabled"
        steps = [s["name"] for s in record["steps"]]
        assert steps == ["composite", "delineate", "postprocess", "metadata", "write"]
        assert all(s["status"] == "success" for s in record["steps"])
        assert gdf.attrs["engine_meta"]["backend"] == "stub"
        assert gdf.attrs["provenance_path"] == str(provenance_path(out))

        loaded = gpd.read_file(out)
        assert set(loaded.columns) >= METADATA_COLUMNS

    def test_seeds_before_running(self, engine, run_kwargs, monkeypatch):
        seen = []
        monkeypatch.setattr("agribound._repro.seed_everything", lambda seed, **k: seen.append(seed))
        delineate(**{**run_kwargs, "seed": 7})
        assert seen == [7]

    def test_reuses_output_with_matching_config(self, engine, run_kwargs):
        first = delineate(**run_kwargs)
        second = delineate(**run_kwargs)
        assert len(engine.calls) == 1
        assert second.attrs["reused"] is True
        assert second.attrs["run_id"] == first.attrs["run_id"]
        assert len(second) == len(first)

    def test_reuse_ignores_output_only_fields(self, engine, run_kwargs):
        delineate(**run_kwargs)
        delineate(**{**run_kwargs, "n_workers": 8, "device": "auto"})
        assert len(engine.calls) == 1

    def test_mismatched_config_raises(self, engine, run_kwargs):
        delineate(**run_kwargs)
        with pytest.raises(FileExistsError, match="different configuration"):
            delineate(**{**run_kwargs, "min_field_area_m2": 10.0})
        assert len(engine.calls) == 1

    def test_overwrite_reruns(self, engine, run_kwargs):
        first = delineate(**run_kwargs)
        second = delineate(**{**run_kwargs, "min_field_area_m2": 10.0, "overwrite": True})
        assert len(engine.calls) == 2
        assert second.attrs["run_id"] != first.attrs["run_id"]
        record = read_provenance(run_kwargs["output_path"])
        assert record["run_id"] == second.attrs["run_id"]
        # The overwritten output can now be reused with the new configuration.
        delineate(**{**run_kwargs, "min_field_area_m2": 10.0})
        assert len(engine.calls) == 2

    def test_existing_output_without_provenance_raises(self, engine, run_kwargs):
        out = Path(run_kwargs["output_path"])
        out.parent.mkdir(parents=True)
        gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_file(out)
        with pytest.raises(FileExistsError, match="no provenance record"):
            delineate(**run_kwargs)
        assert engine.calls == []

    def test_provenance_disabled(self, engine, run_kwargs):
        delineate(**{**run_kwargs, "provenance": False})
        assert Path(run_kwargs["output_path"]).exists()
        assert not provenance_path(run_kwargs["output_path"]).exists()

    def test_parquet_output_and_reuse(self, engine, run_kwargs, tmp_path):
        kwargs = {**run_kwargs, "output_path": str(tmp_path / "fields.parquet")}
        gdf = delineate(**kwargs)
        assert gdf.attrs["run_id"]
        again = delineate(**kwargs)
        assert len(engine.calls) == 1 and len(again) == 2
        assert again.crs.to_epsg() == 4326

    def test_zero_detections_still_written(self, run_kwargs, monkeypatch):
        stub = StubEngine(n=0)
        monkeypatch.setattr("agribound.engines.get_engine", lambda name: stub)
        gdf = delineate(**run_kwargs)
        assert len(gdf) == 0
        assert Path(run_kwargs["output_path"]).exists()
        record = read_provenance(run_kwargs["output_path"])
        assert record["status"] == "success"
        assert "No field boundaries detected" in record["warnings"]
        assert record["facts"]["n_output"] == 0

    def test_engine_returning_empty_frame_without_geometry(self, run_kwargs, monkeypatch):
        class Empty:
            def delineate(self, raster_path, config):
                return gpd.GeoDataFrame()

        monkeypatch.setattr("agribound.engines.get_engine", lambda name: Empty())
        gdf = delineate(**run_kwargs)
        assert len(gdf) == 0 and gdf.crs.to_epsg() == 32611

    def test_caller_config_not_mutated_and_overrides(self, engine, run_kwargs, tmp_path):
        cfg = AgriboundConfig(**run_kwargs)
        before = cfg.to_dict()
        other = str(tmp_path / "other.gpkg")
        gdf = delineate(config=cfg, output_path=other, year=2020)
        assert cfg.to_dict() == before
        assert Path(other).exists()
        assert (gdf["agribound:year"] == 2020).all()

    def test_config_with_equal_named_args_is_not_overridden(self, engine, run_kwargs):
        cfg = AgriboundConfig(**run_kwargs)
        delineate(study_area=cfg.study_area, config=cfg)
        assert Path(run_kwargs["output_path"]).exists()

    def test_aliases(self, engine, run_kwargs):
        kwargs = dict(run_kwargs)
        kwargs.pop("simplify_tolerance")
        gdf = delineate(**kwargs, min_area=50_000, simplify=0)
        assert len(gdf) == 0  # both 40 000 m² squares removed by the aliased area filter

    def test_default_output_path_uses_format_extension(
        self, engine, run_kwargs, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        kwargs = dict(run_kwargs)
        kwargs.pop("output_path")
        delineate(**kwargs, output_format="geojson")
        assert (tmp_path / "fields_local_2024.geojson").exists()


class TestLulcStage:
    def test_filter_applied_and_recorded(self, engine, run_kwargs, monkeypatch):
        def fake_filter(gdf, config):
            out = gdf.iloc[:1].copy()
            out["lulc:crop_fraction"] = [0.9]
            out.attrs["lulc_stats"] = {"dataset": "nlcd", "n_in": len(gdf), "n_kept": 1}
            return out

        monkeypatch.setattr("agribound.postprocess.lulc_filter.filter_by_lulc", fake_filter)
        gdf = delineate(**{**run_kwargs, "lulc_filter": True})
        assert len(gdf) == 1
        assert "lulc:crop_fraction" in gdf.columns
        record = read_provenance(run_kwargs["output_path"])
        assert record["facts"]["lulc_status"] == "applied"
        assert record["facts"]["lulc_stats"]["n_kept"] == 1
        assert gdf.attrs["lulc_stats"]["dataset"] == "nlcd"

    def test_failure_raises_by_default(self, engine, run_kwargs, monkeypatch):
        def boom(gdf, config):
            raise RuntimeError("no GEE credentials")

        monkeypatch.setattr("agribound.postprocess.lulc_filter.filter_by_lulc", boom)
        with pytest.raises(RuntimeError, match="lulc_on_error='warn'"):
            delineate(**{**run_kwargs, "lulc_filter": True})
        assert not Path(run_kwargs["output_path"]).exists()
        record = read_provenance(run_kwargs["output_path"])
        assert record["status"] == "failed"
        assert [s for s in record["steps"] if s["name"] == "lulc_filter"][0]["status"] == "failed"

    def test_failure_warns_when_configured(self, engine, run_kwargs, monkeypatch, caplog):
        def boom(gdf, config):
            raise RuntimeError("no GEE credentials")

        monkeypatch.setattr("agribound.postprocess.lulc_filter.filter_by_lulc", boom)
        with caplog.at_level("WARNING", logger="agribound.pipeline"):
            gdf = delineate(**{**run_kwargs, "lulc_filter": True, "lulc_on_error": "warn"})
        assert len(gdf) == 2
        assert "LULC filter failed" in caplog.text
        record = read_provenance(run_kwargs["output_path"])
        assert record["facts"]["lulc_status"] == "failed"
        assert any("no GEE credentials" in w for w in record["warnings"])


class TestSamRefineStage:
    @pytest.fixture
    def fake_refine(self, monkeypatch):
        calls = []

        def refine(gdf, raster_path, config, **kwargs):
            calls.append((len(gdf), raster_path, config.sam_backend))
            out = gdf.copy()
            out["agribound:sam_refined"] = True
            out.attrs["sam_stats"] = {"backend": config.sam_backend, "n_total": len(gdf)}
            return out

        monkeypatch.setattr("agribound.engines.samgeo_engine.refine_boundaries", refine)
        return calls

    def test_stage_runs_when_enabled(self, engine, run_kwargs, fake_refine):
        gdf = delineate(**{**run_kwargs, "sam_refine": True, "sam_backend": "sam2.1"})
        assert len(fake_refine) == 1
        n, raster_path, backend = fake_refine[0]
        assert n == 2 and backend == "sam2.1" and Path(raster_path).exists()
        assert gdf.attrs["sam_stats"] == {"backend": "sam2.1", "n_total": 2}
        record = read_provenance(run_kwargs["output_path"])
        assert record["facts"]["sam_stats"]["backend"] == "sam2.1"
        assert "sam_refine" in [s["name"] for s in record["steps"]]

    def test_legacy_engine_param_enables_stage(self, engine, run_kwargs, fake_refine):
        params = {"smooth_iterations": 0, "sam_refine": True}
        delineate(**{**run_kwargs, "engine_params": params})
        assert len(fake_refine) == 1

    def test_not_run_by_default(self, engine, run_kwargs, fake_refine):
        delineate(**run_kwargs)
        assert fake_refine == []

    def test_skipped_for_embedding_engine(
        self, run_kwargs, fake_refine, monkeypatch, sample_rgb_tif
    ):
        class Builder:
            def build(self, config):
                return sample_rgb_tif

        stub = StubEngine()
        monkeypatch.setattr("agribound.composites.get_composite_builder", lambda source: Builder())
        monkeypatch.setattr("agribound.engines.get_engine", lambda name: stub)
        delineate(
            study_area="bbox:-117.0,36.1,-116.99,36.2",
            source="google-embedding",
            engine="embedding",
            year=2023,
            output_path=run_kwargs["output_path"],
            lulc_filter=False,
            sam_refine=True,
        )
        assert len(stub.calls) == 1
        assert fake_refine == []


class TestFineTuneAndEvaluation:
    def test_fine_tune_checkpoint_passed_and_no_evaluation(
        self, engine, run_kwargs, monkeypatch, tmp_path
    ):
        ref = tmp_path / "ref.gpkg"
        gpd.GeoDataFrame(
            geometry=[box(500020, 4000020, 500220, 4000220)], crs="EPSG:32611"
        ).to_file(ref)
        monkeypatch.setattr(
            "agribound.engines.finetune.fine_tune", lambda raster, config: "/tmp/best.pt"
        )
        evaluated = []
        monkeypatch.setattr(_evaluate_module(), "evaluate", lambda *a, **k: evaluated.append(a))
        cfg = AgriboundConfig(**run_kwargs, fine_tune=True, reference_boundaries=str(ref))
        gdf = delineate(config=cfg)
        assert engine.calls[0][1]["checkpoint_path"] == "/tmp/best.pt"
        assert "checkpoint_path" not in cfg.engine_params
        assert evaluated == []
        record = read_provenance(run_kwargs["output_path"])
        assert record["facts"]["fine_tuned_checkpoint"] == "/tmp/best.pt"
        assert "checkpoint_path" not in record["config"]["engine_params"]
        assert "evaluation_metrics" not in gdf.attrs

    def test_evaluation_uses_references_in_study_area(
        self, engine, run_kwargs, monkeypatch, tmp_path, sample_rgb_tif
    ):
        ref = tmp_path / "ref.gpkg"
        refs = [
            box(500020, 4000020, 500220, 4000220),
            box(500300, 4000300, 500500, 4000500),
            box(600000, 4100000, 600200, 4100200),  # far outside the study area
        ]
        gpd.GeoDataFrame({"k": [1, 2, 3]}, geometry=refs, crs="EPSG:32611").to_file(ref)
        seen = {}

        def fake_evaluate(pred, reference, *args, **kwargs):
            seen["n_ref"] = len(reference)
            seen["n_pred"] = len(pred)
            return {"f1": np.float64(1.0), "recall": 1.0}

        monkeypatch.setattr(_evaluate_module(), "evaluate", fake_evaluate)
        gdf = delineate(
            **run_kwargs,
            study_area=_aoi_bbox_for(sample_rgb_tif),
            reference_boundaries=str(ref),
        )
        assert seen == {"n_ref": 2, "n_pred": 2}
        assert gdf.attrs["evaluation_metrics"]["f1"] == 1.0
        record = read_provenance(run_kwargs["output_path"])
        assert record["facts"]["evaluation"] == {"f1": 1.0, "recall": 1.0}
        assert record["facts"]["evaluation_reference"] == {
            "n_reference_total": 3,
            "selection": "representative point in study area",
            "n_reference_used": 2,
        }
        # Reused outputs keep the metrics from the provenance record.
        again = delineate(
            **run_kwargs,
            study_area=_aoi_bbox_for(sample_rgb_tif),
            reference_boundaries=str(ref),
        )
        assert again.attrs["evaluation_metrics"]["f1"] == 1.0


# Left half of the 640 m test raster (EPSG:32611) as an EWKT study area.
_HALF_AOI = "SRID=32611;POLYGON ((500000 4000000, 500320 4000000, 500320 4000640, 500000 4000640, 500000 4000000))"  # noqa: E501


class AoiStubEngine:
    """A inside, B crossing the outline (centre outside), C crossing (centre inside), D outside."""

    boxes = {
        "A": box(500020, 4000020, 500220, 4000220),
        "B": box(500250, 4000300, 500450, 4000500),
        "C": box(500280, 4000020, 500340, 4000100),
        "D": box(500400, 4000400, 500600, 4000600),
    }

    def delineate(self, raster_path, config):
        return gpd.GeoDataFrame(
            {"name": list(self.boxes)}, geometry=list(self.boxes.values()), crs="EPSG:32611"
        )


class TestAoiSelection:
    @pytest.fixture
    def aoi_kwargs(self, run_kwargs, monkeypatch):
        monkeypatch.setattr("agribound.engines.get_engine", lambda name: AoiStubEngine())
        return {**run_kwargs, "study_area": _HALF_AOI, "min_field_area_m2": 100.0}

    @pytest.mark.parametrize(
        ("rule", "names", "extra"),
        [
            ("representative_point", ["A", "C"], {}),
            ("intersects", ["A", "B", "C"], {}),
            ("clip", ["A", "B", "C"], {"n_clipped": 3}),  # B, C cut; D dropped
            ("none", ["A", "B", "C", "D"], {}),
        ],
    )
    def test_rules(self, aoi_kwargs, rule, names, extra):
        gdf = delineate(**aoi_kwargs, aoi_selection=rule)
        assert sorted(gdf["name"]) == names
        record = read_provenance(aoi_kwargs["output_path"])
        assert record["facts"]["aoi_selection"] == {
            "rule": rule,
            "n_before": 4,
            "n_after": len(names),
            **extra,
        }
        steps = [s["name"] for s in record["steps"]]
        assert ("aoi_selection" in steps) == (rule != "none")
        assert steps.index("aoi_selection" if rule != "none" else "delineate") < steps.index(
            "postprocess"
        )
        if rule == "clip":
            areas = gdf.to_crs("EPSG:32611").set_index("name").geometry.area
            assert areas["A"] == pytest.approx(200 * 200)
            assert areas["B"] == pytest.approx(70 * 200)  # cut at x = 500320
            assert areas["C"] == pytest.approx(40 * 80)
        else:
            # Selection never changes a geometry.
            kept = gdf.to_crs("EPSG:32611").set_index("name").geometry
            for name in names:
                assert kept[name].equals_exact(AoiStubEngine.boxes[name], 1e-6)

    def test_without_study_area_nothing_is_selected(self, engine, run_kwargs):
        delineate(**run_kwargs)
        record = read_provenance(run_kwargs["output_path"])
        assert record["facts"]["aoi_selection"] == {
            "rule": "representative_point",
            "n_before": 2,
            "n_after": 2,
            "skipped": "no study area",
        }

    def test_selection_changes_the_config_hash(self, aoi_kwargs):
        a = AgriboundConfig(**aoi_kwargs)
        assert config_hash(a) != config_hash(a.merged(aoi_selection="clip"))

    @pytest.mark.parametrize(
        ("rule", "label", "n_used"),
        [
            ("representative_point", "representative point in study area", 2),
            ("intersects", "intersects study area", 3),
            ("clip", "clipped to study area", 3),
            ("none", "intersects study area", 3),
        ],
    )
    def test_references_are_selected_with_the_same_rule(
        self, aoi_kwargs, monkeypatch, tmp_path, rule, label, n_used
    ):
        ref = tmp_path / "ref.gpkg"
        gpd.GeoDataFrame(
            {"name": list(AoiStubEngine.boxes)},
            geometry=list(AoiStubEngine.boxes.values()),
            crs="EPSG:32611",
        ).to_file(ref)
        seen = {}

        def fake_evaluate(pred, reference, *args, **kwargs):
            seen["ref_names"] = sorted(reference["name"])
            seen["ref_area"] = float(reference.geometry.area.sum())
            return {"f1": 1.0}

        monkeypatch.setattr(_evaluate_module(), "evaluate", fake_evaluate)
        delineate(**aoi_kwargs, aoi_selection=rule, reference_boundaries=str(ref))
        record = read_provenance(aoi_kwargs["output_path"])
        assert record["facts"]["evaluation_reference"] == {
            "n_reference_total": 4,
            "selection": label,
            "n_reference_used": n_used,
        }
        if rule == "clip":
            assert seen["ref_area"] == pytest.approx(200 * 200 + 70 * 200 + 40 * 80)


class TestSelectInStudyArea:
    def test_invalid_null_and_empty_geometries(self):
        from shapely.geometry import Polygon

        from agribound.pipeline import select_in_study_area

        aoi = box(0, 0, 10, 10)
        bowtie = Polygon([(1, 1), (4, 4), (4, 1), (1, 4), (1, 1)])  # self-intersecting
        gdf = gpd.GeoDataFrame(
            {"k": [1, 2, 3, 4]},
            geometry=[bowtie, None, Polygon(), box(8, 8, 14, 14)],
            crs="EPSG:32611",
        )
        gdf.attrs["engine_meta"] = {"backend": "x"}
        for rule, expected in (
            ("representative_point", [1]),
            ("intersects", [1, 4]),
            ("clip", [1, 4]),
            ("none", [1, 2, 3, 4]),
        ):
            out, stats = select_in_study_area(gdf, aoi, rule)
            assert out["k"].tolist() == expected, rule
            assert out.attrs["engine_meta"] == {"backend": "x"}
            assert stats["n_before"] == 4 and stats["n_after"] == len(expected)
        clipped, stats = select_in_study_area(gdf, aoi, "clip")
        assert stats["n_clipped"] == 1
        assert clipped.geometry.iloc[1].equals(box(8, 8, 10, 10))
        assert not clipped.geometry.iloc[0].is_valid  # contained: original geometry kept
        with pytest.raises(ValueError, match="Unknown aoi_selection"):
            select_in_study_area(gdf, aoi, "nope")

    def test_clip_drops_non_polygonal_remainders(self):
        from agribound.pipeline import select_in_study_area

        gdf = gpd.GeoDataFrame(geometry=[box(10, 0, 12, 5)], crs="EPSG:32611")
        out, stats = select_in_study_area(gdf, box(0, 0, 10, 10), "clip")  # touches along x=10
        assert len(out) == 0 and stats == {
            "rule": "clip",
            "n_before": 1,
            "n_after": 0,
            "n_clipped": 1,
        }

    def test_study_area_is_densified_before_reprojection(self):
        """A long lon/lat edge becomes a curve in UTM; the vertices-only reprojection misses it."""
        from agribound.pipeline import study_area_in_crs

        cfg = AgriboundConfig(
            source="sentinel2",
            gee_project="p",
            lulc_filter=False,
            study_area="bbox:-117,36,-116,37",
        )
        geom = study_area_in_crs(cfg, "EPSG:32611")
        corners = gpd.GeoSeries([box(-117, 36, -116, 37)], crs="EPSG:4326").to_crs("EPSG:32611")
        assert len(geom.exterior.coords) > 100
        # The densified outline differs from the 4-corner quadrilateral by tens of metres.
        assert geom.symmetric_difference(corners.iloc[0]).area > 1e5
        back = gpd.GeoSeries([geom], crs="EPSG:32611").to_crs("EPSG:4326").iloc[0]
        assert back.symmetric_difference(box(-117, 36, -116, 37)).area < 1e-6


class TestGeeAssetStudyAreaOffline:
    """A GEE-asset study area read in stage A is available to an offline delineation."""

    ASSET = "projects/p/assets/half_aoi"

    @pytest.fixture
    def offline_kwargs(self, monkeypatch, sample_rgb_tif, tmp_path):
        class CachedBuilder:  # a cached composite: the builder does not read the study area
            def build(self, config):
                return sample_rgb_tif

        monkeypatch.setattr(
            "agribound.composites.get_composite_builder", lambda source: CachedBuilder()
        )
        monkeypatch.setattr("agribound.engines.get_engine", lambda name: AoiStubEngine())
        return dict(
            source="sentinel2",
            gee_project="p",
            study_area=self.ASSET,
            engine="delineate-anything",
            output_path=str(tmp_path / "out" / "fields.gpkg"),
            cache_dir=str(tmp_path / "cache"),
            device="cpu",
            lulc_filter=False,
            simplify_tolerance=0,
            engine_params={"smooth_iterations": 0},
            min_field_area_m2=100.0,
            overwrite=True,
        )

    @staticmethod
    def _online(monkeypatch, events):
        import sys
        import types

        from shapely import wkt
        from shapely.geometry import mapping

        half = gpd.GeoSeries([wkt.loads(_HALF_AOI.split(";", 1)[1])], crs="EPSG:32611")
        feature = {"geometry": mapping(half.to_crs("EPSG:4326").iloc[0]), "properties": {}}

        class FakeFC:
            def __init__(self, asset_id):
                events.append(("read", asset_id))

            def getInfo(self):  # noqa: N802 - mirrors ee.FeatureCollection.getInfo
                return {"features": [feature]}

        monkeypatch.setitem(sys.modules, "ee", types.SimpleNamespace(FeatureCollection=FakeFC))
        monkeypatch.setattr("agribound.auth.ensure_gee", lambda config: events.append("ensure"))

    @staticmethod
    def _offline(monkeypatch):
        import sys

        def no_network(config):
            raise RuntimeError("offline node: cannot reach Earth Engine")

        monkeypatch.setitem(sys.modules, "ee", None)
        monkeypatch.setattr("agribound.auth.ensure_gee", no_network)

    def test_composite_online_then_delineate_offline(self, monkeypatch, offline_kwargs):
        from agribound.io.vector import study_area_cache_file

        events = []
        self._online(monkeypatch, events)
        cfg = AgriboundConfig(**offline_kwargs)
        build_composite(cfg)  # 'agribound composite' on a node with network access
        assert events == ["ensure", ("read", self.ASSET)]
        assert study_area_cache_file(cfg).exists()

        self._offline(monkeypatch)
        gdf = delineate(**offline_kwargs)  # 'agribound delineate' on an offline node
        assert sorted(gdf["name"]) == ["A", "C"]
        record = read_provenance(offline_kwargs["output_path"])
        assert record["facts"]["aoi_selection"] == {
            "rule": "representative_point",
            "n_before": 4,
            "n_after": 2,
        }

    def test_offline_without_copy_raises_actionable_error(self, monkeypatch, offline_kwargs):
        from agribound.io.vector import study_area_cache_file

        self._offline(monkeypatch)
        with pytest.raises(RuntimeError, match="aoi_selection='none'") as info:
            delineate(**offline_kwargs)
        message = str(info.value)
        assert str(study_area_cache_file(AgriboundConfig(**offline_kwargs))) in message
        assert "agribound composite" in message and "local vector file" in message
        # Without the selection, nothing else in this run needs the study area.
        gdf = delineate(**offline_kwargs, aoi_selection="none")
        assert sorted(gdf["name"]) == ["A", "B", "C", "D"]


class TestBuildComposite:
    def test_local_returns_raster(self, run_kwargs):
        cfg = AgriboundConfig(**run_kwargs)
        assert Path(build_composite(cfg)).exists()

    def test_requires_study_area_for_remote_sources(self):
        cfg = AgriboundConfig(source="sentinel2", gee_project="p", lulc_filter=False)
        with pytest.raises(ValueError, match="study_area is required"):
            build_composite(cfg)

    def test_calls_builder_for_source(self, monkeypatch, sample_rgb_tif):
        seen = []

        class Builder:
            def build(self, config):
                seen.append(config.source)
                return sample_rgb_tif

        monkeypatch.setattr("agribound.composites.get_composite_builder", lambda source: Builder())
        cfg = AgriboundConfig(
            source="sentinel2", gee_project="p", study_area="bbox:0,0,1,1", lulc_filter=False
        )
        assert build_composite(cfg) == sample_rgb_tif
        assert seen == ["sentinel2"]

    def test_lulc_raster_prefetch(self, run_kwargs, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "agribound.postprocess.lulc_filter.prefetch_lulc_raster",
            lambda config: calls.append(config.lulc_mode) or "/tmp/lulc.tif",
            raising=False,
        )
        build_composite(
            AgriboundConfig(**{**run_kwargs, "lulc_filter": True, "lulc_mode": "raster"})
        )
        assert calls == ["raster"]
        build_composite(AgriboundConfig(**{**run_kwargs, "lulc_filter": True}))  # server mode
        build_composite(AgriboundConfig(**{**run_kwargs, "lulc_mode": "raster"}))  # filter off
        assert calls == ["raster"]

    def test_lulc_prefetch_failure_policy(self, run_kwargs, monkeypatch):
        def boom(config):
            raise RuntimeError("offline")

        monkeypatch.setattr(
            "agribound.postprocess.lulc_filter.prefetch_lulc_raster", boom, raising=False
        )
        base = {**run_kwargs, "lulc_filter": True, "lulc_mode": "raster"}
        with pytest.raises(RuntimeError, match="raster prefetch failed"):
            build_composite(AgriboundConfig(**base))
        path = build_composite(AgriboundConfig(**{**base, "lulc_on_error": "warn"}))
        assert Path(path).exists()


@pytest.mark.slow
class TestImportSmoke:
    """Verify the package can be imported without errors."""

    def test_import_agribound(self):
        import agribound  # noqa: F401

    def test_import_io(self):
        from agribound.io.crs import get_utm_crs  # noqa: F401
        from agribound.io.raster import get_raster_info  # noqa: F401
        from agribound.io.vector import read_vector  # noqa: F401


class TestPublicApi:
    def test_exports(self):
        for name in (
            "__version__",
            "AgriboundConfig",
            "delineate",
            "build_composite",
            "evaluate",
            "list_engines",
            "list_sources",
            "list_ftw_models",
            "query_ftw",
            "show_boundaries",
        ):
            assert hasattr(agribound, name), name
        assert agribound.__version__ == "1.0.0"
        assert "agent" in dir(agribound)

    def test_unknown_attribute(self):
        with pytest.raises(AttributeError):
            agribound.no_such_thing  # noqa: B018

    def test_agent_missing_gives_install_hint(self, monkeypatch):
        import importlib

        real = importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == "agribound.agent.agent":
                exc = ModuleNotFoundError("No module named 'anthropic'")
                exc.name = "anthropic"
                raise exc
            return real(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fake_import)
        monkeypatch.delitem(agribound.__dict__, "agent", raising=False)
        with pytest.raises(ImportError, match=r'pip install "agribound\[agent\]"'):
            agribound.__getattr__("agent")


class TestConfigCreation:
    """AgriboundConfig can be created with various settings."""

    def test_embedding_config(self):
        cfg = AgriboundConfig(source="google-embedding", engine="embedding", year=2023)
        assert cfg.is_embedding_source() is True
        assert cfg.is_gee_source() is False

    def test_config_working_dir(self, tmp_path):
        cfg = AgriboundConfig(
            source="local",
            local_tif_path="/tmp/test.tif",
            output_path=str(tmp_path / "output" / "fields.gpkg"),
        )
        wd = cfg.get_working_dir()
        assert wd.exists()
        assert wd.name == ".agribound_cache"
