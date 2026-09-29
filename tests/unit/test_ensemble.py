"""Tests for the ensemble engine (agribound.engines.ensemble)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from agribound.config import AgriboundConfig
from agribound.engines import ensemble as ens
from agribound.engines.ensemble import EnsembleEngine, _min_votes
from agribound.registry import ENGINE_REGISTRY

UTM = "EPSG:32611"
X0, Y1 = 500000.0, 4001000.0


def _px(c0, r0, c1, r1):
    """Box from pixel coordinates of the 10 m test raster."""
    return box(X0 + c0 * 10, Y1 - r1 * 10, X0 + c1 * 10, Y1 - r0 * 10)


@pytest.fixture
def raster(tmp_path):
    path = tmp_path / "img.tif"
    with rasterio.open(
        path, "w", driver="GTiff", height=100, width=100, count=4, dtype="uint8",
        crs=UTM, transform=from_origin(X0, Y1, 10, 10),
    ) as dst:  # fmt: skip
        dst.write(np.ones((4, 100, 100), dtype=np.uint8))
    return str(path)


def _config(tmp_path, raster, **engine_params):
    return AgriboundConfig(
        source="local",
        engine="ensemble",
        local_tif_path=raster,
        output_path=str(tmp_path / "out.gpkg"),
        lulc_filter=False,
        engine_params=engine_params,
    )


class FakeEngine:
    """Returns the polygons registered for its member label; records how it was called."""

    def __init__(self, name, outputs, calls, fail=()):
        self.name = name
        self.outputs = outputs
        self.calls = calls
        self.fail = fail

    def delineate(self, raster_path, config):
        key = config.engine_params.get("variant", self.name)
        self.calls.append(
            {
                "engine": config.engine,
                "key": key,
                "engine_params": dict(config.engine_params),
                "cache_dir": config.cache_dir,
                "sam_refine": config.sam_refine,
                "seed": config.seed,
                "draw": float(np.random.random()),
            }
        )
        if key in self.fail:
            raise RuntimeError(f"{key} exploded")
        geoms = self.outputs[key]
        gdf = gpd.GeoDataFrame({"score": [0.5] * len(geoms)}, geometry=geoms, crs=UTM)
        gdf.attrs["engine_meta"] = {"backend": f"fake-{key}"}
        return gdf


@pytest.fixture
def fake(monkeypatch):
    state = {"outputs": {}, "calls": [], "fail": set()}

    def get_engine(name):
        return FakeEngine(name, state["outputs"], state["calls"], state["fail"])

    monkeypatch.setattr(ens, "get_engine", get_engine)
    return state


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_class_attributes_match_registry(self):
        info = ENGINE_REGISTRY["ensemble"]
        assert EnsembleEngine.supported_sources == info["supported_sources"]
        assert EnsembleEngine.requires_bands == info["requires_bands"]

    def test_default_members(self, tmp_path, raster):
        specs = EnsembleEngine.member_specs(_config(tmp_path, raster))
        assert [s["engine"] for s in specs] == ["delineate-anything", "ftw"]

    @pytest.mark.parametrize("source", ["naip", "spot", "spot-pan", "usgs-naip-plus"])
    def test_default_members_validated_against_source(self, tmp_path, source):
        """FTW (a default member) cannot run on these sources: raise, never drop it silently.

        The configuration rejects it (before any composite is built); the
        engine repeats the check for configurations that bypass validation.
        """
        kwargs = dict(
            engine="ensemble",
            year=2020,
            gee_project="test-project",
            output_path=str(tmp_path / "o.gpkg"),
            lulc_filter=False,
        )
        with pytest.raises(ValueError, match=r"'ftw' \(a default member\) does not support") as e:
            AgriboundConfig(source=source, **kwargs)
        assert "delineate-anything" in str(e.value).split("members that support")[1]
        cfg = AgriboundConfig(source="sentinel2", **kwargs)
        cfg.source = source  # bypasses __post_init__ validation
        with pytest.raises(ValueError, match=r"'ftw' \(a default member\) does not support"):
            EnsembleEngine.member_specs(cfg)

    def test_labels_are_unique(self, tmp_path, raster):
        cfg = _config(
            tmp_path,
            raster,
            engines=[
                {"engine": "ftw", "engine_params": {"model": "A"}},
                {"engine": "ftw", "engine_params": {"model": "A"}},
                "delineate-anything",
                {"engine": "geoai", "label": "mine"},
            ],
        )
        labels = [s["label"] for s in EnsembleEngine.member_specs(cfg)]
        assert labels == ["A", "A_1", "delineate-anything", "mine"]

    def test_suffixed_label_does_not_collide(self, tmp_path, raster):
        cfg = _config(
            tmp_path,
            raster,
            engines=[
                {"engine": "ftw", "label": "A"},
                {"engine": "ftw", "label": "A_2"},
                {"engine": "ftw", "label": "A"},  # A_2 is taken -> A_3
            ],
        )
        assert [s["label"] for s in EnsembleEngine.member_specs(cfg)] == ["A", "A_2", "A_3"]

    def test_cache_slugs_are_unique(self, tmp_path, raster):
        """Distinct labels with the same slug (or differing only in case) get distinct caches."""
        cfg = _config(
            tmp_path,
            raster,
            engines=[
                {"engine": "ftw", "label": "a/b"},
                {"engine": "ftw", "label": "a_b"},
                {"engine": "ftw", "label": "A_B"},
            ],
        )
        specs = EnsembleEngine.member_specs(cfg)
        assert [s["cache_slug"] for s in specs] == ["a_b", "a_b_1", "A_B_2"]
        dirs = [EnsembleEngine.member_config(cfg, s).cache_dir for s in specs]
        assert len({d.casefold() for d in dirs}) == 3

    def test_malformed_member_spec(self, tmp_path, raster):
        cfg = _config(tmp_path, raster)
        cfg.engine_params = {"engines": [{"engine": "ftw", "params": {}}]}
        with pytest.raises(ValueError, match="Unknown keys"):
            EnsembleEngine.member_specs(cfg)

    def test_unknown_ensemble_level_keys_raise(self, tmp_path, raster):
        cfg = _config(tmp_path, raster, model="FTW_PRUE_EFNET_B5")
        with pytest.raises(ValueError, match="not passed to its members"):
            EnsembleEngine.resolve_params(cfg)
        ok = _config(tmp_path, raster, smooth_iterations=0, regularize="none", sam_model="tiny")
        assert EnsembleEngine.resolve_params(ok)["merge_strategy"] == "intersection"

    def test_member_config(self, tmp_path, raster):
        cfg = _config(
            tmp_path,
            raster,
            engines=[{"engine": "ftw", "engine_params": {"model": "B7/x"}}],
            smooth_iterations=1,
        )
        cfg.sam_refine = True
        spec = EnsembleEngine.member_specs(cfg)[0]
        member = EnsembleEngine.member_config(cfg, spec)
        assert member.engine == "ftw"
        assert member.engine_params == {"model": "B7/x"}  # nothing inherited
        assert member.sam_refine is False
        assert Path(member.cache_dir) == tmp_path / ".agribound_cache" / "ensemble" / "B7_x"
        assert member.seed == cfg.seed
        shared = EnsembleEngine.member_config(cfg, spec, isolate_cache=False)
        assert shared.get_working_dir() == cfg.get_working_dir()

    def test_invalid_values(self, tmp_path, raster):
        for bad in (
            {"merge_strategy": "median"},
            {"vote_threshold": 1.5},
            {"on_member_error": "ignore"},
            {"vote_resolution": 0},
        ):
            with pytest.raises(ValueError):
                EnsembleEngine.resolve_params(_config(tmp_path, raster, **bad))


# ---------------------------------------------------------------------------
# Running members
# ---------------------------------------------------------------------------


class TestDelineate:
    def _members(self, fake):
        fake["outputs"].update(
            {
                "delineate-anything": [_px(10, 10, 30, 30), _px(50, 50, 70, 70)],
                "ftw": [_px(12, 10, 30, 30), _px(80, 80, 90, 90)],
            }
        )

    def test_members_isolated_seeded_and_recorded(self, tmp_path, raster, fake):
        self._members(fake)
        cfg = _config(tmp_path, raster, merge_strategy="union")
        cfg.sam_refine = True
        out = EnsembleEngine().delineate(raster, cfg)
        calls = fake["calls"]
        assert [c["engine"] for c in calls] == ["delineate-anything", "ftw"]
        assert all(c["sam_refine"] is False for c in calls)
        assert calls[0]["cache_dir"] != calls[1]["cache_dir"]
        assert calls[0]["cache_dir"].endswith("ensemble/delineate-anything")
        # Each member starts from the same seeded RNG state.
        assert calls[0]["draw"] == calls[1]["draw"]
        meta = out.attrs["engine_meta"]
        assert meta["merge_strategy"] == "union" and meta["n_members"] == 2
        assert [m["engine_meta"] for m in meta["members"]] == [
            {"backend": "fake-delineate-anything"},
            {"backend": "fake-ftw"},
        ]
        assert [m["n_polygons"] for m in meta["members"]] == [2, 2]
        assert out.crs == UTM

    def test_member_error_raises_by_default(self, tmp_path, raster, fake):
        self._members(fake)
        fake["fail"].add("ftw")
        with pytest.raises(RuntimeError, match="ftw exploded") as info:
            EnsembleEngine().delineate(raster, _config(tmp_path, raster))
        assert any("ensemble member 'ftw'" in n for n in info.value.__notes__)

    def test_member_error_skip_is_recorded(self, tmp_path, raster, fake):
        self._members(fake)
        fake["fail"].add("ftw")
        out = EnsembleEngine().delineate(
            raster, _config(tmp_path, raster, on_member_error="skip", merge_strategy="vote")
        )
        meta = out.attrs["engine_meta"]
        assert meta["failed_members"][0]["label"] == "ftw"
        assert meta["n_members"] == 1
        # Single survivor: its polygons are returned unchanged.
        assert len(out) == 2 and out["engine_count"].tolist() == [1, 1]
        assert out.geometry.iloc[0].equals(fake["outputs"]["delineate-anything"][0])

    def test_all_members_fail(self, tmp_path, raster, fake):
        self._members(fake)
        fake["fail"].update({"ftw", "delineate-anything"})
        with pytest.raises(RuntimeError, match="All ensemble members failed"):
            EnsembleEngine().delineate(raster, _config(tmp_path, raster, on_member_error="skip"))

    def test_same_engine_different_models(self, tmp_path, raster, fake):
        fake["outputs"].update({"a": [_px(0, 0, 10, 10)], "b": [_px(0, 0, 10, 10)]})
        cfg = _config(
            tmp_path,
            raster,
            engines=[
                {"engine": "ftw", "engine_params": {"variant": "a", "model": "m1"}},
                {"engine": "ftw", "engine_params": {"variant": "b", "model": "m2"}},
            ],
        )
        EnsembleEngine().delineate(raster, cfg)
        calls = fake["calls"]
        assert [c["engine_params"]["model"] for c in calls] == ["m1", "m2"]
        assert calls[0]["cache_dir"].endswith("ensemble/m1")
        assert calls[1]["cache_dir"].endswith("ensemble/m2")

    def test_prefetch_collects_member_files(self, tmp_path, raster, monkeypatch):
        class A:
            @classmethod
            def prefetch(cls, config):
                return ["/w/a.pt", "/w/shared.pt"]

        class B:
            @classmethod
            def prefetch(cls, config):
                assert config.engine == "ftw" and config.engine_params == {"model": "x"}
                return ["/w/shared.pt", "/w/b.ckpt"]

        monkeypatch.setattr(
            ens, "get_engine_class", lambda n: A if n == "delineate-anything" else B
        )
        cfg = _config(
            tmp_path,
            raster,
            engines=["delineate-anything", {"engine": "ftw", "engine_params": {"model": "x"}}],
        )
        assert EnsembleEngine.prefetch(cfg) == ["/w/a.pt", "/w/shared.pt", "/w/b.ckpt"]


# ---------------------------------------------------------------------------
# Merge strategies
# ---------------------------------------------------------------------------


def _frames(**named):
    return {k: gpd.GeoDataFrame(geometry=v, crs=UTM) for k, v in named.items()}


class TestUnion:
    def test_fuses_duplicates_keeps_touching_fields(self):
        results = _frames(
            a=[_px(0, 0, 10, 10), _px(10, 0, 20, 10)],  # two touching fields
            b=[_px(0, 0, 10, 9)],  # duplicate of a's first field
        )
        out = EnsembleEngine._merge_union(results)
        assert len(out) == 2
        assert out["ensemble:members"].tolist() == ["a,b", "a"]
        assert out["ensemble:n_members"].tolist() == [2, 1]
        assert out.geometry.iloc[0].equals(_px(0, 0, 10, 10))
        assert out["engine_count"].tolist() == [2, 2]

    def test_empty(self):
        out = EnsembleEngine._merge_union(_frames(a=[], b=[]))
        assert len(out) == 0 and out.crs == UTM


class TestIntersection:
    def test_areas_covered_by_all_members(self):
        results = _frames(
            a=[_px(0, 0, 10, 10), _px(20, 0, 30, 10)],
            b=[_px(5, 0, 15, 10)],
            c=[_px(0, 0, 8, 10)],
        )
        out = EnsembleEngine._merge_intersection(results)
        assert len(out) == 1
        assert out.geometry.iloc[0].equals(_px(5, 0, 8, 10))
        assert out["ensemble:members"].tolist() == ["a,b,c"]

    def test_member_without_polygons_empties_result(self):
        out = EnsembleEngine._merge_intersection(_frames(a=[_px(0, 0, 10, 10)], b=[]))
        assert len(out) == 0 and out.crs == UTM


class TestVote:
    @pytest.mark.parametrize(
        ("n", "threshold", "expected"),
        [
            (1, 0.5, 1),
            (1, 0.0, 1),
            (2, 0.3, 2),  # floor of two votes
            (2, 0.5, 2),
            (3, 0.3, 2),
            (3, 0.5, 2),
            (4, 0.5, 2),  # at least half, not a strict majority
            (5, 0.5, 3),
            (9, 0.3, 3),
            (10, 0.3, 3),
            (25, 0.28, 7),  # 0.28 * 25 = 7.000000000000001 must not round up (0.1.x: 8)
            (4, 0.8, 4),
            (3, 1.0, 3),
            (3, 0.0, 2),
        ],
    )
    def test_min_votes_rule_matches_0_1_x(self, n, threshold, expected):
        assert _min_votes(n, threshold) == expected
        # agribound 0.1.x: max(2 if n >= 2 else 1, ceil(threshold * n)), which rounded
        # float products just above an integer up; 1.0 agrees with it everywhere else.
        old = max(2 if n >= 2 else 1, int(np.ceil(threshold * n)))
        assert expected == old or (n, threshold) == (25, 0.28)

    def test_majority_on_raster_grid(self):
        grid = (rasterio.crs.CRS.from_string(UTM), from_origin(X0, Y1, 10, 10), 100, 100)
        results = _frames(
            a=[_px(0, 0, 20, 10)],
            b=[_px(10, 0, 30, 10)],
            c=[_px(15, 0, 40, 10)],
        )
        out = EnsembleEngine._merge_vote(results, 0.5, grid=grid)
        # Covered by >= 2 of 3: columns 10-30.
        assert len(out) == 1
        assert out.geometry.iloc[0].equals(_px(10, 0, 30, 10))
        assert out["vote_count"].tolist() == [3]  # all three overlap in columns 15-20
        assert out["min_votes"].tolist() == [2] and out["engine_count"].tolist() == [3]
        assert 2 < out["vote_count_mean"].iloc[0] < 3
        assert out.attrs["vote_stats"]["grid"]["width"] == 100

    def test_members_without_polygons_are_left_out(self, caplog):
        grid = (rasterio.crs.CRS.from_string(UTM), from_origin(X0, Y1, 10, 10), 100, 100)
        four = _frames(a=[_px(0, 0, 10, 10)], b=[_px(0, 0, 10, 10)], c=[], d=[None])
        with caplog.at_level("WARNING", logger="agribound.engines.ensemble"):
            out = EnsembleEngine._merge_vote(four, 0.5, grid=grid)
        # n = 2 (a, b), so min_votes = 2 and the shared box is kept (as in 0.1.x).
        assert len(out) == 1 and out.geometry.iloc[0].equals(_px(0, 0, 10, 10))
        assert out["engine_count"].tolist() == [2] and out["min_votes"].tolist() == [2]
        stats = out.attrs["vote_stats"]
        assert stats["empty_members"] == ["c", "d"]
        assert (stats["n_members"], stats["n_members_total"]) == (2, 4)
        assert "left out of the vote" in caplog.text

    def test_single_voting_member_needs_one_vote(self):
        results = _frames(a=[_px(0, 0, 10, 10)], b=[])
        out = EnsembleEngine._merge_vote(results, 0.5)
        assert len(out) == 1 and out["min_votes"].tolist() == [1]

    def test_no_member_with_polygons(self):
        grid = (rasterio.crs.CRS.from_string(UTM), from_origin(X0, Y1, 10, 10), 100, 100)
        out = EnsembleEngine._merge_vote(_frames(a=[], b=[]), 0.5, grid=grid)
        assert len(out) == 0 and out.crs == UTM
        assert out.attrs["vote_stats"]["empty_members"] == ["a", "b"]
        assert len(EnsembleEngine._merge_vote(_frames(a=[], b=[]), 0.5)) == 0

    def test_explicit_min_votes(self):
        results = _frames(a=[_px(0, 0, 10, 10)], b=[_px(5, 0, 15, 10)])
        grid = (rasterio.crs.CRS.from_string(UTM), from_origin(X0, Y1, 10, 10), 100, 100)
        out = EnsembleEngine._merge_vote(results, min_votes=1, grid=grid)
        assert out.geometry.iloc[0].equals(_px(0, 0, 15, 10))
        assert out.attrs["vote_stats"]["rule"] == "explicit min_votes"
        with pytest.raises(ValueError, match="min_votes"):
            EnsembleEngine._merge_vote(results, min_votes=3, grid=grid)
        # More than the members with polygons, but not more than all members: empty result.
        partly_empty = _frames(a=[_px(0, 0, 10, 10)], b=[_px(0, 0, 10, 10)], c=[])
        assert len(EnsembleEngine._merge_vote(partly_empty, min_votes=3, grid=grid)) == 0

    def test_legacy_static_call_without_grid(self):
        """Examples 09 and 12 call _merge_vote(results, threshold=...) on saved outputs."""
        results = _frames(a=[_px(0, 0, 10, 10)], b=[_px(0, 0, 10, 10)], c=[_px(50, 50, 60, 60)])
        out = EnsembleEngine._merge_vote(results, threshold=0.5)
        assert len(out) == 1  # at least 2 of 3; only the shared box survives
        assert out.geometry.iloc[0].equals(_px(0, 0, 10, 10))  # 10 m grid over the extent
        # threshold 0.3 (examples 09/12): ceil(0.9) = 1, but at least two members must agree.
        low = EnsembleEngine._merge_vote(results, threshold=0.3)
        assert len(low) == 1 and low["min_votes"].tolist() == [2]

    def test_delineate_uses_input_raster_grid(self, tmp_path, raster, fake):
        fake["outputs"].update(
            {"delineate-anything": [_px(0, 0, 10, 10)], "ftw": [_px(0, 0, 10, 10)]}
        )
        out = EnsembleEngine().delineate(raster, _config(tmp_path, raster, merge_strategy="vote"))
        vote = out.attrs["engine_meta"]["vote"]
        assert vote["grid"]["width"] == 100 and vote["grid"]["transform"][0] == 10.0
        assert vote["min_votes"] == 2
        assert out.geometry.iloc[0].equals(_px(0, 0, 10, 10))


class TestStageInputs:
    """EnsembleEngine.stage_inputs stages FTW members' windows in their own caches."""

    def _cfg(self, tmp_path, **ep):
        return AgriboundConfig(
            source="sentinel2",
            engine="ensemble",
            year=2024,
            gee_project="test-project",
            study_area="bbox:149.70,-30.40,149.72,-30.38",
            output_path=str(tmp_path / "o.gpkg"),
            lulc_filter=False,
            engine_params=ep,
        )

    def test_members_with_stage_inputs_are_staged(self, tmp_path, monkeypatch):
        from agribound.engines import ftw

        calls = []

        def fake_stage(config, raster_path):
            calls.append((config.engine, config.get_working_dir(), dict(config.engine_params)))
            return {"n_windows": 2, "rasters": [f"{config.get_working_dir()}/a.tif"]}

        monkeypatch.setattr(ftw.FTWEngine, "stage_inputs", staticmethod(fake_stage))
        cfg = self._cfg(
            tmp_path,
            engines=["delineate-anything", {"engine": "ftw", "engine_params": {"window_days": 9}}],
        )
        staged = EnsembleEngine.stage_inputs(cfg, "/x/composite.tif")
        assert [c[0] for c in calls] == ["ftw"]  # DA has no stage_inputs
        assert calls[0][2] == {"window_days": 9}
        assert str(calls[0][1]).endswith("ensemble/ftw")  # the member's isolated cache
        assert staged["rasters"] == [f"{calls[0][1]}/a.tif"]
        assert list(staged["members"]) == ["ftw"] and staged["failed_members"] == []

    def test_member_failure_policy(self, tmp_path, monkeypatch):
        from agribound.engines import ftw

        def failing(config, raster_path):
            raise RuntimeError("FTW window A: no imagery")

        monkeypatch.setattr(ftw.FTWEngine, "stage_inputs", staticmethod(failing))
        cfg = self._cfg(tmp_path)
        with pytest.raises(RuntimeError, match="no imagery") as info:
            EnsembleEngine.stage_inputs(cfg, "/x/composite.tif")
        assert any("ensemble member 'ftw'" in n for n in info.value.__notes__)
        skipped = EnsembleEngine.stage_inputs(
            cfg.merged(engine_params={"on_member_error": "skip"}), "/x/composite.tif"
        )
        assert skipped["rasters"] == [] and skipped["failed_members"][0]["label"] == "ftw"
