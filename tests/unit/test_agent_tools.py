"""Tests for the provider-neutral agent tools (no network, no GEE)."""

from __future__ import annotations

import json
import math

import geopandas as gpd
import pytest
from shapely.geometry import box

pytest.importorskip("pydantic")

from agribound.agent import tools as tools_mod  # noqa: E402
from agribound.agent.errors import ExecutionDisabledError  # noqa: E402
from agribound.agent.gate import ConfirmationGate  # noqa: E402
from agribound.agent.plans import config_defaults  # noqa: E402
from agribound.agent.tools import (  # noqa: E402
    TOOL_SPECS,
    ResolvabilityInput,
    ToolContext,
    ToolRegistry,
    run_approved_plan,
)
from agribound.config import AgriboundConfig  # noqa: E402
from agribound.registry import ENGINE_REGISTRY, SOURCE_REGISTRY  # noqa: E402

# A 0.02 x 0.02 degree AOI near Visalia, California (UTM 11N) and one near Namoi (UTM 56S).
CA_BBOX = (-119.30, 36.30, -119.28, 36.32)
AU_BBOX = (151.00, -31.245, 151.05, -31.195)


def _aoi_file(tmp_path, bbox=CA_BBOX, name="aoi.geojson"):
    path = tmp_path / name
    gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326").to_file(path, driver="GeoJSON")
    return str(path)


def _squares(tmp_path, sides_m, crs="EPSG:32611", origin=(300000.0, 4020000.0), name="ref.gpkg"):
    x0, y0 = origin
    geoms, x = [], x0
    for side in sides_m:
        geoms.append(box(x, y0, x + side, y0 + side))
        x += side + 50.0
    path = tmp_path / name
    gpd.GeoDataFrame({"fid_": range(len(geoms))}, geometry=geoms, crs=crs).to_file(path)
    return str(path)


@pytest.fixture(autouse=True)
def _no_earth_engine(monkeypatch):
    """Fail loudly if a test would initialise Earth Engine (tests stay offline)."""

    def refuse(ctx):
        raise AssertionError("a test tried to initialise Earth Engine")

    monkeypatch.setattr(tools_mod, "_init_gee", refuse)


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(
        workdir=tmp_path / "work",
        study_area=_aoi_file(tmp_path),
        gee_project="test-project",
        allow_network=False,
    )


def _enable_execution(ctx, gate):
    """Execution tests: the fake pipeline replaces every remote call (local AOI file)."""
    ctx.execution_enabled = True
    ctx.allow_network = True
    ctx.gate = gate


@pytest.fixture
def registry(ctx):
    return ToolRegistry(ctx, include_execute=True)


def _call(registry, name, args=None):
    outcome = registry.call(name, args or {})
    assert outcome.ok, outcome.error
    return outcome.output


# ---------------------------------------------------------------------------
# Registry mechanics
# ---------------------------------------------------------------------------


def test_registry_offers_execute_only_when_enabled(ctx):
    assert "execute_plan" not in ToolRegistry(ctx).specs  # execution_enabled=False
    ctx.execution_enabled = True
    assert list(ToolRegistry(ctx).specs)[-1] == "execute_plan"
    assert [s.name for s in TOOL_SPECS][:2] == ["list_sources", "list_engines"]


def test_tool_schemas_are_json_objects_without_extra_properties(registry):
    for definition in registry.definitions():
        schema = definition.input_schema
        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is False, definition.name
        json.dumps(schema)
        assert definition.description


def test_unknown_tool_non_dict_and_extra_fields_are_errors(registry):
    assert "Unknown tool" in registry.call("nope", {}).error
    assert "JSON object" in registry.call("list_sources", "[]").error
    bad = registry.call("list_sources", {"unexpected": 1})
    assert not bad.ok and "unexpected" in bad.error and not bad.arguments_valid


def test_validation_error_message_omits_input_values(registry):
    out = registry.call("check_availability", {"year": 1800})
    assert not out.ok
    assert out.error.startswith("Invalid arguments for check_availability: year:")
    assert "1800" not in out.error


def test_unexpected_exceptions_become_error_outcomes(registry, monkeypatch):
    spec = registry.specs["list_sources"]

    def boom(ctx, inp):
        raise KeyError("x")

    registry.specs["list_sources"] = tools_mod.ToolSpec(
        spec.name, spec.description, spec.input_model, spec.output_model, boom
    )
    out = registry.call("list_sources", {})
    assert not out.ok and out.error.startswith("Unexpected error in list_sources: KeyError")


# ---------------------------------------------------------------------------
# Read-only tools
# ---------------------------------------------------------------------------


def test_list_sources_and_engines_mirror_the_registry(registry):
    sources = {s["source"]: s for s in _call(registry, "list_sources")["sources"]}
    assert set(sources) == set(SOURCE_REGISTRY)
    assert sources["sentinel2"]["year_first"] == 2017
    assert sources["sentinel2"]["year_last"] is None
    assert sources["naip"]["export_resolution_m"] == config_defaults()["naip_resolution_m"]
    assert sources["spot"]["restricted"] is True

    engines = {e["engine"]: e for e in _call(registry, "list_engines")["engines"]}
    assert set(engines) == set(ENGINE_REGISTRY)
    assert engines["geoai"]["label_free"] is False
    assert engines["ftw"]["fine_tunable"] is False
    assert set(engines["delineate-anything"]["python_packages_found"]) == {"ultralytics"}
    # Source-specific caveats are listed too (e.g. 30 m Landsat for Delineate-Anything).
    for key, info in ENGINE_REGISTRY.items():
        assert engines[key]["source_notes"] == dict(info.get("source_notes") or {})
    assert "landsat" in engines["delineate-anything"]["source_notes"]
    assert "landsat" in engines["ftw"]["source_notes"]


def test_describe_study_area(registry):
    out = _call(registry, "describe_study_area")
    assert out["utm_epsg"] == 32611 and out["utm_zone"] == 11
    assert out["utm_zones_spanned"] == [11]
    assert out["bbox_4326"] == pytest.approx(list(CA_BBOX))
    # 0.02 deg x 0.02 deg at 36.31 N: about 1.79 km x 2.22 km.
    assert out["area_km2"] == pytest.approx(3.97, rel=0.02)
    est = {e["source"]: e for e in out["composite_estimates"]}
    s2 = est["sentinel2"]
    assert s2["n_bands"] == 12 and s2["dtype"] == "float32"
    assert s2["uncompressed_mb"] == pytest.approx(
        s2["width_px"] * s2["height_px"] * 12 * 4 / 1e6, abs=0.1
    )
    assert est["naip"]["dtype"] == "uint8"
    assert est["naip"]["width_px"] == pytest.approx(s2["width_px"] * 10, abs=10)
    assert est["usgs-naip-plus"]["width_px"] is None
    assert "local" not in est


def test_describe_study_area_requires_an_area(tmp_path):
    reg = ToolRegistry(ToolContext(workdir=tmp_path / "w", allow_network=False))
    out = reg.call("describe_study_area", {})
    assert not out.ok and "No study area" in out.error
    missing = reg.call("describe_study_area", {"study_area": str(tmp_path / "none.geojson")})
    assert not missing.ok and "Could not read study area" in missing.error


def test_check_availability_registry_ranges(registry):
    out = _call(registry, "check_availability", {"year": 2016, "sources": ["sentinel2", "hls"]})
    res = {r["source"]: r for r in out["results"]}
    assert res["sentinel2"]["in_registry_range"] is False
    assert res["hls"]["in_registry_range"] is True
    assert res["sentinel2"]["live"] is None
    tess = _call(
        registry,
        "check_availability",
        {"year": 2015, "sources": ["tessera-embedding"], "tessera_version": "v1.1"},
    )
    assert tess["results"][0]["in_registry_range"] is True
    naip = _call(registry, "check_availability", {"year": 2024, "sources": ["naip"]})
    assert naip["results"][0]["in_registry_range"] is False  # GEE NAIP ends in 2023


def test_live_check_not_run_without_network(registry):
    out = _call(
        registry, "check_availability", {"year": 2023, "sources": ["sentinel2"], "live": True}
    )
    live = out["results"][0]["live"]
    assert live["status"] == "not_run" and "network" in live["message"]


def test_resolvability_from_a_representative_field_size(registry):
    out = _call(
        registry,
        "estimate_resolvability",
        {"median_field_area_ha": 1.0, "sources": ["sentinel2", "naip", "landsat"]},
    )
    per = {r["source"]: r for r in out["per_source"]}
    # p = A / GSD^2 with A = 10 000 m^2.
    assert per["sentinel2"]["pixels_per_field"]["median"] == pytest.approx(100.0, rel=1e-6)
    assert per["naip"]["pixels_per_field"]["median"] == pytest.approx(10_000.0, rel=1e-6)
    assert per["landsat"]["pixels_per_field"]["median"] == pytest.approx(10_000 / 900, rel=1e-6)
    # A 100 m square: padded box 130 px at 1 m (refined), 13 px at 10 m (skipped).
    assert per["naip"]["sam_refinement"]["eligible_fraction"] == 1.0
    assert per["sentinel2"]["sam_refinement"]["eligible_fraction"] == 0.0
    side = per["sentinel2"]["sam_refinement"]["min_refinable_square_side_m"]
    assert side == pytest.approx(64 * 10 / 1.3, abs=0.2)
    assert out["field_size_source"] == "user_median" and out["n_fields"] == 1


def test_resolvability_from_reference_layer_counts_and_areas(tmp_path, ctx):
    # Squares of 100, 400 and 800 m in the AOI's UTM zone; SAM needs >= 492.3 m at 10 m.
    ref = _squares(tmp_path, [100.0, 400.0, 800.0])
    ctx.study_area = None
    reg = ToolRegistry(ctx)
    out = _call(reg, "estimate_resolvability", {"reference_path": ref, "sources": ["sentinel2"]})
    sam = out["per_source"][0]["sam_refinement"]
    assert out["n_fields"] == 3
    assert sam["eligible_count"] == 1
    assert sam["eligible_fraction"] == pytest.approx(1 / 3)
    areas = [100.0**2, 400.0**2, 800.0**2]
    assert sam["eligible_area_fraction"] == pytest.approx(areas[2] / sum(areas), rel=1e-3)
    assert out["field_area_ha"]["median"] == pytest.approx(16.0, rel=1e-3)


def test_resolvability_restricts_reference_to_study_area(tmp_path, ctx):
    ref = _squares(tmp_path, [300.0, 300.0], crs="EPSG:32611", origin=(300000.0, 4020000.0))
    far = _aoi_file(tmp_path, bbox=(-100.0, 40.0, -99.99, 40.01), name="far.geojson")
    out = ToolRegistry(ctx).call(
        "estimate_resolvability", {"reference_path": ref, "study_area": far}
    )
    assert not out.ok and "No reference polygons" in out.error


def test_resolvability_needs_exactly_one_field_size_source(registry, tmp_path):
    ref = _squares(tmp_path, [100.0])
    out = registry.call(
        "estimate_resolvability", {"reference_path": ref, "median_field_area_ha": 2.0}
    )
    assert not out.ok and "exactly one" in out.error
    none = registry.call("estimate_resolvability", {})
    assert not none.ok and "median_field_area_ha" in none.error


def test_resolvability_gsd_override_and_skipped_sources(registry):
    out = _call(
        registry,
        "estimate_resolvability",
        {
            "median_field_area_ha": 1.0,
            "sources": ["usgs-naip-plus", "local"],
            "gsd_m": {"usgs-naip-plus": 0.6},
        },
    )
    assert [r["source"] for r in out["per_source"]] == ["usgs-naip-plus"]
    assert out["per_source"][0]["gsd_basis"] == "user override"
    assert "local" in out["skipped_sources"]


def test_resolvability_defaults_match_config_defaults():
    defaults = config_defaults()
    fields = ResolvabilityInput.model_fields
    assert fields["min_crop_px"].default == defaults["sam_min_crop_px"]
    assert fields["crop_padding"].default == defaults["sam_crop_padding"]


def test_recommendations_follow_documented_rules(registry, tmp_path):
    out = _call(registry, "recommend_configurations", {"year": 2023, "max_candidates": 50})
    pairs = {(c["source"], c["engine"]) for c in out["candidates"]}
    # No reference boundaries: engines that need a fine-tuned checkpoint are excluded.
    assert ("sentinel2", "geoai") not in pairs and ("sentinel2", "dinov3") not in pairs
    excl = {(e["source"], e["engine"]): e["reasons"] for e in out["excluded"]}
    assert any("no reference" in r for r in excl[("sentinel2", "geoai")])
    assert ("spot", None) in excl and ("local", None) in excl
    assert ("naip", "delineate-anything") in pairs  # California is inside the CONUS box
    # SAR sources rank after every other source; finer GSD first among the others.
    from agribound.registry import SAR_SOURCES

    sar = [c["source"] in SAR_SOURCES for c in out["candidates"]]
    assert any(sar) and sar == sorted(sar)
    gsds = [c["metrics"]["gsd_m"] for c in out["candidates"] if c["source"] not in SAR_SOURCES]
    known = [g for g in gsds if g is not None]
    assert known == sorted(known)
    assert gsds[len(known) :] == [None] * (len(gsds) - len(known))  # unknown GSD ranks last
    assert "conus_bbox_4326" in out["parameters"]
    # Candidates never change thresholds.
    for c in out["candidates"]:
        cfg = c["proposal"].get("config", {})
        assert set(cfg) <= {"engine_params", "sam_refine"}
    prithvi = next(c for c in out["candidates"] if c["engine"] == "prithvi")
    assert prithvi["proposal"]["config"]["engine_params"] == {"mode": "embed"}


def test_recommendations_outside_conus_and_with_reference(tmp_path, ctx):
    ctx.study_area = _aoi_file(tmp_path, bbox=AU_BBOX, name="au.geojson")
    import pyproj

    lon, lat = box(*AU_BBOX).centroid.coords[0]
    # Scalar transform: a one-point GeoSeries.to_crs triggers pyproj's NumPy
    # "ndim > 0 to a scalar" DeprecationWarning.
    x, y = pyproj.Transformer.from_crs(4326, 32756, always_xy=True).transform(lon, lat)
    ref = _squares(tmp_path, [200.0, 300.0], crs="EPSG:32756", origin=(x, y))
    reg = ToolRegistry(ctx)
    out = _call(
        reg,
        "recommend_configurations",
        {"year": 2023, "reference_path": ref, "prefer_label_free": False, "max_candidates": 50},
    )
    excl = {(e["source"], e["engine"]): e["reasons"] for e in out["excluded"]}
    assert any("conus_bbox_4326" in r for r in excl[("naip", None)])
    first = out["candidates"][0]
    assert first["proposal"]["fine_tune"] is True  # prefer_label_free=False ranks fine-tuning first
    assert first["proposal"]["reference_boundaries"] == ref
    engines_ft = {c["engine"] for c in out["candidates"] if c["proposal"]["fine_tune"]}
    assert {"geoai", "dinov3"} <= engines_ft and "ftw" not in engines_ft
    assert out["parameters"]["median_field_area_m2"]["value"] == pytest.approx(
        (200.0**2 + 300.0**2) / 2, rel=1e-3
    )


def test_recommendations_optional_pixels_per_field_threshold(registry):
    out = _call(
        registry,
        "recommend_configurations",
        {"year": 2023, "median_field_area_ha": 1.0, "min_median_pixels_per_field": 50},
    )
    excl = {(e["source"], e["engine"]): e["reasons"] for e in out["excluded"]}
    assert any("min_median_pixels_per_field" in r for r in excl[("landsat", None)])
    assert all(c["source"] not in ("landsat", "hls") for c in out["candidates"])
    assert out["parameters"]["min_median_pixels_per_field"]["value"] == 50


def test_recommendation_rules_are_named_parameters(registry):
    defaults = config_defaults()
    out = _call(
        registry,
        "recommend_configurations",
        {"year": 2018, "want_sam_refine": True, "max_candidates": 50},
    )
    params = out["parameters"]
    for name in (
        "conus_bbox_4326",
        "sentinel2_partial_coverage_years",
        "tessera_v1_near_global_years",
        "sam_min_crop_px",
        "sam_crop_padding",
        "min_median_pixels_per_field",
    ):
        assert "value" in params[name] and params[name].get("rationale"), name
    assert params["sam_min_crop_px"]["value"] == defaults["sam_min_crop_px"]
    assert params["sam_crop_padding"]["value"] == defaults["sam_crop_padding"]
    s2 = next(c for c in out["candidates"] if c["source"] == "sentinel2")
    assert any("sentinel2_partial_coverage_years" in w for w in s2["warnings"])
    tessera = next(c for c in out["candidates"] if c["source"] == "tessera-embedding")
    assert any("near-global only for [2024]" in w for w in tessera["warnings"])
    assert any("v1.1 is regional" in w for w in tessera["warnings"])
    # The SAM warning quotes the same defaults that the parameters report.
    assert any(f"sam_min_crop_px={defaults['sam_min_crop_px']}" in w for w in s2["warnings"])
    # A year inside the named sets carries no such warning.
    out24 = _call(registry, "recommend_configurations", {"year": 2024, "max_candidates": 50})
    for c in out24["candidates"]:
        assert not any("partial_coverage" in w or "near-global" in w for w in c["warnings"])


def test_live_tessera_check_passes_version_and_variant(tmp_path, monkeypatch):
    geotessera = pytest.importorskip("geotessera")
    seen = {}

    class FakeGeoTessera:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def embeddings_count(self, bbox, year=2024):
            seen["count"] = (bbox, year)
            return 7

    monkeypatch.setattr(geotessera, "GeoTessera", FakeGeoTessera)
    ctx = ToolContext(
        workdir=tmp_path / "w",
        study_area=_aoi_file(tmp_path),
        allow_network=True,
        embedding_cache_dir=str(tmp_path / "tcache"),
    )
    out = _call(
        ToolRegistry(ctx),
        "check_availability",
        {
            "year": 2021,
            "sources": ["tessera-embedding"],
            "live": True,
            "tessera_version": "v1.1",
            "tessera_variant": "cambridge",
        },
    )
    live = out["results"][0]["live"]
    assert live["status"] == "ok" and live["tile_count"] == 7
    assert seen["init"] == {
        "dataset_version": "v1.1",
        "dataset_variant": "cambridge",
        "cache_dir": str(tmp_path / "tcache"),
    }
    assert seen["count"][0] == pytest.approx(CA_BBOX) and seen["count"][1] == 2021
    assert "dataset_variant='cambridge'" in live["method"]


def test_year_outside_range_excludes_source(registry):
    out = _call(registry, "recommend_configurations", {"year": 2015, "max_candidates": 50})
    excl = {(e["source"], e["engine"]): e["reasons"] for e in out["excluded"]}
    assert any("outside the registry range" in r for r in excl[("sentinel2", None)])
    # TESSERA: the default v1 range excludes 2015, and the reason names the version that
    # covers it.
    tessera = excl[("tessera-embedding", None)]
    assert any(
        "outside the TESSERA v1 range 2017-2025" in r and "v1.1 (2015-2025)" in r for r in tessera
    )


def test_recommendations_with_a_tessera_version(registry):
    out = _call(
        registry,
        "recommend_configurations",
        {"year": 2015, "tessera_version": "v1.1", "sources": ["tessera-embedding"]},
    )
    assert out["candidates"], out["excluded"]
    cand = out["candidates"][0]
    assert cand["proposal"]["config"]["tessera_version"] == "v1.1"
    assert any("tessera_version v1.1" in r for r in cand["rules_applied"])
    assert any("TESSERA v1.1 coverage is regional" in w for w in cand["warnings"])
    assert out["parameters"]["tessera_version"]["value"] == "v1.1"
    # The candidate is a valid proposal.
    assert registry.call("propose_run", cand["proposal"]).ok


def test_usgs_naip_plus_conus_rule_is_labelled_a_simplification(tmp_path, ctx):
    ctx.study_area = _aoi_file(tmp_path, bbox=AU_BBOX, name="au.geojson")
    out = _call(ToolRegistry(ctx), "recommend_configurations", {"year": 2020})
    excl = {(e["source"], e["engine"]): e["reasons"] for e in out["excluded"]}
    assert any("conterminous part of USGS NAIP Plus" in r for r in excl[("usgs-naip-plus", None)])
    assert any("NAIP in Earth Engine" in r for r in excl[("naip", None)])
    rationale = out["parameters"]["conus_bbox_4326"]["rationale"]
    assert "simplification" in rationale and "Alaska" in rationale


def test_engine_notes_are_source_specific():
    """A Sentinel-2 plan must not show the Landsat/HLS caveat of Delineate-Anything."""
    from agribound.registry import engine_notes

    s2 = " ".join(engine_notes("delineate-anything", "sentinel2"))
    landsat = " ".join(engine_notes("delineate-anything", "landsat"))
    assert "Landsat" not in s2 and "HLS" not in s2
    assert "30 m Landsat" in landsat


def test_open_world_flags_mark_every_tool_that_can_reach_a_remote_service():
    offline_only = {"list_sources", "list_engines"}
    for spec in TOOL_SPECS:
        assert spec.open_world is (spec.name not in offline_only), spec.name
        assert "study_area" not in spec.input_model.model_fields or spec.open_world


def test_network_tools_refuse_when_offline(registry):
    out = registry.call("query_published_ftw", {})
    assert not out.ok and "network access" in out.error


def _ftw_polygons(years):
    """Published-FTW-like polygons (EPSG:4326) inside CA_BBOX, one 200 m square per year."""
    geoms, x = [], 300000.0
    for _ in years:
        geoms.append(box(x, 4020000.0, x + 200.0, 4020200.0))
        x += 300.0
    gdf = gpd.GeoDataFrame(
        {"determination:datetime": [f"{y}-01-01T00:00:00Z" for y in years]},
        geometry=geoms,
        crs="EPSG:32611",
    )
    return gdf.to_crs("EPSG:4326")


@pytest.fixture
def fake_query_ftw(monkeypatch):
    calls = []

    def fake(study_area, **kwargs):
        calls.append({"study_area": study_area, **kwargs})
        gdf = _ftw_polygons(fake.years)
        if kwargs.get("output_path"):
            gdf.to_file(kwargs["output_path"])
        return gdf

    fake.years = [2024, 2024, 2025]
    monkeypatch.setattr("agribound.ftw_query.query_ftw", fake)
    return fake, calls


def test_query_published_ftw_forwards_arguments_and_counts_years(ctx, fake_query_ftw):
    fake, calls = fake_query_ftw
    ctx.allow_network = True
    out = _call(
        ToolRegistry(ctx),
        "query_published_ftw",
        {"min_confidence": 69, "keep_null_confidence": False, "max_features": 50},
    )
    kwargs = calls[0]
    assert kwargs["year"] is None and kwargs["clip"] is True
    assert kwargs["min_confidence"] == 69 and kwargs["keep_null_confidence"] is False
    assert kwargs["max_features"] == 50
    assert kwargs["cache_dir"] == str(ctx.workdir / "ftw_cache")
    assert kwargs["output_path"].startswith(str(ctx.workdir / "ftw"))
    assert list(kwargs["study_area"].total_bounds) == pytest.approx(list(CA_BBOX))
    assert out["output_path"] == kwargs["output_path"]
    assert out["n_polygons"] == 3
    assert out["polygons_per_year"] == {"2024": 2, "2025": 1}
    assert any("field-year predictions" in n for n in out["notes"])
    # One year: no year-mix note; defaults are not forwarded.
    fake.years = [2025]
    single = _call(ToolRegistry(ctx), "query_published_ftw", {"year": 2025})
    assert calls[1]["year"] == 2025 and "min_confidence" not in calls[1]
    assert "keep_null_confidence" not in calls[1]  # query_ftw's default (True) applies
    assert single["polygons_per_year"] == {"2025": 1}
    assert not any("field-year" in n for n in single["notes"])


def test_resolvability_from_published_ftw_uses_one_year(ctx, fake_query_ftw):
    fake, calls = fake_query_ftw
    ctx.allow_network = True
    reg = ToolRegistry(ctx)
    out = _call(
        reg, "estimate_resolvability", {"use_published_ftw": True, "sources": ["sentinel2"]}
    )
    assert calls[0]["clip"] is False and calls[0]["year"] is None
    assert out["field_size_source"] == "published_ftw"
    assert out["published_ftw_years"] == {"2024": 2, "2025": 1}
    assert out["published_ftw_year_used"] == 2025 and out["n_fields"] == 1
    assert any("only the latest year (2025)" in n for n in out["notes"])
    assert out["field_area_ha"]["median"] == pytest.approx(4.0, rel=1e-2)
    # An explicit year is forwarded and used as is.
    fake.years = [2024, 2024]
    explicit = _call(
        reg,
        "estimate_resolvability",
        {"use_published_ftw": True, "year": 2024, "sources": ["sentinel2"]},
    )
    assert calls[1]["year"] == 2024
    assert explicit["published_ftw_year_used"] == 2024 and explicit["n_fields"] == 2


def _fake_live_ee(monkeypatch, ctx, counts: dict[str, int]) -> list:
    """Fake ``ee`` for live checks: ``size()`` of a collection is ``counts[collection]``."""
    import sys
    from types import SimpleNamespace

    seen = []

    class FakeCollection:
        def __init__(self, cid):
            self.cid = cid

        def filterBounds(self, region):  # noqa: N802 - Earth Engine API name
            seen.append(("bounds", self.cid, region))
            return self

        def filterDate(self, start, end):  # noqa: N802 - Earth Engine API name
            seen.append(("date", self.cid, start, end))
            return self

        def size(self):
            return SimpleNamespace(getInfo=lambda: counts[self.cid])

    fake_ee = SimpleNamespace(
        ImageCollection=FakeCollection,
        Geometry=SimpleNamespace(Rectangle=lambda coords, proj, geodesic: (coords, proj, geodesic)),
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    monkeypatch.setattr(tools_mod, "_init_gee", lambda ctx: None)
    ctx.allow_network = True
    return seen


def test_live_gee_check_counts_each_collection(ctx, monkeypatch):
    counts = {"COPERNICUS/S2_SR_HARMONIZED": 37}
    seen = _fake_live_ee(monkeypatch, ctx, counts)
    out = _call(
        ToolRegistry(ctx),
        "check_availability",
        {"year": 2023, "sources": ["sentinel2"], "live": True},
    )
    live = out["results"][0]["live"]
    assert live["status"] == "ok"
    assert live["per_collection"] == counts and live["image_count"] == 37
    assert live["message"] is None
    dates = {s[1]: s[2:] for s in seen if s[0] == "date"}
    assert dates == {cid: ("2023-01-01", "2024-01-01") for cid in counts}
    region = next(s[2] for s in seen if s[0] == "bounds")
    assert region[0] == pytest.approx(list(CA_BBOX)) and region[1:] == ("EPSG:4326", False)


HLSL30 = "NASA/HLS/HLSL30/v002"
HLSS30 = "NASA/HLS/HLSS30/v002"
#: A 5 km square inside MGRS tile T01KAA (Fiji), 0.3 degrees west of 180 degrees.
FIJI_BBOX = (179.6761, -17.6230, 179.7239, -17.5769)


def _fake_hls_live_ee(
    monkeypatch, ctx, counts: dict[str, int], antimeridian: dict[str, int] | None = None
) -> list:
    """Fake ``ee`` for the hls live check.

    Each collection keeps its steps; ``size()`` records them and gives
    ``counts[collection]`` for a collection bounded by the study-area box, and
    ``antimeridian[collection]`` (default 0) for one bounded by a point (the images of
    the MGRS tiles at 180 degrees). Sizes add up as ``ee.Number`` does.
    """
    antimeridian = antimeridian or {}
    import sys
    from types import SimpleNamespace

    sized = []

    class Number:
        def __init__(self, n):
            self.n = n

        def add(self, other):
            return Number(self.n + other.n)

        def getInfo(self):  # noqa: N802 - Earth Engine API name
            return self.n

    class Collection:
        def __init__(self, cid, steps=()):
            self.cid, self.steps = cid, tuple(steps)

        def filterBounds(self, region):  # noqa: N802 - Earth Engine API name
            return Collection(self.cid, (*self.steps, ("bounds", region)))

        def filter(self, value):
            return Collection(self.cid, (*self.steps, ("filter", value)))

        def filterDate(self, start, end):  # noqa: N802 - Earth Engine API name
            return Collection(self.cid, (*self.steps, ("date", start, end)))

        def map(self, fn):
            return Collection(self.cid, (*self.steps, ("map",)))

        def size(self):
            sized.append((self.cid, self.steps))
            bounds = self.steps[0][1]
            return Number(
                antimeridian.get(self.cid, 0) if bounds[0] == "point" else counts[self.cid]
            )

    class OrFilter(tuple):
        def Not(self):  # noqa: N802 - Earth Engine API name
            return ("not", tuple(self))

    fake_ee = SimpleNamespace(
        ImageCollection=Collection,
        Geometry=SimpleNamespace(
            Rectangle=lambda coords, proj, geodesic: (coords, proj, geodesic),
            Point=lambda coords: ("point", tuple(coords)),
        ),
        Filter=SimpleNamespace(
            bounds=lambda geometry: ("bounds", geometry),
            Or=lambda *filters: OrFilter(("or", *filters)),
            lte=lambda prop, value: ("lte", prop, value),
            eq=lambda prop, value: ("eq", prop, value),
        ),
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    monkeypatch.setattr(tools_mod, "_init_gee", lambda ctx: None)
    ctx.allow_network = True
    return sized


def _far_filter(bbox) -> tuple:
    from agribound.composites.gee import hls_far_points

    p1, p2 = hls_far_points(box(*bbox))
    return ("not", ("or", ("bounds", ("point", p1)), ("bounds", ("point", p2))))


def test_live_check_of_hls_drops_the_far_point_images(ctx, monkeypatch):
    counts = {HLSL30: 11, HLSS30: 29}
    sized = _fake_hls_live_ee(monkeypatch, ctx, counts, antimeridian={HLSL30: 7, HLSS30: 99})
    out = _call(
        ToolRegistry(ctx),
        "check_availability",
        {"year": 2023, "sources": ["hls"], "live": True},
    )
    live = out["results"][0]["live"]
    assert live["status"] == "ok"
    assert live["per_collection"] == counts and live["image_count"] == 40
    assert live["message"] is None
    # One count per collection: the study-area box, the far-point exclusion, the calendar
    # year, no cloud filter. Away from 180 degrees nothing else is counted.
    assert [cid for cid, _ in sized] == [HLSL30, HLSS30]
    for _, steps in sized:
        (bounds, far, date) = steps
        assert bounds[0] == "bounds" and bounds[1][0] == pytest.approx(list(CA_BBOX))
        assert bounds[1][1:] == ("EPSG:4326", False)
        assert far == ("filter", _far_filter(CA_BBOX))
        assert date == ("date", "2023-01-01", "2024-01-01")
    method = live["method"]
    assert "before cloud filtering, without the images whose footprint meets the antipode" in method
    assert "within 5 degrees of 180 degrees" not in method


def test_live_check_of_hls_near_180_counts_the_images_of_the_tiles_there(ctx, monkeypatch):
    from agribound.composites.gee import hls_far_points

    counts = {HLSL30: 50, HLSS30: 120}
    sized = _fake_hls_live_ee(monkeypatch, ctx, counts, antimeridian={HLSL30: 2, HLSS30: 29})
    out = _call(
        ToolRegistry(ctx),
        "check_availability",
        {
            "year": 2022,
            "sources": ["hls"],
            "live": True,
            "study_area": "bbox:" + ",".join(str(v) for v in FIJI_BBOX),
        },
    )
    live = out["results"][0]["live"]
    assert live["status"] == "ok"
    assert live["per_collection"] == {HLSL30: 52, HLSS30: 149}
    assert live["image_count"] == 201
    # Per collection: the images over the box without the far-point ones, and the images
    # of the tiles at 180 degrees whose tile meets the box (bounded by the first far point,
    # in the calendar year, flagged per image), added to that collection's count.
    p1, _ = hls_far_points(box(*FIJI_BBOX))
    assert [cid for cid, _ in sized] == [HLSL30, HLSL30, HLSS30, HLSS30]
    for i in (1, 3):
        assert sized[i][1] == (
            ("bounds", ("point", p1)),
            ("date", "2022-01-01", "2023-01-01"),
            ("map",),
            ("filter", ("eq", "agribound_real_footprint", 1)),
        )
    for i in (0, 2):
        assert sized[i][1][1] == ("filter", _far_filter(FIJI_BBOX))
    assert "the bbox lies within 5 degrees of 180 degrees" in live["method"]
    assert "each count includes the collection's images of those tiles" in live["method"]


def _pan_collection(mission: str) -> str:
    return f"LANDSAT/{mission}/C02/T1_TOA"


@pytest.mark.parametrize(
    ("year", "counts", "used"),
    [
        (2012, {"LE07": 23, "LC08": 0, "LC09": 0}, ["LE07"]),
        (2013, {"LE07": 23, "LC08": 18, "LC09": 0}, ["LC08"]),  # 'auto' never mixes
        (2022, {"LE07": 27, "LC08": 23, "LC09": 22}, ["LC08", "LC09"]),
        (2025, {"LE07": 0, "LC08": 20, "LC09": 21}, ["LC08", "LC09"]),
    ],
)
def test_live_gee_check_of_landsat_pan_counts_the_missions_auto_uses(
    ctx, monkeypatch, year, counts, used
):
    seen = _fake_live_ee(monkeypatch, ctx, {_pan_collection(m): n for m, n in counts.items()})
    out = _call(
        ToolRegistry(ctx),
        "check_availability",
        {"year": year, "sources": ["landsat-pan"], "live": True},
    )
    live = out["results"][0]["live"]
    assert live["status"] == "ok"
    # Every collection is still queried, but only the default missions are counted.
    assert {s[1] for s in seen if s[0] == "date"} == {_pan_collection(m) for m in counts}
    assert live["per_collection"] == {_pan_collection(m): counts[m] for m in used}
    assert live["image_count"] == sum(counts[m] for m in used)
    assert "landsat_pan_missions='auto'" in live["method"]
    message = live["message"]
    assert message.startswith(
        f"image_count counts {', '.join(used)}: the missions that the default"
    )
    unused = [m for m in counts if m not in used and counts[m] > 0]
    for m in unused:
        assert f"{_pan_collection(m)} ({counts[m]} images)" in message
    assert ("Not used by default" in message) is bool(unused)


def _fake_s1_ee(monkeypatch, ctx, per_pass: dict[str, dict[str, float]]) -> list:
    """Fake ``ee`` for the sentinel1 live check: one histogram of platform_number per pass.

    Each collection keeps the filters (and ``distinct``) applied to it;
    ``Dictionary(query).getInfo()`` records them and returns ``per_pass[orbit]`` for
    the pass each query filters on.
    """
    import sys
    from types import SimpleNamespace

    requests = []

    class Collection:
        def __init__(self, cid, steps=()):
            self.cid, self.steps = cid, tuple(steps)

        def filterBounds(self, region):  # noqa: N802 - Earth Engine API name
            return Collection(self.cid, (*self.steps, ("bounds", region)))

        def filter(self, value):
            return Collection(self.cid, (*self.steps, value))

        def filterDate(self, start, end):  # noqa: N802 - Earth Engine API name
            return Collection(self.cid, (*self.steps, ("date", start, end)))

        def distinct(self, properties):
            return Collection(self.cid, (*self.steps, ("distinct", tuple(properties))))

        def aggregate_histogram(self, prop):
            return (prop, self)

    class Dictionary:
        def __init__(self, query):
            self.query = query

        def getInfo(self):  # noqa: N802 - Earth Engine API name
            requests.append({k: (prop, c.cid, c.steps) for k, (prop, c) in self.query.items()})
            out = {}
            for key, (_, collection) in self.query.items():
                pass_filter = ("eq", "orbitProperties_pass")
                (orbit,) = [s[2] for s in collection.steps if s[:2] == pass_filter]
                out[key] = per_pass.get(orbit)
            return out

    fake_ee = SimpleNamespace(
        ImageCollection=Collection,
        Dictionary=Dictionary,
        Filter=SimpleNamespace(
            eq=lambda prop, value: ("eq", prop, value),
            listContains=lambda prop, value: ("listContains", prop, value),
        ),
        Geometry=SimpleNamespace(Rectangle=lambda coords, proj, geodesic: (coords, proj, geodesic)),
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    monkeypatch.setattr(tools_mod, "_init_gee", lambda ctx: None)
    ctx.allow_network = True
    return requests


def _s1_live(ctx, monkeypatch, per_pass):
    """Run check_availability(live=True) for sentinel1 in 2025 with a fake ``ee``."""
    requests = _fake_s1_ee(monkeypatch, ctx, per_pass)
    out = _call(
        ToolRegistry(ctx),
        "check_availability",
        {"year": 2025, "sources": ["sentinel1"], "live": True},
    )
    return out["results"][0]["live"], requests


@pytest.mark.parametrize(
    ("per_pass", "used", "count", "texts"),
    [
        (
            {"ASCENDING": {"A": 12.0}, "DESCENDING": {"A": 16.0, "C": 14.0}},
            "DESCENDING",
            30,
            ["ASCENDING 12 acquisitions (S1A 12)", "DESCENDING 30 acquisitions (S1A 16, S1C 14)"],
        ),
        (
            {"ASCENDING": {"C": 20, "D": 9}, "DESCENDING": {"C": 10}},
            "ASCENDING",
            29,
            ["ASCENDING 29 acquisitions (S1C 20, S1D 9)", "DESCENDING 10 acquisitions (S1C 10)"],
        ),
        (
            {"ASCENDING": {"A": 31}, "DESCENDING": {"A": 31}},  # a tie
            "DESCENDING",
            31,
            ["ASCENDING 31 acquisitions (S1A 31)", "DESCENDING 31 acquisitions (S1A 31)"],
        ),
        (
            {"ASCENDING": {"A": 4}},
            "ASCENDING",
            4,
            ["ASCENDING 4 acquisitions (S1A 4)", "DESCENDING 0 acquisitions (none)"],
        ),
    ],
)
def test_live_check_of_sentinel1_counts_acquisitions_per_pass_and_satellite(
    ctx, monkeypatch, per_pass, used, count, texts
):
    live, requests = _s1_live(ctx, monkeypatch, per_pass)
    assert live["status"] == "ok"
    assert live["image_count"] == count
    assert live["per_collection"] == {"COPERNICUS/S1_GRD": count}
    for text in texts:
        assert text in live["message"]
    assert "images" not in live["message"]
    assert f"A calendar-year sentinel1 composite of 2025 would use {used}" in live["message"]
    assert "one pass can combine relative orbits" in live["message"]
    assert "instrumentMode IW" in live["method"] and "VV and VH" in live["method"]
    assert "platform_number and orbitNumber_start count once" in live["method"]
    # One request; both passes are counted with the composite's scene filters, as
    # acquisitions (distinct satellite and absolute orbit).
    (request,) = requests
    assert set(request) == {"ASCENDING", "DESCENDING"}
    for orbit, (prop, cid, steps) in request.items():
        assert prop == "platform_number" and cid == "COPERNICUS/S1_GRD"
        bounds, *filters = steps
        assert bounds[0] == "bounds" and bounds[1][0] == pytest.approx(list(CA_BBOX))
        assert filters == [
            ("eq", "instrumentMode", "IW"),
            ("eq", "resolution_meters", 10),
            ("listContains", "transmitterReceiverPolarisation", "VV"),
            ("listContains", "transmitterReceiverPolarisation", "VH"),
            ("date", "2025-01-01", "2026-01-01"),
            ("distinct", ("platform_number", "orbitNumber_start")),
            ("eq", "orbitProperties_pass", orbit),
        ]


@pytest.mark.parametrize("per_pass", [{}, {"ASCENDING": {}, "DESCENDING": None}])
def test_live_check_of_sentinel1_without_acquisitions_names_no_pass(ctx, monkeypatch, per_pass):
    live, _ = _s1_live(ctx, monkeypatch, per_pass)
    assert live["status"] == "ok"
    assert live["image_count"] == 0 and live["per_collection"] == {"COPERNICUS/S1_GRD": 0}
    assert live["message"] == (
        "No IW acquisition with VV and VH intersects the study-area bounding box in 2025, in "
        "either orbit pass, so a sentinel1 composite of 2025 would fail (NoDataError)."
    )
    assert "ASCENDING" not in live["message"] and "DESCENDING" not in live["message"]


def test_sentinel1_is_listed_and_ranked_after_every_other_source(registry):
    sources = {s["source"]: s for s in _call(registry, "list_sources")["sources"]}
    s1 = sources["sentinel1"]
    assert s1["export_resolution_m"] == 10 and s1["value_scale"] == "uint8"
    assert s1["canonical_bands"] == {"R": "VV", "G": "VH", "B": "VV_VH"}
    assert s1["year_first"] == 2015 and s1["year_last"] is None and s1["requires_gee"]

    out = _call(registry, "recommend_configurations", {"year": 2023, "max_candidates": 50})
    ranked = [(c["source"], c["engine"]) for c in out["candidates"]]
    first_s1 = ranked.index(("sentinel1", "delineate-anything"))
    # Every optical and embedding candidate, including the 30 m Landsat and HLS ones,
    # ranks above Sentinel-1 (10 m).
    assert {s for s, _ in ranked[:first_s1]} >= {"sentinel2", "landsat", "hls", "naip"}
    assert {s for s, _ in ranked[first_s1:]} == {"sentinel1"}
    assert ("sentinel1", "ftw") not in ranked and ("sentinel1", "prithvi") not in ranked
    cand = out["candidates"][first_s1]
    assert cand["metrics"]["gsd_m"] == 10.0
    assert any(
        w.startswith("Delineate-Anything: Sentinel-1 SAR") and "expect to need fine-tuning" in w
        for w in cand["warnings"]
    )
    assert out["ordering"].startswith(
        "1) candidates on the SAR sources ['sentinel1', 'umbra', 'capella'] last"
    )
    ordering = out["ordering"]
    assert "0.18 and 0.39 on two 5 km smoke squares" in ordering and "not held out" in ordering
    assert "0.80 on the tree crops of a Madera square" in ordering
    assert "0.455 near Davis (2024), 0.173 in Lea County (2024)" in ordering
    assert "0.11, 0.06 and 0.00 on oil palm and olive squares" in ordering
    est = {e["source"]: e for e in _call(registry, "describe_study_area")["composite_estimates"]}
    assert est["sentinel1"]["dtype"] == "uint8" and est["sentinel1"]["n_bands"] == 3


def test_sentinel1_ranks_last_before_the_sentinel2_years(tmp_path, ctx):
    """2016, outside the US: Sentinel-1 is the finest source, but still ranks last."""
    ctx.study_area = _aoi_file(tmp_path, bbox=AU_BBOX, name="au.geojson")
    out = _call(ToolRegistry(ctx), "recommend_configurations", {"year": 2016})
    ranked = [(c["source"], c["engine"]) for c in out["candidates"]]
    assert ranked[0] == ("landsat-pan", "delineate-anything")
    assert ranked[-1] == ("sentinel1", "delineate-anything")
    assert {"landsat", "hls"} <= {s for s, _ in ranked[:-1]}
    excl = {(e["source"], e["engine"]) for e in out["excluded"]}
    assert ("sentinel2", None) in excl  # before the registry's Sentinel-2 years


def test_sentinel1_ranks_last_when_fine_tuning_is_preferred(tmp_path, ctx):
    """With reference boundaries and fine-tuning preferred, SAR candidates still come last."""
    ctx.study_area = _aoi_file(tmp_path, bbox=AU_BBOX, name="au.geojson")
    import pyproj

    lon, lat = box(*AU_BBOX).centroid.coords[0]
    x, y = pyproj.Transformer.from_crs(4326, 32756, always_xy=True).transform(lon, lat)
    ref = _squares(tmp_path, [200.0, 300.0], crs="EPSG:32756", origin=(x, y))
    out = _call(
        ToolRegistry(ctx),
        "recommend_configurations",
        {"year": 2023, "reference_path": ref, "prefer_label_free": False, "max_candidates": 50},
    )
    ranked = [(c["source"], c["proposal"]["fine_tune"]) for c in out["candidates"]]
    first_s1 = [s for s, _ in ranked].index("sentinel1")
    assert {s for s, _ in ranked[first_s1:]} == {"sentinel1"}
    assert (("sentinel2", False) in ranked[:first_s1]) and (("sentinel1", True) in ranked)
    # Within the SAR candidates the other rules still apply: fine-tuning first here.
    s1 = [ft for s, ft in ranked if s == "sentinel1"]
    assert s1 == sorted(s1, reverse=True)


def test_plan_network_services(monkeypatch):
    from agribound.agent.tools import plan_network_services

    # GEE sources need a project; never depend on the developer's gcloud config.
    monkeypatch.setenv("GEE_PROJECT", "test-project")

    def cfg(**kw):
        base = {"study_area": "bbox:-119.3,36.3,-119.28,36.32", "year": 2023}
        return AgriboundConfig.from_dict({**base, **kw})

    assert plan_network_services(cfg(source="sentinel2")) == [
        "Earth Engine (composite)",
        "Earth Engine (LULC filter, lulc_mode='server')",
    ]
    assert plan_network_services(cfg(source="sentinel2", lulc_filter=False)) == [
        "Earth Engine (composite)"
    ]
    assert plan_network_services(
        cfg(
            source="google-embedding",
            engine="embedding",
            lulc_filter=False,
            google_embedding_backend="source_coop",
        )
    ) == ["Source Cooperative (Google embeddings)"]
    assert plan_network_services(
        cfg(source="tessera-embedding", engine="embedding", lulc_filter=False)
    ) == ["TESSERA (embedding tiles)"]
    assert plan_network_services(cfg(source="usgs-naip-plus", lulc_filter=False)) == [
        "USGS NAIP Plus ImageServer (composite)"
    ]
    asset = cfg(source="sentinel2", lulc_filter=False, study_area="projects/p/assets/aoi")
    assert plan_network_services(asset)[0] == "Earth Engine (study-area asset)"


def test_plan_network_services_for_crop_maps(monkeypatch):
    from agribound.agent.tools import plan_network_services

    monkeypatch.setenv("GEE_PROJECT", "test-project")
    base = {"study_area": "bbox:-93.6,42.0,-93.5,42.1", "year": 2023, "engine": "crop-sequence"}
    for source in ("cdl-sequence", "aci-sequence"):
        cfg = AgriboundConfig.from_dict({**base, "source": source, "lulc_filter": False})
        assert plan_network_services(cfg) == ["Earth Engine (crop maps)"]
        assert cfg.requires_gee()


def test_crop_sequence_is_not_recommended_unless_named(registry):
    out = _call(registry, "recommend_configurations", {"year": 2023, "max_candidates": 50})
    assert not any(c["engine"] == "crop-sequence" for c in out["candidates"])
    assert not any(c["source"] in ("cdl-sequence", "aci-sequence") for c in out["candidates"])
    excl = {(e["source"], e["engine"]): e["reasons"] for e in out["excluded"]}
    for key in (("cdl-sequence", None), ("aci-sequence", None), ("*", "crop-sequence")):
        assert any(
            r.startswith("opt-in; crop-history zones, not field instances") for r in excl[key]
        ), key


@pytest.mark.parametrize(
    "named", [{"sources": ["cdl-sequence"]}, {"engines": ["crop-sequence", "delineate-anything"]}]
)
def test_crop_sequence_is_recommended_when_named(registry, named):
    out = _call(
        registry,
        "recommend_configurations",
        {"year": 2023, "max_candidates": 50, "want_sam_refine": True, **named},
    )
    crop = [c for c in out["candidates"] if c["engine"] == "crop-sequence"]
    assert [c["source"] for c in crop] == ["cdl-sequence"]  # California: ACI has no map
    (cand,) = crop
    assert "sam_refine" not in cand["proposal"].get("config", {})  # refused for class codes
    assert any("not field instances" in w for w in cand["warnings"])
    assert any("lulc_filter=False" in w for w in cand["warnings"])
    excl = {(e["source"], e["engine"]) for e in out["excluded"]}
    assert ("*", "crop-sequence") not in excl
    plan = registry.call("propose_run", cand["proposal"])
    assert plan.ok, plan.error
    assert "Earth Engine (crop maps)" in plan.output["network_services"]


def test_sam_auto_is_not_recommended_unless_named(registry):
    """Engine sam-auto is opt-in: never proposed unless named, and excluded once for all."""
    out = _call(registry, "recommend_configurations", {"year": 2023, "max_candidates": 50})
    assert not any(c["engine"] == "sam-auto" for c in out["candidates"])
    excl = {(e["source"], e["engine"]): e["reasons"] for e in out["excluded"]}
    (reason,) = excl[("*", "sam-auto")]
    assert reason.startswith("opt-in; SAM 2.1 automatic masks, never chosen by default")
    assert [e for e in out["excluded"] if e["engine"] == "sam-auto"] == [
        {"source": "*", "engine": "sam-auto", "reasons": [reason]}
    ]
    # Engines named without it: not considered, so not listed either.
    out = _call(
        registry, "recommend_configurations", {"year": 2023, "engines": ["delineate-anything"]}
    )
    assert not any(e["engine"] == "sam-auto" for e in out["excluded"])
    engines = {e["engine"]: e for e in _call(registry, "list_engines")["engines"]}
    assert set(engines["sam-auto"]["python_packages_found"]) == {"sam2", "cv2"}


def test_sam_auto_is_recommended_when_named_after_delineate_anything(registry):
    out = _call(
        registry,
        "recommend_configurations",
        {
            "year": 2023,
            "engines": ["delineate-anything", "sam-auto"],
            "sources": ["naip", "sentinel1"],
            "want_sam_refine": True,
            "max_candidates": 50,
        },
    )
    ranked = [(c["source"], c["engine"]) for c in out["candidates"]]
    assert ranked == [
        ("naip", "delineate-anything"),
        ("naip", "sam-auto"),
        ("sentinel1", "delineate-anything"),
    ]
    da, sam = out["candidates"][:2]
    assert da["proposal"]["config"]["sam_refine"] is True
    assert "sam_refine" not in sam["proposal"].get("config", {})  # refused by sam-auto
    assert any("refuses sam_refine" in w for w in sam["warnings"])
    excl = {(e["source"], e["engine"]): e["reasons"] for e in out["excluded"]}
    assert ("*", "sam-auto") not in excl
    (reason,) = excl[("sentinel1", "sam-auto")]
    assert "no default setting for sentinel1" in reason
    plan = registry.call("propose_run", sam["proposal"])
    assert plan.ok, plan.error


def test_crop_sequence_window_before_the_first_map_is_excluded(registry):
    out = _call(
        registry,
        "recommend_configurations",
        {"year": 2014, "sources": ["cdl-sequence"], "max_candidates": 50},
    )
    assert not out["candidates"]
    (reasons,) = [e["reasons"] for e in out["excluded"] if e["source"] == "cdl-sequence"]
    assert any("start in 2007, before 2008" in r for r in reasons)


def test_composite_estimate_of_a_crop_map_stack():
    from agribound.agent.tools import _composite_estimate

    est = _composite_estimate("cdl-sequence", (0.0, 0.0, 3000.0, 1500.0), 10_000)
    assert (est.width_px, est.height_px, est.n_bands, est.dtype) == (120, 70, 8, "uint8")
    assert est.uncompressed_mb == pytest.approx(120 * 70 * 8 / 1e6, abs=0.1)
    est5 = _composite_estimate(
        "aci-sequence", (0.0, 0.0, 3000.0, 1500.0), 10_000, crop_sequence_years=5
    )
    assert est5.n_bands == 5 and "padded by 10 px" in est5.note


def test_offline_execute_plan_is_refused_before_the_reviewer_is_asked(tmp_path, monkeypatch):
    # The Sentinel-2 proposal needs a GEE project; never use the developer's gcloud config.
    monkeypatch.setenv("GEE_PROJECT", "test-project")
    calls, asked = [], []
    monkeypatch.setattr("agribound.pipeline.delineate", _fake_delineate(calls))
    tif = tmp_path / "in.tif"
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=4,
        dtype="uint8",
        crs="EPSG:32611",
        transform=from_origin(300000, 4020000, 1, 1),
    ) as dst:
        dst.write(np.ones((4, 4, 4), dtype="uint8"))
    ctx = ToolContext(
        workdir=tmp_path / "w",
        study_area=_aoi_file(tmp_path),
        allow_network=False,
        execution_enabled=True,
        gate=ConfirmationGate(lambda plan: asked.append(plan) or True),
    )
    reg = ToolRegistry(ctx)
    networked = _propose(reg).output
    assert "execute_plan will refuse it" in networked["next_step"]
    out = reg.call("execute_plan", {"plan_id": networked["plan_id"]})
    assert not out.ok and "needs network access" in out.error
    assert "Earth Engine (composite)" in out.error and networked["yaml_path"] in out.error
    assert asked == [] and calls == [] and ctx.gate.executions == 0
    with pytest.raises(tools_mod.NetworkDisabledError):  # the run itself re-checks
        run_approved_plan(ctx, ctx.plans[networked["plan_id"]])
    # A local raster without the LULC filter needs no remote service and may run offline.
    local = reg.call(
        "propose_run",
        {
            "source": "local",
            "engine": "delineate-anything",
            "year": 2023,
            "local_tif_path": str(tif),
            "config": {"lulc_filter": False},
        },
    )
    assert local.ok, local.error
    assert local.output["network_services"] == []
    ran = reg.call("execute_plan", {"plan_id": local.output["plan_id"]})
    assert ran.ok, ran.error
    assert len(asked) == 1 and len(calls) == 1


def test_evaluate_against_reference(tmp_path, registry):
    ref = _squares(tmp_path, [200.0, 200.0], name="ref.gpkg")
    pred = _squares(tmp_path, [200.0], name="pred.gpkg")
    out = _call(
        registry,
        "evaluate_against_reference",
        {"predicted_path": pred, "reference_path": ref, "restrict_to_study_area": False},
    )
    assert out["n_predicted"] == 1 and out["n_reference"] == 2
    assert out["metrics"]["precision"] == pytest.approx(1.0)
    assert out["metrics"]["recall"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# propose_run / execute_plan
# ---------------------------------------------------------------------------


def _propose(registry, **extra):
    args = {"source": "sentinel2", "engine": "delineate-anything", "year": 2023}
    args.update(extra)
    return registry.call("propose_run", args)


def test_propose_run_writes_yaml_and_confines_outputs(registry, ctx):
    out = _propose(registry, rationale="r", limitations=["l"], alternatives=["a"])
    assert out.ok, out.error
    res = out.output
    assert res["plan_id"] in ctx.plans
    assert res["output_path"].startswith(str(ctx.workdir / "plans"))
    cfg = AgriboundConfig.from_yaml(res["yaml_path"])
    assert cfg.output_path == res["output_path"]
    assert cfg.cache_dir == str(ctx.cache_dir)
    assert cfg.overwrite is False and cfg.provenance is True
    assert cfg.gee_project == "test-project"
    assert res["execution_enabled"] is False and "agribound delineate --config" in res["next_step"]
    assert res["estimated_cost"]["composite"]["n_bands"] == 12
    # Same proposal -> same plan.
    again = _propose(registry, rationale="r", limitations=["l"], alternatives=["a"])
    assert again.output["plan_id"] == res["plan_id"]


def test_propose_run_flags_threshold_changes(registry):
    out = _propose(registry, config={"lulc_crop_threshold": 0.1})
    assert out.ok
    assert any("lulc_crop_threshold" in w for w in out.output["warnings"])
    assert out.output["non_default_fields"]["lulc_crop_threshold"]["value"] == 0.1


@pytest.mark.parametrize(
    ("config", "field", "kind"),
    [
        ({"lulc_on_error": "warn"}, "lulc_on_error", "which polygons are kept"),
        ({"sam_refine": True}, "sam_refine", "which polygons are kept"),
        ({"lulc_tree_crops": True}, "lulc_tree_crops", "which polygons are kept"),
        ({"lulc_mode": "raster"}, "lulc_mode", "input data or the method"),
        ({"s2_cloud_mask": "cloud_score_plus"}, "s2_cloud_mask", "input data or the method"),
        ({"composite_method": "greenest"}, "composite_method", "input data or the method"),
    ],
)
def test_propose_run_flags_method_and_filter_changes(registry, config, field, kind):
    out = _propose(registry, config=config)
    assert out.ok, out.error
    assert any(w.startswith(f"{field} is") and kind in w for w in out.output["warnings"])


@pytest.mark.parametrize("source", ["sentinel1", "landsat-pan"])
def test_propose_run_refuses_ndvi_composites_without_nir_and_red(registry, ctx, source):
    out = _propose(registry, source=source, config={"composite_method": "max_ndvi"})
    assert not out.ok
    assert f"needs NIR and red bands, which source '{source}' does not have" in out.error
    assert ctx.plans == {}
    assert _propose(registry, source=source).ok


def test_propose_run_flags_landsat_pan_missions(registry):
    out = _propose(registry, source="landsat-pan", config={"landsat_pan_missions": ["LC08"]})
    assert out.ok, out.error
    assert any(
        w.startswith("landsat_pan_missions is ['LC08'] (package default 'auto')")
        and "input data or the method" in w
        for w in out.output["warnings"]
    )
    assert out.output["config"]["landsat_pan_missions"] == ["LC08"]
    default = _propose(registry, source="landsat-pan")
    assert default.ok, default.error
    assert not any("landsat_pan_missions" in w for w in default.output["warnings"])


def test_propose_run_rejects_landsat_pan_missions_outside_the_year(registry):
    out = _propose(
        registry, source="landsat-pan", year=2020, config={"landsat_pan_missions": "LC09"}
    )
    assert not out.ok
    assert "Invalid configuration" in out.error
    assert "has no mission whose record overlaps year=2020" in out.error


def _run_key_1_0_1(config: dict) -> str:
    """The plan-directory key that agribound 1.0.1 computed for a frozen plan configuration."""
    from agribound._results import results_versions
    from agribound.agent.plans import canonical_json, compute_plan_hash, input_fingerprints
    from agribound.provenance import HASH_CONDITIONAL_FIELDS

    cfg = AgriboundConfig.from_dict(config)
    # 1.0.1 had none of the fields added since (HASH_CONDITIONAL_FIELDS), and its
    # usgs_retries default was 3 (5 since 1.1.0).
    key = {
        k: v
        for k, v in cfg.to_dict().items()
        if k != "output_path" and k not in HASH_CONDITIONAL_FIELDS
    }
    if key["usgs_retries"] == 5:
        key["usgs_retries"] = 3
    versions = results_versions(cfg)
    if versions:
        key["results_versions"] = versions
    return compute_plan_hash(canonical_json(key), canonical_json(input_fingerprints(cfg)))[:10]


def _plan_dir_name(out) -> str:
    assert out.ok, out.error
    from pathlib import Path

    return Path(out.output["output_path"]).parent.name


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"source": "landsat", "year": 2020},
        {"engine": "ftw", "config": {"lulc_dataset": "c3s"}},
        {"config": {"sam_refine": True}},  # with results versions
        {"config": {"lulc_tree_crops": False, "landsat_pan_missions": "auto"}},
    ],
)
def test_plan_directory_of_a_proposal_that_1_0_1_could_make_is_unchanged(registry, extra):
    """Fields added after 1.0.1 at their defaults do not move the plan directory (or output)."""
    out = _propose(registry, **extra)
    cfg = out.output["config"]
    expected = f"{cfg['source']}_{cfg['year']}_{cfg['engine']}_{_run_key_1_0_1(cfg)}"
    assert _plan_dir_name(out) == expected


def test_plan_directory_changes_with_the_new_fields(registry):
    default = _plan_dir_name(_propose(registry))
    assert _plan_dir_name(_propose(registry, config={"lulc_tree_crops": True})) != default
    # Ignored by sentinel2 (not in config_hash), but an explicit value gets its own directory,
    # so two plans never share one plan YAML.
    le07 = _plan_dir_name(_propose(registry, config={"landsat_pan_missions": ["LE07"]}))
    assert le07 != default
    # landsat-pan always keys its missions: an output of the code that mixed Landsat 7 and
    # 8/9 PAN (keyed without them) is never found, so it cannot block the run.
    pan = _propose(registry, source="landsat-pan")
    cfg = pan.output["config"]
    assert _plan_dir_name(pan) != f"landsat-pan_2023_delineate-anything_{_run_key_1_0_1(cfg)}"
    lc08 = _propose(registry, source="landsat-pan", config={"landsat_pan_missions": "LC08"})
    assert _plan_dir_name(lc08) != _plan_dir_name(pan)


def test_tessera_plan_directory_follows_the_store(
    registry, ctx, monkeypatch, tessera_stamp_factory
):
    """A TESSERA output read from store objects that have changed since is not reused
    (agribound.provenance.reuse_mismatch) and agent runs never overwrite, so with network
    access the plan directory includes the fingerprint of the store's objects."""
    pytest.importorskip("geotessera")
    from agribound.composites import tessera_stamp

    tessera = {"source": "tessera-embedding", "engine": "embedding", "year": 2024}
    out = _propose(registry, **tessera)
    cfg = out.output["config"]
    offline = _plan_dir_name(out)  # no network access: the key of 1.0.1
    assert offline == f"tessera-embedding_2024_embedding_{_run_key_1_0_1(cfg)}"

    ctx.allow_network = True
    first = _plan_dir_name(_propose(registry, **tessera))
    assert first != offline
    assert _plan_dir_name(_propose(registry, **tessera)) == first  # the same store
    monkeypatch.setattr(
        tessera_stamp,
        "store_stamp",
        lambda url, bbox, year, **kw: tessera_stamp_factory(url, bbox, year, "d" * 64),
    )
    assert _plan_dir_name(_propose(registry, **tessera)) not in (first, offline)

    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise tessera_stamp.StoreCheckError("no network")

    monkeypatch.setattr(tessera_stamp, "store_stamp", fail)
    assert _plan_dir_name(_propose(registry, **tessera)) == offline
    assert len(calls) == 1
    assert _propose(registry).ok  # other sources: never checked
    assert len(calls) == 1


GHANA_BBOX = "bbox:-1.60,5.58,-1.57,5.61"  # Twifo Praso, Ghana (oil palm)


def _tree_crop_warnings(out) -> list[str]:
    assert out.ok, out.error
    return [w for w in out.output["warnings"] if w.startswith("Tree crops:")]


def test_tree_crop_risk_covers_the_tree_crop_datasets():
    from agribound.postprocess.lulc_filter import TREE_CROP_DATASETS

    assert set(tools_mod._TREE_CROP_RISK) == set(TREE_CROP_DATASETS)


def test_propose_run_warns_that_the_lulc_filter_can_remove_tree_crops(registry):
    (warning,) = _tree_crop_warnings(_propose(registry, study_area=GHANA_BBOX))
    assert warning.startswith(
        "Tree crops: the study area is outside the conterminous US, so the LULC crop filter "
        "(lulc_dataset='auto') uses Dynamic World;"
    )
    assert "0 of 95 oil-palm blocks" in warning
    assert "the user can ask for lulc_tree_crops=True" in warning and "lulc_filter=False" in warning
    # Before Dynamic World's first year the filter uses C3S.
    (old,) = _tree_crop_warnings(
        _propose(registry, study_area=GHANA_BBOX, source="landsat", year=2010)
    )
    assert "(lulc_dataset='auto') uses C3S; it can map orchards" in old


@pytest.mark.parametrize(
    "extra",
    [
        {},  # California: NLCD, whose class 82 includes orchards
        {"study_area": GHANA_BBOX, "config": {"lulc_tree_crops": True}},
        {"study_area": GHANA_BBOX, "config": {"lulc_filter": False}},
        {"study_area": GHANA_BBOX, "config": {"lulc_dataset": "nlcd"}},
        {"study_area": GHANA_BBOX, "config": {"lulc_dataset": "cdl"}},
    ],
)
def test_no_tree_crop_warning_where_the_filter_keeps_tree_crops(registry, extra):
    assert _tree_crop_warnings(_propose(registry, **extra)) == []


@pytest.mark.parametrize(("dataset", "name"), [("dynamic_world", "Dynamic World"), ("c3s", "C3S")])
def test_tree_crop_warning_for_an_explicit_dataset(registry, dataset, name):
    (warning,) = _tree_crop_warnings(_propose(registry, config={"lulc_dataset": dataset}))
    assert warning.startswith(
        f"Tree crops: the LULC crop filter uses {name} (lulc_dataset={dataset!r});"
    )


def test_tree_crop_warning_when_the_study_area_cannot_be_read(registry, monkeypatch):
    # A GEE-asset study area cannot be read offline: the warning names the condition.
    monkeypatch.setattr(tools_mod, "_init_gee", lambda ctx: ctx.require_network("Earth Engine"))
    (warning,) = _tree_crop_warnings(_propose(registry, study_area="projects/p/assets/aoi"))
    assert warning.startswith(
        "Tree crops: outside the conterminous US the LULC crop filter (lulc_dataset='auto') "
        "uses Dynamic World;"
    )


def test_tree_crop_warning_uses_the_local_raster_footprint(tmp_path, ctx):
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    ctx.study_area = None
    registry = ToolRegistry(ctx)

    def raster(name, bounds):
        path = tmp_path / name
        profile = {
            "driver": "GTiff",
            "width": 8,
            "height": 8,
            "count": 3,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": from_bounds(*bounds, 8, 8),
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.ones((3, 8, 8), dtype="uint8"))
        return str(path)

    ghana = _propose(
        registry, source="local", local_tif_path=raster("gh.tif", (-1.6, 5.58, -1.57, 5.61))
    )
    (warning,) = _tree_crop_warnings(ghana)
    assert "the study area is outside the conterminous US" in warning
    california = _propose(registry, source="local", local_tif_path=raster("ca.tif", CA_BBOX))
    assert _tree_crop_warnings(california) == []


def test_recommendations_note_the_tree_crop_risk_outside_the_us(tmp_path, ctx):
    def tree_crop_notes(year):
        out = _call(ToolRegistry(ctx), "recommend_configurations", {"year": year})
        return [n for n in out["notes"] if n.startswith("Tree crops:")]

    assert tree_crop_notes(2023) == []  # California
    ctx.study_area = _aoi_file(tmp_path, bbox=AU_BBOX, name="au.geojson")
    (note,) = tree_crop_notes(2023)
    assert "(lulc_dataset='auto') uses Dynamic World rather than NLCD" in note
    assert "Candidates keep the default lulc_tree_crops=False" in note
    assert "set them only if the user asks" in note
    (old,) = tree_crop_notes(2010)
    assert "uses C3S rather than NLCD" in old


def test_propose_run_returns_the_frozen_config(registry, ctx):
    out = _propose(registry).output
    plan = ctx.plans[out["plan_id"]]
    assert out["config"] == plan.config
    assert out["config"]["output_path"] == out["output_path"]
    assert out["network_services"] == [
        "Earth Engine (composite)",
        "Earth Engine (LULC filter, lulc_mode='server')",
    ]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"output_path": "x.gpkg"}, "cannot be set by a proposal"),
        ({"overwrite": True}, "cannot be set by a proposal"),
        ({"provenance": False}, "cannot be set by a proposal"),
        ({"cache_dir": "/tmp/x"}, "cannot be set by a proposal"),
        ({"year": 2022}, "top-level"),
        ({"not_a_field": 1}, "Unknown configuration field"),
        ({"lulc_crop_threshold": 3.0}, "Invalid configuration"),
    ],
)
def test_propose_run_rejects_invalid_configs(registry, config, message):
    out = _propose(registry, config=config)
    assert not out.ok and message in out.error


def test_propose_run_rejects_bad_output_names_and_years(registry):
    assert "plain file name" in _propose(registry, output_name="../x.gpkg").error
    assert "must end with" in _propose(registry, output_name="x.shp").error
    assert "outside the available range" in _propose(registry, year=2010).error
    assert "does not support source" in _propose(registry, engine="embedding").error


def test_execute_plan_disabled_in_dry_run(registry, ctx):
    plan_id = _propose(registry).output["plan_id"]
    out = registry.call("execute_plan", {"plan_id": plan_id})
    assert not out.ok and "disabled" in out.error
    with pytest.raises(ExecutionDisabledError):
        run_approved_plan(ctx, ctx.plans[plan_id])


def _fake_delineate(calls, fail=False):
    def fake(config=None, **kwargs):
        calls.append(config)
        if fail:
            raise RuntimeError("GEE quota exceeded")
        gdf = gpd.GeoDataFrame(
            {"metrics:area": [10_000.0, 30_000.0]},
            geometry=[box(0, 0, 100, 100), box(200, 0, 373.2, 173.2)],
            crs="EPSG:32611",
        )
        gdf.attrs.update(run_id="run-1", lulc_stats={"n_in": 3, "n_kept": 2})
        return gdf

    return fake


def test_execute_plan_runs_only_approved_plan(registry, ctx, monkeypatch):
    calls = []
    monkeypatch.setattr("agribound.pipeline.delineate", _fake_delineate(calls))
    _enable_execution(ctx, ConfirmationGate(lambda plan: False))
    plan_id = _propose(registry).output["plan_id"]
    denied = registry.call("execute_plan", {"plan_id": plan_id})
    assert not denied.ok and "not approved" in denied.error and calls == []

    ctx.gate = ConfirmationGate(lambda plan: plan.plan_id == plan_id)
    ok = registry.call("execute_plan", {"plan_id": plan_id})
    assert ok.ok, ok.error
    assert len(calls) == 1 and isinstance(calls[0], AgriboundConfig)
    assert calls[0].output_path == ctx.plans[plan_id].config["output_path"]
    summary = ok.output
    assert summary["n_polygons"] == 2 and summary["run_id"] == "run-1"
    assert summary["area_ha"]["total"] == pytest.approx(4.0)
    assert summary["lulc_stats"] == {"n_in": 3, "n_kept": 2}
    assert summary["approval"]["used_utc"] is not None
    assert ctx.executions[-1]["status"] == "success"

    again = registry.call("execute_plan", {"plan_id": plan_id})
    assert not again.ok and "max_executions=1" in again.error
    assert len(calls) == 1


def test_summary_of_a_reused_output_reads_the_provenance_facts(registry, ctx, monkeypatch):
    from agribound.provenance import write_provenance

    def reused(config=None, **kwargs):
        # What the pipeline returns when it reuses a matching output: attrs carry the
        # run ID and engine metadata, but not the LULC/SAM statistics.
        gdf = gpd.GeoDataFrame(
            {"metrics:area": [10_000.0]}, geometry=[box(0, 0, 100, 100)], crs="EPSG:32611"
        )
        gdf.attrs.update(run_id="run-old", reused=True, engine_meta={"backend": "x"})
        write_provenance(
            config.output_path,
            {
                "status": "success",
                "warnings": ["LULC dataset year 2023 used for 2024"],
                "facts": {
                    "n_detected": 5,
                    "n_postprocessed": 4,
                    "n_after_lulc": 1,
                    "n_output": 1,
                    "lulc_status": "applied",
                    "lulc_stats": {"n_in": 4, "n_kept": 1},
                    "sam_stats": {"n_total": 4, "n_refined": 2},
                },
            },
        )
        return gdf

    monkeypatch.setattr("agribound.pipeline.delineate", reused)
    _enable_execution(ctx, ConfirmationGate(lambda plan: True))
    plan_id = _propose(registry).output["plan_id"]
    out = registry.call("execute_plan", {"plan_id": plan_id})
    assert out.ok, out.error
    summary = out.output
    assert summary["reused_existing_output"] is True and summary["run_id"] == "run-old"
    assert summary["stage_counts"] == {
        "n_detected": 5,
        "n_postprocessed": 4,
        "n_after_lulc": 1,
        "n_output": 1,
    }
    assert summary["lulc_status"] == "applied"
    assert summary["lulc_stats"] == {"n_in": 4, "n_kept": 1}
    assert summary["sam_stats"] == {"n_total": 4, "n_refined": 2}
    assert summary["engine_meta"] == {"backend": "x"}
    assert summary["provenance_warnings"] == ["LULC dataset year 2023 used for 2024"]


def test_failed_execution_is_recorded_and_consumes_the_execution(registry, ctx, monkeypatch):
    calls = []
    monkeypatch.setattr("agribound.pipeline.delineate", _fake_delineate(calls, fail=True))
    _enable_execution(ctx, ConfirmationGate(lambda plan: True))
    plan_id = _propose(registry).output["plan_id"]
    out = registry.call("execute_plan", {"plan_id": plan_id})
    assert not out.ok and "GEE quota exceeded" in out.error
    assert ctx.executions[-1]["status"] == "failed"
    assert ctx.gate.executions == 1


def test_unknown_plan_id(registry, ctx):
    _enable_execution(ctx, ConfirmationGate(lambda plan: True))
    out = registry.call("execute_plan", {"plan_id": "plan-000"})
    assert not out.ok and "Unknown plan_id" in out.error


def test_min_refinable_side_bisection_matches_formula():
    from agribound.engines.samgeo_engine import is_refinable

    for gsd in (0.6, 1.0, 10.0, 30.0):
        side = tools_mod._min_refinable_square_side_m(gsd, 64, 0.15, is_refinable)
        exact = 64 * gsd / 1.3
        assert exact <= side <= exact + 0.2
        assert math.isclose(side * 10, round(side * 10))
        assert is_refinable((0, 0, side, side), (gsd, gsd), 64, 0.15)
        assert not is_refinable((0, 0, side - 0.2, side - 0.2), (gsd, gsd), 64, 0.15)


# ---------------------------------------------------------------------------
# Open VHR SAR sources (umbra, capella)
# ---------------------------------------------------------------------------


def test_vhr_sar_is_recommended_only_when_named_with_the_tested_da_settings(registry):
    out = _call(registry, "recommend_configurations", {"year": 2024, "max_candidates": 50})
    assert not any(c["source"] in ("umbra", "capella") for c in out["candidates"])
    excl = {(e["source"], e["engine"]): e["reasons"] for e in out["excluded"]}
    assert any("few sites" in r for r in excl[("umbra", None)])
    assert any("few sites" in r for r in excl[("capella", None)])
    out = _call(
        registry, "recommend_configurations", {"year": 2024, "sources": ["umbra", "capella"]}
    )
    assert {(c["source"], c["engine"]) for c in out["candidates"]} == {
        ("umbra", "delineate-anything"),
        ("capella", "delineate-anything"),
    }
    for cand in out["candidates"]:
        assert cand["proposal"]["config"]["engine_params"] == {
            "processing_gsd_m": 3,
            "merge_tile_pieces": False,
        }
        assert cand["metrics"]["gsd_m"] == 1.0
        assert any("processing_gsd_m=3" in rule for rule in cand["rules_applied"])
        assert any("CC BY 4.0" in w for w in cand["warnings"])
    assert any("Delineate-Anything's engine_params" in note for note in out["notes"])


def test_plan_network_services_for_vhr_sar():
    from agribound.agent.tools import plan_network_services

    base = {"study_area": "bbox:-121.8,38.5,-121.79,38.51", "year": 2024, "lulc_filter": False}
    umbra = AgriboundConfig.from_dict({**base, "source": "umbra"})
    assert plan_network_services(umbra) == [
        "AWS Registry of Open Data: s3://umbra-open-data-catalog (archive index and GEC downloads)",
        "AWS Registry of Open Data: Copernicus GLO-30 DEM (GLO-90 where GLO-30 has no public "
        "tile), and cdn.proj.org (EGM2008 geoid) for the RPC orthorectification",
        "USGS NAIP Plus ImageServer (co-registration base, where available)",
    ]
    capella = AgriboundConfig.from_dict(
        {**base, "source": "capella", "year": 2023, "vhr_sar_coregister": "sar"}
    )
    assert plan_network_services(capella) == [
        "AWS Registry of Open Data: s3://capella-open-data (archive index and GEO downloads)"
    ]


def test_propose_run_for_vhr_sar(registry):
    out = _propose(
        registry,
        source="umbra",
        year=2024,
        config={"lulc_filter": False, "vhr_sar_max_collects": 4},
    )
    assert out.ok, out.error
    warnings = out.output["warnings"]
    assert any("SAR data: Umbra Lab, Inc., CC BY 4.0" in w for w in warnings)
    assert any("the tested recipe sets engine_params" in w for w in warnings)
    tuned = _propose(
        registry,
        source="capella",
        year=2023,
        config={"lulc_filter": False, "engine_params": {"processing_gsd_m": 3,
                                                        "merge_tile_pieces": False}},
    )  # fmt: skip
    assert tuned.ok, tuned.error
    assert not any("the tested recipe sets engine_params" in w for w in tuned.output["warnings"])
    cached = _propose(registry, source="umbra", year=2024, config={"vhr_sar_cache_dir": "/x"})
    assert not cached.ok and "vhr_sar_cache_dir" in cached.error


def test_live_check_for_vhr_sar_uses_the_index(registry, ctx, monkeypatch):
    from agribound.composites import vhr_sar

    footprint = ((-119.4, 36.2), (-119.2, 36.2), (-119.2, 36.4), (-119.4, 36.4), (-119.4, 36.2))
    records = [
        vhr_sar.CollectRecord(
            source="umbra",
            collect_id=f"c{i}",
            datetime=f"2023-0{i + 5}-01T10:00:00.000000Z",
            polarization="VV",
            orbit_state="ascending",
            look_side="left",
            incidence_deg=40.0 + i,
            view_azimuth_deg=90.0,
            footprint=footprint,
            product_key=f"k/c{i}_GEC.tif",
            product_bytes=1,
            product_etag="e",
            product_last_modified="2023-01-01T00:00:00Z",
        )
        for i in range(3)
    ]
    index = vhr_sar.ArchiveIndex("umbra", "2026-10-09T00:00:00Z", records)
    monkeypatch.setattr(vhr_sar, "load_index", lambda source, **kw: index)
    ctx.allow_network = True
    out = _call(registry, "check_availability", {"year": 2023, "sources": ["umbra"], "live": True})
    live = out["results"][0]["live"]
    assert live["status"] == "ok" and live["image_count"] == 3
    assert "3 collects cover the bounding box in 2023" in live["message"]
    assert "2023-05-01, 2023-06-01, 2023-07-01" in live["message"]
    # Keyed by the bucket that holds the collects (not "s3://umbra").
    assert live["per_collection"] == {"s3://umbra-open-data-catalog": 3}
    from agribound.agent.tools import TOOL_SPECS

    (spec,) = [t for t in TOOL_SPECS if t.name == "check_availability"]
    assert "Umbra and Capella collects" in spec.description
