"""Integration tests for the agribound CLI (and the fine-tuning dispatcher).

Nothing here needs network, GEE or a GPU: ``delineate``/``composite`` are run
with ``--dry-run`` or with the pipeline functions mocked.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import geopandas as gpd
import pytest
import yaml
from click.testing import CliRunner
from shapely.geometry import box

from agribound import cli as cli_module
from agribound.cli import main

SUBCOMMANDS = [
    "delineate",
    "composite",
    "scenes",
    "prefetch",
    "evaluate",
    "query-ftw",
    "list-engines",
    "list-sources",
    "list-ftw-models",
    "auth",
]


def _flat(text: str) -> str:
    """Collapse whitespace so wrapped help text can be searched."""
    return " ".join(text.split())


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _dry_run(runner: CliRunner, args: list[str]) -> dict:
    result = runner.invoke(main, [*args, "--dry-run"])
    assert result.exit_code == 0, result.output
    return yaml.safe_load(result.stdout)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def base_yaml(tmp_path: Path) -> Path:
    return _write_yaml(
        tmp_path / "run.yaml",
        {
            "study_area": "aoi.geojson",
            "source": "sentinel2",
            "year": 2020,
            "engine": "delineate-anything",
            "gee_project": "test-project",
            "output_path": str(tmp_path / "out" / "fields.gpkg"),
            "lulc_filter": False,
            "engine_params": {"da_model": "DelineateAnything-S", "conf": 0.2},
        },
    )


# ---------------------------------------------------------------------------
# Group-level behaviour
# ---------------------------------------------------------------------------


class TestCliVersion:
    """Test --version flag."""

    def test_version_output(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "agribound" in result.output.lower()

    def test_verbose_flag_kept(self, runner):
        result = runner.invoke(main, ["-v", "list-engines"])
        assert result.exit_code == 0, result.output


class TestCliHelp:
    """Every subcommand must render --help."""

    @pytest.mark.parametrize("command", SUBCOMMANDS)
    def test_help_exits_ok(self, runner, command):
        result = runner.invoke(main, [command, "--help"])
        assert result.exit_code == 0, result.output
        assert "Usage:" in result.output

    def test_optional_commands_help(self, runner):
        for name in set(main.commands) - set(SUBCOMMANDS):
            result = runner.invoke(main, [name, "--help"])
            assert result.exit_code == 0, (name, result.output)

    def test_units_in_delineate_help(self, runner):
        text = _flat(runner.invoke(main, ["delineate", "--help"]).output)
        assert "simplification tolerance in meters" in text
        assert "Minimum field area in m^2" in text
        assert "pixels" not in text.split("--simplify", 1)[1].split("--device", 1)[0]

    def test_contract_flags_present(self, runner):
        text = runner.invoke(main, ["delineate", "--help"]).output
        for flag in (
            "--engine-param",
            "--lulc-filter / --no-lulc-filter",
            "--lulc-threshold",
            "--lulc-dataset",
            "--lulc-mode",
            "--lulc-on-error",
            "--seed",
            "--export-crs",
            "--tessera-version",
            "--sam-refine / --no-sam-refine",
            "--sam-backend",
            "--fine-tune-epochs",
            "--fine-tune-split",
            "--tile-size",
            "--cache-dir",
            "--overwrite",
            "--gee-service-account-key",
            "--gee-high-volume",
            "--gee-max-requests",
            "--s2-cloud-mask",
            "--naip-resolution",
            "--no-provenance",
            "--dry-run",
            "--config",
            "--aoi-selection",
            "--embedding-cache-dir",
            "--landsat-pan-missions",
            "--lulc-tree-crops / --no-lulc-tree-crops",
        ):
            assert flag in text, flag
        composite_text = runner.invoke(main, ["composite", "--help"]).output
        assert "--embedding-cache-dir" in composite_text
        assert "--aoi-selection" not in composite_text  # applied after delineation only
        # Stage-A options: they decide the composite and the prefetched LULC raster.
        assert "--landsat-pan-missions" in composite_text
        assert "--lulc-tree-crops / --no-lulc-tree-crops" in composite_text

    def test_composite_has_no_engine_only_flags(self, runner):
        text = runner.invoke(main, ["composite", "--help"]).output
        assert "--study-area" in text and "--dry-run" in text
        for flag in ("--engine-param", "--min-area", "--simplify", "--fine-tune", "--sam-refine"):
            assert flag not in text, flag

    def test_help_defaults_match_config(self, runner):
        from agribound.config import AgriboundConfig

        text = _flat(runner.invoke(main, ["delineate", "--help"]).output)
        cfg_defaults = AgriboundConfig.__dataclass_fields__
        assert f"[default: {cfg_defaults['seed'].default}]" in text
        assert f"[default: {cfg_defaults['min_field_area_m2'].default}]" in text
        missions = text.split("--landsat-pan-missions", 1)[1].split("--export-crs", 1)[0]
        assert "LE07, LC08, LC09. [default: auto]" in missions
        tree = text.split("--lulc-tree-crops / --no-lulc-tree-crops", 1)[1].split("--seed", 1)[0]
        assert "[default: False]" in tree and "Dynamic World crops+trees" in tree


# ---------------------------------------------------------------------------
# Listings
# ---------------------------------------------------------------------------


class TestCliListEngines:
    """Test the list-engines command."""

    def test_lists_every_registry_engine(self, runner):
        from agribound.registry import ENGINE_REGISTRY

        result = runner.invoke(main, ["list-engines"])
        assert result.exit_code == 0
        for name in ENGINE_REGISTRY:
            assert name in result.output, name

    def test_list_engines_shows_approach_and_flags(self, runner):
        result = runner.invoke(main, ["list-engines"])
        text = result.output.lower()
        assert "segmentation" in text or "clustering" in text
        assert "fine-tunable" in text
        assert "label-free" in text


class TestCliListSources:
    """Test the list-sources command."""

    def test_lists_every_registry_source(self, runner):
        from agribound.registry import SOURCE_REGISTRY

        result = runner.invoke(main, ["list-sources"])
        assert result.exit_code == 0
        for name in SOURCE_REGISTRY:
            assert name in result.output, name

    def test_list_sources_shows_resolution(self, runner):
        result = runner.invoke(main, ["list-sources"])
        # Sentinel-2 is 10m
        assert "10m" in result.output

    def test_list_sources_shows_sentinel1(self, runner):
        result = runner.invoke(main, ["list-sources"])
        assert result.exit_code == 0
        lines = result.output.splitlines()
        (row,) = [line for line in lines if line.strip().startswith("sentinel1")]
        assert "Sentinel-1 C-band SAR GRD" in row
        assert "10m" in row and "2015-present" in row and "[GEE]" in row
        assert "restricted" not in row
        rows = [line.split()[0] for line in lines if line.startswith("  ")]
        assert rows.index("sentinel1") == rows.index("spot-pan") + 1


class TestCliListFtwModels:
    def test_forwards_all_flag(self, runner, monkeypatch):
        ftw_mod = importlib.import_module("agribound.engines.ftw")
        calls = []

        def fake(include_legacy=False):
            calls.append(include_legacy)
            return {
                "FTW_PRUE_EFNET_B5": {"title": "PRUE B5", "default": True, "requires_window": True},
                "OLD_MODEL": {"title": "Old", "legacy": True},
            }

        monkeypatch.setattr(ftw_mod, "list_ftw_models", fake)
        result = runner.invoke(main, ["list-ftw-models"])
        assert result.exit_code == 0, result.output
        assert "FTW_PRUE_EFNET_B5" in result.output and "default" in result.output
        result = runner.invoke(main, ["list-ftw-models", "--all"])
        assert result.exit_code == 0
        assert calls == [False, True]

    def test_missing_ftw_tools_is_a_clean_error(self, runner, monkeypatch):
        ftw_mod = importlib.import_module("agribound.engines.ftw")

        def fake(include_legacy=False):
            raise ImportError("ftw-tools is required to list FTW models.")

        monkeypatch.setattr(ftw_mod, "list_ftw_models", fake)
        result = runner.invoke(main, ["list-ftw-models"])
        assert result.exit_code == 1
        assert "ftw-tools is required" in result.output


# ---------------------------------------------------------------------------
# delineate --config / overrides / dry-run
# ---------------------------------------------------------------------------


class TestDelineateConfig:
    def test_config_without_study_area_flag(self, runner, base_yaml):
        data = _dry_run(runner, ["delineate", "--config", str(base_yaml)])
        assert data["study_area"] == "aoi.geojson"
        assert data["year"] == 2020
        assert data["engine"] == "delineate-anything"
        assert data["lulc_filter"] is False
        assert data["engine_params"] == {"da_model": "DelineateAnything-S", "conf": 0.2}

    def test_explicit_flags_override_yaml(self, runner, base_yaml):
        data = _dry_run(
            runner,
            [
                "delineate",
                "--config",
                str(base_yaml),
                "--engine",
                "ftw",
                "--study-area",
                "other.geojson",
                "--lulc-filter",
                "--engine-param",
                "model=FTW_PRUE_EFNET_B5",
                "--engine-param",
                "conf=0.5",
            ],
        )
        assert data["engine"] == "ftw"
        assert data["study_area"] == "other.geojson"
        assert data["lulc_filter"] is True
        assert data["year"] == 2020  # not passed -> YAML value kept
        # engine_params are merged key by key; CLI keys win
        assert data["engine_params"] == {
            "da_model": "DelineateAnything-S",
            "conf": 0.5,
            "model": "FTW_PRUE_EFNET_B5",
        }

    def test_explicit_flag_equal_to_default_still_overrides(self, runner, base_yaml):
        # 2024 is also the CLI/config default: explicitly passing it must still win.
        data = _dry_run(runner, ["delineate", "--config", str(base_yaml), "--year", "2024"])
        assert data["year"] == 2024

    def test_unpassed_flags_do_not_override_yaml(self, runner, tmp_path):
        cfg = _write_yaml(
            tmp_path / "c.yaml",
            {
                "study_area": "a.geojson",
                "gee_project": "p",
                "min_field_area_m2": 999.0,
                "simplify_tolerance": 0.0,
                "seed": 7,
                "provenance": False,
                "output_path": str(tmp_path / "f.gpkg"),
            },
        )
        data = _dry_run(runner, ["delineate", "--config", str(cfg)])
        assert data["min_field_area_m2"] == 999.0
        assert data["simplify_tolerance"] == 0.0
        assert data["seed"] == 7
        assert data["provenance"] is False

    def test_mapped_flags(self, runner, tmp_path):
        data = _dry_run(
            runner,
            [
                "delineate",
                "--study-area",
                "a.geojson",
                "--gee-project",
                "p",
                "--output",
                str(tmp_path / "x.gpkg"),
                "--min-area",
                "1000",
                "--simplify",
                "5",
                "--local-tif",
                "t.tif",
                "--date-range",
                "2024-04-01",
                "2024-09-30",
                "--no-provenance",
                "--overwrite",
                "--seed",
                "3",
                "--naip-resolution",
                "0.6",
                "--lulc-threshold",
                "0.5",
                "--lulc-mode",
                "raster",
                "--lulc-on-error",
                "warn",
                "--s2-cloud-mask",
                "cloud_score_plus",
                "--export-crs",
                "EPSG:5070",
                "--fine-tune-split",
                "random",
                "--sam-backend",
                "sam2.1",
                "--sam-refine",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--gee-max-requests",
                "4",
                "--gee-high-volume",
                "--tile-size",
                "2048",
            ],
        )
        assert data["output_path"] == str(tmp_path / "x.gpkg")
        assert data["min_field_area_m2"] == 1000.0
        assert data["simplify_tolerance"] == 5.0
        assert data["local_tif_path"] == "t.tif"
        assert data["date_range"] == ["2024-04-01", "2024-09-30"]
        assert data["provenance"] is False
        assert data["overwrite"] is True
        assert data["seed"] == 3
        assert data["naip_resolution_m"] == 0.6
        assert data["lulc_crop_threshold"] == 0.5
        assert data["lulc_mode"] == "raster"
        assert data["lulc_on_error"] == "warn"
        assert data["s2_cloud_mask"] == "cloud_score_plus"
        assert data["export_crs"] == "EPSG:5070"
        assert data["fine_tune_split"] == "random"
        assert data["sam_backend"] == "sam2.1"
        assert data["sam_refine"] is True
        assert data["cache_dir"] == str(tmp_path / "cache")
        assert data["gee_max_requests"] == 4
        assert data["gee_high_volume"] is True
        assert data["tile_size"] == 2048

    def test_landsat_pan_missions_and_lulc_tree_crops(self, runner, tmp_path):
        args = [
            "delineate",
            "--study-area",
            "a.geojson",
            "--gee-project",
            "p",
            "--source",
            "landsat-pan",
            "--year",
            "2015",
            "--output",
            str(tmp_path / "x.gpkg"),
        ]
        data = _dry_run(runner, args)
        assert data["landsat_pan_missions"] == "auto"
        assert data["lulc_tree_crops"] is False
        data = _dry_run(
            runner, [*args, "--landsat-pan-missions", "lc08, LE07", "--lulc-tree-crops"]
        )
        assert data["landsat_pan_missions"] == ["LE07", "LC08"]
        assert data["lulc_tree_crops"] is True
        data = _dry_run(runner, [*args, "--landsat-pan-missions", "AUTO", "--no-lulc-tree-crops"])
        assert data["landsat_pan_missions"] == "auto"
        assert data["lulc_tree_crops"] is False
        data = _dry_run(runner, ["composite", *args[1:], "--landsat-pan-missions", "LE07"])
        assert data["landsat_pan_missions"] == ["LE07"]

    def test_crop_sequence_years(self, runner, tmp_path):
        args = [
            "delineate",
            "--study-area",
            "bbox:-93.6,42.0,-93.5,42.1",
            "--gee-project",
            "p",
            "--source",
            "cdl-sequence",
            "--engine",
            "crop-sequence",
            "--year",
            "2023",
            "--no-lulc-filter",
            "--output",
            str(tmp_path / "zones.gpkg"),
        ]
        data = _dry_run(runner, [*args, "--crop-sequence-years", "8"])
        assert data["source"] == "cdl-sequence" and data["engine"] == "crop-sequence"
        assert data["crop_sequence_years"] == 8 and data["year"] == 2023
        assert _dry_run(runner, [*args, "--crop-sequence-years", "5"])["crop_sequence_years"] == 5
        data = _dry_run(runner, ["composite", *args[1:], "--crop-sequence-years", "3"])
        assert data["crop_sequence_years"] == 3
        # A window that starts before the first CDL year is a usage error.
        result = runner.invoke(main, [*args, "--crop-sequence-years", "20", "--dry-run"])
        assert result.exit_code == 2 and "starts in 2008" in _flat(result.output)
        text = _flat(runner.invoke(main, ["delineate", "--help"]).output)
        own = text.split("--crop-sequence-years", 1)[1].split("--vhr-sar-collect-ids", 1)[0]
        assert "ending in --year" in own and "[default: 8]" in own

    @pytest.mark.parametrize("value", ["LT05", "LC08,L9", ""])
    def test_invalid_landsat_pan_missions_is_usage_error(self, runner, value):
        result = runner.invoke(
            main,
            [
                "delineate",
                "--study-area",
                "a.geojson",
                "--gee-project",
                "p",
                "--source",
                "landsat-pan",
                "--landsat-pan-missions",
                value,
                "--dry-run",
            ],
        )
        assert result.exit_code == 2
        assert "Invalid configuration" in result.output
        assert "landsat_pan_missions" in result.output

    @pytest.mark.parametrize("command", ["delineate", "composite"])
    @pytest.mark.parametrize("source", ["sentinel1", "landsat-pan"])
    def test_ndvi_composite_method_without_nir_and_red_is_usage_error(
        self, runner, command, source
    ):
        args = [command, "--study-area", "a.geojson", "--gee-project", "p", "--source", source]
        result = runner.invoke(main, [*args, "--composite-method", "greenest", "--dry-run"])
        assert result.exit_code == 2
        assert "Invalid configuration" in result.output
        assert f"needs NIR and red bands, which source '{source}'" in _flat(result.output)
        assert _dry_run(runner, args)["composite_method"] == "median"

    def test_new_flags_override_yaml_and_round_trip(self, runner, tmp_path):
        cfg = _write_yaml(
            tmp_path / "lp.yaml",
            {
                "study_area": "a.geojson",
                "source": "landsat-pan",
                "year": 2015,
                "gee_project": "p",
                "output_path": str(tmp_path / "f.gpkg"),
                "landsat_pan_missions": ["LC08", "LE07"],
                "lulc_tree_crops": True,
            },
        )
        data = _dry_run(runner, ["delineate", "--config", str(cfg)])
        assert data["landsat_pan_missions"] == ["LE07", "LC08"]
        assert data["lulc_tree_crops"] is True
        data = _dry_run(
            runner,
            [
                "delineate",
                "--config",
                str(cfg),
                "--landsat-pan-missions",
                "LC08",
                "--no-lulc-tree-crops",
            ],
        )
        assert data["landsat_pan_missions"] == ["LC08"]
        assert data["lulc_tree_crops"] is False
        # The resolved YAML reads back unchanged.
        first = runner.invoke(main, ["delineate", "--config", str(cfg), "--dry-run"])
        resolved = tmp_path / "resolved.yaml"
        resolved.write_text(first.stdout)
        second = runner.invoke(main, ["delineate", "--config", str(resolved), "--dry-run"])
        assert second.exit_code == 0, second.output
        assert second.stdout == first.stdout

    def test_aoi_selection_and_embedding_cache_dir(self, runner, base_yaml, tmp_path):
        data = _dry_run(runner, ["delineate", "--config", str(base_yaml)])
        assert data["aoi_selection"] == "representative_point"
        assert data["embedding_cache_dir"] is None
        data = _dry_run(
            runner,
            [
                "delineate",
                "--config",
                str(base_yaml),
                "--aoi-selection",
                "clip",
                "--embedding-cache-dir",
                str(tmp_path / "emb"),
            ],
        )
        assert data["aoi_selection"] == "clip"
        assert data["embedding_cache_dir"] == str(tmp_path / "emb")
        data = _dry_run(
            runner,
            ["composite", "--config", str(base_yaml), "--embedding-cache-dir", "/scratch/e"],
        )
        assert data["embedding_cache_dir"] == "/scratch/e"
        bad = runner.invoke(main, ["delineate", "--config", str(base_yaml), "--aoi-selection", "x"])
        assert bad.exit_code == 2

    def test_study_area_required_without_config(self, runner):
        result = runner.invoke(main, ["delineate", "--dry-run"])
        assert result.exit_code == 2
        assert "--study-area" in result.output

    def test_study_area_required_when_yaml_lacks_it(self, runner, tmp_path):
        cfg = _write_yaml(tmp_path / "c.yaml", {"gee_project": "p"})
        result = runner.invoke(main, ["delineate", "--config", str(cfg), "--dry-run"])
        assert result.exit_code == 2
        assert "--study-area" in result.output
        data = _dry_run(runner, ["delineate", "--config", str(cfg), "--study-area", "a.geojson"])
        assert data["study_area"] == "a.geojson"

    def test_study_area_optional_for_local_source(self, runner, tmp_path):
        data = _dry_run(
            runner,
            [
                "delineate",
                "--source",
                "local",
                "--local-tif",
                "img.tif",
                "--output",
                str(tmp_path / "f.gpkg"),
            ],
        )
        assert data["source"] == "local"
        assert data["study_area"] == ""

    def test_default_output_name(self, runner):
        data = _dry_run(
            runner,
            ["delineate", "--study-area", "a.geojson", "--gee-project", "p", "--year", "2021"],
        )
        assert data["output_path"] == "fields_sentinel2_2021.gpkg"
        assert data["year"] == 2021

    def test_dry_run_round_trips(self, runner, base_yaml, tmp_path):
        first = runner.invoke(main, ["delineate", "--config", str(base_yaml), "--dry-run"])
        assert first.exit_code == 0, first.output
        resolved = tmp_path / "resolved.yaml"
        resolved.write_text(first.stdout)
        second = runner.invoke(main, ["delineate", "--config", str(resolved), "--dry-run"])
        assert second.exit_code == 0, second.output
        assert second.stdout == first.stdout

    def test_invalid_yaml_key_is_usage_error(self, runner, tmp_path):
        cfg = _write_yaml(tmp_path / "c.yaml", {"study_area": "a", "not_a_field": 1})
        result = runner.invoke(main, ["delineate", "--config", str(cfg), "--dry-run"])
        assert result.exit_code == 2
        assert "not_a_field" in result.output

    def test_non_mapping_yaml_is_usage_error(self, runner, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("- a\n- b\n")
        result = runner.invoke(main, ["delineate", "--config", str(cfg), "--dry-run"])
        assert result.exit_code == 2

    def test_fine_tune_non_tunable_engine_reports_hint(self, runner, tmp_path):
        result = runner.invoke(
            main,
            [
                "delineate",
                "--study-area",
                "a.geojson",
                "--gee-project",
                "p",
                "--engine",
                "ftw",
                "--fine-tune",
                "--reference",
                "ref.gpkg",
                "--dry-run",
            ],
        )
        assert result.exit_code == 2
        assert "ftw model fit" in result.output

    def test_dry_run_does_not_run_pipeline(self, runner, base_yaml, monkeypatch):
        pipeline = importlib.import_module("agribound.pipeline")

        def boom(*args, **kwargs):
            raise AssertionError("pipeline must not run with --dry-run")

        monkeypatch.setattr(pipeline, "delineate", boom)
        result = runner.invoke(main, ["delineate", "--config", str(base_yaml), "--dry-run"])
        assert result.exit_code == 0, result.output

    def test_runs_pipeline_with_resolved_config(self, runner, base_yaml, monkeypatch):
        pipeline = importlib.import_module("agribound.pipeline")
        seen = {}

        def fake_delineate(study_area=None, config=None, **kwargs):
            seen["config"] = config
            seen["study_area"] = study_area
            gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
            gdf.attrs["evaluation_metrics"] = {"precision": 0.5, "recall": 0.25, "f1": 1 / 3}
            return gdf

        monkeypatch.setattr(pipeline, "delineate", fake_delineate)
        result = runner.invoke(main, ["delineate", "--config", str(base_yaml), "--year", "2023"])
        assert result.exit_code == 0, result.output
        assert seen["config"].year == 2023
        assert seen["config"].study_area == "aoi.geojson"
        assert seen["study_area"] is None  # everything travels in config
        assert "Delineated 1 field boundaries" in result.output
        assert "precision=0.500" in result.output and "f1=0.333" in result.output
        assert "Provenance" not in result.output  # no record was written

    def test_existing_output_conflict_is_clean_error(self, runner, base_yaml, monkeypatch):
        pipeline = importlib.import_module("agribound.pipeline")

        def conflict(study_area=None, config=None, **kwargs):
            raise FileExistsError("Output exists ... (CLI: --overwrite)")

        monkeypatch.setattr(pipeline, "delineate", conflict)
        result = runner.invoke(main, ["delineate", "--config", str(base_yaml)])
        assert result.exit_code == 1
        assert "--overwrite" in result.output
        assert "Traceback" not in result.output

    def test_reports_provenance_file(self, runner, base_yaml, monkeypatch):
        pipeline = importlib.import_module("agribound.pipeline")
        from agribound.provenance import provenance_path

        def fake_delineate(study_area=None, config=None, **kwargs):
            record = provenance_path(config.output_path)
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text("{}")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        monkeypatch.setattr(pipeline, "delineate", fake_delineate)
        result = runner.invoke(main, ["delineate", "--config", str(base_yaml)])
        assert result.exit_code == 0, result.output
        assert "Provenance →" in result.output
        assert ".provenance.json" in result.output


class TestEngineParamParsing:
    def _params(self, runner, *pairs: str) -> dict:
        args = ["delineate", "--study-area", "a.geojson", "--gee-project", "p"]
        for pair in pairs:
            args += ["--engine-param", pair]
        return _dry_run(runner, args)["engine_params"]

    def test_json_values(self, runner):
        params = self._params(
            runner,
            "batch_size=8",
            "lr=1e-4",
            "use_lora=true",
            "checkpoint=null",
            'engines=["ftw", "delineate-anything"]',
            'window={"a": 1}',
        )
        assert params == {
            "batch_size": 8,
            "lr": 1e-4,
            "use_lora": True,
            "checkpoint": None,
            "engines": ["ftw", "delineate-anything"],
            "window": {"a": 1},
        }

    def test_string_fallback(self, runner):
        params = self._params(runner, "model=FTW_PRUE_EFNET_B5", "expr=a=b", "empty=")
        assert params == {"model": "FTW_PRUE_EFNET_B5", "expr": "a=b", "empty": ""}

    def test_repeated_key_last_wins(self, runner):
        assert self._params(runner, "conf=0.1", "conf=0.3") == {"conf": 0.3}

    @pytest.mark.parametrize("bad", ["novalue", "=5", " =x"])
    def test_invalid_pairs(self, runner, bad):
        result = runner.invoke(
            main,
            ["delineate", "--study-area", "a", "--gee-project", "p", "--engine-param", bad],
        )
        assert result.exit_code == 2
        assert "KEY=VALUE" in result.output

    def test_parser_unit(self):
        ctx = None
        parsed = cli_module._parse_engine_params(ctx, None, ("a=1", "b=x", "c=[1, 2]"))
        assert parsed == {"a": 1, "b": "x", "c": [1, 2]}


# ---------------------------------------------------------------------------
# composite / prefetch
# ---------------------------------------------------------------------------


class TestComposite:
    def test_dry_run(self, runner, base_yaml):
        data = _dry_run(runner, ["composite", "--config", str(base_yaml), "--year", "2019"])
        assert data["year"] == 2019
        assert data["study_area"] == "aoi.geojson"

    def test_calls_build_composite(self, runner, base_yaml, monkeypatch):
        pipeline = importlib.import_module("agribound.pipeline")
        seen = {}

        def fake_build(config):
            seen["config"] = config
            return "/tmp/composite.tif"

        monkeypatch.setattr(pipeline, "build_composite", fake_build, raising=False)
        result = runner.invoke(main, ["composite", "--config", str(base_yaml)])
        assert result.exit_code == 0, result.output
        assert seen["config"].year == 2020
        assert "/tmp/composite.tif" in result.output


class _FakeEngine:
    seen: list = []
    result: list = []

    @classmethod
    def prefetch(cls, config):
        cls.seen.append(config)
        return list(cls.result)


class TestPrefetch:
    @pytest.fixture
    def fake_engine(self, monkeypatch):
        engines = importlib.import_module("agribound.engines")
        _FakeEngine.seen = []
        _FakeEngine.result = []
        requested = []

        def fake_get_engine(name):
            requested.append(name)
            return _FakeEngine()

        monkeypatch.setattr(engines, "get_engine", fake_get_engine)
        return requested

    def test_calls_engine_prefetch(self, runner, fake_engine):
        _FakeEngine.result = ["/w/a.pt", "/w/b.pt"]
        result = runner.invoke(
            main,
            ["prefetch", "--engine", "ftw", "--engine-param", "model=FTW_PRUE_EFNET_B7"],
        )
        assert result.exit_code == 0, result.output
        assert fake_engine == ["ftw"]
        config = _FakeEngine.seen[0]
        assert config.engine_params == {"model": "FTW_PRUE_EFNET_B7"}
        assert config.source == "local"  # no GEE project needed for prefetch
        assert "/w/a.pt" in result.output and "Prefetched 2 file(s)" in result.output

    def test_embedding_engine_default_source(self, runner, fake_engine):
        result = runner.invoke(main, ["prefetch", "--engine", "embedding"])
        assert result.exit_code == 0, result.output
        # The embedding engine also accepts local feature rasters, and prefetch prefers
        # "local" (no GEE project; the placeholder raster is never read).
        assert _FakeEngine.seen[0].source == "local"
        assert "reported no files" in result.output

    def test_engine_from_config(self, runner, base_yaml, fake_engine):
        result = runner.invoke(main, ["prefetch", "--config", str(base_yaml)])
        assert result.exit_code == 0, result.output
        config = _FakeEngine.seen[0]
        assert config.engine == "delineate-anything"
        assert config.source == "sentinel2"
        assert config.engine_params["da_model"] == "DelineateAnything-S"

    def test_engine_required(self, runner, fake_engine):
        result = runner.invoke(main, ["prefetch"])
        assert result.exit_code == 2
        assert "--engine" in result.output

    @pytest.fixture
    def fake_sam(self, monkeypatch):
        samgeo_engine = importlib.import_module("agribound.engines.samgeo_engine")
        calls = []

        def fake_prefetch(config):
            calls.append((config.sam_backend, config.sam_model))
            return ["/hf/sam2.1_hiera_large.pt"]

        monkeypatch.setattr(samgeo_engine, "prefetch", fake_prefetch)
        return calls

    def test_sam_weights_prefetched_with_sam_refine(self, runner, fake_engine, fake_sam):
        _FakeEngine.result = ["/w/da.pt"]
        result = runner.invoke(
            main,
            [
                "prefetch",
                "--engine",
                "delineate-anything",
                "--sam-refine",
                "--sam-backend",
                "sam2.1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert fake_sam == [("sam2.1", None)]
        assert "/w/da.pt" in result.output and "/hf/sam2.1_hiera_large.pt" in result.output
        assert "Prefetched 1 file(s) for engine 'delineate-anything'" in result.output
        assert "Prefetched 1 file(s) for SAM refinement (sam_backend='sam2.1')" in result.output

    def test_sam_refine_from_config(self, runner, tmp_path, fake_engine, fake_sam):
        cfg = _write_yaml(
            tmp_path / "sam.yaml",
            {
                "source": "local",
                "local_tif_path": "x.tif",
                "engine": "ftw",
                "sam_refine": True,
                "sam_model": "facebook/sam2-hiera-small",
                "lulc_filter": False,
            },
        )
        result = runner.invoke(main, ["prefetch", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        assert fake_sam == [("sam2", "facebook/sam2-hiera-small")]
        assert "reported no files" in result.output  # the engine itself had none
        assert "for SAM refinement" in result.output

    def test_no_sam_prefetch_by_default_or_for_embedding(self, runner, fake_engine, fake_sam):
        assert runner.invoke(main, ["prefetch", "--engine", "ftw"]).exit_code == 0
        result = runner.invoke(main, ["prefetch", "--engine", "embedding", "--sam-refine"])
        assert result.exit_code == 0, result.output
        # The embedding engine refines inside the engine and prefetches SAM itself.
        assert fake_sam == []
        assert _FakeEngine.seen[-1].sam_refine is True


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@pytest.fixture
def vector_pair(tmp_path: Path) -> tuple[Path, Path]:
    # ~100 m squares near the equator
    d = 0.0009
    ref = gpd.GeoDataFrame(
        {"zone": ["a", "b"]},
        geometry=[box(0, 0, d, d), box(2 * d, 0, 3 * d, d)],
        crs="EPSG:4326",
    )
    pred = gpd.GeoDataFrame(geometry=[box(0, 0, d, d)], crs="EPSG:4326")
    ref_path = tmp_path / "ref.geojson"
    pred_path = tmp_path / "pred.geojson"
    ref.to_file(ref_path, driver="GeoJSON")
    pred.to_file(pred_path, driver="GeoJSON")
    return pred_path, ref_path


class TestEvaluate:
    def test_writes_metrics_json(self, runner, vector_pair, tmp_path):
        pred, ref = vector_pair
        out = tmp_path / "m" / "metrics.json"
        result = runner.invoke(
            main, ["evaluate", "--predicted", str(pred), "--reference", str(ref), "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        metrics = json.loads(out.read_text())
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(0.5)

    def test_prints_json_without_output(self, runner, vector_pair):
        pred, ref = vector_pair
        result = runner.invoke(main, ["evaluate", "-p", str(pred), "-r", str(ref)])
        assert result.exit_code == 0, result.output
        assert "recall" in json.loads(result.stdout)

    def test_forwards_options(self, runner, vector_pair, monkeypatch):
        pred, ref = vector_pair
        evaluate_mod = importlib.import_module("agribound.evaluate")
        seen = {}

        def fake_evaluate(predicted, reference, **kwargs):
            seen.update(kwargs)
            return {"f1": 0.5, "nan_metric": float("nan")}

        monkeypatch.setattr(evaluate_mod, "evaluate", fake_evaluate)
        result = runner.invoke(
            main,
            [
                "evaluate",
                "-p",
                str(pred),
                "-r",
                str(ref),
                "--iou-threshold",
                "0.7",
                "--strata-column",
                "zone",
                "--size-bins",
                "0,1,5,10",
                "--bootstrap",
                "100",
                "--bootstrap-seed",
                "1",
                "--boundary-tolerance-m",
                "10",
                "--equal-area-crs",
                "EPSG:6933",
                "--matching",
                "many_to_one",
                "--boundary-sample-spacing-m",
                "2.5",
            ],
        )
        assert result.exit_code == 0, result.output
        assert seen == {
            "iou_threshold": 0.7,
            "strata": "zone",
            "size_bins": [0.0, 1.0, 5.0, 10.0],
            "matching": "many_to_one",
            "bootstrap": 100,
            "bootstrap_seed": 1,
            "boundary_tolerance_m": 10.0,
            "boundary_sample_spacing_m": 2.5,
            "equal_area_crs": "EPSG:6933",
        }
        assert json.loads(result.stdout) == {"f1": 0.5, "nan_metric": None}

    def test_defaults_pass_only_iou(self, runner, vector_pair, monkeypatch):
        pred, ref = vector_pair
        evaluate_mod = importlib.import_module("agribound.evaluate")
        seen = {}

        def fake_evaluate(predicted, reference, **kwargs):
            seen.update(kwargs)
            return {}

        monkeypatch.setattr(evaluate_mod, "evaluate", fake_evaluate)
        result = runner.invoke(main, ["evaluate", "-p", str(pred), "-r", str(ref)])
        assert result.exit_code == 0, result.output
        assert seen == {"iou_threshold": 0.5}

    def test_unknown_strata_column(self, runner, vector_pair):
        pred, ref = vector_pair
        result = runner.invoke(
            main, ["evaluate", "-p", str(pred), "-r", str(ref), "--strata-column", "nope"]
        )
        assert result.exit_code == 2
        assert "nope" in result.output

    @pytest.mark.parametrize("bins", ["5", "1,x", "5,1", "1,1", "-1,2", "nan,1"])
    def test_bad_size_bins(self, runner, vector_pair, bins):
        pred, ref = vector_pair
        result = runner.invoke(
            main, ["evaluate", "-p", str(pred), "-r", str(ref), "--size-bins", bins]
        )
        assert result.exit_code == 2

    def test_auto_size_bins_inf_edge_and_spacing_none(self, runner, vector_pair, monkeypatch):
        pred, ref = vector_pair
        evaluate_mod = importlib.import_module("agribound.evaluate")
        seen = []

        def fake_evaluate(predicted, reference, **kwargs):
            seen.append(kwargs)
            return {}

        monkeypatch.setattr(evaluate_mod, "evaluate", fake_evaluate)
        base = ["evaluate", "-p", str(pred), "-r", str(ref)]
        assert runner.invoke(main, [*base, "--size-bins", "AUTO"]).exit_code == 0
        assert runner.invoke(main, [*base, "--size-bins", "0,0.5,inf"]).exit_code == 0
        result = runner.invoke(main, [*base, "--boundary-sample-spacing-m", "none"])
        assert result.exit_code == 0, result.output
        assert seen[0]["size_bins"] == "auto"
        assert seen[1]["size_bins"] == [0.0, 0.5, float("inf")]
        assert seen[2] == {"iou_threshold": 0.5, "boundary_sample_spacing_m": None}

    def test_size_bins_help_says_hectares(self, runner):
        text = _flat(runner.invoke(main, ["evaluate", "--help"]).output)
        assert "in hectares" in text and "'auto'" in text

    @pytest.mark.parametrize("spacing", ["0", "-1", "x", "inf"])
    def test_bad_boundary_spacing(self, runner, vector_pair, spacing):
        pred, ref = vector_pair
        result = runner.invoke(
            main,
            ["evaluate", "-p", str(pred), "-r", str(ref), "--boundary-sample-spacing-m", spacing],
        )
        assert result.exit_code == 2

    def test_baselines_flag_is_forwarded(self, runner, vector_pair, monkeypatch):
        pred, ref = vector_pair
        evaluate_mod = importlib.import_module("agribound.evaluate")
        seen = []

        def fake_evaluate(predicted, reference, **kwargs):
            seen.append(kwargs)
            return {}

        monkeypatch.setattr(evaluate_mod, "evaluate", fake_evaluate)
        base = ["evaluate", "-p", str(pred), "-r", str(ref)]
        assert runner.invoke(main, [*base, "--baselines"]).exit_code == 0
        assert runner.invoke(main, base).exit_code == 0
        assert seen == [{"iou_threshold": 0.5, "baselines": True}, {"iou_threshold": 0.5}]

    def test_real_evaluate_with_baselines(self, runner, vector_pair):
        pred, ref = vector_pair
        result = runner.invoke(
            main,
            [
                "evaluate",
                "-p",
                str(pred),
                "-r",
                str(ref),
                "--baselines",
                "--boundary-tolerance-m",
                "10",
            ],
        )
        assert result.exit_code == 0, result.output
        metrics = json.loads(result.stdout)
        grid = metrics["baselines"]["blind_grid"]
        assert grid["n_grids"] == 36 and "boundary_f1" in grid
        assert "shifted_reference" not in metrics["baselines"]
        assert metrics["recall"] == pytest.approx(0.5)  # the evaluation itself is unchanged

    def test_real_evaluate_with_new_options(self, runner, vector_pair):
        pred, ref = vector_pair
        result = runner.invoke(
            main,
            [
                "evaluate",
                "-p",
                str(pred),
                "-r",
                str(ref),
                "--size-bins",
                "auto",
                "--matching",
                "many_to_one",
                "--boundary-sample-spacing-m",
                "none",
            ],
        )
        assert result.exit_code == 0, result.output
        metrics = json.loads(result.stdout)
        assert metrics["recall"] == pytest.approx(0.5)
        assert metrics["matching"] == "many_to_one"


# ---------------------------------------------------------------------------
# query-ftw
# ---------------------------------------------------------------------------


class TestQueryFtw:
    @pytest.fixture
    def captured(self, monkeypatch):
        ftw_query = importlib.import_module("agribound.ftw_query")
        seen = {}

        def fake_query(**kwargs):
            seen.update(kwargs)
            return gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")

        monkeypatch.setattr(ftw_query, "query_ftw", fake_query)
        return seen

    def test_new_flags_forwarded(self, runner, captured):
        result = runner.invoke(
            main,
            [
                "query-ftw",
                "--study-area",
                "aoi.geojson",
                "--min-confidence",
                "0.69",
                "--drop-null-confidence",
                "--layout",
                "raw",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["min_confidence"] == pytest.approx(0.69)
        assert captured["keep_null_confidence"] is False
        assert captured["layout"] == "raw"
        assert "Queried 1 published FTW polygons" in result.output

    def test_defaults_leave_query_ftw_defaults(self, runner, captured):
        result = runner.invoke(main, ["query-ftw", "--study-area", "aoi.geojson"])
        assert result.exit_code == 0, result.output
        for key in ("min_confidence", "keep_null_confidence", "layout"):
            assert key not in captured, key
        assert captured["label"] == "field"
        assert captured["columns"] is None

    def test_explicit_default_values_forwarded(self, runner, captured):
        result = runner.invoke(
            main,
            [
                "query-ftw",
                "--study-area",
                "aoi.geojson",
                "--keep-null-confidence",
                "--layout",
                "by-admin-conf",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["keep_null_confidence"] is True
        assert captured["layout"] == "by-admin-conf"
        assert "min_confidence" not in captured

    def test_help_defaults_match_query_ftw_contract(self, runner):
        text = _flat(runner.invoke(main, ["query-ftw", "--help"]).output)
        assert "[default: by-admin-conf]" in text
        assert "[default: keep-null-confidence]" in text

    def test_existing_flags_forwarded(self, runner, captured):
        result = runner.invoke(
            main,
            [
                "query-ftw",
                "--study-area",
                "aoi.geojson",
                "--year",
                "2024",
                "--no-clip",
                "--columns",
                "id",
                "--columns",
                "confidence",
                "--max-features",
                "10",
                "--dst-crs",
                "EPSG:5070",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["year"] == 2024
        assert captured["clip"] is False
        assert captured["columns"] == ["id", "confidence"]
        assert captured["max_features"] == 10
        assert captured["dst_crs"] == "EPSG:5070"


# ---------------------------------------------------------------------------
# Optional command registration
# ---------------------------------------------------------------------------


class TestOptionalCommands:
    def _group(self):
        import click

        @click.group()
        def grp():
            pass

        return grp

    def test_missing_module_is_skipped(self):
        grp = self._group()
        names = cli_module._register_optional_commands(
            grp,
            (
                ("agribound.definitely_missing_pkg.cli", "tiles"),
                ("agribound.cli_missing_module_xyz", "agent"),
            ),
        )
        assert names == []
        assert grp.commands == {}

    def test_existing_module_is_registered(self, monkeypatch):
        import click

        @click.command("tiles")
        def tiles():
            """Fake tiles command."""

        fake = types.ModuleType("agribound_fake_optional_cli")
        fake.tiles = tiles
        monkeypatch.setitem(sys.modules, "agribound_fake_optional_cli", fake)
        grp = self._group()
        names = cli_module._register_optional_commands(
            grp, (("agribound_fake_optional_cli", "tiles"),)
        )
        assert names == ["tiles"]
        assert "tiles" in grp.commands

    def test_missing_dependency_inside_module_is_reraised(self, tmp_path, monkeypatch):
        (tmp_path / "agribound_fake_broken_cli.py").write_text(
            "import agribound_nonexistent_dependency_xyz\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        grp = self._group()
        with pytest.raises(ModuleNotFoundError) as excinfo:
            cli_module._register_optional_commands(grp, (("agribound_fake_broken_cli", "agent"),))
        assert excinfo.value.name == "agribound_nonexistent_dependency_xyz"

    def test_is_missing_module(self):
        err = ModuleNotFoundError("x", name="agribound.hpc")
        assert cli_module._is_missing_module(err, "agribound.hpc.cli")
        err = ModuleNotFoundError("x", name="agribound.hpc.cli")
        assert cli_module._is_missing_module(err, "agribound.hpc.cli")
        err = ModuleNotFoundError("x", name="anthropic")
        assert not cli_module._is_missing_module(err, "agribound.agent.cli")


# ---------------------------------------------------------------------------
# Fine-tuning dispatcher (agribound.engines.finetune)
# ---------------------------------------------------------------------------


class TestFineTuneDispatcher:
    """Routing, seeding, error and caching behaviour of ``fine_tune``."""

    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        tif = tmp_path / "composite.tif"
        tif.write_bytes(b"not a real raster")  # never read: data prep is mocked
        ref = tmp_path / "ref.geojson"
        ref.write_text('{"type": "FeatureCollection", "features": []}')

        seeds: list[int] = []
        fake_repro = types.ModuleType("agribound._repro")
        fake_repro.seed_everything = lambda seed, deterministic=False: seeds.append(seed)
        monkeypatch.setitem(sys.modules, "agribound._repro", fake_repro)

        data_mod = importlib.import_module("agribound.engines.finetune._data")
        prepared: list[Path] = []

        def fake_prepare(raster_path, config, engine):
            train_dir = config.get_working_dir() / f"chips_{engine}"
            train_dir.mkdir(parents=True, exist_ok=True)
            prepared.append(config.get_working_dir())
            return train_dir

        monkeypatch.setattr(data_mod, "_prepare_training_data", fake_prepare)

        prithvi_mod = importlib.import_module("agribound.engines.finetune._prithvi")
        trained: list[Path] = []

        def fake_prithvi(train_dir, config):
            ckpt = config.get_working_dir() / "checkpoints" / "prithvi" / "best.ckpt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_bytes(b"weights")
            trained.append(config.get_working_dir())
            return str(ckpt)

        monkeypatch.setattr(prithvi_mod, "_finetune_prithvi", fake_prithvi)

        def make_config(**overrides):
            from agribound.config import AgriboundConfig

            values = {
                "source": "local",
                "local_tif_path": str(tif),
                "engine": "prithvi",
                "output_path": str(tmp_path / "out" / "fields.gpkg"),
                "reference_boundaries": str(ref),
                "lulc_filter": False,
                "seed": 123,
            }
            values.update(overrides)
            return AgriboundConfig(**values)

        return types.SimpleNamespace(
            tif=str(tif),
            ref=ref,
            seeds=seeds,
            prepared=prepared,
            trained=trained,
            make_config=make_config,
            prithvi_mod=prithvi_mod,
        )

    def test_import_path_kept(self):
        from agribound.engines.finetune import fine_tune

        assert callable(fine_tune)

    @pytest.mark.parametrize(
        ("engine", "extra", "needle"),
        [
            ("ftw", {}, "ftw model fit"),
            ("embedding", {"source": "tessera-embedding", "local_tif_path": None}, "no trainable"),
            ("ensemble", {}, "checkpoint_path"),
        ],
    )
    def test_non_tunable_engines_raise(self, env, engine, extra, needle):
        from agribound.engines.finetune import fine_tune

        config = env.make_config(engine=engine, **extra)
        with pytest.raises(ValueError, match="cannot be fine-tuned") as excinfo:
            fine_tune(env.tif, config)
        assert needle in str(excinfo.value)
        if engine == "ftw":
            assert "checkpoint_path" in str(excinfo.value)
        assert env.trained == [] and env.prepared == []

    def test_prithvi_routes_to_trainer_and_seeds(self, env):
        from agribound.engines.finetune import fine_tune

        config = env.make_config()
        ckpt = fine_tune(env.tif, config)
        assert Path(ckpt).is_file() and ckpt.endswith("best.ckpt")
        assert env.seeds == [123]
        # trainers see a keyed run directory inside the working dir, not the shared cache
        run_dir = env.trained[0]
        assert run_dir.parent == config.get_working_dir()
        assert run_dir.name.startswith("finetune_prithvi_")
        assert env.prepared == [run_dir]
        manifest = json.loads((run_dir / "finetune_manifest.json").read_text())
        assert manifest["engine"] == "prithvi"
        assert manifest["seed"] == 123
        assert not Path(manifest["checkpoint"]).is_absolute()

    def test_cached_checkpoint_reused(self, env):
        from agribound.engines.finetune import fine_tune

        first = fine_tune(env.tif, env.make_config())
        second = fine_tune(env.tif, env.make_config())
        assert first == second
        assert len(env.trained) == 1

    @pytest.mark.parametrize(
        "overrides",
        [
            {"fine_tune_epochs": 3},
            {"seed": 7},
            {"fine_tune_split": "random"},
            {"engine_params": {"model_name": "Prithvi-EO-2.0-600M-TL"}},
            {"year": 2020},
        ],
    )
    def test_cache_key_parts(self, env, overrides):
        from agribound.engines.finetune import fine_tune

        fine_tune(env.tif, env.make_config())
        fine_tune(env.tif, env.make_config(**overrides))
        assert len(env.trained) == 2
        assert env.trained[0] != env.trained[1]

    def test_inference_only_params_do_not_retrain(self, env):
        from agribound.engines.finetune import fine_tune

        fine_tune(env.tif, env.make_config())
        fine_tune(
            env.tif,
            env.make_config(engine_params={"checkpoint_path": "x.ckpt", "sam_refine": True}),
        )
        assert len(env.trained) == 1

    def test_reference_change_retrains(self, env):
        from agribound.engines.finetune import fine_tune

        fine_tune(env.tif, env.make_config())
        env.ref.write_text('{"type": "FeatureCollection", "features": [], "name": "v2"}')
        fine_tune(env.tif, env.make_config())
        assert len(env.trained) == 2

    def test_missing_checkpoint_retrains(self, env):
        from agribound.engines.finetune import fine_tune

        ckpt = fine_tune(env.tif, env.make_config())
        Path(ckpt).unlink()
        fine_tune(env.tif, env.make_config())
        assert len(env.trained) == 2

    def test_non_checkpoint_return_is_rejected(self, env, monkeypatch):
        from agribound.engines.finetune import fine_tune

        def returns_yaml(train_dir, config):
            path = config.get_working_dir() / "finetune_config.yaml"
            path.write_text("model: {}\n")
            return str(path)

        monkeypatch.setattr(env.prithvi_mod, "_finetune_prithvi", returns_yaml)
        with pytest.raises(RuntimeError, match="not an existing checkpoint"):
            fine_tune(env.tif, env.make_config())

    def test_reference_required(self, env):
        from agribound.engines.finetune import fine_tune

        with pytest.raises(ValueError, match="reference_boundaries"):
            fine_tune(env.tif, env.make_config(reference_boundaries=None))


# ---------------------------------------------------------------------------
# Open VHR SAR flags (umbra, capella)
# ---------------------------------------------------------------------------

VHR_SAR_FLAGS = (
    "--vhr-sar-collect-ids",
    "--vhr-sar-max-collects",
    "--vhr-sar-polarization",
    "--vhr-sar-resolution",
    "--vhr-sar-coregister",
    "--vhr-sar-coregister-base",
    "--vhr-sar-bands",
    "--vhr-sar-calibration",
    "--vhr-sar-max-download-gb",
    "--vhr-sar-cache-dir",
)


class TestCliVhrSar:
    def test_flags_are_stage_a_options(self, runner):
        for command in ("delineate", "composite"):
            text = runner.invoke(main, [command, "--help"]).output
            for flag in VHR_SAR_FLAGS:
                assert flag in text, (command, flag)

    def test_flags_reach_the_configuration(self, runner, tmp_path):
        args = [
            "delineate",
            "--study-area",
            "bbox:-121.8,38.5,-121.79,38.51",
            "--source",
            "umbra",
            "--year",
            "2024",
            "--no-lulc-filter",
            "--output",
            str(tmp_path / "x.gpkg"),
        ]
        data = _dry_run(runner, args)
        assert data["vhr_sar_collect_ids"] is None and data["vhr_sar_max_collects"] == 6
        assert data["vhr_sar_coregister"] == "auto" and data["bands"] is None
        data = _dry_run(
            runner,
            [
                *args,
                "--vhr-sar-collect-ids",
                "a, b",
                "--vhr-sar-polarization",
                "hh",
                "--vhr-sar-resolution",
                "2",
                "--vhr-sar-coregister",
                "sar",
                "--vhr-sar-bands",
                "median+dates",
                "--vhr-sar-max-download-gb",
                "1.5",
                "--vhr-sar-cache-dir",
                str(tmp_path / "vhr"),
            ],
        )
        assert data["vhr_sar_collect_ids"] == ["a", "b"]
        assert data["vhr_sar_polarization"] == "HH" and data["vhr_sar_resolution_m"] == 2.0
        assert data["vhr_sar_coregister"] == "sar" and data["vhr_sar_bands"] == "median+dates"
        # The median+dates band mapping is derived when the composite is read, not stored.
        assert data["bands"] is None
        assert data["vhr_sar_max_download_gb"] == 1.5
        assert data["vhr_sar_cache_dir"] == str(tmp_path / "vhr")

    def test_invalid_combinations_are_usage_errors(self, runner):
        args = ["delineate", "--study-area", "a.geojson", "--source", "umbra", "--year", "2024"]
        result = runner.invoke(main, [*args, "--vhr-sar-calibration", "full", "--dry-run"])
        assert result.exit_code == 2 and "applies to source='capella' only" in _flat(result.output)
        result = runner.invoke(main, [*args, "--engine", "ftw", "--dry-run"])
        assert result.exit_code == 2 and "does not support source 'umbra'" in _flat(result.output)
        single = ["--vhr-sar-bands", "median+dates", "--vhr-sar-max-collects", "1", "--dry-run"]
        result = runner.invoke(main, [*args, *single])
        assert result.exit_code == 2 and "needs at least 2 collects" in _flat(result.output)

    def test_list_sources_shows_umbra_and_capella(self, runner):
        lines = runner.invoke(main, ["list-sources"]).output.splitlines()
        rows = [line.split()[0] for line in lines if line.startswith("  ")]
        assert rows.index("umbra") == rows.index("sentinel1") + 1
        assert rows.index("capella") == rows.index("umbra") + 1
        (umbra,) = [line for line in lines if line.strip().startswith("umbra")]
        assert "1.0m" in umbra and "2023-present" in umbra and "[no GEE]" in umbra


# ---------------------------------------------------------------------------
# scenes and composite --date-layers
# ---------------------------------------------------------------------------


def _survey_table():
    """A ranked survey table as ``agribound.survey_scenes`` returns it (abridged)."""
    import pandas as pd

    rows = [
        {
            "rank": 1,
            "date": "2020-08-04",
            "acquisition": "Sentinel-2B_2020-08-04_R065",
            "n_images": 1,
            "image_ids": ["COPERNICUS/S2_SR_HARMONIZED/20200804T101559_20200804T103905_T30NXM"],
            "scene_cloud_pct": 10.626962,
            "extent_share": 1.0,
            "clear_share": 1.0,
            "cs_clear_share": 1.0,
            "scl_clear_share": 1.0,
            "cs_mean": 0.903,
            "aot_mean": 0.0556,
            "date_range": ("2020-08-04", "2020-08-04"),
        },
        {
            "rank": 2,
            "date": "2020-01-12",
            "acquisition": "Sentinel-2A_2020-01-12_R065",
            "n_images": 1,
            "image_ids": ["COPERNICUS/S2_SR_HARMONIZED/20200112T102401_20200112T103856_T30NXM"],
            "scene_cloud_pct": 3.08427,
            "extent_share": 1.0,
            "clear_share": 1.0,
            "cs_clear_share": 1.0,
            "scl_clear_share": 0.9838,
            "cs_mean": 0.851,
            "aot_mean": None,
            "date_range": ("2020-01-12", "2020-01-12"),
        },
    ]
    table = pd.DataFrame(rows)
    table.attrs.update(
        {
            "source": "sentinel2",
            "year": 2020,
            "window": ("2020-01-01", "2021-01-01"),
            "cloud_cover_max": 80,
            "cloud_mask": "Cloud Score+ cs_cdf >= 0.6",
            "ranked_by": "share of the study area clear under the configured mask",
            "rank_rule": "clear_share desc, cs_mean desc, date asc, acquisition asc",
            "survey_version": "1",
            "scale_m": 60.0,
            "crs": "EPSG:32630",
            "reduced_over": "the study area",
            "n_acquisitions": 3,
            "n_without_coverage": 1,
        }
    )
    return table


class TestScenes:
    @pytest.fixture
    def fake_survey(self, monkeypatch):
        scenes = importlib.import_module("agribound.composites.scenes")
        seen = []

        def fake(config, *, scale_m=None):
            seen.append({"config": config, "scale_m": scale_m})
            return _survey_table()

        monkeypatch.setattr(scenes, "survey_scenes", fake)
        return seen

    def test_help_lists_the_options(self, runner):
        text = _flat(runner.invoke(main, ["scenes", "--help"]).output)
        for flag in ("--top", "--json", "--csv", "--scale-m", "--study-area", "--date-range"):
            assert flag in text, flag
        assert "[default: 10; x>=0]" in text
        # click may wrap at the hyphens of "Delineate-Anything" and "Sentinel-2"
        assert "the single clearest Sentinel-2 date was a worse input" in text.replace("- ", "-")
        assert "--engine-param" not in text and "--date-layers" not in text

    def test_dry_run(self, runner, base_yaml, fake_survey):
        data = _dry_run(runner, ["scenes", "--config", str(base_yaml), "--cloud-cover-max", "80"])
        assert data["cloud_cover_max"] == 80 and data["source"] == "sentinel2"
        assert fake_survey == []

    @pytest.mark.parametrize("source", ["naip", "sentinel1"])
    def test_unsupported_source_is_usage_error(self, runner, base_yaml, fake_survey, source):
        args = ["scenes", "--config", str(base_yaml), "--source", source, "--dry-run"]
        result = runner.invoke(main, args)
        assert result.exit_code == 2, result.output
        assert "The scene survey supports the sources sentinel2" in _flat(result.output)
        assert fake_survey == []

    def test_table(self, runner, base_yaml, fake_survey):
        args = ["scenes", "--config", str(base_yaml), "--s2-cloud-mask", "cloud_score_plus"]
        result = runner.invoke(main, [*args, "--scale-m", "120"])
        assert result.exit_code == 0, result.output
        (call,) = fake_survey
        assert call["scale_m"] == 120.0
        assert call["config"].s2_cloud_mask == "cloud_score_plus"
        out = result.output
        assert "sentinel2 2020: 2 acquisition(s) over the study area" in out
        assert "1 date(s) whose images miss the study area are not listed." in out
        header = next(line for line in out.splitlines() if line.startswith("rank"))
        for name in ("date", "acquisition", "images", "cloud %", "footprint", "clear", "AOT"):
            assert name in header
        row = next(line for line in out.splitlines() if "Sentinel-2B_2020-08-04_R065" in line)
        assert row.split()[:3] == ["1", "2020-08-04", "Sentinel-2B_2020-08-04_R065"]
        assert "10.6" in row and "0.903" in row and "0.056" in row
        row2 = next(line for line in out.splitlines() if "Sentinel-2A_2020-01-12" in line)
        assert row2.rstrip().endswith("-")  # no AOT: no clear pixel
        assert "Rank 1 alone: --date-range 2020-08-04 2020-08-04" in out

    def test_top_json_and_csv(self, runner, base_yaml, fake_survey, tmp_path):
        csv_path = tmp_path / "survey" / "scenes.csv"
        args = ["scenes", "--config", str(base_yaml), "--top", "1", "--json"]
        result = runner.invoke(main, [*args, "--csv", str(csv_path)])
        assert result.exit_code == 0, result.output
        data = json.loads(result.stdout)
        assert data["n_rows"] == 2 and len(data["rows"]) == 1
        assert data["rows"][0]["date_range"] == ["2020-08-04", "2020-08-04"]
        assert data["window"] == ["2020-01-01", "2021-01-01"]
        assert data["rank_rule"].startswith("clear_share desc")
        import pandas as pd

        written = pd.read_csv(csv_path)
        assert written["date"].tolist() == ["2020-08-04", "2020-01-12"]
        assert "date_range" not in written.columns
        assert written.loc[0, "image_ids"].startswith("COPERNICUS/S2_SR_HARMONIZED/")
        table = runner.invoke(main, ["scenes", "--config", str(base_yaml), "--top", "1"]).output
        assert "1 of 2 rows (--top 0 prints all)." in table

    def test_survey_errors_are_clean(self, runner, base_yaml, monkeypatch):
        scenes = importlib.import_module("agribound.composites.scenes")

        def failing(config, *, scale_m=None):
            raise RuntimeError("The sentinel2 2020 scene survey failed on Earth Engine")

        monkeypatch.setattr(scenes, "survey_scenes", failing)
        result = runner.invoke(main, ["scenes", "--config", str(base_yaml)])
        assert result.exit_code == 1
        assert "scene survey failed on Earth Engine" in result.output


class TestCompositeDateLayers:
    @pytest.fixture
    def fake_stage_a(self, monkeypatch):
        pipeline = importlib.import_module("agribound.pipeline")
        scenes = importlib.import_module("agribound.composites.scenes")
        calls = []
        monkeypatch.setattr(
            pipeline, "build_composite", lambda config: calls.append("composite") or "/c.tif"
        )
        monkeypatch.setattr(
            scenes, "build_date_layers", lambda config: calls.append("dates") or "/d.tif"
        )
        return calls

    def test_help(self, runner):
        text = _flat(runner.invoke(main, ["composite", "--help"]).output)
        assert "--date-layers" in text and "days since 1970-01-01" in text
        assert "--date-layers" not in runner.invoke(main, ["delineate", "--help"]).output

    def test_dry_run(self, runner, base_yaml, fake_stage_a):
        data = _dry_run(runner, ["composite", "--config", str(base_yaml), "--date-layers"])
        assert data["source"] == "sentinel2"
        assert fake_stage_a == []

    def test_builds_the_composite_then_the_layers(self, runner, base_yaml, fake_stage_a):
        result = runner.invoke(main, ["composite", "--config", str(base_yaml), "--date-layers"])
        assert result.exit_code == 0, result.output
        assert fake_stage_a == ["composite", "dates"]
        assert "Composite ready → /c.tif" in result.output
        assert "Date layers ready → /d.tif" in result.output

    def test_without_the_flag_no_layers(self, runner, base_yaml, fake_stage_a):
        result = runner.invoke(main, ["composite", "--config", str(base_yaml)])
        assert result.exit_code == 0 and fake_stage_a == ["composite"]

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            (["--source", "naip"], "Date layers are available for the sources sentinel2"),
            (["--export-method", "gcs", "--gcs-bucket", "b"], "need export_method='local'"),
        ],
    )
    def test_refused_before_the_composite(self, runner, base_yaml, fake_stage_a, extra, message):
        args = ["composite", "--config", str(base_yaml), "--date-layers", *extra]
        result = runner.invoke(main, args)
        assert result.exit_code == 2, result.output
        assert message in _flat(result.output)
        assert fake_stage_a == []
