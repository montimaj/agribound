"""Tests for agribound.config.AgriboundConfig."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import yaml

from agribound.config import (
    VALID_ENGINES,
    VALID_SOURCES,
    AgriboundConfig,
)
from agribound.registry import ENGINE_REGISTRY, SOURCE_REGISTRY


def _local(**kwargs):
    kwargs.setdefault("source", "local")
    kwargs.setdefault("local_tif_path", "/tmp/x.tif")
    return AgriboundConfig(**kwargs)


#: Environment variables that can supply a GEE project to config validation.
_PROJECT_ENV = (
    "GEE_PROJECT",
    "AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


def _env_without_project(**extra):
    env = {k: v for k, v in os.environ.items() if k not in _PROJECT_ENV}
    env.update(extra)
    return env


def _write_key(tmp_path, name="key.json", **content):
    import json

    path = tmp_path / name
    path.write_text(json.dumps(content))
    return str(path)


class TestGeeProjectFromCredentials:
    """gee_project falls back to the project_id of the credentials file."""

    def _config(self, env, **kwargs):
        with (
            patch.dict("os.environ", env, clear=True),
            patch("agribound.auth._get_gcloud_project", return_value=None),
        ):
            return AgriboundConfig(source="sentinel2", **kwargs)

    def test_explicit_service_account_key(self, tmp_path):
        key = _write_key(tmp_path, type="service_account", project_id="key-project")
        cfg = self._config(_env_without_project(), gee_service_account_key=key)
        assert cfg.gee_project == "key-project"

    def test_env_service_account_key(self, tmp_path):
        key = _write_key(tmp_path, type="service_account", project_id="env-key-project")
        env = _env_without_project(AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY=key)
        assert self._config(env).gee_project == "env-key-project"

    def test_google_application_credentials(self, tmp_path):
        key = _write_key(tmp_path, type="service_account", project_id="adc-key-project")
        env = _env_without_project(GOOGLE_APPLICATION_CREDENTIALS=key)
        assert self._config(env).gee_project == "adc-key-project"

    def test_first_set_credentials_file_is_read(self, tmp_path):
        explicit = _write_key(tmp_path, "a.json", project_id="explicit")
        env_key = _write_key(tmp_path, "b.json", project_id="env-key")
        adc = _write_key(tmp_path, "c.json", project_id="adc")
        env = _env_without_project(
            AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY=env_key, GOOGLE_APPLICATION_CREDENTIALS=adc
        )
        assert self._config(env, gee_service_account_key=explicit).gee_project == "explicit"
        assert self._config(env).gee_project == "env-key"

    def test_env_project_and_gcloud_take_precedence(self, tmp_path):
        key = _write_key(tmp_path, project_id="key-project")
        env = _env_without_project(GEE_PROJECT="env-project")
        assert self._config(env, gee_service_account_key=key).gee_project == "env-project"
        with (
            patch.dict("os.environ", _env_without_project(), clear=True),
            patch("agribound.auth._get_gcloud_project", return_value="gcloud-project"),
        ):
            cfg = AgriboundConfig(source="sentinel2", gee_service_account_key=key)
        assert cfg.gee_project == "gcloud-project"

    @pytest.mark.parametrize(
        "content", [{"type": "authorized_user", "quota_project_id": "q"}, None]
    )
    def test_file_without_project_id_still_raises(self, tmp_path, content):
        key = str(tmp_path / "missing.json") if content is None else _write_key(tmp_path, **content)
        env = _env_without_project(GOOGLE_APPLICATION_CREDENTIALS=key)
        with pytest.raises(ValueError, match="project_id of the service-account key"):
            self._config(env)

    def test_explicit_project_does_not_read_the_key(self, tmp_path):
        key = _write_key(tmp_path, project_id="key-project")
        cfg = self._config(_env_without_project(), gee_project="mine", gee_service_account_key=key)
        assert cfg.gee_project == "mine"


class TestAgriboundConfigDefaults:
    """Test default configuration values."""

    def test_default_source(self):
        cfg = AgriboundConfig(source="local", local_tif_path="/tmp/test.tif")
        assert cfg.engine == "delineate-anything"
        assert cfg.year == 2024
        assert cfg.output_format == "gpkg"
        assert cfg.export_method == "local"
        assert cfg.composite_method == "median"
        assert cfg.cloud_cover_max == 20
        assert cfg.min_field_area_m2 == 2500.0
        assert cfg.simplify_tolerance == 2.0
        assert cfg.device == "auto"
        assert cfg.tile_size == 10_000
        assert cfg.n_workers == 4
        assert cfg.fine_tune is False
        assert cfg.fine_tune_epochs == 20
        assert cfg.fine_tune_val_split == 0.2
        assert cfg.engine_params == {}

    def test_1_0_defaults(self):
        cfg = _local()
        assert cfg.seed == 42
        assert cfg.export_crs == "utm"
        assert cfg.s2_cloud_mask == "scl"
        assert cfg.cloud_score_threshold == 0.60
        assert cfg.naip_resolution_m == 1.0
        assert cfg.tessera_version == "v1"
        assert cfg.tessera_variant is None
        assert cfg.embedding_cache_dir is None
        assert cfg.google_embedding_backend == "gee"
        assert cfg.aoi_selection == "representative_point"
        assert cfg.lulc_filter is True
        assert cfg.lulc_crop_threshold == 0.3
        assert cfg.lulc_batch_size == 200
        assert cfg.lulc_dataset == "auto"
        assert cfg.lulc_on_error == "raise"
        assert cfg.lulc_mode == "server"
        assert cfg.lulc_nodata_policy == "keep"
        assert cfg.sam_refine is False
        assert cfg.sam_backend == "sam2"
        assert cfg.sam_model is None
        assert cfg.sam_min_crop_px == 64
        assert cfg.sam_crop_padding == 0.15
        assert cfg.fine_tune_split == "block"
        assert cfg.fine_tune_block_size_m == 5000.0
        assert cfg.fine_tune_split_column is None
        assert cfg.gee_service_account_key is None
        assert cfg.gee_high_volume is False
        assert cfg.gee_max_requests == 8
        assert cfg.gee_workload_tag is None
        assert cfg.cache_dir is None
        assert cfg.overwrite is False
        assert cfg.provenance is True

    def test_local_source_requires_tif(self):
        cfg = AgriboundConfig(source="local", local_tif_path="/data/img.tif")
        assert cfg.source == "local"
        assert cfg.local_tif_path == "/data/img.tif"

    def test_valid_choices_derive_from_registry(self):
        assert tuple(SOURCE_REGISTRY) == VALID_SOURCES
        assert tuple(ENGINE_REGISTRY) == VALID_ENGINES


class TestAgriboundConfigValidation:
    """Test configuration validation."""

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="Invalid source"):
            AgriboundConfig(source="invalid_source")

    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError, match="Invalid engine"):
            _local(engine="nonexistent")

    def test_invalid_output_format_raises(self):
        with pytest.raises(ValueError, match="Invalid output_format"):
            _local(output_format="csv")

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="Invalid device"):
            _local(device="tpu")

    @pytest.mark.parametrize(
        "field,value",
        [
            ("s2_cloud_mask", "qa60"),
            ("tessera_version", "v3"),
            ("lulc_dataset", "worldcover"),
            ("lulc_on_error", "ignore"),
            ("lulc_mode", "local"),
            ("lulc_nodata_policy", "zero"),
            ("sam_backend", "sam1"),
            ("fine_tune_split", "kfold"),
            ("google_embedding_backend", "gcs"),
            ("composite_method", "mean"),
            ("export_method", "s3"),
            ("aoi_selection", "bbox"),
        ],
    )
    def test_invalid_enum_raises(self, field, value):
        with pytest.raises(ValueError, match=f"Invalid {field}"):
            _local(**{field: value})

    def test_enums_are_normalised(self):
        cfg = _local(lulc_mode=" RASTER ", sam_backend="SAM2.1", s2_cloud_mask="Cloud_Score_Plus")
        assert cfg.lulc_mode == "raster"
        assert cfg.sam_backend == "sam2.1"
        assert cfg.s2_cloud_mask == "cloud_score_plus"

    def test_aoi_selection_choices(self):
        from agribound.config import VALID_AOI_SELECTIONS

        assert VALID_AOI_SELECTIONS == ("representative_point", "intersects", "clip", "none")
        for rule in VALID_AOI_SELECTIONS:
            assert _local(aoi_selection=rule.upper()).aoi_selection == rule
        cfg = _local(aoi_selection="clip")
        assert AgriboundConfig.from_dict(cfg.to_dict()).aoi_selection == "clip"

    def test_gee_source_requires_project(self):
        with (
            patch.dict("os.environ", _env_without_project(), clear=True),
            patch("agribound.auth._get_gcloud_project", return_value=None),
            pytest.raises(ValueError, match="gee_project is required"),
        ):
            AgriboundConfig(source="sentinel2")

    def test_spot_pan_requires_project(self):
        with (
            patch.dict("os.environ", _env_without_project(), clear=True),
            patch("agribound.auth._get_gcloud_project", return_value=None),
            pytest.raises(ValueError, match="gee_project is required"),
        ):
            AgriboundConfig(source="spot-pan", year=2020)

    def test_gee_project_from_env(self):
        with patch.dict("os.environ", {"GEE_PROJECT": "env-project"}):
            cfg = AgriboundConfig(source="sentinel2")
        assert cfg.gee_project == "env-project"

    def test_gee_source_with_project_ok(self):
        cfg = AgriboundConfig(source="sentinel2", gee_project="my-project")
        assert cfg.source == "sentinel2"

    def test_local_source_without_tif_raises(self):
        with pytest.raises(ValueError, match="local_tif_path is required"):
            AgriboundConfig(source="local")

    def test_fine_tune_without_reference_raises(self):
        with pytest.raises(ValueError, match="reference_boundaries is required"):
            _local(fine_tune=True)

    def test_fine_tune_non_tunable_engine_raises(self):
        with pytest.raises(ValueError, match="does not support fine-tuning"):
            _local(engine="ftw", fine_tune=True, reference_boundaries="ref.gpkg")

    def test_fine_tune_tunable_engine_ok(self):
        cfg = _local(engine="dinov3", fine_tune=True, reference_boundaries="ref.gpkg")
        assert cfg.fine_tune is True

    def test_column_split_requires_column(self):
        with pytest.raises(ValueError, match="fine_tune_split_column is required"):
            _local(fine_tune_split="column")
        cfg = _local(fine_tune_split="column", fine_tune_split_column="region")
        assert cfg.fine_tune_split_column == "region"

    def test_gcs_export_without_bucket_raises(self):
        with pytest.raises(ValueError, match="gcs_bucket is required"):
            AgriboundConfig(
                source="sentinel2",
                gee_project="proj",
                export_method="gcs",
            )

    def test_source_case_insensitive(self):
        cfg = AgriboundConfig(source="LOCAL", local_tif_path="/tmp/x.tif")
        assert cfg.source == "local"

    @pytest.mark.parametrize(
        "field,value",
        [
            ("cloud_cover_max", 500),
            ("cloud_cover_max", -1),
            ("lulc_crop_threshold", 5.0),
            ("cloud_score_threshold", 1.5),
            ("fine_tune_val_split", 1.5),
            ("fine_tune_val_split", 0.0),
            ("min_field_area_m2", -5),
            ("simplify_tolerance", -1),
            ("naip_resolution_m", 0),
            ("fine_tune_block_size_m", -10),
            ("sam_crop_padding", -0.1),
            ("n_workers", -1),
            ("tile_size", -1),
            ("lulc_batch_size", 0),
            ("gee_max_requests", 0),
            ("sam_min_crop_px", 0),
            ("fine_tune_epochs", 0),
            ("seed", -1),
            ("seed", 2**32),
            ("usgs_retries", -1),
        ],
    )
    def test_numeric_ranges(self, field, value):
        with pytest.raises(ValueError, match=field):
            _local(**{field: value})

    @pytest.mark.parametrize("field,value", [("seed", 1.5), ("year", "abc"), ("seed", True)])
    def test_non_integer_raises(self, field, value):
        with pytest.raises(TypeError):
            _local(**{field: value})

    def test_numeric_string_year_is_coerced(self):
        assert _local(year="2020").year == 2020

    def test_gee_max_requests_above_limit_warns(self, caplog):
        with caplog.at_level("WARNING", logger="agribound.config"):
            _local(gee_max_requests=64)
        assert "exceeds" in caplog.text


class TestEngineSourceCompatibility:
    """Engine/source compatibility is enforced from the registry."""

    def test_ftw_rejects_naip(self):
        with pytest.raises(ValueError, match="does not support source 'naip'"):
            AgriboundConfig(source="naip", engine="ftw", gee_project="p", year=2020)

    def test_error_lists_supported_sources(self):
        with pytest.raises(ValueError, match="sentinel2"):
            AgriboundConfig(source="google-embedding", engine="ftw")

    def test_embedding_engine_rejects_imagery(self):
        with pytest.raises(ValueError, match="does not support source"):
            _local(engine="embedding")

    def test_delineate_anything_accepts_spot_pan_and_usgs(self):
        with pytest.warns(UserWarning):
            cfg = AgriboundConfig(source="spot-pan", gee_project="p", year=2020)
        assert cfg.engine == "delineate-anything"
        cfg = AgriboundConfig(source="usgs-naip-plus", year=2022)
        assert cfg.source == "usgs-naip-plus"

    def test_ensemble_members_checked(self):
        with pytest.raises(ValueError, match="Ensemble member 'ftw'"):
            AgriboundConfig(
                source="naip",
                engine="ensemble",
                gee_project="p",
                year=2020,
                engine_params={"engines": ["delineate-anything", {"engine": "ftw"}]},
            )

    @pytest.mark.parametrize("source", ["naip", "usgs-naip-plus"])
    def test_ensemble_default_members_checked(self, source):
        """Without engine_params['engines'] the default members (DA + FTW) are validated."""
        with pytest.raises(ValueError, match=r"'ftw' \(a default member\)") as info:
            AgriboundConfig(source=source, engine="ensemble", gee_project="p", year=2020)
        message = str(info.value)
        assert "['delineate-anything', 'ftw']" in message
        assert "'delineate-anything', 'geoai', 'dinov3'" in message.split("members that")[1]
        # Explicit members that support the source are accepted.
        cfg = AgriboundConfig(
            source=source,
            engine="ensemble",
            gee_project="p",
            year=2020,
            engine_params={"engines": ["delineate-anything", "geoai"]},
        )
        assert cfg.engine == "ensemble"
        # Sources that both default members support need no engine list.
        assert _local(engine="ensemble").engine == "ensemble"

    def test_ensemble_unknown_member(self):
        with pytest.raises(ValueError, match="Invalid ensemble member"):
            _local(engine="ensemble", engine_params={"engines": ["nope"]})

    def test_ensemble_valid_members(self):
        cfg = _local(
            engine="ensemble",
            engine_params={"engines": ["delineate-anything", {"engine": "ftw"}]},
        )
        assert cfg.engine == "ensemble"


class TestYearValidation:
    """Year ranges come from registry.source_year_range."""

    def test_naip_after_2023_rejected(self):
        with pytest.raises(ValueError, match="2002-2023"):
            AgriboundConfig(source="naip", gee_project="p", year=2024)

    def test_sentinel2_before_2017_rejected(self):
        with pytest.raises(ValueError, match="2017-present"):
            AgriboundConfig(source="sentinel2", gee_project="p", year=2016)

    def test_future_year_rejected_for_open_ranges(self):
        with pytest.raises(ValueError, match="outside the available range"):
            AgriboundConfig(source="landsat", gee_project="p", year=2999)

    def test_landsat_1984_ok(self):
        assert AgriboundConfig(source="landsat", gee_project="p", year=1984).year == 1984

    def test_landsat_pan_year(self):
        assert AgriboundConfig(source="landsat-pan", gee_project="p", year=1999).year == 1999
        with pytest.raises(ValueError, match="1999-present"):
            AgriboundConfig(source="landsat-pan", gee_project="p", year=1998)

    def test_google_embedding_range(self):
        with pytest.raises(ValueError, match="2017-2025"):
            AgriboundConfig(source="google-embedding", engine="embedding", year=2016)
        assert AgriboundConfig(source="google-embedding", engine="embedding", year=2025)

    def test_tessera_version_specific_range(self):
        with pytest.raises(ValueError, match="tessera_version='v1'"):
            AgriboundConfig(source="tessera-embedding", engine="embedding", year=2015)
        cfg = AgriboundConfig(
            source="tessera-embedding", engine="embedding", year=2015, tessera_version="v1.1"
        )
        assert cfg.year == 2015

    def test_local_has_no_year_constraint(self):
        assert _local(year=1900).year == 1900


class TestDateRange:
    def test_valid_date_range_normalised(self):
        cfg = _local(date_range=["2023-06-01", "2023-09-30"])
        assert cfg.date_range == ("2023-06-01", "2023-09-30")

    @pytest.mark.parametrize(
        "value",
        [("bad", "x"), ("2023-13-01", "2023-12-31"), ("2023-06-01",), ("2023-09-30", "2023-06-01")],
    )
    def test_invalid_date_range(self, value):
        with pytest.raises(ValueError, match="date_range"):
            _local(date_range=value)


class TestExportCrs:
    def test_utm_default_and_case(self):
        assert _local(export_crs="UTM").export_crs == "utm"

    def test_epsg_normalised(self):
        assert _local(export_crs="epsg:32611").export_crs == "EPSG:32611"

    def test_invalid_crs(self):
        with pytest.raises(ValueError, match="export_crs"):
            _local(export_crs="mercator")
        with pytest.raises(ValueError, match="unknown EPSG"):
            _local(export_crs="EPSG:999999")

    def test_geographic_crs_warns(self, caplog):
        with caplog.at_level("WARNING", logger="agribound.config"):
            assert _local(export_crs="EPSG:4326").export_crs == "EPSG:4326"
        assert "geographic" in caplog.text


class TestWorkloadTag:
    @pytest.mark.parametrize("tag", ["agribound-run1", "a", "run_1.v2", "A9"])
    def test_valid(self, tag):
        assert _local(gee_workload_tag=tag).gee_workload_tag == tag

    @pytest.mark.parametrize("tag", ["-bad", "bad-", "has space", "x" * 64, ""])
    def test_invalid(self, tag):
        with pytest.raises(ValueError, match="gee_workload_tag"):
            _local(gee_workload_tag=tag)


class TestOutputFormat:
    def test_format_inferred_from_extension(self):
        cfg = _local(output_path="out/fields.parquet")
        assert cfg.output_format == "parquet"
        assert _local(output_path="x.json").output_format == "geojson"

    def test_default_path_follows_format(self):
        cfg = _local(output_format="geojson")
        assert cfg.output_path == "fields.geojson"

    def test_conflict_raises(self):
        with pytest.raises(ValueError, match="conflicts"):
            _local(output_path="x.gpkg", output_format="parquet")

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="unsupported extension"):
            _local(output_path="fields.shp")


class TestSamRefineCompat:
    def test_engine_params_sam_refine_promoted(self):
        cfg = _local(engine_params={"sam_refine": True})
        assert cfg.sam_refine is True

    def test_merged_can_disable_legacy_flag(self):
        cfg = _local(engine_params={"sam_refine": True})
        off = cfg.merged(sam_refine=False)
        assert off.sam_refine is False
        assert off.engine_params["sam_refine"] is False
        assert cfg.sam_refine is True


class TestAgriboundConfigYaml:
    """Test YAML serialization round-trip."""

    def test_yaml_round_trip(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        original = AgriboundConfig(
            source="local",
            local_tif_path="/tmp/test.tif",
            engine="ftw",
            year=2023,
            min_field_area_m2=5000.0,
            simplify_tolerance=3.0,
            device="cpu",
        )
        original.to_yaml(yaml_path)
        loaded = AgriboundConfig.from_yaml(yaml_path)

        assert loaded.source == original.source
        assert loaded.engine == original.engine
        assert loaded.year == original.year
        assert loaded.local_tif_path == original.local_tif_path
        assert loaded.min_field_area_m2 == original.min_field_area_m2
        assert loaded.simplify_tolerance == original.simplify_tolerance
        assert loaded.device == original.device

    def test_yaml_round_trip_all_fields(self, tmp_path):
        original = _local(
            date_range=("2023-06-01", "2023-09-30"),
            seed=7,
            export_crs="EPSG:32611",
            s2_cloud_mask="cloud_score_plus",
            cloud_score_threshold=0.55,
            naip_resolution_m=0.6,
            tessera_version="v1.1",
            tessera_variant="cambridge",
            embedding_cache_dir="/scratch/emb",
            google_embedding_backend="source_coop",
            lulc_filter=False,
            lulc_dataset="c3s",
            lulc_on_error="warn",
            lulc_mode="raster",
            lulc_nodata_policy="drop",
            sam_refine=True,
            sam_backend="sam3",
            sam_model="facebook/sam3",
            sam_min_crop_px=32,
            sam_crop_padding=0.2,
            fine_tune_split="column",
            fine_tune_block_size_m=2000.0,
            fine_tune_split_column="grp",
            gee_service_account_key="/keys/sa.json",
            gee_high_volume=True,
            gee_max_requests=4,
            gee_workload_tag="agribound-test",
            cache_dir="/scratch/cache",
            overwrite=True,
            provenance=False,
            bands={"R": 3, "G": 2, "B": 1},
            engine_params={"model": "x", "window": (1, 2)},
        )
        path = tmp_path / "all.yaml"
        original.to_yaml(path)
        raw = yaml.safe_load(path.read_text())
        assert raw["date_range"] == ["2023-06-01", "2023-09-30"]
        loaded = AgriboundConfig.from_yaml(path)
        expected = original.to_dict()
        expected["engine_params"]["window"] = [1, 2]  # tuples are written as lists
        assert loaded.to_dict() == expected
        assert loaded.date_range == ("2023-06-01", "2023-09-30")

    def test_yaml_with_date_range(self, tmp_path):
        yaml_path = tmp_path / "config_dr.yaml"
        original = AgriboundConfig(
            source="local",
            local_tif_path="/tmp/test.tif",
            date_range=("2023-06-01", "2023-09-30"),
        )
        original.to_yaml(yaml_path)
        loaded = AgriboundConfig.from_yaml(yaml_path)
        assert loaded.date_range == ("2023-06-01", "2023-09-30")

    def test_from_yaml_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AgriboundConfig.from_yaml(tmp_path / "nonexistent.yaml")

    def test_from_yaml_unknown_keys(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("source: local\nlocal_tif_path: /tmp/x.tif\nsead: 1\nfoo: bar\n")
        with pytest.raises(ValueError, match=r"Unknown configuration key\(s\): \['foo', 'sead'\]"):
            AgriboundConfig.from_yaml(path)

    def test_from_yaml_not_mapping(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(ValueError, match="mapping"):
            AgriboundConfig.from_yaml(path)

    def test_from_yaml_empty_file_uses_defaults(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with patch.dict("os.environ", {"GEE_PROJECT": "p"}):
            cfg = AgriboundConfig.from_yaml(path)
        assert cfg.source == "sentinel2"

    def test_to_yaml_str(self):
        text = _local(date_range=("2023-01-01", "2023-02-01")).to_yaml_str()
        assert "date_range:\n- '2023-01-01'" in text


class TestAgriboundConfigDevice:
    """Test device resolution logic."""

    def test_explicit_cpu(self):
        assert _local(device="cpu").resolve_device() == "cpu"

    def test_explicit_cuda(self):
        assert _local(device="cuda").resolve_device() == "cuda"

    def test_auto_without_torch_falls_back_to_cpu(self):
        cfg = _local(device="auto")
        with patch.dict("sys.modules", {"torch": None}):
            assert cfg.resolve_device() == "cpu"

    def test_auto_with_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        cfg = _local(device="auto")
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert cfg.resolve_device() == "cuda"

    def test_auto_with_mps(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        cfg = _local(device="auto")
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert cfg.resolve_device() == "mps"

    def test_auto_cpu_fallback(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        cfg = _local(device="auto")
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert cfg.resolve_device() == "cpu"


class TestAgriboundConfigHelpers:
    """Test helper methods."""

    def test_is_gee_source(self):
        cfg = AgriboundConfig(source="sentinel2", gee_project="proj")
        assert cfg.is_gee_source() is True

    def test_spot_pan_is_gee_source(self):
        with pytest.warns(UserWarning):
            cfg = AgriboundConfig(source="spot-pan", gee_project="proj", year=2020)
        assert cfg.is_gee_source() is True

    def test_is_not_gee_source(self):
        assert _local().is_gee_source() is False

    def test_embedding_is_not_gee_imagery_source(self):
        cfg = AgriboundConfig(source="google-embedding", engine="embedding")
        assert cfg.is_gee_source() is False
        assert cfg.is_embedding_source() is True

    def test_requires_gee(self):
        assert _local(lulc_filter=False).requires_gee() is False
        assert _local(lulc_filter=True).requires_gee() is True
        emb = AgriboundConfig(source="google-embedding", engine="embedding", lulc_filter=False)
        assert emb.requires_gee() is True
        coop = emb.merged(google_embedding_backend="source_coop")
        assert coop.requires_gee() is False

    def test_get_output_extension(self):
        assert _local().get_output_extension() == ".gpkg"

    def test_to_dict(self):
        d = _local().to_dict()
        assert isinstance(d, dict)
        assert d["source"] == "local"
        assert d["local_tif_path"] == "/tmp/x.tif"

    def test_to_dict_date_range_is_list(self):
        d = _local(date_range=("2023-01-01", "2023-02-01")).to_dict()
        assert d["date_range"] == ["2023-01-01", "2023-02-01"]

    def test_from_dict(self):
        d = {"source": "local", "local_tif_path": "/tmp/x.tif", "engine": "ftw"}
        cfg = AgriboundConfig.from_dict(d)
        assert cfg.source == "local"
        assert cfg.engine == "ftw"

    def test_from_dict_does_not_mutate_input(self):
        d = {
            "source": "local",
            "local_tif_path": "/tmp/x.tif",
            "date_range": ["2023-01-01", "2023-01-31"],
        }
        AgriboundConfig.from_dict(d)
        assert d["date_range"] == ["2023-01-01", "2023-01-31"]

    def test_from_dict_unknown_key(self):
        with pytest.raises(ValueError, match="Unknown configuration key"):
            AgriboundConfig.from_dict({"source": "local", "local_tif_path": "x", "bogus": 1})

    def test_merged(self):
        cfg = _local(engine_params={"a": 1})
        other = cfg.merged(year=2020, engine="ftw")
        assert other.year == 2020 and other.engine == "ftw"
        assert cfg.year == 2024 and cfg.engine == "delineate-anything"
        other.engine_params["a"] = 2
        assert cfg.engine_params["a"] == 1

    def test_merged_revalidates(self):
        with pytest.raises(ValueError, match="does not support source"):
            _local().merged(engine="embedding")

    def test_merged_unknown_key(self):
        with pytest.raises(ValueError, match="Unknown configuration key"):
            _local().merged(nope=1)


class TestWorkingDir:
    def test_default_working_dir(self, tmp_path):
        cfg = _local(output_path=str(tmp_path / "out" / "fields.gpkg"))
        wd = cfg.get_working_dir()
        assert wd == tmp_path / "out" / ".agribound_cache"
        assert wd.is_dir()

    def test_cache_dir_override(self, tmp_path):
        cfg = _local(
            output_path=str(tmp_path / "out" / "fields.gpkg"),
            cache_dir=str(tmp_path / "shared_cache"),
        )
        wd = cfg.get_working_dir()
        assert wd == tmp_path / "shared_cache"
        assert wd.is_dir()
        assert not (tmp_path / "out" / ".agribound_cache").exists()
