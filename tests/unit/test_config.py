"""Tests for agribound.config.AgriboundConfig."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agribound.config import AgriboundConfig


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

    def test_local_source_requires_tif(self):
        cfg = AgriboundConfig(source="local", local_tif_path="/data/img.tif")
        assert cfg.source == "local"
        assert cfg.local_tif_path == "/data/img.tif"


class TestAgriboundConfigValidation:
    """Test configuration validation."""

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError, match="Invalid source"):
            AgriboundConfig(source="invalid_source")

    def test_invalid_engine_raises(self):
        with pytest.raises(ValueError, match="Invalid engine"):
            AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", engine="nonexistent")

    def test_invalid_output_format_raises(self):
        with pytest.raises(ValueError, match="Invalid output_format"):
            AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", output_format="csv")

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="Invalid device"):
            AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", device="tpu")

    def test_gee_source_requires_project(self):
        with pytest.raises(ValueError, match="gee_project is required"):
            AgriboundConfig(source="sentinel2")

    def test_gee_source_with_project_ok(self):
        cfg = AgriboundConfig(source="sentinel2", gee_project="my-project")
        assert cfg.source == "sentinel2"

    def test_local_source_without_tif_raises(self):
        with pytest.raises(ValueError, match="local_tif_path is required"):
            AgriboundConfig(source="local")

    def test_fine_tune_without_reference_raises(self):
        with pytest.raises(ValueError, match="reference_boundaries is required"):
            AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", fine_tune=True)

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


class TestAgriboundConfigDevice:
    """Test device resolution logic."""

    def test_explicit_cpu(self):
        cfg = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", device="cpu")
        assert cfg.resolve_device() == "cpu"

    def test_explicit_cuda(self):
        cfg = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", device="cuda")
        assert cfg.resolve_device() == "cuda"

    def test_auto_without_torch_falls_back_to_cpu(self):
        cfg = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", device="auto")
        with patch.dict("sys.modules", {"torch": None}):
            # When torch import fails, should fall back to cpu
            assert cfg.resolve_device() in ("cuda", "mps", "cpu")

    def test_auto_with_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        cfg = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", device="auto")
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert cfg.resolve_device() == "cuda"

    def test_auto_with_mps(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        cfg = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", device="auto")
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert cfg.resolve_device() == "mps"

    def test_auto_cpu_fallback(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        cfg = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", device="auto")
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert cfg.resolve_device() == "cpu"


class TestAgriboundConfigHelpers:
    """Test helper methods."""

    def test_is_gee_source(self):
        cfg = AgriboundConfig(source="sentinel2", gee_project="proj")
        assert cfg.is_gee_source() is True

    def test_is_not_gee_source(self):
        cfg = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif")
        assert cfg.is_gee_source() is False

    def test_is_embedding_source(self):
        cfg = AgriboundConfig(source="google-embedding", engine="embedding")
        assert cfg.is_embedding_source() is True

    def test_get_output_extension(self):
        cfg = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif")
        assert cfg.get_output_extension() == ".gpkg"

    def test_to_dict(self):
        cfg = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif")
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["source"] == "local"
        assert d["local_tif_path"] == "/tmp/x.tif"

    def test_from_dict(self):
        d = {"source": "local", "local_tif_path": "/tmp/x.tif", "engine": "ftw"}
        cfg = AgriboundConfig.from_dict(d)
        assert cfg.source == "local"
        assert cfg.engine == "ftw"
