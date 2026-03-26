"""Basic pipeline smoke tests (no GPU, GEE, or ML dependencies required)."""

from __future__ import annotations

import pytest


@pytest.mark.slow
class TestImportSmoke:
    """Verify the package can be imported without errors."""

    def test_import_agribound(self):
        import agribound  # noqa: F401

    def test_import_config(self):
        from agribound.config import AgriboundConfig  # noqa: F401

    def test_import_engines(self):
        from agribound.engines.base import list_engines  # noqa: F401

    def test_import_composites(self):
        from agribound.composites.base import list_sources  # noqa: F401

    def test_import_evaluate(self):
        from agribound.evaluate import evaluate  # noqa: F401

    def test_import_io(self):
        from agribound.io.raster import get_raster_info  # noqa: F401
        from agribound.io.vector import read_vector  # noqa: F401
        from agribound.io.crs import get_utm_crs  # noqa: F401


@pytest.mark.slow
class TestConfigCreation:
    """Test that AgriboundConfig can be created with various settings."""

    def test_local_config(self):
        from agribound.config import AgriboundConfig

        cfg = AgriboundConfig(
            source="local",
            local_tif_path="/tmp/test.tif",
            engine="delineate-anything",
            device="cpu",
        )
        assert cfg.source == "local"
        assert cfg.resolve_device() == "cpu"

    def test_embedding_config(self):
        from agribound.config import AgriboundConfig

        cfg = AgriboundConfig(
            source="google-embedding",
            engine="embedding",
            year=2023,
        )
        assert cfg.is_embedding_source() is True
        assert cfg.is_gee_source() is False

    def test_config_working_dir(self, tmp_path):
        from agribound.config import AgriboundConfig

        cfg = AgriboundConfig(
            source="local",
            local_tif_path="/tmp/test.tif",
            output_path=str(tmp_path / "output" / "fields.gpkg"),
        )
        wd = cfg.get_working_dir()
        assert wd.exists()
        assert wd.name == ".agribound_cache"
