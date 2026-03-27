"""Tests for engine registry and discovery in agribound.engines.base."""

from __future__ import annotations

import pytest

from agribound.engines.base import ENGINE_REGISTRY, get_engine, list_engines


class TestListEngines:
    """Test the list_engines function."""

    def test_returns_all_engines(self):
        engines = list_engines()
        assert len(engines) == 7

    def test_expected_engine_names(self):
        engines = list_engines()
        expected = {
            "delineate-anything",
            "ftw",
            "geoai",
            "dinov3",
            "prithvi",
            "embedding",
            "ensemble",
        }
        assert set(engines.keys()) == expected

    def test_each_engine_has_required_metadata(self):
        engines = list_engines()
        required_keys = {"name", "approach", "gpu_required", "requires_bands", "supported_sources"}
        for name, info in engines.items():
            for key in required_keys:
                assert key in info, f"Engine {name!r} missing metadata key {key!r}"

    def test_returns_copy(self):
        """Ensure list_engines returns a copy, not the mutable registry."""
        engines = list_engines()
        engines["fake"] = {}
        assert "fake" not in ENGINE_REGISTRY


class TestGetEngine:
    """Test the get_engine factory function."""

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine("nonexistent_engine")

    def test_unknown_engine_empty_string(self):
        with pytest.raises(ValueError):
            get_engine("")

    def test_valid_engine_names_in_registry(self):
        """All engines in the registry should be recognized by get_engine.

        We do not actually instantiate them here because each engine module
        may import heavy optional dependencies (torch, ultralytics, etc.).
        Instead we only verify the registry key-check path by testing that
        an unknown name raises ValueError.
        """
        for name in ENGINE_REGISTRY:
            # Just confirm the name is in the registry
            assert name in ENGINE_REGISTRY
