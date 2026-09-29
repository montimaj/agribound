"""Tests for the fine-tuning dispatcher's model keys (agribound.engines.finetune)."""

from __future__ import annotations

import pytest

from agribound.config import AgriboundConfig
from agribound.engines.finetune import _get_model_key


def _cfg(tmp_path, engine: str, **engine_params) -> AgriboundConfig:
    return AgriboundConfig(
        source="local",
        local_tif_path=str(tmp_path / "x.tif"),
        engine=engine,
        output_path=str(tmp_path / "out.gpkg"),
        lulc_filter=False,
        engine_params=engine_params,
    )


class TestModelKey:
    def test_delineate_anything_default_is_the_base_weights_key(self, tmp_path):
        from agribound.engines.delineate_anything import DEFAULT_DA_MODEL, resolve_da_model_key

        cfg = _cfg(tmp_path, "delineate-anything")
        assert _get_model_key("delineate-anything", cfg) == DEFAULT_DA_MODEL == "large_v2"
        assert _get_model_key("delineate-anything", cfg) == resolve_da_model_key({})

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            ({"da_model": "DelineateAnythingV2"}, "large_v2"),
            ({"da_model": "large"}, "large"),
            ({"model_size": "small"}, "small"),
            ({"model_size": "large"}, "large"),
        ],
    )
    def test_delineate_anything_aliases_are_normalised(self, tmp_path, params, expected):
        cfg = _cfg(tmp_path, "delineate-anything", **params)
        assert _get_model_key("delineate-anything", cfg) == expected

    def test_delineate_anything_invalid_model_raises(self, tmp_path):
        cfg = _cfg(tmp_path, "delineate-anything", da_model="nope")
        with pytest.raises(ValueError, match="Unknown da_model"):
            _get_model_key("delineate-anything", cfg)

    def test_dinov3_default_matches_trainer_alias(self, tmp_path):
        import inspect

        from agribound.engines.finetune import _dinov3

        cfg = _cfg(tmp_path, "dinov3")
        assert _get_model_key("dinov3", cfg) == "large"
        # The trainer's own default alias is "large" as well.
        assert 'params.get("dinov3_model", "large")' in inspect.getsource(_dinov3)
        assert _get_model_key("dinov3", _cfg(tmp_path, "dinov3", dinov3_model="small")) == "small"

    def test_prithvi_default_matches_engine_default(self, tmp_path):
        from agribound.engines.prithvi import DEFAULT_PRITHVI_MODEL

        cfg = _cfg(tmp_path, "prithvi")
        assert _get_model_key("prithvi", cfg) == DEFAULT_PRITHVI_MODEL

    def test_other_engines_use_their_name(self, tmp_path):
        assert _get_model_key("geoai", _cfg(tmp_path, "geoai")) == "geoai"
