"""Tests for DINOv3's opt-in growth ``engine_params["dilate_interior_px"] = "half_band"``.

The band measurement (agribound.engines.finetune._data.predicted_band_thickness),
the rule g = ceil(T / 2) (agribound.engines.dinov3.half_band_dilate_px), its
resolution, and the engine with a fake model (no weights): the default growth
and its record are unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from agribound.config import AgriboundConfig
from agribound.engines.dinov3 import (
    DILATE_HALF_BAND,
    half_band_dilate_px,
    resolve_dilate_px,
    resolve_dilate_rule,
)
from agribound.engines.finetune import _data
from agribound.engines.finetune._data import predicted_band_thickness, semantic_from_instances

UTM = "EPSG:32755"
X0, Y0 = 700_000.0, 6_600_000.0


def _field(band: int, side: int = 400, margin: int = 10) -> np.ndarray:
    """One square field on background, with a boundary band of *band* px inside its edge."""
    pred = np.zeros((side + 2 * margin, side + 2 * margin), dtype=np.uint8)
    pred[margin : margin + side, margin : margin + side] = 2
    inner = slice(margin + band, margin + side - band)
    pred[inner, inner] = 1
    return pred


class TestBandThickness:
    @pytest.mark.parametrize(("band", "growth"), [(8, 4), (12, 6), (24, 12), (25, 13), (3, 2)])
    def test_band_and_growth(self, band, growth):
        pred = _field(band)
        thickness, n_edge = predicted_band_thickness(pred)
        assert thickness == band and n_edge == 4 * 400 - 4
        result = half_band_dilate_px(pred, fallback_px=4)
        assert result == {
            "dilate_interior_px": growth,
            "band_thickness_px": band,
            "band_edge_px": n_edge,
            "fallback": False,
        }
        assert growth == math.ceil(band / 2)

    def test_trained_band_gives_half_the_trained_growth(self):
        """A model that predicts its 4 px training band gets 2 px, not the default 4."""
        instances = np.zeros((300, 300), dtype=np.int32)
        instances[20:280, 20:150] = 1
        instances[20:280, 150:280] = 2  # a neighbouring field
        pred = semantic_from_instances(instances, boundary_erosion=4)
        assert predicted_band_thickness(pred)[0] == 4
        assert half_band_dilate_px(pred, fallback_px=4)["dilate_interior_px"] == 2

    def test_median_is_truncated(self):
        """Half the edge at 8 px, half at 9 px: a median of 8.5 gives T = 8 and g = 4."""
        pred = np.ones((200, 60), dtype=np.uint8)
        pred[:, 0] = 0  # background column
        pred[:100, 1:9] = 2  # 8 px band
        pred[100:, 1:10] = 2  # 9 px band
        assert predicted_band_thickness(pred) == (8, 200)
        assert half_band_dilate_px(pred, fallback_px=4)["dilate_interior_px"] == 4

    def test_array_edge_is_not_background(self):
        pred = np.full((50, 50), 2, dtype=np.uint8)
        pred[10:, 10:] = 1  # interior touching the array's bottom and right edges
        assert predicted_band_thickness(pred) == (None, 0)  # no background pixel at all
        pred[0, :] = 0  # background along the top: row 1 is the band's outer edge
        assert predicted_band_thickness(pred) == (9, 50)

    def test_fallback_without_interior_or_edge(self):
        no_interior = np.zeros((20, 20), dtype=np.uint8)
        no_interior[5:15, 5:15] = 2
        assert predicted_band_thickness(no_interior) == (None, 36)
        result = half_band_dilate_px(no_interior, fallback_px=4)
        assert result["dilate_interior_px"] == 4 and result["fallback"] is True
        assert result["band_thickness_px"] is None


class TestResolution:
    def test_rule(self):
        for value in (None, 4, 4.0, "4", " 6 "):
            assert resolve_dilate_rule(value) == "fixed"
        for value in ("half_band", " Half_Band "):
            assert resolve_dilate_rule(value) == DILATE_HALF_BAND
        for value in ("half", "ceil", ""):
            with pytest.raises(ValueError, match="dilate_interior_px"):
                resolve_dilate_rule(value)

    def test_dilate_px_unchanged_by_default_and_trained_width_as_fallback(self):
        assert resolve_dilate_px({}, {"boundary_erosion": 4}, {}) == 4
        assert resolve_dilate_px({"dilate_interior_px": 5}, {"boundary_erosion": 4}, {}) == 5
        assert resolve_dilate_px({"dilate_interior_px": "3"}, {}, {}) == 3
        assert resolve_dilate_px({}, {}, {}) == 2
        # half_band: the trained width (used where the band cannot be measured)
        params = {"dilate_interior_px": "half_band"}
        assert resolve_dilate_px(params, {"boundary_erosion": 4}, {}) == 4
        assert resolve_dilate_px(params, {}, {"boundary_erosion": 3}) == 3
        assert resolve_dilate_px(params, {}, {}) == 2
        with pytest.raises(ValueError, match="dilate_interior_px"):
            resolve_dilate_px({"dilate_interior_px": "wide"}, {}, {})

    def test_checked_when_the_configuration_is_built(self, tmp_path):
        """DINOv3's rule (directly or as an ensemble member) and Prithvi's refusal of
        "half_band" are checked before any composite is built."""

        def config(engine, **params):
            return AgriboundConfig(
                source="local",
                local_tif_path=str(tmp_path / "x.tif"),
                engine=engine,
                output_path=str(tmp_path / "o.gpkg"),
                engine_params=params,
            )

        for value in ("half_band", 6, "6"):
            config("dinov3", dilate_interior_px=value)
        with pytest.raises(ValueError, match="must be a number of pixels or 'half_band'"):
            config("dinov3", dilate_interior_px="wide")
        member = {"engine": "dinov3", "engine_params": {"dilate_interior_px": "wide"}}
        with pytest.raises(ValueError, match="must be a number of pixels or 'half_band'"):
            config("ensemble", engines=["delineate-anything", member])
        config("prithvi", dilate_interior_px=4)
        for engine, params in (
            ("prithvi", {"dilate_interior_px": "Half_Band"}),
            (
                "ensemble",
                {
                    "engines": [
                        "delineate-anything",
                        {"engine": "prithvi", "engine_params": {"dilate_interior_px": "half_band"}},
                    ]
                },
            ),
        ):
            with pytest.raises(ValueError, match="'half_band' is DINOv3's growth rule"):
                config(engine, **params)

    def test_prithvi_refuses_half_band_before_loading_a_model(self, tmp_path):
        """A configuration changed after it was built still gets the message, not int()'s."""
        from agribound.engines.prithvi import PrithviEngine

        cfg = AgriboundConfig(
            source="local",
            local_tif_path=str(tmp_path / "x.tif"),
            engine="prithvi",
            output_path=str(tmp_path / "o.gpkg"),
            engine_params={"dilate_interior_px": 4},
        )
        cfg.engine_params["dilate_interior_px"] = "half_band"
        with pytest.raises(ValueError, match="'half_band' is DINOv3's growth rule"):
            PrithviEngine()._segment_mode(str(tmp_path / "x.tif"), cfg, str(tmp_path / "c.ckpt"))


# ---------------------------------------------------------------------------
# The engine, with a fake model
# ---------------------------------------------------------------------------

BAND = 12  # px of the boundary band the fake model predicts
EDGE = 10  # px of invalid (background) data on the left


def _write(path, data, res=10.0, nodata=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=str(data.dtype),
        crs=UTM,
        transform=from_origin(X0, Y0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return str(path)


@pytest.fixture
def fake_segmenter(monkeypatch):
    """The model: interior (1) where the red input exceeds 0.5, else boundary (2)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("geoai.dinov3_finetune")
    from agribound.engines import dinov3

    loads = []

    class FakeModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
            logits[:, 1] = torch.where(x[:, 0] > 0.5, 5.0, -5.0)
            logits[:, 2] = 1.0
            return logits

    class FakeSegmenter:
        @staticmethod
        def load_from_checkpoint(path, map_location=None, **kwargs):
            loads.append(path)
            return FakeModel()

    monkeypatch.setattr(dinov3, "dinov3_segmenter_class", lambda: FakeSegmenter)
    return loads


@pytest.fixture
def band_raster(tmp_path):
    """S2-like raster, 96 x 96 px: invalid left edge, then a 12 px dark band, then bright."""
    data = np.full((12, 96, 96), 500.0, dtype=np.float32)
    data[3, :, EDGE + BAND :] = 2500.0
    data[:, :, :EDGE] = np.nan
    return _write(tmp_path / "s2.tif", data, nodata=float("nan"))


def _config(tmp_path, ckpt, **params):
    return AgriboundConfig(
        source="sentinel2",
        year=2023,
        study_area="bbox:150.0,-30.8,150.1,-30.7",
        gee_project="test-project",
        engine="dinov3",
        output_path=str(tmp_path / "out" / "fields.gpkg"),
        lulc_filter=False,
        device="cpu",
        seed=5,
        min_field_area_m2=0,
        engine_params={"checkpoint_path": str(ckpt), "window_size": 64, **params},
    )


def _ckpt(path, boundary_erosion=4):
    import torch

    torch.save(
        {
            "hyper_parameters": {
                "model_name": "dinov3_vitl16",
                "num_classes": 3,
                "boundary_erosion": boundary_erosion,
            }
        },
        path,
    )
    return path


def test_engine_half_band_and_unchanged_default(tmp_path, band_raster, fake_segmenter):
    from agribound.engines.dinov3 import DINOv3Engine

    ckpt = _ckpt(tmp_path / "dinov3.ckpt")
    default = DINOv3Engine().delineate(band_raster, _config(tmp_path, ckpt))
    meta = default.attrs["engine_meta"]
    assert meta["dilate_interior_px"] == 4
    new_keys = {"dilate_interior_rule", "band_thickness_px", "band_edge_px", "dilate_fallback"}
    assert not new_keys & set(meta) and "trained_dilate_px" not in meta
    # the interior (columns 22-95) grown by the trained 4 px into the band
    assert len(default) == 1
    assert default.geometry.iloc[0].area == pytest.approx(96 * (96 - EDGE - BAND + 4) * 100.0)

    half = DINOv3Engine().delineate(
        band_raster, _config(tmp_path, ckpt, dilate_interior_px="half_band")
    )
    assert len(fake_segmenter) == 1  # the cached segmentation is reused
    meta = half.attrs["engine_meta"]
    assert meta["dilate_interior_rule"] == "half_band"
    assert meta["band_thickness_px"] == BAND and meta["band_edge_px"] == 96
    assert meta["dilate_interior_px"] == BAND // 2 and meta["trained_dilate_px"] == 4
    assert meta["dilate_fallback"] is False
    json.dumps(meta)
    assert half.geometry.iloc[0].area == pytest.approx(96 * (96 - EDGE - BAND + BAND // 2) * 100.0)
    # the polygons are those of interior_polygons with the measured growth
    (seg,) = [
        p
        for p in Path(tmp_path / "out" / ".agribound_cache").glob("dinov3_segmentation_*.tif")
        if "_fields_" not in p.name
    ]
    expected = _data.interior_polygons(seg, BAND // 2, 0, out_dir=tmp_path / "check")
    assert half.geometry.iloc[0].equals_exact(expected.geometry.iloc[0], 0.0)

    with pytest.raises(ValueError, match="dilate_interior_px"):
        DINOv3Engine().delineate(band_raster, _config(tmp_path, ckpt, dilate_interior_px="wide"))
