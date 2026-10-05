"""Offline invariants for visible-band detail fusion and scene pairing."""

from __future__ import annotations

import numpy as np
import pytest

from agribound.composites.landsat_matched import match_scenes
from agribound.composites.pan_fusion import (
    block_mean,
    inject_pan_detail,
    reduced_resolution_validation,
)


def inputs():
    intensity = np.array([[0.1, 0.2], [0.3, 0.4]])
    rgb = np.stack([intensity * 1.2, intensity, intensity * 0.8])
    low_pan = np.repeat(np.repeat(intensity + 0.05, 2, axis=0), 2, axis=1)
    detail = np.tile([[0.01, -0.01], [-0.01, 0.01]], (2, 2))
    return rgb, low_pan + detail


def test_fusion_adds_actual_detail_and_preserves_coarse_spectral_means():
    rgb, pan = inputs()
    fused, diag = inject_pan_detail(rgb, pan)
    np.testing.assert_allclose(block_mean(fused), rgb, atol=1e-8)
    assert diag["gain_sr_per_toa"] == pytest.approx(1)
    assert np.ptp(fused[0, :2, :2]) == pytest.approx(0.02)
    # Equal injection preserves within-pixel R-G chromatic differences.
    np.testing.assert_allclose(
        fused[0] - fused[1], np.repeat(np.repeat(rgb[0] - rgb[1], 2, 0), 2, 1), atol=1e-8
    )


def test_strength_zero_is_only_resampling():
    rgb, pan = inputs()
    fused, _ = inject_pan_detail(rgb, pan, strength=0)
    np.testing.assert_allclose(fused, np.repeat(np.repeat(rgb, 2, 1), 2, 2), atol=1e-8)


def test_missing_subpixel_invalidates_entire_common_cell():
    rgb, pan = inputs()
    pan[0, 0] = np.nan
    fused, diag = inject_pan_detail(rgb, pan)
    assert np.isnan(fused[:, :2, :2]).all()
    assert np.isfinite(fused[:, 2:, 2:]).all()
    assert diag["valid_30m_cells"] == 3


def test_negative_prevention_preserves_means_instead_of_clipping():
    rgb, pan = inputs()
    rgb[2] *= 0.01
    pan += np.tile([[0.2, -0.2], [-0.2, 0.2]], (2, 2))
    fused, diag = inject_pan_detail(rgb, pan)
    assert np.min(fused) >= -1e-8
    np.testing.assert_allclose(block_mean(fused), rgb, atol=1e-8)
    assert diag["attenuated_cell_fraction"] > 0


def test_constant_pan_does_not_invent_detail():
    rgb, _ = inputs()
    fused, diag = inject_pan_detail(rgb, np.ones((4, 4)))
    assert diag["gain_sr_per_toa"] == 0
    assert diag["coarse_pan_rgb_correlation"] is None
    np.testing.assert_allclose(block_mean(fused), rgb, atol=1e-8)


@pytest.mark.parametrize("strength", [-1, 2, np.nan])
def test_invalid_strength_rejected(strength):
    with pytest.raises(ValueError, match="strength"):
        inject_pan_detail(*inputs(), strength=strength)


def test_misaligned_shapes_and_empty_support_rejected():
    rgb, pan = inputs()
    with pytest.raises(ValueError, match="aligned"):
        inject_pan_detail(rgb, pan[:2])
    with pytest.raises(ValueError, match="common valid"):
        inject_pan_detail(rgb, np.full_like(pan, np.nan))


def test_reduced_resolution_reports_both_baseline_and_fusion():
    intensity = 0.1 + np.arange(16).reshape(4, 4) / 100
    rgb = np.stack([intensity * 1.2, intensity, intensity * 0.8])
    pan = np.repeat(np.repeat(intensity + 0.05, 2, 0), 2, 1)
    result = reduced_resolution_validation(rgb, pan)
    assert result["valid_pixels"] == 16
    assert len(result["fused"]["band_rmse"]) == 3
    # At the coarser scale PAN exactly predicts the intensity variation.
    assert sum(result["fused"]["band_rmse"]) < sum(result["resampled_sr"]["band_rmse"])


def test_scene_pairing_uses_acquisition_id_not_different_product_ids():
    pan = [{"scene_id": "A", "product_id": "L1_A"}, {"scene_id": "B", "product_id": "L1_B"}]
    sr = [{"scene_id": "C", "product_id": "L2_C"}, {"scene_id": "A", "product_id": "L2_A"}]
    result = match_scenes(pan, sr)
    assert result["pairs"] == [{"scene_id": "A", "pan": pan[0], "sr": sr[1]}]
    assert result["unmatched_pan"] == [pan[1]]
    assert result["unmatched_sr"] == [sr[0]]


def test_ambiguous_scene_pairing_is_rejected():
    row = {"scene_id": "A", "product_id": "L1_A"}
    with pytest.raises(ValueError, match="duplicate"):
        match_scenes([row, row], [row])
