"""Explicit three-channel Landsat experiments; no NIR pansharpening.

Inputs are matched six-band SR (B2--B7, unit surface reflectance), native B8
TOA, and the established fused visible RGB. Complete common 30 m support is
required. Nearest-neighbour replication does not add spectral information.
"""

from __future__ import annotations

import numpy as np
from rasterio import Affine

from agribound.composites.pan_fusion import block_mean, inject_pan_detail

CHANNELS = {
    "false_color": ["SR_B5", "SR_B4", "SR_B3"],
    "false_color_15m": ["SR_B5_resampled", "SR_B4_resampled", "SR_B3_resampled"],
    "pan_nir_red": ["PAN_B8_TOA", "SR_B5_resampled", "SR_B4_resampled"],
    "coarse_pan_nir_red": ["PAN_B8_30m_resampled", "SR_B5_resampled", "SR_B4_resampled"],
    "hybrid": ["SR_B5_resampled", "R_fused", "G_fused"],
}


def repeat_2x(array: np.ndarray) -> np.ndarray:
    """Replicate aligned pixels exactly, with no interpolated detail."""
    return np.repeat(np.repeat(array, 2, axis=-2), 2, axis=-1)


def validate_grids(sr, pan, fused=None) -> None:
    """Require projected metre grids with exact nested extents and orientation."""
    if sr.crs is None or not sr.crs.is_projected or sr.crs.linear_units != "metre":
        raise ValueError("SR requires a projected metre CRS")
    if sr.crs != pan.crs or pan.transform != sr.transform * Affine.scale(0.5):
        raise ValueError("PAN/SR CRS and aligned 2:1 transforms must agree")
    if (pan.height, pan.width) != (2 * sr.height, 2 * sr.width):
        raise ValueError("PAN/SR extents and 2:1 dimensions must agree")
    if sr.transform.b or sr.transform.d or sr.transform.a <= 0 or sr.transform.e >= 0:
        raise ValueError("Expected north-up raster orientation")
    if not np.allclose(sr.res, (30, 30)) or not np.allclose(pan.res, (15, 15)):
        raise ValueError("Expected native 30 m SR and 15 m PAN")
    if sr.count != 6 or pan.count != 1:
        raise ValueError("Expected six SR bands B2--B7 and one PAN band")
    if fused is not None and (
        fused.crs != pan.crs
        or fused.transform != pan.transform
        or fused.shape != pan.shape
        or fused.count != 3
    ):
        raise ValueError("Fused visible RGB must match the PAN grid")


def build_inputs(sr: np.ndarray, pan: np.ndarray, fused_rgb: np.ndarray):
    """Build D--H with one common mask, retaining TOA/SR channel identities."""
    sr = np.asarray(sr, dtype=np.float32)
    pan = np.asarray(pan, dtype=np.float32)
    fused = np.asarray(fused_rgb, dtype=np.float32)
    if sr.ndim != 3 or sr.shape[0] != 6:
        raise ValueError("Expected SR (6,H,W) in B2--B7 order")
    if pan.shape != (2 * sr.shape[1], 2 * sr.shape[2]) or fused.shape != (3, *pan.shape):
        raise ValueError("Expected aligned PAN and fused RGB at twice SR dimensions")
    coarse = block_mean(pan).astype(np.float32)
    fused_valid = block_mean(np.isfinite(fused).all(axis=0).astype(float)) == 1
    common = np.isfinite(sr).all(axis=0) & np.isfinite(coarse) & fused_valid
    if not common.any():
        raise ValueError("No complete common valid observation support")
    fc = sr[[3, 2, 1]].copy()
    fc[:, ~common] = np.nan
    up = repeat_2x(fc)
    high_mask = repeat_2x(common)
    native_stack = np.stack([pan, up[0], up[1]])
    coarse_stack = np.stack([repeat_2x(coarse), up[0], up[1]])
    hybrid = np.stack([up[0], fused[0], fused[1]])
    for data in (native_stack, coarse_stack, hybrid):
        data[:, ~high_mask] = np.nan
    inputs = {
        "false_color": fc,
        "false_color_15m": up,
        "pan_nir_red": native_stack,
        "coarse_pan_nir_red": coarse_stack,
        "hybrid": hybrid,
    }
    diagnostic = {
        "valid_30m_cells": int(common.sum()),
        "valid_fraction": float(common.mean()),
        "sr_all_six_bands_required": True,
        "pan_four_subpixels_required": True,
        "baseline_sr_support_identical": bool(np.array_equal(common, np.isfinite(sr).all(0))),
        "baseline_pan_support_identical": bool(np.array_equal(high_mask, np.isfinite(pan))),
        "baseline_fused_support_identical": bool(
            np.array_equal(high_mask, np.isfinite(fused).all(0))
        ),
        "nir_max_abs_change_after_resampling": float(
            np.max(np.abs(hybrid[0][high_mask] - up[0][high_mask]))
        ),
        "hybrid_visible_max_abs_change": float(
            np.max(np.abs(hybrid[1:, high_mask] - fused[:2, high_mask]))
        ),
        "coarse_pan_mean_max_abs_error": float(
            np.max(np.abs(block_mean(coarse_stack[0])[common] - coarse[common]))
        ),
        "resampling": "aligned nearest-neighbour 2x replication; no added SR detail",
        "channel_radiometry": "PAN unit TOA; SR unit surface reflectance; fused experimental",
    }
    return inputs, diagnostic


def hybrid_reduced_validation(sr_rgb: np.ndarray, pan: np.ndarray) -> dict:
    """Validate H's visible pair at reduced scale using the established RGB gain."""
    h, w = sr_rgb.shape[-2:]
    h, w = h - h % 2, w - w % 2
    truth = sr_rgb[:, :h, :w]
    coarse_rgb = block_mean(truth)
    coarse_pan = block_mean(pan[: 2 * h, : 2 * w])
    fused, _ = inject_pan_detail(coarse_rgb, coarse_pan)
    baseline = repeat_2x(coarse_rgb)
    valid = np.isfinite(fused).all(0) & np.isfinite(truth).all(0)

    def scores(candidate):
        a, b = candidate[:2, valid], truth[:2, valid]
        denominator = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
        use = denominator > 1e-12
        angle = np.degrees(
            np.arccos(np.clip(np.sum(a[:, use] * b[:, use], axis=0) / denominator[use], -1, 1))
        )
        return {
            "red_green_rmse": np.sqrt(np.mean((a - b) ** 2, axis=1)).tolist(),
            "red_green_spectral_angle_median_deg": float(np.median(angle)) if use.any() else None,
        }

    return {
        "valid_pixels": int(valid.sum()),
        "resampled_visible": scores(baseline),
        "fused_visible": scores(fused),
        "limitation": "60 m SR + 30 m PAN against 30 m truth; box, not sensor MTF degradation",
        "nir": "unchanged nearest-neighbour SR; no NIR sharpening claim",
    }
