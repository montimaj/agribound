"""Conservative visible-band PAN detail injection for controlled experiments.

This is an experimental image fusion, not a calibrated 15 m SR product.
Only RGB is sharpened: Landsat 8/9 PAN does not cover NIR or SWIR.
The box-degraded result preserves each input SR pixel's band means.
"""

from __future__ import annotations

import numpy as np


def block_mean(array: np.ndarray, factor: int = 2) -> np.ndarray:
    """Average complete square blocks; a missing subpixel invalidates its block."""
    data = np.asarray(array, dtype=np.float64)
    h, w = data.shape[-2:]
    if factor < 1 or h % factor or w % factor:
        raise ValueError("Image dimensions must be divisible by the positive block factor")
    return data.reshape(*data.shape[:-2], h // factor, factor, w // factor, factor).mean(
        axis=(-3, -1)
    )


def inject_pan_detail(
    sr_rgb: np.ndarray, pan: np.ndarray, *, strength: float = 1.0
) -> tuple[np.ndarray, dict]:
    """Inject zero-mean 15 m PAN detail into 30 m RGB SR (unit reflectance).

    Fit a nonnegative least-squares slope (with intercept) between degraded
    PAN and SR RGB intensity on valid 30 m cells. Add that slope times PAN's
    within-cell residual equally to the three nearest-neighbour SR bands.
    A cell-wide attenuation prevents negative values without changing means.
    Full 2x2 support is required: no interpolation across clouds or nodata.
    """
    rgb = np.asarray(sr_rgb, dtype=np.float64)
    p = np.asarray(pan, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[0] != 3 or p.ndim != 2:
        raise ValueError("Expected RGB (3,H,W) and PAN (2H,2W)")
    if p.shape != (2 * rgb.shape[1], 2 * rgb.shape[2]):
        raise ValueError("PAN must have exactly twice the aligned SR dimensions")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be finite and between 0 and 1")
    low = block_mean(p)
    valid = np.isfinite(rgb).all(axis=0) & np.isfinite(low)
    if not valid.any():
        raise ValueError("No complete common valid PAN/SR cells")
    if np.any(rgb[:, valid] < 0):
        raise ValueError("SR must be nonnegative unit reflectance")
    x, y = low[valid], rgb.mean(axis=0)[valid]
    variance = float(np.var(x))
    gain = (
        max(0.0, float(np.mean((x - x.mean()) * (y - y.mean()))) / variance)
        if variance > 1e-12
        else 0.0
    )
    correlation = float(np.corrcoef(x, y)[0, 1]) if variance > 1e-12 and np.var(y) > 1e-12 else None

    def up(a):
        return np.repeat(np.repeat(a, 2, axis=-2), 2, axis=-1)

    detail = (p - up(low)) * gain * strength
    blocks = detail.reshape(rgb.shape[1], 2, rgb.shape[2], 2)
    negative_peak = -blocks.min(axis=(1, 3))
    alpha = np.ones_like(low)
    np.divide(rgb.min(axis=0), negative_peak, out=alpha, where=negative_peak > 0)
    alpha = np.clip(alpha, 0, 1)
    fused = up(rgb) + up(alpha) * detail
    fused[:, ~up(valid)] = np.nan
    output = fused.astype(np.float32)
    error = block_mean(output) - rgb
    diagnostics = {
        "method": "visible RGB zero-mean block high-pass injection",
        "strength": strength,
        "gain_sr_per_toa": gain,
        "coarse_pan_rgb_correlation": correlation,
        "valid_30m_cells": int(valid.sum()),
        "attenuated_cell_fraction": float(np.mean(alpha[valid] < 1)),
        "coarse_band_rmse": [float(np.sqrt(np.mean(e[valid] ** 2))) for e in error],
        "coarse_max_abs_error": float(np.max(np.abs(error[:, valid]))),
        "detail_rms": float(np.sqrt(np.mean((up(alpha) * detail)[up(valid)] ** 2))),
        "interpretation": "experimental fused RGB; not calibrated 15 m surface reflectance",
    }
    return output, diagnostics


def reduced_resolution_validation(sr_rgb: np.ndarray, pan: np.ndarray) -> dict:
    """Box-degrade SR to 60 m and PAN to 30 m; compare fusion to known 30 m SR.

    This is a reduced-resolution consistency test, not evidence of 15 m SR
    accuracy. The box degradation approximates pixel averaging, not sensor MTF.
    """
    rgb = np.asarray(sr_rgb, dtype=np.float64)
    h, w = rgb.shape[-2:]
    h, w = h - h % 2, w - w % 2
    if h < 2 or w < 2:
        raise ValueError("Reduced-resolution validation needs at least 2x2 SR pixels")
    truth = rgb[:, :h, :w]
    degraded_rgb = block_mean(truth)
    degraded_pan = block_mean(np.asarray(pan)[: 2 * h, : 2 * w])
    fused, diagnostics = inject_pan_detail(degraded_rgb, degraded_pan)
    baseline = np.repeat(np.repeat(degraded_rgb, 2, axis=-2), 2, axis=-1)
    valid = np.isfinite(truth).all(axis=0) & np.isfinite(fused).all(axis=0)

    def scores(candidate):
        rmse = np.sqrt(np.mean((candidate[:, valid] - truth[:, valid]) ** 2, axis=1))
        a, b = candidate[:, valid], truth[:, valid]
        denominator = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
        nonzero = denominator > 1e-12
        cosine = np.sum(a[:, nonzero] * b[:, nonzero], axis=0) / denominator[nonzero]
        angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        return {
            "band_rmse": rmse.tolist(),
            "spectral_angle_median_deg": float(np.median(angles)) if len(angles) else None,
        }

    return {
        "reference_resolution_m": 30,
        "input_sr_resolution_m": 60,
        "input_pan_resolution_m": 30,
        "valid_pixels": int(valid.sum()),
        "resampled_sr": scores(baseline),
        "fused": scores(fused),
        "gain": diagnostics["gain_sr_per_toa"],
        "limitation": "box degradation, not MTF-matched; cannot validate true 15 m SR radiometry",
    }
