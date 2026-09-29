"""
FTW (Fields of The World) semantic-segmentation engine.

Runs an ftw-tools checkpoint on R, G, B and NIR and polygonises the predicted
field class (1) with ``ftw_tools.postprocess.polygonize.polygonize``. The
default model is the ftw-tools ``MODEL_REGISTRY`` entry marked ``default``
(``FTW_PRUE_EFNET_B5`` in ftw-tools 2.0.0b5: a PRUE U-Net with an
EfficientNet-B5 encoder, two input windows). Other registry models are chosen
with ``engine_params["model"]`` (see :func:`list_ftw_models`); a local
checkpoint with ``engine_params["checkpoint_path"]``.
Instance-segmentation entries of the registry (Delineate-Anything) are
rejected; use the ``delineate-anything`` engine for those.

Input windows
-------------
The number of windows follows the model: registry models use
``ModelSpec.requires_window``; for a checkpoint file ``in_channels`` is read
from its ``hyper_parameters`` (4 = one window, 8 = two windows).

Two-window models take ``[R, G, B, NIR]`` of an early-season window A
followed by the same bands of a late-season window B, the band order that
ftw-tools' own inference input builder writes (``ftw inference download``:
``create_input(win_a, win_b)`` stacks the scenes time-major, B04, B03, B02,
B08 per scene) and that ``ftw_tools.inference.inference.run`` passes to the
model unchanged. The window centres are
FTW's summer-crop start and end of season over the study-area bounding box
(``ftw_tools.utils.get_harvest_integer_from_bbox`` and
``harvest_to_datetime``), with the end of season placed in ``year + 1`` when
it falls before the start (southern-hemisphere seasons), as
``ftw_tools.download.download_img.scene_selection`` does. Each window is a
median composite over ``centre +/- window_days`` (engine parameter, default
30) built by the source's composite builder with ``date_range`` set, so each
window has its own cache entry. ``engine_params["window_dates"]`` (two
``"YYYY-MM-DD"`` centres) replaces the crop calendar. If a window has no
imagery (the composite builder raises
:class:`~agribound.composites.base.NoDataError`) the run fails with a
:class:`RuntimeError` (the :class:`NoDataError` chained as its cause),
unless ``engine_params["allow_annual_fallback"]`` is True, in which case the
annual composite is used for that window (logged as a WARNING and recorded in
``engine_meta``); other builder errors (invalid configuration,
authentication, quota, network) always propagate. Each window's record in
``engine_meta["windows"]`` also holds its composite's image count
(``n_images``), valid-pixel fraction (``valid_fraction``) and cloud mask
(``cloud_mask``), read from the composite's tags. A short window has few
images, and the default Sentinel-2 mask (SCL classes 3, 8, 9 and 10) can
leave haze or thin cloud in the median, which the valid fraction does not
reveal (a Namoi test window B of 11 images had haze over part of the study
area with ``valid_fraction`` 1.0). Look at the window composites
(``engine_meta["windows"][...]["raster"]``) or try
``s2_cloud_mask="cloud_score_plus"``.
:meth:`FTWEngine.stage_inputs` builds these inputs without running inference
(used by :mod:`agribound.hpc.tiles` to stage them on a node with network
access). For ``source="local"`` a
two-window model needs either ``stacked_windows=True`` with a raster whose
bands 1-4 and 5-8 are the two windows (R, G, B, NIR each), or
``allow_annual_fallback=True`` (the single raster is used for both windows).

Single-window models use the input raster.

Radiometry
----------
ftw-tools' default preprocessing divides the input by 3000, i.e. it expects
Sentinel-2 L2A surface reflectance x 10000. Sentinel-2, Landsat and HLS
composites are already on that scale after the 1.0 harmonisation and are used
unchanged (Landsat and HLS are nevertheless outside the Sentinel-2 training
distribution: a WARNING is logged and ``engine_meta["out_of_distribution_source"]``
is True); other value scales are converted with
:func:`agribound.io.raster.to_s2_dn`. ``local`` rasters need
``engine_params["value_scale"]``. NaN, infinite and declared nodata values
are replaced with 0 before the input raster is written
(:func:`write_ftw_input`).

Polygonisation
--------------
``polygonize`` applies ``simplify`` and the morphology options in the units of
the prediction raster's CRS, and computes areas (``min_size``) in those units
when they are metres. A prediction raster in a geographic CRS, a CRS whose
linear unit is not the metre (e.g. US survey feet) or a Mercator/Web Mercator
CRS is therefore reprojected (nearest neighbour) to the UTM zone of the
study-area centre first (ftw-baselines issue #271;
:func:`metric_reprojection_reason`), so ``simplify`` and ``min_size`` are in
metres and m² of UTM or of the raster's own metric projection (other metric
projections, e.g. Albers, keep their small scale distortion).
``polygonize`` processes the mask in windows of ``polygonization_stride``
pixels (default 2048); fields crossing a window edge are split there unless
``merge_adjacent`` is set. ``close_interiors`` (default True) fills all holes;
on ftw-tools builds whose ``polygonize`` cannot close the interiors of a
MultiPolygon (2.0.0b5), combining it with ``erode_dilate`` or
``dilate_erode`` raises :class:`ValueError` before any input is built.

Engine parameters (all optional)
--------------------------------
``model``, ``checkpoint_path``, ``window_days`` (30), ``window_dates``,
``allow_annual_fallback`` (False), ``stacked_windows`` (False),
``value_scale`` (local rasters), ``resize_factor`` (2), ``patch_size``
(default: :func:`select_patch_size`), ``batch_size`` (2), ``padding``
(ftw-tools default),
``softmax_threshold`` (polygonise field probability >= threshold instead of
the arg-max class; implies ``save_scores``), ``save_scores`` (only together
with ``softmax_threshold``), ``simplify`` (metres, 0; the pipeline simplifies
later with ``config.simplify_tolerance``), ``close_interiors`` (True),
``merge_adjacent`` (None), ``polygonization_stride`` (2048), ``max_size``
(m², None), ``erode_dilate``, ``dilate_erode``, ``erode_dilate_raster``,
``dilate_erode_raster`` (0) and ``thin_boundaries`` (False).

``gdf.attrs["engine_meta"]`` records the backend (``"ftw-tools"``), the
ftw-tools version, the model key, the checkpoint URL, the path and SHA-256 of
the checkpoint file (for registry models, the copy ftw-tools caches under
``torch.hub.get_dir()/checkpoints``), the window dates and how they were
chosen, and the input units.

References
----------
Kerner, H., et al. (2025). Fields of The World: A Machine Learning Benchmark
Dataset for Global Agricultural Field Boundary Segmentation. Proceedings of
the AAAI Conference on Artificial Intelligence 39(27), 28151-28159.
doi:10.1609/aaai.v39i27.35034.

Muhawenayo, G., et al. (2026). PRUE: A Practical Recipe for Field Boundary
Segmentation at Scale. arXiv:2603.27101.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine, get_canonical_band_indices
from agribound.registry import ENGINE_REGISTRY

logger = logging.getLogger(__name__)

_REGISTRY_ENTRY = ENGINE_REGISTRY["ftw"]

#: Polygonize keyword arguments that may be set through engine_params.
_POLYGONIZE_PARAMS = (
    "close_interiors",
    "merge_adjacent",
    "polygonization_stride",
    "max_size",
    "erode_dilate",
    "dilate_erode",
    "erode_dilate_raster",
    "dilate_erode_raster",
    "thin_boundaries",
)

_RGBN = ["R", "G", "B", "NIR"]

#: Supported sources that are not Sentinel-2 (the FTW training data).
_OUT_OF_DISTRIBUTION_SOURCES = ("landsat", "hls")

#: Version of :func:`write_ftw_input`'s conversion, part of the cache keys of
#: FTW input rasters (2: declared nodata values are set to 0).
FTW_INPUT_VERSION = "2"


def _model_registry() -> dict[str, Any]:
    try:
        from ftw_tools.inference.model_registry import MODEL_REGISTRY
    except ImportError:
        raise ImportError(
            "ftw-tools (>= 2.0.0b5) is required for the FTW engine. Install with: "
            "pip install 'agribound[ftw]'"
        ) from None
    return MODEL_REGISTRY


def _ftw_version() -> str | None:
    try:
        import importlib.metadata as md

        return md.version("ftw-tools")
    except Exception:
        return None


def list_ftw_models(include_legacy: bool = False) -> dict[str, dict]:
    """List the models in the installed ftw-tools ``MODEL_REGISTRY``.

    Parameters
    ----------
    include_legacy : bool
        Include models marked legacy (FTW v1/v2 checkpoints; default False).

    Returns
    -------
    dict[str, dict]
        Model name -> ``{"url", "title", "description", "license", "version",
        "requires_window", "requires_polygonize", "instance_segmentation",
        "default", "legacy"}``. Instance-segmentation entries
        (Delineate-Anything) are listed but are run by the
        ``delineate-anything`` engine, not by this one.

    Raises
    ------
    ImportError
        If ftw-tools is not installed.

    Examples
    --------
    >>> from agribound.engines.ftw import list_ftw_models
    >>> for name, info in list_ftw_models().items():
    ...     print(f"{name}: {info['title']}")
    """
    models = {}
    for name, spec in _model_registry().items():
        if not include_legacy and spec.legacy:
            continue
        models[name] = {
            "url": spec.url,
            "title": spec.title,
            "description": spec.description,
            "license": spec.license,
            "version": spec.version,
            "requires_window": spec.requires_window,
            "requires_polygonize": spec.requires_polygonize,
            "instance_segmentation": spec.instance_segmentation,
            "default": spec.default,
            "legacy": spec.legacy,
        }
    return models


def default_ftw_model() -> str:
    """Return the ``MODEL_REGISTRY`` key marked ``default=True``.

    Raises
    ------
    RuntimeError
        If the registry does not have exactly one default entry.
    """
    defaults = [name for name, spec in _model_registry().items() if spec.default]
    if len(defaults) != 1:
        raise RuntimeError(
            f"Expected exactly one default model in the ftw-tools MODEL_REGISTRY, found "
            f"{defaults}; pass engine_params['model'] explicitly."
        )
    return defaults[0]


def checkpoint_in_channels(path: str | Path) -> int:
    """Read ``hyper_parameters["in_channels"]`` from an FTW Lightning checkpoint.

    The checkpoint is unpickled with ``torch.load(weights_only=False)``, as
    ftw-tools itself loads it; only load checkpoints you trust.
    """
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters") if isinstance(ckpt, dict) else None
    if not isinstance(hparams, dict) or "in_channels" not in hparams:
        raise ValueError(
            f"{path} has no hyper_parameters['in_channels']; it is not an FTW (Lightning) "
            "checkpoint that ftw-tools can load."
        )
    return int(hparams["in_channels"])


@dataclass(frozen=True)
class FTWModelChoice:
    """The model an FTW run uses (see :func:`resolve_ftw_model`)."""

    run_model: str
    """Value passed to ``ftw_tools.inference.inference.run(model=...)``."""
    registry_key: str | None
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    in_channels: int | None
    n_windows: int
    url: str | None
    license: str | None
    version: str | None
    cache_id: str
    """Identifier used in cache keys (never a filesystem path)."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SHA256_MEMO: dict[tuple[str, int, int], str] = {}


def registry_checkpoint_facts(registry_key: str) -> dict[str, str]:
    """``checkpoint_path`` and ``checkpoint_sha256`` of a cached registry checkpoint.

    ftw-tools' ``run()`` loads registry models from
    ``torch.hub.get_dir()/checkpoints/<key>.ckpt`` (downloading it only when
    missing), so hashing that file pins the weights a run used. Empty when
    torch is unavailable or the file does not exist. The digest is memoised
    per process by path, size and modification time.
    """
    try:
        import torch

        path = Path(torch.hub.get_dir()) / "checkpoints" / f"{registry_key}.ckpt"
        st = path.stat()
    except Exception:
        return {}
    memo_key = (str(path), st.st_size, st.st_mtime_ns)
    if memo_key not in _SHA256_MEMO:
        _SHA256_MEMO[memo_key] = _sha256(path)
    return {"checkpoint_path": str(path), "checkpoint_sha256": _SHA256_MEMO[memo_key]}


def resolve_ftw_model(engine_params: dict[str, Any] | None) -> FTWModelChoice:
    """Resolve the FTW model from ``checkpoint_path`` / ``model`` / the registry default.

    Raises
    ------
    FileNotFoundError
        If ``checkpoint_path`` does not exist.
    ValueError
        For unknown registry keys, instance-segmentation registry entries,
        non-``.ckpt`` files, or checkpoints whose ``in_channels`` is not 4
        or 8.
    """
    params = engine_params or {}
    checkpoint = params.get("checkpoint_path")
    model = params.get("model")
    if checkpoint is None and model is not None and str(model).endswith(".ckpt"):
        checkpoint = model  # a checkpoint path passed as the model (ftw-tools accepts both)
    if checkpoint is not None:
        path = Path(str(checkpoint)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"FTW checkpoint not found: {path}")
        if path.suffix != ".ckpt":
            raise ValueError(f"ftw-tools loads only .ckpt checkpoints, got {path.name}")
        in_channels = checkpoint_in_channels(path)
        if in_channels not in (4, 8):
            raise ValueError(
                f"{path.name} has in_channels={in_channels}; agribound builds 4-band (one "
                "window of R, G, B, NIR) or 8-band (two windows) FTW inputs."
            )
        sha = _sha256(path)
        return FTWModelChoice(
            run_model=str(path),
            registry_key=None,
            checkpoint_path=str(path),
            checkpoint_sha256=sha,
            in_channels=in_channels,
            n_windows=in_channels // 4,
            url=None,
            license=None,
            version=None,
            cache_id=f"ckpt-{path.stem}-{sha[:12]}",
        )

    registry = _model_registry()
    key = str(model) if model is not None else default_ftw_model()
    spec = registry.get(key)
    if spec is None:
        raise ValueError(
            f"Unknown FTW model {key!r}. Available: {sorted(registry)} "
            "(see 'agribound list-ftw-models --all')."
        )
    if spec.instance_segmentation:
        raise ValueError(
            f"{key!r} is an instance-segmentation model; run it with "
            "engine='delineate-anything' (engine_params {'backend': 'ftw'} to use ftw-tools, "
            "or the default native backend)."
        )
    return FTWModelChoice(
        run_model=key,
        registry_key=key,
        checkpoint_path=None,
        checkpoint_sha256=None,
        in_channels=8 if spec.requires_window else 4,
        n_windows=2 if spec.requires_window else 1,
        url=spec.url,
        license=spec.license,
        version=spec.version,
        cache_id=key,
    )


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


def crop_calendar_centres(bbox: tuple[float, float, float, float], year: int) -> dict[str, Any]:
    """Window centres from FTW's summer-crop calendar.

    Parameters
    ----------
    bbox : tuple
        ``(minx, miny, maxx, maxy)`` in EPSG:4326.
    year : int
        Year of the start of season.

    Returns
    -------
    dict
        ``sos_doy``, ``eos_doy``, ``centre_a``/``centre_b`` (ISO dates),
        ``year_b`` and ``rollover`` (True when EOS < SOS, so window B is in
        ``year + 1``).

    Raises
    ------
    ValueError
        If the crop calendar has no value under *bbox* (e.g. over water), with
        a pointer to ``engine_params["window_dates"]``.
    """
    from ftw_tools.utils import get_harvest_integer_from_bbox, harvest_to_datetime

    # No calendar value under the bbox: NaN -> int raises ValueError; a bbox
    # outside the calendar raster raises rioxarray's NoDataInBounds.
    no_data_errors: tuple[type[Exception], ...] = (ValueError,)
    try:
        from rioxarray.exceptions import NoDataInBounds

        no_data_errors = (ValueError, NoDataInBounds)
    except ImportError:
        pass
    try:
        sos, eos = get_harvest_integer_from_bbox(bbox=[float(v) for v in bbox])
    except no_data_errors as exc:
        raise ValueError(
            f"FTW's crop calendar has no season dates for bbox {tuple(bbox)} ({exc}). Pass "
            "engine_params['window_dates'] = ['YYYY-MM-DD', 'YYYY-MM-DD'] (window centres)."
        ) from exc
    rollover = eos < sos
    year_b = year + 1 if rollover else year
    centre_a = harvest_to_datetime(sos, year).date()
    centre_b = harvest_to_datetime(eos, year_b).date()
    return {
        "sos_doy": int(sos),
        "eos_doy": int(eos),
        "centre_a": centre_a.isoformat(),
        "centre_b": centre_b.isoformat(),
        "year_b": int(year_b),
        "rollover": bool(rollover),
    }


def window_range(centre: str | _dt.date, days: int) -> tuple[str, str]:
    """Return ``(centre - days, centre + days)`` as ISO date strings."""
    if not isinstance(centre, _dt.date):
        centre = _dt.date.fromisoformat(str(centre))
    delta = _dt.timedelta(days=int(days))
    return (centre - delta).isoformat(), (centre + delta).isoformat()


def _aoi_bounds_4326(config: AgriboundConfig, raster_path: str) -> tuple[float, ...]:
    """EPSG:4326 bounds of the study area (its local copy for a GEE asset), else of the raster."""
    if config.study_area:
        from agribound.io.vector import read_config_study_area

        aoi = read_config_study_area(config)
        if aoi.crs is None:
            aoi = aoi.set_crs("EPSG:4326")
        return tuple(float(v) for v in aoi.to_crs("EPSG:4326").total_bounds)
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(raster_path) as src:
        return tuple(float(v) for v in transform_bounds(src.crs, "EPSG:4326", *src.bounds))


def _window_centres(
    config: AgriboundConfig, raster_path: str, params: dict[str, Any]
) -> dict[str, Any]:
    dates = params.get("window_dates")
    if dates is not None:
        if not isinstance(dates, list | tuple) or len(dates) != 2:
            raise ValueError("engine_params['window_dates'] must be two 'YYYY-MM-DD' strings")
        a, b = (_dt.date.fromisoformat(str(d)) for d in dates)
        if b <= a:
            raise ValueError(f"window_dates must be in season order (A before B), got {dates}")
        return {"method": "engine_params", "centre_a": a.isoformat(), "centre_b": b.isoformat()}
    bbox = _aoi_bounds_4326(config, raster_path)
    info = crop_calendar_centres(bbox, int(config.year))
    return {"method": "ftw_crop_calendar", "bbox_4326": list(bbox), **info}


def _build_window(
    config: AgriboundConfig,
    label: str,
    start: str,
    end: str,
    raster_path: str,
    allow_fallback: bool,
) -> tuple[str, str, str | None]:
    """Build one window composite; return ``(path, status, error)``.

    Composite builders raise :class:`~agribound.composites.base.NoDataError`
    (a :class:`ValueError`) when the window has no imagery over the study
    area (no image matches the filters, or no valid pixel). Only that error
    is treated as "no imagery in the window": it raises a
    :class:`RuntimeError` with guidance (the :class:`NoDataError` is chained
    as ``__cause__``), or, with *allow_fallback*, returns the annual
    composite. Any other error (invalid configuration, authentication,
    quota, network, ...) propagates unchanged, also with *allow_fallback*,
    so it never swaps in the annual composite.
    """
    from agribound.composites import NoDataError, get_composite_builder

    window_config = config.merged(date_range=(start, end), composite_method="median")
    try:
        path = str(get_composite_builder(config.source).build(window_config))
        return path, "composite", None
    except NoDataError as exc:
        error = f"{type(exc).__name__}: {exc}"
        if not allow_fallback:
            raise RuntimeError(
                f"FTW window {label} ({start} to {end}): no {config.source} composite could be "
                f"built ({error}). Two-window FTW models need imagery in both windows. "
                "Choose other windows with engine_params['window_dates'] or 'window_days', or "
                "set engine_params['allow_annual_fallback']=True to use the annual composite "
                "for a missing window (not seasonal; recorded in engine_meta)."
            ) from exc
        logger.warning(
            "FTW window %s (%s to %s) failed (%s); using the annual composite %s instead "
            "(allow_annual_fallback=True)",
            label,
            start,
            end,
            error,
            raster_path,
        )
        return raster_path, "annual_fallback", error


#: Composite tags copied into each window's ``engine_meta["windows"]`` record.
_WINDOW_TAGS = {
    "AGRIBOUND_N_IMAGES": "n_images",
    "AGRIBOUND_VALID_FRACTION": "valid_fraction",
    "AGRIBOUND_CLOUD_MASK": "cloud_mask",
    "AGRIBOUND_DATE_START": "composite_date_start",
    "AGRIBOUND_DATE_END_EXCLUSIVE": "composite_date_end_exclusive",
}


def _window_composite_facts(path: str) -> dict[str, Any]:
    """Image count, valid fraction and cloud mask of a window composite (from its tags)."""
    try:
        import rasterio

        with rasterio.open(path) as src:
            tags = src.tags()
    except Exception:  # informative only
        return {}
    facts: dict[str, Any] = {}
    for tag, key in _WINDOW_TAGS.items():
        if tag not in tags:
            continue
        value: Any = tags[tag]
        with contextlib.suppress(ValueError):
            if key == "n_images":
                value = int(value)
            elif key == "valid_fraction":
                value = float(value)
        facts[key] = value
    return facts


def write_ftw_input(
    out_path: str | Path,
    sources: list[tuple[str, list[int]]],
    source: str,
    value_scale: str | None = None,
    block_rows: int = 512,
) -> str:
    """Write a float32 FTW input stack from ``(raster, band indices)`` pairs.

    Bands are written in the order given (e.g. window A R, G, B, NIR then
    window B), converted with :func:`agribound.io.raster.to_s2_dn` (values
    already on the ``reflectance_x10000`` scale are copied unchanged). Values
    equal to a raster's declared nodata value (e.g. -9999 or 65535) and NaN
    and infinite values are replaced by 0 before the conversion (0 is also
    the default ``nan_fill_value`` of ftw-tools' inference on ftw-baselines
    main). Rasters
    whose grid differs from the first one are resampled onto it (nearest
    neighbour; pixels outside them become nodata, hence 0). The raster is
    processed in strips of *block_rows* rows. The output has no nodata
    value.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    from agribound.io.raster import to_s2_dn
    from agribound.registry import source_value_scale

    scale = value_scale or source_value_scale(source)
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.stem + ".tmp.tif")
    handles = [rasterio.open(path) for path, _ in sources]
    try:
        ref = handles[0]
        readers = []
        for handle in handles:
            same = (
                handle.crs == ref.crs
                and handle.transform == ref.transform
                and (handle.width, handle.height) == (ref.width, ref.height)
            )
            readers.append(
                handle
                if same
                else WarpedVRT(
                    handle,
                    crs=ref.crs,
                    transform=ref.transform,
                    width=ref.width,
                    height=ref.height,
                    resampling=Resampling.nearest,
                )
            )
        count = sum(len(bands) for _, bands in sources)
        profile = {
            "driver": "GTiff",
            "width": ref.width,
            "height": ref.height,
            "count": count,
            "dtype": "float32",
            "crs": ref.crs,
            "transform": ref.transform,
            "nodata": None,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }
        with rasterio.open(tmp, "w", **profile) as dst:
            for row in range(0, ref.height, block_rows):
                window = Window(0, row, ref.width, min(block_rows, ref.height - row))
                parts = []
                for reader, (_, bands) in zip(readers, sources, strict=True):
                    raw = reader.read(bands, window=window)
                    data = raw.astype(np.float32)
                    missing = ~np.isfinite(data)
                    nodata = reader.nodata
                    if nodata is not None and np.isfinite(nodata):
                        missing |= raw == nodata
                    data[missing] = 0.0
                    if scale == "reflectance_x10000":
                        dn = data  # already S2 L2A units; avoid a float32 round trip
                    else:
                        dn = to_s2_dn(data, source, value_scale=value_scale)
                    parts.append(np.where(np.isfinite(dn), dn, 0).astype(np.float32))
                dst.write(np.concatenate(parts, axis=0), window=window)
        for reader, handle in zip(readers, handles, strict=True):
            if reader is not handle:
                reader.close()
    finally:
        for handle in handles:
            handle.close()
    tmp.replace(out_path)
    return str(out_path)


def select_patch_size(height: int, width: int, requested: int | None = None) -> int:
    """Patch size passed to ftw-tools, always smaller than the raster's smaller side.

    Without *requested*, this is ftw-tools' own rule (the largest of 1024,
    512, 256 and 128 that fits) except that the patch must be strictly
    smaller than ``min(height, width)``: with torchgeo 0.10,
    ``GridGeoSampler`` can return no patch when the patch spans the whole
    raster extent (the extent divided by a floating-point resolution falls
    just short of the pixel count), and ftw-tools then writes an
    all-background prediction without an error (observed with ftw-tools
    2.0.0b5 and torchgeo 0.10.0 on a 1024 x 1024 EPSG:4326 raster).

    Raises
    ------
    ValueError
        If *requested* is not a multiple of 32 or not smaller than the
        smaller raster side, or if the raster is 128 px or smaller.
    """
    smaller = min(int(height), int(width))
    if requested is not None:
        requested = int(requested)
        if requested % 32 or requested >= smaller:
            raise ValueError(
                f"patch_size must be a multiple of 32 and smaller than the raster's smaller "
                f"side ({smaller} px), got {requested}"
            )
        return requested
    for size in (1024, 512, 256, 128):
        if size < smaller:
            return size
    raise ValueError(f"The FTW input is too small ({height} x {width} px); it needs > 128 px")


def _fingerprint(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"


def metric_reprojection_reason(crs: Any) -> str | None:
    """Why a prediction raster in *crs* must be reprojected to UTM before polygonising.

    ftw-tools' ``polygonize`` applies ``simplify`` and the morphology
    options in CRS units, and computes areas in those units whenever the
    CRS's linear unit is the metre. Returns ``None`` for projected CRSs in
    metres that are not Mercator projections (e.g. UTM, the default export
    CRS), else a short reason: ``"geographic CRS"``, ``"linear unit <unit>"``
    (e.g. US survey feet) or ``"Mercator projection"`` (the normal Mercator
    variants and Web Mercator, whose "metres" are ground metres divided by
    cos(latitude); Transverse and Oblique Mercator are kept).
    """
    if crs is None:
        raise ValueError("The FTW prediction raster has no CRS")
    import pyproj

    crs_obj = pyproj.CRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
    if crs_obj.is_geographic:
        return "geographic CRS"
    unit = crs_obj.axis_info[0].unit_name if crs_obj.axis_info else ""
    factor = crs_obj.axis_info[0].unit_conversion_factor if crs_obj.axis_info else None
    if factor != 1.0:
        return f"linear unit {unit or 'unknown'}"
    method = crs_obj.coordinate_operation.method_name if crs_obj.coordinate_operation else ""
    method = method.lower()
    if method.startswith("mercator") or "pseudo mercator" in method:
        return "Mercator projection"
    return None


def _check_close_interiors(polygonize: Any, poly_kwargs: dict[str, Any]) -> None:
    """Reject ``close_interiors`` with vector morphology on ftw-tools builds that crash on it.

    ``erode_dilate``/``dilate_erode`` can split a field into a MultiPolygon;
    ftw-tools 2.0.0b5 then calls ``Polygon(geom.exterior)`` on it and fails
    with ``AttributeError``. ftw-baselines main handles MultiPolygons there;
    the check looks for that handling in ``polygonize``'s source.
    """
    morphology = any(float(poly_kwargs.get(k) or 0) > 0 for k in ("erode_dilate", "dilate_erode"))
    if not (morphology and poly_kwargs.get("close_interiors")):
        return
    try:
        source = inspect.getsource(polygonize)
    except (OSError, TypeError):
        source = ""
    if "Polygon(g.exterior)" in source:
        return
    raise ValueError(
        f"The installed ftw-tools ({_ftw_version()}) polygonize() fails with "
        "close_interiors=True when erode_dilate/dilate_erode split a field (MultiPolygon has "
        "no exterior). Set engine_params['close_interiors']=False, use erode_dilate_raster/"
        "dilate_erode_raster instead, or install ftw-baselines main."
    )


def _has_field_pixels(pred_path: str, softmax_threshold: float | None) -> bool:
    import rasterio

    with rasterio.open(pred_path) as src:
        for _, window in src.block_windows(1):
            if softmax_threshold is None:
                if np.any(src.read(1, window=window) == 1):
                    return True
            elif np.any(src.read(2, window=window) >= softmax_threshold * 255):
                return True
    return False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class FTWEngine(DelineationEngine):
    """Field boundary delineation with FTW semantic segmentation (see the module docstring)."""

    name = "ftw"
    supported_sources = list(_REGISTRY_ENTRY["supported_sources"])
    requires_bands = list(_REGISTRY_ENTRY["requires_bands"])

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run FTW inference and polygonisation.

        Parameters
        ----------
        raster_path : str
            Annual composite (or local raster) for the run.
        config : AgriboundConfig
            Pipeline configuration (``engine_params``: module docstring).

        Returns
        -------
        geopandas.GeoDataFrame
            Field polygons (in a projected CRS) with
            ``gdf.attrs["engine_meta"]``.
        """
        try:
            from ftw_tools.inference.inference import run as ftw_run
            from ftw_tools.postprocess.polygonize import polygonize as ftw_polygonize
        except ImportError:
            raise ImportError(
                "ftw-tools (>= 2.0.0b5) is required for the FTW engine. Install with: "
                "pip install 'agribound[ftw]'"
            ) from None
        from agribound._cache import cache_path
        from agribound.registry import source_value_scale

        self.validate_input(raster_path, config)
        params = dict(config.engine_params)
        choice = resolve_ftw_model(params)
        value_scale = params.get("value_scale") or source_value_scale(config.source)
        if value_scale in ("dn", "unknown"):
            raise ValueError(
                f"FTW needs Sentinel-2-like reflectance, but source {config.source!r} has value "
                f"scale {value_scale!r}. For a local raster set engine_params['value_scale'] to "
                "'reflectance_x10000', 'unit' (0-1 reflectance) or 'uint8'."
            )
        softmax_threshold = params.get("softmax_threshold")
        save_scores = bool(params.get("save_scores", softmax_threshold is not None))
        if softmax_threshold is not None:
            softmax_threshold = float(softmax_threshold)
            if not 0 < softmax_threshold < 1:
                raise ValueError(f"softmax_threshold must be in (0, 1), got {softmax_threshold}")
            save_scores = True
        elif save_scores:
            raise ValueError(
                "save_scores=True writes class probabilities, which ftw-tools polygonize only "
                "reads with a softmax_threshold; set engine_params['softmax_threshold']."
            )

        # Polygonize options are checked before any (GEE) input is built.
        poly_kwargs: dict[str, Any] = {
            "simplify": float(params.get("simplify", 0)),
            "min_size": float(config.min_field_area_m2),
            "close_interiors": bool(params.get("close_interiors", True)),
        }
        for key in _POLYGONIZE_PARAMS:
            if key in params and key != "close_interiors":
                poly_kwargs[key] = params[key]
        if softmax_threshold is not None:
            poly_kwargs["softmax_threshold"] = softmax_threshold
        accepted = inspect.signature(ftw_polygonize).parameters
        unknown = sorted(k for k in poly_kwargs if k not in accepted)
        if unknown:
            raise ValueError(f"The installed ftw-tools polygonize() does not accept {unknown}")
        _check_close_interiors(ftw_polygonize, poly_kwargs)

        rgbn = get_canonical_band_indices(config.source, _RGBN, bands=config.bands)
        out_of_distribution = config.source in _OUT_OF_DISTRIBUTION_SOURCES
        if out_of_distribution:
            logger.warning(
                "FTW checkpoints are trained on Sentinel-2 L2A; %s input (harmonised surface "
                "reflectance x 10000) is out of distribution and accuracy is not established.",
                config.source,
            )
        meta: dict[str, Any] = {
            "backend": "ftw-tools",
            "ftw_tools_version": _ftw_version(),
            "model": choice.registry_key or "checkpoint",
            "checkpoint_url": choice.url,
            "checkpoint_path": choice.checkpoint_path,
            "checkpoint_sha256": choice.checkpoint_sha256,
            "model_license": choice.license,
            "model_version": choice.version,
            "in_channels": choice.in_channels,
            "n_windows": choice.n_windows,
            "band_indices_rgbn": rgbn,
            "value_scale": value_scale,
            "out_of_distribution_source": out_of_distribution,
            "input_units": (
                "S2 L2A reflectance x10000 ("
                + (
                    "composite values copied unchanged"
                    if value_scale == "reflectance_x10000"
                    else f"converted from {value_scale} with agribound.io.raster.to_s2_dn"
                )
                + "; NaN, inf and declared nodata -> 0); ftw-tools divides by 3000"
            ),
        }

        # --- build the FTW input -------------------------------------------
        sources, meta["windows"] = self._input_sources(
            config, raster_path, params, choice.n_windows, rgbn
        )

        window_key = json.dumps(
            [(_fingerprint(path), bands) for path, bands in sources], sort_keys=True
        )
        ftw_input = cache_path(
            config,
            "ftw_input",
            ".tif",
            choice.n_windows,
            window_key,
            value_scale,
            FTW_INPUT_VERSION,
        )
        if not ftw_input.exists():
            write_ftw_input(ftw_input, sources, config.source, value_scale=value_scale)
        meta["ftw_input"] = str(ftw_input)

        # --- inference ------------------------------------------------------
        import rasterio

        device = config.resolve_device()
        with rasterio.open(ftw_input) as src:
            patch_size = select_patch_size(src.height, src.width, params.get("patch_size"))
        run_kwargs: dict[str, Any] = {
            "input": str(ftw_input),
            "model": choice.run_model,
            "resize_factor": int(params.get("resize_factor", 2)),
            "gpu": 0 if device == "cuda" else -1,
            "patch_size": patch_size,
            "batch_size": int(params.get("batch_size", 2)),
            "num_workers": config.n_workers,
            "padding": params.get("padding"),
            "overwrite": True,
            "mps_mode": device == "mps",
            "save_scores": save_scores,
        }
        if "nan_fill_value" in inspect.signature(ftw_run).parameters:
            run_kwargs["nan_fill_value"] = 0.0
        pred = cache_path(
            config,
            "ftw_pred",
            ".tif",
            choice.cache_id,
            # Registry weights are identified by their URL (a checkpoint by its SHA-256
            # in cache_id), so a registry update for the same key is not reused.
            choice.url,
            _fingerprint(ftw_input),
            json.dumps(
                {k: v for k, v in run_kwargs.items() if k not in ("input", "model", "num_workers")},
                sort_keys=True,
                default=str,
            ),
            # Preprocessing and patch stitching can change between ftw-tools releases.
            meta["ftw_tools_version"],
        )
        meta.update(
            {
                "device": device,
                "resize_factor": run_kwargs["resize_factor"],
                "patch_size": run_kwargs["patch_size"],
                "batch_size": run_kwargs["batch_size"],
                "save_scores": save_scores,
            }
        )
        if pred.exists():
            logger.info("Using cached FTW prediction: %s", pred)
            meta["cached_prediction"] = True
        else:
            logger.info("Running FTW inference (model=%s, device=%s)", meta["model"], device)
            # Written under a temporary name and renamed, so an interrupted run
            # never leaves a partial raster that a later run would take as cached.
            partial = pred.with_name(pred.stem + ".partial.tif")
            ftw_run(out=str(partial), **run_kwargs)
            if not partial.exists():
                raise RuntimeError(f"FTW inference did not write the prediction raster {partial}")
            os.replace(partial, pred)
        if choice.registry_key is not None:
            # Pin the exact registry weights: ftw-tools loads the cached file.
            meta.update(registry_checkpoint_facts(choice.registry_key))

        # --- polygonise in metres -------------------------------------------
        poly_input = str(pred)
        with rasterio.open(pred) as src:
            pred_crs = src.crs
        reason = metric_reprojection_reason(pred_crs)
        if reason is not None:
            from shapely.geometry import box

            from agribound.io.crs import reproject_raster, utm_crs_for_geometry

            utm = utm_crs_for_geometry(box(*_aoi_bounds_4326(config, raster_path)))
            target = pred.with_name(pred.stem + f"_epsg{utm.to_epsg()}.tif")
            if not target.exists():
                partial = target.with_name(target.stem + ".partial.tif")
                reproject_raster(pred, partial, utm, resampling="nearest")
                os.replace(partial, target)
            poly_input = str(target)
            meta["prediction_reprojected_to"] = f"EPSG:{utm.to_epsg()}"
            meta["prediction_reprojection_reason"] = reason
        meta["prediction_crs"] = str(pred_crs)
        meta["polygonize"] = {
            ("simplify_m" if key == "simplify" else key): value
            for key, value in poly_kwargs.items()
        }

        if not _has_field_pixels(poly_input, softmax_threshold):
            logger.warning("FTW predicted no field pixels")
            with rasterio.open(poly_input) as src:
                crs = src.crs
            gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
            meta["n_output"] = 0
            gdf.attrs["engine_meta"] = meta
            return gdf

        poly_path = cache_path(
            config, "ftw_polygons", ".gpkg", _fingerprint(poly_input), sorted(poly_kwargs.items())
        )
        ftw_polygonize(input=poly_input, out=str(poly_path), overwrite=True, **poly_kwargs)
        if not poly_path.exists():
            raise RuntimeError(f"FTW polygonization did not write {poly_path}")
        gdf = gpd.read_file(poly_path)
        meta["n_output"] = len(gdf)
        logger.info("FTW delineated %d field polygons", len(gdf))
        gdf.attrs["engine_meta"] = meta
        return gdf

    @staticmethod
    def stage_inputs(config: AgriboundConfig, raster_path: str) -> dict[str, Any]:
        """Build (or reuse from the cache) the input rasters :meth:`delineate` reads.

        Runs the input stage of :meth:`delineate` for *config* and
        *raster_path* without running inference: the model is resolved from
        ``engine_params`` with :func:`resolve_ftw_model`, as in
        :meth:`delineate` (ftw-tools' model registry for registry models, the
        checkpoint's channel count for ``checkpoint_path``), and for a
        two-window model on an Earth Engine
        source both seasonal window composites are built by the source's
        composite builder with the same window centres, ``window_days`` and
        cache keys, so a later :meth:`delineate` with the same configuration
        and cache directory finds them without network access (the crop
        calendar comes from ftw-tools' cache; see :meth:`prefetch`).
        Single-window models and local sources build nothing.

        Parameters
        ----------
        config : AgriboundConfig
            Pipeline configuration.
        raster_path : str
            Annual composite (or local raster) of the run.

        Returns
        -------
        dict
            ``n_windows`` (1 or 2), ``band_indices_rgbn`` (1-based R, G, B, NIR
            indices read from each raster), ``windows`` (the record stored in
            ``engine_meta["windows"]``: ``"single"``, or ``"a"``/``"b"`` with
            ``start``, ``end``, ``raster`` and ``status``, plus the window
            ``centres`` for Earth Engine sources) and ``rasters`` (the raster
            of each window in input order, A then B).

        Raises
        ------
        RuntimeError
            If a window has no imagery and ``allow_annual_fallback`` is not
            set (the builder's :class:`~agribound.composites.base.NoDataError`
            is its ``__cause__``).
        ValueError
            For invalid window parameters or band mappings.
        """
        params = dict(config.engine_params)
        choice = resolve_ftw_model(params)
        rgbn = get_canonical_band_indices(config.source, _RGBN, bands=config.bands)
        sources, windows = FTWEngine._input_sources(
            config, raster_path, params, choice.n_windows, rgbn
        )
        return {
            "n_windows": int(choice.n_windows),
            "band_indices_rgbn": list(rgbn),
            "windows": windows,
            "rasters": [str(path) for path, _bands in sources],
        }

    @staticmethod
    def _input_sources(
        config: AgriboundConfig,
        raster_path: str,
        params: dict[str, Any],
        n_windows: int,
        rgbn: list[int],
    ) -> tuple[list[tuple[str, list[int]]], dict[str, Any]]:
        """``(sources, windows record)`` of the FTW input for *n_windows*."""
        if n_windows == 1:
            record = {"single": {"raster": raster_path, "status": "input raster"}}
            return [(raster_path, rgbn)], record
        if config.is_gee_source():
            return FTWEngine._two_windows(config, raster_path, params, rgbn)
        return FTWEngine._two_windows_local(raster_path, params, rgbn)

    @staticmethod
    def _two_windows(
        config: AgriboundConfig, raster_path: str, params: dict[str, Any], rgbn: list[int]
    ) -> tuple[list[tuple[str, list[int]]], dict[str, Any]]:
        days = int(params.get("window_days", 30))
        if days < 1:
            raise ValueError(f"window_days must be >= 1, got {days}")
        allow_fallback = bool(params.get("allow_annual_fallback", False))
        centres = _window_centres(config, raster_path, params)
        record: dict[str, Any] = {"centres": centres, "window_days": days}
        sources = []
        for label in ("a", "b"):
            start, end = window_range(centres[f"centre_{label}"], days)
            path, status, error = _build_window(
                config, label.upper(), start, end, raster_path, allow_fallback
            )
            record[label] = {"start": start, "end": end, "raster": path, "status": status}
            record[label].update(_window_composite_facts(path))
            if error:
                record[label]["error"] = error
            sources.append((path, rgbn))
        logger.info(
            "FTW windows: A %s..%s (%s), B %s..%s (%s)",
            record["a"]["start"],
            record["a"]["end"],
            record["a"]["status"],
            record["b"]["start"],
            record["b"]["end"],
            record["b"]["status"],
        )
        return sources, record

    @staticmethod
    def _two_windows_local(
        raster_path: str, params: dict[str, Any], rgbn: list[int]
    ) -> tuple[list[tuple[str, list[int]]], dict[str, Any]]:
        import rasterio

        if params.get("stacked_windows"):
            with rasterio.open(raster_path) as src:
                if src.count < 8:
                    raise ValueError(
                        f"stacked_windows=True needs at least 8 bands, {raster_path} has "
                        f"{src.count}"
                    )
            return [(raster_path, [1, 2, 3, 4]), (raster_path, [5, 6, 7, 8])], {
                "a": {"raster": raster_path, "bands": [1, 2, 3, 4], "status": "stacked"},
                "b": {"raster": raster_path, "bands": [5, 6, 7, 8], "status": "stacked"},
            }
        if params.get("allow_annual_fallback"):
            logger.warning(
                "Local raster %s is used for both FTW windows (allow_annual_fallback=True); the "
                "input is not seasonal.",
                raster_path,
            )
            return [(raster_path, rgbn), (raster_path, rgbn)], {
                "a": {"raster": raster_path, "status": "annual_fallback"},
                "b": {"raster": raster_path, "status": "annual_fallback"},
            }
        raise ValueError(
            "This FTW model needs two seasonal windows, which cannot be built from a local "
            "raster. Provide an 8-band raster (window A R, G, B, NIR; window B R, G, B, NIR) "
            "with engine_params['stacked_windows']=True, use a single-window model "
            "(e.g. model='FTW_v2_3_Class_FULL_singleWindow'), or set "
            "engine_params['allow_annual_fallback']=True to use the raster for both windows."
        )

    @classmethod
    def prefetch(cls, config: AgriboundConfig) -> list[str]:
        """Download the FTW checkpoint (and the crop calendar for two-window models).

        Registry checkpoints are saved where ftw-tools' ``run()`` looks for
        them (``torch.hub.get_dir()/checkpoints/<model>.ckpt``); the crop
        calendar goes to ``$FTW_CACHE_DIR/crop_calendar`` (default
        ``~/.cache/ftw-tools``). Set ``TORCH_HOME`` and ``FTW_CACHE_DIR`` to
        shared storage on HPC systems.

        Returns
        -------
        list[str]
            Local paths of the checkpoint and crop-calendar files.
        """
        choice = resolve_ftw_model(config.engine_params)
        paths: list[str] = []
        if choice.checkpoint_path:
            paths.append(choice.checkpoint_path)
        else:
            import torch

            target = Path(torch.hub.get_dir()) / "checkpoints" / f"{choice.registry_key}.ckpt"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                torch.hub.download_url_to_file(choice.url, str(target), progress=True)
            paths.append(str(target))
        if choice.n_windows == 2 and config.engine_params.get("window_dates") is None:
            from ftw_tools.download.crop_calendar import ensure_crop_calendar_exists
            from ftw_tools.settings import CROP_CAL_SUMMER_END, CROP_CAL_SUMMER_START

            calendar_dir = ensure_crop_calendar_exists()
            paths += [
                str(calendar_dir / CROP_CAL_SUMMER_START),
                str(calendar_dir / CROP_CAL_SUMMER_END),
            ]
        return paths
