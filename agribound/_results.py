"""
Results versions: releases that change a component's output for the same configuration.

:func:`agribound.provenance.config_hash` covers the configuration, not the
code. When a release changes what a component produces for an unchanged
configuration, its number in :data:`RESULTS_VERSIONS` is raised. The pipeline
records the numbers that apply to a run (:func:`results_versions`) in the
provenance fact ``results_versions``, and an existing output is reused only
when its recorded numbers equal the current ones
(:func:`agribound.provenance.reuse_mismatch`). A record without the fact, or
without a component, counts as version 1 for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

RESULTS_VERSIONS: dict[str, int] = {
    "embedding": 2,
    "sam_refine": 2,
    "delineate_anything_finetune": 3,
    "delineate_anything": 2,
    "dinov3_finetune": 2,
    "dinov3": 2,
    "dinov3_processing_grid": 2,
    "sam_auto": 1,
}
"""Version of each component's results; 1 = agribound <= 1.0.0.

Bump a component's number when its output changes for the same
configuration, so that outputs of earlier releases are no longer reused.

- ``"embedding"``: the embedding engine's clustering (2: agribound 1.0.1).
- ``"sam_refine"``: SAM refinement (2: agribound 1.0.1).
- ``"delineate_anything_finetune"``: fine-tuning Delineate-Anything (2: the
  bias warmup learning rate of AdamW is 0, as in Ultralytics'
  ``optimizer="auto"``; agribound <= 1.0.1 used 0.1. 3: the training chips
  overlap, half a chip apart, and are split by chip-size cells; the optimizer
  steps after every batch (``nbs`` equal to the batch size); label parts that
  the chip frame cuts to a sliver are dropped; and pixels that are 0 in every
  band are nodata in the chips, as in the native backend).
- ``"delineate_anything"``: the native backend of Delineate-Anything, which
  combines the detections of overlapping tiles (2: a detection is tile-cut
  within 4 px of an interior tile edge, overlaps go to complete detections
  first, and tile-cut pieces that meet at a tile edge can be joined
  (``merge_tile_seams``); agribound <= 1.0.1 needed the outermost pixel,
  gave overlaps to the higher confidence and joined only overlapping pieces).
- ``"dinov3_finetune"``: fine-tuning DINOv3 (2: inverse-frequency class weights,
  the checkpoint with the best validation mean IoU instead of the lowest
  validation loss, the backbone's input normalisation, and a 4 px boundary
  band; agribound <= 1.0.1 trained with none of these and a 2 px band).
- ``"dinov3"``: DINOv3 inference (2: the class probabilities of overlapping
  windows are averaged with a cosine taper; agribound <= 1.0.1 kept each
  window's prediction up to the middle of its overlaps, geoai's rule).
- ``"dinov3_processing_grid"``: DINOv3 on composites with pixels coarser than
  10 m (2: fine-tuning and inference read a bilinear copy with 10 m pixels,
  ``engine_params["processing_gsd_m"]``; agribound <= 1.0.1 read every
  raster as it is). It applies only to the runs that read such a copy, so
  the results versions of DINOv3 runs at 10 m or finer do not change.
- ``"sam_auto"``: engine ``"sam-auto"`` (1: agribound 1.1.0, which adds the
  engine). Its output also depends on the device, which the configuration
  hash leaves out (:mod:`agribound.engines.sam_auto`).

A record without an entry for a component counts as version 1, so a
component added later starts at 1 when its output is unchanged, or at 2
when the release that adds it also changes its output; :func:`results_versions`
must say when it applies.
"""


def _ensemble_members(config: Any) -> list[tuple[str, dict]]:
    """``(engine name, engine_params)`` of the configured ensemble members (malformed ignored)."""
    members = (getattr(config, "engine_params", None) or {}).get("engines")
    if not isinstance(members, list | tuple):
        from agribound.registry import ENSEMBLE_DEFAULT_MEMBERS

        return [(name, {}) for name in ENSEMBLE_DEFAULT_MEMBERS]
    out = []
    for entry in members:
        name = entry.get("engine") if isinstance(entry, dict) else entry
        params = entry.get("engine_params") if isinstance(entry, dict) else None
        if isinstance(name, str):
            out.append((name.lower().strip(), params if isinstance(params, dict) else {}))
    return out


def _ensemble_member_engines(config: Any) -> list[str]:
    """Engine names of the configured ensemble members (malformed entries ignored)."""
    return [name for name, _ in _ensemble_members(config)]


def _runs_engine(config: Any, name: str) -> bool:
    """True if *config* runs engine *name*, directly or as an ensemble member."""
    engine = str(getattr(config, "engine", "") or "").lower().strip()
    return engine == name or (engine == "ensemble" and name in _ensemble_member_engines(config))


def _native_delineate_anything(engine: str, engine_params: dict) -> bool:
    """True if *engine* is Delineate-Anything with its native backend (the default)."""
    backend = str(engine_params.get("backend", "native")).strip().lower()
    return engine == "delineate-anything" and backend == "native"


def _export_metres_per_unit(config: Any) -> float | None:
    """Metres per unit of the export CRS of *config* (1 for ``"utm"``; None if geographic).

    The export grids are laid out in the CRS's units
    (:func:`agribound.composites.gee.compute_export_grid`), so in a projected
    CRS in US survey feet a 30 m source has pixels of 30 ft (9.14 m).
    """
    text = str(getattr(config, "export_crs", None) or "utm").strip()
    if text.lower() == "utm":
        return 1.0
    import pyproj

    try:
        crs = pyproj.CRS.from_user_input(text)
    except pyproj.exceptions.CRSError:
        return 1.0  # the configuration rejects it
    if not crs.is_projected:
        return None
    return float(crs.axis_info[0].unit_conversion_factor)


def _composite_pixel_sizes_m(config: Any) -> tuple[float, float] | None:
    """Smallest and largest pixel size in metres that DINOv3 can measure on the composite.

    For a source exported by agribound: its export resolution
    (``naip_resolution_m`` for ``naip``, ``vhr_sar_resolution_m`` for
    ``umbra`` and ``capella``, else ``resolution_m`` of
    :data:`agribound.registry.SOURCE_REGISTRY`), in metres in a projected
    ``export_crs`` in other units (the grid is laid out in the CRS's units,
    :func:`_export_metres_per_unit`); in an ``export_crs`` in degrees the
    resolution itself. None for ``usgs-naip-plus``, whose composite has the
    finest resolution of its footprints, about 1 m or finer. Both sizes are
    equal.

    For ``local`` the sizes are read from ``local_tif_path`` with the
    engine's measure (the composite is a crop of it): in a geographic CRS the
    range over the latitudes of the file, because the engine measures a
    crop at its own centre
    (:func:`agribound.engines.finetune._data._grid_pixel_size_range_m`).
    None when the file cannot be read.
    """
    from agribound.registry import SOURCE_REGISTRY, VHR_SAR_SOURCES

    source = str(getattr(config, "source", "") or "").lower().strip()
    if source == "local":
        path = getattr(config, "local_tif_path", None)
        if not path:
            return None
        import rasterio
        from rasterio.errors import RasterioError

        from agribound.engines.finetune._data import _grid_pixel_size_range_m

        try:
            with rasterio.open(Path(path).expanduser()) as src:
                return _grid_pixel_size_range_m(src)
        except (OSError, RasterioError):
            return None
    if source == "naip":
        size = float(getattr(config, "naip_resolution_m", 1.0))
    elif source in VHR_SAR_SOURCES:
        size = float(getattr(config, "vhr_sar_resolution_m", 1.0))
    else:
        resolution = (SOURCE_REGISTRY.get(source) or {}).get("resolution_m")
        if resolution is None:
            return None
        size = float(resolution)
    factor = _export_metres_per_unit(config)
    if factor is not None and factor != 1.0:
        size *= factor
    return size, size


def _dinov3_reads_a_grid(config: Any, params: dict, fine_tuned: bool) -> bool:
    """True if DINOv3 with *params* can read the composite of *config* on a processing grid.

    The rule is :func:`agribound.engines.finetune._data.dinov3_grid_gsd` on
    the sizes of :func:`_composite_pixel_sizes_m`: True when it gives a grid
    (or raises) at either of them. Its "no grid" sizes form one interval, so
    for a local raster in a geographic CRS this is True whenever a crop at
    some latitude of the file would be read on a grid. A run that loads a
    checkpoint without ``params["processing_gsd_m"]`` reads the grid
    recorded with the checkpoint (none for checkpoints made before the
    grid); the checkpoint's path in the configuration hash covers it, so it
    is False here.
    """
    from agribound.engines.finetune._data import dinov3_grid_gsd

    value = params.get("processing_gsd_m")
    if value is None and not fine_tuned:
        return False
    sizes = _composite_pixel_sizes_m(config)
    if sizes is None:
        # Pixels of about 1 m or finer (usgs-naip-plus), or an unreadable local raster:
        # "auto" and "native" read it as it is; a number or an invalid value counts.
        keyword = isinstance(value, str) and value.strip().lower() in ("auto", "native")
        return value is not None and not keyword
    try:
        return any(dinov3_grid_gsd(size, params) is not None for size in sizes)
    except ValueError:
        # The run raises this; an output made when DINOv3 ignored the setting is not reused.
        return True


def _dinov3_processing_grid(config: Any) -> bool:
    """True if a DINOv3 run of *config*, or a DINOv3 ensemble member, reads a processing grid."""
    engine = str(getattr(config, "engine", "") or "").lower().strip()
    if engine == "dinov3":
        params = getattr(config, "engine_params", None) or {}
        return _dinov3_reads_a_grid(config, params, bool(getattr(config, "fine_tune", False)))
    if engine == "ensemble":
        members = [params for name, params in _ensemble_members(config) if name == "dinov3"]
        return any(_dinov3_reads_a_grid(config, params, False) for params in members)
    return False


def results_versions(config: Any) -> dict[str, int]:
    """Return the :data:`RESULTS_VERSIONS` entries of the components *config* runs.

    - ``"embedding"``: ``config.engine == "embedding"``, or an ensemble
      member is the embedding engine (no source supports both engines
      today; checked so that a registry change cannot make a stale output
      reusable).
    - ``"sam_refine"``: ``config.sam_refine`` (which also absorbs the legacy
      ``engine_params["sam_refine"]``). The pipeline refines the output of
      every engine, including an ensemble's; the embedding engine refines
      its own polygons.
    - ``"delineate_anything_finetune"``: ``config.fine_tune`` with
      ``config.engine == "delineate-anything"`` (a run that loads a fine-tuned
      checkpoint through ``engine_params["checkpoint_path"]`` is covered by
      the checkpoint's path in the configuration hash).
    - ``"delineate_anything"``: ``config.engine == "delineate-anything"``
      with the native backend (``engine_params["backend"]`` unset or
      ``"native"``), fine-tuned or not, or an ensemble member that is (the
      default members include it).
    - ``"dinov3_finetune"``: ``config.fine_tune`` with
      ``config.engine == "dinov3"`` (a run that loads a checkpoint through
      ``engine_params["checkpoint_path"]`` is covered by the checkpoint's
      path in the configuration hash).
    - ``"dinov3"``: ``config.engine == "dinov3"``, fine-tuned or not, or an
      ensemble member that is.
    - ``"dinov3_processing_grid"``: DINOv3 reads the composite on a processing
      grid (:func:`agribound.engines.finetune._data.dinov3_processing_gsd`):
      ``config.fine_tune`` with ``config.engine == "dinov3"`` on a source
      with pixels coarser than 10.1 m (``landsat`` and ``hls``, 30 m;
      ``landsat-pan``, 15 m; ``naip`` with ``naip_resolution_m`` above
      10.1 m; a ``local`` raster, read from ``local_tif_path``) unless
      ``engine_params["processing_gsd_m"]`` is ``"native"``, or on any source
      with a number there that gives a grid; or a DINOv3 run that loads a
      checkpoint, or an ensemble member, whose ``engine_params`` set
      ``processing_gsd_m`` to a value that gives a grid. An invalid value
      counts too (the run raises it). Pixel sizes are in metres, also for an
      ``export_crs`` or a local raster in feet. For a source exported by
      agribound the size is the export resolution, so a composite in an
      export CRS in degrees, whose pixels DINOv3 measures along the
      parallel, can count here although the engine reads it as it is. A
      local raster in degrees counts when a crop of it at any of its
      latitudes would be read on a grid. Both errors lead only to running
      again an output that would not change.
    - ``"sam_auto"``: ``config.engine == "sam-auto"``, or an ensemble member
      that is.

    Parameters
    ----------
    config : AgriboundConfig
        Configuration of the run.

    Returns
    -------
    dict
        Component name to version number, in :data:`RESULTS_VERSIONS` order;
        empty when no versioned component runs.
    """
    engine = str(getattr(config, "engine", "") or "").lower().strip()
    engine_params = getattr(config, "engine_params", None) or {}
    applies = {
        "embedding": engine == "embedding"
        or (engine == "ensemble" and "embedding" in _ensemble_member_engines(config)),
        "sam_refine": bool(getattr(config, "sam_refine", False))
        or bool(engine_params.get("sam_refine", False)),
        "delineate_anything_finetune": bool(getattr(config, "fine_tune", False))
        and engine == "delineate-anything",
        "delineate_anything": _native_delineate_anything(engine, engine_params)
        or (
            engine == "ensemble"
            and any(_native_delineate_anything(n, p) for n, p in _ensemble_members(config))
        ),
        "dinov3_finetune": bool(getattr(config, "fine_tune", False)) and engine == "dinov3",
        "dinov3": _runs_engine(config, "dinov3"),
        "dinov3_processing_grid": _dinov3_processing_grid(config),
        "sam_auto": _runs_engine(config, "sam-auto"),
    }
    return {name: version for name, version in RESULTS_VERSIONS.items() if applies[name]}


__all__ = ["RESULTS_VERSIONS", "results_versions"]
