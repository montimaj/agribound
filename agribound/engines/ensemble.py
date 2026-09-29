"""
Ensemble engine: several engines (or one engine with different models) on the
same raster, combined by intersection, union or pixel vote.

Merge strategies
----------------
``"intersection"`` (default)
    Successive :func:`geopandas.overlay` ``how="intersection"`` of the
    members' polygons: the output pieces are the areas every member covers,
    one piece per combination of overlapping polygons. Pieces are not
    re-merged, so where members' boundaries disagree small sliver pieces can
    appear next to the main piece of a field; the pipeline's area filter
    removes those below ``min_field_area_m2``.
``"union"``
    All members' polygons are pooled and duplicates are fused with
    :func:`agribound.postprocess.merge.merge_polygons` (IoU >=
    ``union_iou_threshold`` or containment >= ``union_containment_threshold``;
    defaults 0.3 and 0.8). Fields that only touch stay separate.
``"vote"``
    Each member's polygons are rasterised (pixel centres) onto the input
    raster's grid; a pixel is kept when at least ``min_votes`` members cover
    it, and the kept pixels are polygonised (4-connectivity). As in
    agribound 0.1.x, members that returned no polygons are left out of the
    vote (logged at WARNING and listed in ``vote_stats["empty_members"]``),
    and by default ``min_votes = max(min(2, n), ceil(vote_threshold * n))``
    for the ``n`` remaining members: at least ``vote_threshold`` of them,
    and at least two whenever there are two or more, so that one member's
    false positives never pass on their own. ``engine_params["min_votes"]``
    sets it directly (1..n). Members skipped after an error
    (``on_member_error="skip"``) do not count either. Adjacent fields that
    are both kept become one polygon wherever they touch on the pixel grid,
    because the vote raster is binary (field / not field).

With a single member (configured, or the only one left with
``on_member_error="skip"``) that member's frame is returned as it is (all
its columns), plus an ``engine_count`` column.
"""

from __future__ import annotations

import copy
import logging
import math
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine, get_engine, get_engine_class
from agribound.registry import ENGINE_REGISTRY, ENSEMBLE_DEFAULT_MEMBERS, engine_supports_source

logger = logging.getLogger(__name__)

#: Default members (the registry's :data:`~agribound.registry.ENSEMBLE_DEFAULT_MEMBERS`).
DEFAULT_MEMBERS: tuple[str, ...] = ENSEMBLE_DEFAULT_MEMBERS
MERGE_STRATEGIES: tuple[str, ...] = ("intersection", "union", "vote")

#: engine_params read by the ensemble itself.
ENSEMBLE_KEYS: frozenset[str] = frozenset(
    {
        "engines",
        "merge_strategy",
        "vote_threshold",
        "min_votes",
        "vote_resolution",
        "union_iou_threshold",
        "union_containment_threshold",
        "on_member_error",
        "isolate_member_caches",
    }
)
#: engine_params read by the pipeline or the SAM stage for the ensemble output.
PIPELINE_KEYS: frozenset[str] = frozenset(
    {
        "smooth_iterations",
        "regularize",
        "sam_refine",
        "sam_model",
        "sam_batch_size",
        "sam_window_px",
        "sam_rgb_bands",
        "sam_overlaps",
    }
)

MEMBERS_COLUMN = "ensemble:members"
N_MEMBERS_COLUMN = "ensemble:n_members"


class EnsembleEngine(DelineationEngine):
    """Multi-engine or multi-model ensemble (see the module docstring for the strategies).

    Engine parameters (``config.engine_params``)
    --------------------------------------------
    engines : list[str | dict]
        Members; each is an engine name or a dict with ``"engine"``, optional
        ``"engine_params"`` and optional ``"label"`` (default: the member's
        ``engine_params["model"]`` or the engine name; a repeated label gets
        an ``_<index>`` suffix, counting up from the member's position until
        it is unique). Default ``["delineate-anything", "ftw"]``. Every
        member, including the defaults, must support ``config.source``.
    merge_strategy : str
        ``"intersection"`` (default), ``"union"`` or ``"vote"``.
    vote_threshold : float
        Vote strategy: fraction in [0, 1] of the members with polygons that
        must agree (default 0.5); at least two members must agree whenever
        two or more have polygons.
    min_votes : int or None
        Vote strategy: explicit minimum number of agreeing members (1..n,
        where n counts the members with polygons); overrides
        *vote_threshold*.
    vote_resolution : float or None
        Vote grid cell size in CRS units. *None* (default) uses the input
        raster's own grid.
    union_iou_threshold, union_containment_threshold : float
        Union strategy duplicate criteria (defaults 0.3, 0.8).
    on_member_error : str
        ``"raise"`` (default): a failing member aborts the run. ``"skip"``:
        continue without it; the failure is logged as a warning and listed
        in ``engine_meta["failed_members"]``.
    isolate_member_caches : bool
        Give every member its own cache directory
        ``<working dir>/ensemble/<slug>`` (default *True*), so members with
        different models never reuse each other's intermediates. The slug is
        the label with characters other than letters, digits, ``.``, ``_``
        and ``-`` replaced by ``_``; a slug that repeats, ignoring case, gets
        an ``_<index>`` suffix. Members that build their own window composites
        (FTW) then download them into their own directory. *False* shares
        the ensemble's cache directory.

    Members receive only the ``engine_params`` of their own spec; they do
    not inherit the ensemble-level ``engine_params``. Ensemble-level keys
    other than the ones above and the pipeline/SAM keys in
    :data:`PIPELINE_KEYS` raise ``ValueError`` (put them in a member spec).
    Every member runs with ``sam_refine=False`` (the pipeline refines the
    ensemble output instead) and is seeded with ``config.seed`` right before
    it runs, so its result does not depend on the member order.

    Output columns: ``engine_count`` (intersection and union: number of
    members that ran without an error; vote: number of those with polygons)
    plus, per strategy,
    ``ensemble:members`` (intersection: all member labels; union: labels of
    the members whose polygons were fused, comma-separated),
    ``ensemble:n_members`` (union), or ``vote_count`` (maximum number of
    agreeing members inside the polygon; in 0.1.x this column held the
    constant ``min_votes``), ``vote_count_mean`` and ``min_votes`` (vote).
    Member attribute columns are not carried over, except with a single
    member, whose frame is returned as it is plus ``engine_count``.
    ``attrs["engine_meta"]`` lists every member's label, engine, parameters,
    polygon count, cache directory and own ``engine_meta``.
    """

    name = "ensemble"
    supported_sources = list(ENGINE_REGISTRY["ensemble"]["supported_sources"])
    requires_bands = list(ENGINE_REGISTRY["ensemble"]["requires_bands"])

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def member_specs(config: AgriboundConfig) -> list[dict[str, Any]]:
        """Return the normalised member specs.

        Each spec is ``{"label", "engine", "engine_params", "cache_slug"}``;
        labels and cache slugs are unique.

        Raises
        ------
        ValueError
            For an empty or malformed member list, an unknown engine, a nested
            ensemble, or a member that does not support ``config.source``.
        """
        raw = (config.engine_params or {}).get("engines")
        if raw is None:
            raw = list(DEFAULT_MEMBERS)
        if not isinstance(raw, list | tuple) or not raw:
            raise ValueError("engine_params['engines'] must be a non-empty list for ensemble")
        specs: list[dict[str, Any]] = []
        used: set[str] = set()
        used_slugs: set[str] = set()
        for i, entry in enumerate(raw):
            if isinstance(entry, str):
                name, params, label = entry, {}, None
            elif isinstance(entry, dict) and isinstance(entry.get("engine"), str):
                unknown = set(entry) - {"engine", "engine_params", "label"}
                if unknown:
                    raise ValueError(f"Unknown keys {sorted(unknown)} in ensemble member {entry!r}")
                name = entry["engine"]
                params = entry.get("engine_params") or {}
                label = entry.get("label")
                if not isinstance(params, dict):
                    raise ValueError(f"engine_params of ensemble member {entry!r} must be a dict")
            else:
                raise ValueError(f"Invalid ensemble member spec: {entry!r}")
            name = name.lower().strip()
            if name not in ENGINE_REGISTRY or name == "ensemble":
                raise ValueError(
                    f"Invalid ensemble member {name!r}. Choose from "
                    f"{[n for n in ENGINE_REGISTRY if n != 'ensemble']}"
                )
            if not engine_supports_source(name, config.source):
                default = " (a default member)" if "engines" not in config.engine_params else ""
                raise ValueError(
                    f"Ensemble member {name!r}{default} does not support source "
                    f"{config.source!r} (supported: {ENGINE_REGISTRY[name]['supported_sources']}). "
                    "Set engine_params['engines'] to members that support it."
                )
            label = _unique(str(label or params.get("model") or name), used, i)
            used.add(label)
            # Distinct labels can share a slug ("a/b" and "a_b"), and macOS file
            # systems ignore case: keep the directories apart.
            slug = _unique(_slug(label, name), used_slugs, i, casefold=True)
            used_slugs.add(slug.casefold())
            specs.append(
                {
                    "label": label,
                    "engine": name,
                    "engine_params": copy.deepcopy(params),
                    "cache_slug": slug,
                }
            )
        return specs

    @staticmethod
    def resolve_params(config: AgriboundConfig) -> dict[str, Any]:
        """Return the validated ensemble-level parameters.

        Raises
        ------
        ValueError
            For unknown ensemble-level keys or invalid values.
        """
        ep = dict(config.engine_params or {})
        unknown = sorted(set(ep) - ENSEMBLE_KEYS - PIPELINE_KEYS)
        if unknown:
            raise ValueError(
                f"engine_params {unknown} are not used by the ensemble and are not passed to its "
                "members; put member settings in each member's 'engine_params' "
                "(engine_params={'engines': [{'engine': ..., 'engine_params': {...}}]})."
            )
        strategy = str(ep.get("merge_strategy", "intersection")).lower().strip()
        if strategy not in MERGE_STRATEGIES:
            raise ValueError(f"Unknown merge strategy {strategy!r}. Choose from {MERGE_STRATEGIES}")
        threshold = float(ep.get("vote_threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"vote_threshold must be in [0, 1], got {threshold}")
        on_error = str(ep.get("on_member_error", "raise")).lower().strip()
        if on_error not in ("raise", "skip"):
            raise ValueError(f"on_member_error must be 'raise' or 'skip', got {on_error!r}")
        resolution = ep.get("vote_resolution")
        if resolution is not None and float(resolution) <= 0:
            raise ValueError(f"vote_resolution must be > 0, got {resolution}")
        min_votes = ep.get("min_votes")
        return {
            "merge_strategy": strategy,
            "vote_threshold": threshold,
            "min_votes": None if min_votes is None else int(min_votes),
            "vote_resolution": None if resolution is None else float(resolution),
            "union_iou_threshold": float(ep.get("union_iou_threshold", 0.3)),
            "union_containment_threshold": float(ep.get("union_containment_threshold", 0.8)),
            "on_member_error": on_error,
            "isolate_member_caches": bool(ep.get("isolate_member_caches", True)),
        }

    @staticmethod
    def member_config(
        config: AgriboundConfig, spec: dict[str, Any], isolate_cache: bool = True
    ) -> AgriboundConfig:
        """Return the validated configuration a member runs with.

        Same fields as *config* except ``engine``, ``engine_params`` (the
        member's own), ``sam_refine=False`` and, with *isolate_cache*,
        ``cache_dir=<working dir>/ensemble/<spec["cache_slug"]>`` (the slug
        of the label when the spec has no ``cache_slug``).
        """
        overrides: dict[str, Any] = {
            "engine": spec["engine"],
            "engine_params": copy.deepcopy(spec["engine_params"]),
            "sam_refine": False,
        }
        if isolate_cache:
            slug = spec.get("cache_slug") or _slug(spec["label"], spec["engine"])
            overrides["cache_dir"] = str(Path(config.get_working_dir()) / "ensemble" / slug)
        return config.merged(**overrides)

    # ------------------------------------------------------------------
    # Engine API
    # ------------------------------------------------------------------

    @classmethod
    def prefetch(cls, config: AgriboundConfig) -> list[str]:
        """Prefetch every member's weights (duplicates removed, order kept)."""
        paths: list[str] = []
        for spec in cls.member_specs(config):
            member = cls.member_config(config, spec, isolate_cache=False)
            for path in get_engine_class(spec["engine"]).prefetch(member) or []:
                if str(path) not in paths:
                    paths.append(str(path))
        return paths

    @classmethod
    def stage_inputs(cls, config: AgriboundConfig, raster_path: str) -> dict[str, Any]:
        """Build the inputs that members build themselves, without running them.

        For every member whose engine class has a ``stage_inputs`` method
        (currently FTW: :meth:`agribound.engines.ftw.FTWEngine.stage_inputs`,
        the two seasonal window composites), calls it with the member's own
        configuration (:meth:`member_config`, including its isolated cache
        directory) and *raster_path*, so a later :meth:`delineate` with the
        same configuration finds the inputs in the member caches without
        network access. Used by :mod:`agribound.hpc.tiles` to stage ensemble
        tiles.

        Parameters
        ----------
        config : AgriboundConfig
            Ensemble configuration.
        raster_path : str
            Input raster shared by the members.

        Returns
        -------
        dict
            ``members`` (label -> the member's ``stage_inputs`` result),
            ``rasters`` (all staged rasters, member order) and
            ``failed_members`` (``{"label", "engine", "error"}``; only with
            ``on_member_error="skip"``, as in :meth:`delineate`).

        Raises
        ------
        Exception
            A member's staging error when ``on_member_error="raise"``.
        """
        specs = cls.member_specs(config)
        params = cls.resolve_params(config)
        members: dict[str, Any] = {}
        rasters: list[str] = []
        failed: list[dict[str, str]] = []
        for spec in specs:
            stage = getattr(get_engine_class(spec["engine"]), "stage_inputs", None)
            if stage is None:
                continue
            member = cls.member_config(config, spec, params["isolate_member_caches"])
            try:
                staged = stage(member, raster_path)
            except Exception as exc:
                if params["on_member_error"] == "raise":
                    exc.add_note(f"(ensemble member {spec['label']!r}, engine {spec['engine']!r})")
                    raise
                logger.warning(
                    "Staging the inputs of ensemble member %s failed; it will be retried (and "
                    "skipped if it fails) during delineation: %s",
                    spec["label"],
                    exc,
                )
                failed.append(
                    {
                        "label": spec["label"],
                        "engine": spec["engine"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            members[spec["label"]] = staged
            rasters.extend(str(p) for p in staged.get("rasters", []))
        return {"members": members, "rasters": rasters, "failed_members": failed}

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run every member on *raster_path* and merge the results.

        Parameters
        ----------
        raster_path : str
            Input GeoTIFF shared by all members.
        config : AgriboundConfig
            Pipeline configuration (``engine="ensemble"``).

        Returns
        -------
        geopandas.GeoDataFrame
            Merged polygons in the raster CRS, with ``attrs["engine_meta"]``.

        Raises
        ------
        ValueError
            For invalid ensemble parameters or members.
        RuntimeError
            If every member failed (``on_member_error="skip"``).
        """
        from agribound._repro import seed_everything
        from agribound.io.raster import get_raster_info

        specs = self.member_specs(config)
        params = self.resolve_params(config)
        info = get_raster_info(raster_path)
        target_crs = info.crs

        results: dict[str, gpd.GeoDataFrame] = {}
        members_meta: list[dict[str, Any]] = []
        failed: list[dict[str, str]] = []
        for i, spec in enumerate(specs):
            label = spec["label"]
            member = self.member_config(config, spec, params["isolate_member_caches"])
            logger.info(
                "Ensemble [%d/%d]: running %s (%s)", i + 1, len(specs), label, spec["engine"]
            )
            seed_everything(member.seed)
            try:
                gdf = get_engine(spec["engine"]).delineate(raster_path, member)
            except Exception as exc:
                if params["on_member_error"] == "raise":
                    exc.add_note(f"(ensemble member {label!r}, engine {spec['engine']!r})")
                    raise
                logger.warning("Ensemble member %s failed and is skipped: %s", label, exc)
                failed.append(
                    {
                        "label": label,
                        "engine": spec["engine"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            gdf = _as_frame(gdf, target_crs)
            results[label] = gdf
            members_meta.append(
                {
                    "label": label,
                    "engine": spec["engine"],
                    "engine_params": spec["engine_params"],
                    "n_polygons": int(len(gdf)),
                    "cache_dir": str(member.get_working_dir()),
                    "seed": int(member.seed),
                    "engine_meta": copy.deepcopy(gdf.attrs.get("engine_meta")),
                }
            )
            logger.info("%s produced %d polygons", label, len(gdf))

        if not results:
            raise RuntimeError(
                "All ensemble members failed:\n"
                + "\n".join(f"  - {f['label']}: {f['error']}" for f in failed)
            )

        strategy = params["merge_strategy"]
        meta: dict[str, Any] = {
            "backend": "ensemble",
            "merge_strategy": strategy,
            "n_members": len(results),
            "members": members_meta,
            "failed_members": failed,
            "isolate_member_caches": params["isolate_member_caches"],
        }
        if len(results) == 1:
            merged = next(iter(results.values())).copy()
            merged["engine_count"] = 1
            meta["note"] = "single member: its polygons are returned unchanged"
        elif strategy == "union":
            merged = self._merge_union(
                results,
                iou_threshold=params["union_iou_threshold"],
                containment_threshold=params["union_containment_threshold"],
            )
            meta["union_iou_threshold"] = params["union_iou_threshold"]
            meta["union_containment_threshold"] = params["union_containment_threshold"]
        elif strategy == "intersection":
            merged = self._merge_intersection(results)
        else:
            grid = None
            if params["vote_resolution"] is None:
                grid = (info.crs, info.transform, info.width, info.height)
            merged = self._merge_vote(
                results,
                params["vote_threshold"],
                min_votes=params["min_votes"],
                resolution=params["vote_resolution"],
                grid=grid,
            )
            meta["vote_threshold"] = params["vote_threshold"]
            meta["vote"] = copy.deepcopy(merged.attrs.get("vote_stats"))
        merged.attrs = {"engine_meta": meta}
        return merged

    # ------------------------------------------------------------------
    # Merge strategies (static; also usable on saved member outputs)
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_union(
        results: dict[str, gpd.GeoDataFrame],
        iou_threshold: float = 0.3,
        containment_threshold: float = 0.8,
    ) -> gpd.GeoDataFrame:
        """Pool all members' polygons and fuse duplicates (see module docstring)."""
        from agribound.postprocess.merge import merge_polygons

        frames, crs = _aligned(results)
        labels = list(frames)
        rows = []
        for label in labels:
            gdf = frames[label]
            geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
            rows.append(
                gpd.GeoDataFrame({"_member": [label] * len(geoms)}, geometry=geoms, crs=crs)
            )
        pooled = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), geometry="geometry", crs=crs)
        if len(pooled) == 0:
            return _empty(crs, [MEMBERS_COLUMN, N_MEMBERS_COLUMN, "engine_count"])
        merged, groups = merge_polygons(
            pooled,
            iou_threshold=iou_threshold,
            containment_threshold=containment_threshold,
            return_groups=True,
        )
        member_of = pooled["_member"].to_numpy()
        names = [sorted({str(member_of[p]) for p in group}) for group in groups]
        out = gpd.GeoDataFrame(
            {
                MEMBERS_COLUMN: [",".join(n) for n in names],
                N_MEMBERS_COLUMN: [len(n) for n in names],
                "engine_count": len(results),
            },
            geometry=list(merged.geometry),
            crs=crs,
        )
        logger.info("Union merge: %d polygons from %d pooled", len(out), len(pooled))
        return out

    @staticmethod
    def _merge_intersection(results: dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        """Areas covered by every member, via successive overlays (see module docstring)."""
        frames, crs = _aligned(results)
        labels = list(frames)
        columns = [MEMBERS_COLUMN, "engine_count"]
        base = _geometry_only(frames[labels[0]])
        for label in labels[1:]:
            other = _geometry_only(frames[label])
            if len(base) == 0 or len(other) == 0:
                base = base.iloc[0:0]
                break
            base = gpd.overlay(base, other, how="intersection", keep_geom_type=True)
            base = _geometry_only(base)
        base = base[~base.geometry.isna() & ~base.geometry.is_empty]
        if len(base) == 0:
            logger.warning("Intersection merge produced no overlapping polygons")
            return _empty(crs, columns)
        n_out = len(base)
        out = gpd.GeoDataFrame(
            {
                MEMBERS_COLUMN: [",".join(sorted(labels))] * n_out,
                "engine_count": [len(results)] * n_out,
            },
            geometry=list(base.geometry),
            crs=crs,
        )
        logger.info("Intersection merge: %d polygons", len(out))
        return out

    @staticmethod
    def _merge_vote(
        results: dict[str, gpd.GeoDataFrame],
        threshold: float = 0.5,
        *,
        min_votes: int | None = None,
        resolution: float | None = None,
        grid: tuple[Any, Any, int, int] | None = None,
    ) -> gpd.GeoDataFrame:
        """Pixel vote (see module docstring).

        Parameters
        ----------
        results : dict[str, geopandas.GeoDataFrame]
            Member polygons by label. Members without polygons are left out
            of the vote (``n`` counts the others), as in agribound 0.1.x.
        threshold : float
            Vote threshold in [0, 1] (ignored when *min_votes* is given):
            ``min_votes = max(min(2, n), ceil(threshold * n))``.
        min_votes : int or None
            Explicit minimum number of agreeing members (1..n).
        resolution : float or None
            Cell size in CRS units for a grid over the members' combined
            extent. Used when *grid* is *None*; *None* then means 10 units in
            a projected CRS or 1e-4 degrees in a geographic one (the 0.1.x grid).
        grid : tuple or None
            ``(crs, transform, width, height)`` of the vote raster, e.g. the
            input raster's grid (what :meth:`delineate` passes).

        Returns
        -------
        geopandas.GeoDataFrame
            Polygons with ``vote_count`` (max agreement inside), ``vote_count_mean``,
            ``min_votes`` and ``engine_count`` (``n``); ``attrs["vote_stats"]``
            records ``min_votes``, the rule, ``threshold``, ``n_members``
            (``n``), ``n_members_total``, ``empty_members`` and the grid
            (*None* when no member has polygons). An explicit *min_votes*
            larger than ``n`` gives an empty result (logged at WARNING).

        Raises
        ------
        ValueError
            For no member results, a threshold outside [0, 1] or *min_votes*
            outside 1..n.
        """
        import rasterio
        from rasterio.features import rasterize, shapes
        from scipy import ndimage
        from shapely.geometry import shape as shapely_shape

        from agribound.postprocess.simplify import make_polygonal

        n_total = len(results)
        if n_total == 0:
            raise ValueError("_merge_vote needs at least one member result")
        if min_votes is None and not 0.0 <= threshold <= 1.0:
            raise ValueError(f"vote threshold must be in [0, 1], got {threshold}")
        if min_votes is not None and not 1 <= int(min_votes) <= n_total:
            raise ValueError(f"min_votes must be in [1, {n_total}], got {min_votes}")
        columns = ["vote_count", "vote_count_mean", "min_votes", "engine_count"]

        # As in 0.1.x, members without polygons are left out of the vote.
        empty_members = [k for k, v in results.items() if not _has_polygons(v)]
        voting = {k: v for k, v in results.items() if k not in empty_members}
        n = len(voting)
        if min_votes is None:
            rule = "max(min(2, n), ceil(threshold * n)) (agribound 0.1.x)"
            min_votes = _min_votes(n, threshold) if n else None
        else:
            rule = "explicit min_votes"
            min_votes = int(min_votes)
        stats: dict[str, Any] = {
            "min_votes": min_votes,
            "rule": rule,
            "n_members": n,
            "n_members_total": n_total,
            "empty_members": empty_members,
            "threshold": threshold,
            "grid": None,
        }
        if empty_members:
            logger.warning(
                "Vote merge: %d of %d members returned no polygons and are left out of the "
                "vote (%s)",
                len(empty_members),
                n_total,
                ", ".join(empty_members),
            )
        if n == 0:
            crs = grid[0] if grid is not None else _aligned(results)[1]
            out = _empty(crs, columns)
            out.attrs["vote_stats"] = stats
            return out
        if min_votes > n:
            logger.warning(
                "Vote merge: min_votes=%d but only %d member(s) returned polygons; the "
                "result is empty",
                min_votes,
                n,
            )

        if grid is not None:
            crs, transform, width, height = grid
            frames = {
                k: (v.to_crs(crs) if v.crs is not None and v.crs != crs else v)
                for k, v in voting.items()
            }
        else:
            frames, crs = _aligned(voting)
            bounds = np.array([f.total_bounds for f in frames.values()])
            minx, miny = np.nanmin(bounds[:, 0]), np.nanmin(bounds[:, 1])
            maxx, maxy = np.nanmax(bounds[:, 2]), np.nanmax(bounds[:, 3])
            if resolution is None:
                resolution = 1e-4 if crs is not None and crs.is_geographic else 10.0
            width = int(np.ceil((maxx - minx) / resolution))
            height = int(np.ceil((maxy - miny) / resolution))
            if width == 0 or height == 0:
                out = _empty(crs, columns)
                out.attrs["vote_stats"] = stats
                return out
            transform = rasterio.transform.from_origin(minx, maxy, resolution, resolution)
        stats["grid"] = {
            "crs": str(crs) if crs is not None else None,
            "transform": list(transform)[:6],
            "width": int(width),
            "height": int(height),
        }

        votes = np.zeros((height, width), dtype=np.uint16)
        for gdf in frames.values():
            geoms = [make_polygonal(g) for g in gdf.geometry if g is not None and not g.is_empty]
            burn = [(g, 1) for g in geoms if g is not None and not g.is_empty]
            if burn:
                votes += rasterize(
                    burn, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8
                )

        consensus = votes >= min_votes
        labels, n_labels = ndimage.label(consensus, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        if n_labels == 0:
            out = _empty(crs, columns)
            out.attrs["vote_stats"] = stats
            return out
        index = np.arange(1, n_labels + 1)
        max_votes = np.asarray(ndimage.maximum(votes, labels, index), dtype=int)
        mean_votes = np.asarray(ndimage.mean(votes, labels, index), dtype=float)

        polys: dict[int, Any] = {}
        for geom, val in shapes(
            labels.astype(np.int32), mask=consensus, transform=transform, connectivity=4
        ):
            polys[int(val)] = shapely_shape(geom)
        ids = sorted(polys)
        out = gpd.GeoDataFrame(
            {
                "vote_count": [int(max_votes[i - 1]) for i in ids],
                "vote_count_mean": [float(mean_votes[i - 1]) for i in ids],
                "min_votes": min_votes,
                "engine_count": n,
            },
            geometry=[polys[i] for i in ids],
            crs=crs,
        )
        out.attrs["vote_stats"] = stats
        logger.info("Vote merge: %d polygons (at least %d of %d members)", len(out), min_votes, n)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _min_votes(n: int, threshold: float) -> int:
    """``max(min(2, n), ceil(threshold * n))``: the agribound 0.1.x vote rule.

    At least *threshold* of the *n* members, and at least two members when
    ``n >= 2``. A tolerance of 1e-9 keeps float products that land just
    above an integer (``0.28 * 25 = 7.000000000000001``) from rounding up;
    0.1.x had no tolerance and required 8 votes in that case.
    """
    return int(max(min(2, n), math.ceil(threshold * n - 1e-9)))


def _slug(label: str, fallback: str) -> str:
    """Directory-safe form of a member label."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or fallback


def _unique(name: str, used: set[str], index: int, casefold: bool = False) -> str:
    """*name*, or ``name_<index>`` (``_<index+1>``, ...) until it is not in *used*."""
    key = (lambda s: s.casefold()) if casefold else (lambda s: s)
    candidate, k = name, index
    while key(candidate) in used:
        candidate = f"{name}_{k}"
        k += 1
    return candidate


def _has_polygons(gdf: gpd.GeoDataFrame) -> bool:
    """True if *gdf* has at least one non-missing, non-empty geometry."""
    if gdf is None or len(gdf) == 0:
        return False
    geoms = gdf.geometry
    return bool((~geoms.isna() & ~geoms.is_empty).any())


def _as_frame(gdf: Any, crs: Any) -> gpd.GeoDataFrame:
    """Member output as a GeoDataFrame in *crs* (empty/CRS-less outputs are allowed)."""
    if gdf is None:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"Ensemble member returned {type(gdf).__name__}, expected a GeoDataFrame")
    if gdf.crs is None:
        if len(gdf) > 0:
            raise ValueError("Ensemble member returned polygons without a CRS")
        return gdf.set_crs(crs, allow_override=True) if crs is not None else gdf
    if crs is not None and gdf.crs != crs:
        return gdf.to_crs(crs)
    return gdf


def _aligned(results: dict[str, gpd.GeoDataFrame]) -> tuple[dict[str, gpd.GeoDataFrame], Any]:
    """Reproject every frame to the first non-missing CRS."""
    crs = next((g.crs for g in results.values() if g.crs is not None), None)
    frames = {
        k: (g.to_crs(crs) if crs is not None and g.crs is not None and g.crs != crs else g)
        for k, g in results.items()
    }
    return frames, crs


def _geometry_only(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=list(gdf.geometry), crs=gdf.crs)


def _empty(crs: Any, columns: list[str]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({c: [] for c in columns}, geometry=[], crs=crs)
