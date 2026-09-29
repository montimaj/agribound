"""USGS NAIP Plus ImageServer composite builder.

The USGS NAIP Plus ImageServer serves only the **latest NAIP/HRO vintage of
each state** (years 2012-2023 as of 2026-09, most states 2019-2023), not the
historical NAIP archive, so most years are unavailable for a given state. Use
``source="naip"`` (Earth Engine, 2002-2023) for other years. The service's
pixel size is 0.3 m; state vintages are 0.3-0.6 m.

The output covers the export grid: the bounding box of the study area in
``config.export_crs`` (``"utm"``: the UTM zone of the study-area centroid), as
for the other builders; pixels are not masked to the study-area polygons.

:class:`USGSNAIPPlusCompositeBuilder`:

1. queries catalogue items (``Category = 1``, ``Year = config.year``,
   optional ``State``) intersecting the export-grid outline; when none match it
   raises :class:`ValueError` listing the years the service has for the area;
2. selects footprints greedily (closest year, 4 bands, largest overlap,
   finest resolution) until the grid outline is covered or the service's
   ``maxMosaicImageCount`` is reached, and locks the mosaic to them
   (``esriMosaicLockRaster``); a WARNING is logged when the selected
   footprints cover less than 99 % of the grid outline;
3. exports tiles of at most ``min(config.tile_size, maxImageWidth,
   maxImageHeight)`` pixels in EPSG:3857 at the finest ground resolution of the
   selected footprints (the service resamples with bilinear interpolation),
   with the Web Mercator pixel size scaled by ``1 / cos(latitude)`` of the
   study-area centroid so that the ground sampling matches;
4. merges the tiles and resamples them (bilinear) onto the export grid;
5. writes a uint8 GeoTIFF (nodata 0; pixels no selected footprint covers are
   0) with a JSON manifest of the query and selection, and records the share
   of study-area pixels that have imagery (``AGRIBOUND_VALID_FRACTION``; a
   WARNING is logged below 99 %, and :class:`ValueError` is raised at 0).

The band order is assumed to be R, G, B, N (the service reports generic band
names ``Band_1``-``Band_4`` and sensor type ``CNIR``).
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from shapely.geometry.base import BaseGeometry

from agribound.clients.usgs_naip_plus import (
    DEFAULT_USGS_NAIPPLUS_URL,
    USGSNAIPPlusClient,
    USGSRasterCandidate,
)
from agribound.composites.base import SOURCE_REGISTRY, CompositeBuilder, NoDataError
from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

_METRE_UNITS = {"meter", "meters", "metre", "metres", "m"}
_FOOT_UNITS = {"foot", "feet", "ft", "foot_us", "us_feet", "us survey feet"}
_DEFAULT_RESOLUTION_M = 1.0
#: Below this share of coverage (selected footprints over the grid outline, or
#: study-area pixels with imagery) a WARNING is logged.
LOW_COVERAGE_FRACTION = 0.99


def _to_3857(geom_4326: BaseGeometry) -> BaseGeometry:
    """Transform an EPSG:4326 geometry to EPSG:3857."""
    import pyproj
    from shapely.ops import transform as shapely_transform

    t = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return shapely_transform(t.transform, geom_4326)


class USGSNAIPPlusCompositeBuilder(CompositeBuilder):
    """Build a local uint8 raster for the study area from the USGS NAIP Plus ImageServer.

    Attributes
    ----------
    SELECTION_VERSION : int
        Version of the selection/export recipe (part of the cache key).
    last_metadata : dict
        Facts about the most recent :meth:`build`.
    """

    SELECTION_VERSION = 3

    def __init__(self) -> None:
        self.last_metadata: dict[str, Any] = {}

    def build(self, config: AgriboundConfig) -> str:
        """Download and mosaic a USGS NAIP Plus raster covering the study-area extent.

        Returns
        -------
        str
            Path to the uint8 GeoTIFF (cached with
            :func:`agribound._cache.cache_path`).

        Raises
        ------
        NoDataError
            (A :class:`ValueError`.) If the service has no imagery for the
            requested year (the message lists the years it has for the area),
            or no pixel inside the study area has imagery.
        ValueError
            If no study area is set or the study area is empty.
        """
        from agribound._cache import cache_path

        if not config.study_area:
            raise ValueError("study_area is required when source='usgs-naip-plus'")

        from agribound.composites.gee import (
            compute_export_grid,
            grid_footprint_4326,
            raster_valid_fraction,
            resolve_export_crs,
            split_grid,
            study_area_geometry_4326,
        )

        geom_4326 = study_area_geometry_4326(config)

        final_path = cache_path(config, "usgs_naip_plus", ".tif", self.SELECTION_VERSION)
        manifest_path = final_path.with_suffix(".json")
        if final_path.exists() and manifest_path.exists():
            logger.info("Using cached USGS NAIP Plus raster: %s", final_path)
            return str(final_path)

        client = USGSNAIPPlusClient(
            service_url=getattr(config, "usgs_service_url", DEFAULT_USGS_NAIPPLUS_URL),
            timeout_s=getattr(config, "usgs_timeout_s", 120),
            retries=getattr(config, "usgs_retries", 3),
        )

        service_metadata = client.get_service_metadata()
        # The query/selection target is the outline of a provisional export grid
        # (the study-area bounding box in the export CRS), in EPSG:3857.
        export_crs = resolve_export_crs(config.export_crs, geom_4326)
        provisional = compute_export_grid(geom_4326, export_crs, _DEFAULT_RESOLUTION_M)
        aoi_3857 = _to_3857(grid_footprint_4326(provisional))
        where = self._build_where_clause(config)
        candidates = client.query_candidates(aoi_3857.bounds, where, out_sr=3857)
        candidates = self._filter_candidates(candidates, aoi_3857)

        if not candidates and getattr(config, "usgs_allow_year_fallback", False):
            where = self._build_where_clause(config, allow_year_fallback=True)
            candidates = client.query_candidates(aoi_3857.bounds, where, out_sr=3857)
            candidates = self._filter_candidates(candidates, aoi_3857)

        if not candidates:
            raise self._no_imagery_error(client, config, aoi_3857)

        max_ids = int(service_metadata.get("maxMosaicImageCount", 50))
        selected_ids, ranked_candidates = self._select_lock_raster_ids(
            candidates,
            aoi_3857,
            target_year=int(config.year),
            max_ids=max_ids,
        )
        if not selected_ids:
            raise RuntimeError("Candidate selection returned no LockRaster IDs")
        selected = [c for c in ranked_candidates if c.object_id in set(selected_ids)]
        footprint_share = self._footprint_coverage(selected, aoi_3857)
        if footprint_share < LOW_COVERAGE_FRACTION:
            reason = (
                f"the selection reached the service's maxMosaicImageCount ({max_ids})"
                if len(selected_ids) >= max_ids
                else "no other candidate footprint adds coverage"
            )
            logger.warning(
                "The %d selected USGS NAIP Plus footprint(s) cover %.1f%% of the study-area "
                "extent (%s); uncovered pixels are 0 (nodata).",
                len(selected_ids),
                100.0 * footprint_share,
                reason,
            )

        export_max_px = min(
            int(getattr(config, "tile_size", 10_000)),
            int(service_metadata.get("maxImageWidth", 4000)),
            int(service_metadata.get("maxImageHeight", 4000)),
        )
        resolution_m = self._estimate_resolution_m(selected)
        centroid_lat = float(geom_4326.centroid.y)
        resolution_3857 = resolution_m / math.cos(math.radians(centroid_lat))

        grid = compute_export_grid(geom_4326, export_crs, resolution_m)
        grid_3857 = compute_export_grid(grid_footprint_4326(grid), "EPSG:3857", resolution_3857)
        windows = split_grid(grid_3857, export_max_px)
        tile_dir = final_path.with_name(final_path.stem + "_tiles")
        tile_dir.mkdir(parents=True, exist_ok=True)

        raw_tile_paths: list[str] = []
        for idx, (row_off, col_off, height, width) in enumerate(windows):
            tile_grid = grid_3857.window(row_off, col_off, height, width)
            t = tile_grid.transform
            bounds = (t.c, t.f + height * t.e, t.c + width * t.a, t.f)
            tile_path = tile_dir / f"tile_{idx:03d}.tif"
            if not tile_path.exists():
                logger.info(
                    "Exporting USGS NAIP Plus tile %d/%d (%d x %d px) with %d LockRaster IDs",
                    idx + 1,
                    len(windows),
                    width,
                    height,
                    len(selected_ids),
                )
                client.export_image(
                    bbox_3857=bounds,
                    width=width,
                    height=height,
                    lock_raster_ids=selected_ids,
                    output_path=tile_path,
                )
            self._validate_export(tile_path)
            raw_tile_paths.append(str(tile_path))

        band_count = self._write_final(raw_tile_paths, final_path, grid)
        if band_count < 4:
            logger.warning(
                "USGS NAIP Plus export has %d bands; the NIR band (N) is missing", band_count
            )
        # Uncovered pixels are 0 in every band; a dark pixel can be 0 in one band.
        valid_fraction = raster_valid_fraction(final_path, geom_4326, require="any")
        if valid_fraction <= 0.0:
            final_path.unlink(missing_ok=True)
            raise NoDataError(
                f"The USGS NAIP Plus export for {config.year} has no imagery inside the study "
                f"area (LockRaster IDs {selected_ids}). Use source='naip' (Earth Engine)."
            )
        if valid_fraction < LOW_COVERAGE_FRACTION:
            logger.warning(
                "USGS NAIP Plus imagery covers %.1f%% of the study area; the other pixels are 0 "
                "(nodata).",
                100.0 * valid_fraction,
            )

        self.last_metadata = {
            "AGRIBOUND_SOURCE": "usgs-naip-plus",
            "AGRIBOUND_VALUE_SCALE": SOURCE_REGISTRY["usgs-naip-plus"]["value_scale"],
            "AGRIBOUND_YEAR": config.year,
            "AGRIBOUND_EXPORT_CRS": grid.crs,
            "AGRIBOUND_RESOLUTION_M": resolution_m,
            "AGRIBOUND_IMAGE_YEARS": ",".join(
                sorted({str(c.year) for c in selected if c.year is not None})
            ),
            "AGRIBOUND_LOCK_RASTER_IDS": ",".join(str(i) for i in selected_ids),
            "AGRIBOUND_FOOTPRINT_COVERAGE": round(footprint_share, 6),
            "AGRIBOUND_VALID_FRACTION": round(valid_fraction, 6),
        }
        with rasterio.open(final_path, "r+") as dst:
            dst.update_tags(**{k: str(v) for k, v in self.last_metadata.items()})

        self._write_manifest(
            manifest_path=manifest_path,
            config=config,
            service_metadata=service_metadata,
            where=where,
            selected_ids=selected_ids,
            ranked_candidates=ranked_candidates,
            resolution_m=resolution_m,
            resolution_3857=resolution_3857,
            export_grid=grid,
            raw_tile_paths=raw_tile_paths,
            final_path=final_path,
            aoi_3857=aoi_3857,
            coverage={
                "footprint_coverage_of_extent": round(footprint_share, 6),
                "valid_fraction_in_study_area": round(valid_fraction, 6),
                "max_mosaic_image_count_reached": len(selected_ids) >= max_ids,
            },
        )
        import shutil

        shutil.rmtree(tile_dir, ignore_errors=True)
        logger.info("USGS NAIP Plus raster written to: %s", final_path)
        return str(final_path)

    def get_band_mapping(self, source: str) -> dict[str, str]:
        """Return the canonical band mapping (R, G, B, NIR -> R, G, B, N)."""
        info = SOURCE_REGISTRY.get(source, {})
        return dict(info.get("canonical_bands") or {})

    # ------------------------------------------------------------------
    # Query and selection
    # ------------------------------------------------------------------

    def _build_where_clause(
        self,
        config: AgriboundConfig,
        *,
        allow_year_fallback: bool = False,
        include_year: bool = True,
    ) -> str:
        parts = ["Category = 1"]

        state = getattr(config, "usgs_state", None)
        if state:
            state_clean = str(state).strip().upper().replace("'", "''")
            parts.append(f"State = '{state_clean}'")

        if not include_year:
            pass
        elif allow_year_fallback:
            parts.append(f"Year >= {int(config.year) - 1}")
            parts.append(f"Year <= {int(config.year) + 1}")
        else:
            parts.append(f"Year = {int(config.year)}")

        return " AND ".join(parts)

    def _no_imagery_error(
        self, client: USGSNAIPPlusClient, config: AgriboundConfig, aoi_3857: BaseGeometry
    ) -> NoDataError:
        """Build the 'no imagery' error, listing the years the service has for the area."""
        state = getattr(config, "usgs_state", None)
        where_any_year = self._build_where_clause(config, include_year=False)
        try:
            counts = client.query_year_counts(aoi_3857.bounds, where_any_year)
            years = ", ".join(f"{y} ({n} items)" for y, n in counts.items()) or "none"
        except Exception as exc:  # the listing is only diagnostic
            logger.warning("Could not list the available USGS NAIP Plus years: %s", exc)
            years = "unknown (year query failed)"
        requested = (
            f"{int(config.year) - 1}-{int(config.year) + 1}"
            if getattr(config, "usgs_allow_year_fallback", False)
            else str(config.year)
        )
        where = f" in state {state!r}" if state else ""
        return NoDataError(
            f"USGS NAIP Plus has no imagery for {requested}{where} over the study-area extent. "
            f"Years available for this area: {years}. The service holds only the latest "
            "vintage of each state; use source='naip' (Earth Engine, 2002-2023) for other years."
        )

    def _filter_candidates(
        self,
        candidates: list[USGSRasterCandidate],
        aoi_3857: BaseGeometry,
    ) -> list[USGSRasterCandidate]:
        filtered: list[USGSRasterCandidate] = []
        for candidate in candidates:
            if candidate.geometry is None or candidate.geometry.is_empty:
                continue
            if not candidate.geometry.intersects(aoi_3857):
                continue
            filtered.append(candidate)
        return filtered

    def _select_lock_raster_ids(
        self,
        candidates: list[USGSRasterCandidate],
        aoi_3857: BaseGeometry,
        *,
        target_year: int,
        max_ids: int,
    ) -> tuple[list[int], list[USGSRasterCandidate]]:
        ranked = sorted(
            candidates,
            key=lambda candidate: self._candidate_sort_key(candidate, aoi_3857, target_year),
        )

        selected: list[USGSRasterCandidate] = []
        covered = None

        for candidate in ranked:
            if len(selected) >= max_ids:
                break

            footprint = candidate.geometry.intersection(aoi_3857)
            if footprint.is_empty:
                continue

            if covered is None:
                selected.append(candidate)
                covered = footprint
                if covered.area >= aoi_3857.area * 0.999:
                    break
                continue

            additional = footprint.difference(covered)
            if additional.is_empty:
                continue

            selected.append(candidate)
            covered = covered.union(footprint)
            if covered.area >= aoi_3857.area * 0.999:
                break

        if not selected and ranked:
            selected = [ranked[0]]

        selected_ids = [candidate.object_id for candidate in selected]
        return selected_ids, ranked

    @staticmethod
    def _footprint_coverage(selected: list[USGSRasterCandidate], aoi_3857: BaseGeometry) -> float:
        """Share of *aoi_3857* (the export-grid outline) covered by the selected footprints."""
        from shapely.ops import unary_union

        if aoi_3857.is_empty or aoi_3857.area <= 0:
            return 0.0
        parts = [
            c.geometry.intersection(aoi_3857)
            for c in selected
            if c.geometry is not None and not c.geometry.is_empty
        ]
        if not parts:
            return 0.0
        return float(min(1.0, unary_union(parts).area / aoi_3857.area))

    def _candidate_sort_key(
        self,
        candidate: USGSRasterCandidate,
        aoi_3857: BaseGeometry,
        target_year: int,
    ) -> tuple[Any, ...]:
        intersection_area = 0.0
        if candidate.geometry is not None and not candidate.geometry.is_empty:
            intersection_area = candidate.geometry.intersection(aoi_3857).area

        year_distance = abs(candidate.year - target_year) if candidate.year is not None else 9999

        return (
            0 if candidate.category == 1 else 1,
            year_distance,
            0 if candidate.band_count and candidate.band_count >= 4 else 1,
            -intersection_area,
            candidate.resolution_value if candidate.resolution_value is not None else float("inf"),
            candidate.acquisition_date or "",
            candidate.object_id,
        )

    def _estimate_resolution_m(self, candidates: list[USGSRasterCandidate]) -> float:
        """Finest ground resolution (m) among *candidates* (the selected footprints).

        ``resolution_value`` is interpreted with ``resolution_units`` (metres
        or feet); candidates with other or missing units are ignored. Returns
        1.0 m, with a WARNING, when no candidate has a usable resolution.
        """
        values = []
        for candidate in candidates:
            value = candidate.resolution_value
            if value is None or value <= 0:
                continue
            units = (candidate.resolution_units or "").strip().lower().replace("-", "_")
            if units in _METRE_UNITS:
                values.append(float(value))
            elif units in _FOOT_UNITS:
                values.append(float(value) * 0.3048)
            else:
                logger.warning(
                    "Ignoring resolution %s %r of LockRaster %d (unknown units)",
                    value,
                    candidate.resolution_units,
                    candidate.object_id,
                )
        if values:
            return float(min(values))
        logger.warning(
            "No usable resolution in the selected USGS footprints; exporting at %.1f m",
            _DEFAULT_RESOLUTION_M,
        )
        return _DEFAULT_RESOLUTION_M

    # ------------------------------------------------------------------
    # Raster handling
    # ------------------------------------------------------------------

    def _validate_export(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Expected export not found: {path}")

        with rasterio.open(path) as src:
            if src.width <= 0 or src.height <= 0:
                raise RuntimeError(f"Invalid raster dimensions in {path}")
            if src.count < 3:
                raise RuntimeError(f"Expected at least 3 bands in {path}, found {src.count}")
            if src.crs is None:
                raise RuntimeError(f"Raster CRS is missing in {path}")

    @staticmethod
    def _write_final(tile_paths: list[str], out_path: Path, grid: Any) -> int:
        """Merge the exported tiles and resample them (bilinear) onto *grid*.

        Tiles are merged in their own CRS (``rasterio.merge.merge``), then read
        through a ``WarpedVRT`` on *grid* block by block; pixels that no tile
        covers are set to 0 (nodata). Pixels are not masked to the study-area
        polygons.

        Returns
        -------
        int
            Number of bands written.
        """
        from rasterio.enums import Resampling
        from rasterio.merge import merge
        from rasterio.vrt import WarpedVRT
        from rasterio.windows import Window

        mosaic_path = out_path.with_name(out_path.stem + "_mosaic.tif")
        part = out_path.with_name(out_path.name + ".part")
        try:
            if len(tile_paths) == 1:
                src_path = tile_paths[0]
            else:
                merge(
                    tile_paths,
                    dst_path=str(mosaic_path),
                    dst_kwds={"compress": "deflate", "tiled": True, "BIGTIFF": "IF_SAFER"},
                )
                src_path = str(mosaic_path)

            with rasterio.open(src_path) as src:
                count = src.count
                profile = {
                    "driver": "GTiff",
                    "dtype": "uint8",
                    "count": count,
                    "width": grid.width,
                    "height": grid.height,
                    "crs": grid.crs,
                    "transform": grid.transform,
                    "nodata": 0,
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                    "compress": "deflate",
                    "predictor": 2,
                    "BIGTIFF": "IF_SAFER",
                }
                with (
                    WarpedVRT(
                        src,
                        crs=grid.crs,
                        transform=grid.transform,
                        width=grid.width,
                        height=grid.height,
                        resampling=Resampling.bilinear,
                    ) as vrt,
                    rasterio.open(part, "w", **profile) as dst,
                ):
                    names = ["R", "G", "B", "N"][:count]
                    for i, name in enumerate(names, start=1):
                        dst.set_band_description(i, name)
                    rows = 1024
                    for r0 in range(0, grid.height, rows):
                        window = Window(0, r0, grid.width, min(rows, grid.height - r0))
                        data = vrt.read(window=window)
                        covered = vrt.read_masks(1, window=window) > 0
                        data[:, ~covered] = 0
                        dst.write(data.astype(np.uint8, copy=False), window=window)
            os.replace(part, out_path)
        finally:
            if part.exists():
                part.unlink()
            if mosaic_path.exists():
                mosaic_path.unlink()
        return count

    def _write_manifest(
        self,
        *,
        manifest_path: Path,
        config: AgriboundConfig,
        service_metadata: dict[str, Any],
        where: str,
        selected_ids: list[int],
        ranked_candidates: list[USGSRasterCandidate],
        resolution_m: float,
        resolution_3857: float,
        export_grid: Any,
        raw_tile_paths: list[str],
        final_path: Path,
        aoi_3857: BaseGeometry,
        coverage: dict[str, Any] | None = None,
    ) -> None:
        manifest = {
            "source": "usgs-naip-plus",
            "selection_version": self.SELECTION_VERSION,
            "service_url": getattr(config, "usgs_service_url", DEFAULT_USGS_NAIPPLUS_URL),
            "year": config.year,
            "usgs_state": getattr(config, "usgs_state", None),
            "allow_year_fallback": getattr(config, "usgs_allow_year_fallback", False),
            "where": where,
            "selected_lock_raster_ids": selected_ids,
            "resolution_m": resolution_m,
            "export_resolution_3857_units": resolution_3857,
            "export_crs": export_grid.crs,
            "export_shape": [export_grid.height, export_grid.width],
            "export_transform": list(export_grid.crs_transform),
            "query_bounds_3857": [float(v) for v in aoi_3857.bounds],
            # The tiles are deleted once the final raster is written.
            "n_export_tiles": len(raw_tile_paths),
            "coverage": dict(coverage or {}),
            "final_raster_path": str(final_path),
            "service_limits": {
                "maxImageWidth": service_metadata.get("maxImageWidth"),
                "maxImageHeight": service_metadata.get("maxImageHeight"),
                "maxMosaicImageCount": service_metadata.get("maxMosaicImageCount"),
            },
            "ranked_candidates": [
                {
                    "object_id": candidate.object_id,
                    "year": candidate.year,
                    "state": candidate.state,
                    "acquisition_date": candidate.acquisition_date,
                    "resolution_value": candidate.resolution_value,
                    "resolution_units": candidate.resolution_units,
                    "band_count": candidate.band_count,
                    "category": candidate.category,
                    "name": candidate.name,
                    "download_url": candidate.download_url,
                    "bounds_3857": (
                        list(candidate.geometry.bounds)
                        if candidate.geometry is not None and not candidate.geometry.is_empty
                        else None
                    ),
                }
                for candidate in ranked_candidates
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
