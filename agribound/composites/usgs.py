"""USGS ImageServer-backed composite builders."""

from __future__ import annotations

import json
import logging
import math
from hashlib import sha1
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from shapely.geometry.base import BaseGeometry

from agribound.clients.usgs_naip_plus import (
    DEFAULT_USGS_NAIPPLUS_URL,
    USGSNAIPPlusClient,
    USGSRasterCandidate,
)
from agribound.composites.base import SOURCE_REGISTRY, CompositeBuilder
from agribound.config import AgriboundConfig
from agribound.io.raster import clip_raster_to_geometry
from agribound.io.vector import get_study_area_geometry, read_study_area

logger = logging.getLogger(__name__)


class USGSNAIPPlusCompositeBuilder(CompositeBuilder):
    """Build local rasters from the USGS NAIP Plus ImageServer."""

    SELECTION_VERSION = 1

    def build(self, config: AgriboundConfig) -> str:
        """Download, mosaic, and clip a USGS NAIP Plus raster for the study area."""
        if not config.study_area:
            raise ValueError("study_area is required when source='usgs-naip-plus'")

        study_gdf = read_study_area(config.study_area)
        if study_gdf.empty:
            raise ValueError("Study area is empty")

        study_3857 = study_gdf.to_crs("EPSG:3857")
        aoi_3857 = get_study_area_geometry(study_3857)
        if aoi_3857.is_empty:
            raise ValueError("Study area geometry is empty after projection to EPSG:3857")

        cache_dir = config.get_working_dir()
        run_key = self._fingerprint(config, aoi_3857)
        final_path = cache_dir / f"usgs_naip_plus_{config.year}_{run_key}.tif"
        manifest_path = final_path.with_suffix(".json")
        aoi_path = cache_dir / f"usgs_naip_plus_{run_key}_study_area_3857.geojson"

        if final_path.exists() and manifest_path.exists():
            logger.info("Using cached USGS NAIP Plus raster: %s", final_path)
            return str(final_path)

        if not aoi_path.exists():
            self._write_aoi_geojson(study_3857, aoi_path)

        client = USGSNAIPPlusClient(
            service_url=getattr(config, "usgs_service_url", DEFAULT_USGS_NAIPPLUS_URL),
            timeout_s=getattr(config, "usgs_timeout_s", 120),
            retries=getattr(config, "usgs_retries", 3),
        )

        service_metadata = client.get_service_metadata()
        where = self._build_where_clause(config)
        candidates = client.query_candidates(aoi_3857.bounds, where, out_sr=3857)
        candidates = self._filter_candidates(candidates, aoi_3857)

        if not candidates and getattr(config, "usgs_allow_year_fallback", False):
            where = self._build_where_clause(config, allow_year_fallback=True)
            candidates = client.query_candidates(aoi_3857.bounds, where, out_sr=3857)
            candidates = self._filter_candidates(candidates, aoi_3857)

        if not candidates:
            raise RuntimeError(
                f"No USGS NAIP Plus candidates found for year={config.year}, "
                f"state={getattr(config, 'usgs_state', None)!r}, "
                f"bounds={tuple(round(v, 2) for v in aoi_3857.bounds)}"
            )

        selected_ids, ranked_candidates = self._select_lock_raster_ids(
            candidates,
            aoi_3857,
            target_year=int(config.year),
            max_ids=int(service_metadata.get("maxMosaicImageCount", 50)),
        )
        if not selected_ids:
            raise RuntimeError("Candidate selection returned no LockRaster IDs")

        export_max_width = int(service_metadata.get("maxImageWidth", 4000))
        export_max_height = int(service_metadata.get("maxImageHeight", 4000))
        export_max_px = min(
            int(getattr(config, "tile_size", 10_000)),
            export_max_width,
            export_max_height,
        )

        resolution_m = self._estimate_resolution_m(ranked_candidates)
        export_tiles = self._compute_export_tiles(
            bounds_3857=aoi_3857.bounds,
            resolution_m=resolution_m,
            max_tile_px=export_max_px,
        )

        raw_tile_paths: list[str] = []
        for idx, tile in enumerate(export_tiles):
            tile_path = cache_dir / f"usgs_naip_plus_{run_key}_tile_{idx:03d}.tif"
            if not tile_path.exists():
                logger.info(
                    "Exporting USGS NAIP Plus tile %d/%d with %d LockRaster IDs",
                    idx + 1,
                    len(export_tiles),
                    len(selected_ids),
                )
                client.export_image(
                    bbox_3857=tile["bounds"],
                    width=tile["width_px"],
                    height=tile["height_px"],
                    lock_raster_ids=selected_ids,
                    output_path=tile_path,
                )
            self._validate_export(tile_path)
            raw_tile_paths.append(str(tile_path))

        if len(raw_tile_paths) == 1:
            raw_mosaic_path = Path(raw_tile_paths[0])
        else:
            raw_mosaic_path = cache_dir / f"usgs_naip_plus_{run_key}_mosaic_raw.tif"
            if not raw_mosaic_path.exists():
                self._mosaic_tiles(raw_tile_paths, str(raw_mosaic_path))

        # Geometry is already in EPSG:3857, which matches the export request.
        clip_raster_to_geometry(str(raw_mosaic_path), str(final_path), aoi_3857)

        self._validate_export(final_path)

        self._write_manifest(
            manifest_path=manifest_path,
            config=config,
            service_metadata=service_metadata,
            where=where,
            selected_ids=selected_ids,
            ranked_candidates=ranked_candidates,
            resolution_m=resolution_m,
            raw_tile_paths=raw_tile_paths,
            final_path=final_path,
            aoi_3857=aoi_3857,
        )

        logger.info("USGS NAIP Plus composite written to: %s", final_path)
        return str(final_path)

    def get_band_mapping(self, source: str) -> dict[str, str]:
        """Return canonical band mapping for the USGS source."""
        info = SOURCE_REGISTRY.get(source, {})
        return info.get("canonical_bands") or {}

    def _build_where_clause(
        self,
        config: AgriboundConfig,
        *,
        allow_year_fallback: bool = False,
    ) -> str:
        parts = ["Category = 1"]

        state = getattr(config, "usgs_state", None)
        if state:
            state_clean = str(state).strip().upper().replace("'", "''")
            parts.append(f"State = '{state_clean}'")

        if allow_year_fallback:
            parts.append(f"Year >= {int(config.year) - 1}")
            parts.append(f"Year <= {int(config.year) + 1}")
        else:
            parts.append(f"Year = {int(config.year)}")

        return " AND ".join(parts)

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
        resolution_values = [
            candidate.resolution_value
            for candidate in candidates
            if candidate.resolution_value is not None and candidate.resolution_value > 0
        ]
        if resolution_values:
            return float(min(resolution_values))
        return 1.0

    def _compute_export_tiles(
        self,
        *,
        bounds_3857: tuple[float, float, float, float],
        resolution_m: float,
        max_tile_px: int,
    ) -> list[dict[str, Any]]:
        xmin, ymin, xmax, ymax = bounds_3857
        width_m = max(xmax - xmin, resolution_m)
        height_m = max(ymax - ymin, resolution_m)

        width_px = max(1, int(math.ceil(width_m / resolution_m)))
        height_px = max(1, int(math.ceil(height_m / resolution_m)))

        n_cols = max(1, int(math.ceil(width_px / max_tile_px)))
        n_rows = max(1, int(math.ceil(height_px / max_tile_px)))

        x_edges = np.linspace(xmin, xmax, n_cols + 1)
        y_edges = np.linspace(ymin, ymax, n_rows + 1)

        tiles: list[dict[str, Any]] = []
        for row in range(n_rows):
            for col in range(n_cols):
                txmin = float(x_edges[col])
                txmax = float(x_edges[col + 1])
                tymin = float(y_edges[row])
                tymax = float(y_edges[row + 1])

                tile_width_px = max(1, int(math.ceil((txmax - txmin) / resolution_m)))
                tile_height_px = max(1, int(math.ceil((tymax - tymin) / resolution_m)))

                tiles.append(
                    {
                        "bounds": (txmin, tymin, txmax, tymax),
                        "width_px": min(tile_width_px, max_tile_px),
                        "height_px": min(tile_height_px, max_tile_px),
                    }
                )
        return tiles

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
        raw_tile_paths: list[str],
        final_path: Path,
        aoi_3857: BaseGeometry,
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
            "estimated_resolution_m": resolution_m,
            "aoi_bounds_3857": [float(v) for v in aoi_3857.bounds],
            "raw_tile_paths": raw_tile_paths,
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

    def _write_aoi_geojson(self, gdf_3857: gpd.GeoDataFrame, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()
        gdf_3857.to_file(out_path, driver="GeoJSON")

    def _fingerprint(self, config: AgriboundConfig, aoi_3857: BaseGeometry) -> str:
        payload = {
            "source": config.source,
            "year": config.year,
            "study_bounds_3857": [round(v, 3) for v in aoi_3857.bounds],
            "usgs_state": getattr(config, "usgs_state", None),
            "allow_year_fallback": getattr(config, "usgs_allow_year_fallback", False),
            "selection_version": self.SELECTION_VERSION,
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha1(text.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _mosaic_tiles(tile_paths: list[str], output_path: str) -> None:
        """Mosaic multiple GeoTIFF tiles into a single file."""
        datasets = [rasterio.open(path) for path in tile_paths]
        try:
            mosaic, transform = merge(datasets)
            mosaic = np.where(np.isfinite(mosaic), mosaic, 0)

            meta = datasets[0].meta.copy()
            meta.update(
                {
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": transform,
                    "count": mosaic.shape[0],
                    "compress": "lzw",
                    "BIGTIFF": "YES",
                }
            )

            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(mosaic)
        finally:
            for dataset in datasets:
                dataset.close()
