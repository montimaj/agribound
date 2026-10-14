"""USGS NAIP Plus ArcGIS ImageServer client.

A small, dependency-light client for the USGS NAIP Plus ImageServer
(``https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer``):
query catalogue items and per-year counts, and export AOI-bounded TIFFs for
:class:`agribound.composites.usgs.USGSNAIPPlusCompositeBuilder`.

The service holds only the latest NAIP/HRO vintage of each state (years
2012-2023 as of 2026-09; dense only from 2019), not the historical NAIP
archive; its native pixel size is 0.3 m and the state vintages are 0.3-0.6 m.

Requests and downloads are retried after transient failures: HTTP 429 and
5xx responses (also ArcGIS error payloads with those codes), timeouts,
connection errors and truncated or garbled responses. The n-th retry waits
``min(RETRY_MAX_WAIT_S, RETRY_BASE_WAIT_S * 2 ** (n - 1))`` seconds plus up
to ``RETRY_JITTER`` of that at random (4, 8, 16, 32 and 60 s for the default
5 retries) and is logged at WARNING. Other HTTP errors (4xx) are not retried.
"""

from __future__ import annotations

import errno
import http.client
import json
import logging
import random
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

DEFAULT_USGS_NAIPPLUS_URL = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer"
)
#: Retries of a request or download after a transient failure (6 attempts in all).
DEFAULT_RETRIES = 5
#: Wait before the first retry (s); it doubles with each retry up to RETRY_MAX_WAIT_S.
RETRY_BASE_WAIT_S = 4.0
RETRY_MAX_WAIT_S = 60.0
#: Random extra wait, as a share of the wait, so that parallel clients spread out.
RETRY_JITTER = 0.25

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class USGSImageServerError(RuntimeError):
    """Raised when the USGS ImageServer request or response is invalid."""


class _ArcGISError(USGSImageServerError):
    """An ArcGIS error payload (``{"error": {...}}``); *code* is its error code."""

    def __init__(self, message: str, code: int | None) -> None:
        super().__init__(message)
        self.code = code


#: Local file-system failures (e.g. writing the download's .part file): never transient.
_LOCAL_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.ENOSPC, errno.EROFS, errno.EDQUOT})


def _transient(exc: BaseException) -> bool:
    """Whether a failed request may succeed when repeated (see the module docstring)."""
    if isinstance(exc, HTTPError | _ArcGISError):  # HTTPError before OSError: it is one
        return exc.code is not None and (exc.code == 429 or exc.code >= 500)
    if isinstance(exc, PermissionError | IsADirectoryError | NotADirectoryError):
        return False
    if isinstance(exc, OSError) and exc.errno in _LOCAL_ERRNOS:
        return False
    return isinstance(exc, OSError | http.client.HTTPException | ValueError)


def _retry_wait(retry: int) -> float:
    """Seconds to wait before retry number *retry* (1, 2, ...), jitter included."""
    wait_s = min(RETRY_MAX_WAIT_S, RETRY_BASE_WAIT_S * 2 ** (retry - 1))
    return wait_s * (1 + random.uniform(0, RETRY_JITTER))


def _failure_text(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, _ArcGISError):
        return f"ArcGIS error {exc}"
    return type(exc).__name__


@dataclass(frozen=True)
class USGSRasterCandidate:
    """Catalog item returned by the USGS NAIP Plus ImageServer."""

    object_id: int
    year: int | None
    state: str | None
    acquisition_date: str | None
    resolution_value: float | None
    resolution_units: str | None
    band_count: int | None
    category: int | None
    name: str | None
    download_url: str | None
    geometry: BaseGeometry | None
    attributes: dict[str, Any]


class USGSNAIPPlusClient:
    """Thin ArcGIS ImageServer client for USGS NAIP Plus."""

    def __init__(
        self, service_url: str, timeout_s: int = 120, retries: int = DEFAULT_RETRIES
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries

    def get_service_metadata(self) -> dict[str, Any]:
        """Fetch service metadata."""
        return self._request_json("", {"f": "json"})

    def query_object_ids(
        self,
        bounds_3857: tuple[float, float, float, float],
        where: str,
    ) -> list[int]:
        """Query matching object IDs for a 3857 AOI envelope."""
        params = {
            "f": "json",
            "where": where,
            "geometry": self._format_envelope(bounds_3857),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 3857,
            "spatialRel": "esriSpatialRelIntersects",
            "returnIdsOnly": "true",
        }
        payload = self._request_json("/query", params)
        object_ids = payload.get("objectIds") or []
        return sorted(int(v) for v in object_ids)

    def query_year_counts(
        self,
        bounds_3857: tuple[float, float, float, float],
        where: str,
    ) -> dict[int, int]:
        """Count catalogue items per ``Year`` intersecting a 3857 envelope.

        Uses an ``outStatistics`` query (count of ``OBJECTID`` grouped by
        ``Year``), so no footprints are transferred.

        Returns
        -------
        dict[int, int]
            ``{year: n_items}``, sorted by year.
        """
        params = {
            "f": "json",
            "where": where,
            "geometry": self._format_envelope(bounds_3857),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 3857,
            "spatialRel": "esriSpatialRelIntersects",
            "outStatistics": json.dumps(
                [
                    {
                        "statisticType": "count",
                        "onStatisticField": "OBJECTID",
                        "outStatisticFieldName": "n_items",
                    }
                ],
                separators=(",", ":"),
            ),
            "groupByFieldsForStatistics": "Year",
            "returnGeometry": "false",
        }
        payload = self._request_json("/query", params)
        counts: dict[int, int] = {}
        for feature in payload.get("features") or []:
            attrs = feature.get("attributes") or {}
            year = self._to_int(self._first_not_none(attrs, "Year", "YEAR", "year"))
            n = self._to_int(self._first_not_none(attrs, "n_items", "N_ITEMS"))
            if year is not None and n is not None:
                counts[year] = counts.get(year, 0) + n
        return dict(sorted(counts.items()))

    def query_candidates(
        self,
        bounds_3857: tuple[float, float, float, float],
        where: str,
        *,
        out_sr: int = 3857,
        batch_size: int = 50,
    ) -> list[USGSRasterCandidate]:
        """Query full catalog items intersecting a 3857 AOI envelope."""
        object_ids = self.query_object_ids(bounds_3857, where)
        if not object_ids:
            return []

        features: list[dict[str, Any]] = []
        for batch in self._chunked(object_ids, batch_size):
            params = {
                "f": "json",
                "objectIds": ",".join(str(v) for v in batch),
                "outFields": ",".join(
                    [
                        "OBJECTID",
                        "Name",
                        "State",
                        "Year",
                        "Category",
                        "download_url",
                        "acquisition_date",
                        "resolution_value",
                        "resolution_units",
                        "band_count",
                        "sensor_type",
                        "projection_name",
                        "projection_zone",
                        "datum",
                    ]
                ),
                "returnGeometry": "true",
                "outSR": out_sr,
            }
            payload = self._request_json("/query", params)
            features.extend(payload.get("features") or [])

        candidates = [self._feature_to_candidate(feature) for feature in features]
        candidates = [candidate for candidate in candidates if candidate.object_id is not None]
        return sorted(candidates, key=lambda c: c.object_id)

    def export_image(
        self,
        *,
        bbox_3857: tuple[float, float, float, float],
        width: int,
        height: int,
        lock_raster_ids: list[int],
        output_path: Path,
        compression: str = "LZ77",
    ) -> dict[str, Any]:
        """Export a TIFF of *bbox_3857* using a deterministic LockRaster mosaic rule.

        The request is in EPSG:3857 (``bboxSR``/``imageSR``) with
        ``size=width,height``; the service resamples the selected rasters with
        bilinear interpolation.
        """
        if not lock_raster_ids:
            raise ValueError("lock_raster_ids must be non-empty")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")

        mosaic_rule = {
            "mosaicMethod": "esriMosaicLockRaster",
            "lockRasterIds": [int(v) for v in lock_raster_ids],
            "ascending": True,
            "mosaicOperation": "MT_FIRST",
        }

        params = {
            "f": "json",
            "bbox": self._format_envelope(bbox_3857),
            "bboxSR": 3857,
            "imageSR": 3857,
            "size": f"{int(width)},{int(height)}",
            "format": "tiff",
            "pixelType": "UNKNOWN",
            "interpolation": "RSP_BilinearInterpolation",
            "compression": compression,
            "adjustAspectRatio": "false",
            "validateExtent": "true",
            "mosaicRule": json.dumps(mosaic_rule, separators=(",", ":")),
        }

        payload = self._request_json("/exportImage", params)
        href = payload.get("href")
        if not href:
            raise USGSImageServerError(
                f"exportImage response missing 'href': {json.dumps(payload)[:500]}"
            )

        self._download_file(href, output_path)
        return payload

    def _with_retries(self, attempt: Callable[[], _T], what: str, url: str) -> _T:
        """Return ``attempt()``, repeating it up to ``self.retries`` times after a transient
        failure (see the module docstring); each retry is logged at WARNING.

        An ArcGIS error payload is raised as it is; another failed request (an HTTP 4xx
        at once, a transient failure after the last retry) as ``USGSImageServerError``
        ``"Failed <what>: <url>"``.
        """
        for n in range(self.retries + 1):
            try:
                return attempt()
            except _ArcGISError as exc:
                if n >= self.retries or not _transient(exc):
                    raise
                failure: BaseException = exc
            except (OSError, http.client.HTTPException, ValueError) as exc:
                if n >= self.retries or not _transient(exc):
                    raise USGSImageServerError(f"Failed {what}: {url}") from exc
                failure = exc
            wait_s = _retry_wait(n + 1)
            logger.warning(
                "USGS %s failed (%s). Retrying in %.0f s [%d/%d]: %s",
                what,
                _failure_text(failure),
                wait_s,
                n + 1,
                self.retries,
                url,
            )
            time.sleep(wait_s)
        raise AssertionError("unreachable")  # the last attempt returns or raises

    def _request_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """Issue a GET request and parse JSON, retrying transient failures."""
        url = self._build_url(path, params)

        def attempt() -> dict[str, Any]:
            with urlopen(url, timeout=self.timeout_s) as response:
                raw = response.read()
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict) and "error" in payload:
                error = payload["error"] if isinstance(payload["error"], dict) else {}
                raise _ArcGISError(self._format_arcgis_error(error), self._error_code(error))
            if not isinstance(payload, dict):
                raise USGSImageServerError(f"Expected JSON object response from {url}")
            return payload

        return self._with_retries(attempt, "request", url)

    def _download_file(self, url: str, output_path: Path) -> None:
        """Download a file atomically, retrying transient failures.

        The response is streamed to ``<output_path>.part`` with
        ``urlopen(url, timeout=self.timeout_s)`` (the timeout applies to the
        connection and to each read, so a stalled transfer fails and is
        retried) and renamed when complete.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".part")

        def attempt() -> None:
            if tmp_path.exists():
                tmp_path.unlink()
            with urlopen(url, timeout=self.timeout_s) as response, open(tmp_path, "wb") as fh:
                shutil.copyfileobj(response, fh, length=1 << 20)
            tmp_path.replace(output_path)

        try:
            self._with_retries(attempt, "download", url)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _error_code(error: dict[str, Any]) -> int | None:
        """The code of an ArcGIS error payload, or None."""
        try:
            return int(error.get("code"))
        except (TypeError, ValueError):
            return None

    def _feature_to_candidate(self, feature: dict[str, Any]) -> USGSRasterCandidate:
        """Convert an ArcGIS feature JSON object into a typed candidate."""
        attributes = dict(feature.get("attributes") or {})
        geometry_json = feature.get("geometry")
        geometry = self._esri_geometry_to_shapely(geometry_json) if geometry_json else None

        object_id = self._first_not_none(attributes, "OBJECTID", "ObjectID", "objectid")
        if object_id is None:
            raise USGSImageServerError("Feature missing OBJECTID")

        return USGSRasterCandidate(
            object_id=int(object_id),
            year=self._to_int(self._first_not_none(attributes, "Year", "year")),
            state=self._to_str(self._first_not_none(attributes, "State", "state")),
            acquisition_date=self._epoch_millis_to_iso(
                self._first_not_none(attributes, "acquisition_date", "AcquisitionDate")
            ),
            resolution_value=self._to_float(
                self._first_not_none(attributes, "resolution_value", "ResolutionValue")
            ),
            resolution_units=self._to_str(
                self._first_not_none(attributes, "resolution_units", "ResolutionUnits")
            ),
            band_count=self._to_int(self._first_not_none(attributes, "band_count", "BandCount")),
            category=self._to_int(self._first_not_none(attributes, "Category", "category")),
            name=self._to_str(self._first_not_none(attributes, "Name", "raster_name", "name")),
            download_url=self._to_str(self._first_not_none(attributes, "download_url")),
            geometry=geometry,
            attributes=attributes,
        )

    def _build_url(self, path: str, params: dict[str, Any]) -> str:
        query = urlencode({k: v for k, v in params.items() if v is not None})
        return f"{self.service_url}{path}?{query}"

    @staticmethod
    def _format_envelope(bounds: tuple[float, float, float, float]) -> str:
        xmin, ymin, xmax, ymax = bounds
        return f"{xmin},{ymin},{xmax},{ymax}"

    @staticmethod
    def _format_arcgis_error(error_payload: dict[str, Any]) -> str:
        code = error_payload.get("code")
        message = error_payload.get("message", "ArcGIS service error")
        details = error_payload.get("details") or []
        detail_text = " | ".join(str(v) for v in details if v)
        if detail_text:
            return f"[{code}] {message}: {detail_text}"
        return f"[{code}] {message}"

    @staticmethod
    def _chunked(values: list[int], size: int) -> Iterable[list[int]]:
        if size <= 0:
            raise ValueError("size must be positive")
        for start in range(0, len(values), size):
            yield values[start : start + size]

    @staticmethod
    def _first_not_none(mapping: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return None

    @staticmethod
    def _to_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    @staticmethod
    def _to_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _epoch_millis_to_iso(value: Any) -> str | None:
        if value is None or value == "":
            return None
        try:
            millis = int(value)
        except (TypeError, ValueError):
            return str(value)
        dt = datetime.fromtimestamp(millis / 1000.0, tz=UTC)
        return dt.isoformat()

    @staticmethod
    def _esri_rings_to_polygon(rings: list[list[list[float]]]) -> BaseGeometry:
        """Build a (multi)polygon from Esri JSON rings.

        Esri polygons list outer rings clockwise and holes counter-clockwise.
        Holes are subtracted from the outer ring that contains them. If no ring
        is clockwise (orientation not followed), every ring is treated as an
        outer ring and they are unioned.
        """
        from shapely.geometry import LinearRing

        shells: list[Polygon] = []
        holes: list[Polygon] = []
        for ring in rings:
            if len(ring) < 4:
                continue
            coords = [(float(x), float(y)) for x, y in ring]
            poly = Polygon(coords)
            if poly.is_empty or poly.area == 0:
                continue
            (holes if LinearRing(coords).is_ccw else shells).append(poly)
        if not shells:
            shells, holes = holes, []
        if not shells:
            raise USGSImageServerError("Esri polygon geometry contained no valid rings")

        parts: list[BaseGeometry] = []
        for shell in shells:
            if not shell.is_valid:
                shell = shell.buffer(0)
            inner = [h for h in holes if shell.contains(h.representative_point())]
            for hole in inner:
                shell = shell.difference(hole if hole.is_valid else hole.buffer(0))
            if not shell.is_empty:
                parts.append(shell)
        if not parts:
            raise USGSImageServerError("Esri polygon geometry contained no valid rings")
        merged = parts[0] if len(parts) == 1 else unary_union(parts)
        if not merged.is_valid:
            merged = merged.buffer(0)
        return merged

    @classmethod
    def _esri_geometry_to_shapely(cls, geometry: dict[str, Any]) -> BaseGeometry:
        """Convert Esri JSON geometry into a shapely geometry."""
        if {"xmin", "ymin", "xmax", "ymax"}.issubset(geometry):
            return box(
                float(geometry["xmin"]),
                float(geometry["ymin"]),
                float(geometry["xmax"]),
                float(geometry["ymax"]),
            )

        if "x" in geometry and "y" in geometry:
            return Point(float(geometry["x"]), float(geometry["y"]))

        if "points" in geometry:
            return MultiPoint([(float(x), float(y)) for x, y in geometry["points"]])

        if "paths" in geometry:
            paths = geometry["paths"] or []
            if len(paths) == 1:
                return LineString([(float(x), float(y)) for x, y in paths[0]])
            return MultiLineString(
                [[(float(x), float(y)) for x, y in path] for path in paths if path]
            )

        if "rings" in geometry:
            return cls._esri_rings_to_polygon(geometry["rings"] or [])

        raise USGSImageServerError(f"Unsupported Esri geometry keys: {sorted(geometry.keys())}")


__all__ = [
    "DEFAULT_RETRIES",
    "DEFAULT_USGS_NAIPPLUS_URL",
    "USGSImageServerError",
    "USGSRasterCandidate",
    "USGSNAIPPlusClient",
]
