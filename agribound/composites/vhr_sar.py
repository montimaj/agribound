"""
Open very-high-resolution (VHR) SAR amplitude composites: Umbra and Capella open data.

Sources ``"umbra"`` and ``"capella"`` (:data:`agribound.registry.VHR_SAR_SOURCES`)
build a composite locally, without Earth Engine, from the public AWS buckets of
the Umbra Open Data Program (``s3://umbra-open-data-catalog``) and of Capella
Space (``s3://capella-open-data``), both licensed CC BY 4.0. One module serves
both through provider adapters, so that licence, credit and provenance stay
per vendor. Only amplitude products are read (Umbra GEC, Capella GEO); complex
products (CPHD, SICD, SLC, SIDD, CSI) are never downloaded.

The recipe (:data:`VHR_SAR_RECIPE_TAG`) was fixed in advance from a prototype
at four sites and a Capella pilot (2026-10-08):

1. **Index.** The bucket listing plus the per-collect metadata: the Umbra
   ``.stac.v2.json`` sidecars, or the Capella static STAC items plus the head of
   each GEO ``_extended.json``. Umbra copies of one collect are merged by
   ``umbra:collect_id`` (the copy with the newest GEC, whose GEC and sidecar
   come from the same folder); polarization and geometry come from the sidecar
   (``METADATA.json`` disagreed with it for 36 of 204 collects at Centerfield),
   and folders without one are not used. Capella incidence comes at full
   precision from ``_extended.json`` (STAC rounds it to 0.1 degree), with the
   calibration label and the amplitude scale factor. Records dated after the
   index snapshot (Umbra lists collects dated 2035 and 2036) are dropped. The
   index is cached with its snapshot time (:func:`load_index`); selection is
   deterministic for one snapshot.
2. **Selection** (:func:`select_collects`). Collects in the date window whose
   footprint covers the study area, grouped by orbit pass, look side and
   polarization (HH and VV are never mixed); in each group the largest subset
   within :data:`INCIDENCE_TOLERANCE_DEG` of incidence and
   :data:`VIEW_AZIMUTH_TOLERANCE_DEG` of view azimuth (circular); the largest
   such subset wins, and up to ``vhr_sar_max_collects`` of its collects, spread
   over the window, are used. ``vhr_sar_collect_ids`` pins collects instead.
3. **Products.** Umbra GEC (8-bit, a rotated EPSG:4326 grid with RPCs) is
   orthorectified with its RPCs and the Copernicus GLO-30 DEM (GLO-90 where
   GLO-30 has no public tile) converted to ellipsoidal heights with EGM2008
   (:func:`build_dem_hae`); without RPCs its
   geotransform is used, with a WARNING. Capella GEO (terrain-corrected UTM) is
   resampled; it is corrected with a 30 m DSM, which does not resolve trees
   (a WARNING says so). Products are averaged onto the grid (Umbra: the mean
   of the log-scaled DN; Capella: the mean intensity, GDAL ``rms`` of DN).
4. **Radiometry.** Capella: sigma0 dB = 20 log10(scale_factor x DN), stored
   as dB + :data:`CAPELLA_DB_OFFSET`; ``vhr_sar_calibration="full"`` drops
   collects whose calibration is ``"limited"`` or ``"partial"``. Umbra: the DN
   behaves as a log scale (about :data:`UMBRA_DN_PER_DB` DN per dB; DN 1 holds
   23-40 % of the pixels) and is treated as uncalibrated. Each date is
   level-normalised (its median over the area all dates cover is shifted to
   the mean of the dates' medians) before the median; there is no beam
   flattening (it lowered F1 in the prototype).
5. **Co-registration** (``vhr_sar_coregister``). Every date is registered to a
   reference date (normalised cross-correlation of edge maps on tiles,
   falling back to 2 m and 4 m block means for single-look 1 m products), then
   the stack to an optical base of 1 m or finer (``vhr_sar_coregister_base``,
   or USGS NAIP Plus where it has imagery); each date is then warped once with
   its total shift. A registration counts only when a majority of the tile
   estimates agree (:func:`register_adaptive`); otherwise the build stops with
   an error. Offsets after RPC + DEM were 2.3-25.9 m (median 6.8 m) in
   the prototype, and 0.9 m or less after this correction. Sentinel-2 is never
   used as the base (it was 4.5-7 m off at two sites).
6. **Output.** float32 on the study-area grid at ``vhr_sar_resolution_m``
   (default 1 m; 0.5 m scored worse), 0 = nodata: band ``MEDIAN`` (read as R, G
   and B), and for ``vhr_sar_bands="median+dates"`` also ``PAIR_MEAN`` (the
   mean of the two dates closest in time) and one band per date.

Delineate-Anything reads these composites best with
``engine_params={"processing_gsd_m": 3, "merge_tile_pieces": False}`` (not the
engine defaults; see :data:`DELINEATE_ANYTHING_PARAMS`).

Every composite carries ``AGRIBOUND_ATTRIBUTION`` (the vendor credit),
``AGRIBOUND_LICENSE``/``AGRIBOUND_LICENSE_URL``, ``AGRIBOUND_DATA_CITATION``
(the AWS Registry of Open Data citation with the access date),
``AGRIBOUND_CHANGES`` (the note of changes), ``AGRIBOUND_SAR_COLLECTS`` (one
JSON entry per collect) and ``AGRIBOUND_INDEX_SNAPSHOT``; a JSON manifest is
written next to it. Its cache key holds the recipe, the settings (with the
path, size and modification time of ``vhr_sar_coregister_base``), the sorted
collect IDs with their ETags and the index snapshot. Downloads and DEM builds
hold a lock file, so processes that share ``vhr_sar_cache_dir`` fetch each
product once.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import datetime as _dt
import errno
import glob
import gzip
import http.client
import json
import logging
import math
import os
import re
import threading
import time
import urllib.parse
import warnings
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from agribound.composites.base import SOURCE_REGISTRY, CompositeBuilder, NoDataError
from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recipe constants
# ---------------------------------------------------------------------------

#: Revision of the amplitude recipe (part of the cache key; bump when it changes).
VHR_SAR_RECIPE_VERSION = "1"
#: Recipe tag written to ``AGRIBOUND_SAR_RECIPE``.
VHR_SAR_RECIPE_TAG = f"vhr-sar-amplitude-{VHR_SAR_RECIPE_VERSION}"
#: Maximum incidence-angle spread (degrees) of the collects of one composite.
INCIDENCE_TOLERANCE_DEG = 5.0
#: Maximum circular view-azimuth spread (degrees) of the collects of one composite.
VIEW_AZIMUTH_TOLERANCE_DEG = 25.0
#: Share of the study area a collect footprint must cover. The metadata footprints are
#: approximate: the prototype's evaluation areas, drawn inside the image data, were
#: 99.0-99.99 % inside the footprints of the collects used.
MIN_FOOTPRINT_COVER = 0.98
#: Offset added to Capella sigma0 (dB) so that valid pixels are positive (0 = nodata).
CAPELLA_DB_OFFSET = 50.0
#: Approximate Umbra GEC digital numbers per dB (prototype; documentation only).
UMBRA_DN_PER_DB = 6.6
#: What the Umbra prototype measured for single dates (for the single-date messages):
#: Delineate-Anything F1 with its default settings, and read at 2 m without tile
#: merging against the median of the same area (REPORT section 3 of the prototype).
SINGLE_DATE_NOTE = (
    "in the Umbra prototype single dates scored F1 0.00-0.08 with Delineate-Anything's "
    "default settings; read at 2 m without tile merging they scored 0.36-0.40 near UC Davis, "
    "0.15-0.25 near Purdue and 0.16-0.27 on Mallorca, against 0.42, 0.22 and 0.31 for the "
    "median of the dates, and about 0 at Centerfield"
)
#: Margin (m) around the study-area grid used for co-registration.
REGISTRATION_MARGIN_M = 500.0
#: Co-registration consensus (see :func:`estimate_shift` and :func:`register_adaptive`):
#: the inlier radius grows with the scatter of the tile estimates up to this multiple of
#: the tolerance, and a level succeeds only when its inliers are at least this share of
#: the tiles with a correlation peak. Without the cap, 98-100 % of the tile estimates of
#: unrelated images (spread 55-133 m) counted as inliers at the 2 m and 4 m levels.
INLIER_RADIUS_CAP = 3.0
MIN_INLIER_FRACTION = 0.5
#: Coarsest pixel size (m) accepted for an optical co-registration base.
MAX_BASE_RESOLUTION_M = 1.0
#: Delineate-Anything settings recommended for these sources (not engine defaults).
DELINEATE_ANYTHING_PARAMS: dict[str, Any] = {"processing_gsd_m": 3, "merge_tile_pieces": False}
#: Revision of the cached index file format.
INDEX_FORMAT = "1"
#: A cached index is refreshed when the date window ends after its snapshot and it
#: is older than this (see :func:`load_index`).
INDEX_REFRESH_MIN_AGE = _dt.timedelta(days=1)
#: Copernicus GLO-30 DEM tiles (public COGs, heights above EGM2008).
GLO30_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
#: Copernicus GLO-90 DEM tiles, read where GLO-30 has no public tile (Armenia and
#: Azerbaijan are not in the public GLO-30 release).
GLO90_URL = "https://copernicus-dem-90m.s3.amazonaws.com/{name}/{name}.tif"
#: EGM2008 geoid undulation grid (2.5 arc-minutes) of the PROJ data CDN.
EGM2008_URL = "https://cdn.proj.org/us_nga_egm08_25.tif"
#: Product types that are never downloaded (complex or derived products).
FORBIDDEN_PRODUCTS = ("CPHD", "SICD", "SLC", "SIDD", "CSI")

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_UTC = _dt.UTC


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Provider:
    """Facts about one open-data archive (licence, credit, bucket, products).

    Attributes
    ----------
    source : str
        Agribound source name.
    vendor : str
        Data provider.
    bucket, region : str
        Public S3 bucket and its AWS region.
    product : str
        Amplitude product used (``"GEC"`` or ``"GEO"``).
    attribution : str
        Credit line written to ``AGRIBOUND_ATTRIBUTION``.
    license_name, license_url : str
        Licence of the data.
    registry_name, registry_url : str
        Dataset name and page in the AWS Registry of Open Data (citation).
    changes : str
        Note of the changes agribound makes (``AGRIBOUND_CHANGES``).
    """

    source: str
    vendor: str
    bucket: str
    region: str
    product: str
    attribution: str
    license_name: str
    license_url: str
    registry_name: str
    registry_url: str
    changes: str

    def citation(self, access_date: str) -> str:
        """The AWS Registry of Open Data citation ("How to Cite") with *access_date*."""
        return f"{self.registry_name} was accessed on {access_date} from {self.registry_url}."


PROVIDERS: dict[str, Provider] = {
    "umbra": Provider(
        source="umbra",
        vendor="Umbra Lab, Inc.",
        bucket="umbra-open-data-catalog",
        region="us-west-2",
        product="GEC",
        # Umbra publishes no credit string (checked 2026-10-08); this is agribound's.
        attribution="SAR data: Umbra Lab, Inc., CC BY 4.0",
        license_name="CC BY 4.0",
        license_url="https://umbra.space/open-license",
        registry_name="Umbra Synthetic Aperture Radar (SAR) Open Data",
        registry_url="https://registry.opendata.aws/umbra-open-data",
        changes=(
            "Modified by agribound: GEC amplitude orthorectified with its RPCs and the "
            "Copernicus GLO-30 (or GLO-90) DEM (EGM2008 to ellipsoidal heights), averaged to "
            "the output grid, co-registered, level-normalised per date and median-composited"
        ),
    ),
    "capella": Provider(
        source="capella",
        vendor="Capella Space Corp",
        bucket="capella-open-data",
        region="us-west-2",
        product="GEO",
        attribution="© Capella Space Corp, All Rights Reserved.",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        registry_name="Capella Space Synthetic Aperture Radar (SAR) Open Dataset",
        registry_url="https://registry.opendata.aws/capella_opendata",
        changes=(
            "Modified by agribound: GEO amplitude converted to sigma0 (dB) plus "
            f"{CAPELLA_DB_OFFSET:g}, averaged (mean intensity) to the output grid, "
            "co-registered, level-normalised per date and median-composited"
        ),
    ),
}
"""Provider facts per source (licences checked 2026-10-08: the AWS registry pages,
https://www.capellaspace.com/legal/data-licensing and https://umbra.space/open-license)."""


def get_provider(source: str) -> Provider:
    """Return the :class:`Provider` of an open VHR SAR *source*."""
    try:
        return PROVIDERS[str(source).lower().strip()]
    except KeyError:
        raise ValueError(
            f"Source {source!r} is not an open VHR SAR source. Choose from {sorted(PROVIDERS)}"
        ) from None


# ---------------------------------------------------------------------------
# Anonymous HTTPS access to a public S3 bucket
# ---------------------------------------------------------------------------


class BucketError(RuntimeError):
    """An HTTP request to a public bucket failed."""


class ProductChangedError(BucketError):
    """The object changed in the bucket since it was indexed (ETag mismatch)."""


@dataclasses.dataclass(frozen=True)
class S3Object:
    """One object of a bucket listing."""

    key: str
    size: int
    etag: str
    last_modified: str


def _user_agent() -> str:
    from agribound._version import __version__

    return f"agribound/{__version__}"


def _etag_tag(etag: str) -> str:
    """*etag* reduced to file-name characters (S3 ETags are hex digits, maybe ``-<parts>``)."""
    return re.sub(r"[^0-9A-Za-z-]", "", etag)[:64] or "no-etag"


_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def _exclusive(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive lock on *lock_path* across threads and processes.

    Threads of one process share a :class:`threading.Lock` per path;
    processes lock the file with ``fcntl.flock`` (POSIX; NFS through its lock
    manager) or ``msvcrt.locking`` (Windows). The operating system releases the
    lock when a process ends. Where the file system does not support locks
    (for example Lustre mounted without ``flock``), a WARNING is logged and
    the block runs without the inter-process lock.
    """
    key = os.path.abspath(lock_path)
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.Lock())
    with thread_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as fh:
            locked = False
            try:
                if os.name == "nt":
                    import msvcrt

                    while True:
                        try:
                            fh.seek(0)
                            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                            locked = True
                            break
                        except OSError as exc:
                            if exc.errno not in (errno.EACCES, errno.EDEADLK):
                                raise
                            time.sleep(1.0)
                else:
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                    locked = True
            except OSError as exc:
                logger.warning(
                    "Could not lock %s (%s); continuing without a lock, so another process "
                    "that uses the same vhr_sar_cache_dir at the same time may fail.",
                    lock_path,
                    exc,
                )
            try:
                yield
            finally:
                if locked:
                    with contextlib.suppress(OSError):
                        if os.name == "nt":
                            import msvcrt

                            fh.seek(0)
                            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl

                            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class PublicBucket:
    """Anonymous HTTPS client of a public S3 bucket (virtual-hosted style).

    Requests reuse one keep-alive connection per thread and are retried on
    connection errors and HTTP 5xx with exponential back-off.

    Parameters
    ----------
    bucket, region : str
        Bucket name and AWS region.
    timeout_s : float
        Socket timeout per request.
    retries : int
        Retries after the first attempt.
    """

    def __init__(
        self, bucket: str, region: str = "us-west-2", *, timeout_s: float = 60.0, retries: int = 4
    ) -> None:
        self.bucket = bucket
        self.host = f"{bucket}.s3.{region}.amazonaws.com"
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self._local = threading.local()

    def url(self, key: str) -> str:
        """HTTPS URL of *key*."""
        return f"https://{self.host}/{urllib.parse.quote(key)}"

    # -- transport --------------------------------------------------------

    def _connection(self) -> http.client.HTTPSConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = http.client.HTTPSConnection(self.host, timeout=self.timeout_s)
            self._local.conn = conn
        return conn

    def _drop_connection(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
        self._local.conn = None

    def request(
        self, method: str, path: str, headers: dict[str, str] | None = None
    ) -> tuple[int, dict[str, str], bytes]:
        """Send one request; return ``(status, lower-case headers, body)``.

        Retries connection errors and HTTP 500/502/503/504; other statuses are
        returned to the caller.
        """
        hdrs = {"User-Agent": _user_agent(), **(headers or {})}
        last: BaseException | None = None
        for attempt in range(self.retries + 1):
            try:
                conn = self._connection()
                conn.request(method, path, headers=hdrs)
                resp = conn.getresponse()
                body = resp.read()
                if resp.status in (500, 502, 503, 504):
                    raise BucketError(f"HTTP {resp.status}")
                return resp.status, {k.lower(): v for k, v in resp.getheaders()}, body
            except (OSError, http.client.HTTPException, BucketError) as exc:
                last = exc
                self._drop_connection()
                if attempt < self.retries:
                    time.sleep(min(20.0, 1.5 * 2**attempt))
        raise BucketError(
            f"{method} https://{self.host}{path} failed after {self.retries + 1} attempts: {last}"
        ) from last

    def _stream(
        self, path: str, headers: dict[str, str]
    ) -> tuple[int, dict[str, str], Iterable[bytes], Callable[[], None]]:
        """Open a streaming GET: ``(status, headers, chunk iterator, close)``."""
        conn = http.client.HTTPSConnection(self.host, timeout=self.timeout_s)
        conn.request("GET", path, headers={"User-Agent": _user_agent(), **headers})
        resp = conn.getresponse()

        def chunks() -> Iterable[bytes]:
            while True:
                buf = resp.read(8 << 20)
                if not buf:
                    return
                yield buf

        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, chunks(), conn.close

    # -- operations ---------------------------------------------------------

    def list(
        self, prefix: str, *, delimiter: str | None = None
    ) -> tuple[list[S3Object], list[str]]:
        """ListObjectsV2 of *prefix*: ``(objects, common prefixes)`` over all pages."""
        objects: list[S3Object] = []
        prefixes: list[str] = []
        token = None
        while True:
            query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if delimiter:
                query["delimiter"] = delimiter
            if token:
                query["continuation-token"] = token
            status, _, body = self.request("GET", "/?" + urllib.parse.urlencode(query))
            if status != 200:
                raise BucketError(f"Listing s3://{self.bucket}/{prefix} failed: HTTP {status}")
            root = ET.fromstring(body)
            for item in root.findall(f"{_S3_NS}Contents"):
                objects.append(
                    S3Object(
                        key=item.findtext(f"{_S3_NS}Key") or "",
                        size=int(item.findtext(f"{_S3_NS}Size") or 0),
                        etag=(item.findtext(f"{_S3_NS}ETag") or "").strip('"'),
                        last_modified=item.findtext(f"{_S3_NS}LastModified") or "",
                    )
                )
            for item in root.findall(f"{_S3_NS}CommonPrefixes"):
                prefixes.append(item.findtext(f"{_S3_NS}Prefix") or "")
            token = root.findtext(f"{_S3_NS}NextContinuationToken")
            if root.findtext(f"{_S3_NS}IsTruncated") != "true" or not token:
                return objects, prefixes

    def list_all(self, prefix: str, *, workers: int = 16, _depth: int = 0) -> list[S3Object]:
        """Every object under *prefix*, listed in parallel partitions.

        The sub-folders one level below *prefix* are listed recursively (up to
        two levels) when there are at most 64 of them; otherwise they are
        grouped by the first character after their common prefix (for example
        the first hex digit of UUID folders) and each group is listed as one
        prefix. Partitions are listed concurrently.
        """
        direct, subs = self.list(prefix, delimiter="/")
        found = {o.key: o for o in direct}
        jobs: list[Callable[[], list[S3Object]]]
        if len(subs) > 64 or (subs and _depth >= 2):
            common = os.path.commonprefix(subs)
            parts = sorted({sub[: len(common) + 1] for sub in subs})
            jobs = [lambda part=part: self.list(part)[0] for part in parts]
        else:
            jobs = [
                lambda sub=sub: self.list_all(sub, workers=workers, _depth=_depth + 1)
                for sub in subs
            ]
        if jobs:
            with concurrent.futures.ThreadPoolExecutor(min(workers, len(jobs))) as pool:
                for objects in pool.map(lambda job: job(), jobs):
                    for obj in objects:
                        found[obj.key] = obj
        return sorted(found.values(), key=lambda o: o.key)

    def get(self, key: str, *, byte_range: tuple[int, int] | None = None) -> bytes:
        """GET *key* (optionally bytes ``[first, last]``)."""
        headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
        status, _, body = self.request("GET", "/" + urllib.parse.quote(key), headers)
        if status not in (200, 206):
            raise BucketError(f"GET s3://{self.bucket}/{key} failed: HTTP {status}")
        return body

    def head(self, key: str) -> S3Object:
        """HEAD *key*."""
        status, headers, _ = self.request("HEAD", "/" + urllib.parse.quote(key))
        if status != 200:
            raise BucketError(f"HEAD s3://{self.bucket}/{key} failed: HTTP {status}")
        return S3Object(
            key=key,
            size=int(headers.get("content-length", 0)),
            etag=headers.get("etag", "").strip('"'),
            last_modified=headers.get("last-modified", ""),
        )

    def download(self, key: str, dest: str | Path, *, size: int, etag: str) -> None:
        """Download *key* to *dest*, resuming a partial ``<dest>.<etag>.part``.

        The partial file is named after the ETag its bytes came from, and
        partial files of *dest* with another ETag are deleted first; every
        request carries ``If-Match: <etag>``. A download therefore never mixes
        two versions of the object. The result must have *size* bytes.
        Concurrent calls for one *dest* must be serialised by the caller
        (:func:`ensure_product` holds a lock).

        Raises
        ------
        ProductChangedError
            If the object no longer has *etag* (HTTP 412).
        BucketError
            If the download fails or ends with the wrong size.
        """
        dest = Path(dest)
        part = dest.with_name(f"{dest.name}.{_etag_tag(etag)}.part")
        for stale in dest.parent.glob(f"{glob.escape(dest.name)}.*part"):
            if stale != part and stale.name.endswith(".part"):
                stale.unlink(missing_ok=True)
        path = "/" + urllib.parse.quote(key)
        last: BaseException | None = None
        for attempt in range(self.retries + 1):
            have = part.stat().st_size if part.exists() else 0
            if have > size:
                part.unlink()
                have = 0
            if have == size:
                break
            headers = {"If-Match": f'"{etag}"'}
            if have:
                headers["Range"] = f"bytes={have}-"
            close: Callable[[], None] = lambda: None  # noqa: E731
            try:
                status, rh, chunks, close = self._stream(path, headers)
                if status == 412:
                    raise ProductChangedError(
                        f"s3://{self.bucket}/{key} changed since it was indexed (ETag {etag}); "
                        "refresh the index (agribound.composites.vhr_sar.refresh_index)."
                    )
                if status not in (200, 206):
                    raise BucketError(f"HTTP {status}")
                got = rh.get("etag", "").strip('"')
                if got and got != etag:
                    raise ProductChangedError(
                        f"s3://{self.bucket}/{key} has ETag {got}, the index lists {etag}; "
                        "refresh the index."
                    )
                mode = "ab" if (have and status == 206) else "wb"
                with open(part, mode) as fh:
                    for buf in chunks:
                        fh.write(buf)
            except ProductChangedError:
                raise
            except (OSError, http.client.HTTPException, BucketError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(min(30.0, 3.0 * (attempt + 1)))
            finally:
                with contextlib.suppress(Exception):
                    close()
        got_size = part.stat().st_size if part.exists() else 0
        if got_size != size:
            raise BucketError(
                f"Download of s3://{self.bucket}/{key} incomplete ({got_size} of {size} bytes): "
                f"{last}"
            )
        os.replace(part, dest)


# ---------------------------------------------------------------------------
# Index records
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CollectRecord:
    """One collect of an archive index (one amplitude product).

    ``datetime`` is ISO 8601 UTC; ``footprint`` is the lon/lat outer ring;
    angles are in degrees (``view_azimuth_deg`` in [0, 360)). The product
    fields describe the GEC (Umbra) or GEO (Capella) object in the bucket.
    """

    source: str
    collect_id: str
    datetime: str
    polarization: str
    orbit_state: str
    look_side: str
    incidence_deg: float
    view_azimuth_deg: float
    footprint: tuple[tuple[float, float], ...]
    product_key: str
    product_bytes: int
    product_etag: str
    product_last_modified: str
    platform: str | None = None
    metadata_key: str | None = None
    mode: str | None = None
    squint_deg: float | None = None
    resolution_range_m: float | None = None
    resolution_azimuth_m: float | None = None
    looks_range: float | None = None
    looks_azimuth: float | None = None
    processor: str | None = None
    calibration: str | None = None
    scale_factor: float | None = None
    reference_height_m: float | None = None
    n_copies: int = 1

    @property
    def date(self) -> str:
        """``YYYY-MM-DD`` (UTC)."""
        return self.datetime[:10]

    @property
    def time(self) -> _dt.datetime:
        """Acquisition time (timezone-aware UTC)."""
        return _parse_time(self.datetime)

    def geometry(self) -> Any:
        """Footprint as a shapely polygon (EPSG:4326)."""
        from shapely.geometry import Polygon

        return Polygon(self.footprint)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["footprint"] = [list(p) for p in self.footprint]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollectRecord:
        names = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in data.items() if k in names}
        values["footprint"] = tuple(tuple(float(c) for c in p[:2]) for p in values["footprint"])
        return cls(**values)


def _parse_time(text: str) -> _dt.datetime:
    value = _dt.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=_UTC)
    return value.astimezone(_UTC)


def _iso(value: _dt.datetime) -> str:
    return value.astimezone(_UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _outer_ring(geometry: dict[str, Any]) -> tuple[list[tuple[float, float]], list[float]]:
    """Lon/lat outer ring and the heights (third coordinates) of a GeoJSON polygon."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "MultiPolygon":
        from shapely.geometry import shape

        largest = max(shape(geometry).geoms, key=lambda g: g.area)
        ring = [(float(x), float(y)) for x, y, *_ in largest.exterior.coords]
        return ring, []
    if kind != "Polygon" or not coords:
        raise ValueError(f"unsupported footprint geometry {kind!r}")
    ring = [(float(c[0]), float(c[1])) for c in coords[0]]
    heights = [float(c[2]) for c in coords[0][:-1] if len(c) > 2 and c[2] is not None]
    return ring, heights


def parse_umbra_stac(item: dict[str, Any]) -> dict[str, Any]:
    """Collect fields of an Umbra ``.stac.v2.json`` sidecar.

    Raises
    ------
    ValueError
        If the collect ID, time, polarization or geometry angles are missing,
        or the item holds more than one polarization.
    """
    props = item.get("properties") or {}
    collect_id = props.get("umbra:collect_id")
    when = props.get("datetime")
    pols = [str(p).upper() for p in props.get("sar:polarizations") or []]
    incidence = _float(props.get("view:incidence_angle"))
    azimuth = _float(props.get("view:azimuth"))
    if not collect_id or not when:
        raise ValueError("no umbra:collect_id or datetime")
    if len(pols) != 1:
        raise ValueError(f"polarizations {pols} (one expected)")
    if incidence is None or azimuth is None:
        raise ValueError("no view:incidence_angle or view:azimuth")
    ring, heights = _outer_ring(item.get("geometry") or {})
    return {
        "collect_id": str(collect_id),
        "datetime": _iso(_parse_time(when)),
        "platform": props.get("platform"),
        "polarization": pols[0],
        "orbit_state": str(props.get("sat:orbit_state") or "").lower(),
        "look_side": str(props.get("sar:observation_direction") or "").lower(),
        "incidence_deg": incidence,
        "view_azimuth_deg": azimuth % 360.0,
        "squint_deg": _float(props.get("umbra:squint_angle_degrees_off_broadside")),
        "mode": props.get("sar:instrument_mode"),
        "resolution_range_m": _float(props.get("sar:resolution_range")),
        "resolution_azimuth_m": _float(props.get("sar:resolution_azimuth")),
        "looks_range": _float(props.get("sar:looks_range")),
        "looks_azimuth": _float(props.get("sar:looks_azimuth")),
        "processor": props.get("processing:version"),
        "footprint": ring,
        "reference_height_m": float(np.mean(heights)) if heights else None,
    }


def parse_capella_item(item: dict[str, Any]) -> dict[str, Any]:
    """Collect fields of a Capella GEO STAC item (incidence rounded to 0.1 degree)."""
    props = item.get("properties") or {}
    collect_id = props.get("capella:collect_id")
    when = props.get("datetime")
    pols = [str(p).upper() for p in props.get("sar:polarizations") or []]
    incidence = _float(props.get("view:incidence_angle"))
    azimuth = _float(props.get("view:azimuth"))
    if not collect_id or not when:
        raise ValueError("no capella:collect_id or datetime")
    if len(pols) != 1:
        raise ValueError(f"polarizations {pols} (one expected)")
    if incidence is None or azimuth is None:
        raise ValueError("no view:incidence_angle or view:azimuth")
    ring, _ = _outer_ring(item.get("geometry") or {})
    software = props.get("processing:software")
    processor = (
        ",".join(str(v) for v in software.values()) if isinstance(software, dict) else software
    )
    return {
        "collect_id": str(collect_id),
        "datetime": _iso(_parse_time(when)),
        "platform": props.get("platform"),
        "polarization": pols[0],
        "orbit_state": str(props.get("sat:orbit_state") or "").lower(),
        "look_side": str(props.get("sar:observation_direction") or "").lower(),
        "incidence_deg": incidence,
        "view_azimuth_deg": azimuth % 360.0,
        "squint_deg": _float(props.get("capella:squint_angle")),
        "mode": props.get("sar:instrument_mode"),
        "resolution_range_m": _float(props.get("sar:resolution_range")),
        "resolution_azimuth_m": _float(props.get("sar:resolution_azimuth")),
        "looks_range": _float(props.get("sar:looks_range")),
        "looks_azimuth": _float(props.get("sar:looks_azimuth")),
        "processor": processor,
        "footprint": ring,
    }


_JSON_DECODER = json.JSONDecoder()


def _json_value_after(text: str, name: str) -> Any:
    """The JSON value that follows the first ``"name":`` in *text* (raises ValueError)."""
    match = re.search(rf'"{re.escape(name)}"\s*:\s*', text)
    if match is None:
        raise ValueError(f"{name!r} not found")
    value, _ = _JSON_DECODER.raw_decode(text, match.end())
    return value


def parse_capella_extended(text: str) -> dict[str, Any]:
    """Fields of the head of a Capella ``_extended.json`` (full precision incidence).

    *text* may be truncated after the ``collect.image`` object.

    Raises
    ------
    ValueError
        If the ``image`` object cannot be decoded from *text*.
    """
    image = _json_value_after(text, "image")
    if not isinstance(image, dict):
        raise ValueError("'image' is not an object")
    centre = image.get("center_pixel") or {}
    out = {
        "incidence_deg": _float(centre.get("incidence_angle")),
        "calibration": image.get("calibration"),
        "scale_factor": _float(image.get("scale_factor")),
        "radiometry": image.get("radiometry"),
    }
    with contextlib.suppress(ValueError):
        out["collect_id"] = _json_value_after(text, "collect_id")
    with contextlib.suppress(ValueError):
        out["software_version"] = _json_value_after(text, "software_version")
    return out


# ---------------------------------------------------------------------------
# Archive index
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ArchiveIndex:
    """The collects of one archive at one snapshot.

    Attributes
    ----------
    source : str
        ``"umbra"`` or ``"capella"``.
    snapshot : str
        UTC time the bucket was listed (ISO 8601); selection is deterministic
        for one snapshot.
    records : list[CollectRecord]
        One record per collect (copies merged), sorted by time.
    stats : dict
        Counts of the build (listed keys, products, metadata read, records
        dropped and why).
    path : Path or None
        Cache file.
    """

    source: str
    snapshot: str
    records: list[CollectRecord]
    stats: dict[str, Any] = dataclasses.field(default_factory=dict)
    path: Path | None = None
    copies: dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)

    @property
    def snapshot_time(self) -> _dt.datetime:
        return _parse_time(self.snapshot)

    def by_id(self) -> dict[str, CollectRecord]:
        return {r.collect_id: r for r in self.records}


def default_cache_dir() -> Path:
    """``$XDG_CACHE_HOME/agribound/vhr_sar``, else ``~/.cache/agribound/vhr_sar``."""
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "agribound" / "vhr_sar"


def resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    """*cache_dir* (``~`` expanded) or :func:`default_cache_dir`."""
    return Path(cache_dir).expanduser() if cache_dir else default_cache_dir()


def index_path(source: str, cache_dir: str | Path | None = None) -> Path:
    """Cache file of the index of *source*: ``<cache_dir>/<source>/index.json.gz``."""
    return resolve_cache_dir(cache_dir) / get_provider(source).source / "index.json.gz"


def _write_json_gz(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
    os.replace(tmp, path)


def _read_index(path: Path) -> ArchiveIndex | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, EOFError, zlib.error) as exc:
        # OSError covers gzip.BadGzipFile; EOFError a truncated file; zlib.error damaged data.
        logger.warning("Ignoring unreadable VHR SAR index %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or data.get("format") != INDEX_FORMAT:
        return None
    try:
        copies = data.get("copies", {})
        records, _ = _merge_copies(data["source"], copies, _parse_time(data["snapshot"]))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        logger.warning("Ignoring malformed VHR SAR index %s: %s", path, exc)
        return None
    return ArchiveIndex(
        source=data["source"],
        snapshot=data["snapshot"],
        records=records,
        stats=data.get("stats", {}),
        path=path,
        copies=copies,
    )


def _merge_copies(
    source: str, copies: dict[str, Any], snapshot: _dt.datetime
) -> tuple[list[CollectRecord], dict[str, int]]:
    """Merge the parsed product copies into one record per collect.

    The copy with the newest product upload (then the largest key) is kept;
    records dated after *snapshot* are dropped.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    stats = {"copies": 0, "copies_unparsed": 0, "future_dropped": 0}
    for entry in copies.values():
        record = entry.get("record")
        if record is None:
            stats["copies_unparsed"] += 1
            continue
        stats["copies"] += 1
        groups.setdefault(record["collect_id"], []).append(record)
    records = []
    for cid in sorted(groups):
        group = groups[cid]
        best = max(group, key=lambda r: (r["product_last_modified"], r["product_key"]))
        if _parse_time(best["datetime"]) > snapshot:
            stats["future_dropped"] += 1
            continue
        records.append(CollectRecord.from_dict({**best, "source": source, "n_copies": len(group)}))
    records.sort(key=lambda r: (r.datetime, r.collect_id))
    stats["collects"] = len(records)
    return records, stats


def _fetch_parallel(
    jobs: list[tuple[str, Callable[[], dict[str, Any]]]], workers: int, label: str
) -> dict[str, dict[str, Any]]:
    """Run ``(name, fn)`` jobs in a thread pool; a failure becomes ``{"record": None, ...}``.

    The failure's reason is kept, and the next refresh reads that file again.
    """
    out: dict[str, dict[str, Any]] = {}
    if not jobs:
        return out
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        futures = {pool.submit(fn): name for name, fn in jobs}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            name = futures[fut]
            try:
                out[name] = fut.result()
            except Exception as exc:  # one bad sidecar must not stop the index
                out[name] = {"record": None, "reason": f"{type(exc).__name__}: {exc}"[:300]}
            if i % 2000 == 0:
                logger.info("%s: %d of %d read (%.0f s)", label, i, len(jobs), time.time() - t0)
    return out


def _build_umbra_copies(
    bucket: PublicBucket, previous: dict[str, Any], workers: int
) -> tuple[dict[str, Any], dict[str, int]]:
    objects: list[S3Object] = []
    for prefix in ("sar-data/", "open-data/"):
        objects += bucket.list_all(prefix, workers=workers)
    folders: dict[str, dict[str, S3Object]] = {}
    for obj in objects:
        folder, _, name = obj.key.rpartition("/")
        folders.setdefault(folder, {})[name] = obj
    copies: dict[str, Any] = {}
    jobs = []
    stats = {"listed_keys": len(objects), "products": 0, "products_without_stac": 0}
    for folder in sorted(folders):
        files = folders[folder]
        gecs = [o for n, o in files.items() if n.endswith("_GEC.tif")]
        if not gecs:
            continue
        stats["products"] += 1
        gec = max(gecs, key=lambda o: (o.last_modified, o.key))
        sidecars = [o for n, o in files.items() if n.endswith(".stac.v2.json")]
        if not sidecars:
            stats["products_without_stac"] += 1
            continue
        sidecar = max(sidecars, key=lambda o: (o.last_modified, o.key))
        product = {
            "product_key": gec.key,
            "product_bytes": gec.size,
            "product_etag": gec.etag,
            "product_last_modified": gec.last_modified,
            "metadata_key": sidecar.key,
        }
        old = previous.get(sidecar.key)
        if old and old.get("etag") == sidecar.etag and old.get("record") is not None:
            copies[sidecar.key] = {**old, "record": {**old["record"], **product}}
            continue

        def fetch(sidecar: S3Object = sidecar, product: dict[str, Any] = product) -> dict:
            item = json.loads(bucket.get(sidecar.key))
            return {"etag": sidecar.etag, "record": {**parse_umbra_stac(item), **product}}

        jobs.append((sidecar.key, fetch))
    logger.info(
        "Umbra index: %d GEC folders (%d without a .stac.v2.json sidecar); reading %d sidecars",
        stats["products"],
        stats["products_without_stac"],
        len(jobs),
    )
    copies.update(_fetch_parallel(jobs, workers, "Umbra sidecars"))
    return copies, stats


_CAPELLA_ID = r"CAPELLA_[A-Z0-9]+_[A-Z]+_GEO_[A-Z]{2}_\d{14}_\d{14}"
_CAPELLA_ITEM_RE = re.compile(rf"^stac/capella-open-data-by-datetime/.*/({_CAPELLA_ID})/\1\.json$")
_CAPELLA_DATA_RE = re.compile(rf"^data/.*/({_CAPELLA_ID})/\1(\.tif|_extended\.json)$")
#: Bytes of ``_extended.json`` read for the index (the image object is near the start).
CAPELLA_EXTENDED_HEAD_BYTES = 65536


def _build_capella_copies(
    bucket: PublicBucket, previous: dict[str, Any], workers: int
) -> tuple[dict[str, Any], dict[str, int]]:
    items = {}
    for obj in bucket.list_all("stac/capella-open-data-by-datetime/", workers=workers):
        match = _CAPELLA_ITEM_RE.match(obj.key)
        if match:
            items[match.group(1)] = obj
    data: dict[str, dict[str, S3Object]] = {}
    listed = 0
    for obj in bucket.list_all("data/", workers=workers):
        listed += 1
        match = _CAPELLA_DATA_RE.match(obj.key)
        if match:
            data.setdefault(match.group(1), {})[match.group(2)] = obj
    stats = {
        "listed_keys": listed + len(items),
        "products": 0,
        "products_without_stac": 0,
        "products_without_extended": 0,
    }
    copies: dict[str, Any] = {}
    jobs = []
    for pid in sorted(data):
        files = data[pid]
        tif = files.get(".tif")
        if tif is None:
            continue
        stats["products"] += 1
        item = items.get(pid)
        ext = files.get("_extended.json")
        if item is None:
            stats["products_without_stac"] += 1
            continue
        if ext is None:
            stats["products_without_extended"] += 1
            continue
        product = {
            "product_key": tif.key,
            "product_bytes": tif.size,
            "product_etag": tif.etag,
            "product_last_modified": tif.last_modified,
            "metadata_key": ext.key,
        }
        version = f"{item.etag}:{ext.etag}"
        old = previous.get(item.key)
        if old and old.get("etag") == version and old.get("record") is not None:
            copies[item.key] = {**old, "record": {**old["record"], **product}}
            continue

        def fetch(
            item: S3Object = item, ext: S3Object = ext, product: dict = product, version=version
        ) -> dict:
            record = parse_capella_item(json.loads(bucket.get(item.key)))
            head = bucket.get(ext.key, byte_range=(0, CAPELLA_EXTENDED_HEAD_BYTES - 1))
            text = head.decode("utf-8", "replace")
            try:
                extended = parse_capella_extended(text)
            except ValueError:
                extended = parse_capella_extended(bucket.get(ext.key).decode("utf-8", "replace"))
            if extended.get("collect_id") and extended["collect_id"] != record["collect_id"]:
                raise ValueError(
                    f"collect_id {record['collect_id']} (STAC) != {extended['collect_id']} "
                    "(_extended.json)"
                )
            if extended.get("incidence_deg") is not None:
                record["incidence_deg"] = extended["incidence_deg"]
            record["calibration"] = extended.get("calibration")
            record["scale_factor"] = extended.get("scale_factor")
            if record["scale_factor"] is None:
                raise ValueError("no scale_factor in _extended.json")
            return {"etag": version, "record": {**record, **product}}

        jobs.append((item.key, fetch))
    logger.info(
        "Capella index: %d GEO products (%d without a STAC item, %d without _extended.json); "
        "reading %d",
        stats["products"],
        stats["products_without_stac"],
        stats["products_without_extended"],
        len(jobs),
    )
    copies.update(_fetch_parallel(jobs, workers, "Capella GEO metadata"))
    return copies, stats


_COPY_BUILDERS = {"umbra": _build_umbra_copies, "capella": _build_capella_copies}


def refresh_index(
    source: str,
    cache_dir: str | Path | None = None,
    *,
    bucket: PublicBucket | None = None,
    workers: int = 32,
) -> ArchiveIndex:
    """List the archive of *source* and (re)build its index at a new snapshot.

    Metadata files already in the previous index with the same ETag are not
    read again. The Umbra index reads one ``.stac.v2.json`` sidecar per GEC
    folder (about 18,000 on 2026-10-08) on its first build.

    Parameters
    ----------
    source : str
        ``"umbra"`` or ``"capella"``.
    cache_dir : str, Path or None
        Index cache directory (:func:`resolve_cache_dir`).
    bucket : PublicBucket or None
        Client to use (tests pass a fake).
    workers : int
        Concurrent requests.

    Returns
    -------
    ArchiveIndex
    """
    provider = get_provider(source)
    path = index_path(provider.source, cache_dir)
    previous = _read_index(path) if path.exists() else None
    bucket = bucket or PublicBucket(provider.bucket, provider.region)
    snapshot = _dt.datetime.now(_UTC).replace(microsecond=0)
    t0 = time.time()
    copies, stats = _COPY_BUILDERS[provider.source](
        bucket, previous.copies if previous else {}, workers
    )
    records, merge_stats = _merge_copies(provider.source, copies, snapshot)
    stats.update(merge_stats)
    stats["seconds"] = round(time.time() - t0, 1)
    snapshot_text = snapshot.strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_json_gz(
        path,
        {
            "format": INDEX_FORMAT,
            "source": provider.source,
            "bucket": provider.bucket,
            "snapshot": snapshot_text,
            "stats": stats,
            "copies": copies,
        },
    )
    logger.info(
        "%s index at snapshot %s: %d collects (%s) in %.0f s -> %s",
        provider.source,
        snapshot_text,
        len(records),
        ", ".join(f"{k} {v}" for k, v in stats.items() if k != "seconds"),
        stats["seconds"],
        path,
    )
    return ArchiveIndex(provider.source, snapshot_text, records, stats, path, copies)


def load_index(
    source: str,
    cache_dir: str | Path | None = None,
    *,
    window_end: str | None = None,
    refresh: bool = False,
    bucket: PublicBucket | None = None,
    now: _dt.datetime | None = None,
) -> ArchiveIndex:
    """Return the cached index of *source*, building or refreshing it when needed.

    The index is built when there is no cache file, and refreshed when
    *refresh* is set or when the date window ends after the snapshot
    (*window_end*, exclusive ``YYYY-MM-DD``) and the snapshot is older than
    :data:`INDEX_REFRESH_MIN_AGE`: collects of such a window may have been
    uploaded since. Otherwise the cached snapshot is used as it is, so repeated
    runs select the same collects. Delete the file, or call
    :func:`refresh_index`, to pick up newer uploads for an earlier window.
    """
    path = index_path(source, cache_dir)
    index = None if refresh or not path.exists() else _read_index(path)
    if index is not None and window_end is not None:
        now = now or _dt.datetime.now(_UTC)
        end = _dt.datetime.fromisoformat(f"{window_end}T00:00:00+00:00")
        if end > index.snapshot_time and now - index.snapshot_time > INDEX_REFRESH_MIN_AGE:
            logger.info(
                "The %s index snapshot (%s) predates the end of the date window (%s); "
                "refreshing it.",
                source,
                index.snapshot,
                window_end,
            )
            index = None
    if index is None:
        index = refresh_index(source, cache_dir, bucket=bucket)
    return index


def cached_index(source: str, cache_dir: str | Path | None = None) -> ArchiveIndex | None:
    """Return the cached index of *source* as it is, or None; read only.

    Unlike :func:`load_index`, this never lists the archive, refreshes or
    writes anything (for dry runs and reports). It returns None when there
    is no cache file, or when the file is unreadable or of another
    :data:`INDEX_FORMAT` (:func:`load_index` then rebuilds it).
    """
    path = index_path(source, cache_dir)
    return _read_index(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Selection:
    """Collects chosen for one composite.

    Attributes
    ----------
    chosen : list[CollectRecord]
        Collects in time order.
    reference : CollectRecord
        Reference date of the SAR-to-SAR co-registration (closest to the
        median incidence and the middle of the chosen dates).
    group : dict
        Orbit pass, look side and polarization of the chosen group, with its
        incidence and view-azimuth spreads.
    groups : list[dict]
        Every group of covering collects with its largest consistent subset,
        best first.
    pinned : bool
        Whether ``vhr_sar_collect_ids`` chose the collects.
    counts : dict
        Collects in the window, covering the study area, excluded and why.
    warnings : list[str]
        Messages logged as WARNING.
    """

    chosen: list[CollectRecord]
    reference: CollectRecord
    group: dict[str, Any]
    groups: list[dict[str, Any]]
    pinned: bool
    counts: dict[str, Any]
    warnings: list[str] = dataclasses.field(default_factory=list)


def azimuth_spread(values: Sequence[float]) -> float:
    """Circular spread (degrees) of azimuths: 360 minus the largest gap between them."""
    vals = sorted(float(v) % 360.0 for v in values)
    if len(vals) < 2:
        return 0.0
    gaps = [(vals[(i + 1) % len(vals)] - vals[i]) % 360.0 for i in range(len(vals))]
    return float(360.0 - max(gaps))


def footprint_cover(footprint: Any, aoi_4326: Any) -> float:
    """Share of *aoi_4326* inside *footprint* (both EPSG:4326), measured in the AOI's UTM zone."""
    import pyproj
    from shapely.ops import transform as shapely_transform

    from agribound.io.crs import utm_crs_for_geometry

    if aoi_4326.is_empty or aoi_4326.area == 0:
        return 1.0 if footprint.covers(aoi_4326) else 0.0
    crs = utm_crs_for_geometry(aoi_4326)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    aoi = shapely_transform(to_utm, aoi_4326)
    fp = shapely_transform(to_utm, footprint)
    if not fp.is_valid:
        fp = fp.buffer(0)
    return float(fp.intersection(aoi).area / aoi.area) if aoi.area > 0 else 0.0


def _consistent_subset(records: list[CollectRecord]) -> list[CollectRecord]:
    """Largest subset within the incidence and view-azimuth tolerances.

    Every feasible subset has a lowest incidence and a first azimuth (going
    clockwise) among its members, so trying each member as the anchor of both
    windows finds the largest one. Ties: smaller incidence spread, smaller
    azimuth spread, longer time span, then the sorted collect IDs.
    """
    if not records:
        return []
    inc = np.array([r.incidence_deg for r in records], dtype=float)
    az = np.array([r.view_azimuth_deg % 360.0 for r in records], dtype=float)
    eps = 1e-9
    in_inc = (inc[None, :] >= inc[:, None] - eps) & (
        inc[None, :] <= inc[:, None] + INCIDENCE_TOLERANCE_DEG + eps
    )
    in_az = ((az[None, :] - az[:, None]) % 360.0) <= VIEW_AZIMUTH_TOLERANCE_DEG + eps
    counts = in_inc.astype(np.int64) @ in_az.T.astype(np.int64)
    best = int(counts.max())
    subsets = set()
    for i, j in zip(*np.nonzero(counts == best), strict=True):
        subsets.add(tuple(np.flatnonzero(in_inc[i] & in_az[j]).tolist()))

    def rank(members: tuple[int, ...]) -> tuple[Any, ...]:
        chosen = [records[k] for k in members]
        times = [r.time for r in chosen]
        return (
            float(np.ptp(inc[list(members)])),
            azimuth_spread([r.view_azimuth_deg for r in chosen]),
            -(max(times) - min(times)).total_seconds(),
            sorted(r.collect_id for r in chosen),
        )

    return [records[k] for k in min(subsets, key=rank)]


def _spread_pick(records: list[CollectRecord], k: int) -> list[CollectRecord]:
    """Up to *k* records spread evenly in time (always the first and last when k >= 2)."""
    ordered = sorted(records, key=lambda r: (r.datetime, r.collect_id))
    if len(ordered) <= k:
        return ordered
    if k == 1:
        return [_reference(ordered)]
    idx = sorted({round(i * (len(ordered) - 1) / (k - 1)) for i in range(k)})
    return [ordered[i] for i in idx]


def _reference(records: list[CollectRecord]) -> CollectRecord:
    """The record closest to the median incidence and the middle date."""
    incs = sorted(r.incidence_deg for r in records)
    mid_inc = incs[len(incs) // 2]
    times = [r.time for r in records]
    mid = min(times) + (max(times) - min(times)) / 2
    return min(
        records,
        key=lambda r: (
            abs(r.incidence_deg - mid_inc) / INCIDENCE_TOLERANCE_DEG
            + abs((r.time - mid).total_seconds()) / (60.0 * 86400.0),
            r.datetime,
            r.collect_id,
        ),
    )


def _describe(records: list[CollectRecord]) -> dict[str, Any]:
    first = records[0]
    times = [r.time for r in records]
    incs = [r.incidence_deg for r in records]
    return {
        "orbit_state": first.orbit_state,
        "look_side": first.look_side,
        "polarization": first.polarization,
        "n_collects": len(records),
        "incidence_spread_deg": round(max(incs) - min(incs), 3),
        "view_azimuth_spread_deg": round(azimuth_spread([r.view_azimuth_deg for r in records]), 3),
        "span_days": round((max(times) - min(times)).total_seconds() / 86400.0, 2),
        "dates": [r.date for r in sorted(records, key=lambda r: r.datetime)],
    }


def _window_label(window: tuple[str, str]) -> str:
    end = _dt.date.fromisoformat(window[1]) - _dt.timedelta(days=1)
    return f"{window[0]} to {end.isoformat()}"


def select_collects(
    records: Sequence[CollectRecord],
    aoi_4326: Any,
    window: tuple[str, str],
    *,
    source: str,
    polarization: str = "auto",
    max_collects: int = 6,
    calibration: str = "any",
    pinned_ids: Sequence[str] | None = None,
    snapshot: str | None = None,
) -> Selection:
    """Choose the collects of one composite (deterministic for one index snapshot).

    Parameters
    ----------
    records : sequence of CollectRecord
        Index records (:attr:`ArchiveIndex.records`).
    aoi_4326 : shapely geometry
        Study area (EPSG:4326).
    window : tuple[str, str]
        ``(start, end_exclusive)`` dates (:func:`agribound.composites.gee.date_window`).
    source : str
        ``"umbra"`` or ``"capella"`` (for messages).
    polarization : str
        ``"auto"``, ``"HH"`` or ``"VV"``.
    max_collects : int
        Maximum collects of an automatic selection; 1 builds a single date.
    calibration : str
        ``"any"`` or ``"full"`` (Capella calibration label).
    pinned_ids : sequence of str or None
        Collect IDs to use instead of the automatic selection.
    snapshot : str or None
        Index snapshot (for messages).

    Raises
    ------
    NoDataError
        If no collect in the window covers the study area, or fewer than two
        consistent collects do (with ``max_collects >= 2``).
    ValueError
        If the study area is larger than one scene, or pinned collects are
        unknown, inconsistent, outside the window or do not cover the study area.
    """
    label = f"{source} (index snapshot {snapshot})" if snapshot else source
    if pinned_ids:
        return _select_pinned(
            records, aoi_4326, window, label, list(pinned_ids), polarization, calibration
        )
    start, end = window
    in_window = [r for r in records if start <= r.date < end]
    counts: dict[str, Any] = {"in_window": len(in_window)}
    minx, miny, maxx, maxy = aoi_4326.bounds
    touching = []
    covering = []
    for r in in_window:
        fx0, fy0, fx1, fy1 = (
            min(p[0] for p in r.footprint),
            min(p[1] for p in r.footprint),
            max(p[0] for p in r.footprint),
            max(p[1] for p in r.footprint),
        )
        if fx1 < minx or fx0 > maxx or fy1 < miny or fy0 > maxy:
            continue
        geom = r.geometry()
        if not geom.intersects(aoi_4326):
            continue
        touching.append(r)
        if footprint_cover(geom, aoi_4326) >= MIN_FOOTPRINT_COVER:
            covering.append(r)
    counts["intersecting"] = len(touching)
    counts["covering"] = len(covering)
    if not covering:
        raise _no_cover_error(records, aoi_4326, window, label, touching)
    pool = covering
    if polarization != "auto":
        pool = [r for r in pool if r.polarization == polarization]
        counts["polarization_dropped"] = len(covering) - len(pool)
    if calibration == "full":
        kept = [r for r in pool if (r.calibration or "").lower() == "full"]
        counts["calibration_dropped"] = len(pool) - len(kept)
        pool = kept
    if not pool:
        raise NoDataError(
            f"No {label} collect from {_window_label(window)} that covers the study area has "
            + (f"polarization {polarization}" if polarization != "auto" else "")
            + (" and " if polarization != "auto" and calibration == "full" else "")
            + ("calibration 'full'" if calibration == "full" else "")
            + f" ({len(covering)} cover it: "
            + ", ".join(sorted({f"{r.polarization}/{r.calibration}" for r in covering}))
            + ")."
        )
    by_group: dict[tuple[str, str, str], list[CollectRecord]] = {}
    for r in pool:
        by_group.setdefault((r.orbit_state, r.look_side, r.polarization), []).append(r)
    ranked = []
    for key, members in by_group.items():
        subset = _consistent_subset(members)
        desc = _describe(subset)
        desc["n_in_group"] = len(members)
        ranked.append(
            (
                (
                    -len(subset),
                    desc["incidence_spread_deg"],
                    desc["view_azimuth_spread_deg"],
                    -desc["span_days"],
                    key,
                ),
                desc,
                subset,
            )
        )
    ranked.sort(key=lambda x: x[0])
    _, group, subset = ranked[0]
    groups = [d for _, d, _ in ranked]
    warnings_: list[str] = []
    if len(subset) < 2 and max_collects >= 2:
        raise NoDataError(
            f"Only one consistent {label} collect from {_window_label(window)} covers the study "
            f"area ({len(pool)} cover it; groups by orbit pass, look side and polarization: "
            + "; ".join(
                f"{g['orbit_state']}/{g['look_side']}/{g['polarization']}: {g['n_in_group']} "
                f"({g['n_collects']} within {INCIDENCE_TOLERANCE_DEG:g} deg incidence and "
                f"{VIEW_AZIMUTH_TOLERANCE_DEG:g} deg view azimuth)"
                for g in groups
            )
            + "). The recipe needs at least 2 dates; widen the date window, or set "
            f"vhr_sar_max_collects=1 for a single-date composite ({SINGLE_DATE_NOTE})."
        )
    chosen = _spread_pick(subset, max_collects)
    if len(chosen) == 1:
        warnings_.append(
            f"The {source} composite uses a single date ({chosen[0].date}); {SINGLE_DATE_NOTE}."
        )
    counts["consistent"] = len(subset)
    counts["chosen"] = len(chosen)
    return Selection(
        chosen=chosen,
        reference=_reference(chosen),
        group=_describe(chosen) | {"n_consistent": len(subset)},
        groups=groups,
        pinned=False,
        counts=counts,
        warnings=warnings_,
    )


def _no_cover_error(
    records: Sequence[CollectRecord],
    aoi_4326: Any,
    window: tuple[str, str],
    label: str,
    touching: list[CollectRecord],
) -> Exception:
    """NoDataError (nothing covers) or ValueError (the study area exceeds one scene)."""
    import geopandas as gpd

    from agribound.io.crs import utm_crs_for_geometry

    crs = utm_crs_for_geometry(aoi_4326)
    aoi = gpd.GeoSeries([aoi_4326], crs="EPSG:4326").to_crs(crs).iloc[0]
    w, s, e, n = aoi.bounds
    if touching:
        fps = gpd.GeoSeries([r.geometry() for r in touching], crs="EPSG:4326").to_crs(crs)
        largest = float(fps.area.max())
        best = max(footprint_cover(r.geometry(), aoi_4326) for r in touching)
        if aoi.area > largest or aoi.convex_hull.area > largest:
            return ValueError(
                f"The study area ({(e - w) / 1000:.1f} x {(n - s) / 1000:.1f} km, "
                f"{aoi.area / 1e6:.1f} km2) is larger than one {label} scene (the largest "
                f"footprint in the window is {largest / 1e6:.1f} km2). Every collect of a VHR "
                "SAR composite must cover the whole study area; split it into areas of at "
                "most one scene."
            )
        return NoDataError(
            f"No {label} collect from {_window_label(window)} covers the whole study area: "
            f"{len(touching)} intersect it, covering at most {100 * best:.1f} % of it. Shrink "
            "the study area to the footprint of the collects, or widen the date window."
        )
    nearby = []
    for r in records:
        geom = r.geometry()
        if geom.intersects(aoi_4326):
            nearby.append(r)
    covering = [r for r in nearby if footprint_cover(r.geometry(), aoi_4326) >= 0.5]
    if covering:
        start = _dt.date.fromisoformat(window[0])

        def distance(r: CollectRecord) -> float:
            return abs((_dt.date.fromisoformat(r.date) - start).days)

        dates = sorted({r.date for r in sorted(covering, key=distance)[:10]})
        hint = f" Collects over it exist on {', '.join(dates)} (nearest to the window)."
    else:
        hint = " The archive has no collect over it."
    return NoDataError(
        f"No {label} collect from {_window_label(window)} covers the study area.{hint}"
    )


def _select_pinned(
    records: Sequence[CollectRecord],
    aoi_4326: Any,
    window: tuple[str, str],
    label: str,
    pinned_ids: list[str],
    polarization: str,
    calibration: str,
) -> Selection:
    by_id = {r.collect_id: r for r in records}
    unknown = [cid for cid in pinned_ids if cid not in by_id]
    if unknown:
        raise ValueError(
            f"vhr_sar_collect_ids not in the {label} index: {unknown}. Check the IDs, or "
            "refresh the index (agribound.composites.vhr_sar.refresh_index) for newer uploads."
        )
    chosen = sorted((by_id[cid] for cid in pinned_ids), key=lambda r: (r.datetime, r.collect_id))
    problems = []
    for name, attr in (
        ("polarizations", "polarization"),
        ("orbit passes", "orbit_state"),
        ("look sides", "look_side"),
    ):
        values = sorted({getattr(r, attr) for r in chosen})
        if len(values) > 1:
            problems.append(f"mixed {name} {values}")
    if polarization != "auto" and any(r.polarization != polarization for r in chosen):
        problems.append(f"vhr_sar_polarization={polarization!r} but collects are not all it")
    incs = [r.incidence_deg for r in chosen]
    if max(incs) - min(incs) > INCIDENCE_TOLERANCE_DEG + 1e-9:
        problems.append(
            f"incidence spread {max(incs) - min(incs):.2f} deg > {INCIDENCE_TOLERANCE_DEG:g}"
        )
    spread = azimuth_spread([r.view_azimuth_deg for r in chosen])
    if spread > VIEW_AZIMUTH_TOLERANCE_DEG + 1e-9:
        problems.append(f"view-azimuth spread {spread:.1f} deg > {VIEW_AZIMUTH_TOLERANCE_DEG:g}")
    start, end = window
    outside = [f"{r.collect_id} ({r.date})" for r in chosen if not start <= r.date < end]
    if outside:
        problems.append(f"outside the date window {_window_label(window)}: {outside}")
    if calibration == "full":
        partial = [f"{r.collect_id} ({r.calibration})" for r in chosen if r.calibration != "full"]
        if partial:
            problems.append(f"not fully calibrated (vhr_sar_calibration='full'): {partial}")
    short = []
    for r in chosen:
        cover = footprint_cover(r.geometry(), aoi_4326)
        if cover < MIN_FOOTPRINT_COVER:
            short.append(f"{r.collect_id} ({100 * cover:.1f} %)")
    if short:
        problems.append(f"footprints that do not cover the study area: {short}")
    if problems:
        raise ValueError(
            f"The pinned {label} collects cannot form one composite: " + "; ".join(problems) + "."
        )
    warnings_ = []
    if len(chosen) == 1:
        warnings_.append(
            f"The composite uses a single date ({chosen[0].date}); {SINGLE_DATE_NOTE}."
        )
    return Selection(
        chosen=chosen,
        reference=_reference(chosen),
        group=_describe(chosen) | {"n_consistent": len(chosen)},
        groups=[],
        pinned=True,
        counts={"pinned": len(chosen)},
        warnings=warnings_,
    )


# ---------------------------------------------------------------------------
# Products, DEM and geoid
# ---------------------------------------------------------------------------


def check_product_key(source: str, key: str) -> None:
    """Refuse any key that is not the amplitude product of *source* (never CPHD/SICD/SLC)."""
    name = key.rsplit("/", 1)[-1]
    ok = (
        name.endswith("_GEC.tif")
        if source == "umbra"
        else bool(re.fullmatch(rf"{_CAPELLA_ID}\.tif", name))
    )
    if not ok or any(f"_{p}" in name.upper() for p in FORBIDDEN_PRODUCTS):
        raise ValueError(f"Refusing to download {key!r}: not a {get_provider(source).product}")


def product_path(cache_dir: str | Path | None, record: CollectRecord) -> Path:
    """Local file of the product of *record*.

    ``<cache>/<source>/products/<collect_id>/<copy>/<name>``, where ``<copy>`` is
    the first 10 hex digits of the SHA-1 of the bucket key: copies of one Umbra
    collect in different folders share the file name (292 collects of the
    2026-10-09 index have such copies with different ETags and sizes).
    """
    import hashlib

    name = record.product_key.rsplit("/", 1)[-1]
    copy = hashlib.sha1(record.product_key.encode("utf-8")).hexdigest()[:10]
    folder = resolve_cache_dir(cache_dir) / record.source / "products" / record.collect_id
    return folder / copy / name


def _matching_product(path: Path, record: CollectRecord) -> dict[str, Any] | None:
    """The metadata of *path* if the file holds the indexed object of *record*, else None."""
    meta = path.with_name(path.name + ".json")
    if not (path.exists() and meta.exists()):
        return None
    try:
        info = json.loads(meta.read_text())
    except (OSError, ValueError):
        return None
    if path.stat().st_size != record.product_bytes or info.get("etag") != record.product_etag:
        return None
    if info.get("key", record.product_key) != record.product_key:
        return None
    return info


def cached_product(cache_dir: str | Path | None, record: CollectRecord) -> Path | None:
    """The cached product of *record* if its key, size and ETag match the index, else *None*.

    Products cached by the first version of this module, at
    ``<cache>/<source>/products/<collect_id>/<name>``, are reused when their
    metadata names the same key and ETag.
    """
    path = product_path(cache_dir, record)
    if _matching_product(path, record) is not None:
        return path
    legacy = path.parent.parent / path.name
    if _matching_product(legacy, record) is not None:
        return legacy
    return None


def ensure_product(
    bucket: PublicBucket, record: CollectRecord, cache_dir: str | Path | None
) -> tuple[Path, dict[str, Any]]:
    """Return the local product of *record*, downloading it when needed.

    The file is reused when its key, size and ETag (recorded in
    ``<file>.json``) match the index; a partial download of the same ETag is
    resumed. Downloads hold a lock (``<file>.lock``), so processes that share
    ``vhr_sar_cache_dir`` download a product once: the others wait and then
    reuse it.
    """
    check_product_key(record.source, record.product_key)
    hit = cached_product(cache_dir, record)
    if hit is not None:
        return hit, json.loads(hit.with_name(hit.name + ".json").read_text())
    path = product_path(cache_dir, record)
    meta_path = path.with_name(path.name + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive(path.with_name(path.name + ".lock")):
        hit = cached_product(cache_dir, record)  # downloaded by another process meanwhile
        if hit is not None:
            return hit, json.loads(hit.with_name(hit.name + ".json").read_text())
        logger.info(
            "Downloading %s (%.0f MB): s3://%s/%s",
            record.collect_id,
            record.product_bytes / 1e6,
            bucket.bucket,
            record.product_key,
        )
        t0 = time.time()
        meta_path.unlink(missing_ok=True)  # the file is replaced: no stale metadata
        bucket.download(
            record.product_key, path, size=record.product_bytes, etag=record.product_etag
        )
        info = {
            "key": record.product_key,
            "bucket": bucket.bucket,
            "size": record.product_bytes,
            "etag": record.product_etag,
            "last_modified": record.product_last_modified,
            "downloaded_utc": _dt.datetime.now(_UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seconds": round(time.time() - t0, 1),
        }
        tmp = meta_path.with_name(meta_path.name + ".tmp")
        tmp.write_text(json.dumps(info, indent=1))
        os.replace(tmp, meta_path)
    return path, info


def glo30_tile_name(lat: int, lon: int) -> str:
    """Copernicus GLO-30 tile name for the 1-degree cell with south-west corner (*lat*, *lon*)."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def _read_url_window(
    url: str, bounds: tuple[float, float, float, float], shape: tuple[int, int]
) -> np.ndarray | None:
    """Read *url* resampled (bilinear) onto the EPSG:4326 grid *bounds*/*shape*; None if missing."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject

    env = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_USERAGENT": _user_agent(),
        "GDAL_HTTP_MAX_RETRY": "4",
        "GDAL_HTTP_RETRY_DELAY": "2",
    }
    dst = np.full(shape, np.nan, dtype=np.float32)
    try:
        with rasterio.Env(**env), rasterio.open(f"/vsicurl/{url}") as src:
            reproject(
                rasterio.band(src, 1),
                dst,
                dst_transform=from_bounds(*bounds, shape[1], shape[0]),
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
    except rasterio.errors.RasterioIOError as exc:
        if "404" in str(exc) or "No such file" in str(exc) or "not recognized" in str(exc):
            return None
        raise
    return dst


def read_copernicus_dem(
    bounds: tuple[float, float, float, float], shape: tuple[int, int]
) -> tuple[np.ndarray, dict[str, str]]:
    """Copernicus DEM heights (m above EGM2008) on an EPSG:4326 grid, and their source per tile.

    Each 1-degree tile is read from GLO-30, else from GLO-90 (GLO-30 has no
    public tiles over Armenia and Azerbaijan), with a WARNING. Where neither
    has a tile the heights are 0, which is right over the sea only, so a
    WARNING says so: over land, the RPC orthorectification would then be off by
    about h / tan(incidence). Returns ``(heights, {tile: "GLO-30" | "GLO-90" |
    "none"})`` with tiles named like ``N40_E044``.
    """
    w, s, e, n = bounds
    out = np.full(shape, np.nan, dtype=np.float32)
    sources: dict[str, str] = {}
    for lat in range(math.floor(s), math.ceil(n)):
        for lon in range(math.floor(w), math.ceil(e)):
            name = glo30_tile_name(lat, lon)
            ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
            tile = f"{ns}{abs(lat):02d}_{ew}{abs(lon):03d}"
            part = _read_url_window(GLO30_URL.format(name=name), bounds, shape)
            sources[tile] = "GLO-30"
            if part is None:
                name90 = name.replace("_COG_10_", "_COG_30_")
                part = _read_url_window(GLO90_URL.format(name=name90), bounds, shape)
                sources[tile] = "GLO-90"
                if part is None:
                    sources[tile] = "none"
                    logger.warning(
                        "No Copernicus DEM tile %s (GLO-30 or GLO-90): heights there are set to "
                        "0 m, which is right over the sea only; over land the orthorectification "
                        "would be off by about h / tan(incidence).",
                        tile,
                    )
                    continue
                logger.warning(
                    "Copernicus GLO-30 has no public tile %s; using GLO-90 (90 m) there.", tile
                )
            fill = np.isnan(out) & np.isfinite(part)
            out[fill] = part[fill]
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32), sources


def read_egm2008(bounds: tuple[float, float, float, float], shape: tuple[int, int]) -> np.ndarray:
    """EGM2008 geoid undulation N (m; ellipsoidal height = orthometric height + N)."""
    grid = _read_url_window(EGM2008_URL, bounds, shape)
    if grid is None or not np.isfinite(grid).all():
        raise RuntimeError(f"Could not read the EGM2008 geoid grid {EGM2008_URL} for {bounds}")
    return grid


def build_dem_hae(
    bounds: tuple[float, float, float, float],
    out_path: str | Path,
    *,
    step_deg: float = 1.0 / 3600.0,
    read_dem: Callable[..., np.ndarray] | None = None,
    read_geoid: Callable[..., np.ndarray] | None = None,
) -> dict[str, Any]:
    """Write a DEM of ellipsoidal heights (WGS 84) for RPC orthorectification.

    Heights are Copernicus GLO-30 (GLO-90 where GLO-30 has no public tile;
    orthometric, EGM2008) plus the EGM2008 geoid undulation, on a 1 arc-second
    EPSG:4326 grid over *bounds* (lon/lat). Ignoring the geoid would misplace a
    scene by roughly 25-100 m (a 50 m height error moved a warp by 117 m in the
    prototype).

    *read_dem* (``(bounds, shape) -> heights``) and *read_geoid* default to
    :func:`read_copernicus_dem` and :func:`read_egm2008` (looked up when
    called); with the default, the source of every 1-degree tile is recorded
    (``"dem_tiles"``).

    Returns
    -------
    dict
        Path, grid, height statistics and, with the default *read_dem*, the
        DEM source per tile.
    """
    import rasterio
    from rasterio.transform import from_bounds

    read_geoid = read_geoid or read_egm2008
    w, s, e, n = bounds
    width = max(2, math.ceil((e - w) / step_deg))
    height = max(2, math.ceil((n - s) / step_deg))
    bounds = (w, n - height * step_deg, w + width * step_deg, n)
    tiles: dict[str, str] | None = None
    if read_dem is None:
        dem, tiles = read_copernicus_dem(bounds, (height, width))
    else:
        dem = read_dem(bounds, (height, width))
    geoid = read_geoid(bounds, (height, width))
    hae = (dem + geoid).astype(np.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    with rasterio.open(
        tmp,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_bounds(*bounds, width, height),
        compress="deflate",
        predictor=3,
        tiled=True,
    ) as dst:
        dst.write(hae, 1)
        other = sorted(f"{t} {src}" for t, src in (tiles or {}).items() if src != "GLO-30")
        dem_name = "Copernicus GLO-30" + (f" ({', '.join(other)})" if other else "")
        dst.update_tags(
            AGRIBOUND_DEM=f"{dem_name} + EGM2008 undulation (ellipsoidal heights, WGS 84)"
        )
    os.replace(tmp, out_path)
    return {
        "path": str(out_path),
        "bounds": [float(v) for v in bounds],
        "shape": [height, width],
        "orthometric_m": [float(np.min(dem)), float(np.median(dem)), float(np.max(dem))],
        "geoid_undulation_m": [float(np.min(geoid)), float(np.max(geoid))],
        **({"dem_tiles": tiles} if tiles is not None else {}),
    }


# ---------------------------------------------------------------------------
# Warping
# ---------------------------------------------------------------------------


class _RpcOptionNoise(logging.Filter):
    """Drop GDAL's notice that the warper ignores the RPC transformer options.

    rasterio passes ``RPC_DEM`` and ``RPC_DEMINTERPOLATION`` both to the RPC
    transformer, which uses them, and to the warp options, where GDAL logs
    "does not support option" (the DEM is applied: a +50 m DEM moves a UC Davis
    GEC by about 100 m).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "does not support option RPC_" not in record.getMessage()


@contextlib.contextmanager
def _quiet_rpc_options() -> Any:
    target = logging.getLogger("rasterio._env")
    noise = _RpcOptionNoise()
    target.addFilter(noise)
    try:
        yield
    finally:
        target.removeFilter(noise)


def _cached_dem_info(dem_path: Path) -> dict[str, Any]:
    """The JSON record of a cached DEM, with ``path`` set to where the DEM is now.

    The record keeps the path the DEM was built at; a cache directory that was
    moved, or that is shared under another mount point, would otherwise send
    the warp to a file that no longer exists there.
    """
    info = json.loads(dem_path.with_suffix(".json").read_text())
    info["path"] = str(dem_path)
    return info


def _shifted_transform(transform: Any, shift: tuple[float, float]) -> Any:
    """Grid transform sampled at ``P + shift`` (so content displaced by *shift* lands at P)."""
    from rasterio.transform import Affine

    t = transform
    return Affine(t.a, t.b, t.c + float(shift[0]), t.d, t.e, t.f + float(shift[1]))


def warp_product(
    path: str | Path,
    grid: Any,
    *,
    source: str,
    method: str,
    dem_path: str | Path | None = None,
    shift: tuple[float, float] = (0.0, 0.0),
    num_threads: int | None = None,
) -> np.ndarray:
    """Resample one amplitude product onto *grid*; 0 is nodata.

    Parameters
    ----------
    path : str or Path
        Umbra GEC or Capella GEO GeoTIFF.
    grid : ExportGrid
        Destination grid.
    source : str
        ``"umbra"`` or ``"capella"``.
    method : str
        ``"rpc_dem"`` (Umbra: RPCs + *dem_path*), ``"geotransform"`` (Umbra:
        the GEC's own rotated geotransform) or ``"geo"`` (Capella GEO).
    shift : tuple[float, float]
        ``(dx, dy)`` metres by which the product's content is displaced from
        the truth (east, north); the grid is sampled at ``P + shift``.

    Returns
    -------
    numpy.ndarray
        float32 ``(height, width)``: Umbra, the mean DN of the source pixels
        (``average``); Capella, the root mean square of the DN (``rms``), so
        ``(scale x rms)^2`` is the mean sigma0 intensity.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    if source not in PROVIDERS:
        raise ValueError(f"Unknown source {source!r}")
    threads = num_threads or max(1, min(8, os.cpu_count() or 1))
    dst = np.zeros((grid.height, grid.width), dtype=np.float32)
    common = {
        "destination": dst,
        "dst_transform": _shifted_transform(grid.transform, shift),
        "dst_crs": grid.crs,
        "src_nodata": 0,
        "dst_nodata": 0,
        "num_threads": threads,
        "warp_mem_limit": 1024,
    }
    try:
        with rasterio.open(path) as src, _quiet_rpc_options():
            if method == "rpc_dem":
                if dem_path is None:
                    raise ValueError("method='rpc_dem' needs dem_path")
                reproject(
                    rasterio.band(src, 1),
                    rpcs=src.rpcs,
                    src_crs="EPSG:4326",
                    resampling=Resampling.average,
                    RPC_DEM=str(dem_path),
                    RPC_DEMINTERPOLATION="bilinear",
                    **common,
                )
            elif method == "geotransform":
                reproject(rasterio.band(src, 1), resampling=Resampling.average, **common)
            elif method == "geo":
                reproject(rasterio.band(src, 1), resampling=Resampling.rms, **common)
            else:
                raise ValueError(f"Unknown warp method {method!r}")
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Warping {Path(path).name} ({method}) failed: {exc}") from exc
    return dst


def capella_db(rms_dn: np.ndarray, scale_factor: float) -> np.ndarray:
    """Capella GEO DN (``rms`` average) to sigma0 dB + :data:`CAPELLA_DB_OFFSET`; 0 stays 0.

    sigma0 dB = 20 log10(scale_factor x DN) (Capella SAR Products Format
    Specification); values are clipped at 0.01 so that only nodata is 0.
    """
    rms_dn = np.asarray(rms_dn, dtype=np.float32)
    valid = rms_dn > 0
    out = np.zeros(rms_dn.shape, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20.0 * np.log10(np.float32(scale_factor) * rms_dn[valid]) + CAPELLA_DB_OFFSET
    out[valid] = np.maximum(db, 0.01).astype(np.float32)
    return out


def to_amplitude(arr: np.ndarray, record: CollectRecord) -> np.ndarray:
    """Warped product values to the composite's amplitude scale (0 = nodata).

    Umbra: the mean DN as it is (an uncalibrated log scale). Capella:
    :func:`capella_db` with the record's scale factor.
    """
    if record.source == "capella":
        if not record.scale_factor:
            raise ValueError(f"Capella collect {record.collect_id} has no scale_factor")
        return capella_db(arr, record.scale_factor)
    return np.asarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Co-registration
# ---------------------------------------------------------------------------


def edge_map(img: np.ndarray, sigma: float = 2.0, *, log: bool = True) -> np.ndarray:
    """Edge strength of an amplitude image: (log,) normalised Gaussian smoothing, Sobel.

    Pixels with value 0 are nodata; edges within ``3 sigma + 2`` pixels of
    nodata are set to 0.
    """
    from scipy import ndimage

    valid = img > 0
    data = np.where(valid, np.log(np.maximum(img, 1e-3)) if log else img, 0.0).astype(np.float64)
    num = ndimage.gaussian_filter(data, sigma)
    den = ndimage.gaussian_filter(valid.astype(np.float64), sigma)
    smooth = np.where(den > 0.5, num / np.maximum(den, 1e-6), 0.0)
    grad = np.hypot(ndimage.sobel(smooth, 1), ndimage.sobel(smooth, 0))
    grad[~ndimage.binary_erosion(valid, iterations=int(3 * sigma) + 2)] = 0.0
    return grad


def optical_edge_map(bands: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Summed Sobel edge strength of Gaussian-smoothed optical bands (0 = nodata)."""
    from scipy import ndimage

    bands = np.asarray(bands, dtype=np.float64)
    if bands.ndim == 2:
        bands = bands[None]
    valid = np.all(bands > 0, axis=0)
    grad = np.zeros(bands.shape[1:], dtype=np.float64)
    for band in bands:
        smooth = ndimage.gaussian_filter(band, sigma)
        grad += np.hypot(ndimage.sobel(smooth, 1), ndimage.sobel(smooth, 0))
    grad[~ndimage.binary_erosion(valid, iterations=int(3 * sigma) + 2)] = 0.0
    return grad


def _normalise_edges(edges: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.where(valid, edges, 0.0)
    hi = np.percentile(x[valid], 99) if valid.any() else 1.0
    x = np.clip(x / max(hi, 1e-9), 0, 1)
    return np.where(valid, x - x[valid].mean(), 0.0) if valid.any() else x


def register_tile(
    ref: np.ndarray, mov: np.ndarray, max_shift_px: int
) -> tuple[float, float, float, float]:
    """Displacement ``(d_row, d_col)`` of *mov*'s content relative to *ref* (pixels).

    Normalised circular cross-correlation (FFT) of the normalised edge maps,
    restricted to ``|shift| <= max_shift_px``, with a parabolic sub-pixel
    fit. Returns ``(d_row, d_col, peak, peak_ratio)``; *peak_ratio* is the
    peak over the highest value more than 6 pixels away.
    """
    a = _normalise_edges(ref, ref != 0)
    b = _normalise_edges(mov, mov != 0)
    spec = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    corr = np.real(np.fft.ifft2(spec)) / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12)
    n0, n1 = corr.shape
    k0 = np.fft.fftfreq(n0, 1 / n0)
    k1 = np.fft.fftfreq(n1, 1 / n1)
    win = (np.abs(k0)[:, None] <= max_shift_px) & (np.abs(k1)[None, :] <= max_shift_px)
    masked = np.where(win, corr, -np.inf)
    i, j = np.unravel_index(int(np.argmax(masked)), corr.shape)
    peak = float(corr[i, j])

    def subpixel(cm: float, cp: float, c0: float) -> float:
        den = cm - 2 * c0 + cp
        return 0.0 if den == 0 else 0.5 * (cm - cp) / den

    di = subpixel(corr[(i - 1) % n0, j], corr[(i + 1) % n0, j], peak)
    dj = subpixel(corr[i, (j - 1) % n1], corr[i, (j + 1) % n1], peak)
    d0 = np.minimum(np.abs(k0 - k0[i]), n0 - np.abs(k0 - k0[i]))[:, None]
    d1 = np.minimum(np.abs(k1 - k1[j]), n1 - np.abs(k1 - k1[j]))[None, :]
    far = np.hypot(d0, d1) > 6
    second = float(np.max(np.where(win & far, corr, -np.inf)))
    ratio = peak / second if second > 0 else float("inf")
    return float(k0[i] + di), float(k1[j] + dj), peak, ratio


def block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block average of valid (> 0) pixels; blocks with half or fewer valid pixels are 0."""
    if factor == 1:
        return arr
    if arr.ndim == 3:
        return np.stack([block_mean(a, factor) for a in arr])
    h, w = (arr.shape[0] // factor) * factor, (arr.shape[1] // factor) * factor
    blocks = arr[:h, :w].reshape(h // factor, factor, w // factor, factor)
    valid = blocks > 0
    n = valid.sum((1, 3))
    total = (blocks * valid).sum((1, 3))
    return np.where(n > factor * factor * 0.5, total / np.maximum(n, 1), 0).astype(np.float32)


def estimate_shift(
    ref_edges: np.ndarray,
    mov_edges: np.ndarray,
    res: float,
    *,
    tile_px: int,
    max_shift_m: float,
    min_valid: float = 0.6,
    min_peak: float = 0.2,
    tol_m: float = 3.0,
    radius_cap: float = INLIER_RADIUS_CAP,
) -> dict[str, Any]:
    """Robust displacement of *mov* relative to *ref* from tiles (metres, east/north).

    Tiles of *tile_px* pixels with 50 % overlap where at least *min_valid* of
    the pixels have edges in both maps are registered with
    :func:`register_tile`; tiles with a peak of at least *min_peak* vote, and
    the result is the median of the inlier tiles: those within
    ``min(max(tol_m, 3 MAD), radius_cap * tol_m)`` of the first median. The cap
    keeps a scatter of unrelated tile estimates from widening the radius until
    every tile counts as an inlier. ``dx`` is east and ``dy`` north: the
    displacement of the moving content relative to the reference.
    ``inlier_fraction`` is the share of the voting tiles that are inliers
    (:func:`register_adaptive` needs a majority).
    """
    h, w = ref_edges.shape
    tile_px = int(min(tile_px, h, w))
    step = max(1, tile_px // 2)
    max_px = int(max(1, min(max_shift_m / res, tile_px // 4)))
    tiles = []
    for r0 in range(0, h - tile_px + 1, step):
        for c0 in range(0, w - tile_px + 1, step):
            a = ref_edges[r0 : r0 + tile_px, c0 : c0 + tile_px]
            b = mov_edges[r0 : r0 + tile_px, c0 : c0 + tile_px]
            if ((a != 0) & (b != 0)).mean() < min_valid:
                continue
            drow, dcol, peak, ratio = register_tile(a, b, max_px)
            tiles.append(
                {
                    "row": r0,
                    "col": c0,
                    "dx": dcol * res,
                    "dy": -drow * res,
                    "peak": peak,
                    "ratio": ratio,
                }
            )
    out: dict[str, Any] = {"n_tiles": len(tiles), "tile_px": tile_px, "res_m": res}
    good = [t for t in tiles if t["peak"] >= min_peak]
    out["n_peak_ok"] = len(good)
    if not good:
        out["n_inliers"] = 0
        out["tiles"] = tiles
        return out
    dx = np.array([t["dx"] for t in good])
    dy = np.array([t["dy"] for t in good])
    mx, my = float(np.median(dx)), float(np.median(dy))
    dist = np.hypot(dx - mx, dy - my)
    radius = min(max(tol_m, 3 * 1.4826 * float(np.median(dist))), radius_cap * tol_m)
    inl = dist <= radius
    spread = None
    if inl.any():
        mx, my = float(np.median(dx[inl])), float(np.median(dy[inl]))
        spread = float(np.sqrt(np.mean((dx[inl] - mx) ** 2 + (dy[inl] - my) ** 2)))
    for t, keep in zip(good, inl, strict=True):
        t["inlier"] = bool(keep)
    out.update(
        dx=mx,
        dy=my,
        n_inliers=int(inl.sum()),
        inlier_fraction=float(inl.mean()),
        inlier_radius_m=float(radius),
        peak_median=float(np.median([t["peak"] for t in good])),
        spread_rms_m=spread,
        tiles=tiles,
    )
    return out


def _tile_px(shape: tuple[int, int]) -> int:
    side = min(shape)
    return 1024 if side >= 3072 else 512 if side >= 1536 else 256


def register_adaptive(
    ref: np.ndarray,
    mov: np.ndarray,
    res: float,
    *,
    kind: str,
    log: bool = True,
) -> dict[str, Any]:
    """Register *mov* to *ref* at *res*, falling back to 2x and 4x block means.

    *kind* ``"sar"``: both are SAR amplitudes (max shift 100 m, inlier
    tolerance 3 m, peak >= 0.3 at full resolution); ``"optical"``: *ref* holds
    optical bands ``(bands, H, W)`` (max shift 150 m, tolerance 6 m, peak >=
    0.2). A level succeeds when its inliers (tile estimates within at most
    :data:`INLIER_RADIUS_CAP` tolerances of the median) are at least
    :data:`MIN_INLIER_FRACTION` of the tiles with a peak and number at least
    ``min(n, max(3, tiles // 2))``, where *n* is 8 (SAR) or 15 (optical) at full
    resolution and 6 or 8 at the coarser levels. Unrelated images fail every
    level. Returns the :func:`estimate_shift` result of the level used (``ok``
    tells whether it succeeded; ``scale`` is the block factor).
    """
    if kind not in ("sar", "optical"):
        raise ValueError(f"kind must be 'sar' or 'optical', got {kind!r}")
    result: dict[str, Any] = {}
    for factor in (1, 2, 4):
        a = block_mean(ref, factor)
        b = block_mean(mov, factor)
        ref_edges = edge_map(a, log=log) if kind == "sar" else optical_edge_map(a)
        mov_edges = edge_map(b, log=log)
        level_res = res * factor
        tile_px = _tile_px(mov_edges.shape) if factor == 1 else _tile_px(mov_edges.shape) // 2
        tile_px = max(128, tile_px)
        params = (
            {"max_shift_m": 100.0, "tol_m": 3.0, "min_peak": 0.3 if factor == 1 else 0.2}
            if kind == "sar"
            else {"max_shift_m": 150.0, "tol_m": 6.0, "min_peak": 0.2}
        )
        result = estimate_shift(ref_edges, mov_edges, level_res, tile_px=tile_px, **params)
        result["scale"] = factor
        need = (8 if kind == "sar" else 15) if factor == 1 else (6 if kind == "sar" else 8)
        need = min(need, max(3, result["n_tiles"] // 2))
        result["min_inliers"] = need
        if (
            result.get("n_inliers", 0) >= need
            and result.get("inlier_fraction", 0.0) >= MIN_INLIER_FRACTION
        ):
            result["ok"] = True
            return result
    result["ok"] = False
    return result


def _consensus_text(result: dict[str, Any], res: float) -> str:
    """Summary of a failed :func:`register_adaptive` level for error messages."""
    radius = result.get("inlier_radius_m")
    within = f" within {radius:.1f} m" if radius is not None else ""
    return (
        f"at {result['scale'] * res:g} m: {result['n_tiles']} tiles, "
        f"{result.get('n_peak_ok', 0)} with a correlation peak, "
        f"{result.get('n_inliers', 0)} of these consistent{within}; at least "
        f"{result.get('min_inliers')} and a majority are needed"
    )


def shift_image(arr: np.ndarray, dx: float, dy: float, res: float) -> np.ndarray:
    """Move content displaced by ``(dx, dy)`` metres back (bilinear; 0 = nodata)."""
    from scipy import ndimage

    valid = (arr > 0).astype(np.float32)
    # Content at P + (dx, dy) moves to P: rows +dy/res (north is up), cols -dx/res.
    offset = (dy / res, -dx / res)
    data = ndimage.shift(np.where(arr > 0, arr, 0).astype(np.float32), offset, order=1, cval=0)
    weight = ndimage.shift(valid, offset, order=1, cval=0)
    return np.where(weight > 0.99, data / np.maximum(weight, 1e-6), 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Radiometry and compositing
# ---------------------------------------------------------------------------


def level_normalise(
    arrays: Sequence[np.ndarray], *, min_common: int = 1000
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Shift each date additively so its median matches the mean of the dates' medians.

    Medians are taken over the pixels every date covers (each date's own
    valid pixels when fewer than *min_common* are common). Values are
    log-scaled (Umbra DN, Capella dB), so the shift is additive. Input pixels
    of 0 are nodata; in the output they are NaN (a valid value may become 0 or
    negative after the shift).
    """
    valid = [a > 0 for a in arrays]
    common = np.logical_and.reduce(valid) if arrays else np.zeros(0, bool)
    use_common = bool(arrays) and int(common.sum()) >= min_common
    levels = []
    for a, v in zip(arrays, valid, strict=True):
        values = a[common] if use_common else a[v]
        levels.append(float(np.median(values)) if values.size else float("nan"))
    finite = [lvl for lvl in levels if math.isfinite(lvl)]
    target = float(np.mean(finite)) if finite else 0.0
    offsets = [target - lvl if math.isfinite(lvl) else 0.0 for lvl in levels]
    out = [
        np.where(v, a + np.float32(off), np.nan).astype(np.float32)
        for a, v, off in zip(arrays, valid, offsets, strict=True)
    ]
    return out, {
        "levels": levels,
        "target": target,
        "offsets": offsets,
        "basis": "common" if use_common else "per-date",
        "n_common": int(common.sum()) if arrays else 0,
    }


def median_composite(
    arrays: Sequence[np.ndarray],
    *,
    lower: float,
    upper: float | None = None,
    rows_per_block: int = 512,
) -> np.ndarray:
    """Per-pixel median of the finite values, clipped to [*lower*, *upper*]; 0 = nodata.

    *arrays* are NaN-coded (:func:`level_normalise`). A pixel needs
    ``min(2, len(arrays))`` valid dates; other pixels are 0.
    """
    need = min(2, len(arrays))
    h, w = arrays[0].shape
    out = np.zeros((h, w), dtype=np.float32)
    for r0 in range(0, h, rows_per_block):
        r1 = min(h, r0 + rows_per_block)
        stack = np.stack([a[r0:r1] for a in arrays]).astype(np.float32)
        count = np.isfinite(stack).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
        med = np.clip(med, lower, upper if upper is not None else np.inf)
        out[r0:r1] = np.where((count >= need) & np.isfinite(med), med, 0)
    return out


def closest_pair(records: Sequence[CollectRecord]) -> tuple[int, int]:
    """Indices of the two records closest in time (earliest pair on ties)."""
    order = sorted(range(len(records)), key=lambda i: records[i].datetime)
    best = None
    for a, b in zip(order, order[1:], strict=False):
        gap = (records[b].time - records[a].time).total_seconds()
        if best is None or gap < best[0]:
            best = (gap, a, b)
    if best is None:
        raise ValueError("closest_pair needs at least two records")
    return best[1], best[2]


def pair_mean(
    a: np.ndarray, b: np.ndarray, *, lower: float, upper: float | None = None
) -> np.ndarray:
    """Mean of two NaN-coded dates where both are valid (clipped), else 0."""
    both = np.isfinite(a) & np.isfinite(b)
    mean = np.clip((a + b) / 2, lower, upper if upper is not None else np.inf)
    return np.where(both, mean, 0).astype(np.float32)


def date_band(a: np.ndarray, *, lower: float, upper: float | None = None) -> np.ndarray:
    """One NaN-coded date as an output band (clipped; NaN -> 0)."""
    return np.where(
        np.isfinite(a), np.clip(a, lower, upper if upper is not None else np.inf), 0
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Composite builder
# ---------------------------------------------------------------------------


def _expand_grid(grid: Any, margin_m: float) -> Any:
    """*grid* grown by *margin_m* on every side (aligned to its pixels)."""
    from rasterio.transform import Affine

    from agribound.composites.gee import ExportGrid

    t = grid.transform
    px = max(0, int(math.ceil(margin_m / abs(t.a))))
    return ExportGrid(
        grid.crs,
        Affine(t.a, 0.0, t.c - px * t.a, 0.0, t.e, t.f - px * t.e),
        grid.width + 2 * px,
        grid.height + 2 * px,
    )


def _read_base_on_grid(path: str | Path, grid: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Up to three bands of an optical base, averaged onto *grid* (float32, 0 = nodata)."""
    import pyproj
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(path) as src:
        crs = pyproj.CRS.from_user_input(src.crs) if src.crs else None
        if crs is None:
            raise ValueError(f"The co-registration base {path} has no CRS")
        res = abs(src.transform.a)
        if crs.is_geographic:
            lat = (src.bounds.bottom + src.bounds.top) / 2
            res = res * 111_320.0 * math.cos(math.radians(lat))
        else:
            # Pixel sizes are in the CRS's linear unit (US survey feet for many State
            # Plane orthophotos: a 2 ft pixel is 0.61 m).
            axes = crs.axis_info
            factor = axes[0].unit_conversion_factor if axes else None
            res = res * (factor or 1.0)
        if res > MAX_BASE_RESOLUTION_M * 1.05:
            raise ValueError(
                f"The co-registration base {path} has {res:.2f} m pixels; a base of "
                f"{MAX_BASE_RESOLUTION_M:g} m or finer is needed (the offsets to correct are "
                "2-26 m)."
            )
        count = min(3, src.count)
        out = np.zeros((count, grid.height, grid.width), dtype=np.float32)
        for i in range(count):
            reproject(
                rasterio.band(src, i + 1),
                out[i],
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                resampling=Resampling.average,
                src_nodata=src.nodata if src.nodata is not None else 0,
                dst_nodata=0,
            )
    return out, {"path": str(path), "resolution_m": round(res, 3), "bands": count}


def _naip_plus_base(config: AgriboundConfig, grid: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
    """The latest USGS NAIP Plus vintage over *grid* at 1 m, or ``(None, reason)``."""
    from agribound.clients.usgs_naip_plus import USGSNAIPPlusClient
    from agribound.composites.gee import grid_footprint_4326
    from agribound.composites.usgs import USGSNAIPPlusCompositeBuilder, _to_3857

    outline = grid_footprint_4326(grid)
    client = USGSNAIPPlusClient(
        service_url=config.usgs_service_url,
        timeout_s=config.usgs_timeout_s,
        retries=config.usgs_retries,
    )
    counts = client.query_year_counts(_to_3857(outline).bounds, "Category = 1")
    first, last = SOURCE_REGISTRY["usgs-naip-plus"]["year_range"]
    years = sorted(y for y in counts if first <= y <= (last or y))
    if not years:
        return None, {"reason": "USGS NAIP Plus has no imagery over the study area"}
    year = years[-1]
    w, s, e, n = outline.bounds
    base_config = AgriboundConfig(
        source="usgs-naip-plus",
        engine="delineate-anything",
        year=year,
        study_area=f"bbox:{w},{s},{e},{n}",
        export_crs=grid.crs,
        lulc_filter=False,
        cache_dir=str(config.get_working_dir()),
        output_path=str(Path(config.get_working_dir()) / "vhr_sar_base.gpkg"),
        usgs_service_url=config.usgs_service_url,
        usgs_timeout_s=config.usgs_timeout_s,
        usgs_retries=config.usgs_retries,
        tile_size=config.tile_size,
    )
    logger.info("Co-registration base: USGS NAIP Plus %d at 1 m", year)
    path = USGSNAIPPlusCompositeBuilder().build(base_config, resolution_m=MAX_BASE_RESOLUTION_M)
    bands, info = _read_base_on_grid(path, grid)
    info.update(kind="usgs-naip-plus", year=year, years_available=sorted(counts))
    return bands, info


def _optical_base(config: AgriboundConfig, grid: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
    """The co-registration base on *grid*: the user's file, else NAIP Plus, else None."""
    if config.vhr_sar_coregister_base:
        base = Path(config.vhr_sar_coregister_base).expanduser()
        if not base.exists():
            raise FileNotFoundError(f"vhr_sar_coregister_base not found: {base}")
        bands, info = _read_base_on_grid(base, grid)
        info["kind"] = "user"
        return bands, info
    try:
        return _naip_plus_base(config, grid)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read the USGS NAIP Plus co-registration base ({type(exc).__name__}: "
            f"{exc}). Set vhr_sar_coregister_base to an orthophoto of 1 m or finer, or "
            "vhr_sar_coregister='sar' to register the dates to each other only."
        ) from exc


def _cache_key_parts(
    config: AgriboundConfig, selection: Selection, snapshot: str
) -> tuple[str, ...]:
    from agribound.composites.gee import COMPOSITE_RECIPE_VERSION

    settings = {
        "resolution_m": float(config.vhr_sar_resolution_m),
        "coregister": config.vhr_sar_coregister,
        "coregister_base": config.vhr_sar_coregister_base,
        "bands": config.vhr_sar_bands,
        "polarization": config.vhr_sar_polarization,
        "max_collects": config.vhr_sar_max_collects,
        "calibration": config.vhr_sar_calibration,
        "pinned": list(config.vhr_sar_collect_ids or []),
    }
    if config.vhr_sar_coregister_base:
        # The file, not only its name: a re-exported or replaced orthophoto at the same
        # path (or the same relative path elsewhere) gives another composite.
        from agribound._cache import _file_signature

        resolved = str(Path(config.vhr_sar_coregister_base).expanduser().resolve())
        mtime, size = _file_signature(resolved)
        settings["coregister_base_file"] = f"{resolved}:{size}:{mtime}"
    collects = sorted(f"{r.collect_id}:{r.product_etag}" for r in selection.chosen)
    return (
        COMPOSITE_RECIPE_VERSION,
        VHR_SAR_RECIPE_TAG,
        json.dumps(settings, sort_keys=True),
        ",".join(collects),
        f"snapshot={snapshot}",
    )


class VHRSARCompositeBuilder(CompositeBuilder):
    """Amplitude composite of Umbra GEC or Capella GEO collects (see the module docstring).

    Attributes
    ----------
    last_metadata : dict
        The ``AGRIBOUND_*`` tags of the most recent :meth:`build`.
    bucket_factory : callable
        ``(provider) -> PublicBucket`` (tests replace it with a fake).
    """

    bucket_factory: Callable[[Provider], PublicBucket] = staticmethod(
        lambda provider: PublicBucket(provider.bucket, provider.region)
    )

    def __init__(self) -> None:
        self.last_metadata: dict[str, Any] = {}

    def get_band_mapping(self, source: str) -> dict[str, str]:
        """Return the canonical band mapping (R, G and B -> MEDIAN)."""
        return dict(SOURCE_REGISTRY.get(source, {}).get("canonical_bands") or {})

    def build(self, config: AgriboundConfig) -> str:
        """Select, download, orthorectify, co-register and composite the collects.

        Returns
        -------
        str
            Path of the float32 composite (cached with
            :func:`agribound._cache.cache_path`).

        Raises
        ------
        NoDataError
            (A :class:`ValueError`.) If no collect in the window covers the study
            area, fewer than two consistent collects do, or the composite has no
            valid pixel inside it.
        ValueError
            If the study area is larger than one scene, pinned collects are
            invalid, or the downloads exceed ``vhr_sar_max_download_gb``.
        RuntimeError
            If the DEM, a warp or the co-registration fails.
        """
        import pyproj

        from agribound._cache import cache_key, cache_path
        from agribound.composites.gee import (
            compute_export_grid,
            date_window,
            raster_valid_fraction,
            resolve_export_crs,
            study_area_geometry_4326,
        )

        provider = get_provider(config.source)
        if not config.study_area:
            raise ValueError(f"study_area is required when source={config.source!r}")
        geom = study_area_geometry_4326(config)
        crs = resolve_export_crs(config.export_crs, geom)
        if pyproj.CRS.from_user_input(crs).is_geographic:
            raise ValueError(
                f"export_crs={config.export_crs!r} is geographic; the {provider.source} composite "
                "and its co-registration need a projected CRS in metres. Use 'utm' (default) or "
                "a projected EPSG code."
            )
        res = float(config.vhr_sar_resolution_m)
        grid = compute_export_grid(geom, crs, res)
        window = date_window(config)
        cache_dir = resolve_cache_dir(config.vhr_sar_cache_dir)
        bucket = self.bucket_factory(provider)

        index = load_index(provider.source, cache_dir, window_end=window[1], bucket=bucket)
        selection = select_collects(
            index.records,
            geom,
            window,
            source=provider.source,
            polarization=config.vhr_sar_polarization,
            max_collects=config.vhr_sar_max_collects,
            calibration=config.vhr_sar_calibration,
            pinned_ids=config.vhr_sar_collect_ids,
            snapshot=index.snapshot,
        )
        for message in selection.warnings:
            logger.warning(message)
        if config.vhr_sar_bands == "median+dates" and len(selection.chosen) < 2:
            raise ValueError(
                "vhr_sar_bands='median+dates' needs at least 2 collects (Delineate-Anything "
                "reads the pair mean as G); use vhr_sar_bands='median' for a single date."
            )
        parts = _cache_key_parts(config, selection, index.snapshot)
        out_path = cache_path(config, f"{provider.source}_composite", ".tif", *parts)
        manifest_path = out_path.with_suffix(".json")
        if out_path.exists() and manifest_path.exists():
            from agribound.composites.gee import read_composite_tags

            logger.info("Using cached %s composite: %s", provider.source, out_path)
            self.last_metadata = read_composite_tags(out_path)
            return str(out_path)

        # Downloads (within the budget) -------------------------------------------
        missing = [r for r in selection.chosen if cached_product(cache_dir, r) is None]
        need_gb = sum(r.product_bytes for r in missing) / 1e9
        if need_gb > config.vhr_sar_max_download_gb:
            raise ValueError(
                f"The {len(selection.chosen)} selected {provider.source} collects need "
                f"{need_gb:.2f} GB of {provider.product} downloads ({len(missing)} not cached), "
                f"more than vhr_sar_max_download_gb={config.vhr_sar_max_download_gb:g}. Raise "
                "the limit, or lower vhr_sar_max_collects."
            )
        products = {}
        downloads = {}
        for record in selection.chosen:
            path, info = ensure_product(bucket, record, cache_dir)
            products[record.collect_id] = path
            downloads[record.collect_id] = info
        access_date = max(i["downloaded_utc"][:10] for i in downloads.values())

        # Orthorectification inputs -------------------------------------------------
        t0 = time.time()
        margin_grid = (
            _expand_grid(grid, REGISTRATION_MARGIN_M)
            if config.vhr_sar_coregister != "none"
            else grid
        )
        methods, dem_info = self._methods(provider, selection, products, margin_grid, cache_dir)

        # Co-registration -------------------------------------------------------------
        shifts = {r.collect_id: (0.0, 0.0) for r in selection.chosen}
        reg_info: dict[str, Any] = {"mode": config.vhr_sar_coregister, "dates": {}}
        if config.vhr_sar_coregister != "none":
            shifts, reg_info = self._coregister(
                provider, selection, products, methods, margin_grid, dem_info, config
            )

        # Final warps, radiometry, composite ---------------------------------------
        arrays = []
        for record in selection.chosen:
            raw = warp_product(
                products[record.collect_id],
                grid,
                source=provider.source,
                method=methods[record.collect_id],
                dem_path=dem_info.get("path"),
                shift=shifts[record.collect_id],
            )
            arrays.append(to_amplitude(raw, record))
        normalised, levels = level_normalise(arrays)
        lower, upper = (1.0, 255.0) if provider.source == "umbra" else (0.01, None)
        bands = [median_composite(normalised, lower=lower, upper=upper)]
        names = ["MEDIAN"]
        if config.vhr_sar_bands == "median+dates" and len(normalised) >= 2:
            i, j = closest_pair(selection.chosen)
            bands.append(pair_mean(normalised[i], normalised[j], lower=lower, upper=upper))
            names.append("PAIR_MEAN")
            for k, (record, arr) in enumerate(zip(selection.chosen, normalised, strict=True)):
                bands.append(date_band(arr, lower=lower, upper=upper))
                names.append(f"D{k + 1:02d}_{record.date}")
        del arrays, normalised

        tags = self._tags(
            config,
            provider,
            selection,
            index,
            grid,
            names,
            methods,
            shifts,
            reg_info,
            levels,
            downloads,
            access_date,
            cache_key(config, *parts),
        )
        self._write(out_path, np.stack(bands), grid, names, tags)
        fraction = raster_valid_fraction(out_path, geom, require="all")
        if fraction <= 0.0:
            out_path.unlink(missing_ok=True)
            raise NoDataError(
                f"The {provider.source} composite has no valid pixel inside the study area."
            )
        import rasterio

        with rasterio.open(out_path, "r+") as dst:
            dst.update_tags(AGRIBOUND_VALID_FRACTION=str(round(fraction, 6)))
        tags["AGRIBOUND_VALID_FRACTION"] = round(fraction, 6)
        if fraction < 0.95:
            logger.warning(
                "The %s composite has valid data in only %.1f%% of the study area.",
                provider.source,
                100 * fraction,
            )
        self.last_metadata = dict(tags)
        manifest = {
            "source": provider.source,
            "recipe": VHR_SAR_RECIPE_TAG,
            "index": {
                "path": str(index.path) if index.path else None,
                "snapshot": index.snapshot,
                "stats": index.stats,
            },
            "window": list(window),
            "grid": {
                "crs": grid.crs,
                "transform": list(grid.crs_transform),
                "width": grid.width,
                "height": grid.height,
            },
            "selection": {
                "pinned": selection.pinned,
                "counts": selection.counts,
                "group": selection.group,
                "groups": selection.groups,
                "reference": selection.reference.collect_id,
                "collects": [r.to_dict() for r in selection.chosen],
            },
            "downloads": downloads,
            "orthorectification": {"methods": methods, "dem": dem_info},
            "coregistration": reg_info,
            "levels": levels,
            "bands": names,
            "tags": {k: (v if isinstance(v, int | float) else str(v)) for k, v in tags.items()},
            "seconds": round(time.time() - t0, 1),
        }
        manifest_path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
        logger.info("%s composite written to: %s", provider.source, out_path)
        return str(out_path)

    # -- steps ---------------------------------------------------------------------

    def _methods(
        self,
        provider: Provider,
        selection: Selection,
        products: dict[str, Path],
        grid: Any,
        cache_dir: Path,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Warp method per collect and, for Umbra RPCs, the ellipsoidal DEM."""
        import rasterio

        if provider.source == "capella":
            logger.warning(
                "Capella GEO is terrain-corrected with a 30 m DSM, which does not resolve "
                "trees: tree tops are displaced (about 12 m for a 10 m tree at 40 degrees "
                "incidence)."
            )
            return {r.collect_id: "geo" for r in selection.chosen}, {}
        methods = {}
        for record in selection.chosen:
            with rasterio.open(products[record.collect_id]) as src:
                has_rpc = src.rpcs is not None
            if has_rpc:
                methods[record.collect_id] = "rpc_dem"
            else:
                logger.warning(
                    "Umbra GEC %s has no RPCs; using its geotransform (no terrain correction; "
                    "tile-to-tile geolocation scatter was 3.7-6.6 m with it, against 1.0-3.5 m "
                    "with RPCs + DEM, and 18-24 m on Mallorca).",
                    record.collect_id,
                )
                methods[record.collect_id] = "geotransform"
        if "rpc_dem" not in methods.values():
            return methods, {}
        from agribound.composites.gee import grid_footprint_4326

        outline = grid_footprint_4326(grid)
        for record in selection.chosen:
            outline = outline.union(record.geometry())
        w, s, e, n = outline.bounds
        pad = 0.02
        bounds = (
            math.floor((w - pad) * 100) / 100,
            math.floor((s - pad) * 100) / 100,
            math.ceil((e + pad) * 100) / 100,
            math.ceil((n + pad) * 100) / 100,
        )
        name = "glo30_egm2008_hae_" + "_".join(f"{v:.2f}" for v in bounds) + ".tif"
        dem_path = cache_dir / "dem" / name
        info_path = dem_path.with_suffix(".json")
        if dem_path.exists() and info_path.exists():
            return methods, _cached_dem_info(dem_path)
        dem_path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive(dem_path.with_name(dem_path.name + ".lock")):
            if dem_path.exists() and info_path.exists():  # built by another process meanwhile
                return methods, _cached_dem_info(dem_path)
            try:
                info = build_dem_hae(bounds, dem_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not build the Copernicus DEM + EGM2008 heights for {bounds} ({exc}); "
                    "they are needed for the RPC orthorectification of Umbra GEC products."
                ) from exc
            tmp = info_path.with_name(info_path.name + ".tmp")
            tmp.write_text(json.dumps(info, indent=1))
            os.replace(tmp, info_path)
        return methods, info

    def _coregister(
        self,
        provider: Provider,
        selection: Selection,
        products: dict[str, Path],
        methods: dict[str, str],
        grid: Any,
        dem_info: dict[str, Any],
        config: AgriboundConfig,
    ) -> tuple[dict[str, tuple[float, float]], dict[str, Any]]:
        """Total shift per collect: SAR to SAR (reference date), then stack to an optical base."""
        res = abs(grid.transform.a)
        warped = {}
        for record in selection.chosen:
            raw = warp_product(
                products[record.collect_id],
                grid,
                source=provider.source,
                method=methods[record.collect_id],
                dem_path=dem_info.get("path"),
            )
            warped[record.collect_id] = to_amplitude(raw, record)
        ref_id = selection.reference.collect_id
        info: dict[str, Any] = {
            "mode": config.vhr_sar_coregister,
            "reference": ref_id,
            "grid_margin_m": REGISTRATION_MARGIN_M,
            "dates": {},
        }
        rel = {ref_id: (0.0, 0.0)}
        for record in selection.chosen:
            cid = record.collect_id
            if cid == ref_id:
                info["dates"][cid] = {"reference": True}
                continue
            result = register_adaptive(warped[ref_id], warped[cid], res, kind="sar")
            info["dates"][cid] = {k: v for k, v in result.items() if k != "tiles"}
            if not result["ok"]:
                raise RuntimeError(
                    f"Co-registration of {provider.source} collect {cid} ({record.date}) to the "
                    f"reference date {selection.reference.date} failed: the tile estimates do "
                    f"not agree ({_consensus_text(result, res)}). Set vhr_sar_coregister='none', "
                    "or pin other collects with vhr_sar_collect_ids."
                )
            rel[cid] = (result["dx"], result["dy"])
        base_shift = (0.0, 0.0)
        info["base"] = None
        if config.vhr_sar_coregister in ("auto", "optical"):
            base, base_info = _optical_base(config, grid)
            info["base"] = base_info
            if base is None:
                if config.vhr_sar_coregister == "optical":
                    raise RuntimeError(
                        f"vhr_sar_coregister='optical' but no base is available "
                        f"({base_info.get('reason')}); set vhr_sar_coregister_base to an "
                        "orthophoto of 1 m or finer, or use vhr_sar_coregister='sar'."
                    )
                logger.warning(
                    "No optical base of %g m or finer for the %s composite (%s): the dates are "
                    "co-registered to each other only, so the composite keeps the absolute "
                    "geolocation error of the reference date (2.3-25.9 m, median 6.8 m, in the "
                    "prototype). Set vhr_sar_coregister_base to an orthophoto to correct it.",
                    MAX_BASE_RESOLUTION_M,
                    provider.source,
                    base_info.get("reason"),
                )
            else:
                aligned = [
                    shift_image(warped.pop(r.collect_id), *rel[r.collect_id], res)
                    for r in selection.chosen
                ]
                normalised, _ = level_normalise(aligned)
                del aligned
                lower = 1.0 if provider.source == "umbra" else 0.01
                stack = median_composite(normalised, lower=lower)
                del normalised
                result = register_adaptive(base, stack, res, kind="optical")
                info["base_registration"] = {k: v for k, v in result.items() if k != "tiles"}
                if not result["ok"]:
                    raise RuntimeError(
                        f"Co-registration of the {provider.source} stack to the optical base "
                        f"failed: the tile estimates do not agree ({_consensus_text(result, res)}"
                        "). Check that the base shows the study area, or set "
                        "vhr_sar_coregister='sar' to skip the optical base."
                    )
                base_shift = (result["dx"], result["dy"])
        shifts = {}
        for record in selection.chosen:
            cid = record.collect_id
            dx, dy = rel[cid]
            shifts[cid] = (dx + base_shift[0], dy + base_shift[1])
            info["dates"][cid]["total_shift_m"] = [round(v, 3) for v in shifts[cid]]
        return shifts, info

    def _tags(
        self,
        config: AgriboundConfig,
        provider: Provider,
        selection: Selection,
        index: ArchiveIndex,
        grid: Any,
        names: list[str],
        methods: dict[str, str],
        shifts: dict[str, tuple[float, float]],
        reg_info: dict[str, Any],
        levels: dict[str, Any],
        downloads: dict[str, Any],
        access_date: str,
        key: str,
    ) -> dict[str, Any]:
        from agribound._version import __version__
        from agribound.composites.gee import COMPOSITE_RECIPE_VERSION, date_window

        window = date_window(config)
        collects = []
        for k, record in enumerate(selection.chosen):
            reg = reg_info.get("dates", {}).get(record.collect_id, {})
            collects.append(
                {
                    "id": record.collect_id,
                    "time": record.datetime,
                    "platform": record.platform,
                    "pol": record.polarization,
                    "pass": record.orbit_state,
                    "look_side": record.look_side,
                    "incidence_deg": round(record.incidence_deg, 4),
                    "view_azimuth_deg": round(record.view_azimuth_deg, 3),
                    "resolution_m": [record.resolution_range_m, record.resolution_azimuth_m],
                    "looks": [record.looks_range, record.looks_azimuth],
                    "processor": record.processor,
                    "calibration": record.calibration,
                    "key": record.product_key,
                    "etag": record.product_etag,
                    "orthorectification": methods.get(record.collect_id),
                    "shift_m": [round(v, 3) for v in shifts.get(record.collect_id, (0, 0))],
                    "registration": (
                        "reference date"
                        if reg.get("reference")
                        else {
                            name: reg[name]
                            for name in (
                                "scale",
                                "n_tiles",
                                "n_inliers",
                                "inlier_fraction",
                                "spread_rms_m",
                            )
                            if name in reg
                        }
                        or None
                    ),
                    "level_offset": round(levels["offsets"][k], 4),
                    "reference": record.collect_id == selection.reference.collect_id,
                }
            )
        years = sorted({r.date[:4] for r in selection.chosen})
        base = reg_info.get("base") or {}
        coreg = {
            "mode": config.vhr_sar_coregister,
            "reference": selection.reference.collect_id,
            "base": base.get("kind"),
            "base_year": base.get("year"),
            "base_shift_m": (
                [
                    round(reg_info["base_registration"]["dx"], 3),
                    round(reg_info["base_registration"]["dy"], 3),
                ]
                if reg_info.get("base_registration")
                else None
            ),
        }
        tags: dict[str, Any] = {
            "AGRIBOUND_VERSION": __version__,
            "AGRIBOUND_SOURCE": provider.source,
            "AGRIBOUND_VALUE_SCALE": SOURCE_REGISTRY[provider.source]["value_scale"],
            "AGRIBOUND_YEAR": config.year,
            "AGRIBOUND_DATE_START": window[0],
            "AGRIBOUND_DATE_END_EXCLUSIVE": window[1],
            "AGRIBOUND_COMPOSITE_METHOD": "median",
            "AGRIBOUND_N_IMAGES": len(selection.chosen),
            "AGRIBOUND_IMAGE_YEARS": ",".join(years),
            "AGRIBOUND_COLLECTIONS": f"s3://{provider.bucket} ({provider.product})",
            "AGRIBOUND_EXPORT_CRS": grid.crs,
            "AGRIBOUND_RESOLUTION_M": float(config.vhr_sar_resolution_m),
            "AGRIBOUND_CACHE_KEY": key,
            "AGRIBOUND_RECIPE_VERSION": COMPOSITE_RECIPE_VERSION,
            "AGRIBOUND_SAR_RECIPE": VHR_SAR_RECIPE_TAG,
            "AGRIBOUND_ATTRIBUTION": provider.attribution,
            "AGRIBOUND_LICENSE": provider.license_name,
            "AGRIBOUND_LICENSE_URL": provider.license_url,
            "AGRIBOUND_DATA_CITATION": provider.citation(access_date),
            "AGRIBOUND_CHANGES": provider.changes,
            "AGRIBOUND_SAR_PRODUCT": provider.product,
            "AGRIBOUND_SAR_POLARIZATION": selection.group["polarization"],
            "AGRIBOUND_SAR_ORBIT_STATE": selection.group["orbit_state"],
            "AGRIBOUND_SAR_LOOK_SIDE": selection.group["look_side"],
            "AGRIBOUND_SAR_SELECTION": "pinned" if selection.pinned else "automatic",
            "AGRIBOUND_SAR_REFERENCE_COLLECT": selection.reference.collect_id,
            "AGRIBOUND_SAR_COLLECTS": json.dumps(collects, separators=(",", ":")),
            "AGRIBOUND_SAR_COREGISTRATION": json.dumps(coreg, separators=(",", ":")),
            "AGRIBOUND_SAR_BANDS": ",".join(names),
            "AGRIBOUND_INDEX_SNAPSHOT": index.snapshot,
        }
        if provider.source == "capella":
            tags["AGRIBOUND_DB_OFFSET"] = CAPELLA_DB_OFFSET
            tags["AGRIBOUND_SAR_CALIBRATION"] = ",".join(
                str(r.calibration) for r in selection.chosen
            )
        return tags

    @staticmethod
    def _write(
        path: Path, data: np.ndarray, grid: Any, names: list[str], tags: dict[str, Any]
    ) -> None:
        import rasterio

        part = path.with_name(path.name + ".part")
        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": int(data.shape[0]),
            "width": grid.width,
            "height": grid.height,
            "crs": grid.crs,
            "transform": grid.transform,
            "nodata": 0.0,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            "predictor": 3,
            "BIGTIFF": "IF_SAFER",
        }
        try:
            with rasterio.open(part, "w", **profile) as dst:
                dst.write(data.astype(np.float32, copy=False))
                for i, name in enumerate(names, start=1):
                    dst.set_band_description(i, name)
                dst.update_tags(**{k: str(v) for k, v in tags.items()})
            os.replace(part, path)
        finally:
            if part.exists():
                part.unlink()


__all__ = [
    "CAPELLA_DB_OFFSET",
    "DELINEATE_ANYTHING_PARAMS",
    "INCIDENCE_TOLERANCE_DEG",
    "MIN_FOOTPRINT_COVER",
    "PROVIDERS",
    "VHR_SAR_RECIPE_TAG",
    "VHR_SAR_RECIPE_VERSION",
    "VIEW_AZIMUTH_TOLERANCE_DEG",
    "ArchiveIndex",
    "BucketError",
    "CollectRecord",
    "ProductChangedError",
    "Provider",
    "PublicBucket",
    "S3Object",
    "Selection",
    "VHRSARCompositeBuilder",
    "azimuth_spread",
    "build_dem_hae",
    "cached_index",
    "capella_db",
    "check_product_key",
    "default_cache_dir",
    "ensure_product",
    "footprint_cover",
    "get_provider",
    "level_normalise",
    "load_index",
    "median_composite",
    "parse_capella_extended",
    "parse_capella_item",
    "parse_umbra_stac",
    "refresh_index",
    "register_adaptive",
    "register_tile",
    "select_collects",
    "warp_product",
]
