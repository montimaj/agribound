"""
State of the TESSERA Zarr store under a read: the objects it reads and their ETags.

The TESSERA stores are rewritten in place: the v1 store was converted again
upstream on 2026-10-06/08 (geotessera issue #429) under the same URL, and its
metadata documents (``zarr.json``) kept their contents and dates; only the shard
objects changed. A cache key made of the configuration, the store URL and the
variant cannot see such a rewrite, so agribound records the state of the objects
a TESSERA raster was read from (a *store stamp*):

- :func:`read_object_keys` lists the objects that geotessera's
  ``GeoTesseraZarr.read_region`` reads for an EPSG:4326 box and a year. The box
  is split by UTM zone as :func:`agribound.composites.local._read_tessera_zones`
  splits it; each part goes to the zone that holds its centre (geotessera's
  routing) and selects the pixels whose centres lie in the zone-CRS envelope of
  its four corners. The pixel window is computed from the zone's
  ``spatial:transform`` and ``spatial:shape`` and padded by one pixel on each
  side (so that a rounding difference at a window edge cannot drop a shard);
  the year's index comes from the zone's ``time`` array; the shards of the
  ``embeddings`` and ``scales`` arrays that hold the window, and the chunks of
  the ``x`` and ``y`` coordinate arrays that hold it (geotessera selects the
  window and builds the returned transform from their values), follow from
  each array's own ``zarr.json`` (shape, regular chunk grid, chunk key encoding
  and dimension names). Nothing is hard-coded about the shard size or key
  layout.
- :func:`compute_stamp` gets each object's ETag and Last-Modified with HTTP
  HEAD and returns a :class:`StoreStamp`: the SHA-256 of the zones' grid
  attributes (``proj:code`` and ``spatial:transform``) and of the sorted
  ``"<key> <ETag>"`` lines (a missing object, a missing zone or a missing year
  counts as ``absent``), and the newest Last-Modified.

:func:`store_stamp` is the entry point used by the TESSERA builder
(:func:`agribound.composites.local.build_tessera_embedding`), the output reuse
check (:func:`agribound.provenance.reuse_mismatch`) and the HPC composite stage
and tile status. The stamp is written to the raster's tags
(``TESSERA_STORE_FINGERPRINT``, ``TESSERA_STORE_LAST_MODIFIED``,
``TESSERA_STORE_OBJECTS``) and, through the composite tags, into the provenance
record. When the stamp cannot be computed (no network, an HTTP error, a store
layout this module does not know, or the check turned off with
``AGRIBOUND_TESSERA_STORE_CHECK=0``), :class:`StoreCheckError` is raised and the
callers fall back on what they have locally.

geotessera's own cache (``GeoTesseraZarr(cache_dir=...)``) keeps every object it
reads whole on disk, the shards included, and never checks them against the
store again (zarr's ``CacheStore`` with ``max_age_seconds="infinity"``).
agribound caches the rasters it builds, so it reads through
:func:`geotessera_store` instead, which keeps only the metadata documents and the
chunks of the coordinate arrays on disk (``config.embedding_cache_dir``, in the
directory geotessera uses). :func:`refresh_local_cache` deletes those copies when
their ETag differs from the one recorded when they were cached (or none is
recorded), and :func:`record_local_cache` records the ETags after the read, in
``<cache_dir>/<store>.agribound-etags.json``.

Requests use :mod:`urllib` with a ``User-Agent`` of ``agribound/<version>``
(the Source Cooperative server answers HTTP 403 to Python's default one), a
timeout of :data:`HTTP_TIMEOUT_S` seconds, and up to :data:`HTTP_RETRIES`
retries after HTTP 429 or 5xx, a read timeout or a dropped connection, waiting
1, 2 and 4 s plus up to 25 % at random. A request whose connection cannot be
made (failed name lookup, refused connection, connection timeout, unreachable
network or host, connection not permitted, certificate not verified) fails at
once. A redirected HEAD request stays a HEAD request. HEAD requests run
:data:`HTTP_WORKERS` at a time. Answers are reused within a process for
:data:`MEMO_TTL_S` seconds (the stamp computed right before a read asks the
store again). A request that failed is not repeated for :data:`FAILURE_TTL_S`
seconds, and a store whose connection could not be made is not asked again for
that long, so that checking many tiles without network costs one failure.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import email.utils
import errno
import functools
import hashlib
import http.client
import itertools
import json
import logging
import math
import os
import random
import socket
import ssl
import threading
import time
from collections.abc import Iterable, Sequence
from datetime import UTC
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

logger = logging.getLogger(__name__)

#: Version of the stamp recipe (the objects covered and the fingerprint text). It is
#: the first line of the fingerprinted text, so changing the recipe changes every
#: fingerprint and every TESSERA raster and output is checked against the new one.
STAMP_VERSION = 1
#: Tags written to a TESSERA raster (and copied into ``facts["composite"]``).
TAG_FINGERPRINT = "TESSERA_STORE_FINGERPRINT"
TAG_LAST_MODIFIED = "TESSERA_STORE_LAST_MODIFIED"
TAG_OBJECTS = "TESSERA_STORE_OBJECTS"
STAMP_TAGS = (TAG_FINGERPRINT, TAG_LAST_MODIFIED, TAG_OBJECTS)

#: Environment variable that turns the check off (``0``, ``false``, ``no`` or ``off``).
CHECK_ENV = "AGRIBOUND_TESSERA_STORE_CHECK"
#: Seconds allowed per HTTP request.
HTTP_TIMEOUT_S = 15.0
#: Retries of a request after HTTP 429 or 5xx, a read timeout or a dropped connection.
HTTP_RETRIES = 3
#: Wait before the first retry (s); it doubles with each retry up to RETRY_MAX_WAIT_S.
RETRY_BASE_WAIT_S = 1.0
RETRY_MAX_WAIT_S = 8.0
#: Random extra wait, as a share of the wait.
RETRY_JITTER = 0.25
#: HEAD requests in flight.
HTTP_WORKERS = 8
#: Seconds for which metadata documents and HEAD answers are reused in a process.
MEMO_TTL_S = 60.0
#: Seconds for which a failed request is not repeated, and a store whose connection
#: could not be made is not asked again (later checks fail at once).
FAILURE_TTL_S = 60.0

#: Arrays that ``read_region`` reads for its pixel window.
READ_ARRAYS = ("embeddings", "scales")
#: Coordinate arrays that geotessera loads whole when it opens a zone.
COORDINATE_ARRAYS = ("time", "band", "x", "y")
#: Coordinate arrays whose values place the window (dimension name = array name).
WINDOW_COORDINATES = ("x", "y")
#: Pixels added on each side of the computed window.
WINDOW_PAD_PX = 1
#: Largest number of chunks read from a small array (the ``time`` coordinate).
_MAX_SMALL_ARRAY_CHUNKS = 64

#: ``OSError.errno`` values of a connection that cannot be made: no route to the
#: network or host, the network or host is down, no local address, or the
#: connection is refused or not permitted (a local firewall).
_NO_CONNECTION_ERRNOS = frozenset(
    code
    for code in (
        getattr(errno, name, None)
        for name in (
            "ENETUNREACH",
            "EHOSTUNREACH",
            "ENETDOWN",
            "EHOSTDOWN",
            "EADDRNOTAVAIL",
            "ECONNREFUSED",
            "EACCES",
            "EPERM",
        )
    )
    if code is not None
)

_LOCK = threading.Lock()
# (method, store URL, key) -> (monotonic time, answer)
_MEMO: dict[tuple[str, str, str], tuple[float, Any]] = {}
# (method, store URL, key) -> (monotonic time, message, connection failure)
_REQUEST_FAILURES: dict[tuple[str, str, str], tuple[float, str, bool]] = {}
# store URL -> (monotonic time, message): stores whose connection could not be made
_FAILURES: dict[str, tuple[float, str]] = {}


class StoreCheckError(RuntimeError):
    """The state of the TESSERA store could not be determined.

    Raised without network access, after an HTTP error, for a store layout
    this module does not know, and when the check is turned off
    (:data:`CHECK_ENV`). ``unreachable`` is True when the connection to the
    store could not be made (no network); such a failure is remembered for
    the whole store (:data:`FAILURE_TTL_S`).
    """

    def __init__(self, message: str, *, unreachable: bool = False) -> None:
        super().__init__(message)
        self.unreachable = bool(unreachable)


@dataclasses.dataclass(frozen=True)
class ObjectState:
    """An object of the store: its key and, when it exists, its ETag and Last-Modified.

    ``etag`` is *None* for an object that does not exist (HTTP 404) and for
    the markers of a missing zone or year. ``last_modified`` is ISO 8601 UTC
    (``YYYY-MM-DDTHH:MM:SSZ``) or *None*.
    """

    key: str
    etag: str | None
    last_modified: str | None = None

    @property
    def exists(self) -> bool:
        return self.etag is not None


@dataclasses.dataclass(frozen=True)
class StoreStamp:
    """State of the objects a TESSERA read of *bbox* and *year* depends on.

    Attributes
    ----------
    store_url : str
        Zarr store URL (without a trailing slash).
    bbox : tuple
        The EPSG:4326 box the stamp was computed for.
    year : int
        Year read.
    fingerprint : str
        SHA-256 (hex) of the text ``"agribound-tessera-store-stamp <version>"``,
        the store URL, one line per zone read with its grid attributes
        (:attr:`grids`) and one ``"<key> <ETag>"`` line per object, sorted by
        key (``absent`` for a missing object).
    last_modified : str or None
        Newest Last-Modified of the objects (ISO 8601 UTC), *None* when no
        object exists.
    objects : tuple of ObjectState
        The objects (shards of the read arrays and chunks of the ``x`` and
        ``y`` coordinate arrays under the window), sorted by key.
    zones : tuple of int
        UTM zones whose groups were read.
    grids : tuple of str
        ``"utm<NN> <proj:code> <spatial:transform>"`` of each zone read.
    """

    store_url: str
    bbox: tuple[float, float, float, float]
    year: int
    fingerprint: str
    last_modified: str | None
    objects: tuple[ObjectState, ...]
    zones: tuple[int, ...] = ()
    grids: tuple[str, ...] = ()

    @property
    def n_objects(self) -> int:
        """Number of objects that exist."""
        return sum(1 for o in self.objects if o.exists)

    def tags(self) -> dict[str, str]:
        """GeoTIFF tags of the stamp."""
        return {
            TAG_FINGERPRINT: self.fingerprint,
            TAG_LAST_MODIFIED: self.last_modified or "",
            TAG_OBJECTS: str(self.n_objects),
        }

    def describe(self) -> str:
        """Short description for log messages: fingerprint prefix, objects, newest date."""
        newest = self.last_modified or "none"
        return (
            f"fingerprint {self.fingerprint[:12]}, {self.n_objects} objects, newest "
            f"modified {newest}"
        )


# ---------------------------------------------------------------------------
# Switch and memo
# ---------------------------------------------------------------------------


def check_disabled_reason() -> str | None:
    """Why the check is turned off (:data:`CHECK_ENV`), or *None* when it is on."""
    value = os.environ.get(CHECK_ENV)
    if value is not None and value.strip().lower() in {"0", "false", "no", "off"}:
        return f"the TESSERA store check is turned off ({CHECK_ENV}={value})"
    return None


def clear_memo() -> None:
    """Forget the answers and failures remembered in this process (mainly for tests)."""
    with _LOCK:
        _MEMO.clear()
        _REQUEST_FAILURES.clear()
        _FAILURES.clear()


def _memo_get(key: tuple[str, str, str]) -> tuple[bool, Any]:
    with _LOCK:
        hit = _MEMO.get(key)
    if hit is None or time.monotonic() - hit[0] > MEMO_TTL_S:
        return False, None
    return True, hit[1]


def _memo_put(key: tuple[str, str, str], value: Any) -> None:
    with _LOCK:
        _MEMO[key] = (time.monotonic(), value)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _user_agent() -> str:
    from agribound._version import __version__

    return f"agribound/{__version__}"


def _connection_failure(exc: BaseException) -> bool:
    """Whether a failed request shows that no connection to the store can be made.

    urllib raises :class:`URLError` when the connection cannot be opened: the
    name lookup failed, the connection was refused, not permitted or timed out,
    the network or host is unreachable or down, or the server's certificate
    could not be verified. These fail every request to the store alike.
    """
    if isinstance(exc, HTTPError) or not isinstance(exc, URLError):
        return False
    reason = exc.reason
    if isinstance(
        reason,
        socket.gaierror
        | ConnectionRefusedError
        | TimeoutError
        | PermissionError
        | ssl.SSLCertVerificationError,
    ):
        return True
    return isinstance(reason, OSError) and getattr(reason, "errno", None) in _NO_CONNECTION_ERRNOS


def _transient(exc: BaseException) -> bool:
    """Whether a failed request may succeed when repeated (see the module docstring)."""
    if isinstance(exc, HTTPError):  # before URLError and OSError: it is both
        return exc.code == 429 or exc.code >= 500
    if _connection_failure(exc):  # repeating it would only wait longer
        return False
    if isinstance(exc, URLError):
        return isinstance(exc.reason, OSError | http.client.HTTPException)
    return isinstance(exc, TimeoutError | ConnectionError | http.client.HTTPException)


def _describe(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, URLError):
        return f"{type(exc.reason).__name__}: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"


def _retry_wait(retry: int) -> float:
    """Seconds to wait before retry number *retry* (1, 2, ...), jitter included."""
    wait_s = min(RETRY_MAX_WAIT_S, RETRY_BASE_WAIT_S * 2 ** (retry - 1))
    return wait_s * (1 + random.uniform(0, RETRY_JITTER))


def _iso_utc(value: str | None) -> str | None:
    """An HTTP date (``Wed, 07 Oct 2026 12:40:56 GMT``) as ``2026-10-07T12:40:56Z``."""
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _etag(headers: dict[str, str]) -> str:
    """The ETag without quotes or weak prefix; Last-Modified and length when there is none."""
    etag = (headers.get("etag") or "").strip()
    if etag.startswith("W/"):
        etag = etag[2:]
    etag = etag.strip('"')
    if etag:
        return etag
    return f"no-etag;{headers.get('last-modified', '')};{headers.get('content-length', '')}"


class _SameMethodRedirectHandler(HTTPRedirectHandler):
    """Follow a redirect of a HEAD request with HEAD.

    urllib's handler builds the redirected request without a method before
    Python 3.13, so a redirected HEAD request was sent as GET, which opens a
    download of the whole object (an embeddings shard is up to 1.7 GB).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and req.get_method() == "HEAD":
            new.method = "HEAD"
        return new


@functools.cache
def _opener():
    return build_opener(_SameMethodRedirectHandler)


def urlopen(request: Request, timeout: float):
    """Open *request* (redirects keep a HEAD request a HEAD request)."""
    return _opener().open(request, timeout=timeout)


class StoreClient:
    """Anonymous HTTPS reads of a Zarr store's objects (HEAD and small GETs).

    Parameters
    ----------
    store_url : str
        Store URL, e.g. ``https://data.source.coop/tessera/tessera/zarr/v1``.
    timeout_s, retries, workers : float, int, int or None
        Per-request timeout, retries and HEAD requests in flight (defaults:
        :data:`HTTP_TIMEOUT_S`, :data:`HTTP_RETRIES`, :data:`HTTP_WORKERS`).
    fresh : bool
        Ask the store again instead of using the answers remembered in this
        process (each answer is still asked once per client, and remembered
        for later clients).
    """

    def __init__(
        self,
        store_url: str,
        *,
        timeout_s: float | None = None,
        retries: int | None = None,
        workers: int | None = None,
        fresh: bool = False,
    ) -> None:
        self.store_url = str(store_url).rstrip("/")
        self.timeout_s = float(HTTP_TIMEOUT_S if timeout_s is None else timeout_s)
        self.retries = int(HTTP_RETRIES if retries is None else retries)
        self.workers = max(1, int(HTTP_WORKERS if workers is None else workers))
        self.fresh = bool(fresh)
        self._answers: dict[tuple[str, str, str], Any] = {}
        self._answers_lock = threading.Lock()

    def url(self, key: str) -> str:
        """URL of the object *key*."""
        return f"{self.store_url}/{quote(key, safe='/')}"

    def _request(self, method: str, key: str) -> tuple[int, dict[str, str], bytes]:
        """Send one request; return ``(status, lower-case headers, body)``; 404 is returned."""
        url = self.url(key)
        request = Request(url, method=method, headers={"User-Agent": _user_agent()})
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    body = response.read() if method == "GET" else b""
                    headers = {k.lower(): v for k, v in response.headers.items()}
                    return int(response.status), headers, body
            except HTTPError as exc:
                if exc.code == 404:
                    return 404, {}, b""
                error: BaseException = exc
            except (OSError, http.client.HTTPException) as exc:
                error = exc
            if attempt >= self.retries or not _transient(error):
                raise StoreCheckError(
                    f"{method} {url} failed: {_describe(error)}",
                    unreachable=_connection_failure(error),
                ) from error
            wait_s = _retry_wait(attempt + 1)
            logger.debug(
                "TESSERA store %s %s failed (%s); retrying in %.1f s [%d/%d]",
                method,
                url,
                _describe(error),
                wait_s,
                attempt + 1,
                self.retries,
            )
            time.sleep(wait_s)
        raise AssertionError("unreachable")  # pragma: no cover - the last attempt returns or raises

    def _send(self, method: str, key: str) -> tuple[int, dict[str, str], bytes]:
        """:meth:`_request`, unless the same request failed within :data:`FAILURE_TTL_S`."""
        failure_key = (method, self.store_url, key)
        with _LOCK:
            failed = _REQUEST_FAILURES.get(failure_key)
        if failed is not None:
            age = time.monotonic() - failed[0]
            if age < FAILURE_TTL_S:
                raise StoreCheckError(
                    f"{failed[1]} ({age:.0f} s ago; not repeated within {FAILURE_TTL_S:.0f} s)",
                    unreachable=failed[2],
                )
        try:
            return self._request(method, key)
        except StoreCheckError as exc:
            with _LOCK:
                _REQUEST_FAILURES[failure_key] = (time.monotonic(), str(exc), exc.unreachable)
            raise

    def _remembered(self, memo_key: tuple[str, str, str]) -> tuple[bool, Any]:
        if self.fresh:
            with self._answers_lock:
                if memo_key in self._answers:
                    return True, self._answers[memo_key]
            return False, None
        return _memo_get(memo_key)

    def _remember(self, memo_key: tuple[str, str, str], value: Any) -> None:
        with self._answers_lock:
            self._answers[memo_key] = value
        _memo_put(memo_key, value)

    def get(self, key: str) -> bytes | None:
        """Body of the object *key*, or *None* when it does not exist."""
        memo_key = ("GET", self.store_url, key)
        found, value = self._remembered(memo_key)
        if found:
            return value
        status, _headers, body = self._send("GET", key)
        value = None if status == 404 else body
        self._remember(memo_key, value)
        return value

    def get_json(self, key: str) -> dict[str, Any] | None:
        """The JSON object stored at *key*, or *None* when it does not exist."""
        body = self.get(key)
        if body is None:
            return None
        try:
            value = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StoreCheckError(f"{self.url(key)} is not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise StoreCheckError(f"{self.url(key)} is not a JSON object")
        return value

    def head(self, key: str) -> ObjectState:
        """ETag and Last-Modified of the object *key* (``etag=None`` when it does not exist)."""
        memo_key = ("HEAD", self.store_url, key)
        found, value = self._remembered(memo_key)
        if found:
            return value
        status, headers, _body = self._send("HEAD", key)
        if status == 404:
            state = ObjectState(key, None, None)
        else:
            state = ObjectState(key, _etag(headers), _iso_utc(headers.get("last-modified")))
        self._remember(memo_key, state)
        return state

    def head_many(self, keys: Iterable[str]) -> dict[str, ObjectState]:
        """:meth:`head` of every key, :attr:`workers` at a time."""
        unique = sorted(set(keys))
        if not unique:
            return {}
        workers = min(self.workers, len(unique))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(zip(unique, pool.map(self.head, unique), strict=True))


# ---------------------------------------------------------------------------
# Object keys of a read
# ---------------------------------------------------------------------------


def zone_for_lon(lon: float) -> int:
    """UTM zone (1-60) holding *lon*, as geotessera routes a read (``_zone_for_lon``)."""
    return max(1, min(60, int(math.floor((float(lon) + 180.0) / 6.0)) + 1))


def zone_parts(
    bbox: Sequence[float],
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """``(zone, sub_bbox)`` for each UTM-zone part of an EPSG:4326 *bbox*.

    The parts are those of :func:`agribound.composites.local._read_tessera_zones`
    (:func:`agribound.io.crs.utm_zones_for_bounds` and
    :func:`agribound.composites.local.zone_sub_bbox`); *zone* is the zone
    geotessera reads each part from (the one holding its centre).
    """
    from agribound.composites.local import zone_sub_bbox
    from agribound.io.crs import utm_zones_for_bounds

    bounds = tuple(float(v) for v in bbox)
    parts = []
    for zone in utm_zones_for_bounds(bounds):
        sub = zone_sub_bbox(bounds, zone)
        if sub is None:
            continue
        parts.append((zone_for_lon((sub[0] + sub[2]) / 2.0), sub))
    return parts


def pixel_window(
    attrs: dict[str, Any], sub_bbox: Sequence[float], pad: int = WINDOW_PAD_PX
) -> tuple[int, int, int, int] | None:
    """Pixel window ``(col0, col1, row0, row1)`` (inclusive) a zone read selects.

    geotessera projects the four corners of *sub_bbox* to the zone's CRS
    (``proj:code``) and selects the pixels whose centres lie inside the
    envelope of the projected corners. The pixel centres follow from the
    zone's ``spatial:transform`` (``spatial:registration`` ``"pixel"``: centre
    at half a pixel from the corner). The window is padded by *pad* pixels
    and clipped to ``spatial:shape``.

    Parameters
    ----------
    attrs : dict
        Attributes of the zone group (``utm<NN>/zarr.json``).
    sub_bbox : sequence of float
        ``(min_lon, min_lat, max_lon, max_lat)`` inside the zone.
    pad : int
        Pixels added on each side.

    Returns
    -------
    tuple or None
        The window, or *None* when the envelope misses the grid.

    Raises
    ------
    StoreCheckError
        If the attributes are missing or describe a rotated or flipped grid.
    """
    import pyproj

    try:
        crs = str(attrs["proj:code"])
        a, b, x0, d, e, y0 = (float(v) for v in list(attrs["spatial:transform"])[:6])
        height, width = (int(v) for v in list(attrs["spatial:shape"])[:2])
    except (KeyError, TypeError, ValueError) as exc:
        raise StoreCheckError(f"zone attributes without a usable grid: {exc!r}") from exc
    if b != 0.0 or d != 0.0 or a <= 0.0 or e >= 0.0:
        raise StoreCheckError(f"unsupported spatial:transform {attrs['spatial:transform']!r}")
    offset = 0.5 if str(attrs.get("spatial:registration", "pixel")) == "pixel" else 0.0
    to_zone = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = [], []
    for lon in (sub_bbox[0], sub_bbox[2]):
        for lat in (sub_bbox[1], sub_bbox[3]):
            x, y = to_zone.transform(float(lon), float(lat))
            xs.append(x)
            ys.append(y)
    if not all(math.isfinite(v) for v in xs + ys):
        raise StoreCheckError(f"cannot project {tuple(sub_bbox)} to {crs}")
    col0 = math.ceil((min(xs) - x0) / a - offset) - pad
    col1 = math.floor((max(xs) - x0) / a - offset) + pad
    row0 = math.ceil((y0 - max(ys)) / -e - offset) - pad
    row1 = math.floor((y0 - min(ys)) / -e - offset) + pad
    col0, col1 = max(0, col0), min(width - 1, col1)
    row0, row1 = max(0, row0), min(height - 1, row1)
    if col0 > col1 or row0 > row1:
        return None
    return col0, col1, row0, row1


def chunk_key(encoding: Any, coords: Sequence[int]) -> str:
    """Key of the chunk (shard) at grid position *coords*, from a v3 chunk key encoding.

    ``default`` encoding: ``c`` and the indices joined by the separator
    (default ``/``); ``v2``: the indices joined by the separator (default
    ``.``), ``0`` for a 0-d array.
    """
    if isinstance(encoding, dict):
        name = str(encoding.get("name", "default"))
        config = encoding.get("configuration") or {}
    else:
        name, config = str(encoding or "default"), {}
    indices = [str(int(i)) for i in coords]
    if name == "default":
        separator = str(config.get("separator", "/"))
        return separator.join(["c", *indices])
    if name == "v2":
        separator = str(config.get("separator", "."))
        return separator.join(indices) if indices else "0"
    raise StoreCheckError(f"unknown chunk key encoding {name!r}")


def chunk_keys(meta: dict[str, Any], selection: dict[str, tuple[int, int]]) -> list[str]:
    """Keys (relative to the array) of the chunks holding *selection*.

    Parameters
    ----------
    meta : dict
        The array's ``zarr.json`` (Zarr v3): ``shape``, a ``regular`` chunk grid
        (for a sharded array its chunks are the shards, the stored objects),
        ``chunk_key_encoding`` and ``dimension_names``.
    selection : dict
        Dimension name -> inclusive index range; other dimensions are read whole.

    Raises
    ------
    StoreCheckError
        For another chunk grid, a dimension without a name, or a selection
        outside the array.
    """
    try:
        shape = [int(v) for v in meta["shape"]]
        grid = meta["chunk_grid"]
        if str(grid.get("name")) != "regular":
            raise StoreCheckError(f"unsupported chunk grid {grid.get('name')!r}")
        chunk_shape = [int(v) for v in grid["configuration"]["chunk_shape"]]
        dims = meta.get("dimension_names") or [None] * len(shape)
        encoding = meta.get("chunk_key_encoding", {"name": "default"})
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise StoreCheckError(f"array metadata without a usable chunk grid: {exc!r}") from exc
    if not (len(shape) == len(chunk_shape) == len(dims)):
        raise StoreCheckError("array metadata with inconsistent shape, chunks and dimensions")
    unknown = set(selection) - {d for d in dims if d is not None}
    if unknown:
        raise StoreCheckError(f"array has no dimension {sorted(unknown)} (dimensions {dims})")
    ranges = []
    for size, step, dim in zip(shape, chunk_shape, dims, strict=True):
        if step <= 0:
            raise StoreCheckError(f"chunk shape {chunk_shape} is not positive")
        lo, hi = selection.get(dim, (0, size - 1)) if dim is not None else (0, size - 1)
        if size == 0 or lo < 0 or hi >= size or lo > hi:
            raise StoreCheckError(f"selection {lo}..{hi} outside dimension {dim!r} of size {size}")
        ranges.append(range(lo // step, hi // step + 1))
    return [chunk_key(encoding, coords) for coords in itertools.product(*ranges)]


def read_small_array(client: StoreClient, path: str, meta: dict[str, Any]) -> Any:
    """Read a small array (e.g. ``utm<NN>/time``) whole and decode it with zarr.

    The array's ``zarr.json`` and its chunks are put into an in-memory store,
    so zarr applies the array's own codecs.
    """
    import numpy as np

    try:
        import zarr
        from zarr.core.buffer import default_buffer_prototype
        from zarr.storage import MemoryStore
    except ImportError as exc:  # geotessera depends on zarr >= 3
        raise StoreCheckError(f"zarr >= 3 is needed to read {path}: {exc}") from exc
    keys = chunk_keys(meta, {})
    if len(keys) > _MAX_SMALL_ARRAY_CHUNKS:
        raise StoreCheckError(f"{path} has {len(keys)} chunks; expected a small array")
    buffer = default_buffer_prototype().buffer
    items = {"zarr.json": buffer.from_bytes(json.dumps(meta).encode("utf-8"))}
    for key in keys:
        data = client.get(f"{path}/{key}")
        if data is not None:
            items[key] = buffer.from_bytes(data)
    try:
        array = zarr.open_array(store=MemoryStore(store_dict=items), mode="r")
        return np.asarray(array[...])
    except Exception as exc:  # malformed metadata or chunk: report, do not crash the caller
        raise StoreCheckError(f"cannot decode {client.url(path)}: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class ZoneMetadata:
    """What a read needs to know about a zone group: attributes, arrays and years."""

    zone: int
    attrs: dict[str, Any]
    arrays: dict[str, dict[str, Any]]
    times: tuple[int, ...]

    def grid(self) -> str:
        """``"utm<NN> <proj:code> <spatial:transform>"``: the zone's grid in the fingerprint."""
        try:
            code = str(self.attrs["proj:code"])
            transform = ",".join(repr(float(v)) for v in list(self.attrs["spatial:transform"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreCheckError(f"zone attributes without a usable grid: {exc!r}") from exc
        return f"utm{self.zone:02d} {code} {transform}"


def zone_metadata(client: StoreClient, zone: int) -> ZoneMetadata | None:
    """Metadata of ``utm<NN>`` (*None* when the store has no such zone).

    Reads ``utm<NN>/zarr.json``, the ``zarr.json`` of :data:`READ_ARRAYS`, of
    ``time`` and of :data:`WINDOW_COORDINATES`, and the ``time`` values.
    """
    prefix = f"utm{int(zone):02d}"
    group = client.get_json(f"{prefix}/zarr.json")
    if group is None:
        return None
    arrays: dict[str, dict[str, Any]] = {}
    for name in (*READ_ARRAYS, "time", *WINDOW_COORDINATES):
        meta = client.get_json(f"{prefix}/{name}/zarr.json")
        if meta is None:
            raise StoreCheckError(f"{client.url(prefix)} has no array {name!r}")
        arrays[name] = meta
    times = tuple(int(v) for v in read_small_array(client, f"{prefix}/time", arrays["time"]))
    return ZoneMetadata(int(zone), dict(group.get("attributes") or {}), arrays, times)


@dataclasses.dataclass(frozen=True)
class _ReadObjects:
    keys: list[str]
    markers: list[ObjectState]
    zones: list[int]
    grids: list[str]


def _read_objects(client: StoreClient, bbox: Sequence[float], year: int) -> _ReadObjects:
    keys: list[str] = []
    markers: list[ObjectState] = []
    zones: list[int] = []
    grids: dict[int, str] = {}
    for zone, sub in zone_parts(bbox):
        prefix = f"utm{zone:02d}"
        meta = zone_metadata(client, zone)
        if meta is None:
            markers.append(ObjectState(f"{prefix}/zarr.json", None))
            continue
        if int(year) not in meta.times:
            markers.append(ObjectState(f"{prefix}/time[{int(year)}]", None))
            continue
        window = pixel_window(meta.attrs, sub)
        if window is None:
            continue
        col0, col1, row0, row1 = window
        selection = {
            "time": (meta.times.index(int(year)),) * 2,
            "y": (row0, row1),
            "x": (col0, col1),
        }
        for name in READ_ARRAYS:
            keys += [f"{prefix}/{name}/{k}" for k in chunk_keys(meta.arrays[name], selection)]
        for name in WINDOW_COORDINATES:
            window_range = {name: selection[name]}
            keys += [f"{prefix}/{name}/{k}" for k in chunk_keys(meta.arrays[name], window_range)]
        grids[zone] = meta.grid()
        if zone not in zones:
            zones.append(zone)
    return _ReadObjects(sorted(set(keys)), markers, zones, [grids[z] for z in sorted(grids)])


def read_object_keys(
    client: StoreClient, bbox: Sequence[float], year: int
) -> tuple[list[str], list[ObjectState], list[int]]:
    """Objects a TESSERA read of *bbox* and *year* depends on.

    Returns
    -------
    tuple
        ``(keys, markers, zones)``: the shard keys of :data:`READ_ARRAYS` and
        the chunk keys of :data:`WINDOW_COORDINATES` that hold the pixel
        window of each zone part; :class:`ObjectState` markers (``etag=None``)
        for a zone the store does not have (``utm<NN>/zarr.json``) and for a
        zone without *year* (``utm<NN>/time[<year>]``); and the zones read.
    """
    found = _read_objects(client, bbox, year)
    return found.keys, found.markers, found.zones


def _fingerprint(store_url: str, objects: Sequence[ObjectState], grids: Sequence[str] = ()) -> str:
    lines = [f"agribound-tessera-store-stamp {STAMP_VERSION}", store_url, *grids]
    lines += [f"{o.key} {o.etag if o.etag is not None else 'absent'}" for o in objects]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def compute_stamp(
    store_url: str,
    bbox: Sequence[float],
    year: int,
    *,
    client: StoreClient | None = None,
    fresh: bool = False,
) -> StoreStamp:
    """Compute the :class:`StoreStamp` of a read (no switch, no store failure memo).

    *fresh* (when no *client* is given): ask the store again instead of using
    the answers remembered in this process.

    Raises
    ------
    StoreCheckError
        If a request fails or the store's metadata cannot be used.
    """
    store_url = str(store_url).rstrip("/")
    client = client or StoreClient(store_url, fresh=fresh)
    try:
        found = _read_objects(client, bbox, int(year))
        states = client.head_many(found.keys)
    except StoreCheckError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise StoreCheckError(f"unexpected metadata in {store_url}: {exc!r}") from exc
    objects = tuple(sorted([*states.values(), *found.markers], key=lambda o: o.key))
    dates = [o.last_modified for o in objects if o.exists and o.last_modified]
    return StoreStamp(
        store_url=store_url,
        bbox=tuple(float(v) for v in bbox),
        year=int(year),
        fingerprint=_fingerprint(store_url, objects, found.grids),
        last_modified=max(dates) if dates else None,
        objects=objects,
        zones=tuple(found.zones),
        grids=tuple(found.grids),
    )


def store_stamp(
    store_url: str,
    bbox: Sequence[float],
    year: int,
    *,
    client: StoreClient | None = None,
    fresh: bool = False,
) -> StoreStamp:
    """The :class:`StoreStamp` of a TESSERA read of *bbox* (EPSG:4326) and *year*.

    Raises :class:`StoreCheckError` at once when the check is turned off
    (:data:`CHECK_ENV`), or when the connection to the store could not be
    made within the last :data:`FAILURE_TTL_S` seconds. Other failures (an
    HTTP error, unusable metadata) are not carried over to other reads; a
    request that failed is not repeated within :data:`FAILURE_TTL_S`
    (:class:`StoreClient`). *fresh*: see :func:`compute_stamp`.

    Raises
    ------
    StoreCheckError
        If the stamp cannot be computed.
    """
    reason = check_disabled_reason()
    if reason:
        raise StoreCheckError(reason)
    store_url = str(store_url).rstrip("/")
    with _LOCK:
        failure = _FAILURES.get(store_url)
    if failure is not None:
        age = time.monotonic() - failure[0]
        if age < FAILURE_TTL_S:
            raise StoreCheckError(
                f"{failure[1]} (no connection to the store {age:.0f} s ago)", unreachable=True
            )
    try:
        return compute_stamp(store_url, bbox, year, client=client, fresh=fresh)
    except StoreCheckError as exc:
        if exc.unreachable:
            with _LOCK:
                _FAILURES[store_url] = (time.monotonic(), str(exc))
        raise


# ---------------------------------------------------------------------------
# geotessera's on-disk cache
# ---------------------------------------------------------------------------


def _store_cache_key(store_url: str) -> str:
    """geotessera's subdirectory name for *store_url* (``geotessera.store._store_cache_key``)."""
    try:
        from geotessera.store import _store_cache_key as geotessera_key
    except ImportError as exc:
        raise StoreCheckError(f"geotessera's cache layout is unknown: {exc}") from exc
    return str(geotessera_key(str(store_url).rstrip("/")))


def local_cache_root(cache_dir: str | os.PathLike[str], store_url: str) -> Path:
    """Directory where geotessera caches the objects of *store_url* under *cache_dir*.

    The path is built as geotessera builds it (``Path(cache_dir) / key``,
    without expanding ``~``), so it names the same directory.
    """
    return Path(cache_dir) / _store_cache_key(store_url)


def disk_cached_key(key: str) -> bool:
    """Whether :func:`geotessera_store` keeps the object *key* on disk.

    Metadata documents (``zarr.json``) and the chunks of the coordinate arrays
    (:data:`COORDINATE_ARRAYS`) are kept; the chunks (shards) of the data
    arrays are not.
    """
    parts = key.split("/")
    return parts[-1] == "zarr.json" or (len(parts) >= 3 and parts[1] in COORDINATE_ARRAYS)


@functools.cache
def _metadata_cache_store_type() -> type:
    from zarr.experimental.cache_store import CacheStore

    class MetadataCacheStore(CacheStore):
        """zarr's ``CacheStore`` that caches only the keys of :func:`disk_cached_key`."""

        async def get(self, key, prototype, byte_range=None):
            if disk_cached_key(key):
                return await super().get(key, prototype, byte_range)
            return await self._store.get(key, prototype, byte_range)

    return MetadataCacheStore


def geotessera_store(store_url: str, cache_dir: str | os.PathLike[str]) -> Any:
    """The Zarr store agribound reads *store_url* through, cached under *cache_dir*.

    geotessera's own store (:func:`geotessera.store.zarr_store`, the same HTTP
    settings) wrapped in zarr's ``CacheStore`` as geotessera wraps it for
    ``GeoTesseraZarr(cache_dir=...)``, in the same directory
    (:func:`local_cache_root`), except that only the metadata documents and
    the coordinate-array chunks are kept on disk (:func:`disk_cached_key`).
    Shards of the ``embeddings`` and ``scales`` arrays (up to 1.7 GB each) are
    read from the store every time and not kept, on disk or in memory:
    agribound caches the rasters it builds, and a shard on disk could not be
    tied to the version of the store it came from when several processes
    share the directory. Pass the result to ``GeoTesseraZarr``.
    """
    from geotessera.store import zarr_store
    from zarr.storage import LocalStore

    root = local_cache_root(cache_dir, store_url)
    root.mkdir(parents=True, exist_ok=True)
    source = zarr_store(str(store_url).rstrip("/"))
    return _metadata_cache_store_type()(source, cache_store=LocalStore(root))


def _index_path(cache_dir: str | os.PathLike[str], store_url: str) -> Path:
    root = local_cache_root(cache_dir, store_url)
    return root.with_name(f"{root.name}.agribound-etags.json")


def _read_index(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    objects = data.get("objects") if isinstance(data, dict) else None
    if not isinstance(objects, dict):
        return {}
    return {str(k): str(v) for k, v in objects.items()}


def _write_index(path: Path, store_url: str, objects: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps({"store": store_url, "objects": dict(sorted(objects.items()))}))
    os.replace(tmp, path)


def coordinate_object_keys(client: StoreClient, zone: int) -> list[str]:
    """Chunk keys of the coordinate arrays (:data:`COORDINATE_ARRAYS`) of ``utm<NN>``."""
    prefix = f"utm{int(zone):02d}"
    keys: list[str] = []
    for name in COORDINATE_ARRAYS:
        meta = client.get_json(f"{prefix}/{name}/zarr.json")
        if meta is None:
            continue
        keys += [f"{prefix}/{name}/{k}" for k in chunk_keys(meta, {})]
    return keys


def _local_file(root: Path, key: str) -> Path:
    return root.joinpath(*key.split("/"))


def refresh_local_cache(
    cache_dir: str | os.PathLike[str], stamp: StoreStamp, *, client: StoreClient | None = None
) -> dict[str, str | None]:
    """Delete the cached copies of a read's objects that may be out of date.

    The objects are those :func:`geotessera_store` keeps on disk for the read:
    the store's root ``zarr.json`` (which holds the consolidated metadata
    geotessera reads) and the chunks of the coordinate arrays of the zones
    read. A cached copy is deleted when the object's ETag now differs from the
    one recorded when it was cached, when no ETag was recorded for it, or when
    the object no longer exists, so that geotessera downloads it again. Shard
    copies (kept by geotessera's own cache) are left alone: agribound does not
    read them.

    Returns
    -------
    dict
        Key -> current ETag (*None*: missing) of every object checked; pass it
        to :func:`record_local_cache` after the read.

    Raises
    ------
    StoreCheckError
        If a request fails.
    """
    client = client or StoreClient(stamp.store_url)
    keys = ["zarr.json"]
    for zone in stamp.zones:
        keys += coordinate_object_keys(client, zone)
    current: dict[str, str | None] = {k: s.etag for k, s in client.head_many(keys).items()}
    root = local_cache_root(cache_dir, stamp.store_url)
    index = _read_index(_index_path(cache_dir, stamp.store_url))
    removed = []
    for key, etag in sorted(current.items()):
        path = _local_file(root, key)
        if path.is_file() and (etag is None or index.get(key) != etag):
            path.unlink(missing_ok=True)
            removed.append(key)
    if removed:
        logger.info(
            "Deleted %d object(s) from geotessera's cache %s that changed in the store or have "
            "no recorded ETag (they are downloaded again): %s",
            len(removed),
            root,
            ", ".join(removed[:10]) + (", ..." if len(removed) > 10 else ""),
        )
    return current


def record_local_cache(
    cache_dir: str | os.PathLike[str], store_url: str, current: dict[str, str | None]
) -> None:
    """Record the ETags of the objects of *current* that are now cached on disk.

    *current* is the result of :func:`refresh_local_cache` before the read, so
    an object rewritten during the read is recorded with its older ETag and
    downloaded again by the next read. Only the keys of
    :func:`disk_cached_key` are recorded.
    """
    root = local_cache_root(cache_dir, store_url)
    path = _index_path(cache_dir, store_url)
    index = _read_index(path)
    changed = False
    for key, etag in current.items():
        if not disk_cached_key(key):
            continue
        cached = _local_file(root, key).is_file()
        if cached and etag is not None and index.get(key) != etag:
            index[key] = etag
            changed = True
        elif not cached and key in index:
            del index[key]
            changed = True
    if changed:
        _write_index(path, str(store_url).rstrip("/"), index)


__all__ = [
    "CHECK_ENV",
    "STAMP_TAGS",
    "STAMP_VERSION",
    "TAG_FINGERPRINT",
    "TAG_LAST_MODIFIED",
    "TAG_OBJECTS",
    "ObjectState",
    "StoreCheckError",
    "StoreClient",
    "StoreStamp",
    "ZoneMetadata",
    "check_disabled_reason",
    "chunk_key",
    "chunk_keys",
    "clear_memo",
    "compute_stamp",
    "coordinate_object_keys",
    "disk_cached_key",
    "geotessera_store",
    "local_cache_root",
    "pixel_window",
    "read_object_keys",
    "read_small_array",
    "record_local_cache",
    "refresh_local_cache",
    "store_stamp",
    "urlopen",
    "zone_for_lon",
    "zone_metadata",
    "zone_parts",
]
