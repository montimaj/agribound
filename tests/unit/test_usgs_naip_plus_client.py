from __future__ import annotations

from pathlib import Path

from shapely.geometry import Point

from agribound.clients.usgs_naip_plus import USGSNAIPPlusClient


class TestUSGSNAIPPlusClient:
    def test_query_candidates_parses_features(
        self,
        monkeypatch,
        sample_usgs_query_features,
    ):
        client = USGSNAIPPlusClient("https://example.com/ImageServer")

        monkeypatch.setattr(
            client,
            "query_object_ids",
            lambda bounds_3857, where: [102, 101],
        )

        def fake_request_json(path, params):
            assert path == "/query"
            return sample_usgs_query_features

        monkeypatch.setattr(client, "_request_json", fake_request_json)

        candidates = client.query_candidates(
            bounds_3857=(-13024380.0, 5265000.0, -13022380.0, 5266000.0),
            where="Category = 1 AND Year = 2023 AND State = 'MI'",
        )

        assert [candidate.object_id for candidate in candidates] == [101, 102]
        assert candidates[0].state == "MI"
        assert candidates[0].year == 2023
        assert candidates[0].band_count == 4
        assert candidates[0].geometry is not None

    def test_export_image_downloads_to_output_path(self, monkeypatch, tmp_path):
        client = USGSNAIPPlusClient("https://example.com/ImageServer")
        output_path = tmp_path / "out.tif"

        def fake_request_json(path, params):
            assert path == "/exportImage"
            assert params["format"] == "tiff"
            assert "mosaicRule" in params
            return {"href": "https://example.com/download/out.tif"}

        def fake_download(url, local_path):
            Path(local_path).write_bytes(b"fake")

        monkeypatch.setattr(client, "_request_json", fake_request_json)
        monkeypatch.setattr(client, "_download_file", fake_download)

        payload = client.export_image(
            bbox_3857=(-13024380.0, 5265000.0, -13023380.0, 5266000.0),
            width=512,
            height=512,
            lock_raster_ids=[101, 102],
            output_path=output_path,
        )

        assert payload["href"].endswith("out.tif")
        assert output_path.exists()


class TestUSGSClientQueryAndGeometry:
    def test_query_year_counts(self, monkeypatch):
        client = USGSNAIPPlusClient("https://example.com/ImageServer")
        captured = {}

        def fake_request_json(path, params):
            captured.update(params)
            return {
                "features": [
                    {"attributes": {"Year": 2022, "n_items": 7}},
                    {"attributes": {"Year": 2019, "n_items": 2}},
                ]
            }

        monkeypatch.setattr(client, "_request_json", fake_request_json)
        counts = client.query_year_counts((0, 0, 1, 1), "Category = 1")
        assert counts == {2019: 2, 2022: 7}
        assert list(counts) == [2019, 2022]
        assert captured["groupByFieldsForStatistics"] == "Year"
        assert '"statisticType":"count"' in captured["outStatistics"]
        assert captured["returnGeometry"] == "false"

    def test_esri_holes_are_subtracted(self):
        # outer ring clockwise, hole counter-clockwise (Esri convention)
        outer = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]
        hole = [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]]
        geom = USGSNAIPPlusClient._esri_geometry_to_shapely({"rings": [outer, hole]})
        assert geom.area == 100 - 4
        assert not geom.contains(Point(5, 5))

    def test_esri_two_outer_rings(self):
        a = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]
        b = [[5, 5], [5, 6], [6, 6], [6, 5], [5, 5]]
        geom = USGSNAIPPlusClient._esri_geometry_to_shapely({"rings": [a, b]})
        assert geom.geom_type == "MultiPolygon" and geom.area == 2


class TestDownloadFile:
    def test_streams_with_timeout_and_renames(self, monkeypatch, tmp_path):
        import io

        import agribound.clients.usgs_naip_plus as mod

        seen = {}

        class _Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        def fake_urlopen(url, timeout=None):
            seen["url"], seen["timeout"] = url, timeout
            return _Response(b"TIFFDATA")

        monkeypatch.setattr(mod, "urlopen", fake_urlopen)
        client = USGSNAIPPlusClient("https://example.com/ImageServer", timeout_s=17)
        out = tmp_path / "tile.tif"
        client._download_file("https://example.com/x.tif", out)
        assert out.read_bytes() == b"TIFFDATA"
        assert seen == {"url": "https://example.com/x.tif", "timeout": 17}
        assert not (tmp_path / "tile.tif.part").exists()

    def test_stalled_download_retries_then_fails_cleanly(self, monkeypatch, tmp_path):
        import pytest

        import agribound.clients.usgs_naip_plus as mod

        calls = []

        def stalled(url, timeout=None):
            calls.append(timeout)
            raise TimeoutError("The read operation timed out")

        monkeypatch.setattr(mod, "urlopen", stalled)
        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        client = USGSNAIPPlusClient("https://example.com/ImageServer", timeout_s=5, retries=2)
        with pytest.raises(mod.USGSImageServerError, match="Failed download"):
            client._download_file("https://example.com/x.tif", tmp_path / "tile.tif")
        assert calls == [5, 5, 5]
        assert not list(tmp_path.iterdir())


class TestRetries:
    """Transient failures are retried with a capped exponential backoff (no network)."""

    @staticmethod
    def _opener(monkeypatch, replies):
        """Patch urlopen to play *replies* in turn: bytes (a body) or an exception."""
        import io

        import agribound.clients.usgs_naip_plus as mod

        class _Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.close()

        calls, sleeps = [], []

        def fake_urlopen(url, timeout=None):
            calls.append(url)
            reply = replies[len(calls) - 1]
            if isinstance(reply, BaseException):
                raise reply
            return _Response(reply)

        monkeypatch.setattr(mod, "urlopen", fake_urlopen)
        monkeypatch.setattr(mod.time, "sleep", sleeps.append)
        monkeypatch.setattr(mod.random, "uniform", lambda a, b: b)  # the largest jitter
        return mod, calls, sleeps

    @staticmethod
    def _http_error(code):
        from urllib.error import HTTPError

        return HTTPError("https://example.com/ImageServer", code, f"HTTP {code}", {}, None)

    def test_gateway_errors_and_timeouts_are_retried(self, monkeypatch, caplog):
        import logging
        from urllib.error import URLError

        replies = [
            self._http_error(502),
            self._http_error(504),
            URLError(TimeoutError("timed out")),
            TimeoutError("The read operation timed out"),
            self._http_error(429),
            b'{"ok": 1}',
        ]
        mod, calls, sleeps = self._opener(monkeypatch, replies)
        client = USGSNAIPPlusClient("https://example.com/ImageServer")
        assert client.retries == mod.DEFAULT_RETRIES == 5
        with caplog.at_level(logging.WARNING, logger=mod.__name__):
            assert client._request_json("", {"f": "json"}) == {"ok": 1}
        assert len(calls) == 6
        # 4, 8, 16, 32 and (capped) 60 s, each with its jitter (here the largest, 25 %)
        assert sleeps == [5.0, 10.0, 20.0, 40.0, 75.0]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 5
        assert "HTTP 502" in warnings[0].getMessage() and "[1/5]" in warnings[0].getMessage()

    def test_persistent_server_error_fails_after_the_last_retry(self, monkeypatch):
        import pytest

        mod, calls, sleeps = self._opener(monkeypatch, [self._http_error(503)] * 6)
        client = USGSNAIPPlusClient("https://example.com/ImageServer")
        with pytest.raises(mod.USGSImageServerError, match="Failed request") as info:
            client._request_json("/query", {"f": "json"})
        assert len(calls) == 6 and len(sleeps) == 5
        assert max(sleeps) <= mod.RETRY_MAX_WAIT_S * (1 + mod.RETRY_JITTER)
        assert info.value.__cause__.code == 503

    def test_client_errors_are_not_retried(self, monkeypatch, tmp_path):
        import pytest

        for code in (400, 403, 404):
            mod, calls, sleeps = self._opener(monkeypatch, [self._http_error(code)])
            client = USGSNAIPPlusClient("https://example.com/ImageServer")
            with pytest.raises(mod.USGSImageServerError, match="Failed request"):
                client._request_json("/query", {"f": "json"})
            assert len(calls) == 1 and sleeps == []
        mod, calls, sleeps = self._opener(monkeypatch, [self._http_error(404)])
        with pytest.raises(mod.USGSImageServerError, match="Failed download"):
            client._download_file("https://example.com/x.tif", tmp_path / "x.tif")
        assert len(calls) == 1 and sleeps == [] and not list(tmp_path.iterdir())

    def test_local_file_system_errors_are_not_retried(self, monkeypatch, tmp_path):
        import errno

        import pytest

        local = [
            PermissionError(errno.EACCES, "Permission denied"),
            OSError(errno.ENOSPC, "No space left on device"),
            OSError(errno.EROFS, "Read-only file system"),
        ]
        for exc in local:
            mod, calls, sleeps = self._opener(monkeypatch, [exc])
            client = USGSNAIPPlusClient("https://example.com/ImageServer")
            with pytest.raises(mod.USGSImageServerError, match="Failed download"):
                client._download_file("https://example.com/x.tif", tmp_path / "x.tif")
            assert len(calls) == 1 and sleeps == [], exc
        assert mod._transient(ConnectionResetError(errno.ECONNRESET, "reset"))
        assert mod._transient(TimeoutError("timed out"))

    def test_arcgis_error_payloads(self, monkeypatch):
        import pytest

        busy = b'{"error": {"code": 500, "message": "Error exporting image", "details": []}}'
        mod, calls, sleeps = self._opener(monkeypatch, [busy, b'{"href": "x"}'])
        client = USGSNAIPPlusClient("https://example.com/ImageServer")
        assert client._request_json("/exportImage", {}) == {"href": "x"}
        assert len(calls) == 2 and len(sleeps) == 1
        bad = b'{"error": {"code": 400, "message": "Invalid parameters", "details": []}}'
        mod, calls, sleeps = self._opener(monkeypatch, [bad])
        with pytest.raises(mod.USGSImageServerError, match=r"\[400\] Invalid parameters"):
            client._request_json("/exportImage", {})
        assert len(calls) == 1 and sleeps == []

    def test_download_retries_a_bad_gateway(self, monkeypatch, tmp_path):
        mod, calls, sleeps = self._opener(monkeypatch, [self._http_error(502), b"TIFFDATA"])
        client = USGSNAIPPlusClient("https://example.com/ImageServer", retries=1)
        out = tmp_path / "tile.tif"
        client._download_file("https://example.com/x.tif", out)
        assert out.read_bytes() == b"TIFFDATA" and len(calls) == 2 and sleeps == [5.0]
        assert not (tmp_path / "tile.tif.part").exists()

    def test_retry_wait(self, monkeypatch):
        import agribound.clients.usgs_naip_plus as mod

        monkeypatch.setattr(mod.random, "uniform", lambda a, b: a)  # no jitter
        assert [mod._retry_wait(n) for n in range(1, 8)] == [4, 8, 16, 32, 60, 60, 60]
