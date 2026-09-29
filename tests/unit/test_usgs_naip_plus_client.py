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
