from __future__ import annotations

from pathlib import Path

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