"""Tests for agribound.io.vector read/write round-trips."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from agribound.io.vector import read_vector, write_vector


class TestVectorRoundTripGpkg:
    """Test GeoPackage write then read."""

    def test_gpkg_round_trip(self, tmp_path, sample_geodataframe):
        path = tmp_path / "output.gpkg"
        write_vector(sample_geodataframe, path)
        loaded = read_vector(path)

        assert len(loaded) == len(sample_geodataframe)
        assert loaded.crs is not None

    def test_gpkg_geometry_preserved(self, tmp_path):
        poly = box(0, 0, 1, 1)
        gdf = gpd.GeoDataFrame(
            {"val": [42]}, geometry=[poly], crs="EPSG:4326"
        )
        path = tmp_path / "test.gpkg"
        write_vector(gdf, path)
        loaded = read_vector(path)
        assert loaded.geometry.iloc[0].equals(poly)


class TestVectorRoundTripGeojson:
    """Test GeoJSON write then read."""

    def test_geojson_round_trip(self, tmp_path):
        poly = box(-1, -1, 1, 1)
        gdf = gpd.GeoDataFrame(
            {"name": ["square"]}, geometry=[poly], crs="EPSG:4326"
        )
        path = tmp_path / "test.geojson"
        write_vector(gdf, path)
        loaded = read_vector(path)

        assert len(loaded) == 1
        assert loaded.crs is not None

    def test_geojson_reprojects_to_4326(self, tmp_path, sample_geodataframe):
        """GeoJSON output must be EPSG:4326; write_vector should reproject."""
        path = tmp_path / "reprojected.geojson"
        write_vector(sample_geodataframe, path)
        loaded = read_vector(path)
        # GeoJSON is always 4326
        assert loaded.crs.to_epsg() == 4326


class TestReadVectorErrors:
    """Test error handling in read_vector."""

    def test_missing_file_raises(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            read_vector(tmp_path / "nonexistent.gpkg")

    def test_unsupported_extension_raises(self, tmp_path):
        import pytest

        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("not a vector")
        with pytest.raises(ValueError, match="Unsupported vector format"):
            read_vector(bad_file)
