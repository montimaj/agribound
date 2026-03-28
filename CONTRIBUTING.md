# Contributing to Agribound

We welcome contributions! Whether it's a bug fix, new engine, satellite source, or documentation improvement, here's how to get started.

## Development Setup

```bash
git clone https://github.com/montimaj/agribound.git
cd agribound
conda create -n agribound python=3.12 gdal rasterio geopandas fiona shapely pyproj -c conda-forge
conda activate agribound
pip install -e ".[all,dev,docs]"

# Required for DA instance segmentation on Sentinel-2
git clone https://github.com/fieldsoftheworld/ftw-baselines.git ../ftw-baselines
pip install -e ../ftw-baselines
```

## Running Tests

```bash
# All tests (excluding GPU and GEE)
pytest -m "not gpu and not gee"

# With coverage
pytest --cov=agribound --cov-report=html

# Only fast tests
pytest -m "not slow"
```

Test markers: `gpu` (requires CUDA), `gee` (requires GEE auth), `slow` (long-running).

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting (config in `pyproject.toml`):

```bash
ruff check agribound/ examples/ tests/
ruff format agribound/
```

- Target: Python 3.10+
- Line length: 100
- Rules: E, F, W, I, N, UP, B, SIM

## Adding a New Engine

1. Create `agribound/engines/my_engine.py` implementing `DelineationEngine`:

    ```python
    from agribound.engines.base import DelineationEngine
    from agribound.config import AgriboundConfig
    import geopandas as gpd

    class MyEngine(DelineationEngine):
        name = "my-engine"
        supported_sources = ["sentinel2", "local"]
        requires_bands = ["R", "G", "B"]

        def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
            ...
    ```

2. Register in `agribound/engines/base.py` (`ENGINE_REGISTRY` + `get_engine()`)
3. Add engine name to `VALID_ENGINES` in `agribound/config.py`
4. Add optional dependencies in `pyproject.toml`
5. Write tests in `tests/`
6. Add an example script in `examples/`

## Adding a New Satellite Source

1. Register in `agribound/composites/base.py` (`SOURCE_REGISTRY`)
2. Add source name to `VALID_SOURCES` in `agribound/config.py`
3. Implement cloud masking (if GEE-based) in `agribound/composites/gee.py`
4. Add a factory branch in `get_composite_builder()` if needed

## Building Documentation

```bash
pip install agribound[docs]
mkdocs serve    # local preview at http://127.0.0.1:8000
mkdocs build    # build static site
```

## Pull Request Guidelines

- Create a feature branch from `main`
- Keep PRs focused — one feature or fix per PR
- Add tests for new functionality
- Run `ruff check` and `pytest` before submitting
- Update documentation and examples if applicable
- Follow existing code patterns and naming conventions

## Reporting Issues

Please open an issue on [GitHub](https://github.com/montimaj/agribound/issues) with:

- A clear description of the problem
- Steps to reproduce
- Python version, OS, and agribound version (`agribound --version`)
- Full error traceback
