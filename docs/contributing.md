# Contributing

## Setting Up the Development Environment

Clone the repository and install in editable mode with all dependencies:

```bash
git clone https://github.com/montimaj/agribound.git
cd agribound
pip install -e ".[all,dev,docs]"
```

## Running Tests

Agribound uses pytest. Tests are located in the `tests/` directory.

```bash
# Run all tests
pytest

# Run tests excluding GPU and GEE requirements
pytest -m "not gpu and not gee"

# Run with coverage
pytest --cov=agribound --cov-report=html

# Run only fast tests
pytest -m "not slow"
```

Test markers:

| Marker | Description |
|---|---|
| `gpu` | Requires a CUDA-capable GPU |
| `gee` | Requires GEE authentication |
| `slow` | Long-running tests |

## Code Style

Agribound uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Configuration is in `pyproject.toml`:

- Target: Python 3.10
- Line length: 100
- Selected rules: E, F, W, I, N, UP, B, SIM

Run the linter:

```bash
ruff check agribound/
ruff format agribound/
```

## Adding a New Engine

To add a new delineation engine:

1. **Create the engine module** at `agribound/engines/my_engine.py`:

    ```python
    from agribound.engines.base import DelineationEngine
    from agribound.config import AgriboundConfig
    import geopandas as gpd


    class MyEngine(DelineationEngine):
        name = "my-engine"
        supported_sources = ["sentinel2", "local"]
        requires_bands = ["R", "G", "B"]

        def delineate(
            self, raster_path: str, config: AgriboundConfig
        ) -> gpd.GeoDataFrame:
            # Implement delineation logic here.
            # Must return a GeoDataFrame with at least a geometry column.
            ...
    ```

2. **Register the engine** in `agribound/engines/base.py` by adding an entry to `ENGINE_REGISTRY`:

    ```python
    ENGINE_REGISTRY["my-engine"] = {
        "name": "My Engine",
        "approach": "Description of the approach",
        "strengths": "When this engine excels",
        "gpu_required": True,
        "requires_bands": ["R", "G", "B"],
        "supported_sources": ["sentinel2", "local"],
        "reference": "Paper or project reference",
        "install_extra": "my-engine",
    }
    ```

3. **Add the factory branch** in `get_engine()` in `agribound/engines/base.py`:

    ```python
    elif engine_name == "my-engine":
        from agribound.engines.my_engine import MyEngine
        return MyEngine()
    ```

4. **Add the engine name** to `VALID_ENGINES` in `agribound/config.py`.

5. **Add optional dependencies** (if any) as a new extra in `pyproject.toml`:

    ```toml
    [project.optional-dependencies]
    my-engine = ["some-package>=1.0"]
    ```

6. **Write tests** in `tests/test_my_engine.py`.

## Adding a New Satellite Source

To add a new satellite source:

1. **Register the source** in `agribound/composites/base.py` by adding to `SOURCE_REGISTRY`:

    ```python
    SOURCE_REGISTRY["my-source"] = {
        "name": "My Source",
        "collection": "GEE/COLLECTION/ID",
        "resolution_m": 10,
        "all_bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"],
        "canonical_bands": {"R": "B4", "G": "B3", "B": "B2", "NIR": "B8"},
        "coverage": "Description of spatial and temporal coverage",
        "requires_gee": True,
    }
    ```

2. **Add the source name** to `VALID_SOURCES` in `agribound/config.py`.

3. **Implement a cloud masking function** (if GEE-based) in `agribound/composites/gee.py`.

4. **Add a factory branch** in `get_composite_builder()` in `agribound/composites/base.py` if the source requires a custom builder. Otherwise, the existing `GEECompositeBuilder` handles it automatically for GEE sources.

5. **Write tests** covering composite generation for the new source.

## Building the Documentation

```bash
pip install agribound[docs]
mkdocs serve    # local preview at http://127.0.0.1:8000
mkdocs build    # build static site to site/
```

## Project Structure

```
agribound/
    __init__.py          # Public API exports
    _version.py          # Version string
    auth.py              # GEE authentication
    cli.py               # Click CLI
    config.py            # AgriboundConfig dataclass
    pipeline.py          # Main pipeline orchestrator
    evaluate.py          # Accuracy metrics
    visualize.py         # leafmap visualization
    composites/
        __init__.py
        base.py          # CompositeBuilder ABC + source registry
        gee.py           # GEE composite builder
        local.py         # Local/embedding composite builders
    engines/
        __init__.py
        base.py          # DelineationEngine ABC + engine registry
        delineate_anything.py
        ftw.py
        geoai_field.py
        prithvi.py
        embedding.py
        ensemble.py
        finetune.py      # Fine-tuning orchestrator
    io/
        __init__.py
        raster.py        # GeoTIFF read/write
        vector.py        # Vector read/write
        crs.py           # CRS utilities
    postprocess/
        __init__.py
        polygonize.py    # Mask to polygon conversion
        simplify.py      # Douglas-Peucker simplification
        regularize.py    # Polygon regularization
        filter.py        # Area and hole filtering
        merge.py         # Cross-tile polygon merging
```
