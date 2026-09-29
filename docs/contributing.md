# Contributing

Contributions are welcome: bug fixes, engines, sources, documentation and
examples. The repository's
[CONTRIBUTING.md](https://github.com/montimaj/agribound/blob/main/CONTRIBUTING.md)
has the short version of this page.

## Development environment

```bash
git clone https://github.com/montimaj/agribound.git
cd agribound
conda env create -f environment.yml        # core: pip install -e .[all,dev]
conda activate agribound
pip install -e ".[docs]"                   # documentation toolchain
```

Prithvi (terratorch) needs the separate `environment-gfm.yml` environment,
because terratorch and ftw-tools 2.x require incompatible `lightning`
versions.

## Tests

```bash
# offline suite (what CI runs)
python -m pytest -m "not gpu and not gee and not slow and not network"

# everything, including Earth Engine and network tests
python -m pytest

# coverage
python -m pytest --cov=agribound --cov-report=html
```

| Marker | Meaning |
|---|---|
| `gpu` | needs a GPU |
| `gee` | needs Earth Engine authentication |
| `network` | needs network access |
| `slow` | long-running |

Every behaviour change needs a unit test that runs offline (no network, GPU or
Earth Engine); stub the remote service or the model. CI runs Ruff and a check
that the example notebooks match their scripts, the offline suite on Python
3.12 and 3.13 on Linux, macOS and Windows (core install), an engine job on
Linux, and a strict documentation build with a link check.

## Style

[Ruff](https://docs.astral.sh/ruff/) is configured in `pyproject.toml`
(line length 100, target Python 3.12, rules E, F, W, I, N, UP, B, SIM). CI
checks the whole repository:

```bash
ruff check .
ruff format --check .      # ruff format . applies the formatting
```

Shell scripts (`*.sh`, `*.sbatch`) and the HPC profiles
(`examples/hpc/profiles/*.env`, which bash sources) keep LF line endings in
every checkout through `.gitattributes`, also on Windows with
`core.autocrlf=true`: bash cannot run a script with CRLF endings.
`tests/unit/test_repo_files.py` checks the endings and the attributes.

Code conventions: numpy-style docstrings; `from __future__ import annotations`;
heavy or optional dependencies imported inside functions;
`logger = logging.getLogger(__name__)`. Docstrings and documentation must be
true of the code. No silent fallbacks that change the method (a different
model, band set or engine): raise an actionable error or require an explicit
opt-in recorded in `engine_meta`, and log at WARNING when anything degrades.

## Adding an engine

1. Write the engine class, for example `agribound/engines/my_engine.py`:

    ```python
    from __future__ import annotations

    import geopandas as gpd

    from agribound.config import AgriboundConfig
    from agribound.engines.base import DelineationEngine
    from agribound.registry import ENGINE_REGISTRY


    class MyEngine(DelineationEngine):
        name = "my-engine"
        supported_sources = list(ENGINE_REGISTRY["my-engine"]["supported_sources"])
        requires_bands = list(ENGINE_REGISTRY["my-engine"]["requires_bands"])

        def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
            self.validate_input(raster_path, config)
            ...  # return polygons with a CRS
            gdf.attrs["engine_meta"] = {"model": "...", "weights_sha256": "..."}
            return gdf

        @classmethod
        def prefetch(cls, config: AgriboundConfig) -> list[str]:
            """Download the weights for offline nodes; return their local paths."""
    ```

2. Register it in `agribound/registry.py`: an `ENGINE_REGISTRY` entry with all
   keys (`name`, `approach`, `strengths`, `gpu_recommended`, `requires_bands`,
   `supported_sources`, `label_free`, `fine_tunable`, `reference`,
   `install_extra`, `notes`, optionally `source_notes`) and an
   `ENGINE_CLASSES` entry `"my-engine": "agribound.engines.my_engine:MyEngine"`.
3. Read bands with `get_canonical_band_indices(config.source, names,
   bands=config.bands)` and convert values with `agribound.io.raster`
   (`to_unit_reflectance`, `to_s2_dn`, `percentile_stretch_uint8`) according to
   `registry.source_value_scale(config.source)`; never assume Sentinel-2
   units.
4. Name every intermediate with `agribound._cache.cache_path` and seed random
   choices with `agribound._repro.get_rng`.
5. If it is fine-tunable, add a trainer module to `agribound/engines/finetune/`
   and an entry to its dispatcher (`_TRAINERS` in its `__init__.py`), and give
   the engine a chip format (`CHIP_FORMATS`) and a default chip size in
   `agribound/engines/finetune/_data.py`. A default that is derived rather
   than fixed (as for Delineate-Anything and GeoAI) also needs its own branch
   in `chip_size_rule()`: the rule's identifier is part of the fine-tuning
   cache key, so a changed rule does not reuse old checkpoints.
6. Add an optional-dependency extra in `pyproject.toml`, tests, and
   documentation (engines page, README table).

## Adding a source

1. Add a `SOURCE_REGISTRY` entry in `agribound/registry.py` with every key
   (`name`, `collection`, `resolution_m`, `native_resolution_m`, `all_bands`,
   `canonical_bands`, `value_scale`, `year_range`, `coverage`, `requires_gee`,
   `restricted`); facts must be checked against the provider's catalogue.
2. For an Earth Engine collection, add a collection builder to
   `_COLLECTION_BUILDERS` in `agribound/composites/gee.py` and the source to
   `_OPTICAL_GEE_SOURCES` in `agribound/registry.py` (which defines
   `GEE_IMAGERY_SOURCES`); otherwise write a `CompositeBuilder` and register
   it in `_BUILDERS` in `agribound/composites/base.py`.
3. The builder must write a GeoTIFF on the grid of
   `agribound.composites.gee.compute_export_grid` (the study-area bounding
   box in `export_crs`), cache it with `cache_path`, and raise
   `agribound.composites.NoDataError` when the source has no data for the
   study area.
4. Add the source to the `supported_sources` of the engines that can use it.

## Documentation

```bash
mkdocs serve
mkdocs build --strict
```

The API reference is generated from the docstrings with mkdocstrings.

## Examples, notebooks and figures

The scripts `examples/NN_*.py` are the source of the examples. The notebooks
in `examples/notebooks/` hold the same code and are generated from them, so
edit the script and regenerate (the tool needs `nbformat`):

```bash
python tools/sync_notebooks.py           # rewrite out-of-date notebooks
python tools/sync_notebooks.py --check   # what CI runs; exit 1 if a notebook differs
```

The gallery images in `assets/gallery_1.0/` (and the facts their captions
quote, `gallery_stats.json`) are rendered from the example outputs by
`tools/make_gallery.py`, and the workflow diagram
`assets/agribound_workflow_1.0.{png,svg}` by `tools/make_workflow_diagram.py`;
`assets/README.md` lists the files.

## Packaging

```bash
python -m build                     # sdist and wheel in dist/
twine check --strict dist/*
```

The licence metadata follows PEP 639 (`license = "Apache-2.0"` and
`license-files = ["LICENSE"]` in `pyproject.toml`, no licence classifier), so
the build needs `setuptools>=77`; `python -m build` installs it in its
isolated build environment. The version comes from `agribound/_version.py`.
setuptools-scm puts the files tracked by git into the sdist, and
`MANIFEST.in` adds or removes others: it adds every `agribound/*.py` and
prunes `assets/`, `paper/`, `outputs/`, `examples/notebooks/` and `.github/`.
Other new files (tests, documentation) reach the sdist only once they are
committed.

## Pull requests

1. Branch from `main`.
2. Add tests and documentation for the change.
3. Run the offline test suite and `ruff check`.
4. Describe what changed and why.
