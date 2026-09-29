# Contributing to Agribound

We welcome contributions: bug fixes, engines, satellite sources, documentation
and examples. The full guide is the
[Contributing page](https://montimaj.github.io/agribound/contributing/) of the
documentation.

## Development setup

```bash
git clone https://github.com/montimaj/agribound.git
cd agribound
conda env create -f environment.yml        # core environment, installs -e .[all,dev]
conda activate agribound
pip install -e ".[docs]"                   # optional: documentation toolchain
```

Python >= 3.12 is required. Prithvi (terratorch) needs the separate
`environment-gfm.yml` environment: terratorch requires `lightning>=2.6` and
ftw-tools 2.x requires `lightning<2.6`.

## Running tests

```bash
# Offline suite (what CI runs)
python -m pytest -m "not gpu and not gee and not slow and not network"

# With coverage
python -m pytest --cov=agribound --cov-report=html
```

Test markers: `gpu` (needs a GPU), `gee` (needs Earth Engine authentication),
`network` (needs network access), `slow` (long-running). New behaviour needs an
offline unit test.

## Code style

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

Use numpy-style docstrings, `from __future__ import annotations`, lazy imports
of heavy or optional dependencies, and `logging.getLogger(__name__)`. Do not
add silent fallbacks that change the method (another model, band set or
engine): raise an actionable error or require an explicit opt-in recorded in
`engine_meta`.

## Adding an engine

1. Implement `DelineationEngine` in `agribound/engines/my_engine.py`
   (`delineate(raster_path, config)` returning a GeoDataFrame with a CRS and
   `gdf.attrs["engine_meta"]`; `prefetch(config)` for offline weights).
2. Register it in `agribound/registry.py`: an `ENGINE_REGISTRY` entry and an
   `ENGINE_CLASSES` entry (`"module:Class"`). `VALID_ENGINES` in
   `agribound/config.py` is derived from the registry.
3. Use `get_canonical_band_indices(..., bands=config.bands)`, the value-scale
   helpers in `agribound.io.raster`, `agribound._cache.cache_path` for
   intermediates and `agribound._repro.get_rng` for randomness.
4. If the engine is fine-tunable: a trainer in `agribound/engines/finetune/`
   registered in `_TRAINERS` (`finetune/__init__.py`), and in
   `finetune/_data.py` a `CHIP_FORMATS` entry, a default chip size and, if the
   default is derived from the data, a `chip_size_rule()` branch (the rule is
   part of the fine-tuning cache key).
5. Add an extra in `pyproject.toml`, offline tests, documentation and, if
   useful, an example.

## Adding a satellite source

1. Add a `SOURCE_REGISTRY` entry in `agribound/registry.py` (`VALID_SOURCES`
   is derived from it).
2. Earth Engine sources: add a collection builder with cloud masking to
   `_COLLECTION_BUILDERS` in `agribound/composites/gee.py` and the source to
   `_OPTICAL_GEE_SOURCES` in `agribound/registry.py` (which defines
   `GEE_IMAGERY_SOURCES`). Other sources: write a `CompositeBuilder` and add it
   to `_BUILDERS` in `agribound/composites/base.py`.
3. Raise `agribound.composites.NoDataError` when the source has no data for
   the study area, and add the source to the engines that support it.

## Building documentation

```bash
mkdocs serve            # local preview at http://127.0.0.1:8000
mkdocs build --strict   # what CI runs
```

## Building the package

```bash
python -m build                 # sdist and wheel; needs setuptools>=77 (PEP 639 licence metadata)
twine check --strict dist/*
```

## Pull request guidelines

- Create a feature branch from `main`
- Keep PRs focused: one feature or fix per PR
- Add tests for new functionality
- Run `ruff check .`, `ruff format --check .`, the offline test suite and,
  if you changed an example script, `python tools/sync_notebooks.py` (CI runs
  it with `--check`) before submitting
- Update documentation and examples if applicable

## Reporting issues

Please open an issue on [GitHub](https://github.com/montimaj/agribound/issues) with:

- A clear description of the problem
- Steps to reproduce
- Python version, OS, and agribound version (`agribound --version`)
- The full error traceback, and the run's `*.provenance.json` if there is one
