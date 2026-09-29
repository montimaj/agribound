"""
Batch execution of large study areas (HPC job arrays).

- :mod:`agribound.hpc.tiles`: cut a study area into tiles with halos, write a
  manifest with one configuration per tile, run tiles idempotently, report
  their status and merge the results.
- :mod:`agribound.hpc.regions`: read the region definitions in
  ``examples/regions/*.yaml``.
- :mod:`agribound.hpc.cli`: the ``agribound tiles`` command group.

Submodules are imported on first attribute access, so importing this package
(which ``agribound`` does to register the CLI group) stays cheap.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "make_tiles": "agribound.hpc.tiles",
    "write_tile_manifest": "agribound.hpc.tiles",
    "load_manifest": "agribound.hpc.tiles",
    "load_tile_config": "agribound.hpc.tiles",
    "run_tile": "agribound.hpc.tiles",
    "merge_tiles": "agribound.hpc.tiles",
    "tile_status": "agribound.hpc.tiles",
    "assign_tile_ids": "agribound.hpc.tiles",
    "format_index_ranges": "agribound.hpc.tiles",
    "load_region": "agribound.hpc.regions",
    "find_region_file": "agribound.hpc.regions",
    "plan_runs": "agribound.hpc.regions",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'agribound.hpc' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


__all__ = sorted(_EXPORTS)
