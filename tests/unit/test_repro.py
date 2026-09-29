"""Tests for agribound._repro (seeding, generators, versions, run ids)."""

from __future__ import annotations

import os
import random
import re
import subprocess
import sys

import numpy as np
import pytest

from agribound._repro import collect_versions, get_rng, new_run_id, seed_everything
from agribound._version import __version__
from agribound.config import AgriboundConfig


class TestSeedEverything:
    def test_python_and_numpy_reproducible(self):
        seed_everything(123)
        a = (random.random(), np.random.rand(3).tolist())
        seed_everything(123)
        b = (random.random(), np.random.rand(3).tolist())
        assert a == b
        seed_everything(124)
        assert (random.random(), np.random.rand(3).tolist()) != a

    def test_sets_pythonhashseed(self):
        seed_everything(7)
        assert os.environ["PYTHONHASHSEED"] == "7"

    def test_torch_reproducible(self):
        torch = pytest.importorskip("torch")
        seed_everything(5)
        a = torch.rand(4)
        seed_everything(5)
        b = torch.rand(4)
        assert torch.equal(a, b)

    def test_lightning_env(self):
        pytest.importorskip("lightning")
        seed_everything(11)
        assert os.environ.get("PL_GLOBAL_SEED") == "11"
        assert os.environ.get("PL_SEED_WORKERS") == "1"

    def test_deterministic_flags(self):
        torch = pytest.importorskip("torch")
        prev_alg = torch.are_deterministic_algorithms_enabled()
        prev_cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        try:
            seed_everything(1, deterministic=True)
            assert torch.are_deterministic_algorithms_enabled()
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False
        finally:
            torch.use_deterministic_algorithms(prev_alg)
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev_cudnn

    @pytest.mark.parametrize("seed", [-1, 2**32])
    def test_out_of_range(self, seed):
        with pytest.raises(ValueError, match="seed must be"):
            seed_everything(seed)


class TestGetRng:
    def test_same_seed_and_salt_same_stream(self):
        cfg = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", seed=3)
        assert get_rng(cfg, "split").random() == get_rng(cfg, "split").random()

    def test_salt_and_seed_separate_streams(self):
        cfg = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", seed=3)
        base = get_rng(cfg, "split").random()
        assert get_rng(cfg, "other").random() != base
        assert get_rng(cfg).random() != base
        assert get_rng(cfg.merged(seed=4), "split").random() != base

    def test_accepts_int(self):
        cfg = AgriboundConfig(source="local", local_tif_path="/tmp/x.tif", seed=9)
        assert get_rng(9, "a", 1).integers(0, 10**9) == get_rng(cfg, "a", 1).integers(0, 10**9)

    def test_independent_of_global_state(self):
        a = get_rng(1, "x").random()
        np.random.seed(999)
        random.seed(999)
        assert get_rng(1, "x").random() == a

    def test_stable_across_processes(self):
        """Salt hashing must not depend on PYTHONHASHSEED."""
        code = (
            "from agribound._repro import get_rng; "
            "print(repr(get_rng(42, 'split', 'tile-7').random()))"
        )
        outs = set()
        for hashseed in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=hashseed)
            res = subprocess.run(
                [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
            )
            outs.add(res.stdout.strip())
        assert len(outs) == 1
        assert outs.pop() == repr(get_rng(42, "split", "tile-7").random())


class TestCollectVersions:
    def test_core_entries(self):
        versions = collect_versions()
        assert versions["agribound"] == __version__ == "1.0.0"
        assert versions["python"] == sys.version.split()[0]
        assert "numpy" in versions
        assert "gdal" in versions
        assert all(isinstance(v, str) for v in versions.values())

    def test_extra_and_missing(self):
        versions = collect_versions(extra=["pyyaml", "definitely-not-installed-xyz"])
        assert "pyyaml" in versions
        assert "definitely-not-installed-xyz" not in versions

    def test_does_not_import_packages(self):
        code = (
            "import sys; from agribound._repro import collect_versions; collect_versions(); "
            "print('ultralytics' in sys.modules, 'terratorch' in sys.modules)"
        )
        res = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert res.stdout.strip() == "False False"


class TestRunId:
    def test_format(self):
        assert re.match(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$", new_run_id())

    def test_unique(self):
        assert len({new_run_id() for _ in range(50)}) == 50
