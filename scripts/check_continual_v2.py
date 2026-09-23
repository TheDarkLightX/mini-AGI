#!/usr/bin/env python3
"""Runtime invariants for the Continual V2 experimental branch."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Running this file as `python scripts/check_continual_v2.py` otherwise makes
# scripts/ the import root rather than the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from minagi.recur import RecurConfig, RecurCoder
from train import _ReplayMemory, _split_trunk_router_experts


def check_parameter_partition() -> None:
    cfg = RecurConfig(
        vocab_size=265,
        d_model=16,
        n_head=4,
        d_ff=32,
        n_prelude=1,
        n_recur=1,
        n_coda=0,
        max_steps=1,
        block=16,
        use_pool=True,
        pool_experts=4,
        pool_d_ff=24,
        pool_top_k=2,
        pool_max=4,
    )
    model = RecurCoder(cfg)
    trunk, routers, experts = _split_trunk_router_experts(model)

    groups = [{id(p) for p in x} for x in (trunk, routers, experts)]
    all_ids = {id(p) for p in model.parameters()}

    assert all(groups), "all three plasticity roles must be non-empty"
    assert not (groups[0] & groups[1])
    assert not (groups[0] & groups[2])
    assert not (groups[1] & groups[2])
    assert groups[0] | groups[1] | groups[2] == all_ids


def check_replay() -> None:
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(4):
            p = os.path.join(td, f"d{i}.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(("domain-%d " % i) * 30)
            paths.append(p)

        state = os.path.join(td, "replay.json")
        mem = _ReplayMemory(capacity=2, state_path=state, seed=17)
        for i, p in enumerate(paths):
            mem.add(f"d{i}", p, i)
        assert len(mem.marks) == 2
        assert mem.seen == 4

        # Exact continuation: saving and restoring must preserve which mark the
        # next random draw selects, not merely preserve the reservoir contents.
        mem.save()
        expected = mem.sample()
        restored = _ReplayMemory(capacity=2, state_path=state, seed=999)
        actual = restored.sample()
        assert expected == actual

        # A rewritten file is not the old episode. A one-mark reservoir makes
        # the stale-eviction outcome deterministic.
        stale_state = os.path.join(td, "stale.json")
        stale = _ReplayMemory(capacity=1, state_path=stale_state, seed=3)
        stale.add("old", paths[0], 0)
        with open(paths[0], "a", encoding="utf-8") as f:
            f.write("changed")
        assert stale.sample() is None
        assert stale.marks == []


def main() -> None:
    check_parameter_partition()
    check_replay()
    print("continual-v2 runtime invariants: OK")


if __name__ == "__main__":
    main()
