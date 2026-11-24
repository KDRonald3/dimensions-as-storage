"""Tests of v3/grid.py (the runs, their order and groups) and v3/choose_rates.py (the rule)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from v2.batch import group_key, group_name, make_groups
from v2.run import complete_config
from v3 import choose_rates, grid
from v3.orchestrate import Orchestrator

PROJECT = Path(__file__).resolve().parents[2]
SAME_RATE = {"rate": 0.01}


def test_grid_imports_only_the_standard_library():
    check = ("import sys\nbefore = set(sys.modules)\nimport v3.grid, v3.orchestrate\n"
             "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
             "print(sorted(m for m in new if m not in ('v2', 'v3') "
             "and m not in sys.stdlib_module_names))\n")
    done = subprocess.run([sys.executable, "-c", check], cwd=PROJECT, capture_output=True,
                          text=True, check=True)
    assert done.stdout.strip() == "[]"


def test_practice_grid():
    runs = grid.practice()
    assert len(runs) == 240 and len({c["name"] for c in runs}) == 240
    groups = grid.groups(runs)
    assert len(groups) == 40 and all(len(g) == 6 for g in groups)
    for config in runs:
        assert config["seed"] == 100 and config["usage"] == "equal"
        assert config["steps"] == 40000 and "length_rule" not in config
        assert config["text_seed"] == 0 and config["table_start"] == "nested64"
        assert config["name"] == (f"d{config['columns']}_c{config['creatures']}_b"
                                  f"{config['body']}_equal_k{config['kinds']}_s100_lr"
                                  f"{config['lr']}")
    for group in groups:
        # A group: one columns, kinds and body; both creature counts and all three rates.
        assert len({(c["columns"], c["kinds"], c["body"]) for c in group}) == 1
        assert sorted((c["creatures"], c["lr"]) for c in group) == sorted(
            (n, r) for n in (16, 1024) for r in (0.003, 0.01, 0.03))
        # The group trainer takes it as one group (rates may differ).
        assert len(make_groups(group, by_rate=False)) == 1
    combos = {(c["columns"], c["kinds"]) for c in runs}
    assert combos == {(d, k) for d in grid.COLUMNS for k in grid.KINDS}


def test_counted_grid_order_and_counts():
    settings = {"rate": 0.03}
    runs = grid.counted(settings)
    assert len(runs) == 2880 and len({c["name"] for c in runs}) == 2880
    groups = grid.groups(runs)
    assert len(groups) == 120 and all(len(g) == 24 for g in groups)
    order = [(g[0]["body"], g[0]["kinds"], g[0]["columns"]) for g in groups]
    assert order == [(b, k, d) for b in (128, 4, 32, 8, 64, 16) for k in (4, 8, 16, 32, 64)
                     for d in (8, 16, 32, 64)]
    for config in runs:
        assert config["lr"] == 0.03  # one rate for every counted run
        assert config["length_rule"] == "doubling" and config["first_look"] == 40000
        assert config["steps"] == 640000 and config["text_seed"] == 0
        assert config["name"] == (f"d{config['columns']}_c{config['creatures']}_b"
                                  f"{config['body']}_{config['usage']}_k{config['kinds']}"
                                  f"_s{config['seed']}")
        complete_config(config)  # accepted by the trainers
    for group in groups:
        assert {(c["creatures"], c["usage"], c["seed"]) for c in group} == {
            (n, u, s) for n in (16, 64, 256, 1024) for u in ("equal", "zipf")
            for s in (0, 1, 2)}
        assert len({group_key(complete_config(c)) for c in group}) == 1
    # Every tenth run of the fixed list is carried on, and no other.
    assert [i for i, c in enumerate(runs) if c.get("carry_on")] == list(range(0, 2880, 10))
    with pytest.raises(ValueError):
        grid.counted({"rates": {}})


def test_checkpoint_names_agree():
    """The orchestrator finds a group's checkpoint by the name the group trainer gives it."""
    for group in grid.groups(grid.practice())[:3] + grid.groups(grid.counted(SAME_RATE))[:3]:
        assert Orchestrator.checkpoint_folder(group) == group_name(
            [complete_config(c) for c in group])


# ------------------------------------------------------------------------------------------------
# The rate chooser
# ------------------------------------------------------------------------------------------------


def made_up_summaries(scores):
    """Practice summaries with made-up scores: scores(config) -> (facts, questions_one)."""
    found = {}
    for config in grid.practice():
        facts, one = scores(config)
        found[config["name"]] = {"name": config["name"], "config": config,
                                 "final": {"facts": facts, "questions_one": one}}
    return found


# The rate chooser is tested in v3/tests/test_v3_review.py.
