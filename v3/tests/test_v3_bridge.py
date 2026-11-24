"""The bridging stage (v3/grid.py, bridge): the body widths really give the narrower model the size
of the wider one, and the runs are the counted runs' protocol with only the body changed.

    .venv/bin/python -m pytest v3/tests/test_v3_bridge.py -q
"""
import json
from pathlib import Path

import pytest

from v2.lang import Language
from v2.net import count_numbers, make_net
from v2.run import complete_config
from v3 import grid
from v3.orchestrate import stage_groups

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = grid.read_settings()


def trained(columns, body, kinds):
    lang = Language(grid.BRIDGE_CREATURES, kinds, grid.BRIDGE_USAGE)
    model = make_net(lang, body, columns, grid.LAYERS, grid.HEADS, "trained", "nested64")
    return count_numbers(model, lang)["trained"]


@pytest.mark.parametrize("kinds,columns", sorted(grid.BRIDGE_BODIES))
def test_body_gives_the_size_of_the_wider_model(kinds, columns):
    body = grid.BRIDGE_BODIES[(kinds, columns)]
    target = trained(2 * columns, 4, kinds)
    miss = abs(trained(columns, body, kinds) - target)
    # the closest whole number: one neuron more or less is further away
    assert miss <= abs(trained(columns, body - 1, kinds) - target)
    assert miss <= abs(trained(columns, body + 1, kinds) - target)
    assert miss / target < 0.001


@pytest.mark.parametrize("kinds,columns", sorted(grid.BRIDGE_BODIES))
def test_target_is_the_size_the_counted_run_recorded(kinds, columns):
    """The wider model's size here is the size its counted runs wrote in their summaries."""
    for seed in grid.SEEDS:
        name = f"d{2 * columns}_c1024_b4_equal_k{kinds}_s{seed}"
        summary = json.loads((ROOT / "runs_v3" / "counted" / name / "summary.json").read_text())
        assert summary["parameters"]["trained"] == trained(2 * columns, 4, kinds)


def test_word_use_does_not_change_the_size():
    """Only equal use is run; check that the size would be the same with unequal use."""
    for (kinds, columns), body in grid.BRIDGE_BODIES.items():
        sizes = set()
        for usage in grid.USAGES:
            lang = Language(grid.BRIDGE_CREATURES, kinds, usage)
            sizes.add(count_numbers(make_net(lang, body, columns, 2, 2, "trained", "nested64"),
                                    lang)["trained"])
        assert len(sizes) == 1


def test_runs_follow_the_counted_protocol():
    runs = grid.bridge(SETTINGS)
    assert len(runs) == 45
    assert len({r["name"] for r in runs}) == 45
    counted = {r["name"]: r for r in grid.counted(SETTINGS)}
    assert not {r["name"] for r in runs} & set(counted)  # never the name of a counted run
    for run in runs:
        # the counted run of the same language, columns, kinds and seed, with the body of 4
        twin = dict(counted[f"d{run['columns']}_c1024_b4_equal_k{run['kinds']}_s{run['seed']}"])
        twin.pop("carry_on", None)
        assert run["carry_on"] is True  # every bridging run is carried on
        differ = {key for key in set(run) | set(twin) if run.get(key) != twin.get(key)}
        assert differ == {"body", "name", "carry_on"}
        assert run["body"] == grid.BRIDGE_BODIES[(run["kinds"], run["columns"])]
        assert run["lr"] == SETTINGS["rate"] == 0.003
        assert run["steps"] == 640000 and run["first_look"] == 40000
        assert run["length_rule"] == "doubling"
        complete_config(run)  # the trainer accepts it


def test_groups():
    groups = stage_groups("bridge")
    assert len(groups) == 15
    for group in groups:
        assert sorted(c["seed"] for c in group) == [0, 1, 2]
        assert len({(c["columns"], c["kinds"], c["body"]) for c in group}) == 1
    # the longest first: 128 bits (kinds 64), 32 columns
    assert (groups[0][0]["kinds"], groups[0][0]["columns"]) == (64, 32)
