"""Tests of v3/session_job.py on the processor: two workers at once, stopping on the budget, and
a second session that picks up from the first one's output (as Kaggle attaches it)."""

import json
import shutil

from v3 import session_job


def tiny_groups():
    """Two tiny groups of two members (mixed rates in the first, one rate in the second)."""
    groups = []
    for columns, rates in ((8, (0.01, 0.03)), (16, (0.01, 0.01))):
        group = []
        for seed, lr in enumerate(rates):
            group.append({"creatures": 16, "kinds": 4, "usage": "equal", "body": 4,
                          "seed": seed, "steps": 160, "lr": lr, "threads": 1, "batch": 32,
                          "columns": columns, "text_seed": 0, "table_start": "nested64",
                          "length_rule": "doubling", "first_look": 40,
                          "name": f"t_d{columns}_s{seed}"})
        groups.append(group)
    return groups


def without_time(summary):
    return {k: v for k, v in summary.items() if k not in ("minutes", "steps_per_second")}


def summaries(working):
    return {p.parent.name: without_time(json.loads(p.read_text()))
            for p in (working / "out" / "test").glob("*/summary.json")}


def test_session_two_workers_stop_and_pick_up(tmp_path):
    groups = tiny_groups()
    names = {c["name"] for g in groups for c in g}

    # A whole session with two workers on the processor.
    whole = session_job.run_session(groups, "test", 5, input_root=tmp_path / "none",
                                    working=tmp_path / "whole", devices=["cpu", "cpu"],
                                    end_reserve=0, min_start=1, safety=0)
    assert set(whole["finished"]) == names and not whole["unfinished_groups"]
    reports = [json.loads(p.read_text())
               for p in (tmp_path / "whole" / "out" / "_reports").glob("*.json")]
    assert sorted(r["worker"] for r in reports) == [0, 1]  # each worker took one group
    assert set(whole["speed"]) == {"cpu:0", "cpu:1"}
    assert all(v["per_second"] > 0 for v in whole["speed"].values())
    saved = json.loads((tmp_path / "whole" / "out" / "status.json").read_text())
    assert saved["finished"] == whole["finished"]

    # A session whose budget is used up before the groups are done: it stops cleanly, with a
    # checkpoint for every group.
    short = session_job.run_session(groups, "test", 0.3, input_root=tmp_path / "none",
                                    working=tmp_path / "first", devices=["cpu", "cpu"],
                                    end_reserve=0, min_start=1, safety=0)
    assert len(short["unfinished_groups"]) == 2
    assert all(g["checkpoint"] for g in short["unfinished_groups"])
    assert not short["failed_groups"]

    # The next session gets the first one's output under its input folder (as Kaggle attaches
    # an earlier session) and ends exactly as the whole session did.
    attached = tmp_path / "input" / "air-v3-test-1"
    shutil.copytree(tmp_path / "first", attached)
    second = session_job.run_session(groups, "test", 5, input_root=tmp_path / "input",
                                     working=tmp_path / "second", devices=["cpu"],
                                     end_reserve=0, min_start=1, safety=0)
    assert set(second["finished"]) == names
    log = (tmp_path / "second" / "out" / "logs" / "worker0.log").read_text()
    assert "going on from the checkpoint" in log
    assert summaries(tmp_path / "second") == summaries(tmp_path / "whole")
