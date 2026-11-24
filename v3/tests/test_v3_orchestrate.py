"""Tests of v3/orchestrate.py against a pretend Kaggle tool (v3/tests/fake_kaggle.py).

The pretend tool really runs each session's script on this computer (on the processor), so
these tests go through everything: packing the code, the session job, downloading, copying the
results, and sending the unfinished groups again with the earlier session attached.
"""

import json
import sys
from pathlib import Path

import pytest

from v3 import orchestrate
from v3.tests.test_v3_session import tiny_groups

FAKE = Path(__file__).with_name("fake_kaggle.py")


class Stop(Exception):
    """Raised by the pretend sleep to stop the orchestrator, as Ctrl+C would."""


def three_groups():
    groups = tiny_groups()
    extra = [dict(c, columns=32, name=c["name"].replace("t_d8", "t_d32")) for c in groups[0]]
    return groups + [extra]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A pretend Kaggle tool with its state file; returns (tool path, state reader/writer)."""
    state_path = tmp_path / "fake" / "state.json"
    state_path.parent.mkdir()
    tool = tmp_path / "fake" / "kaggle"
    tool.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE} \"$@\"\n")
    tool.chmod(0o755)
    monkeypatch.setenv("FAKE_KAGGLE_STATE", str(state_path))
    monkeypatch.setattr(orchestrate, "stage_groups", lambda stage, *rest: three_groups())
    # One group per card, so that the three groups need two sessions (as these tests expect).
    monkeypatch.setattr(orchestrate, "GROUPS_PER_CARD", 1)

    def state(**changes):
        current = json.loads(state_path.read_text()) if state_path.exists() else {
            "quota_remaining": 29.7, "max_running": 1, "fail": [], "running_polls": 1,
            "kernels": {}, "pushes": [], "refusals": 0}
        current.update(changes)
        state_path.write_text(json.dumps(current))
        return current

    return tool, state


def arguments(tmp_path, tool, *more):
    return ["test", "--kaggle", str(tool), "--runs-root", str(tmp_path / "runs"),
            "--logs-dir", str(tmp_path / "logs"), "--speed", "1", "--hours", "0.1",
            "--poll-seconds", "0", *more]


def kaggle(tool):
    return orchestrate.Kaggle(tool, user="someone")


def test_sessions_refused_failed_and_restart(tmp_path, fake):
    tool, state = fake
    state(fail=["air-v3-test-1"])  # the first session breaks down
    calls = []

    def sleep_then_stop(seconds):
        calls.append(seconds)
        if len(calls) == 3:  # after the first session failed and a second one was sent
            raise Stop()  # the owner stops the program

    with pytest.raises(Stop):
        orchestrate.main(arguments(tmp_path, tool), kaggle=kaggle(tool), sleep=sleep_then_stop)
    saved = json.loads((tmp_path / "logs" / "state.json").read_text())
    assert saved["sessions"]["air-v3-test-1"]["state"] == "failed"
    assert saved["stages"]["test"]["groups"][0]["failures"] == 1

    # Started again: it goes on from logs/v3/state.json and finishes the stage.
    result = orchestrate.main(arguments(tmp_path, tool), kaggle=kaggle(tool),
                              sleep=lambda s: None)
    assert result == "done"
    final = state()
    names = [p["name"] for p in final["pushes"]]
    assert len(names) == len(set(names))  # no session was sent twice
    assert final["refusals"] >= 1  # Kaggle refused while another session was running
    runs = {p.parent.name for p in (tmp_path / "runs" / "test").glob("*/summary.json")}
    assert runs == {c["name"] for g in three_groups() for c in g}
    assert all((tmp_path / "runs" / "test" / name / "model.pt").exists() for name in runs)
    # Every session asked for its time and the quota was respected.
    assert all(p["seconds"] == 0.1 * 3600 for p in final["pushes"])
    log = (tmp_path / "logs" / "orchestrate.log").read_text()
    assert "did not take" in log and "someone" not in log
    # The status command reads it all back.
    orchestrate.main(["status", "--kaggle", str(tool), "--runs-root", str(tmp_path / "runs"),
                      "--logs-dir", str(tmp_path / "logs")], kaggle=kaggle(tool))


def test_quota_running_out(tmp_path, fake):
    tool, state = fake
    state(quota_remaining=0.62)  # room for one session of 0.1 hours, then not
    result = orchestrate.main(arguments(tmp_path, tool), kaggle=kaggle(tool),
                              sleep=lambda s: None)
    assert result == "quota"
    assert len(state()["pushes"]) == 1
    assert state()["quota_remaining"] >= orchestrate.QUOTA_MARGIN_HOURS - 1e-9
    assert "quota is used up" in (tmp_path / "logs" / "orchestrate.log").read_text()


def test_pick_up_through_an_attached_session(tmp_path, fake, monkeypatch):
    """A session whose groups are not finished: the next session has it attached as a kernel
    source and goes on from its checkpoints."""
    tool, state = fake
    state(max_running=2)
    # Sessions so short that the first one stops at once, with checkpoints; then normal ones.
    sizes = iter([0.004] + [0.1] * 10)
    real_write = orchestrate.Orchestrator.write_session

    def write_session(self, name, indexes, groups, sources, hours):
        return real_write(self, name, indexes, groups, sources, next(sizes))

    monkeypatch.setattr(orchestrate.Orchestrator, "write_session", write_session)
    monkeypatch.setattr(orchestrate, "stage_groups", lambda stage, *rest: tiny_groups())
    result = orchestrate.main(arguments(tmp_path, tool), kaggle=kaggle(tool),
                              sleep=lambda s: None)
    assert result == "done"
    pushes = state()["pushes"]
    assert pushes[0]["sources"] == []
    assert pushes[1]["sources"] == ["someone/air-v3-test-1"]
    saved = json.loads((tmp_path / "logs" / "state.json").read_text())
    assert saved["stages"]["test"]["groups"][0]["source"] == "air-v3-test-1"
    second_log = (tmp_path / "fake" / "kernels" / "air-v3-test-2" / "log.txt").read_text()
    assert "brought in 0 finished runs and 2 checkpoints" in second_log
    assert "going on from the checkpoint" in second_log
