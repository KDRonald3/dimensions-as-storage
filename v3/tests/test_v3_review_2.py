"""Tests added by the second review (23 November)."""

import json

import torch

from v2.batch_text import GRID, BatchText
from v2.lang import Language
from v3 import orchestrate


def test_line_is_on_the_grid_and_text_does_not_depend_on_place_in_group():
    """Every end of a piece of the line is a whole number of grid steps, so a member's draws
    are exact wherever it lies on the line; and its text is the same alone and as the 17th
    member of a group."""
    target = Language(1024, 64, "zipf")
    others = [Language(c, 64, u) for c in (16, 64, 256, 1024) for u in ("equal", "zipf")] * 2
    crowd = BatchText(others + [target], list(range(1, 17)) + [0], 64)
    steps = (crowd.piece_high * GRID)
    assert torch.equal(steps, steps.round())
    alone = BatchText([target], [0], 64).make_steps(range(3))[:, 0]
    assert torch.equal(alone, crowd.make_steps(range(3))[:, 16])


class StuckKaggle:
    """A pretend Kaggle on which one session stays queued."""
    user = "someone"

    def hide(self, text):
        return text

    def status(self, name):
        return "queued"


def test_a_session_waiting_very_long_is_reported_once(tmp_path):
    orchestrator = orchestrate.Orchestrator("test", StuckKaggle(), tmp_path / "runs",
                                            tmp_path / "logs", 0.1, 1, 0, quiet=True)
    sent = orchestrate.time.time() - (orchestrate.LONG_WAIT_HOURS + 1) * 3600
    orchestrator.state["sessions"]["air-v3-test-1"] = {
        "stage": "test", "groups": [0], "hours": 0.1, "state": "running", "pushed": sent}
    for _ in range(3):
        orchestrator.refresh([], {"groups": []})
    log = (tmp_path / "logs" / "orchestrate.log").read_text()
    assert log.count("still queued") == 1
    saved = json.loads((tmp_path / "logs" / "state.json").read_text())
    assert saved["sessions"]["air-v3-test-1"]["state"] == "running"  # still watched, not resent
