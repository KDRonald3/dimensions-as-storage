"""Tests of the plan changes made after the build (v3/PLAN.md B.2): 2 layers and 2 heads
everywhere, the stop evidence of "no more gain" runs, and the two rate-check stages."""

import json

import pytest
import torch

from v2.lang import Language
from v2.net import make_net
from v2.run import add_stop_evidence
from v3 import grid, orchestrate

SETTINGS = {"rate": 0.01}


def test_two_layers_and_two_heads_at_every_column_count():
    runs = grid.practice() + grid.counted(SETTINGS)
    assert all(c["layers"] == 2 and c["heads"] == 2 for c in runs)
    lang = Language(16, 4, "equal")
    for columns in grid.COLUMNS:
        net = make_net(lang, 4, columns, 2, 2, table_start="nested64")
        assert len(net.layers) == 2
        attention = net.layers[0].attention
        assert attention.heads == 2 and attention.output.in_features == columns


def test_stop_evidence_only_for_no_more_gain():
    curve = [{"step": s, "facts": f, "loss": l}
             for s, f, l in ((10, 40.0, 3.0), (20, 50.0, 2.0), (40, 50.5, 1.9), (80, 50.9, 1.85))]
    summary = {"curve": curve, "ended": {"step": 80, "reason": "no more gain"}}
    add_stop_evidence(summary)
    evidence = summary["stop_evidence"]
    assert (evidence["facts_quarter"], evidence["facts_half"], evidence["facts_whole"]) == (
        50.0, 50.5, 50.9)
    assert evidence["loss_fall"] == pytest.approx(1.9 - 1.85)
    for other in ({"curve": curve, "ended": {"step": 80, "reason": "learned"}}, {"curve": curve}):
        add_stop_evidence(other)
        assert "stop_evidence" not in other


def fake_counted(settings):
    """Made-up counted summaries: seed 1 best everywhere, except a tie on facts in the first
    setting, which goes to the better questions about one creature (seed 2)."""
    found = {}
    for i, config in enumerate(grid.counted(settings)):
        facts = 60.0 if config["seed"] == 1 else 50.0
        one = 70.0
        if i < 3:  # the first setting: seeds 0, 1, 2 with facts 60, 60, 60
            facts, one = 60.0, (80.0 if config["seed"] == 2 else 70.0)
        found[config["name"]] = {"final": {"facts": facts, "questions_one": one}}
    return found


def test_rate_check_runs():
    summaries = fake_counted(SETTINGS)
    for stage, divisor in grid.RATE_CHECKS.items():
        runs = grid.rate_check(SETTINGS, summaries, divisor)
        assert len(runs) == 960 and len({c["name"] for c in runs}) == 960
        assert all(c["lr"] == SETTINGS["rate"] / divisor for c in runs)
        assert runs[0]["seed"] == 2 and all(c["seed"] == 1 for c in runs[1:])
        assert all(c["name"].endswith(f"_lr{SETTINGS['rate'] / divisor:.4g}") for c in runs)
        assert all(c["length_rule"] == "doubling" and "carry_on" not in c for c in runs)
        assert sum(len(g) for g in grid.groups(runs)) == 960
    assert grid.best_seed({0: {"final": {"facts": 5, "questions_one": 1}},
                           1: {"final": {"facts": 5, "questions_one": 1}}}) == 0


def test_rate_check_refuses_before_every_counted_run_has_ended(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(json.dumps(SETTINGS))
    monkeypatch.setattr(grid, "SETTINGS_FILE", settings_file)
    summaries = fake_counted(SETTINGS)
    counted_dir = tmp_path / "runs" / "counted"
    names = list(summaries)
    for name in names[:-1]:  # all but one counted run
        (counted_dir / name).mkdir(parents=True)
        (counted_dir / name / "summary.json").write_text(json.dumps(summaries[name]))
    for stage in grid.RATE_CHECKS:
        with pytest.raises(SystemExit, match="cannot start"):
            orchestrate.stage_groups(stage, tmp_path / "runs")
    (counted_dir / names[-1]).mkdir(parents=True)
    (counted_dir / names[-1] / "summary.json").write_text(json.dumps(summaries[names[-1]]))
    groups = orchestrate.stage_groups("rate_check_third", tmp_path / "runs")
    assert sum(len(g) for g in groups) == 960


def test_sentences_of_many_steps_at_once_are_those_of_each_step():
    """The trainer makes the sentences of many steps in one pass: they must be exactly those of
    each step made alone (so a run's text depends only on its text seed, language and step)."""
    from v2.batch_text import BatchText
    languages = [Language(16, 4, "zipf"), Language(64, 4, "equal")]
    maker = BatchText(languages, [0, 3], batch=64, device="cpu")
    together = maker.make_steps(range(5, 9))
    for i, step in enumerate(range(5, 9)):
        assert torch.equal(together[i], maker.make(step))
