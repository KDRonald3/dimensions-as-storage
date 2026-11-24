"""Tests of v2/run.py: the same config gives the same result, and a stopped run picks up exactly
where it stopped."""

import json

import pytest
import torch

from v2 import run

CONFIG = {"creatures": 16, "kinds": 4, "usage": "zipf", "body": 8, "seed": 3, "steps": 150,
          "lr": 0.01, "threads": 1, "batch": 64}


@pytest.fixture
def short_checkpoints(monkeypatch):
    monkeypatch.setattr(run, "CHECKPOINT_EVERY", 50)


def weights(out_root, name):
    return torch.load(out_root / name / "model.pt", weights_only=False)


def without_time(summary):
    return {k: v for k, v in summary.items() if k not in ("minutes", "steps_per_second")}


def test_grading_steps():
    assert run.grading_steps(1000) == [100, 141, 200, 283, 400, 566, 800, 1000]
    assert run.grading_steps(150) == [100, 141, 150]
    assert run.grading_steps(20000)[-2:] == [18102, 20000]


def test_same_config_same_result_and_exact_resume(tmp_path, short_checkpoints):
    first = run.train(CONFIG, tmp_path / "a", quiet=True)
    second = run.train(CONFIG, tmp_path / "b", quiet=True)
    name = first["name"]
    assert name == "c16_b8_zipf_k4_s3"
    a, b = weights(tmp_path / "a", name), weights(tmp_path / "b", name)
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
    assert without_time(first) == without_time(second)

    # Stopped right after the checkpoint at step 100, then started again.
    assert run.train(CONFIG, tmp_path / "c", stop_after=100, quiet=True) is None
    assert (tmp_path / "c" / name / "checkpoint.pt").exists()
    assert not (tmp_path / "c" / name / "summary.json").exists()
    resumed = run.run_one(CONFIG, str(tmp_path / "c"))
    c = weights(tmp_path / "c", name)
    for key in a["model"]:
        assert torch.equal(a["model"][key], c["model"][key]), key
    assert without_time(first) == without_time(resumed)
    assert not (tmp_path / "c" / name / "checkpoint.pt").exists()

    # A finished run is returned as it is, without training.
    again = run.run_one(CONFIG, str(tmp_path / "c"))
    assert again == json.loads((tmp_path / "c" / name / "summary.json").read_text())


def test_summary_format(tmp_path):
    summary = run.run_one(dict(CONFIG, steps=100), str(tmp_path))
    assert list(summary) == ["name", "config", "parameters", "steps", "sentences", "minutes",
                             "steps_per_second", "final", "curve"]
    assert set(summary["config"]) == {"creatures", "kinds", "usage", "body", "seed", "steps",
                                      "lr", "threads", "table", "columns", "layers", "heads",
                                      "batch", "warmup", "average", "name"}
    assert summary["config"]["table"] == "trained"
    assert list(summary["parameters"]) == ["trained", "trained_without_table",
                                           "table_rows_in_use", "table_numbers", "body_middle"]
    assert set(summary["final"]) == {"facts", "facts_by_kind", "questions_one",
                                     "questions_new", "questions_new_detail", "questions_seen",
                                     "questions_seen_detail", "not_known", "loss"}
    assert set(summary["final"]["questions_new_detail"]) == {"yes", "no_one", "no_more",
                                                             "count"}
    assert summary["curve"][0]["step"] == 100 and summary["sentences"] == 100 * 64
    assert set(summary["curve"][0]) == {"step", "sentences", "facts", "questions_one",
                                        "questions_new", "questions_seen", "not_known", "loss"}
    saved = torch.load(tmp_path / summary["name"] / "model.pt", weights_only=False)
    assert saved["config"] == summary["config"]
    assert "word_table" in saved["model"]
