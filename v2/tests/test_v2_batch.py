"""Tests of the group trainer (v2/batch.py) and its sentence maker (v2/batch_text.py).

Everything runs on the processor and is kept small (the whole file takes a few minutes at most).

  1. Exact agreement with the reference trainer of v2/run.py, with the table trained and frozen.
  2. The sentence maker: every sentence true, no held-back pair, at most 13 tokens, and the same
     shares of everything as the reference sentence maker of v2/lang.py.
  3. A small group learns the facts in about 600 steps.
  4. summary.json has exactly the keys of a run_one summary, plus "trainer".
  5. A group stopped after a checkpoint and started again ends exactly as one never stopped.
  6. The batch of step t is the same when it is made twice.
"""

import copy
import json
import math

import numpy as np
import pytest
import torch
from torch.func import functional_call

from v2 import batch, run
from v2.batch_text import BatchText
from v2.lang import Language, is_true, read_sentence
from v2.net import make_net

torch.set_num_threads(4)  # the computer is busy with other work; keep to a few threads

# ------------------------------------------------------------------------------------------------
# 1. Exact agreement with the reference trainer
# ------------------------------------------------------------------------------------------------

EXACT_STEPS = 40
EXACT_MEMBERS = [(16, "equal", 0), (64, "zipf", 1), (256, "equal", 2)]


def key_bias_part(name, tensor, columns=32):
    """The part of a tensor that is the bias of the attention keys, or None.

    Why it is special: adding the same number to every key changes no attention weight (the
    softmax only sees differences), so the key bias has no effect at all on what the model
    computes, and its true gradient is exactly zero. What the computer gets instead is rounding
    noise of about 1e-10, and AdamW divides by the size of the gradient, so this noise moves the
    key bias by amounts that depend on the rounding, in the reference trainer as much as in the
    group trainer. It is checked separately, below."""
    if name.endswith("attention.query_key_value.bias"):
        return slice(columns, 2 * columns)
    return None


def reference_training(config, lang, batches, member):
    """Train one member with the reference code of v2/run.py (the same steps as run.train)."""
    torch.manual_seed(config["seed"])
    model = make_net(lang, config["body"], config["columns"], config["layers"], config["heads"],
                     config["table"])
    averaged = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=run.WEIGHT_DECAY)
    for step, tokens in enumerate(batches):
        inputs, targets = run.inputs_and_targets(lang, tokens[member])
        run.train_step(model, optimizer, inputs, targets, run.learning_rate_at(step, config))
        run.update_average(averaged, model, step, config)
    return model, averaged


@pytest.mark.parametrize("table", ["trained", "frozen"])
def test_exact_agreement_with_reference(tmp_path, table):
    configs = [{"creatures": n, "kinds": 4, "usage": u, "body": 8, "seed": s,
                "steps": EXACT_STEPS, "lr": 0.01, "threads": 1, "table": table}
               for n, u, s in EXACT_MEMBERS]
    full = [run.complete_config(c) for c in configs]
    languages = [Language(n, 4, u) for n, u, _ in EXACT_MEMBERS]
    # The batches of the reference sentence maker, member by member, fed to both trainers.
    batches = [np.stack([lang.sample_batch(s, step, 512)
                         for lang, (_, _, s) in zip(languages, EXACT_MEMBERS)])
               for step in range(EXACT_STEPS)]
    _, state = batch.train_group(configs, tmp_path, "cpu", quiet=True,
                                 batch_maker=lambda step: torch.from_numpy(batches[step]))

    probe = torch.from_numpy(batches[-1][:, :64, :-1].copy())  # some token rows to compare on
    for i, (config, lang) in enumerate(zip(full, languages)):
        model, averaged = reference_training(config, lang, batches, i)
        for copy_name, ours, theirs in (("weights", state["params"], model),
                                        ("averaged", state["averaged"], averaged)):
            names = [name for name, _ in theirs.named_parameters()]
            assert set(names) == set(ours)
            assert ("word_table" in names) == (table == "trained")
            for name, reference in theirs.named_parameters():
                mine = ours[name][i]
                assert mine.shape == reference.shape
                key = key_bias_part(name, mine)
                if key is None:
                    assert torch.allclose(mine, reference, rtol=0, atol=1e-5), (copy_name, name)
                    continue
                keep = torch.ones(mine.shape[0], dtype=torch.bool)
                keep[key] = False
                assert torch.allclose(mine[keep], reference[keep], rtol=0, atol=1e-5), name
                # The key bias only moves by rounding noise (see key_bias_part) ...
                assert (mine[key] - reference[key]).abs().max() < 1e-3, name
        # ... and has no effect: the averaged copies give the same scores.
        with torch.no_grad():
            predicted = functional_call(
                averaged, {k: v[i] for k, v in state["averaged"].items()}, (probe[i],))
            assert torch.allclose(predicted, averaged(probe[i]), rtol=0, atol=1e-5)
        # The frozen table never moves; nor do the rows of creatures not in use.
        start = make_net(lang, 8).word_table.detach()
        if table == "frozen":
            assert torch.equal(state["buffers"]["word_table"][i], start)
        else:
            unused = slice(lang.rows_in_use, None)
            assert torch.equal(state["params"]["word_table"][i][unused], start[unused])
            assert torch.equal(state["averaged"]["word_table"][i][unused], start[unused])


# ------------------------------------------------------------------------------------------------
# 2. The sentence maker
# ------------------------------------------------------------------------------------------------

SAMPLE_STEPS = 40  # 40 batches of 512 = 20,480 sentences per setting
SAMPLE_SETTINGS = [(16, "equal"), (16, "zipf"), (256, "equal"), (256, "zipf")]
SMALLEST_P = 1e-4  # a share test fails only when a difference this unlikely is seen


def chi_square_p(counts_a, counts_b):
    """The chance of a difference at least as large as the one seen between two sets of counts,
    if both come from the same shares (the chi-square test of homogeneity). Categories with too
    few expected counts are put together first. The chance is found with the Wilson-Hilferty
    rule (accurate to a few per cent of itself, which is enough here)."""
    a, b = np.asarray(counts_a, float), np.asarray(counts_b, float)
    total = a + b
    order = np.argsort(total)
    groups, current, size = [], [], 0.0
    expected_min = lambda t: min(t * a.sum(), t * b.sum()) / (a.sum() + b.sum())
    for i in order:  # put the smallest categories together until each has enough
        current.append(i)
        size += total[i]
        if expected_min(size) >= 5:
            groups.append(current)
            current, size = [], 0.0
    if current:
        if groups:
            groups[-1] += current
        else:
            groups.append(current)
    a = np.array([a[g].sum() for g in groups])
    b = np.array([b[g].sum() for g in groups])
    if len(a) < 2:
        return 1.0
    share = (a + b) / (a.sum() + b.sum())
    statistic = (((a - share * a.sum()) ** 2) / (share * a.sum())).sum() + \
                (((b - share * b.sum()) ** 2) / (share * b.sum())).sum()
    df = len(a) - 1
    z = ((statistic / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def tally(lang, sentences):
    """Counts of everything the shares are compared on, from sentences read by the separate
    reader of v2/lang.py."""
    counts = {"type": np.zeros(4), "length": np.zeros(4), "type x length": np.zeros(16),
              "kind": np.zeros(lang.kinds), "type x value": np.zeros(16),
              "length x f": np.zeros(16), "creature": np.zeros(lang.creatures)}
    for ids in sentences:
        read = read_sentence(lang, ids)
        t, k = read["type"], len(read["creatures"])
        counts["type"][t] += 1
        counts["length"][k - 1] += 1
        counts["type x length"][4 * t + k - 1] += 1
        counts["kind"][read["kind"]] += 1
        counts["type x value"][4 * t + read["value"]] += 1
        if read["answer"] == "no":
            f = sum(lang.facts[c, read["kind"]] != read["value"] for c in read["creatures"])
            counts["length x f"][4 * (k - 1) + f - 1] += 1
        for c in read["creatures"]:
            counts["creature"][c] += 1
    return counts


@pytest.mark.parametrize("kinds", [4, 16])
def test_sentence_maker_matches_reference(kinds):
    languages = [Language(n, kinds, u) for n, u in SAMPLE_SETTINGS]
    maker = BatchText(languages, [7] * len(languages), batch=512, device="cpu")
    made = torch.cat([maker.make(step) for step in range(SAMPLE_STEPS)], dim=1).numpy()
    assert made.shape == (len(languages), SAMPLE_STEPS * 512, 13)  # never longer than 13
    failures = []
    for m, lang in enumerate(languages):
        ours = made[m]
        reference = lang.draw_sentences(np.random.default_rng([99, m, kinds]),
                                        len(ours))["tokens"]
        # Every sentence is a complete, true sentence (so nothing was cut off at 13 tokens),
        # by the separate truth check of v2/lang.py ...
        assert all(is_true(lang, row) for row in ours.tolist())
        # ... and none names both creatures of a held-back pair.
        creatures = np.where(ours >= lang.creature_base, ours - lang.creature_base, -1)
        for row in creatures:
            named = row[row >= 0]
            assert not lang.held[np.ix_(named, named)].any()
        ours_counts, their_counts = tally(lang, ours.tolist()), tally(lang, reference.tolist())
        for what in ours_counts:
            p = chi_square_p(ours_counts[what], their_counts[what])
            if p < SMALLEST_P:
                failures.append(f"{lang.creatures} {lang.usage} kinds {kinds}: {what} "
                                f"(p = {p:.2g})")
        if lang.creatures == 16:
            # The awkward case: with 16 creatures many lists of 4 cannot be made, so they are
            # made shorter: fewer than the planned 20% of lists have 4 creatures. The share
            # tests above check that both makers agree on how many fewer.
            share = lambda c: c["length"][3] / c["length"].sum()
            assert share(ours_counts) < 0.18 and share(their_counts) < 0.18
    assert not failures, failures


def test_same_batch_when_made_twice():
    languages = [Language(16, 4, "zipf"), Language(64, 4, "equal")]
    maker = BatchText(languages, [3, 3], batch=128, device="cpu")
    assert torch.equal(maker.make(5), maker.make(5))
    again = BatchText(languages, [3, 3], batch=128, device="cpu")  # a new maker, as after a restart
    assert torch.equal(maker.make(5), again.make(5))
    assert not torch.equal(maker.make(5), maker.make(6))
    other_seeds = BatchText(languages, [4, 4], batch=128, device="cpu")
    assert not torch.equal(maker.make(5), other_seeds.make(5))


# ------------------------------------------------------------------------------------------------
# 3 and 4. Learning, and the summary format
# ------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def learned(tmp_path_factory):
    out_root = tmp_path_factory.mktemp("learned")
    configs = [{"creatures": n, "kinds": 4, "usage": "equal", "body": 32, "seed": 0,
                "steps": 600, "lr": 0.01, "threads": 1} for n in (16, 64)]
    return batch.run_group(configs, str(out_root), "cpu"), out_root


def test_small_group_learns(learned):
    summaries, _ = learned
    # At least 97 (the "learned" mark of v2/analyze.py), not exactly 100: the reference trainer
    # of v2/run.py itself ends at 99.2 for 64 creatures with seed 0 after 600 steps.
    assert all(s["final"]["facts"] >= 97 for s in summaries), [s["final"] for s in summaries]


def test_summary_has_the_keys_of_run_one(learned):
    summaries, out_root = learned
    for summary in summaries:
        on_disk = json.loads((out_root / summary["name"] / "summary.json").read_text())
        assert on_disk == summary
        assert list(summary) == ["name", "config", "parameters", "steps", "sentences",
                                 "minutes", "steps_per_second", "final", "curve", "trainer"]
        assert summary["trainer"] == "batch"
        assert summary["config"] == run.complete_config(summary["config"])
        assert set(summary["final"]) == {"facts", "facts_by_kind", "questions_one",
                                         "questions_new", "questions_new_detail",
                                         "questions_seen", "questions_seen_detail",
                                         "not_known", "loss"}
        assert set(summary["curve"][0]) == {"step", "sentences", "facts", "questions_one",
                                            "questions_new", "questions_seen", "not_known",
                                            "loss"}
        assert [p["step"] for p in summary["curve"]] == run.grading_steps(600)
        saved = torch.load(out_root / summary["name"] / "model.pt", weights_only=False)
        assert saved["config"] == summary["config"]
        # model.pt holds one member only, in the form run_one saves.
        lang = Language(summary["config"]["creatures"], 4, "equal")
        reference = make_net(lang, 32).state_dict()
        assert list(saved["model"]) == list(reference)
        for name, tensor in saved["model"].items():
            assert tensor.shape == reference[name].shape
            assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


# ------------------------------------------------------------------------------------------------
# 5. Stopping and picking up again
# ------------------------------------------------------------------------------------------------

RESUME_CONFIGS = [{"creatures": n, "kinds": 4, "usage": u, "body": 8, "seed": s, "steps": 60,
                   "lr": 0.01, "threads": 1, "batch": 64}
                  for n, u, s in ((16, "equal", 0), (16, "zipf", 1), (64, "equal", 2))]


def without_time(summary):
    return {k: v for k, v in summary.items() if k not in ("minutes", "steps_per_second")}


def test_stop_and_pick_up_again(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "CHECKPOINT_EVERY", 20)
    whole, whole_state = batch.train_group(RESUME_CONFIGS, tmp_path / "a", "cpu", quiet=True)

    stopped, _ = batch.train_group(RESUME_CONFIGS, tmp_path / "b", "cpu", stop_after=20,
                                   quiet=True)
    assert stopped is None
    assert list((tmp_path / "b" / "_groups").glob("*/checkpoint.pt"))
    resumed, resumed_state = batch.train_group(RESUME_CONFIGS, tmp_path / "b", "cpu",
                                               quiet=True)
    assert not list((tmp_path / "b" / "_groups").glob("*/checkpoint.pt"))
    for name in whole_state["params"]:
        assert torch.equal(whole_state["params"][name], resumed_state["params"][name]), name
        assert torch.equal(whole_state["averaged"][name], resumed_state["averaged"][name]), name
    assert [without_time(s) for s in whole] == [without_time(s) for s in resumed]
    for summary in whole:
        a = torch.load(tmp_path / "a" / summary["name"] / "model.pt", weights_only=False)
        b = torch.load(tmp_path / "b" / summary["name"] / "model.pt", weights_only=False)
        for key in a["model"]:
            assert torch.equal(a["model"][key], b["model"][key]), key

    # Finished members are returned as they are, without training.
    again, state = batch.train_group(RESUME_CONFIGS, tmp_path / "b", "cpu", quiet=True)
    assert state is None and again == resumed


def test_make_groups():
    configs = RESUME_CONFIGS + [dict(RESUME_CONFIGS[0], body=16),
                                dict(RESUME_CONFIGS[0], table="frozen")]
    groups = batch.make_groups(configs)
    assert [len(g) for g in groups] == [3, 1, 1]
    with pytest.raises(ValueError):
        batch.train_group(configs, "unused", "cpu", quiet=True)
