"""Tests of v2/net.py and the loss of v2/run.py: sizes, both forms of the word table, the
targets."""

import numpy as np
import pytest
import torch

from v2 import lang as L
from v2 import net, run


@pytest.mark.parametrize("body", [8, 128])
@pytest.mark.parametrize("kinds", [4, 32])
@pytest.mark.parametrize("table_form", ["trained", "frozen"])
def test_trained_numbers_do_not_depend_on_creatures(body, kinds, table_form):
    without_table, frozen_totals = [], []
    for creatures in (16, 64, 256, 1024):
        lang = L.Language(creatures, kinds)
        torch.manual_seed(0)
        model = net.make_net(lang, body, table_form=table_form)
        counts = net.count_numbers(model, lang)
        without_table.append(counts["trained_without_table"])
        assert counts["table_rows_in_use"] == lang.creature_base + creatures
        assert counts["table_numbers"] == 32 * counts["table_rows_in_use"]
        names = [name for name, _ in model.named_parameters()]
        if table_form == "frozen":
            assert "word_table" not in names
            assert "word_table" in dict(model.named_buffers())
            assert counts["trained"] == counts["trained_without_table"]
            frozen_totals.append(counts["trained"])
        else:
            assert "word_table" in names
            # It grows by 32 numbers for each word in use.
            assert counts["trained"] == counts["trained_without_table"] + 32 * (
                lang.creature_base + creatures)
        # Counted by hand: everything except the word table.
        assert counts["trained_without_table"] == sum(
            p.numel() for n, p in model.named_parameters() if n != "word_table")
    assert len(set(without_table)) == 1
    assert len(set(frozen_totals)) <= 1


def test_same_seed_gives_same_starting_values_for_every_vocabulary():
    torch.manual_seed(7)
    small = net.make_net(L.Language(16, 4), 32)
    torch.manual_seed(7)
    large = net.make_net(L.Language(1024, 4), 32)
    for (name, a), (_, b) in zip(small.named_parameters(), large.named_parameters()):
        assert torch.equal(a, b), name


def test_word_table_is_frozen_and_not_in_the_optimizer():
    lang = L.Language(64, 4)
    torch.manual_seed(0)
    model = net.make_net(lang, 16, table_form="frozen")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    in_optimizer = [p for group in optimizer.param_groups for p in group["params"]]
    assert all(p is not model.word_table for p in in_optimizer)
    assert all(p.data_ptr() != model.word_table.data_ptr() for p in in_optimizer)
    assert not model.word_table.requires_grad
    before = model.word_table.clone()
    weights_before = model.output.weight.clone()
    config = run.complete_config({"creatures": 64, "kinds": 4, "usage": "equal", "body": 16,
                                  "seed": 0, "steps": 20, "lr": 0.01, "threads": 1})
    for step in range(20):
        inputs, targets = run.inputs_and_targets(lang, lang.sample_batch(0, step, 64))
        run.train_step(model, optimizer, inputs, targets, run.learning_rate_at(step, config))
    assert torch.equal(model.word_table, before)  # bit for bit
    assert not torch.equal(model.output.weight, weights_before)  # the rest did train


@pytest.mark.parametrize("word", ["creature 5", "red", "and", "pools-if-used"])
def test_a_word_has_the_same_row_everywhere(word):
    rows = []
    for creatures in (16, 64, 256, 1024):
        for kinds in (4, 8, 16, 32):
            lang = L.Language(creatures, kinds)
            table = net.word_table(lang)
            assert table.shape == (lang.n_tokens, 32)
            if word == "creature 5":
                token = lang.creature_base + 5
            elif word == "pools-if-used":
                if kinds < 16:
                    continue
                token = lang.word_id["pools"]
            else:
                token = lang.word_id[word]
            rows.append(table[token])
    assert all(np.array_equal(rows[0], row) for row in rows)
    assert abs(float(net.full_word_table().std()) - 1.0) < 0.05


def test_output_layer_cannot_say_a_creature():
    lang = L.Language(256, 8)
    model = net.make_net(lang, 8)
    scores = model(torch.from_numpy(lang.sample_batch(0, 0, 4)))
    assert scores.shape == (4, 13, lang.n_out)
    assert lang.n_out == lang.creature_base - 1  # every word except <pad> and the creatures


def test_no_loss_on_padding_or_creature_names():
    lang = L.Language(64, 4, "zipf")
    tokens = lang.sample_batch(1, 2, 512)
    inputs, targets = run.inputs_and_targets(lang, tokens)
    inputs, targets = inputs.numpy(), targets.numpy()
    assert np.array_equal(inputs, tokens[:, :-1])
    following = tokens[:, 1:]
    no_loss = (following == L.PAD_ID) | (following >= lang.creature_base)
    assert (targets[no_loss] == run.IGNORE).all()
    assert (targets[~no_loss] == following[~no_loss] - 1).all()
    assert no_loss.any() and (following >= lang.creature_base).any()
    assert (targets[~no_loss] < lang.n_out).all()


def test_sizes_of_the_body():
    lang = L.Language(16, 4)
    for body in (8, 16, 32, 64, 128):
        counts = net.count_numbers(net.make_net(lang, body), lang)
        assert counts["body_middle"] == 2 * (32 * body + body + body * 32 + 32)
