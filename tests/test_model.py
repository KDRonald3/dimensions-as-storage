"""Tests of the tiny GPT in model.py. Nothing here trains a model.

Run them with:

    pytest tests/test_model.py

They check the promises the experiment relies on: the exact number of
parameters, that the word table is tied, that the model cannot look ahead,
and that padding is ignored by the loss.
"""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Let "import model" find the files at the top of the repository, however
# pytest was started.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import ModelConfig, TinyGPT, count_parameters  # noqa: E402


# The vocabulary of the largest language: 27 fixed words + 1,024 animals.
LARGE_VOCAB = 1051

# Body parameters of a 2-layer model, as promised in the specification.
BODY_PARAMETERS = {4: 880, 8: 2_528, 16: 8_128, 32: 28_544, 64: 106_240}


def make_model(vocab_size=50, width=8, seed=0):
    """A small model with fixed starting weights, so tests are repeatable."""
    torch.manual_seed(seed)
    return TinyGPT(ModelConfig(vocab_size=vocab_size, width=width))


def random_tokens(vocab_size, batch, time, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size, (batch, time), generator=generator)


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("vocab_size", [43, LARGE_VOCAB])
def test_body_parameter_counts_are_exact(width, vocab_size):
    # The same number for the small and the large vocabulary: the body does
    # not depend on how many words there are.
    model = TinyGPT(ModelConfig(vocab_size=vocab_size, width=width, layers=2))
    assert count_parameters(model)["body"] == BODY_PARAMETERS[width]


@pytest.mark.parametrize("width", [4, 32])
def test_count_parameters_reports_body_table_and_total(width):
    model = TinyGPT(ModelConfig(vocab_size=LARGE_VOCAB, width=width))
    counts = count_parameters(model)

    assert set(counts) == {"body", "table", "total"}
    assert counts["table"] == LARGE_VOCAB * width
    assert counts["total"] == counts["body"] + counts["table"]

    # Count again in a different way: add up every tensor by name.
    table = 0
    body = 0
    for name, parameter in model.named_parameters():
        if name == "table.weight":
            table += parameter.numel()
        else:
            body += parameter.numel()
    assert counts["table"] == table
    assert counts["body"] == body


def test_one_more_layer_adds_one_more_block():
    # Each block of width w has 12 w^2 + 13 w parameters.
    two = count_parameters(TinyGPT(ModelConfig(vocab_size=50, width=8, layers=2)))
    three = count_parameters(TinyGPT(ModelConfig(vocab_size=50, width=8, layers=3)))
    assert three["body"] - two["body"] == 12 * 8 * 8 + 13 * 8
    assert three["table"] == two["table"]


# ---------------------------------------------------------------------------
# The tied word table
# ---------------------------------------------------------------------------


def test_only_one_tensor_depends_on_the_vocabulary():
    # 50 is not a size that any other tensor could have at width 8
    # (those are 8, 24, 32 and 96), so the search below cannot be fooled.
    model = make_model(vocab_size=50, width=8)
    with_vocab_size = [name for name, parameter in model.named_parameters()
                       if 50 in parameter.shape]
    assert with_vocab_size == ["table.weight"]
    assert tuple(model.table.weight.shape) == (50, 8)


def test_table_row_is_used_on_the_way_out():
    model = make_model(vocab_size=50, width=8)
    tokens = random_tokens(10, batch=3, time=12)      # only words 0 to 9 are read
    before = model(tokens).detach()

    # Word 20 is never read here, so its row can only matter on the way out.
    with torch.no_grad():
        model.table.weight[20] = 0.0
    after = model(tokens).detach()

    others = [word for word in range(50) if word != 20]
    assert before[:, :, 20].abs().max() > 1e-4          # it had a score before
    assert after[:, :, 20].abs().max() == 0             # an all-zero row scores exactly 0
    assert torch.allclose(after[:, :, others], before[:, :, others], atol=1e-6)


def test_table_row_is_used_on_the_way_in():
    model = make_model(vocab_size=50, width=8)
    tokens = torch.full((2, 6), 3)                     # the model reads word 3 only
    before = model(tokens).detach()

    # Change ONE number of the row. (Adding the same amount to all eight would
    # change nothing: a layer norm subtracts the average of a vector first.)
    with torch.no_grad():
        model.table.weight[3, 0] += 1.0
    after = model(tokens).detach()

    # The scores of OTHER words change too. That can only happen if the row of
    # word 3 was read on the way in, because their own rows were not touched.
    others = [word for word in range(50) if word != 3]
    assert (after[:, :, others] - before[:, :, others]).abs().max() > 1e-4


# ---------------------------------------------------------------------------
# Shapes and limits
# ---------------------------------------------------------------------------


def test_output_shape():
    model = make_model(vocab_size=50, width=8)
    logits = model(random_tokens(50, batch=3, time=17))
    assert logits.shape == (3, 17, 50)

    # A single word and a full row both work.
    assert model(random_tokens(50, batch=1, time=1)).shape == (1, 1, 50)
    assert model(random_tokens(50, batch=2, time=96)).shape == (2, 96, 50)


def test_a_row_longer_than_the_context_is_refused():
    model = make_model(vocab_size=50, width=8)
    with pytest.raises(ValueError):
        model(random_tokens(50, batch=1, time=97))


def test_width_must_be_divisible_by_heads():
    with pytest.raises(ValueError):
        TinyGPT(ModelConfig(vocab_size=50, width=6, heads=4))


def test_same_seed_gives_the_same_starting_weights():
    first = make_model(seed=5)
    second = make_model(seed=5)
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("width", [4, 32])
def test_the_body_starts_the_same_whatever_the_size_of_the_table(width):
    # The same seed with a table of 43 words (16 animals) and of 1,051 words
    # (1,024 animals). If the table were drawn first it would use up random
    # numbers, and the two bodies would start from different values.
    small = dict(make_model(vocab_size=43, width=width, seed=3).named_parameters())
    large = dict(make_model(vocab_size=LARGE_VOCAB, width=width, seed=3).named_parameters())
    body = [name for name in small if name != "table.weight"]
    assert list(small) == list(large) and len(body) == len(small) - 1
    for name in body:
        assert torch.equal(small[name], large[name]), name
    assert small["table.weight"].shape != large["table.weight"].shape

    # The check can tell bodies apart: another seed gives another body.
    other = dict(make_model(vocab_size=43, width=width, seed=4).named_parameters())
    assert not torch.equal(small["position_table.weight"], other["position_table.weight"])
    # And the table does get starting values of the usual small size.
    assert 0.01 < large["table.weight"].std().item() < 0.03


# ---------------------------------------------------------------------------
# Causality: the model must never see the word it is asked to predict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [4, 32])
def test_later_words_never_change_earlier_predictions(width):
    vocab_size, time = 50, 12
    model = make_model(vocab_size=vocab_size, width=width)
    tokens = random_tokens(vocab_size, batch=2, time=time)
    original = model(tokens).detach()

    for cut in range(1, time):
        changed_tokens = tokens.clone()
        changed_tokens[:, cut:] = (tokens[:, cut:] + 1) % vocab_size   # every word from `cut` on is different
        changed = model(changed_tokens).detach()

        # Positions before the cut must give exactly the same scores ...
        assert torch.allclose(changed[:, :cut], original[:, :cut], atol=1e-6), cut
        # ... and from the cut on they must differ, or this test would prove nothing.
        assert (changed[:, cut:] - original[:, cut:]).abs().max() > 1e-4, cut


def test_earlier_words_do_change_later_predictions():
    # The other direction: information must flow forward, otherwise the model
    # could not use "foxes" when it reaches "eat".
    vocab_size = 50
    model = make_model(vocab_size=vocab_size, width=8)
    tokens = random_tokens(vocab_size, batch=2, time=12)
    original = model(tokens).detach()

    changed_tokens = tokens.clone()
    changed_tokens[:, 0] = (tokens[:, 0] + 1) % vocab_size             # only the first word differs
    changed = model(changed_tokens).detach()

    assert (changed[:, -1] - original[:, -1]).abs().max() > 1e-4


# ---------------------------------------------------------------------------
# A real-sized pass, and the loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [4, 32])
def test_forward_and_backward_pass_with_the_large_vocabulary(width):
    model = make_model(vocab_size=LARGE_VOCAB, width=width)
    tokens = random_tokens(LARGE_VOCAB, batch=8, time=96, seed=1)
    targets = random_tokens(LARGE_VOCAB, batch=8, time=96, seed=2)

    logits = model(tokens)
    assert logits.shape == (8, 96, LARGE_VOCAB)

    loss = F.cross_entropy(logits.reshape(-1, LARGE_VOCAB), targets.reshape(-1))
    assert torch.isfinite(loss)
    # An untrained model guesses almost evenly, so its loss is close to
    # ln(vocabulary size), the loss of a blind even guess.
    assert abs(loss.item() - math.log(LARGE_VOCAB)) < 0.1

    # The backward pass must reach every tensor of the model.
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_targets_of_minus_100_are_ignored():
    vocab_size = 50
    model = make_model(vocab_size=vocab_size, width=8)
    tokens = random_tokens(vocab_size, batch=4, time=20, seed=1)
    targets = random_tokens(vocab_size, batch=4, time=20, seed=2)
    targets[:, 12:] = -100          # padding at the end of every row, as pack_rows makes it
    targets[0, 3] = -100            # and one in the middle, to be thorough
    kept = targets != -100

    logits = model(tokens)
    logits.retain_grad()            # keep the gradient at the logits for the check below
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=-100)

    # By hand: the average surprise at the right word, over kept positions only.
    log_probabilities = torch.log_softmax(logits.detach(), dim=-1)
    surprise = -log_probabilities[kept, targets[kept]]
    assert surprise.numel() == 4 * 12 - 1
    assert torch.allclose(loss.detach(), surprise.mean(), atol=1e-6)

    # Ignored positions push on nothing: the gradient there is exactly zero.
    loss.backward()
    assert logits.grad[~kept].abs().max() == 0
    assert logits.grad[kept].abs().sum() > 0


def test_padding_at_the_end_of_a_row_cannot_change_the_loss():
    # Whatever stands in the padded places, the loss is the same: their own
    # targets are ignored, and causality keeps them from changing anything earlier.
    vocab_size = 50
    model = make_model(vocab_size=vocab_size, width=8)
    tokens = random_tokens(vocab_size, batch=4, time=20, seed=1)
    targets = random_tokens(vocab_size, batch=4, time=20, seed=2)
    targets[:, 12:] = -100

    other_padding = tokens.clone()
    other_padding[:, 12:] = (tokens[:, 12:] + 7) % vocab_size

    def loss_of(inputs):
        logits = model(inputs).detach()
        return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=-100)

    assert torch.allclose(loss_of(tokens), loss_of(other_padding), atol=1e-6)
