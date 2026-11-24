"""A tiny GPT: the model whose size is varied in the experiment.

What it does: it reads a row of words and, at every position, guesses which
word comes next. Nothing else. Whatever it "knows" about foxes or bears has to
be stored in its PARAMETERS, the numbers that training adjusts.

The data flows through these parts, in this order:

  1. the word table      one row of `width` numbers for every word
  2. the position table  one row of `width` numbers for every place in a row of text
  3. `layers` blocks     each block lets a word look back at the earlier words
                         (attention) and then thinks about what it gathered (MLP)
  4. a final layer norm
  5. the word table again, to score every word as a candidate for the next word

Parts 2 to 4 are called the BODY here. The body has the same number of
parameters whatever the vocabulary is; only the word table grows with it.

Run this file to see where the parameters are (it trains nothing):

    python model.py
    python model.py --width 4
    python model.py --vocab-size 43 --width 16

Other files use it like this:

    model = TinyGPT(ModelConfig(vocab_size=1051, width=32))
    logits = model(tokens)        # tokens: whole numbers [batch, time]
"""

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Before training, every weight is a small random number drawn from a bell
# curve of this spread (standard deviation). Small starting values make an
# untrained model guess almost evenly among all words.
INIT_STD = 0.02

# Only for running this file directly: the vocabulary of the largest language
# (27 fixed words + 1,024 animal words, see language.py) ...
DEMO_VOCAB_SIZE = 1051
# ... and the widths listed side by side in the printout.
DEMO_WIDTHS = [4, 8, 16, 32, 64]


@dataclass
class ModelConfig:
    vocab_size: int     # rows in the word table: one for every word the model can read or write
    width: int = 32     # how many numbers describe one word; this is the model-size dial
    layers: int = 2     # how many blocks are stacked on top of each other
    heads: int = 2      # attention heads in each block; width must be divisible by heads
    context: int = 96   # the longest row of words the model can read at once


class Attention(nn.Module):
    """Lets every position gather information from the positions before it.

    For each position the layer computes three vectors: a QUERY (what am I
    looking for?), a KEY (what do I have to offer?) and a VALUE (what I hand
    over if I am picked). The query of one position is compared with the key
    of every earlier position; the better they match, the larger the share of
    that position's value it receives.

    CAUSAL means a position may look at itself and at earlier positions, never
    ahead. Without this rule the model could simply read the word it is
    supposed to predict.

    The work is split over `heads` independent heads, each using its own slice
    of the width. Two heads can look at two different earlier words at once,
    for example the animal and the verb.
    """

    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        # One layer computes query, key and value together (3 x width numbers).
        self.query_key_value = nn.Linear(config.width, 3 * config.width)
        self.output = nn.Linear(config.width, config.width)

    def forward(self, x):
        batch, time, width = x.shape
        query, key, value = self.query_key_value(x).split(width, dim=2)
        gathered = F.scaled_dot_product_attention(
            self.split_heads(query),
            self.split_heads(key),
            self.split_heads(value),
            is_causal=True,
        )
        # [batch, heads, time, width / heads] -> back to [batch, time, width]
        gathered = gathered.transpose(1, 2).reshape(batch, time, width)
        return self.output(gathered)

    def split_heads(self, x):
        """[batch, time, width] -> [batch, heads, time, width / heads]"""
        batch, time, width = x.shape
        return x.reshape(batch, time, self.heads, width // self.heads).transpose(1, 2)


class Block(nn.Module):
    """One attention step followed by one small network (the MLP).

    Attention moves information BETWEEN positions. The MLP (multi-layer
    perceptron: two linear layers with a bend, GELU, in between) then works on
    each position separately.
    """

    def __init__(self, config):
        super().__init__()
        # A layer norm shifts and rescales a vector so that its numbers have
        # average 0 and a standard spread, which keeps training stable.
        # "Pre-layer-norm" means it is applied BEFORE attention and before
        # the MLP, not after.
        self.norm_1 = nn.LayerNorm(config.width)
        self.attention = Attention(config)
        self.norm_2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
        )

    def forward(self, x):
        # "x = x + ..." : each step ADDS a correction to the vector it was
        # given instead of replacing it, so nothing learned earlier is lost.
        x = x + self.attention(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


def set_starting_values(part, init_std):
    """Give one part of a model small random weights and zero biases.
    Layer norms keep PyTorch's defaults."""
    for module in part.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.bias)


class TinyGPT(nn.Module):
    """A small GPT: predicts the next word at every position of a row."""

    def __init__(self, config, init_std=INIT_STD):
        super().__init__()
        if config.width % config.heads != 0:
            raise ValueError(
                f"width ({config.width}) must be divisible by heads ({config.heads})"
            )
        self.config = config

        # The BODY is made first, and given its starting values, before the
        # word table exists. The reason is how random numbers work: they come
        # out of one stream, one after another, so whatever draws first decides
        # which numbers are left for the rest. If the word table were made
        # first, a table of 43 words and a table of 1,051 words would use up
        # different amounts of the stream, and the same seed would give the
        # body different starting values. Made in this order, the body starts
        # exactly the same whatever the size of the table.
        #
        # The model has no built-in sense of word order, so every place in the
        # row gets a learned vector, which is added to the word's vector.
        self.position_table = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.width)
        set_starting_values(self, init_std)    # everything made so far: the body

        # The word table, made last: row number i is the vector for the word
        # with token id i. (A TOKEN is one unit of text; in this project one
        # token is one word, and its id is simply its row number in this table.)
        self.table = nn.Embedding(config.vocab_size, config.width)
        set_starting_values(self.table, init_std)

    def forward(self, tokens):
        """tokens: token ids [batch, time]. Returns logits [batch, time, vocab_size]."""
        batch, time = tokens.shape
        if time > self.config.context:
            raise ValueError(
                f"this row has {time} words but the model reads at most {self.config.context}"
            )
        places = torch.arange(time, device=tokens.device)
        x = self.table(tokens) + self.position_table(places)   # [batch, time, width]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        # The table is TIED: the same rows that turned words into vectors on
        # the way in now score every word on the way out. A word's score is
        # high when the model's vector points the same way as that word's row.
        # So there is only ONE set of numbers per word, and it has to serve as
        # both "what this word means when read" and "when to say this word".
        #
        # The scores are called LOGITS: one number per word in the vocabulary,
        # higher = more likely to come next. Softmax turns them into
        # probabilities. The logits at position t are the guess for the word
        # at position t + 1.
        logits = x @ self.table.weight.T                       # [batch, time, vocab_size]
        return logits


def count_parameters(model):
    """How many numbers the model can adjust, split into body and word table.

    table: vocab_size x width, the only part that depends on the vocabulary.
    body:  everything else (blocks, position table, final layer norm).
    """
    table = model.table.weight.numel()
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"body": total - table, "table": table, "total": total}


# ---------------------------------------------------------------------------
# Everything below only runs with "python model.py". It prints; it trains nothing.
# ---------------------------------------------------------------------------


def count(module):
    """Number of parameters inside one part of the model."""
    return sum(parameter.numel() for parameter in module.parameters())


def print_parts(model):
    """Where the parameters are, part by part."""
    config = model.config
    line = "  {:<28}{:>10,}   {}"
    print(line.format("word table", count(model.table),
                      f"{config.vocab_size:,} words x {config.width} numbers; "
                      "used on the way in and on the way out (tied)"))
    print(line.format("position table", count(model.position_table),
                      f"{config.context} places x {config.width} numbers"))
    for number, block in enumerate(model.blocks, start=1):
        norms = count(block.norm_1) + count(block.norm_2)
        print(line.format(f"block {number}: attention", count(block.attention),
                          f"{config.heads} heads"))
        print(line.format(f"block {number}: MLP", count(block.mlp),
                          f"{config.width} -> {4 * config.width} -> {config.width} numbers"))
        print(line.format(f"block {number}: two layer norms", norms, "").rstrip())
    print(line.format("final layer norm", count(model.final_norm), "").rstrip())

    counts = count_parameters(model)
    print()
    print(line.format("body", counts["body"], "everything except the word table"))
    print(line.format("table", counts["table"], "the only part that depends on the vocabulary"))
    print(line.format("total", counts["total"], "").rstrip())


def print_widths(config, widths):
    """The same model at several widths, side by side."""
    print(f"  {'width':>5}{'body':>12}{'table':>12}{'total':>12}")
    for width in widths:
        if width % config.heads != 0:
            print(f"  {width:>5}   (skipped: not divisible by {config.heads} heads)")
            continue
        other = ModelConfig(config.vocab_size, width, config.layers, config.heads, config.context)
        counts = count_parameters(TinyGPT(other))
        print(f"  {width:>5}{counts['body']:>12,}{counts['table']:>12,}{counts['total']:>12,}")


def print_starting_loss(model):
    """An untrained model should be about as unsure as a blind, even guess."""
    config = model.config
    tokens = torch.randint(0, config.vocab_size, (8, config.context))
    targets = torch.randint(0, config.vocab_size, (8, config.context))
    with torch.no_grad():
        logits = model(tokens)
    # Cross-entropy is the model's average surprise at the right word. PyTorch
    # reports it in natural-log units; dividing by ln 2 gives bits.
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
    print(f"  untrained model on random words: {loss.item() / math.log(2):.2f} bits per word")
    print(f"  an even guess among {config.vocab_size:,} words:   "
          f"{math.log2(config.vocab_size):.2f} bits per word")


def main():
    parser = argparse.ArgumentParser(description="Show the parameters of a TinyGPT. Trains nothing.")
    parser.add_argument("--vocab-size", type=int, default=DEMO_VOCAB_SIZE,
                        help="rows in the word table")
    parser.add_argument("--width", type=int, default=ModelConfig.width,
                        help="numbers per word: the model-size dial")
    parser.add_argument("--layers", type=int, default=ModelConfig.layers,
                        help="blocks stacked on top of each other")
    parser.add_argument("--heads", type=int, default=ModelConfig.heads,
                        help="attention heads per block")
    parser.add_argument("--context", type=int, default=ModelConfig.context,
                        help="longest row of words the model can read")
    parser.add_argument("--init-std", type=float, default=INIT_STD,
                        help="spread of the random starting weights")
    parser.add_argument("--widths", type=int, nargs="+", default=DEMO_WIDTHS,
                        help="widths to list side by side")
    parser.add_argument("--seed", type=int, default=0,
                        help="decides the random starting weights")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = ModelConfig(args.vocab_size, args.width, args.layers, args.heads, args.context)
    model = TinyGPT(config, init_std=args.init_std)

    print(f"TinyGPT: {config.vocab_size:,} words, width {config.width}, "
          f"{config.layers} layers, {config.heads} heads, context {config.context}")
    print()
    print("Where the parameters are:")
    print_parts(model)
    print()
    print(f"The same model at other widths ({config.vocab_size:,} words):")
    print_widths(config, args.widths)
    print()
    print("Before any training:")
    print_starting_loss(model)


if __name__ == "__main__":
    main()
