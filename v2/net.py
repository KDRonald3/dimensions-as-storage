"""
v2/net.py - the model of the second experiment (PLAN.md, part B.6).

The data flows through these parts, in this order:

  1. the word table      a row of 32 numbers for every token. Every run starts from the same
                         rows, drawn once with a fixed seed. With table "trained" (the default)
                         the rows are trained with everything else. With table "frozen" they are
                         never trained. Nothing else differs between the two forms.
  2. the position table  a trained row of 32 numbers for each of the 13 places of a row
  3. 2 layers            each: normalise, attention with 2 heads, add;
                         normalise, middle network 32 -> body -> 32, add
                         (the same arrangement as the first experiment's model.py)
  4. a last normalise
  5. the output layer    32 -> one score for every word except <pad> and the creature names, so
                         the model can never say a creature name

Model size is `body`: the number of neurons in the middle network of each layer. Apart from the
word table, every trained part has the same size whatever the number of creatures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v2.lang import FULL_TOKENS, NESTED_TOKENS, SENTENCE_LENGTH

# Starting values of trained parts, as in the first experiment: weights drawn from a bell curve
# with this spread, biases zero, normalising layers at PyTorch's defaults.
INIT_STD = 0.02
TABLE_STREAM = 13  # tags the random numbers of the word table (see v2/lang.py)
NESTED_TABLE_STREAM = 17  # tags the random numbers of the third experiment's nested table
NESTED_COLUMNS = 64
TABLE_FORMS = ("trained", "frozen")


def full_word_table(columns=32):
    """The starting word table of the full vocabulary (32 kinds, 1,024 creatures): numbers
    drawn from a bell curve with spread 1, always with the same fixed seed."""
    rng = np.random.default_rng([TABLE_STREAM, columns])
    return rng.standard_normal((FULL_TOKENS, columns)).astype(np.float32)


def nested_word_table():
    """The starting word table of the third experiment ("nested64", v3/PLAN.md B.1): ONE table
    with 64 columns for the full vocabulary of 64 kinds and 1,024 creatures, drawn from a bell
    curve with spread 1, always with the same fixed seed. A model with fewer columns uses its
    first columns, so a word starts from the same numbers at every column count."""
    rng = np.random.default_rng([NESTED_TABLE_STREAM])
    return rng.standard_normal((NESTED_TOKENS, NESTED_COLUMNS)).astype(np.float32)


def word_table(lang, columns=32, table_start="v2"):
    """The rows of the full table that this language uses, so a word has the same starting row
    in every vocabulary. Returns [tokens, columns].

    table_start "v2" (the default) is the second experiment's table, drawn anew for each number
    of columns; "nested64" is the third experiment's table (see nested_word_table)."""
    if table_start == "nested64":
        if columns > NESTED_COLUMNS:
            raise ValueError(f"the nested64 table has {NESTED_COLUMNS} columns; got {columns}")
        return np.ascontiguousarray(nested_word_table()[lang.nested_ids, :columns])
    if table_start != "v2":
        raise ValueError(f'table_start must be "nested64" (or left out); got "{table_start}".')
    if lang.full_ids is None:
        raise ValueError("64 kinds need the nested64 starting table (table_start: nested64)")
    return full_word_table(columns)[lang.full_ids]


class Attention(nn.Module):
    """Lets every place gather information from itself and the places before it (copied from
    the first experiment). Each place makes a query (what am I looking for?), a key (what do I
    offer?) and a value (what I hand over). The work is split over `heads` heads."""

    def __init__(self, columns, heads):
        super().__init__()
        self.heads = heads
        self.query_key_value = nn.Linear(columns, 3 * columns)
        self.output = nn.Linear(columns, columns)

    def forward(self, x):
        batch, time, columns = x.shape
        query, key, value = self.query_key_value(x).split(columns, dim=2)
        gathered = F.scaled_dot_product_attention(
            self.split_heads(query), self.split_heads(key), self.split_heads(value),
            is_causal=True,  # never look ahead at the word to be predicted
        )
        gathered = gathered.transpose(1, 2).reshape(batch, time, columns)
        return self.output(gathered)

    def split_heads(self, x):
        """[batch, time, columns] -> [batch, heads, time, columns / heads]"""
        batch, time, columns = x.shape
        return x.reshape(batch, time, self.heads, columns // self.heads).transpose(1, 2)


class Layer(nn.Module):
    """Normalise, attention, add; normalise, middle network, add."""

    def __init__(self, columns, heads, body):
        super().__init__()
        self.norm_1 = nn.LayerNorm(columns)
        self.attention = Attention(columns, heads)
        self.norm_2 = nn.LayerNorm(columns)
        # The middle network: its `body` neurons are the model-size dial of this experiment.
        # GELU is the same bend as in the first experiment.
        self.middle = nn.Sequential(
            nn.Linear(columns, body),
            nn.GELU(),
            nn.Linear(body, columns),
        )

    def forward(self, x):
        x = x + self.attention(self.norm_1(x))
        x = x + self.middle(self.norm_2(x))
        return x


def set_starting_values(part, init_std=INIT_STD):
    """Small random weights and zero biases (as in the first experiment)."""
    for module in part.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.bias)


class Net(nn.Module):
    """The model. forward(tokens [batch, time]) -> scores [batch, time, n_out], where score
    number j is for token id j + 1."""

    def __init__(self, table, n_out, body, columns=32, layers=2, heads=2, table_form="trained"):
        super().__init__()
        if columns % heads != 0:
            raise ValueError(f"columns ({columns}) must be divisible by heads ({heads})")
        if table_form not in TABLE_FORMS:
            raise ValueError(f'table must be "trained" or "frozen"; got "{table_form}".')
        self.table_form = table_form
        # All other trained parts first, then their starting values. Their sizes do not depend
        # on the number of creatures, so the same seed gives the same starting values for every
        # vocabulary with the same kinds, and for both forms of the word table.
        self.position_table = nn.Embedding(SENTENCE_LENGTH, columns)
        self.layers = nn.ModuleList([Layer(columns, heads, body) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(columns)
        self.output = nn.Linear(columns, n_out)
        set_starting_values(self)
        # The word table. Its starting rows come from the fixed table, not from the run's
        # seed, so making it uses no random numbers of the run.
        start = torch.as_tensor(np.asarray(table, dtype=np.float32))
        if table_form == "trained":
            # A parameter: trained with everything else. Rows of tokens that never appear
            # (creatures not in use) get a gradient of exactly zero, so they never move.
            self.word_table = nn.Parameter(start)
        else:
            # A buffer: saved with the model but never trained and never handed to the
            # optimizer.
            self.register_buffer("word_table", start)

    def forward(self, tokens):
        batch, time = tokens.shape
        places = torch.arange(time, device=tokens.device)
        x = self.word_table[tokens] + self.position_table(places)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.final_norm(x))


def make_net(lang, body, columns=32, layers=2, heads=2, table_form="trained", table_start="v2"):
    """A fresh model for a language. Set the random seed before calling it."""
    return Net(word_table(lang, columns, table_start), lang.n_out, body, columns, layers, heads,
               table_form)


def count_numbers(model, lang):
    """How many numbers the model has, under the names of PLAN.md B.8:

      trained                trained numbers: every trained part, plus, when the table is
                             trained, the rows of the word table that are in use
      trained_without_table  trained numbers outside the word table (the same for every
                             number of creatures)
      table_rows_in_use      rows of the word table a run can meet (<pad>, the other
                             non-creature words, and the creatures in use)
      table_numbers          numbers in those rows (rows x columns)
      body_middle            trained numbers of the middle networks only
    """
    without_table = sum(p.numel() for name, p in model.named_parameters()
                        if name != "word_table")
    columns = model.word_table.shape[1]
    table_numbers = lang.rows_in_use * columns
    trained = without_table + (table_numbers if model.table_form == "trained" else 0)
    middle = sum(p.numel() for layer in model.layers for p in layer.middle.parameters())
    return {"trained": trained, "trained_without_table": without_table,
            "table_rows_in_use": lang.rows_in_use, "table_numbers": table_numbers,
            "body_middle": middle}
