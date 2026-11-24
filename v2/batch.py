"""
v2/batch.py - train MANY models of the second experiment together, in one pass, so that one
graphics card is kept busy (one model of this experiment alone leaves it almost idle).

    from v2.batch import run_group
    summaries = run_group(configs, "runs_v2_batch/main8")      # device: "cuda" if there is one

Every member of the group ends with the same files as a run of v2/run.py:
OUT_ROOT/NAME/summary.json (the same format, with one more key, "trainer": "batch") and
OUT_ROOT/NAME/model.pt (the averaged copy of the weights, with the config).

What can be trained together
----------------------------
A GROUP is a set of runs that agree on everything except the number of creatures, the word use
and the seed (see GROUP_KEYS). All members then have tensors of exactly the same shapes: the word
table always has all 1,024 creature rows, and the output layer depends only on the kinds.
For the third experiment (v3/PLAN.md B.1) members may also have different learning rates
(make_groups(..., by_rate=False)); see "The third experiment" below.

How it works
------------
1. Every member is built with the reference code, exactly as run_one builds it (the same seed,
   so it starts from the same values).
2. The weights of all members are stacked: every weight gets one more dimension in front, one
   entry per member ("model dimension").
3. forward_members (below) computes the model of v2/net.py for all members in one pass: the
   same parts in the same order, written with the member dimension in front (batched matrix
   products). The tests check that it agrees with v2/net.py.
4. The loss is the SUM of the members' own losses. A member's loss depends only on its own
   weights, so each member gets exactly its own gradient; the members never mix.
5. Gradient clipping is done per member (each member's own gradient length over all its
   tensors). AdamW works number by number, so one AdamW over the stacked weights behaves exactly
   as one AdamW per member. The averaged copy is updated number by number as in run_one.
6. Every member is graded with the exam of v2/lang.py, at the same steps as run_one, on its
   averaged copy (one member after another).

Picking up again
----------------
Every 2,000 steps the whole group is saved to OUT_ROOT/_groups/GROUP_NAME/checkpoint.pt. A group
that finds its checkpoint goes on from it. The batch of step t depends only on (group seed, t),
so it ends as if it had never stopped (on the same kind of device). Members that already have a
summary.json with the same config are skipped.

The third experiment (all of this is off unless a config asks for it)
---------------------------------------------------------------------
- Different learning rates in one group. torch's AdamW has one rate per tensor, and a tensor
  here holds every member. So when the rates differ, the group uses its own AdamW (own_adamw_step):
  the same sums, in the same order, as torch's AdamW does for one tensor, but with each member's
  own rate. A group with one rate uses torch's AdamW as before.
- The doubling length rule (v2/run.py end_reason). A member that ends gets its summary.json
  (with "ended") and model.pt written at once and leaves the group; the group goes on with the
  rest. The checkpoint then also records which members are still in the group.
- Stopping on request: train_group(..., deadline=T) trains until just before the time T (seconds
  since 1970, as time.time() gives), writes its checkpoint and returns, saying which members are
  unfinished. A later call picks the group up from that checkpoint.
- A member whose training breaks down (its loss is no longer a number) is ended with the reason
  "broke down" instead of stopping the whole group (only in this mode; the second experiment's
  groups still stop with an error, as before).
- groups_root: where the group checkpoints go (default OUT_ROOT/_groups).
"""

import contextlib
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

# Graphics cards: matrix sums in a fixed order (needed for exact picking up again; see
# repeatable_sums). It must be set before the graphics card is first used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch.optim.adam import adam as torch_adam  # the function torch.optim.AdamW.step calls
from torch.func import stack_module_state

from v2.batch_text import BatchText
from v2.tiny_attention import available as tiny_attention_available, tiny_attention
from v2.lang import EXAM_SETS, Language, grade, grade_picks
from v2.net import count_numbers, make_net
from v2.run import (GRADIENT_CLIP, IGNORE, NEW_KEYS, WEIGHT_DECAY, add_stop_evidence,
                    carried_on_summary,
                    carries_on, carry_until, complete_config, end_reason, grading_points,
                    learning_rate_at, make_summary, same_run, text_seed_of, write_safely)

CHECKPOINT_EVERY = 2000  # steps between checkpoints of the whole group

# Members of a group must agree on all of these. They may differ in creatures, usage, seed (and
# name and threads, which change nothing that is trained).
GROUP_KEYS = ["kinds", "body", "columns", "layers", "heads", "table", "lr", "steps", "batch",
              "warmup", "average"]
# The third experiment's settings that members must also share (when they are given).
SHARED_NEW_KEYS = ["table_start", "length_rule", "first_look"]

# AdamW's fixed settings (torch's defaults, which run_one uses).
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8

# Stopping on request: the time kept free before the deadline, on top of what the next step,
# grading and checkpoint are expected to take.
DEADLINE_SAFETY_SECONDS = 30.0
FIRST_GRADING_GUESS_SECONDS = 2.0  # per member, before a grading has been timed
FIRST_STEP_GUESS_SECONDS = 5.0  # before a step has been timed
FIRST_SAVE_GUESS_SECONDS = 10.0  # before a checkpoint has been timed

# On a graphics card, sums in a fixed order (repeatable_sums) cost speed, and the card's fastest
# way of computing attention has no fixed-order form anyway. So on a card they are off: a group
# picked up again from a checkpoint then ends the same up to the last digits (rounding), not bit
# for bit. On the processor (where the tests run) they stay on.
FIXED_ORDER_ON_CARD = False
# On a card the loss and its gradient are computed by torch.compile, which joins many small
# steps of the sums into a few larger ones (several times faster). It computes the same sums,
# so results differ only in the last digits (rounding). On the processor it is not used.
COMPILE_ON_CARD = True
try:  # a group compiles again when members leave it; allow that many times
    torch._dynamo.config.recompile_limit = 64
except AttributeError:  # older torch: the same setting under its old name
    torch._dynamo.config.cache_size_limit = 64
# On a card with Triton, attention is computed by v2/tiny_attention.py (the same sums, rounded
# differently in the last digits).
ATTENTION_KERNEL_ON_CARD = True
TEXT_STEPS_AT_ONCE = 64  # the sentences of this many steps are made in one pass
# On a card, more at once: measured on an A100, 1.0 ms a step at 64 steps, 0.46 at 256 (the
# sentences are the same whatever this number; they depend only on the seed and the step).
TEXT_STEPS_AT_ONCE_ON_CARD = 256
# Under the compiler, attention is written as "multiply, then add up" for heads of at most this
# many columns; larger heads use v2/tiny_attention.py (or, without Triton, matrix products).
# Each is the faster for its heads, measured on a T4.
ATTENTION_SUMS_UP_TO = 8
EXAM_ROWS_AT_ONCE = 8192  # exam prompts per member handed to the card at once
# On a card, the whole training step (loss, gradients, clipping, AdamW, averaged copy) is
# recorded once as a "CUDA graph" and then replayed with one call per step. The card does the
# same sums; the processor no longer has to hand it some hundred small jobs one by one, which
# for narrow models took longer than the sums themselves. (Only for groups with one learning
# rate; the AdamW of a recorded step is torch's fused one with capturable=True: the same sums.)
# The environment variable AIR_GRAPH=0 switches it off.
GRAPH_ON_CARD = os.environ.get("AIR_GRAPH", "1") != "0"
GRAPH_WARMUP_STEPS = 3  # plain steps before recording (the compiler must have done its work)
# Lower-precision matrix sums on the card (TF32: 10 bits of fraction instead of 23 in the
# products). Faster for wide models, but it changes results beyond rounding, so it is OFF unless
# the environment variable AIR_TF32=1 is set. See logs/v3/speed_report_2.md.
LOWER_PRECISION_ON_CARD = os.environ.get("AIR_TF32", "0") == "1"
# The compiler's mode on a card. "max-autotune-no-cudagraphs": the compiler tries several ways
# of doing each matrix product and keeps the fastest (the same sums in 32-bit numbers, in another
# order). Measured on an A100: 1.8 times faster for narrow groups, 1.2 times for the widest, but
# 1 to 2 minutes of compiling for every new shape of a group (much less when a similar shape has
# been compiled before on the machine). AIR_COMPILE_MODE=default: torch's default mode.
COMPILE_MODE = os.environ.get("AIR_COMPILE_MODE", "max-autotune-no-cudagraphs")
if COMPILE_MODE == "default":
    COMPILE_MODE = ""


def layer_norm(x, weight, bias):
    """nn.LayerNorm for every member: x [members, rows, columns], weight and bias [members,
    columns]."""
    return torch.addcmul(bias[:, None, :], F.layer_norm(x, x.shape[-1:]), weight[:, None, :])


WEIGHT_SUM_PIECES = 8  # see MemberLinear


class MemberLinear(torch.autograd.Function):
    """nn.Linear for every member, with a faster way of working out the weights' gradient.

    That gradient is a sum over all rows of the batch (some thousands) for a small table (for
    example 192 x 64). Done in one go, the card works on only a few small tables at once and
    most of it stands idle. So the rows are cut into WEIGHT_SUM_PIECES pieces, each piece is
    summed at the same time as the others, and the pieces are added up. The same sums, in a
    different order (so the last digits can differ)."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        return torch.baddbmm(bias[:, None, :], x, weight.transpose(1, 2))

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        members, rows, inputs = x.shape
        outputs = weight.shape[1]
        pieces = WEIGHT_SUM_PIECES if rows % WEIGHT_SUM_PIECES == 0 else 1
        grad_x = torch.bmm(grad, weight)
        grad_weight = torch.bmm(
            grad.reshape(members * pieces, rows // pieces, outputs).transpose(1, 2),
            x.reshape(members * pieces, rows // pieces, inputs),
        ).view(members, pieces, outputs, inputs).sum(dim=1)
        return grad_x, grad_weight, grad.sum(dim=1)


def linear(x, weight, bias):
    """nn.Linear for every member: x [members, rows, in], weight [members, out, in], bias
    [members, out] -> [members, rows, out]."""
    return MemberLinear.apply(x, weight, bias)


def attention(query, key, value, products=True):
    """F.scaled_dot_product_attention(query, key, value, is_causal=True) written out
    ([rows, heads, time, size] each). PyTorch's own ways are made for long texts; ours has 12
    places. products=True: as matrix products (fastest without the compiler). products=False:
    as "multiply, then add up", which the compiler turns into one fast step of the card."""
    time, size = query.shape[-2], query.shape[-1]
    if products:
        scores = torch.matmul(query, key.transpose(-1, -2))
    else:
        scores = (query.unsqueeze(-2) * key.unsqueeze(-3)).sum(dim=-1)
    scores = scores * (1.0 / math.sqrt(size))
    ahead = torch.ones(time, time, dtype=torch.bool, device=query.device).triu(1)
    weights = scores.masked_fill(ahead, -math.inf).softmax(dim=-1)  # never look ahead
    if products:
        return torch.matmul(weights, value)
    return (weights.unsqueeze(-1) * value.unsqueeze(-3)).sum(dim=-2)


def forward_members(weights, tokens, layers, heads, last=None, products=True, kernel=False):
    """The model of v2/net.py (Net.forward) for all members at once.

    weights: {name: [members, ...]} with the names of Net's state_dict. tokens: [members, rows,
    time]. Returns the scores [members, rows, time, n_out]; with `last` ([rows] places), only
    the scores at those places, [members, rows, n_out]. products: see attention. kernel: use the
    card's small attention program (v2/tiny_attention.py) instead."""
    members, rows, time = tokens.shape
    table = weights["word_table"]
    words, columns = table.shape[1], table.shape[2]
    # 1 and 2. The word table (each member looks up its own rows) and the position table.
    offsets = (torch.arange(members, device=tokens.device) * words).view(-1, 1, 1)
    x = F.embedding(tokens + offsets, table.reshape(members * words, columns))
    x = (x + weights["position_table.weight"][:, None, :time, :]).reshape(members, rows * time,
                                                                         columns)
    # 3. The layers: normalise, attention, add; normalise, middle network, add.
    size = columns // heads
    for i in range(layers):
        w = lambda name: weights[f"layers.{i}.{name}"]
        h = layer_norm(x, w("norm_1.weight"), w("norm_1.bias"))
        qkv = linear(h, w("attention.query_key_value.weight"), w("attention.query_key_value.bias"))
        if kernel:
            gathered = tiny_attention(qkv.view(members * rows, time, 3 * columns), heads)
        else:
            query, key, value = (part.reshape(members * rows, time, heads, size).transpose(1, 2)
                                 for part in qkv.split(columns, dim=2))
            gathered = attention(query, key, value, products)
            gathered = gathered.transpose(1, 2)
        gathered = gathered.reshape(members, rows * time, columns)
        x = x + linear(gathered, w("attention.output.weight"), w("attention.output.bias"))
        h = layer_norm(x, w("norm_2.weight"), w("norm_2.bias"))
        h = F.gelu(linear(h, w("middle.0.weight"), w("middle.0.bias")))
        x = x + linear(h, w("middle.2.weight"), w("middle.2.bias"))
    # 4 and 5. The last normalise and the output layer.
    if last is not None:
        x = x.view(members, rows, time, columns)[:, torch.arange(rows, device=x.device), last]
    x = layer_norm(x, weights["final_norm.weight"], weights["final_norm.bias"])
    scores = linear(x, weights["output.weight"], weights["output.bias"])
    return scores if last is not None else scores.view(members, rows, time, -1)


def member_losses(weights, inputs, targets, graded, layers, heads, products=True, kernel=False):
    """Each member's own loss: the average over its graded places, as in run_one. inputs,
    targets and graded: [members, batch, 12]. Returns [members]. products, kernel: see
    forward_members."""
    members = inputs.shape[0]
    scores = forward_members(weights, inputs, layers, heads, products=products, kernel=kernel)
    each = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), targets.reshape(-1),
                           ignore_index=IGNORE, reduction="none").view(members, -1)
    return each.sum(dim=1) / graded.view(members, -1).sum(dim=1)


@contextlib.contextmanager
def repeatable_sums(device=None):
    """Inside this block PyTorch adds up numbers in a fixed order. Without it, the gradient of the
    word table is summed by several threads (or, on a graphics card, many cores) in an order that
    changes from run to run, so the last digits change, and a group picked up again from a
    checkpoint would not end exactly as one that never stopped. warn_only: an operation that has
    no fixed-order form gives a warning instead of an error. The setting before the block is put
    back after it. On a graphics card it does nothing unless FIXED_ORDER_ON_CARD is set."""
    if device is not None and torch.device(device).type == "cuda" and not FIXED_ORDER_ON_CARD:
        yield
        return
    was_on = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_on, warn_only=was_warn_only)


def group_key(config, by_rate=True):
    """The settings that all members of a group share (of a complete config). With
    by_rate=False the learning rate is left out, so members may have different rates."""
    keys = [key for key in GROUP_KEYS if by_rate or key != "lr"]
    return (tuple(config[key] for key in keys)
            + tuple(config.get(key) for key in SHARED_NEW_KEYS))


def make_groups(configs, by_rate=True):
    """Sort configs into groups that can be trained together. Returns a list of lists, in the
    order in which each group first appears. by_rate=False puts runs with different learning
    rates into the same group (the third experiment)."""
    groups = {}
    for config in configs:
        full = complete_config(config)
        groups.setdefault(group_key(full, by_rate), []).append(config)
    return list(groups.values())


def new_style(configs):
    """True when a group uses anything of the third experiment: one of its new settings, or
    more than one learning rate. Such a group's checkpoint is named after ALL its members and
    records which members are still training (members may leave it)."""
    complete = [complete_config(c) for c in configs]
    return (any(key in c for c in complete for key in NEW_KEYS)
            or len({c["lr"] for c in complete}) > 1)


def group_name(configs):
    """The name of a group's checkpoint folder, from its members: what the members share, then
    a short code made from the member names."""
    first = configs[0]
    names = "\n".join(sorted(config["name"] for config in configs))
    code = hashlib.sha256(names.encode()).hexdigest()
    rates = {config["lr"] for config in configs}
    rate = first["lr"] if len(rates) == 1 else "mixed"
    return (f"k{first['kinds']}_b{first['body']}_{first['table']}_lr{rate}"
            f"_steps{first['steps']}_{len(configs)}members_{code[:10]}")


def fused_on(device):
    """On a card, torch's fused AdamW (one step of the card for all weights; the same sums,
    rounded differently in the last digits). On the processor, torch's default, as run_one."""
    return {"fused": True} if torch.device(device).type == "cuda" else {}


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_group(configs: list[dict], out_root: str, device: str | None = None) -> list[dict]:
    """Train a group of models together. Writes out_root/<name>/summary.json and model.pt for
    every member and returns the summaries, in the order of `configs`."""
    summaries, _ = train_group(configs, out_root, device)
    return summaries


def run_group_until(configs, out_root, device=None, deadline=None, groups_root=None,
                    quiet=False, pause_at=None, so_far_root=None, stop_check=None):
    """Train a group until it is finished or until just before `deadline` (time.time() seconds),
    whichever comes first. Returns a dictionary:

      finished      names of the members that have their summary.json
      unfinished    names of the members still to be trained (empty when the group is done)
      member_steps  how many member-steps were trained in this sitting (one step of one member)
      step          the step the group reached (None when nothing was trained)
      seconds       how long this sitting took
    """
    started = time.perf_counter()
    summaries, state = train_group(configs, out_root, device, quiet=quiet, deadline=deadline,
                                   groups_root=groups_root, pause_at=pause_at,
                                   so_far_root=so_far_root, stop_check=stop_check)
    names = [complete_config(c)["name"] for c in configs]
    unfinished = state["unfinished"] if summaries is None else []
    return {"finished": [n for n in names if n not in unfinished], "unfinished": unfinished,
            "member_steps": state["member_steps"] if state else 0,
            "step": state["step"] if state else None,
            "seconds": time.perf_counter() - started}


def write_so_far(trainer, so_far_root):
    """so_far_root/<name>.json for every member still in the group: its scores at the last
    grading, its curve, and the step the group has reached ("still training")."""
    folder = Path(so_far_root)
    folder.mkdir(parents=True, exist_ok=True)
    for i, config in enumerate(trainer.configs):
        carry = trainer.carry.get(config["name"])
        note = {"name": config["name"], "status": "still training", "step": trainer.step,
                "final": trainer.final[i], "curve": trainer.curves[i], "config": config}
        if carry is not None:
            note["ended_by_rule"] = carry["ended"]  # a marked run being carried on
        write_safely(folder / f"{config['name']}.json",
                     lambda path: path.write_text(json.dumps(note, indent=1)))


def train_group(configs, out_root, device=None, batch_maker=None, stop_after=None, quiet=False,
                deadline=None, groups_root=None, pause_at=None, so_far_root=None,
                stop_check=None):
    """The training behind run_group. Returns (summaries, state), where state holds the trained
    weights and averaged copies (stacked), for the tests.

    batch_maker (for tests only): a function step -> token ids [members, batch, 13] that is used
        instead of the sentence maker, so the tests can feed the same batches to the reference.
    stop_after (for tests only): stop right after the checkpoint of that step, as if the
        computer had been switched off. Returns (None, state) then.
    deadline: stop on request (see the top of the file). Returns (None, state) when it stopped
        before the group was done; state["unfinished"] names the members still to be trained.
    groups_root: the folder of the group checkpoints (default out_root/_groups).
    pause_at: stop (with a checkpoint) once the group has reached this step, like a deadline.
        Used to train many groups "breadth first": all to 40,000 steps, then on to 80,000, ...
    so_far_root: when given, every member still training when the group stops on request or
        pauses gets so_far_root/<name>.json with its scores so far (removed when it ends).
    stop_check: a function that returns True when the group is asked to stop now (checked
        every few seconds); it then stops as at a deadline.
    """
    device = torch.device(device or default_device())
    say = (lambda *args: None) if quiet else (lambda *args: print(*args, flush=True))
    configs = [complete_config(config) for config in configs]
    if len({group_key(c, by_rate=False) for c in configs}) != 1:
        raise ValueError(f"these configs cannot be trained together; they must agree on "
                         f"{', '.join(k for k in GROUP_KEYS if k != 'lr')} and "
                         f"{', '.join(SHARED_NEW_KEYS)}")
    if len({c["name"] for c in configs}) != len(configs):
        raise ValueError("two members of the group have the same name")
    third = new_style(configs)

    # Members that are finished already are returned as they are.
    out_root = Path(out_root)
    finished = {}
    for config in configs:
        path = out_root / config["name"] / "summary.json"
        if path.exists():
            summary = json.loads(path.read_text())
            if not same_run(summary["config"], config):
                raise ValueError(f"{path} belongs to another config; use another out_root or "
                                 f"name.")
            finished[config["name"]] = summary
    todo = [config for config in configs if config["name"] not in finished]
    if not todo:
        return [finished[c["name"]] for c in configs], None
    if batch_maker is not None and len(todo) != len(configs):
        raise ValueError("batch_maker needs every member of the group to be trained")

    groups_root = Path(groups_root) if groups_root is not None else out_root / "_groups"
    # A group of the third experiment is named after all its members, because members leave it
    # while it trains (the second experiment's groups: after the members still to train).
    checkpoint_path = groups_root / group_name(configs if third else todo) / "checkpoint.pt"
    own_adamw = len({c["lr"] for c in configs}) > 1
    saved = None
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if third:
            # The members still in the group when it was saved (some may have finished since,
            # but they go on until they end again, so the group stays exactly as it was).
            by_name = {c["name"]: c for c in configs}
            missing = [c["name"] for c in saved["configs"] if c["name"] not in by_name]
            if missing:
                raise ValueError(f"{checkpoint_path} belongs to another group; remove it first.")
            todo = [by_name[c["name"]] for c in saved["configs"]]
    trainer = GroupTrainer(todo, device, own_adamw=own_adamw)
    if saved is not None:
        trainer.load_saved(saved, checkpoint_path)
        say(f"group {checkpoint_path.parent.name}: going on from the checkpoint at step "
            f"{trainer.step:,} ({trainer.members} members)")
    else:
        say(f"group {checkpoint_path.parent.name}: {len(todo)} members on {device}")
    trainer.first_step = trainer.step  # where this sitting started (for the speed)
    if batch_maker is None:
        # Each member's sentences depend only on its own seed (or text_seed) and the step
        # (PLAN.md B.7), not on the other members of the group.
        batch_maker = trainer.make_text

    shared = todo[0]
    gradings = set(grading_points(shared))
    started = time.perf_counter()
    clock = DeadlineClock(deadline)

    def save():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with clock.timing("save"):
            trainer.save(checkpoint_path, trainer.seconds + time.perf_counter() - started)

    def finish(ending):
        """Members that end now: their files are written and they leave the group."""
        seconds = trainer.seconds + time.perf_counter() - started
        for summary in trainer.finish_members(ending, out_root, seconds):
            finished.setdefault(summary["name"], summary)
            if so_far_root is not None:
                (Path(so_far_root) / f"{summary['name']}.json").unlink(missing_ok=True)
            say(f"  {summary['name']} ended at step {summary['ended']['step']:,}: "
                f"{summary['ended']['reason']}")

    asked_to_stop = StopChecker(stop_check)
    while trainer.members and trainer.step < shared["steps"]:
        pausing = pause_at is not None and trainer.step >= pause_at
        if pausing or asked_to_stop() or not clock.room_for_step(trainer.step + 1 in gradings,
                                                                 trainer.members):
            # Stopped on request: save where the group is, and say who is unfinished.
            trainer.seconds += time.perf_counter() - started
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save(checkpoint_path, trainer.seconds)
            if so_far_root is not None:
                write_so_far(trainer, so_far_root)
            state = trainer.state()
            state["unfinished"] = [c["name"] for c in trainer.configs]
            say(f"  {'paused' if pausing else 'stopped on request'} at step {trainer.step:,}; "
                f"{trainer.members} members unfinished")
            return None, state
        with clock.timing("step"):
            tokens = batch_maker(trainer.step).to(device)
            trainer.train_step(tokens)
        graded_now = trainer.step in gradings
        if graded_now:
            with clock.timing("grading", trainer.members):
                trainer.grade_all()
            line = "  ".join(f"{c['name']} {trainer.final[i]['facts']:.0f}"
                             for i, c in enumerate(trainer.configs))
            say(f"  step {trainer.step:>7,}  facts: {line}")
        at_checkpoint = trainer.step % CHECKPOINT_EVERY == 0 and trainer.step < shared["steps"]
        if third and (graded_now or at_checkpoint):
            # A member whose training broke down leaves the group (third experiment only).
            broken = trainer.broken_members()
            if broken:
                finish(broken)
        if graded_now:
            # The doubling rule: which members end at this step (none with a fixed length).
            ending = []
            for i, config in enumerate(trainer.configs):
                carry = trainer.carry.get(config["name"])
                if carry is not None:  # a marked member being carried on: stops at "until"
                    if trainer.step >= carry["until"]:
                        ending.append((i, "carried"))
                    continue
                reason = end_reason(config, trainer.step, trainer.curves[i])
                if reason is not None and carries_on(config, trainer.step, reason):
                    trainer.start_carry(i, reason)
                    say(f"  {config['name']} ended at step {trainer.step:,}: {reason}; carried "
                        f"on to step {trainer.carry[config['name']]['until']:,} (a marked run)")
                elif reason is not None:
                    ending.append((i, reason))
            if ending:
                finish(ending)
        if trainer.members and at_checkpoint:
            trainer.check_finite()
            save()
            if stop_after is not None and trainer.step >= stop_after:
                return None, trainer.state()
    trainer.seconds += time.perf_counter() - started
    if trainer.members:
        trainer.check_finite()
        if any(final is None for final in trainer.final):  # a finished checkpoint picked up again
            trainer.grade_all(keep_loss=True)
        summaries = trainer.write_results(out_root)
        for summary in summaries:
            finished[summary["name"]] = summary
            if so_far_root is not None:
                (Path(so_far_root) / f"{summary['name']}.json").unlink(missing_ok=True)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    state = trainer.state()
    # (A member that ended in an earlier sitting whose summary is not in out_root has None.)
    return [finished.get(c["name"]) for c in configs], state


class StopChecker:
    """Calls stop_check (a function) at most every few seconds, so that asking, for example,
    whether a file exists does not slow down every step. Without a function: never stop."""

    EVERY_SECONDS = 5.0

    def __init__(self, stop_check):
        self.stop_check = stop_check
        self.last = time.monotonic()

    def __call__(self):
        if self.stop_check is None or time.monotonic() - self.last < self.EVERY_SECONDS:
            return False
        self.last = time.monotonic()
        return bool(self.stop_check())


class DeadlineClock:
    """Measures how long a step, a grading and a checkpoint take, so that a group asked to stop
    by a deadline knows whether one more step (and the grading after it, and a checkpoint) still
    fits. Without a deadline it always says yes."""

    def __init__(self, deadline):
        self.deadline = deadline
        self.step_seconds = []  # the last few steps
        self.grading_per_member = FIRST_GRADING_GUESS_SECONDS
        self.save_seconds = FIRST_SAVE_GUESS_SECONDS

    @contextlib.contextmanager
    def timing(self, what, members=1):
        start = time.perf_counter()
        yield
        spent = time.perf_counter() - start
        if what == "step":
            self.step_seconds = (self.step_seconds + [spent])[-50:]
        elif what == "grading":
            self.grading_per_member = spent / max(members, 1)
        else:
            self.save_seconds = spent

    def room_for_step(self, grading_next, members):
        if self.deadline is None:
            return True
        # The slowest of the last steps, twice over: steps can vary.
        step = 2 * max(self.step_seconds) if self.step_seconds else FIRST_STEP_GUESS_SECONDS
        need = step + 2 * self.save_seconds + DEADLINE_SAFETY_SECONDS
        if grading_next:
            need += 1.5 * self.grading_per_member * members
        return time.time() + need < self.deadline


class GroupTrainer:
    """The stacked weights, averaged copies and optimizer of one group, and its training step."""

    def __init__(self, configs, device, own_adamw=False):
        self.configs = configs
        self.device = device
        self.members = len(configs)
        shared = configs[0]
        self.languages = [Language(c["creatures"], c["kinds"], c["usage"]) for c in configs]
        self.creature_base = self.languages[0].creature_base
        self.rows = list(range(self.members))  # each member's place in the starting list

        # 1. Every member built by the reference code, exactly as run_one builds it.
        models = []
        for config, lang in zip(configs, self.languages):
            torch.manual_seed(config["seed"])  # the starting values
            extra = ({"table_start": config["table_start"]} if "table_start" in config else {})
            models.append(make_net(lang, config["body"], config["columns"], config["layers"],
                                   config["heads"], config["table"], **extra))
        self.models = models  # kept on the processor: used to write model.pt at the end
        self.numbers = [count_numbers(m, lang) for m, lang in zip(models, self.languages)]

        # 2. The weights stacked, with the member number in front. With a frozen table, the
        # word table is a buffer, so it is stacked with the buffers and never trained.
        params, buffers = stack_module_state(models)
        self.params = {k: v.detach().to(device).requires_grad_(True) for k, v in params.items()}
        self.buffers = {k: v.detach().to(device) for k, v in buffers.items()}
        self.averaged = {k: v.detach().clone() for k, v in self.params.items()}
        # The optimizer: torch's AdamW when every member has the same rate (as before), our own
        # copy of it (own_adamw_step) when the rates differ.
        self.own_adamw = own_adamw
        self.optimizer = None
        self.adam = None
        on_card = torch.device(device).type == "cuda"
        if on_card:  # TF32 only when asked for (see LOWER_PRECISION_ON_CARD)
            torch.backends.cuda.matmul.allow_tf32 = LOWER_PRECISION_ON_CARD
        # The recorded step (see GRAPH_ON_CARD): the learning rate and the weight of the averaged
        # copy then live on the card, and are set before every replay.
        self.graphed = on_card and GRAPH_ON_CARD and not own_adamw
        self.graph = None
        self.plain_steps = 0  # plain steps since the group was (re)built
        if self.graphed:
            self.rate_now = torch.tensor(shared["lr"], dtype=torch.float32, device=device)
            self.average_weight = torch.zeros((), dtype=torch.float32, device=device)
        if own_adamw:
            self.adam = {"step": 0,
                         "exp_avg": {k: torch.zeros_like(v) for k, v in self.params.items()},
                         "exp_avg_sq": {k: torch.zeros_like(v) for k, v in self.params.items()}}
        else:
            self.optimizer = self.new_optimizer(shared["lr"])

        self.layers, self.heads = shared["layers"], shared["heads"]
        # On a card the loss is computed by a compiled copy of member_losses (see
        # COMPILE_ON_CARD); made at the first step.
        self.compiled = None
        self.compile = torch.device(device).type == "cuda" and COMPILE_ON_CARD
        # The card's small attention program (v2/tiny_attention.py), when there is Triton.
        # (Only for heads of more than ATTENTION_SUMS_UP_TO columns: for smaller heads the
        # compiler's own "multiply, then add up" is faster.)
        self.kernel = (ATTENTION_KERNEL_ON_CARD and tiny_attention_available(device)
                       and shared["columns"] // shared["heads"] > ATTENTION_SUMS_UP_TO)

        self.step = 0
        self.first_step = 0
        self.member_steps = 0  # steps of one member, added up, in this sitting
        self.seconds = 0.0
        self.curves = [[] for _ in configs]
        self.final = [None] * self.members
        # The training loss of every member since the last grading (kept on the device, so the
        # processor does not have to wait for the graphics card at every step).
        self.loss_sum = torch.zeros(self.members, dtype=torch.float64, device=device)
        self.loss_count = 0
        self.text = None  # the sentence maker, made when it is first needed
        # Marked members being carried on (carry_on), by name: where the rule ended them
        # ("ended"), their scores ("final") and averaged weights ("averaged", on the processor)
        # there, and the step at which carrying on stops ("until").
        self.carry = {}

    def start_carry(self, i, reason):
        """Member i is ended by the rule now but is carried on: keep its result of this step."""
        config = self.configs[i]
        self.carry[config["name"]] = {
            "ended": {"step": self.step, "reason": reason},
            "final": copy.deepcopy(self.final[i]),
            "averaged": {k: v[i].detach().cpu().clone() for k, v in self.averaged.items()},
            "until": carry_until(config, self.step)}

    def make_text(self, step):
        """The training batch of every member at this step (v2/batch_text.py).

        A member's sentences depend only on its language (creatures, word use) and its text
        seed. Members that share both (with text_seed, the runs that differ only in seed) read
        the same sentences, so these are made once and handed to each of them. That is exactly
        what each would get on its own, and saves most of the time of the sentence maker."""
        if self.text is None:
            keys = [(c["creatures"], c["usage"], text_seed_of(c)) for c in self.configs]
            unique = list(dict.fromkeys(keys))
            self.text_rows = torch.tensor([unique.index(k) for k in keys], device=self.device)
            self.text_shared = len(unique) < len(keys)
            self.text = BatchText([self.languages[keys.index(k)] for k in unique],
                                  [k[2] for k in unique], self.configs[0]["batch"], self.device)
            self.text_made = None  # (first step, sentences of TEXT_STEPS_AT_ONCE steps)
        # The sentences of several steps are made in one pass and kept until they are used.
        if self.text_made is None or not (
                0 <= step - self.text_made[0] < self.text_made[1].shape[0]):
            at_once = (TEXT_STEPS_AT_ONCE_ON_CARD if torch.device(self.device).type == "cuda"
                       else TEXT_STEPS_AT_ONCE)
            self.text_made = (step, self.text.make_steps(range(step, step + at_once)))
        tokens = self.text_made[1][step - self.text_made[0]]
        return tokens.index_select(0, self.text_rows) if self.text_shared else tokens

    def scores(self, tokens, weights=None, last=None):
        """Scores of every member: tokens [members, rows, time] -> [members, rows, time, n_out]
        (with `last`, only at those places). weights: the trained ones unless given."""
        weights = {**(weights if weights is not None else self.params), **self.buffers}
        return forward_members(weights, tokens, self.layers, self.heads, last)

    def train_step(self, tokens):
        """One step of every member, on its own batch: tokens [members, batch, 13]."""
        if tokens.shape[0] != self.members:  # a test's batch_maker, after members have left
            tokens = tokens[torch.as_tensor(self.rows, device=tokens.device)]
        if self.graphed:
            self.graphed_step(tokens)
        else:
            if not self.own_adamw:
                rate = learning_rate_at(self.step, self.configs[0])
                for group in self.optimizer.param_groups:
                    group["lr"] = rate
            self.step_sums(tokens, 1 - self.average_keep())
        self.step += 1
        self.member_steps += self.members
        self.loss_count += 1

    def average_keep(self):
        """How much of the averaged copy is kept at this step (as in run_one)."""
        return min(self.configs[0]["average"], (1 + self.step) / (10 + self.step))

    def graphed_step(self, tokens):
        """The training step on a card, recorded once and then replayed (GRAPH_ON_CARD). The
        first steps after the group is built are plain ones, so that the compiler and the card
        libraries have set themselves up before the recording."""
        self.rate_now.fill_(learning_rate_at(self.step, self.configs[0]))
        self.average_weight.fill_(1 - self.average_keep())
        if self.graph is None and self.plain_steps < GRAPH_WARMUP_STEPS:
            self.plain_steps += 1
            self.step_sums(tokens, self.average_weight)
            return
        if self.graph is None:
            self.recorded_tokens = tokens.clone()
            for p in self.params.values():
                p.grad = None
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):  # records the sums; nothing is computed yet
                self.step_sums(self.recorded_tokens, self.average_weight)
        else:
            self.recorded_tokens.copy_(tokens)
        self.graph.replay()

    def forget_graph(self):
        """The group has changed (members left, or a checkpoint was read): record again."""
        self.graph = None
        self.plain_steps = 0

    def step_sums(self, tokens, average_weight):
        """The sums of one training step: loss, gradients, clipping, AdamW, averaged copy, and
        the training loss added up. average_weight: a number, or (recorded step) a tensor on
        the card."""
        # The model reads the first 12 tokens; the target at each place is the next token, with
        # no loss where that is <pad> or a creature name (as in v2/run.py inputs_and_targets).
        inputs = tokens[:, :, :-1]
        following = tokens[:, :, 1:]
        graded = (following != 0) & (following < self.creature_base)
        targets = torch.where(graded, following - 1, torch.full_like(following, IGNORE))
        with repeatable_sums(self.device):
            member_loss = self.losses({**self.params, **self.buffers}, inputs, targets, graded)
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            else:
                for p in self.params.values():
                    p.grad = None
            member_loss.sum().backward()  # the sum: each member still gets its own gradient
        self.clip_per_member()
        if self.own_adamw:
            self.own_adamw_step([learning_rate_at(self.step, c) for c in self.configs])
        else:
            self.optimizer.step()
        self.update_average(average_weight)
        self.loss_sum += member_loss.detach().double() / math.log(2)  # bits per graded word

    def new_optimizer(self, rate):
        """torch's AdamW over the stacked weights. For the recorded step its learning rate is a
        tensor on the card (rate_now) and it is "capturable" (it can be recorded)."""
        if self.graphed:
            return torch.optim.AdamW(list(self.params.values()), lr=self.rate_now,
                                     weight_decay=WEIGHT_DECAY, fused=True, capturable=True)
        return torch.optim.AdamW(list(self.params.values()), lr=rate,
                                 weight_decay=WEIGHT_DECAY, **fused_on(self.device))

    def losses(self, weights, inputs, targets, graded):
        """Each member's loss on its batch (member_losses), compiled on a card."""
        if self.compile:
            if self.compiled is None:
                # COMPILE_MODE "max-autotune-no-cudagraphs": the compiler tries several ways of
                # doing each matrix product and keeps the fastest (the same sums).
                self.compiled = torch.compile(member_losses, dynamic=False,
                                              **({"mode": COMPILE_MODE} if COMPILE_MODE else {}))
            try:
                size = weights["word_table"].shape[-1] // self.heads
                return self.compiled(weights, inputs, targets, graded, self.layers, self.heads,
                                     products=size > ATTENTION_SUMS_UP_TO,
                                     kernel=self.kernel)
            except Exception as error:  # no compiler on this machine: the plain code, slower
                print(f"torch.compile failed ({type(error).__name__}: {error}); going on "
                      f"without it", flush=True)
                self.compile = False
        return member_losses(weights, inputs, targets, graded, self.layers, self.heads)

    def own_adamw_step(self, rates):
        """One AdamW step in which every member has its own learning rate (rates: one per
        member). For each member, torch's own AdamW code (torch.optim.adam.adam, the function
        that torch.optim.AdamW.step calls) is run on that member's part of every stacked tensor,
        with the member's rate. So each member is updated by exactly the sums a separate AdamW
        at its own rate would do, and the numbers are written straight into the stacked tensors
        (the parts are views of them)."""
        beta1, beta2 = ADAM_BETAS
        steps_before = self.adam["step"]
        self.adam["step"] += 1
        names = list(self.params)
        with torch.no_grad():
            for i, rate in enumerate(rates):
                # torch counts the steps in a small tensor per weight, adding 1 before using it.
                counts = [torch.tensor(float(steps_before), dtype=torch.float32) for _ in names]
                torch_adam(
                    [self.params[k][i] for k in names],
                    [self.params[k].grad[i] for k in names],
                    [self.adam["exp_avg"][k][i] for k in names],
                    [self.adam["exp_avg_sq"][k][i] for k in names],
                    [], counts, foreach=None, capturable=False, differentiable=False,
                    fused=None, has_complex=False, decoupled_weight_decay=True,
                    amsgrad=False, beta1=beta1, beta2=beta2, lr=rate,
                    weight_decay=WEIGHT_DECAY, eps=ADAM_EPS, maximize=False)

    def clip_per_member(self):
        """Gradient clipping as torch.nn.utils.clip_grad_norm_ does it in run_one, but for each
        member on its own: the length of the member's whole gradient (over all its tensors),
        and if that is more than GRADIENT_CLIP, the member's gradient is scaled down to it."""
        grads = [p.grad for p in self.params.values()]
        lengths = torch.stack([g.reshape(self.members, -1).norm(dim=1) for g in grads])
        total = lengths.norm(dim=0)  # [members]
        factor = (GRADIENT_CLIP / (total + 1e-6)).clamp(max=1.0)
        for g in grads:
            g.mul_(factor.view(-1, *([1] * (g.dim() - 1))))

    def update_average(self, weight=None):
        """The averaged copy, number by number, as in run_one (with lerp, so a number that has
        not moved stays exactly as it was). weight: how far to move towards the trained weights
        (1 - keep); a tensor on the card for the recorded step."""
        if weight is None:
            weight = 1 - self.average_keep()
        with torch.no_grad():
            if isinstance(weight, torch.Tensor):
                for average, param in zip(self.averaged.values(), self.params.values()):
                    average.lerp_(param, weight)
            else:
                torch._foreach_lerp_(list(self.averaged.values()), list(self.params.values()),
                                     weight)

    def check_finite(self):
        """Stop when a member's training has broken down (its loss is no longer a number)."""
        if not torch.isfinite(self.loss_sum).all():
            broken = [c["name"] for c, ok in zip(self.configs, torch.isfinite(self.loss_sum))
                      if not ok]
            raise RuntimeError(f"Training broke down by step {self.step:,} for "
                               f"{', '.join(broken)}: the loss is no longer a number. This is a "
                               f"failed run, not a result. Try a smaller lr.")

    def broken_members(self):
        """(member, "broke down") for every member whose loss is no longer a number, graded once
        more so that its summary has scores (third experiment only)."""
        ok = torch.isfinite(self.loss_sum).tolist()
        # Right after a grading the loss sums start again from zero; the loss of the grading
        # then tells.
        for i, final in enumerate(self.final):
            if final is not None and not math.isfinite(final["loss"]):
                ok[i] = False
        broken = [i for i, fine in enumerate(ok) if not fine]
        for i in broken:
            self.final[i] = grade(self.languages[i], self.predictor(i))
            self.final[i]["loss"] = float("nan")
        return [(i, "broke down") for i in broken]

    # --------------------------------------------------------------------------------------------
    # Members leaving the group (third experiment)
    # --------------------------------------------------------------------------------------------

    def finish_members(self, ending, out_root, seconds):
        """Write summary.json and model.pt of the members in `ending` ([(member, reason)]), at
        the step they are at now, then take them out of the group. A member that has a
        summary.json already (it ended before a restart) keeps it. Returns the summaries."""
        summaries = []
        for i, reason in ending:
            ended = {"step": self.step, "reason": reason}
            path = Path(out_root) / self.configs[i]["name"] / "summary.json"
            if path.exists():
                summaries.append(json.loads(path.read_text()))
            elif self.configs[i]["name"] in self.carry:
                # A carried-on member: its result is that of the point where the rule ended it.
                why = {"carried": ("limit" if self.step >= self.configs[i]["steps"]
                                   else "two more doublings")}.get(reason, reason)
                summaries.append(self.write_member(i, out_root, seconds, carried=why))
            else:
                summaries.append(self.write_member(i, out_root, seconds, ended))
            self.carry.pop(self.configs[i]["name"], None)
        leaving = {i for i, _ in ending}
        self.keep_only([i for i in range(self.members) if i not in leaving])
        return summaries

    def keep_only(self, keep):
        """Go on with the members numbered `keep` only (in that order)."""
        index = torch.as_tensor(keep, dtype=torch.long, device=self.device)
        pick = lambda tensor: tensor.detach().index_select(0, index).clone()
        old_params = self.params
        self.params = {k: pick(v).requires_grad_(True) for k, v in old_params.items()}
        self.averaged = {k: pick(v) for k, v in self.averaged.items()}
        self.buffers = {k: pick(v) for k, v in self.buffers.items()}
        if self.own_adamw:
            for part in ("exp_avg", "exp_avg_sq"):
                self.adam[part] = {k: pick(v) for k, v in self.adam[part].items()}
        elif self.members and keep:
            # A new torch AdamW over the smaller tensors, with the kept members' state.
            old_state = self.optimizer.state
            self.optimizer = self.new_optimizer(self.optimizer.param_groups[0]["lr"])
            for name, new in self.params.items():
                state = old_state.get(old_params[name])
                if state:
                    self.optimizer.state[new] = {
                        "step": state["step"].clone(),
                        "exp_avg": pick(state["exp_avg"]),
                        "exp_avg_sq": pick(state["exp_avg_sq"])}
        for attribute in ("configs", "languages", "models", "numbers", "curves", "final",
                          "rows"):
            values = getattr(self, attribute)
            setattr(self, attribute, [values[i] for i in keep])
        self.loss_sum = self.loss_sum.index_select(0, index).clone()
        self.members = len(keep)
        self.text = None  # made again for the members left (their sentences do not change)
        self.forget_graph()

    # --------------------------------------------------------------------------------------------

    def predictor(self, member):
        """The averaged copy of one member as an exam taker (as run.predictor)."""
        weights = {k: v[member:member + 1] for k, v in {**self.averaged, **self.buffers}.items()}

        def predict(prompts, last):
            with torch.no_grad():
                tokens = torch.as_tensor(prompts, device=self.device)[None]
                last = torch.as_tensor(last, device=self.device)
                scores = forward_members(weights, tokens, self.layers, self.heads, last)
                return scores[0].cpu().numpy()

        return predict

    def exam_picks(self, members, exam_set, lowest):
        """The option chosen by each of `members` (averaged copies) for every prompt of an exam
        set: [len(members), prompts]. The same choice as lang.chosen_options (a tie goes to the
        first option), but for many members at once, and only the chosen options leave the
        card."""
        index = torch.as_tensor(members, device=self.device)
        weights = {k: v.index_select(0, index)
                   for k, v in {**self.averaged, **self.buffers}.items()}
        prompts = torch.as_tensor(exam_set["prompts"], device=self.device)
        last = torch.as_tensor(exam_set["last"], device=self.device)
        options = torch.as_tensor(exam_set["options"] - 1, device=self.device)
        count = len(members)
        picks = []
        with torch.no_grad():
            for start in range(0, len(prompts), EXAM_ROWS_AT_ONCE):
                part = slice(start, start + EXAM_ROWS_AT_ONCE)
                tokens = prompts[part].expand(count, -1, -1)
                scores = forward_members(weights, tokens, self.layers, self.heads, last[part],
                                         kernel=self.kernel)
                chosen = scores.gather(2, options[part].expand(count, -1, -1))
                picks.append(chosen.argmin(dim=2) if lowest else chosen.argmax(dim=2))
        return torch.cat(picks, dim=1).cpu().numpy()

    def grade_all(self, keep_loss=False):
        """Grade every member on its averaged copy, and add a point to its curve."""
        losses = (self.loss_sum / max(self.loss_count, 1)).tolist()
        # Members with the same number of creatures (and kinds) take the same exam: their
        # options are chosen together.
        by_exam = {}
        for i, lang in enumerate(self.languages):
            by_exam.setdefault(lang.creatures, []).append(i)
        finals = {}
        for members in by_exam.values():
            lang = self.languages[members[0]]
            exam = lang.exam()
            picks = {name: self.exam_picks(members, exam[name], lowest)
                     for name, lowest in EXAM_SETS.items()}
            for j, i in enumerate(members):
                finals[i] = grade_picks(self.languages[i],
                                        {name: found[j] for name, found in picks.items()})
        for i, config in enumerate(self.configs):
            final = finals[i]
            if keep_loss:
                final["loss"] = self.curves[i][-1]["loss"] if self.curves[i] else 0.0
                self.final[i] = final
                continue
            final["loss"] = losses[i]  # average training loss since the last grading
            self.final[i] = final
            self.curves[i].append({"step": self.step, "sentences": self.step * config["batch"],
                                   "facts": final["facts"],
                                   "questions_one": final["questions_one"],
                                   "questions_new": final["questions_new"],
                                   "questions_seen": final["questions_seen"],
                                   "not_known": final["not_known"], "loss": final["loss"]})
        if not keep_loss:
            self.loss_sum.zero_()
            self.loss_count = 0

    # --------------------------------------------------------------------------------------------

    def state(self):
        state = {"configs": self.configs,
                 "params": {k: v.detach() for k, v in self.params.items()},
                 "averaged": self.averaged, "buffers": self.buffers,
                 "optimizer": (self.optimizer.state_dict() if self.optimizer is not None
                               else self.adam),
                 "step": self.step, "first_step": self.first_step,
                 "member_steps": self.member_steps,
                 "curves": self.curves, "final": self.final,
                 "loss_sum": self.loss_sum, "loss_count": self.loss_count}
        if self.carry:  # only when a member is being carried on (third experiment)
            state["carry"] = self.carry
        return state

    def save(self, path, seconds):
        state = dict(self.state(), seconds=seconds)
        write_safely(path, lambda temporary: torch.save(state, temporary))

    def load(self, path):
        saved = torch.load(path, map_location=self.device, weights_only=False)
        self.load_saved(saved, path)

    def load_saved(self, saved, path):
        """Go on from a saved checkpoint (already read from `path`)."""
        if len(saved["configs"]) != len(self.configs) or not all(
                same_run(a, b) for a, b in zip(saved["configs"], self.configs)):
            raise ValueError(f"{path} belongs to another group; remove it first.")
        with torch.no_grad():
            for k in self.params:
                self.params[k].copy_(saved["params"][k])
                self.averaged[k].copy_(saved["averaged"][k])
        if self.own_adamw:
            self.adam = saved["optimizer"]
        else:
            self.optimizer.load_state_dict(saved["optimizer"])
            if self.graphed:  # reading the checkpoint replaced these settings of the AdamW
                for group in self.optimizer.param_groups:
                    group.update(lr=self.rate_now, fused=True, capturable=True)
                for state in self.optimizer.state.values():  # the step count on the card
                    state["step"] = state["step"].to(self.device, torch.float32)
            self.forget_graph()
        self.step, self.curves, self.final = saved["step"], saved["curves"], saved["final"]
        self.loss_sum = saved["loss_sum"].to(self.device)
        self.loss_count, self.seconds = saved["loss_count"], saved["seconds"]
        self.carry = {name: dict(item, averaged={k: v.cpu() for k, v in item["averaged"].items()})
                      for name, item in saved.get("carry", {}).items()}

    def write_member(self, i, out_root, seconds, ended=None, carried=None):
        """model.pt and summary.json of member i, in the format of run_one. carried: for a
        carried-on member, why the carrying on stopped; its result is then the one kept where
        the rule ended it (self.carry), and the scores now go under "carried_on"."""
        config = self.configs[i]
        run_dir = Path(out_root) / config["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        carry = self.carry[config["name"]] if carried is not None else None
        # The member's averaged copy, in the same form as run_one saves it. clone() makes
        # each tensor its own: a slice would save the numbers of every member.
        weights = self.models[i].state_dict()
        for k in weights:
            if carry is not None and k in carry["averaged"]:
                weights[k] = carry["averaged"][k].clone()
                continue
            source = self.averaged[k] if k in self.averaged else self.buffers[k]
            weights[k] = source[i].detach().cpu().clone()
        write_safely(run_dir / "model.pt",
                     lambda path: torch.save({"config": config, "model": weights}, path))
        if carry is not None:
            summary = make_summary(config, self.numbers[i], carry["ended"]["step"], seconds,
                                   self.curves[i], carry["final"])
            ended = carry["ended"]
        else:
            summary = make_summary(config, self.numbers[i], self.step, seconds,
                                   self.curves[i], self.final[i])
        summary["trainer"] = "batch"
        if ended is not None:
            summary["ended"] = ended
        if carry is not None:
            summary["carried_on"] = carried_on_summary(self.final[i], carry["final"], self.step,
                                                       carried)
        add_stop_evidence(summary)
        # The summary is written last: if summary.json exists, the run is complete.
        write_safely(run_dir / "summary.json",
                     lambda path: path.write_text(json.dumps(summary, indent=1)))
        return summary

    def write_results(self, out_root):
        """model.pt and summary.json of every member, in the format of run_one."""
        return [self.write_member(i, out_root, self.seconds,
                                  carried="limit" if c["name"] in self.carry else None)
                for i, c in enumerate(self.configs)]
