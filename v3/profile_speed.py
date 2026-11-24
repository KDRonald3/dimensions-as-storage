"""
v3/profile_speed.py - measure how fast the group trainer (v2/batch.py) is on this machine's card.

    python -m v3.profile_speed                 # the three shapes below, 24 members each
    python -m v3.profile_speed --table         # also torch.profiler's table of the slowest parts

For each shape: making the sentences (per step), one training step and its parts, one grading
of all members, and member-steps per second without grading. Timings wait for the card to
finish (torch.cuda.synchronize), so they are real.
"""

import argparse
import time

import torch

from v2 import batch
from v2.run import complete_config

SHAPES = [(4, 8, 128), (64, 64, 128), (64, 64, 4)]  # (kinds, columns, body)


def configs(kinds, columns, body):
    found = []
    for creatures in (16, 64, 256, 1024):
        for usage in ("equal", "zipf"):
            for seed in (0, 1, 2):
                found.append(complete_config({
                    "creatures": creatures, "kinds": kinds, "usage": usage, "body": body,
                    "seed": seed, "steps": 640000, "lr": 0.01, "threads": 1, "columns": columns,
                    "text_seed": 0, "table_start": "nested64", "length_rule": "doubling",
                    "first_look": 40000, "name": f"p{creatures}{usage}{seed}"}))
    return found


def wait():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(work, repeats):
    """Average seconds of work() over `repeats` runs (after one run to warm up)."""
    work()
    wait()
    start = time.perf_counter()
    for _ in range(repeats):
        work()
    wait()
    return (time.perf_counter() - start) / repeats


def parts_of_a_step(trainer, tokens, repeats, columns):
    """Seconds of the parts of a training step, each timed on its own."""
    t = {}
    inputs = tokens[:, :, :-1]
    following = tokens[:, :, 1:]
    graded = (following != 0) & (following < trainer.creature_base)
    targets = torch.where(graded, following - 1, torch.full_like(following, batch.IGNORE))

    weights = lambda: {**trainer.params, **trainer.buffers}

    def forward_backward():
        loss = trainer.losses(weights(), inputs, targets, graded)
        for p in trainer.params.values():
            p.grad = None
        loss.sum().backward()

    def forward_only():
        with torch.no_grad():
            trainer.losses(weights(), inputs, targets, graded)

    # The compiled loss against the plain one (the same sums, rounded differently).
    start = time.perf_counter()
    forward_backward()
    wait()
    t["first step (compiling)"] = time.perf_counter() - start
    compiled = [p.grad.clone() for p in trainer.params.values()]
    compiled_loss = trainer.losses(weights(), inputs, targets, graded).detach()
    for p in trainer.params.values():
        p.grad = None
    plain_loss = batch.member_losses(weights(), inputs, targets, graded, trainer.layers,
                                     trainer.heads,
                                     products=columns // trainer.heads > batch.ATTENTION_SUMS_UP_TO)
    plain_loss.sum().backward()
    plain = [p.grad for p in trainer.params.values()]
    loss_gap = ((compiled_loss - plain_loss.detach()).abs() / plain_loss.detach().abs()).max()
    grad_gap = max(((a - b).norm() / b.norm().clamp(min=1e-30)).item()
                   for a, b in zip(compiled, plain))
    print(f"  compiled against plain: loss differs by at most {loss_gap.item():.1e} (share), "
          f"gradients by at most {grad_gap:.1e} (share of their length)")
    t["forward"] = timed(forward_only, repeats)
    t["forward+backward"] = timed(forward_backward, repeats)
    t["clipping"] = timed(trainer.clip_per_member, repeats)
    t["update average"] = timed(trainer.update_average, repeats)
    t["optimiser"] = timed(trainer.optimizer.step, repeats)
    return t


def measure(kinds, columns, body, repeats, table):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = batch.GroupTrainer(configs(kinds, columns, body), torch.device(device))
    # Sentences: one block of TEXT_STEPS_AT_ONCE steps, per step.
    trainer.make_text(0)  # builds the maker
    block = batch.TEXT_STEPS_AT_ONCE
    counter = iter(range(10 ** 9))
    text = timed(lambda: trainer.text.make_steps(
        range(block * next(counter), block * next(counter) + block)), 3) / block
    tokens = trainer.make_text(0)
    parts = parts_of_a_step(trainer, tokens, repeats, columns)
    step = timed(lambda: trainer.train_step(trainer.make_text(trainer.step)), repeats)
    grading = timed(trainer.grade_all, 1)
    per_second = trainer.members / step
    print(f"kinds {kinds} columns {columns} body {body}, {trainer.members} members on {device}:")
    print(f"  sentences {1000 * text:.1f} ms a step (made {block} steps at once)")
    print("  " + ", ".join(f"{k} {1000 * v:.1f} ms" for k, v in parts.items()))
    print(f"  whole step with sentences {1000 * step:.1f} ms -> {per_second:,.0f} member-steps "
          f"a second (gradings not counted); one grading of all members {grading:.1f} s",
          flush=True)
    if table:
        show_table(lambda: trainer.text.make_steps(range(1000, 1000 + block)), device,
                   "the sentences of 16 steps")
        show_table(lambda: trainer.train_step(trainer.make_text(trainer.step)), device,
                   "5 training steps")
    return {"text": text, "step": step, "per_second": per_second, "grading": grading, **parts}


def show_table(work, device, what):
    """torch.profiler's table of the slowest parts of 5 runs of work()."""
    if True:
        from torch.profiler import ProfilerActivity, profile
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda"
                                               else [])
        with profile(activities=activities) as prof:
            for _ in range(5):
                work()
            wait()
        key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
        print(f"  profile of {what}:")
        print(prof.key_averages().table(sort_by=key, row_limit=14, max_name_column_width=45))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure the speed of the group trainer.")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--plain", action="store_true", help="without torch.compile")
    parser.add_argument("--sums-up-to", type=int, default=None,
                        help="batch.ATTENTION_SUMS_UP_TO for this measurement")
    parser.add_argument("--no-kernel", action="store_true", help="without v2/tiny_attention.py")
    parser.add_argument("--decompose", action="store_true",
                        help="let the compiler write small matrix products as sums")
    parser.add_argument("--shapes", default="all", help="e.g. 4,8,128;64,64,128")
    args = parser.parse_args(argv)
    if args.plain:
        batch.COMPILE_ON_CARD = False
    if args.sums_up_to is not None:
        batch.ATTENTION_SUMS_UP_TO = args.sums_up_to
    if args.no_kernel:
        batch.ATTENTION_KERNEL_ON_CARD = False
    if args.decompose:
        import torch._inductor.config as inductor
        inductor.decompose_mem_bound_mm = True
    print(f"settings: compile {batch.COMPILE_ON_CARD}, attention sums up to head size "
          f"{batch.ATTENTION_SUMS_UP_TO}, decompose {args.decompose}, attention kernel "
          f"{batch.ATTENTION_KERNEL_ON_CARD}", flush=True)
    shapes = SHAPES if args.shapes == "all" else [
        tuple(int(x) for x in part.split(",")) for part in args.shapes.split(";")]
    if torch.cuda.is_available():
        print("card:", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
    return [measure(*shape, args.repeats, args.table) for shape in shapes]


if __name__ == "__main__":
    main()
