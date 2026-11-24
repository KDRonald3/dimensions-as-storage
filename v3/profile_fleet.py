"""
v3/profile_fleet.py - how many member-steps a second one card gives when it trains SEVERAL
groups in one program (each on its own card stream), with and without the recorded step
(CUDA graphs) and with other settings of the compiler (--mode, --tf32, ...).

    python -m v3.profile_fleet --groups "8,4,128x4"            # four groups of 24 at once
    python -m v3.profile_fleet --groups "64,64,128x2" --graph 0
    python -m v3.profile_fleet --groups "8,4,4x8" --members 6  # like the practice groups

--groups: a list "columns,kinds,body x copies; ...". Every group has `members` members (24 as in
the counted stage, or 6 as in practice). Gradings are not counted; one grading is timed apart.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="8,4,128x1")
    parser.add_argument("--members", type=int, default=24)
    parser.add_argument("--graph", type=int, default=1)
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--tf32", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=20.0, help="measured time")
    parser.add_argument("--grade", type=int, default=0, help="also time one grading")
    parser.add_argument("--parts", type=int, default=0, help="also time sentences and step apart")
    parser.add_argument("--table", type=int, default=0, help="torch.profiler table")
    parser.add_argument("--mode", default="default", help="torch.compile mode")
    parser.add_argument("--blas", default="", help="cublas or cublaslt")
    parser.add_argument("--decompose", type=int, default=0)
    parser.add_argument("--check-pause", type=int, default=0,
                        help="check pausing and the recorded step against plain training")
    parser.add_argument("--train-steps", type=int, default=0,
                        help="train the first group to this step and print its scores")
    parser.add_argument("--gemm-backends", default="", help="e.g. TRITON")
    parser.add_argument("--pieces", type=int, default=0, help="batch.WEIGHT_SUM_PIECES")
    parser.add_argument("--text-steps", type=int, default=None,
                        help="batch.TEXT_STEPS_AT_ONCE for this measurement")
    args = parser.parse_args(argv)

    from v2 import batch
    batch.GRAPH_ON_CARD = bool(args.graph)
    batch.LOWER_PRECISION_ON_CARD = bool(args.tf32)
    batch.COMPILE_MODE = "" if args.mode == "default" else args.mode
    if args.decompose:
        import torch._inductor.config as inductor
        inductor.decompose_mem_bound_mm = True
    if args.gemm_backends:
        import torch._inductor.config as inductor
        inductor.max_autotune_gemm_backends = args.gemm_backends
    if args.blas and torch.cuda.is_available():
        torch.backends.cuda.preferred_blas_library(args.blas)
    if args.pieces:
        batch.WEIGHT_SUM_PIECES = args.pieces
    if args.text_steps:
        batch.TEXT_STEPS_AT_ONCE = args.text_steps
    from v3.profile_speed import configs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.check_pause:
        check_pause(args.members, device)
        return
    trainers = []
    for part in args.groups.split(";"):
        shape, copies = part.split("x")
        columns, kinds, body = (int(x) for x in shape.split(","))
        for copy in range(int(copies)):
            members = configs(kinds, columns, body)[:args.members]
            for config in members:
                config["name"] += f"_{copy}"
            trainer = batch.GroupTrainer(members, device)
            trainer.stream = (torch.cuda.Stream(device) if device.type == "cuda" and args.streams
                              else None)
            trainers.append(trainer)

    def round_of_steps():
        for trainer in trainers:
            if trainer.stream is not None:
                with torch.cuda.stream(trainer.stream):
                    trainer.train_step(trainer.make_text(trainer.step))
            else:
                trainer.train_step(trainer.make_text(trainer.step))

    def wait():
        if device.type == "cuda":
            torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(batch.GRAPH_WARMUP_STEPS + 3):  # compiling and recording
        round_of_steps()
    wait()
    if args.parts:  # the sentence maker and the step on their own, in the first group
        first = trainers[0]
        for block in (64, 256, 1024):
            begin = time.perf_counter()
            first.text.make_steps(range(10000, 10000 + block))
            wait()
            middle = time.perf_counter()
            first.text.make_steps(range(20000, 20000 + block))
            wait()
            print(f"  sentences, {block} steps at once: {1000 * (time.perf_counter() - middle) / block:.2f}"
                  f" ms a step (first time {1000 * (middle - begin) / block:.2f})", flush=True)
        tokens = first.make_text(first.step)
        for _ in range(3):
            first.train_step(tokens)
        wait()
        begin = time.perf_counter()
        for _ in range(50):
            first.train_step(tokens)
        launched = time.perf_counter()
        wait()
        print(f"  training step alone: {1000 * (time.perf_counter() - begin) / 50:.2f} ms "
              f"(the processor is done handing it over after {1000 * (launched - begin) / 50:.2f} ms)",
              flush=True)
    warm = time.perf_counter() - start
    rounds, start = 0, time.perf_counter()
    while time.perf_counter() - start < args.seconds:
        for _ in range(10):
            round_of_steps()
        rounds += 10
        wait()
    spent = time.perf_counter() - start
    member_steps = rounds * sum(t.members for t in trainers)
    print(f"RESULT groups={args.groups} members={args.members} graph={args.graph} "
          f"streams={args.streams} tf32={args.tf32}: {member_steps / spent:,.0f} member-steps a "
          f"second in total ({1000 * spent / rounds:.1f} ms a round of {len(trainers)} groups; "
          f"setting up {warm:.0f} s)", flush=True)
    if args.train_steps:  # the first group trained on, then graded: its scores (for TF32)
        first = trainers[0]
        while first.step < args.train_steps:
            first.train_step(first.make_text(first.step))
        first.grade_all()
        for config, final in zip(first.configs, first.final):
            print(f"  SCORE {config['name']} step {first.step}: facts {final['facts']:.2f} "
                  f"one {final['questions_one']:.2f} loss {final['loss']:.5f}", flush=True)
    if args.table:  # torch.profiler: the card's slowest parts in 5 rounds
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                round_of_steps()
            wait()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=18,
                                        max_name_column_width=50), flush=True)
    if args.grade:
        start = time.perf_counter()
        trainers[0].grade_all()
        wait()
        print(f"  one grading of {trainers[0].members} members: "
              f"{time.perf_counter() - start:.2f} s", flush=True)
    if device.type == "cuda":
        print(f"  card memory used at most {torch.cuda.max_memory_allocated() / 1e9:.1f} GB",
              flush=True)


def check_pause(members, device):
    """On the card: a group trained straight to 2,000 steps, the same group paused at its look
    at 1,000 and picked up from the checkpoint (recorded step on), and the same without the
    recorded step. Prints the final loss and facts of each member in each way: they must agree
    up to rounding."""
    import tempfile
    from v2 import batch
    from v3.profile_speed import configs
    group = [dict(c, first_look=1000, steps=2000) for c in configs(4, 8, 128)[:members]]
    folder = Path(tempfile.mkdtemp())
    finals = {}
    for way in ("straight", "paused", "no graph"):
        batch.GRAPH_ON_CARD = way != "no graph"
        out = folder / way.replace(" ", "_")
        if way == "paused":
            batch.run_group_until(group, out, device, pause_at=1000, quiet=True)
        batch.run_group_until(group, out, device, quiet=True)
        finals[way] = [json.loads((out / c["name"] / "summary.json").read_text())["final"]
                       for c in group]
    for i, config in enumerate(group):
        print("  CHECK " + config["name"] + "  " + "  ".join(
            f"{way}: loss {finals[way][i]['loss']:.5f} facts {finals[way][i]['facts']:.1f}"
            for way in finals), flush=True)


if __name__ == "__main__":
    main()
