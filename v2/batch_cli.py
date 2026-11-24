"""
v2/batch_cli.py - train a whole stage with the group trainer (v2/batch.py).

    python -m v2.batch_cli main8
    python -m v2.batch_cli main8 --only body=8+16,usage=zipf --device cuda
    python -m v2.batch_cli bits16 --out-root runs_v2_batch --max-members 24

The configs of the stage come from v2.grid.grid. They are sorted into groups (runs that can be
trained together) and the groups are trained one after another. Results go to
OUT_ROOT/STAGE/NAME/, in the same form as v2/run.py writes them. Finished runs are skipped, so
the same command can be started again after a stop; a stopped group goes on from its last
checkpoint.

--only key=v1+v2,key2=v3   keep only the runs whose key has one of the values given
--max-members N            split groups larger than N (to fit in the memory of the card)
--threads N                processor threads; use a few on a busy computer (default: all)
--steps N                  timing tests only: train every run for N steps instead. The results
                           then go to runs_v2_batch_timing (unless --out-root is given), so they
                           are never mixed with real runs.

A timing test of one small group (8 runs, body width 128, 1,000 steps):

    python -m v2.batch_cli main8 --only body=128,seed=0 --steps 1000 --device cuda
"""

import argparse
import time

import torch

from v2.batch import default_device, make_groups, train_group
from v2.grid import grid


def parse_only(text):
    """"body=8+16,usage=zipf" -> {"body": {"8", "16"}, "usage": {"zipf"}}"""
    wanted = {}
    for part in filter(None, (text or "").split(",")):
        key, _, values = part.partition("=")
        if not values:
            raise SystemExit(f"--only: write key=value, not {part!r}")
        wanted[key.strip()] = {v.strip() for v in values.split("+")}
    return wanted


def keep(config, wanted):
    return all(str(config.get(key)) in values for key, values in wanted.items())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a stage with the group trainer.")
    parser.add_argument("stage")
    parser.add_argument("--only", default="")
    parser.add_argument("--out-root", default=None,
                        help="default: runs_v2_batch (runs_v2_batch_timing with --steps)")
    parser.add_argument("--device", default=None, help='"cuda" or "cpu" (default: cuda if '
                                                       'there is one)')
    parser.add_argument("--max-members", type=int, default=0, help="0 = no limit")
    parser.add_argument("--steps", type=int, default=0, help="timing tests only (see above)")
    parser.add_argument("--threads", type=int, default=0,
                        help="processor threads (0 = PyTorch's choice: every core)")
    args = parser.parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.out_root is None:
        args.out_root = "runs_v2_batch_timing" if args.steps else "runs_v2_batch"

    configs = [c for c in grid(args.stage) if keep(c, parse_only(args.only))]
    if args.steps:
        configs = [dict(c, steps=args.steps) for c in configs]
    if not configs:
        raise SystemExit("no runs left after --only")
    groups = make_groups(configs)
    if args.max_members > 0:
        groups = [group[i:i + args.max_members] for group in groups
                  for i in range(0, len(group), args.max_members)]
    device = args.device or default_device()
    out_root = f"{args.out_root}/{args.stage}"
    print(f"{args.stage}: {len(configs)} runs in {len(groups)} groups, on {device}",
          flush=True)
    if device == "cuda":
        print(f"graphics card: {torch.cuda.get_device_name(0)}", flush=True)

    all_member_steps, all_seconds = 0, 0.0
    for number, group in enumerate(groups, 1):
        print(f"group {number} of {len(groups)}: {len(group)} members", flush=True)
        started = time.perf_counter()
        summaries, state = train_group(group, out_root, device)
        seconds = time.perf_counter() - started
        for summary in summaries:
            final = summary["final"]
            print(f"  {summary['name']}: facts {final['facts']:.1f}, questions one "
                  f"{final['questions_one']:.1f}, new {final['questions_new']:.1f}, seen "
                  f"{final['questions_seen']:.1f}, not {final['not_known']:.1f}", flush=True)
        if state is None:
            print("  (all finished before; nothing trained)", flush=True)
            continue
        # Member-steps: one step of one member. Only the steps trained in this sitting count.
        member_steps = len(state["configs"]) * (state["step"] - state["first_step"])
        if seconds > 0:
            print(f"  speed: {member_steps / seconds:,.0f} member-steps per second "
                  f"({seconds / 60:.1f} minutes, grading included)", flush=True)
        all_member_steps += member_steps
        all_seconds += seconds
    if all_seconds > 0:
        print(f"all groups: {all_member_steps / all_seconds:,.0f} member-steps per second",
              flush=True)


if __name__ == "__main__":
    main()
