"""
v3/session_job.py - what runs inside one Kaggle session of the third experiment (v3/PLAN.md B.3).

The program on the owner's computer (v3/orchestrate.py) sends a script that unpacks the code and
then calls

    run_session(groups, stage, budget_minutes)

groups          a list of groups, each a list of configs (v3/grid.py)
stage           "practice", "counted" or "test"
budget_minutes  how long the session may last; it stops cleanly before that

What it does, in order
----------------------
1. Earlier results: the outputs of earlier sessions are attached to this one as "kernel
   sources", so they appear (read-only) under /kaggle/input. Every earlier output folder (it
   holds a session.json) is searched for summaries and model.pt of this session's runs, and for
   checkpoints of its groups. They are copied into /kaggle/working/out, so finished runs are
   skipped and stopped groups go on from their checkpoint. When several earlier sessions have a
   checkpoint of the same group, the one furthest on (the highest step) is taken.
2. One worker process per graphics card (one on the processor when there is no card). The groups
   wait as small files in out/_tasks; a worker takes the next one by renaming its file (a rename
   cannot succeed twice, so no group is ever taken by two workers), trains it with the group
   trainer (v2/batch.py) on its own card until it is done or the budget is used, writes a report,
   and takes the next.
3. At the end: out/status.json (finished runs, unfinished groups, minutes used, member-steps per
   second on every card) and a short summary printed in the log.

Where things go
---------------
    out/<stage>/<name>/summary.json, model.pt    results
    out/_groups/<group name>/checkpoint.pt       checkpoints of unfinished groups
    out/session.json                              what this session was given (written first)
    out/status.json                               what it did (written last)
    out/logs/worker<n>.log                        what each worker printed

The paths /kaggle/working and /kaggle/input can be changed (for tests on this computer) with the
arguments of run_session.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # the folder that holds v2/ and v3/
WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

END_RESERVE_SECONDS = 180  # kept free at the end, to write the status and save the output
MIN_START_SECONDS = 120  # a worker does not start a new group with less time than this left
THREADS_PER_WORKER = 2  # processor threads of each worker on a card (a session has 4 cores)


def say(*args):
    print(*args, flush=True)


# ------------------------------------------------------------------------------------------------
# 1. Earlier results
# ------------------------------------------------------------------------------------------------


def earlier_outputs(input_root):
    """The output folders of earlier sessions found under input_root (each holds a session.json)."""
    input_root = Path(input_root)
    if not input_root.exists():
        return []
    return sorted({path.parent for path in input_root.rglob("session.json")})


def checkpoint_step(path):
    """The step at which a group checkpoint was saved (read on the processor)."""
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)["step"]


def bring_in(groups, stage, input_root, out):
    """Copy the earlier results and checkpoints that this session's groups need into out.
    Returns how many runs and checkpoints were copied."""
    from v2.batch import group_name
    from v2.run import complete_config

    earlier = earlier_outputs(input_root)
    runs = checkpoints = 0
    for configs in groups:
        for config in configs:
            target = out / stage / config["name"]
            if (target / "summary.json").exists():
                continue
            for folder in earlier:
                source = folder / stage / config["name"]
                if (source / "summary.json").exists() and (source / "model.pt").exists():
                    target.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / "model.pt", target / "model.pt")
                    shutil.copy2(source / "summary.json", target / "summary.json")  # last
                    runs += 1
                    break
        name = group_name([complete_config(c) for c in configs])
        found = [folder / "_groups" / name / "checkpoint.pt" for folder in earlier]
        found = [path for path in found if path.exists()]
        target = out / "_groups" / name / "checkpoint.pt"
        if found and not target.exists():
            best = max(found, key=checkpoint_step)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best, target)
            checkpoints += 1
    return runs, checkpoints


# ------------------------------------------------------------------------------------------------
# 2. Workers
# ------------------------------------------------------------------------------------------------


def take_next(out, number):
    """Take the next group nobody has taken: (its task file, the job), or (None, None)."""
    for task in sorted((out / "_tasks").glob("*.json")):
        claimed = out / "_taken" / f"{task.stem}.worker{number}.json"
        try:
            os.rename(task, claimed)  # only one worker can do this for a given file
        except FileNotFoundError:
            continue  # another worker took it first
        return claimed, json.loads(claimed.read_text())
    return None, None


def worker(number, device, out, stage, deadline, threads, min_start=MIN_START_SECONDS,
           safety=None):
    """One worker: takes groups from out/_tasks until there are none left or time is short.
    safety (for tests): the seconds the group trainer keeps free before the deadline."""
    import torch
    import v2.batch
    from v2.batch import run_group_until

    torch.set_num_threads(threads)
    if safety is not None:
        v2.batch.DEADLINE_SAFETY_SECONDS = safety
    name = f"cuda:{number}" if device == "cuda" else f"{device}:{number}"
    if device == "cuda":
        say(f"worker {number}: card {torch.cuda.get_device_name(0)}")
    while time.time() + min_start < deadline:
        _, job = take_next(out, number)
        if job is None:
            break
        index, configs = job["group"], job["configs"]
        say(f"worker {number}: group {index} ({len(configs)} members) at "
            f"{time.strftime('%H:%M:%S')}")
        report = {"group": index, "worker": number, "device": name}
        try:
            report.update(run_group_until(configs, out / stage, device, deadline=deadline,
                                          groups_root=out / "_groups"))
        except Exception:  # a broken group must not stop the others
            report["error"] = traceback.format_exc()
            say(report["error"])
        (out / "_reports" / f"{index:04d}.json").write_text(json.dumps(report, indent=1))
    say(f"worker {number}: done at {time.strftime('%H:%M:%S')}")


def card_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def start_worker(number, device, out, stage, deadline, threads, min_start, safety):
    """Start one worker as a separate Python process. A graphics card is given to it alone
    (CUDA_VISIBLE_DEVICES), so inside the worker it is simply "cuda"."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    if device == "cuda":
        # One card per worker on Kaggle. Where several workers share a card (a large card on
        # Colab, see AIR_WORKERS_PER_CARD below), worker n uses card n modulo the number of cards.
        environment["CUDA_VISIBLE_DEVICES"] = str(number % max(1, card_count()))
    log = open(out / "logs" / f"worker{number}.log", "w")
    command = [sys.executable, "-m", "v3.session_job", "worker", "--number", str(number),
               "--device", device, "--out", str(out), "--stage", stage,
               "--deadline", repr(deadline), "--threads", str(threads),
               "--min-start", repr(min_start)]
    if safety is not None:
        command += ["--safety", repr(safety)]
    return subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                            stderr=subprocess.STDOUT)


# ------------------------------------------------------------------------------------------------
# 3. The session
# ------------------------------------------------------------------------------------------------


def run_session(groups, stage, budget_minutes, input_root=INPUT, working=WORKING, devices=None,
                started=None, end_reserve=END_RESERVE_SECONDS, min_start=MIN_START_SECONDS,
                safety=None):
    """Train the groups until done or until the budget is used. Returns the status (also written
    to out/status.json). devices: for tests, for example ["cpu", "cpu"] (default: one worker per
    graphics card, or one on the processor when there is none). end_reserve, min_start and
    safety: see the constants at the top and worker (smaller values for tests)."""
    started = started or time.time()
    deadline = started + budget_minutes * 60 - end_reserve
    out = Path(working) / "out"
    for folder in ("_tasks", "_taken", "_reports", "logs", stage, "_groups"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    (out / "session.json").write_text(json.dumps(
        {"stage": stage, "budget_minutes": budget_minutes, "started": started,
         "groups": [[c["name"] for c in configs] for configs in groups]}, indent=1))

    runs, checkpoints = bring_in(groups, stage, input_root, out)
    say(f"session: stage {stage}, {len(groups)} groups, budget {budget_minutes:.0f} minutes; "
        f"brought in {runs} finished runs and {checkpoints} checkpoints")

    for index, configs in enumerate(groups):
        (out / "_tasks" / f"{index:04d}.json").write_text(
            json.dumps({"group": index, "configs": configs}))
    if devices is None:
        cards = card_count()
        # AIR_WORKERS_PER_CARD (default 1): a large card can train several groups at once.
        per_card = max(1, int(os.environ.get("AIR_WORKERS_PER_CARD", "1")))
        devices = ["cuda"] * (cards * per_card) if cards else ["cpu"]
    threads = int(os.environ.get("AIR_THREADS_PER_WORKER", THREADS_PER_WORKER)) \
        if devices[0] == "cuda" else 1
    say(f"session: {len(devices)} workers on {', '.join(devices)}")
    workers = [start_worker(n, d, out, stage, deadline, threads, min_start, safety)
               for n, d in enumerate(devices)]
    # The workers stop by themselves before the deadline. One that has not stopped a little
    # after it (half the reserve, at least 5 seconds) is stopped; its group goes on from its last
    # checkpoint in the next session.
    stop_at = deadline + max(5.0, end_reserve / 2)
    for process in workers:
        try:
            process.wait(timeout=max(1.0, stop_at - time.time()))
        except subprocess.TimeoutExpired:
            say("session: a worker did not stop in time and is stopped now")
            process.kill()
    # Tasks nobody took are put back, so the folder shows only what was done.
    for task in (out / "_tasks").glob("*.json"):
        task.unlink()
    for number in range(len(devices)):
        say(f"----- worker {number} -----")
        say((out / "logs" / f"worker{number}.log").read_text()[-6000:])
    status = write_status(groups, stage, out, started, budget_minutes)
    print_summary(status)
    return status


def write_status(groups, stage, out, started, budget_minutes):
    """out/status.json: what this session did."""
    from v2.batch import group_name
    from v2.run import complete_config

    reports = {}
    for path in sorted((out / "_reports").glob("*.json")):
        report = json.loads(path.read_text())
        reports[report["group"]] = report
    finished, unfinished_groups, failed_groups, speed = [], [], [], {}
    for index, configs in enumerate(groups):
        names = [c["name"] for c in configs]
        done = [n for n in names if (out / stage / n / "summary.json").exists()]
        finished += done
        report = reports.get(index)
        if report and "error" in report:
            failed_groups.append({"group": index, "names": names,
                                  "error": report["error"][-3000:]})
        left = [n for n in names if n not in done]
        if left:
            name = group_name([complete_config(c) for c in configs])
            unfinished_groups.append({"group": index, "unfinished": left,
                                      "started": report is not None,
                                      # the step the group reached in this session (None: no
                                      # training done here)
                                      "step": report.get("step") if report else None,
                                      "checkpoint": (out / "_groups" / name /
                                                     "checkpoint.pt").exists()})
        if report and report.get("seconds"):
            card = speed.setdefault(report["device"], {"member_steps": 0, "seconds": 0.0})
            card["member_steps"] += report.get("member_steps", 0)
            card["seconds"] += report["seconds"]
    for card in speed.values():
        card["per_second"] = card["member_steps"] / card["seconds"] if card["seconds"] else 0.0
    status = {"stage": stage, "budget_minutes": budget_minutes,
              "minutes": (time.time() - started) / 60, "groups": len(groups),
              "finished": finished, "unfinished_groups": unfinished_groups,
              "failed_groups": failed_groups, "speed": speed}
    (out / "status.json").write_text(json.dumps(status, indent=1))
    return status


def print_summary(status):
    say(f"session done after {status['minutes']:.1f} of {status['budget_minutes']:.0f} minutes: "
        f"{len(status['finished'])} runs finished, {len(status['unfinished_groups'])} of "
        f"{status['groups']} groups unfinished, {len(status['failed_groups'])} failed")
    for device, card in sorted(status["speed"].items()):
        say(f"  {device}: {card['per_second']:,.0f} member-steps per second "
            f"({card['member_steps']:,} member-steps in {card['seconds'] / 60:.1f} minutes)")
    for failed in status["failed_groups"]:
        say(f"  group {failed['group']} failed: {failed['error'].strip().splitlines()[-1]}")


def main(argv=None):
    """Only used to start a worker process (see start_worker)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["worker"])
    parser.add_argument("--number", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--threads", type=int, default=THREADS_PER_WORKER)
    parser.add_argument("--min-start", type=float, default=MIN_START_SECONDS)
    parser.add_argument("--safety", type=float, default=None)
    args = parser.parse_args(argv)
    worker(args.number, args.device, Path(args.out), args.stage, args.deadline, args.threads,
           args.min_start, args.safety)


if __name__ == "__main__":
    main()
