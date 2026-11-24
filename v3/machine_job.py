"""
v3/machine_job.py - what runs on ONE rented machine (a Colab machine) of the fleet
(v3/colab_fleet.py): it trains the groups it is given, breadth first.

    python -m v3.machine_job --out /content/air_run/out --workers 2

Breadth first
-------------
A group of the doubling rule (v3/PLAN.md B.1) is looked at when it reaches 40,000 steps and each
time its length has doubled. Here every group is first trained to its first look (40,000 steps),
then the groups with unfinished members to the next look (80,000), then 160,000, and so on. So
a first picture of the whole stage comes early. A worker always takes the waiting group that is
least far on (ties: the lower number), trains it up to its next look, saves its checkpoint and
puts it back. Members that end are written at once, as always. Groups with a fixed length (the
practice stage) are trained to their end in one go.

The folder `out` (the controller on the owner's computer reads and writes it)
---------------------------------------------------------------------------
    tasks/<n>.json           a group waiting to be trained: {"group": n, "stage": .., "configs": [..]}
    taken/<n>.w<k>.json      the same file while worker k trains it (taken by renaming)
    progress/<n>.json        {"step": reached, "done": true/false, "error": text or absent}
    <stage>/<name>/          summary.json and model.pt of every run that has ended
    _so_far/<name>.json      scores so far of every run still training (removed when it ends)
    _groups/<group>/checkpoint.pt   the group trainer's checkpoints
    handoff/<n>.pt           a copy of group n's checkpoint, made each time it is put back
    close                    (written by the controller) no more groups will come: stop when idle
    stop                     (written by the controller) stop now: every group saves and is put back
    park/<n>                 (written by the controller) group n goes to a faster machine: the
                             worker training it saves as for `stop`, but puts its task file into
                             parked/ instead of tasks/ (so no worker here takes it again) and
                             removes the signal. A group waiting in tasks/ is parked by the next
                             worker that looks for work. A finished group just loses the signal.
    parked/<n>.json          a parked group: the controller deletes it once the checkpoint is home
    abilities.json           written when the job starts: {"pid": .., "park": true}. A job started
                             by an older version of this file has none, so the controller knows
                             it cannot ask that machine to park one group.
    logs/worker<k>.log       what each worker printed
    job.json                 written when the job ends
A group that is taken away by the controller (to give it to another machine) is simply renamed
out of tasks/ (or parked/) by it; a rename succeeds only once, so a group is never trained twice.
A task file is always in exactly one of tasks/, taken/ and parked/, and a worker trains a group
only while its file is in taken/.
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
IDLE_SECONDS = 5  # an idle worker looks for new groups this often
WORKER_RESTARTS = 5  # crashed workers started again, in all
FOLDERS = ("tasks", "taken", "progress", "_so_far", "_groups", "handoff", "logs", "park", "parked")


def say(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, value):
    """Write through a temporary file, so a reader never sees half a file."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1))
    os.replace(temporary, path)


def next_pause(config, step):
    """The step up to which a group at `step` is trained in this round: its next look (doubling
    rule), or None (train to the end) for a fixed length. config: a complete config."""
    if config.get("length_rule") != "doubling":
        return None
    look = config["first_look"]
    while look <= step:
        look *= 2
    return min(look, config["steps"])


def reached(out, number):
    return read_json(Path(out) / "progress" / f"{number}.json", {}).get("step") or 0


def park_signal(out, number):
    """The file by which the controller asks for group `number` to be parked (see the top)."""
    return Path(out) / "park" / str(number)


def drop_park_signal(out, number):
    # (The controller may have removed it at the same moment: that is fine.)
    park_signal(out, number).unlink(missing_ok=True)


def take_next(out, worker):
    """Take the waiting group least far on: (its file now in taken/, the task) or (None, None).
    A waiting group that the controller asked to park is parked instead of taken."""
    out = Path(out)
    waiting = []
    for task in out.glob("tasks/*.json"):
        number = int(task.stem)
        waiting.append((reached(out, number), number, task))
    for _, number, task in sorted(waiting):
        if park_signal(out, number).exists():
            # It finished its round before a worker saw the signal. Its checkpoint and progress
            # were written then, so it can go to the faster machine straight from the queue.
            (out / "parked").mkdir(exist_ok=True)
            try:
                os.rename(task, out / "parked" / f"{number}.json")  # only one can do this
            except FileNotFoundError:
                continue  # another worker (or the controller) took it first
            drop_park_signal(out, number)
            say(f"group {number}: parked (it goes to another machine)")
            continue
        claimed = out / "taken" / f"{number}.w{worker}.json"
        try:
            os.rename(task, claimed)  # only one can do this for a given file
        except FileNotFoundError:
            continue  # another worker (or the controller) took it first
        return claimed, read_json(claimed)
    return None, None


def train_one_round(task, out, device):
    """Train the group of `task` up to its next pause. Returns its progress."""
    from v2.batch import group_name, run_group_until
    from v2.run import complete_config

    out = Path(out)
    configs = task["configs"]
    complete = [complete_config(c) for c in configs]
    number = task["group"]
    step = reached(out, number)
    pause = next_pause(complete[0], step)
    say(f"group {number}: from step {step:,} to {pause or complete[0]['steps']:,}")
    # Stop (and save) when the whole job is told to stop, or when only this group is asked to
    # park because a faster machine is waiting for it.
    result = run_group_until(configs, out / task["stage"], device, groups_root=out / "_groups",
                             pause_at=pause, so_far_root=out / "_so_far",
                             stop_check=lambda: ((out / "stop").exists()
                                                 or park_signal(out, number).exists()))
    progress = {"step": result["step"] if result["step"] is not None else step,
                "done": not result["unfinished"], "member_steps": result["member_steps"],
                "seconds": result["seconds"], "unfinished": result["unfinished"]}
    if result["unfinished"]:  # a copy of the checkpoint for the controller (see the top)
        checkpoint = out / "_groups" / group_name(complete) / "checkpoint.pt"
        if checkpoint.exists():
            handoff = out / "handoff" / f"{number}.pt"
            shutil.copy(checkpoint, handoff.with_suffix(".part"))  # dated now (see POLL)
            os.replace(handoff.with_suffix(".part"), handoff)
    return progress


def free_card_memory():
    """Give back the card memory this worker no longer needs, between two groups. PyTorch keeps
    whatever it once used. With many workers sharing one card (NVIDIA's MPS), the memory kept by
    16 workers filled an 80 GB H100 on 23 November 2026, and new groups failed for lack of it."""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # (never a reason to stop a worker)
        pass


def worker(number, out, device):
    """Take groups until there are none and the controller has said "close", or until "stop"."""
    out = Path(out)
    while not (out / "stop").exists():
        claimed, task = take_next(out, number)
        if task is None:
            if (out / "close").exists():
                break
            time.sleep(IDLE_SECONDS)
            continue
        index = task["group"]
        try:
            progress = train_one_round(task, out, device)
        except Exception:  # a broken group must not stop the others
            error = traceback.format_exc()
            say(error)
            progress = dict(read_json(out / "progress" / f"{index}.json", {}), error=error)
            progress.setdefault("step", 0)
            progress["done"] = False
        free_card_memory()
        # The progress (and the hand-off checkpoint, in train_one_round) are written before the
        # task file is moved, so a group seen in tasks/ or parked/ has its final files.
        write_json(out / "progress" / f"{index}.json", progress)
        parked = False
        if progress["done"] or "error" in progress:
            claimed.unlink()  # finished (or failed: the controller decides what to do)
            drop_park_signal(out, index)  # (a signal that came too late means nothing now)
        elif park_signal(out, index).exists():
            # Asked to park: set aside, not back in the queue, so no worker here trains it again.
            (out / "parked").mkdir(exist_ok=True)
            os.replace(claimed, out / "parked" / f"{index}.json")
            drop_park_signal(out, index)
            parked = True
        else:
            os.replace(claimed, out / "tasks" / f"{index}.json")  # back in the queue
        say(f"group {index}: at step {progress['step']:,}"
            + (", done" if progress["done"] else "") + (", FAILED" if "error" in progress else "")
            + (", parked (it goes to another machine)" if parked else ""))
    say(f"worker {number}: done")


def run_job(out, workers, device=None):
    """Start the workers (separate programs, so that one compiling does not hold up the
    others on the card) and wait for them. Groups left in taken/ by an earlier job that was
    cut short are put back first."""
    out = Path(out)
    for folder in FOLDERS:
        (out / folder).mkdir(parents=True, exist_ok=True)
    for left in (out / "taken").glob("*.json"):
        os.replace(left, out / "tasks" / f"{left.name.split('.')[0]}.json")
    # Say what this job can do. The controller's look reads it (and checks that this process is
    # still the one running), so it never sends a park signal to a job that would ignore it.
    write_json(out / "abilities.json", {"pid": os.getpid(), "park": True, "started": time.time()})
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")

    def start(number):
        log = open(out / "logs" / f"worker{number}.log", "a")
        return subprocess.Popen(
            [sys.executable, "-m", "v3.machine_job", "--worker", str(number), "--out", str(out),
             "--device", device], cwd=ROOT, env=environment, stdout=log,
            stderr=subprocess.STDOUT)

    processes = {number: start(number) for number in range(workers)}
    say(f"job: {workers} workers on {device}")
    # A worker that crashed (not a Python error in a group, which it survives, but for example
    # killed for lack of memory) would leave its group in taken/ for ever. Its group is put back
    # and the worker started again (at most WORKER_RESTARTS times in all).
    restarts = 0
    while processes:
        for number, process in list(processes.items()):
            code = process.poll()
            if code is None:
                continue
            del processes[number]
            if code != 0 and not (out / "stop").exists() and restarts < WORKER_RESTARTS:
                for left in (out / "taken").glob(f"*.w{number}.json"):
                    os.replace(left, out / "tasks" / f"{left.name.split('.')[0]}.json")
                restarts += 1
                say(f"worker {number} ended with code {code}; started again")
                processes[number] = start(number)
        time.sleep(1)
    write_json(out / "job.json", {"ended": time.time(), "stopped": (out / "stop").exists()})
    say("job: all workers done")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=2, help="groups trained at once")
    parser.add_argument("--worker", type=int, default=None, help="(inside) be worker number n")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if args.worker is not None:
        if args.device == "cpu":
            import torch
            torch.set_num_threads(1)
        worker(args.worker, args.out, args.device)
    else:
        run_job(args.out, args.workers, args.device)


if __name__ == "__main__":
    main()
