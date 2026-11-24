"""Run runs of the second experiment on THIS computer, several at a time, so that its
processors are never idle while work is waiting.

    .venv/bin/python -m v2.local_queue pilot2
    .venv/bin/python -m v2.local_queue bits32:seed=2 bits64:body=128

Each argument is a stage of v2/grid.py, optionally cut down to the runs whose setting has one of
the given values (several values are joined with "+", for example body=64+128).

The runs are the same programs as on the cloud machines (v2.run.run_one). Results go to
runs_v2/<stage>/<name>/, the folder the cloud results are downloaded into as well. A run that is
already finished there is skipped, and a run that was interrupted goes on from its checkpoint.

The computer is cut into slots, each with its own processors, so runs do not fight over them:
the 8 fast cores make 4 slots and the 16 slow cores make 8 slots. Fast slots take the heaviest
runs first, slow slots take the lightest first.
"""

import collections
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from v2.grid import grid

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python"
LOG_DIR = ROOT / "logs" / "v2" / "local"

FAST_SLOTS = ["0-3", "4-7", "8-11", "12-15"]               # 2 fast cores (4 threads of them) each
SLOW_SLOTS = [f"{first}-{first + 1}" for first in range(16, 32, 2)]   # 2 slow cores each
THREADS = 2

# What a child process runs: one run, from its settings given as JSON.
CHILD = "import json, sys; from v2.run import run_one; run_one(json.loads(sys.argv[1]), sys.argv[2])"


def wanted_runs(argument):
    """'bits32:seed=2' -> the (stage, config) pairs of stage bits32 with seed 2."""
    stage, _, rule = argument.partition(":")
    configs = grid(stage)
    for part in filter(None, rule.split(",")):
        key, _, values = part.partition("=")
        allowed = set(values.split("+"))
        configs = [config for config in configs if str(config[key]) in allowed]
    return [(stage, config) for config in configs]


def weight(item):
    """How heavy a run is: longer and with a larger body is heavier."""
    _, config = item
    return config["steps"] * (config["body"] + 40) * config["kinds"]


def work(slot, fast, todo, lock):
    while True:
        with lock:
            if not todo:
                return
            stage, config = todo.popleft() if fast else todo.pop()
        config = dict(config, threads=THREADS)
        out_root = ROOT / "runs_v2" / stage
        if (out_root / config["name"] / "summary.json").exists():
            print(f"{time.strftime('%H:%M:%S')} skip   {stage}/{config['name']} (already finished)",
                  flush=True)
            continue
        started = time.time()
        with open(LOG_DIR / f"{stage}_{config['name']}.log", "w") as log:
            code = subprocess.call(
                ["taskset", "-c", slot, str(PYTHON), "-u", "-c", CHILD,
                 json.dumps(config), str(out_root)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        word = "done  " if code == 0 else "FAILED"
        print(f"{time.strftime('%H:%M:%S')} {word} {stage}/{config['name']} on processors {slot} "
              f"in {(time.time() - started) / 60:.1f} min", flush=True)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    runs = [item for argument in sys.argv[1:] for item in wanted_runs(argument)]
    todo = collections.deque(sorted(runs, key=weight, reverse=True))   # heaviest first
    lock = threading.Lock()
    print(f"{time.strftime('%H:%M:%S')} {len(todo)} runs for this computer, "
          f"{len(FAST_SLOTS) + len(SLOW_SLOTS)} at a time", flush=True)
    workers = [threading.Thread(target=work, args=(slot, True, todo, lock)) for slot in FAST_SLOTS]
    workers += [threading.Thread(target=work, args=(slot, False, todo, lock)) for slot in SLOW_SLOTS]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    print(f"{time.strftime('%H:%M:%S')} Finished: this computer has no more runs waiting.", flush=True)


if __name__ == "__main__":
    main()
