"""Launcher: runs the second experiment on Modal (modal.com).

Commands (run from the project folder, see v2/MODAL.md):
    modal run --detach v2/modal_app.py::main --stage smoke
    modal run --detach v2/modal_app.py::main --stage pilot
    modal run v2/modal_app.py::fetch --stage pilot

Each training run gets its own container (2 cores, 2 GiB). Results are kept on
a Modal Volume called "air-v2-runs", in /runs/<stage>/<name>/.
"""

import json
import sys
import threading
import time
from pathlib import Path

import modal

# Make "import v2.grid" work on this computer, whatever folder we start from.
PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# ---- Settings --------------------------------------------------------------

CORES = 2
MEMORY_GIB = 2
MAX_CONTAINERS = 100
COMMIT_EVERY_SECONDS = 180  # save the volume this often while training

# Modal prices (dollars per second).
PRICE_PER_CORE_SECOND = 0.0000131
PRICE_PER_GIB_SECOND = 0.00000222

# ---- Image and volume ------------------------------------------------------

# Debian with Python 3.12, CPU-only torch, and the same numpy as the project.
# The local "v2" folder is added to the image when a container starts.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.15.0",
        "numpy==2.5.5",
        index_url="https://download.pytorch.org/whl/cpu",
        extra_index_url="https://pypi.org/simple",
    )
    .add_local_python_source("v2")
)

volume = modal.Volume.from_name("air-v2-runs", create_if_missing=True)

app = modal.App("air-v2")


# ---- The remote function ---------------------------------------------------


def train_one(config: dict, stage: str) -> dict:
    """Train one model and save the result on the volume. Used by the two remote functions
    below, which differ only in the size of the machine."""
    from v2.run import run_one

    # If this is a retry, the earlier container may have left a checkpoint.
    # Reload so that we can see it.
    volume.reload()

    # Save the volume every few minutes while training, so a checkpoint
    # survives if this container is stopped.
    stop = threading.Event()

    def commit_often():
        while not stop.wait(COMMIT_EVERY_SECONDS):
            try:
                volume.commit()
            except Exception as error:  # never let saving stop the training
                print("commit failed (will try again):", error, flush=True)

    saver = threading.Thread(target=commit_often, daemon=True)
    saver.start()
    try:
        summary = run_one(config, "/runs/" + stage)
    finally:
        stop.set()
        saver.join()
        volume.commit()  # final save, also after an error (keeps the checkpoint)

    return {
        "name": summary["name"],
        "facts": summary["final"]["facts"],
        "minutes": summary["minutes"],
    }


@app.function(
    image=image,
    volumes={"/runs": volume},
    cpu=CORES,
    memory=MEMORY_GIB * 1024,  # in MiB
    timeout=4 * 60 * 60,
    retries=2,
    max_containers=MAX_CONTAINERS,
)
def train(config: dict, stage: str) -> dict:
    """A run on 2 cores: for the runs that use 2 threads (the large bodies)."""
    return train_one(config, stage)


@app.function(
    image=image,
    volumes={"/runs": volume},
    cpu=1,
    memory=1024,  # in MiB
    timeout=4 * 60 * 60,
    retries=2,
    max_containers=MAX_CONTAINERS,
)
def train_small(config: dict, stage: str) -> dict:
    """A run on 1 core: for the runs that use 1 thread (the small bodies). A small body trains
    almost as fast on 1 thread as on 2, so this halves what those runs cost."""
    return train_one(config, stage)


# ---- Helpers that run on this computer -------------------------------------


def run_name(config: dict) -> str:
    """The folder name of a run (same rule as v2/run.py)."""
    if config.get("name"):
        return config["name"]
    return "c{creatures}_b{body}_{usage}_k{kinds}_s{seed}".format(**config)


def finished_runs(stage: str) -> set:
    """Names of the runs of this stage that already have a summary.json on the volume."""
    try:
        entries = volume.listdir("/" + stage, recursive=True)
    except Exception:  # the stage folder does not exist yet
        return set()
    done = set()
    for entry in entries:
        parts = entry.path.strip("/").split("/")
        if parts[-1] == "summary.json":
            done.add(parts[-2])
    return done


def estimate_cost(minutes_list) -> float:
    """Dollars for these runs if every one used 2 cores and 2 GiB. Runs with 1 thread use a
    1-core machine and cost about half of this, so the figure is an upper estimate."""
    seconds = sum(minutes_list) * 60
    per_second = CORES * PRICE_PER_CORE_SECOND + MEMORY_GIB * PRICE_PER_GIB_SECOND
    return seconds * per_second


def print_cost(minutes_list):
    total = sum(minutes_list)
    print(f"Cost estimate: {len(minutes_list)} runs, {total:.1f} minutes of training "
          f"in total, about ${estimate_cost(minutes_list):.2f} "
          "at most (counts training time only, not start-up).")


# ---- Entry points ----------------------------------------------------------


@app.local_entrypoint()
def main(stage: str, limit: int = 0, exclude: str = ""):
    """Run every run of a stage that is not finished yet.

    --stage   smoke, pilot, main8, bits16, bits32 or long
    --limit   if above 0, only the first few runs (for a trial)
    """
    from v2.grid import grid

    configs = grid(stage)
    if exclude:
        # exclude="seed=2,body=8+128" leaves out the runs that match EVERY condition. Those
        # runs are done on the owner's own computer by v2/local_queue.py, with the same rule.
        conditions = [part.partition("=")[::2] for part in exclude.split(",")]

        def left_out(config):
            return all(str(config[key]) in values.split("+") for key, values in conditions)

        configs = [config for config in configs if not left_out(config)]
    if limit > 0:
        configs = configs[:limit]

    done = finished_runs(stage)
    todo = [c for c in configs if run_name(c) not in done]
    print(f"Stage {stage}: {len(configs)} runs, {len(configs) - len(todo)} already "
          f"on the volume, {len(todo)} to run.", flush=True)

    # spawn() sends the work to Modal and returns at once. The runs keep going
    # even if this terminal closes (when started with --detach).
    calls = {}
    for config in todo:
        machine = train_small if config.get("threads", 2) == 1 else train
        calls[run_name(config)] = machine.spawn(config, stage)
    if calls:
        print(f"Sent {len(calls)} runs. At most {MAX_CONTAINERS} run at the same time.", flush=True)

    failures = []
    minutes_list = []
    waiting = dict(calls)
    while waiting:
        for name, call in list(waiting.items()):
            try:
                result = call.get(timeout=0)
            except modal.exception.OutputExpiredError:
                failures.append((name, "result expired; check the volume"))
                del waiting[name]
                continue
            except (TimeoutError, modal.exception.TimeoutError):
                continue  # not finished yet (Modal raises Python's own TimeoutError here)
            except Exception as error:  # the run failed after all retries
                why = f"{type(error).__name__}: {error}"[:200]
                failures.append((name, why))
                print(f"FAILED  {name}: {why}", flush=True)
                del waiting[name]
                continue
            minutes_list.append(result["minutes"])
            print(f"done  {result['name']}  facts {result['facts']:.3f}  "
                  f"{result['minutes']:.1f} min", flush=True)
            del waiting[name]
        if waiting:
            time.sleep(10)

    print(f"Finished: {len(minutes_list)} runs ok, {len(failures)} failed.")
    for name, why in failures:
        print(f"  failed: {name}: {why}")
    if minutes_list:
        print_cost(minutes_list)
    print(f"To download: modal run v2/modal_app.py::fetch --stage {stage}")


@app.local_entrypoint()
def fetch(stage: str):
    """Download summary.json and model.pt of every run of a stage into runs_v2/<stage>/."""
    out_root = PROJECT / "runs_v2" / stage
    try:
        entries = volume.listdir("/" + stage, recursive=True)
    except Exception:
        print(f"Nothing on the volume for stage {stage}.")
        return

    wanted = [e for e in entries
              if e.path.split("/")[-1] in ("summary.json", "model.pt")]
    minutes_list = []
    for entry in wanted:
        relative = entry.path.strip("/")  # <stage>/<name>/<file>
        target = PROJECT / "runs_v2" / relative
        if not (target.exists() and target.stat().st_size == entry.size):
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as handle:
                for chunk in volume.read_file("/" + relative):
                    handle.write(chunk)
            print("downloaded", relative)
        if target.name == "summary.json":
            minutes_list.append(json.loads(target.read_text())["minutes"])

    print(f"{len(minutes_list)} runs of stage {stage} are in {out_root}")
    if minutes_list:
        print_cost(minutes_list)
