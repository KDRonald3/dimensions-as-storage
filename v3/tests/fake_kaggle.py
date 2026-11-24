"""A pretend Kaggle command-line tool, for the tests of v3/orchestrate.py.

It understands the four commands the orchestrator uses (quota, kernels push, kernels status,
kernels output) and keeps its state in the JSON file named by the environment variable
FAKE_KAGGLE_STATE:

    quota_remaining   weekly hours left (a pushed session takes its asked time off at once)
    max_running       sessions that may run at once; another push is refused
    fail              names of sessions that break down (status "error", no output)
    running_polls     how many status calls a session stays "running"

A session that is not set to fail really runs its script at push time, on this computer, with
/kaggle/working and /kaggle/input replaced by folders next to the state file. The outputs of the
sessions named in kernel_sources are put into its input folder, as Kaggle does.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

STATE = Path(os.environ["FAKE_KAGGLE_STATE"])


def load():
    return json.loads(STATE.read_text())


def save(state):
    STATE.write_text(json.dumps(state, indent=1))


def kernel_folder(name):
    return STATE.parent / "kernels" / name


def push(folder, seconds):
    state = load()
    meta = json.loads((Path(folder) / "kernel-metadata.json").read_text())
    name = meta["id"].split("/")[1]
    running = [k for k, v in state["kernels"].items() if v["running"]]
    if len(running) >= state["max_running"]:
        state["refusals"] = state.get("refusals", 0) + 1
        save(state)
        print("409 Client Error: Conflict. Maximum batch GPU session count of "
              f"{state['max_running']} reached.")
        return 1
    state["quota_remaining"] -= seconds / 3600
    state["pushes"].append({"name": name, "sources": meta["kernel_sources"],
                            "seconds": seconds})
    state["kernels"][name] = {"running": True, "polls": 0,
                              "fails": name in state.get("fail", [])}
    save(state)
    here = kernel_folder(name)
    if here.exists():
        shutil.rmtree(here)
    working, given = here / "working", here / "input"
    working.mkdir(parents=True)
    given.mkdir(parents=True)
    for source in meta["kernel_sources"]:
        earlier = kernel_folder(source.split("/")[1]) / "working"
        if earlier.exists():
            shutil.copytree(earlier, given / source.split("/")[1])
    if name not in state.get("fail", []):
        environment = dict(os.environ, AIR_KAGGLE_WORKING=str(working),
                           AIR_KAGGLE_INPUT=str(given), AIR_ALLOW_NO_CARD="1",
                           AIR_CODE_DIR=str(here / "code"))
        with open(here / "log.txt", "w") as log:
            subprocess.run([sys.executable, str(Path(folder) / meta["code_file"])],
                           cwd=str(here), env=environment, stdout=log,
                           stderr=subprocess.STDOUT, timeout=600)
    print(f"Kernel version 1 successfully pushed.  Please check progress at "
          f"https://www.kaggle.com/code/{meta['id']}")
    return 0


def status(ref):
    state = load()
    name = ref.split("/")[1]
    if name not in state["kernels"]:
        print("404 Client Error: Not Found")
        return 1
    kernel = state["kernels"][name]
    kernel["polls"] += 1
    if kernel["running"] and kernel["polls"] > state.get("running_polls", 1):
        kernel["running"] = False
    save(state)
    if kernel["running"]:
        word = "RUNNING"
    else:
        word = "ERROR" if kernel["fails"] else "COMPLETE"
    print(f'{ref} has status "KernelWorkerStatus.{word}"')
    return 0


def output(ref, folder):
    name = ref.split("/")[1]
    working = kernel_folder(name) / "working"
    Path(folder).mkdir(parents=True, exist_ok=True)
    if working.exists():
        shutil.copytree(working, folder, dirs_exist_ok=True)
    log = kernel_folder(name) / "log.txt"
    if log.exists():
        shutil.copy2(log, Path(folder) / f"{name}.log")
    print(f"Output file downloaded to {folder}")
    return 0


def main(argv):
    if argv[:1] == ["quota"]:
        left = load()["quota_remaining"]
        print("resource  used   remaining  total   refreshAt")
        print("--------  -----  ---------  ------  ---------")
        print(f"GPU       {30 - left:.2f}h  {left:.2f}h     30.00h  2026-11-28T00:00:00+00:00")
        return 0
    if argv[:2] == ["kernels", "push"]:
        return push(argv[argv.index("-p") + 1], float(argv[argv.index("-t") + 1]))
    if argv[:2] == ["kernels", "status"]:
        return status(argv[2])
    if argv[:2] == ["kernels", "output"]:
        return output(argv[2], argv[argv.index("-p") + 1])
    print(f"fake kaggle: unknown command {argv}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
