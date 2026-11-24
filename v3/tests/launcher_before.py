"""A copy of v3/colab_fleet.py as it ran the counted stage on 23 November from 08:56 (the
launcher the upgraded one takes over from), kept only for the take-over test
(test_v3_fleet_takeover.py). Not used by the programs.

Run a stage of the third experiment on SEVERAL Colab machines at once, and bring the results
home as they come.

    .venv/bin/python -m v3.colab_fleet counted --machines 6              # in a terminal
    systemd-run --user --unit air-fleet-counted --working-directory=$PWD \\
        .venv/bin/python -m v3.colab_fleet counted --machines 6         # as a user service
    tail -f logs/v3/fleet.log                                           # watch it
    .venv/bin/python -m v3.colab_fleet counted --stop                   # stop all machines

Stopping only this program (Ctrl-C, systemctl --user stop) leaves the machines training and
using units; start it again to go on, or use --stop (after stopping the service) to end them.

What it does, every minute or so
--------------------------------
1. Compute units: reads the balance (`colab usage`). If the balance would fall below the
   reserve (--reserve, default 100 units) before the next look, it stops everything cleanly:
   every machine saves its groups at a checkpoint, the checkpoints and results come home, and
   the machines are stopped. Started again later, it goes on from there.
2. Machines: while there are groups that no machine has and fewer machines than --machines,
   it asks for more (v3/colab_gpu.py: every card type at once, the fastest granted are kept).
3. Each machine (v3/machine_job.py runs there) is asked how far it is. Runs that have ended are
   downloaded into runs_v3/<stage>/, the "so far" scores of runs still training into
   runs_v3/<stage>/_so_far/, and the checkpoint of every group put back at the end of a round
   into runs_v3/_fleet/<stage>/checkpoints/ (so a group can go on elsewhere if a machine is lost).
4. Groups are handed out so that no group is on two machines. A machine that has nothing
   waiting takes over a waiting group of the machine with the most waiting (the group is taken
   off that machine first, by a rename that can succeed only once, then sent with its checkpoint).
5. A machine that does not answer for --lost-minutes, or that Colab no longer lists, is given
   up: its groups go to other machines from their last checkpoint here (training since that
   checkpoint is lost and is done again).
6. When every run of the stage is here, every machine is stopped. A machine with nothing left
   to do while others still train is stopped at once.

It keeps what it knows in logs/v3/fleet_<stage>.json, so it can be stopped (Ctrl-C, or
`systemctl --user stop air-fleet-counted`) and started again without losing or repeating work.
It writes a plain log to logs/v3/fleet.log.
"""

import argparse
import base64
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from v3 import colab_gpu

ROOT = Path(__file__).resolve().parent.parent.parent  # (one folder deeper than the original)
LOG = ROOT / "logs" / "v3" / "fleet.log"
BASE = "/content/air_run"  # the job's folder on every machine
OUT = f"{BASE}/out"
RETRIES = 4  # a request to a machine is tried this often ("Connection was lost" happens)
EXEC_TIMEOUT = 120  # --timeout of `colab exec`: the longest silence (0 can hang for ever)


def say(text, log=None):
    line = f"{time.strftime('%F %T')}  {text}"
    print(line, flush=True)
    log = Path(log or LOG)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as handle:
        handle.write(line + "\n")


# ------------------------------------------------------------------------------------------------
# Talking to Colab
# ------------------------------------------------------------------------------------------------


class Colab:
    """The Colab command-line tool. Every machine of the fleet has its own state file
    (colab_gpu.config_of). Requests are tried again when the connection was lost."""

    def __init__(self, tool=colab_gpu.COLAB, log=None):
        self.tool = Path(tool)
        self.log = log

    def run(self, *arguments, machine=None, text=None, timeout=300, tries=RETRIES):
        command = [str(self.tool)]
        if machine is not None:
            command += ["--config", str(colab_gpu.config_of(machine))]
        command += list(arguments)
        printed = ""
        for attempt in range(tries):
            try:
                done = subprocess.run(command, input=text, capture_output=True, text=True,
                                      timeout=timeout)
                printed = done.stdout + done.stderr
                if done.returncode == 0:
                    return True, printed
            except subprocess.TimeoutExpired:
                printed = f"no answer within {timeout} s"
            if attempt + 1 < tries:
                time.sleep(3 * (attempt + 1))
        return False, printed

    def execute(self, machine, code, timeout=300):
        """Run Python on the machine; returns (worked, what it printed)."""
        return self.run("exec", "-s", machine, "--timeout", str(EXEC_TIMEOUT), machine=machine,
                        text=code, timeout=timeout)

    def upload(self, machine, local, remote):
        return self.run("upload", "-s", machine, str(local), remote, machine=machine)[0]

    def download(self, machine, remote, local):
        worked, _ = self.run("download", "-s", machine, remote, str(local), machine=machine)
        return worked and Path(local).exists()

    def stop(self, machine):
        return self.run("stop", "-s", machine, machine=machine, tries=2)[0]

    def exists(self, machine):
        """False when Colab says the machine is not there (any more), True when it is, None
        when it could not be asked."""
        worked, printed = self.run("status", "-s", machine, machine=machine, tries=2,
                                   timeout=120)
        if "not found" in printed.lower():
            return False
        return True if worked else None

    def balance(self):
        """(compute units left, units used an hour), or None."""
        worked, printed = self.run("usage", tries=2, timeout=120)
        left = re.search(r"balance:\s*([\d.,]+)", printed)
        rate = re.search(r"rate:\s*([\d.,]+)", printed)
        if not worked or not left:
            return None
        number = lambda m: float(m.group(1).replace(",", "")) if m else 0.0
        return number(left), number(rate)


# ------------------------------------------------------------------------------------------------
# Code sent to the machines (each is run there by `colab exec`)
# ------------------------------------------------------------------------------------------------


def packed_code():
    """v2/ and v3/ (programs only, no tests, reports or results) as base64 text."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for folder in ("v2", "v3"):
            for path in sorted((ROOT / folder).rglob("*.py")):
                parts = path.relative_to(ROOT).parts
                if not ({"tests", "report", "writeup", "__pycache__"} & set(parts)):
                    tar.add(path, arcname="/".join(parts))
    return base64.b64encode(buffer.getvalue()).decode()


# Unpack the programs and start the job, unless it is running already.
START = r'''
import base64, io, os, subprocess, sys, tarfile
os.makedirs(BASE + "/code", exist_ok=True)
os.makedirs(BASE + "/out", exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PACKED))) as packed:
    packed.extractall(BASE + "/code")
alive = subprocess.run("pgrep -f '[v]3.machine_job --out " + BASE + "/out' | wc -l", shell=True,
                       capture_output=True, text=True).stdout.strip()
if alive != "0":
    print("ALREADY RUNNING")
else:
    for name in ("stop", "close"):
        if os.path.exists(BASE + "/out/" + name):
            os.remove(BASE + "/out/" + name)
    environment = dict(os.environ, PYTHONPATH=BASE + "/code")
    subprocess.Popen([sys.executable, "-m", "v3.machine_job", "--out", BASE + "/out",
                      "--workers", str(WORKERS)], cwd=BASE + "/code", env=environment,
                     stdout=open(BASE + "/job.log", "a"), stderr=subprocess.STDOUT,
                     start_new_session=True)
    print("STARTED")
'''

# Give the machine groups: write their task files (a checkpoint was uploaded before, if any).
GIVE = r'''
import json, os
for task in TASKS:
    folder = BASE + "/out/tasks"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(BASE + "/out/progress", exist_ok=True)
    progress = BASE + "/out/progress/%d.json" % task["group"]
    if task.get("step"):
        open(progress, "w").write(json.dumps({"step": task["step"], "done": False}))
    elif os.path.exists(progress):
        os.remove(progress)
    for name, text in task.get("ended", {}).items():
        os.makedirs(BASE + "/out/" + task["stage"] + "/" + name, exist_ok=True)
        open(BASE + "/out/" + task["stage"] + "/" + name + "/summary.json", "w").write(text)
    task = {k: v for k, v in task.items() if k not in ("step", "ended")}
    open(folder + "/%d.tmp" % task["group"], "w").write(json.dumps(task))
    os.replace(folder + "/%d.tmp" % task["group"], folder + "/%d.json" % task["group"])
print("GIVEN", len(TASKS))
'''

# Take a waiting group back (only if no worker has taken it: a rename succeeds only once).
TAKE_BACK = r'''
import os
try:
    os.rename(BASE + "/out/tasks/%d.json" % NUMBER, BASE + "/out/taken_back_%d.json" % NUMBER)
    print("TAKEN BACK")
except FileNotFoundError:
    print("NOT WAITING")
'''

# Say how far the machine is, and pack what is new since the last download.
POLL = r'''
import glob, json, os, subprocess, tarfile, time
out = BASE + "/out"
marker = BASE + "/synced.txt"
last = float(open(marker).read()) if os.path.exists(marker) else 0.0
now = time.time() - 5
new = []
for summary in glob.glob(out + "/" + STAGE + "/*/summary.json"):
    folder = os.path.dirname(summary)
    if last <= os.path.getmtime(summary) < now:
        new += glob.glob(folder + "/*")
new += [p for p in glob.glob(out + "/_so_far/*.json") + glob.glob(out + "/handoff/*.pt")
        if last <= os.path.getmtime(p) < now]
with tarfile.open(BASE + "/new.tar", "w") as tar:
    for p in new:
        tar.add(p, arcname=os.path.relpath(p, out))
open(marker + ".next", "w").write(repr(now))
def read(path):
    try:
        return json.load(open(path))
    except Exception:
        return None
progress = {}
for path in glob.glob(out + "/progress/*.json"):
    found = read(path)
    if found is not None:
        progress[os.path.basename(path)[:-5]] = {k: found.get(k) for k in ("step", "done", "error")}
answer = {
    "new": len(new),
    "waiting": sorted(int(os.path.basename(p)[:-5]) for p in glob.glob(out + "/tasks/*.json")),
    "taken": sorted(int(os.path.basename(p).split(".")[0]) for p in glob.glob(out + "/taken/*.json")),
    "progress": progress,
    "alive": int(subprocess.run("pgrep -f '[v]3.machine_job --out " + BASE + "/out' | wc -l", shell=True,
                                capture_output=True, text=True).stdout.strip() or 0),
    "card": subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
                           shell=True, capture_output=True, text=True).stdout.strip(),
}
print("ANSWER " + json.dumps(answer))
'''

COMMIT = r'''
import os
os.replace(BASE + "/synced.txt.next", BASE + "/synced.txt")
print("COMMITTED")
'''

SIGNAL = r'''
open(BASE + "/out/" + WHAT, "w").write("")
print("SIGNALLED")
'''


def with_values(code, **values):
    """The code with BASE and the given names set at its top."""
    head = f"BASE = {BASE!r}\n" + "".join(f"{k} = {v!r}\n" for k, v in values.items())
    return head + code


# ------------------------------------------------------------------------------------------------
# The fleet
# ------------------------------------------------------------------------------------------------


class Fleet:
    def __init__(self, stage, groups, colab, runs_root=None, state_path=None, log=None,
                 machines=3, reserve=100.0, workers=2, cards=("H100", "G4", "A100"),
                 lost_minutes=15.0, poll_seconds=60.0, prefix="air", grab=None):
        self.stage, self.groups, self.colab = stage, groups, colab
        self.runs = Path(runs_root or ROOT / "runs_v3")
        self.state_path = Path(state_path or ROOT / "logs" / "v3" / f"fleet_{stage}.json")
        self.log = log
        self.want, self.reserve, self.workers = machines, reserve, workers
        self.cards, self.lost_minutes, self.poll_seconds = list(cards), lost_minutes, poll_seconds
        self.prefix = prefix
        # 60 seconds: on 23 November an A100 once took longer than 10 seconds to be granted.
        self.grab = grab or (lambda count: colab_gpu.grab(count, prefix, wait_seconds=60.0,
                                                          cards=self.cards,
                                                          colab=self.colab.tool))
        self.checkpoints = self.runs / "_fleet" / stage / "checkpoints"
        self.state = {"machines": {}, "groups": {}, "failures": {}}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())

    def say(self, text):
        say(text, self.log)

    def save(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=1))
        temporary.replace(self.state_path)

    # --- what is done ----------------------------------------------------------------------------

    def finished_names(self):
        return {p.parent.name for p in (self.runs / self.stage).glob("*/summary.json")}

    def group_done(self, index, finished=None):
        finished = self.finished_names() if finished is None else finished
        return all(c["name"] in finished for c in self.groups[index])

    def open_groups(self):
        """Groups not done here, in the stage's order."""
        finished = self.finished_names()
        return [i for i in range(len(self.groups)) if not self.group_done(i, finished)]

    def free_groups(self):
        """Open groups that no machine has (and that have not failed too often)."""
        held = {int(i) for i, g in self.state["groups"].items() if g.get("machine")}
        return [i for i in self.open_groups() if i not in held
                and self.state["failures"].get(str(i), 0) < 3]

    # --- machines ----------------------------------------------------------------------------

    def add_machines(self):
        """Ask for machines while there is work no machine has and fewer than wanted."""
        free = self.free_groups()
        missing = self.want - len(self.state["machines"])
        if not free or missing <= 0:
            return
        count = min(missing, len(free))
        answer = self.grab(count)
        for kept in answer.get("kept", []):
            name = kept["session"]
            self.state["machines"][name] = {"card": kept["card"], "last_answer": time.time(),
                                            "started": False}
            self.say(f"machine {name} ({kept['card']}) added")
        if not answer.get("kept"):
            self.say(f"no machine granted (asked for {count}): {answer.get('refused')}")
        self.save()

    def start_job(self, name):
        worked, printed = self.colab.execute(
            name, with_values(START, PACKED=packed_code(), WORKERS=self.workers), timeout=600)
        if worked and ("STARTED" in printed or "ALREADY RUNNING" in printed):
            self.state["machines"][name]["started"] = True
            self.say(f"machine {name}: job {'started' if 'STARTED' in printed else 'running'}")
            self.save()
            return True
        self.say(f"machine {name}: the job could not be started: {printed.strip()[-300:]}")
        return False

    def give(self, name, indices):
        """Send groups to a machine, each with its latest checkpoint here (if any)."""
        tasks = []
        for index in indices:
            group = self.state["groups"].setdefault(str(index), {})
            step = group.get("step") or 0
            checkpoint = self.checkpoints / f"{index}.pt"
            if step and checkpoint.exists():
                from v2.batch import group_name
                from v2.run import complete_config
                folder = group_name([complete_config(c) for c in self.groups[index]])
                remote = f"{OUT}/_groups/{folder}/checkpoint.pt"
                self.colab.execute(name, f"import os\nos.makedirs({str(Path(remote).parent)!r}, "
                                         f"exist_ok=True)\nprint('OK')\n")
                if not self.colab.upload(name, checkpoint, remote):
                    self.say(f"machine {name}: upload of the checkpoint of group {index} failed")
                    continue
            else:
                step = 0  # no checkpoint here: the group starts again from the beginning
            # The summaries of members that have ended, so they are not trained again.
            ended = {}
            for config in self.groups[index]:
                summary = self.runs / self.stage / config["name"] / "summary.json"
                if summary.exists():
                    ended[config["name"]] = summary.read_text()
            tasks.append({"group": index, "stage": self.stage, "configs": self.groups[index],
                          "step": step, "ended": ended})
        if not tasks:
            return
        worked, printed = self.colab.execute(name, with_values(GIVE, TASKS=tasks))
        if worked and "GIVEN" in printed:
            for task in tasks:
                self.state["groups"][str(task["group"])]["machine"] = name
            self.say(f"machine {name}: given groups {[t['group'] for t in tasks]}")
            self.save()

    def poll(self, name):
        """Ask the machine how far it is and download what is new. Returns its answer or None."""
        worked, printed = self.colab.execute(name, with_values(POLL, STAGE=self.stage))
        lines = [line for line in printed.splitlines() if line.startswith("ANSWER ")]
        if not worked or not lines:
            return None
        answer = json.loads(lines[-1][len("ANSWER "):])
        if answer["new"]:
            with tempfile.TemporaryDirectory() as folder:
                local = Path(folder) / "new.tar"
                if not self.colab.download(name, f"{BASE}/new.tar", local):
                    self.say(f"machine {name}: download failed; it comes again next time")
                    return answer
                self.take_in(local, answer)
        self.colab.execute(name, with_values(COMMIT))
        return answer

    def take_in(self, tar_path, answer):
        """Put the downloaded files in their places: results (model.pt before summary.json, so a
        summary.json always has its model), so-far files and checkpoints."""
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with tarfile.open(tar_path) as tar:
                tar.extractall(folder, filter="data")
            stage = folder / self.stage
            for run in sorted(stage.glob("*")) if stage.exists() else []:
                target = self.runs / self.stage / run.name
                target.mkdir(parents=True, exist_ok=True)
                for path in sorted(run.iterdir(), key=lambda p: p.name == "summary.json"):
                    shutil.copy2(path, target / path.name)
            finished = self.finished_names()
            so_far = self.runs / self.stage / "_so_far"
            for path in (folder / "_so_far").glob("*.json"):
                if path.stem not in finished:
                    so_far.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, so_far / path.name)
            for name in finished:  # a run that has ended is no longer "so far"
                (so_far / f"{name}.json").unlink(missing_ok=True)
            for path in (folder / "handoff").glob("*.pt"):
                self.checkpoints.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, self.checkpoints / path.name)
                # the step of this checkpoint is the one the machine reported for the group
                reported = answer["progress"].get(path.stem, {})
                if reported.get("step") is not None:
                    self.state["groups"].setdefault(path.stem, {})["saved_step"] = reported["step"]

    def note_progress(self, name, answer):
        """Record what the machine said about its groups."""
        for key, found in answer["progress"].items():
            group = self.state["groups"].setdefault(key, {})
            if group.get("machine") != name:
                continue
            # the step from which another machine could go on: that of the checkpoint here
            group["step"] = group.get("saved_step", 0)
            group["reached"] = found.get("step")
            if found.get("error") and not found.get("done"):
                self.state["failures"][key] = self.state["failures"].get(key, 0) + 1
                group["machine"] = None
                self.say(f"group {key} failed on {name} (failure "
                         f"{self.state['failures'][key]}): "
                         f"{(found['error'] or '').strip().splitlines()[-1:]}")
        self.save()

    def give_up(self, name, why):
        """The machine is lost: its groups go back to the pool, and it is stopped."""
        self.say(f"machine {name} given up ({why}); its groups go to other machines")
        for group in self.state["groups"].values():
            if group.get("machine") == name:
                group["machine"] = None
                group["step"] = group.get("saved_step", 0)
        self.colab.stop(name)
        del self.state["machines"][name]
        self.save()

    def reconcile(self, name, answer):
        """After a restart in an unlucky moment, a machine may have a group this program does
        not know it gave: it is taken as that machine's, unless another machine has it (then
        it is taken back from this one, so that no group trains twice)."""
        for index in answer["waiting"] + answer["taken"]:
            group = self.state["groups"].setdefault(str(index), {})
            if group.get("machine") in (None, name):
                group["machine"] = name
            elif index in answer["waiting"]:
                self.colab.execute(name, with_values(TAKE_BACK, NUMBER=index))
                answer["waiting"].remove(index)
            else:
                self.say(f"group {index} is training on {name} and was given to "
                         f"{group['machine']} too; it is taken back from {group['machine']}")
                self.colab.execute(group["machine"], with_values(TAKE_BACK, NUMBER=index))
                group["machine"] = name

    def held_by(self, name):
        return [int(i) for i, g in self.state["groups"].items() if g.get("machine") == name]

    def balance_work(self, answers):
        """Free groups to machines with room; then a machine with nothing waiting takes over a
        waiting group of the machine with the most waiting."""
        # Every free group is given out at once, to the machine holding fewest, so that each
        # machine can train its groups breadth first (v3/machine_job.py).
        share = {name: [] for name in answers}
        held = {name: len(self.held_by(name)) for name in answers}
        for index in self.free_groups() if answers else []:
            name = min(answers, key=lambda n: (held[n] + len(share[n]), n))
            share[name].append(index)
        for name, indices in share.items():
            if indices:
                self.give(name, indices)
        idle = [n for n, a in answers.items() if not a["waiting"]]
        for name in idle:
            donors = sorted(((len(a["waiting"]), n) for n, a in answers.items()
                             if n != name and len(a["waiting"]) >= 2), reverse=True)
            if not donors:
                return
            donor = donors[0][1]
            number = answers[donor]["waiting"][-1]
            worked, printed = self.colab.execute(donor, with_values(TAKE_BACK, NUMBER=number))
            if not (worked and "TAKEN BACK" in printed):
                continue
            answers[donor]["waiting"].remove(number)
            self.say(f"group {number} moves from {donor} to {name}")
            # the newest checkpoint of the group (made when it was last put back)
            local = self.checkpoints / f"{number}.pt"
            group = self.state["groups"][str(number)]
            reached = group.get("reached") or 0
            if reached and self.colab.download(donor, f"{OUT}/handoff/{number}.pt",
                                               local.with_suffix(".part")):
                local.with_suffix(".part").replace(local)
                group["saved_step"] = reached
            group["step"] = group.get("saved_step", 0)
            group["machine"] = None
            self.give(name, [number])

    # --- the main loop -------------------------------------------------------------------------

    def units_ok(self):
        found = self.colab.balance()
        if found is None:
            return True  # cannot tell; asked again next time
        left, rate = found
        # the units that the machines (and those we may still add) use before the next look
        extra = 7.0 * max(0, self.want - len(self.state["machines"]))
        need = (rate + extra) * (3 * self.poll_seconds / 3600) + 1.0
        if left - need < self.reserve:
            self.say(f"compute units: {left:.1f} left, using {rate:.1f} an hour; the reserve of "
                     f"{self.reserve:.0f} would be reached: stopping everything cleanly")
            return False
        return True

    def stop_all(self, wait_minutes=8.0):
        """Ask every job to stop, bring home what they saved, stop the machines."""
        for name in list(self.state["machines"]):
            self.colab.execute(name, with_values(SIGNAL, WHAT="stop"))
        until = time.time() + wait_minutes * 60
        while self.state["machines"] and time.time() < until:
            for name in list(self.state["machines"]):
                answer = self.poll(name)
                if answer is None or answer["alive"] == 0:
                    if answer is not None:
                        self.note_progress(name, answer)
                    for group in self.state["groups"].values():
                        if group.get("machine") == name:
                            group["machine"] = None
                    self.colab.stop(name)
                    self.say(f"machine {name} stopped")
                    del self.state["machines"][name]
                    self.save()
            time.sleep(10)
        for name in list(self.state["machines"]):  # did not stop in time
            self.colab.stop(name)
            del self.state["machines"][name]
            self.say(f"machine {name} stopped (its job had not finished saving)")
        for group in self.state["groups"].values():
            group["machine"] = None
            group["step"] = group.get("saved_step", 0)
        self.save()

    def step(self):
        """One look at everything. Returns False when the stage is done (or must stop)."""
        if not self.units_ok():
            self.stop_all()
            return False
        if not self.open_groups():
            self.say("every run of the stage is here")
            self.stop_all(wait_minutes=1)
            return False
        self.add_machines()
        answers = {}
        for name in list(self.state["machines"]):
            machine = self.state["machines"][name]
            if self.colab.exists(name) is False:
                self.give_up(name, "Colab says it is not there any more")
                continue
            if not machine.get("started") and not self.start_job(name):
                continue
            answer = self.poll(name)
            if answer is None:
                silent = (time.time() - machine["last_answer"]) / 60
                if silent > self.lost_minutes:
                    self.give_up(name, f"no answer for {silent:.0f} minutes")
                continue
            machine["last_answer"] = time.time()
            self.reconcile(name, answer)
            self.note_progress(name, answer)
            held = self.held_by(name)
            if answer["alive"] == 0 and held:
                self.say(f"machine {name}: its job is not running; starting it again")
                machine["started"] = False
                continue
            answers[name] = answer
            done_here = sum(1 for i in held if self.group_done(i))
            self.say(f"machine {name}: {len(answer['waiting'])} waiting, {len(answer['taken'])} "
                     f"training, card {answer['card']}; {done_here} of its {len(held)} groups done")
        for name, answer in answers.items():  # groups whose runs are all here are let go
            for index in self.held_by(name):
                if self.group_done(index) and index not in answer["waiting"] + answer["taken"]:
                    self.state["groups"][str(index)]["machine"] = None
        self.balance_work(answers)
        for name, answer in answers.items():  # nothing left for this machine: stop it
            if not self.held_by(name) and not self.free_groups():
                self.colab.execute(name, with_values(SIGNAL, WHAT="close"))
                self.colab.stop(name)
                del self.state["machines"][name]
                self.say(f"machine {name}: nothing left to do; stopped")
        self.save()
        finished = self.finished_names()
        total = sum(len(g) for g in self.groups)
        here = sum(1 for g in self.groups for c in g if c["name"] in finished)
        self.say(f"stage {self.stage}: {here} of {total} runs here; "
                 f"{len(self.open_groups())} groups open; {len(self.state['machines'])} machines")
        return True

    def run(self):
        self.say(f"fleet for stage {self.stage}: {len(self.groups)} groups, up to {self.want} "
                 f"machines, reserve {self.reserve:.0f} units")
        while self.step():
            time.sleep(self.poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a stage on several Colab machines.")
    parser.add_argument("stage")
    parser.add_argument("--machines", type=int, default=3)
    parser.add_argument("--reserve", type=float, default=100.0,
                        help="compute units that are never used (default 100)")
    parser.add_argument("--workers", type=int, default=2, help="groups at once on a machine")
    parser.add_argument("--cards", default="H100,G4,A100", help="card types, fastest first")
    parser.add_argument("--lost-minutes", type=float, default=15.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--prefix", default="air")
    parser.add_argument("--stop", action="store_true",
                        help="stop every machine of the fleet cleanly (they save first) and end")
    args = parser.parse_args(argv)
    from v3.orchestrate import stage_groups
    fleet = Fleet(args.stage, stage_groups(args.stage), Colab(), machines=args.machines,
                  reserve=args.reserve, workers=args.workers, cards=args.cards.split(","),
                  lost_minutes=args.lost_minutes, poll_seconds=args.poll_seconds,
                  prefix=args.prefix)
    if args.stop:
        fleet.say("asked to stop every machine")
        fleet.stop_all()
        return
    try:
        fleet.run()
    except KeyboardInterrupt:
        fleet.say("stopped by hand; the machines go on. Start this program again to go on "
                  "watching them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
