"""Run a stage of the third experiment on SEVERAL Colab machines at once, and bring the results
home as they come.

    .venv/bin/python -m v3.colab_fleet counted                          # in a terminal
    systemd-run --user --unit air-fleet-counted --working-directory=$PWD \\
        .venv/bin/python -u -m v3.colab_fleet counted                   # as a user service
    tail -f logs/v3/fleet.log                                           # watch it
    .venv/bin/python -m v3.colab_fleet counted --stop                   # stop all machines

Stopping only this program (Ctrl-C, systemctl --user stop) leaves the machines training and
using units; start it again to go on (it takes over the machines in its state file, also one
written by the launcher before 23 November 10:00), or use --stop (after stopping the service) to
end them.

Does work move to the faster cards by itself? (the owner's question of 23 November 2026)
-----------------------------------------------------------------------------------------
Yes, in three ways, all at every look (about every 40 seconds):
  - A group that is free, or waiting in a machine's queue between two rounds, goes to the machine
    that makes the stage end soonest (the planner, step 3 below). A shared fast card with free
    workers also takes groups that only wait in another machine's queue (fill_free_workers).
  - "Move up": a group in the MIDDLE of a round on a slower machine moves to a faster machine
    that has a free worker, when it should end there clearly sooner: at least 2 minutes sooner
    and in less than 0.8 of the time, counting 4 minutes for the move (saving, bringing the
    checkpoint home, the upload, compiling). The slower machine is told to park only that group:
    it saves it, the checkpoint comes home with the next look, and the group goes on on the
    faster machine from exactly there. The largest gains go first, one free worker each; a group
    that moved stays put for 20 minutes. The other groups on the slower machine train on.
    Only measured speeds count: a card that has not finished a round yet (its speed is a guess)
    neither gives nor takes groups this way, and one odd reading cannot cause a move.
  - A machine whose job is too old to park one group (started before this rule existed) is
    stopped as a whole (it saves first, it is given back, its groups go on elsewhere) only when
    EVERY group it trains would move up; otherwise it is left alone. This holds for A100s too.
So when an H100 has idle workers, the groups that gain most go to it within a look or two, as
many as it has free workers. A machine left with nothing is then handled by the rules below
(a fast card waits a few looks for work, a slow one is given back).

What it does, about every 40 seconds
------------------------------------
1. Compute units: reads the balance (`colab usage`). If the balance would fall below the
   reserve (--reserve, default 100 units) before the next look, it stops everything cleanly:
   every machine saves its groups at a checkpoint, the checkpoints and results come home, and
   the machines are stopped. Started again later, it goes on from there.
2. Each machine (v3/machine_job.py runs there) is asked how far it is. Runs that have ended are
   downloaded into runs_v3/<stage>/, the "so far" scores of runs still training into
   runs_v3/<stage>/_so_far/, and the checkpoint of every group put back at the end of a round
   into runs_v3/_fleet/<stage>/checkpoints/ (so a group can go on elsewhere if a machine is lost).
   Each round that ends tells how fast that machine trains that group shape (v3/fleet_plan.py
   keeps these speeds; until a card type has been measured, a guess is used).
3. Planning (v3/fleet_plan.py): it estimates when the stage ends under the present hand-out
   and makes the moves that bring that end earlier, counting what a move costs (upload,
   compiling the shape on the new machine: 60 to 165 s). Only groups between rounds whose
   checkpoint is already here are moved (taken off their machine by a rename that can succeed
   only once, then sent with the checkpoint), so no group is ever on two machines. A machine in
   the middle of its rounds is stopped (it saves first; its groups go on elsewhere) only when
   the stage then ends clearly earlier (by 10 minutes and 15 %). Cost is only a tie-breaker:
   a machine is given back when the stage does not end later without it, and a machine with
   nothing to do is given back at once.
4. Machines: it asks Colab once for a machine of every card type that would make the stage end
   clearly earlier (fastest first, up to --machines; when full, only cards faster than the
   slowest machine). Colab grants a free card in about 3 s and says "no card free" after 30 s
   (v3/colab_gpu.py asks without the tool's hidden retries). A granted machine is put to work at
   once. When Colab refuses for another reason (its limit on machines at once), the log says so.
5. A machine that does not answer for --lost-minutes, or that Colab no longer lists, is given
   up: its groups go to other machines from their last checkpoint here (training since that
   checkpoint is lost and is done again).
6. When every run of the stage is here, every machine is stopped.

It keeps what it knows in logs/v3/fleet_<stage>.json, so it can be stopped (Ctrl-C, or
`systemctl --user stop air-fleet-counted`) and started again without losing or repeating work.
It writes a plain log to logs/v3/fleet.log.
"""

import argparse
import base64
import concurrent.futures
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from v3 import colab_gpu, fleet_plan

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "v3" / "fleet.log"
BASE = "/content/air_run"  # the job's folder on every machine
OUT = f"{BASE}/out"
RETRIES = 4  # a request to a machine is tried this often ("Connection was lost" happens)
EXEC_TIMEOUT = 120  # --timeout of `colab exec`: the longest silence (0 can hang for ever)
# Fast cards are rare (the owner's rule of 23 November 2026): once one is ours it is kept. It is
# never stopped to make room or to save units, and a problem on it is repaired, not answered by
# giving the machine back.
FAST_CARDS = ("H100", "A100", "G4")

# "Move up" (Fleet.move_up): a group in the middle of a round goes to a faster machine with a free
# worker. A move costs: the worker saves the group, the checkpoint comes home with the next look
# (up to a minute), it is uploaded to the new machine and the group's shape is compiled there (60
# to 165 s). About 4 minutes in all, counted on the side of the move.
MOVE_UP_SECONDS = 240.0
# ... and the group must end at least this much sooner after the move (the estimates are rough) ...
MOVE_UP_MARGIN = 120.0
# ... and in less than this share of the time it would need where it is.
MOVE_UP_SHARE = 0.8
# A group that was moved stays where it is this long, so it is not sent back and forth.
PIN_SECONDS = 1200.0
# A parked group's checkpoint normally comes home with the next look or the one after (looks
# are 40 s apart). If it has not after this long, the machine is asked to send it again.
RESEND_SECONDS = 120.0
# If it has still not come after this long, the move is given up: the group goes back into the
# slower machine's queue and trains on there (never left parked for ever).
PARK_WAIT_SECONDS = 600.0
# A move up is decided only from measured speeds: a card with no measurement yet has only a guess
# (fleet_plan.CARD_GUESS), and single readings for one shape can be far off (on 23 November the
# H100's readings ranged from 614 to 13,065 member-steps a second). So the decision takes, for the
# slower machine, the higher of its speed for the shape and its typical speed, and for the faster
# machine the lower of the two (see Fleet.move_up_rate).


# A machine running at Colab that no launcher knows is taken in only after it has been seen
# twice this many minutes apart (a request that was just granted writes its file a moment later).
STRAY_MINUTES = 3.0
# Sign-in secrets never go into our log (the tool's error texts can hold addresses with them).
SECRETS = re.compile(r"(token=|Bearer\s+|ya29\.)[^\s&'\"]+", re.I)


def say(text, log=None):
    text = SECRETS.sub(lambda m: m.group(1) + "<hidden>", str(text))
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
    if SHARED:
        # NVIDIA's MPS lets the workers' work on the card overlap (without it the card serves
        # one program at a time). It must run before the workers start.
        # (Started without asking first whether it runs: a second start only says that it is
        # running already. A test with pgrep matched its own command line and never started it.)
        subprocess.run("nvidia-cuda-mps-control -d", shell=True, capture_output=True)
        import time
        time.sleep(2)
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
    # Sent twice (a request tried again after a lost connection) while a worker has already
    # taken it: writing it again would let a second worker train the same group at once.
    taken = BASE + "/out/taken"
    if os.path.isdir(taken) and any(name.startswith("%d." % task["group"])
                                    for name in os.listdir(taken)):
        continue
    # A park signal or a parked copy left from an earlier stay of this group here would make a
    # worker set the new task aside at once: the task given now replaces them.
    for old in (BASE + "/out/park/%d" % task["group"], BASE + "/out/parked/%d.json" % task["group"]):
        if os.path.exists(old):
            os.remove(old)
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
        progress[os.path.basename(path)[:-5]] = dict(
            {k: found.get(k) for k in ("step", "done", "error", "member_steps", "seconds")},
            members=len(found["unfinished"]) if isinstance(found.get("unfinished"), list) else None)
def can_park():
    # Only a job started by a machine_job.py that knows the park signal writes this file, and
    # only while that job runs does it count (an older job started later would ignore signals).
    found = read(out + "/abilities.json") or {}
    if not found.get("park") or not found.get("pid"):
        return False
    try:
        os.kill(int(found["pid"]), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
answer = {
    "new": len(new),
    "waiting": sorted(int(os.path.basename(p)[:-5]) for p in glob.glob(out + "/tasks/*.json")),
    "taken": sorted(int(os.path.basename(p).split(".")[0]) for p in glob.glob(out + "/taken/*.json")),
    # (only of this stage: a parked file left by an earlier stage on a machine kept for this one
    # must not make this stage's group of the same number look parked)
    "parked": sorted(int(os.path.basename(p)[:-5]) for p in glob.glob(out + "/parked/*.json")
                     if (read(p) or {}).get("stage", STAGE) == STAGE),
    "park_signals": sorted(int(os.path.basename(p)) for p in glob.glob(out + "/park/*")
                           if os.path.basename(p).isdigit()),
    "can_park": can_park(),
    "progress": progress,
    "alive": int(subprocess.run("pgrep -f '[v]3.machine_job --out " + BASE + "/out' | wc -l", shell=True,
                                capture_output=True, text=True).stdout.strip() or 0),
    "card": subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
                           shell=True, capture_output=True, text=True).stdout.strip(),
}
print("ANSWER " + json.dumps(answer))
'''

# Take several waiting groups back at once (each only if no worker has taken it).
TAKE_BACK_MANY = r'''
import json, os
taken = []
for number in NUMBERS:
    try:
        os.rename(BASE + "/out/tasks/%d.json" % number, BASE + "/out/taken_back_%d.json" % number)
        taken.append(number)
    except FileNotFoundError:
        pass
print("TAKEN " + json.dumps(taken))
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

# Ask the machine to park one group (see v3/machine_job.py): the worker training it saves it and
# sets it aside, so it can go on on a faster machine.
PARK = r'''
import os
os.makedirs(BASE + "/out/park", exist_ok=True)
open(BASE + "/out/park/%d" % NUMBER, "w").write("")
print("PARK SIGNALLED")
'''

# Take a parked group off the machine (WHERE = ("parked",)), or a group with a park signal that is
# waiting in the queue because its round ended before a worker saw the signal (WHERE = ("tasks",)).
# A rename succeeds only once, so the group cannot also be taken by a worker. The signal is
# removed only when the group was taken.
TAKE_PARKED = r'''
import os
out = BASE + "/out"
taken = False
for where in WHERE:
    try:
        os.rename(out + "/" + where + "/%d.json" % NUMBER, out + "/taken_back_parked_%d.json" % NUMBER)
        taken = True
        break
    except FileNotFoundError:
        pass
if taken and os.path.exists(out + "/park/%d" % NUMBER):
    os.remove(out + "/park/%d" % NUMBER)
print("PARKED TAKEN" if taken else "PARKED NOT THERE")
'''

# Remove park signals that no longer stand for a planned move (harmless if already gone).
UNPARK = r'''
import os
for number in NUMBERS:
    try:
        os.remove(BASE + "/out/park/%d" % number)
    except FileNotFoundError:
        pass
print("UNPARKED")
'''

# Put a parked group back into the machine's queue (its move up is given up): a worker there
# trains it on from its newest checkpoint. A rename succeeds only once.
UNPARK_BACK = r'''
import os
out = BASE + "/out"
try:
    os.rename(out + "/parked/%d.json" % NUMBER, out + "/tasks/%d.json" % NUMBER)
    print("BACK IN THE QUEUE")
except FileNotFoundError:
    print("NOT PARKED")
'''

# Date a parked group's hand-off checkpoint now, so the next look brings it home again (for when
# it came home once but this program was cut off before it noted that).
RESEND = r'''
import os
path = BASE + "/out/handoff/%d.pt" % NUMBER
if os.path.exists(path):
    os.utime(path)
print("RESENT")
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
                 machines=12, reserve=100.0, workers=2, cards=tuple(colab_gpu.ORDER),
                 lost_minutes=15.0, poll_seconds=40.0, prefix="air", prober=None,
                 grant_wait=8.0, ask_gain_minutes=5.0, ask_gain_share=0.05, shared_cards=None,
                 keep_machines=False):
        self.stage, self.groups, self.colab = stage, groups, colab
        self.runs = Path(runs_root or ROOT / "runs_v3")
        self.state_path = Path(state_path or ROOT / "logs" / "v3" / f"fleet_{stage}.json")
        self.log = log
        self.want, self.reserve, self.workers = machines, reserve, workers
        self.cards, self.lost_minutes, self.poll_seconds = list(cards), lost_minutes, poll_seconds
        self.prefix = prefix
        # A card type is asked for when one more machine of it would bring the end of the stage
        # earlier by at least this many seconds and this share of the time left.
        self.ask_gain_seconds, self.ask_gain_share = ask_gain_minutes * 60.0, ask_gain_share
        # Cards on which several workers share the card at the same moment through NVIDIA's MPS,
        # with the number of workers there: {"G4": 8, "A100": 3}. Measured on 23 November: with
        # MPS an A100 trains about 40 % more with 4 to 6 workers than with 2 the usual way.
        self.shared_cards = dict(shared_cards or {})
        self.keep_machines = keep_machines
        # Requests for machines: one per card type, no hidden retries (v3/colab_gpu.py). After
        # asking, a look waits up to `grant_wait` seconds, so a machine granted at once (Colab
        # grants in about 3 s) is put to work in the same look.
        self.prober = prober or colab_gpu.Prober(prefix, self.colab.tool)
        self.grant_wait = grant_wait
        self.checkpoints = self.runs / "_fleet" / stage / "checkpoints"
        self.state = {"machines": {}, "groups": {}, "failures": {}}
        if self.state_path.exists():
            # Also the state file of the launcher before 23 November 10:00 (adopted as it is:
            # the fields added since are filled in below and as the machines answer).
            self.state = json.loads(self.state_path.read_text())
        for key, empty in (("machines", {}), ("groups", {}), ("failures", {}), ("speeds", {}),
                           ("machine_speeds", {}), ("availability", {}), ("doubtful", {})):
            self.state.setdefault(key, empty)
        # Requests for machines that were waiting when this program was last stopped: Colab may
        # have granted them to nobody, so their cards are in doubt (see find_lost_machines).
        for card in self.state.pop("asking", {}).values():
            self.state["doubtful"][card] = time.time()
        self.looks = 0
        for machine in self.state["machines"].values():
            machine.setdefault("last_answer", time.time())
            machine.setdefault("started", False)
        self.speeds = fleet_plan.Speeds(self.state["speeds"], self.state["machine_speeds"])
        # The machines are asked at the same time, each in a thread of its own (fetch_all):
        # `fetching` holds the requests still on their way, `fetched` the answers of this look.
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)
        self.fetching, self.fetched = {}, {}
        self.estimate = None  # seconds until the stage ends, as estimated at the last look

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

    def take_answers(self):
        """The answers of Colab to the requests for machines: a granted machine joins the fleet
        (or is given back at once when the fleet is full). Returns the names of the new ones."""
        added = []
        order = {card: i for i, card in enumerate(colab_gpu.ORDER)}
        answers = sorted(self.prober.answers(), key=lambda a: order.get(a["card"], 99))
        for answer in answers:  # fastest card first
            card, outcome = answer["card"], answer["outcome"]
            self.state.setdefault("asking", {}).pop(answer["session"], None)
            self.state["availability"][card] = {"outcome": outcome, "at": time.time(),
                                                "seconds": answer["seconds"]}
            if outcome == "granted":
                name = answer["session"]
                count = len(self.state["machines"])
                # When the fleet is full, a card faster than its slowest is still taken (one over
                # the number for a while): the slow machine's groups move to it and the slow
                # machine is then given back.
                faster = count and self.speeds.card_factor(card) > min(
                    self.speeds.card_factor(m["card"]) for m in self.state["machines"].values())
                if count >= self.want + 1 or (count >= self.want and not faster):
                    colab_gpu.stop([name], self.colab.tool)
                    self.say(f"machine {name} ({card}) granted but the fleet is full: given back")
                    continue
                self.state["machines"][name] = {"card": card, "last_answer": time.time(),
                                                "started": False, "added": time.time()}
                added.append(name)
                self.say(f"machine {name} ({card}) added (granted after {answer['seconds']} s)")
            elif outcome == "limit":
                # Colab refused for another reason than "no card free" (412: too many machines
                # at once, or a usage limit). Noted, so the owner learns Colab's limit.
                count = len(self.state["machines"])
                have = sum(1 for m in self.state["machines"].values() if m["card"] == card)
                # On 23 November Colab refused a fourth G4 and a fourth A100 with 9 and with 10
                # machines in the fleet, and a fourth T4 with 3 machines: its limit seems to be
                # 3 machines of one card type. So the count per type is what is remembered.
                if self.state.get("card_limit", {}).get(card) != have:
                    self.say(f"Colab refused a {card} with 'precondition failed' while the fleet "
                             f"had {have} {card} ({count} machines in all): Colab's limit for "
                             f"this card type (or a usage limit) is reached")
                self.state.setdefault("card_limit", {})[card] = have
                self.state["limit"] = {"machines": count, "at": time.time(), "card": card}
            elif outcome not in ("no card free", "not allowed"):
                # Neither a grant nor a clear no (a time-out, an error, no answer): Colab may
                # still have granted the machine to nobody. find_lost_machines looks for it.
                self.state["doubtful"][card] = time.time()
                self.say(f"request for a {card}: {outcome} ({answer['text']})")
        if added:
            self.save()
        return added

    def find_lost_machines(self):
        """Machines that Colab runs for this account but that no launcher knows: granted to a
        request whose answer was lost (the request timed out or failed after Colab said yes, or
        the launcher was stopped while it waited). Such a machine uses units for ever, so it is
        taken into the fleet (or given back at once when the fleet is full).

        Two kinds:
          - in a state file of this fleet's tool (its name starts with our prefix) but in no
            launcher's state file: ours for certain;
          - in no state file at all: the request was cut off before the tool could write it. It
            might also be a notebook the owner opened in a browser, so only a machine of a card
            type that had such a doubtful request in the half hour before is taken.
        A machine is taken only when it was found twice, STRAY_MINUTES apart, with no request
        for its card waiting meanwhile (a request just granted writes its file a moment later)."""
        running = colab_gpu.server_machines(self.colab.tool)
        if running is None:
            return
        held = set(self.state["machines"])
        for path in self.state_path.parent.glob("fleet_*.json"):
            if path != self.state_path:
                try:
                    held |= set(json.loads(path.read_text()).get("machines", {}))
                except (OSError, ValueError):
                    continue
        known = colab_gpu.known_sessions()
        doubtful = self.state.setdefault("doubtful", {})
        seen = self.state.setdefault("strays", {})
        now = time.time()
        found = {}
        for endpoint, card in running.items():
            name = known.get(endpoint, (None, 0))[0]
            if name is not None and (name in held or not name.startswith(self.prefix + "-")):
                continue  # a machine of a launcher, or not of this fleet's tool (the owner's)
            if name is None and now - doubtful.get(card, -1e12) > 1800:
                continue  # unknown, but no request for this card was in doubt: not ours
            found[endpoint] = (name, card)
        for endpoint in list(seen):
            if endpoint not in found:
                del seen[endpoint]
        pending = self.prober.pending_cards()
        for endpoint, (name, card) in found.items():
            first = seen.setdefault(endpoint, now)
            if now - first < STRAY_MINUTES * 60 or card in pending:
                continue
            del seen[endpoint]
            if name is None:
                name = f"{self.prefix}-{card.lower()}-found-{time.strftime('%H%M%S')}"
                colab_gpu.take_in_endpoint(name, endpoint)
            self.say(f"machine {endpoint} ({card}) is running but no launcher knew it (a grant "
                     f"whose answer was lost); it is taken as {name}")
            self.prober.found.append({"session": name, "card": card, "outcome": "granted",
                                      "seconds": round(now - first), "text": "found"})
        self.save()

    def workers_of(self, name):
        """How many groups a machine trains at once: more on a card that is shared with MPS."""
        card = self.state["machines"].get(name, {}).get("card")
        return self.shared_cards.get(card, self.workers)

    def start_job(self, name):
        card = self.state["machines"][name]["card"]
        worked, printed = self.colab.execute(
            name, with_values(START, PACKED=packed_code(), WORKERS=self.workers_of(name),
                              SHARED=card in self.shared_cards), timeout=600)
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

    def fetch(self, name, started):
        """What a look at one machine needs from Colab: whether it is still there, and (if its
        job was started) its answer with its new files. It runs in a thread of its own and
        changes nothing in the state: what it learns is taken in by look_at."""
        exists = self.colab.exists(name)
        answer = self.poll(name, apply=False) if started and exists is not False else None
        return {"exists": exists, "polled": bool(started), "answer": answer}

    def fetch_all(self, wait_seconds=75.0):
        """Ask every machine at the same time. Before 23 November 10:45 they were asked one
        after another, and a request to a machine in a far-away data centre that stalled for
        minutes held up the look at all the others and the search for new cards. A machine that
        has not answered within `wait_seconds` is left out of this look; its request goes on,
        no second request is sent to it meanwhile, and its answer is used at the next look."""
        for name, machine in self.state["machines"].items():
            if name not in self.fetching:
                self.fetching[name] = self.pool.submit(self.fetch, name,
                                                       bool(machine.get("started")))
        concurrent.futures.wait(list(self.fetching.values()), timeout=wait_seconds)
        self.fetched = {}
        for name, request in list(self.fetching.items()):
            if not request.done():
                continue
            del self.fetching[name]
            try:
                self.fetched[name] = request.result()
            except Exception as error:  # (a look that failed counts as no answer)
                self.say(f"machine {name}: the look failed ({error!r})")

    def apply_saved(self, answer):
        """The checkpoints that came home with an answer: note the step each one stands at."""
        for key, step in (answer or {}).pop("saved", {}).items():
            self.state["groups"].setdefault(key, {})["saved_step"] = step

    def poll(self, name, apply=True):
        """Ask the machine how far it is and download what is new. Returns its answer or None.
        With apply=False (in a thread) the state is not touched; look_at calls apply_saved."""
        # At most 2 minutes a try: on 23 November a look at a new L4 hung twice for the whole
        # 5 minutes (and held up the look at every other machine); the third try answered at once.
        asked = time.time()
        worked, printed = self.colab.execute(name, with_values(POLL, STAGE=self.stage),
                                             timeout=120)
        lines = [line for line in printed.splitlines() if line.startswith("ANSWER ")]
        if not worked or not lines:
            return None
        answer = json.loads(lines[-1][len("ANSWER "):])
        # When it was asked (on this computer's clock): an answer asked for before a park signal
        # was written cannot tell whether the signal arrived (see settle_parking).
        answer["asked_at"] = asked
        if answer["new"]:
            with tempfile.TemporaryDirectory() as folder:
                local = Path(folder) / "new.tar"
                if not self.colab.download(name, f"{BASE}/new.tar", local):
                    self.say(f"machine {name}: download failed; it comes again next time")
                    answer["incomplete"] = True  # (its new files are still only there)
                    return answer
                self.take_in(local, answer)
        self.colab.execute(name, with_values(COMMIT))
        if apply:
            self.apply_saved(answer)
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
                    answer.setdefault("saved", {})[path.stem] = reported["step"]

    def note_progress(self, name, answer):
        """Record what the machine said about its groups."""
        for key, found in answer["progress"].items():
            group = self.state["groups"].setdefault(key, {})
            if group.get("machine") != name:
                continue
            # the step from which another machine could go on: that of the checkpoint here
            group["step"] = group.get("saved_step", 0)
            if found.get("step") is not None and found.get("step") != group.get("reached"):
                self.round_ended(name, key, found, len(answer["taken"]))
            group["reached"] = found.get("step")
            if found.get("members") is not None:
                group["members"] = found["members"]
            if found.get("error") and not found.get("done"):
                self.state["failures"][key] = self.state["failures"].get(key, 0) + 1
                group["machine"] = None
                self.say(f"group {key} failed on {name} (failure "
                         f"{self.state['failures'][key]}): "
                         f"{(found['error'] or '').strip().splitlines()[-1:]}")
        self.save()

    def round_ended(self, name, key, found, training=1):
        """A group finished a round on machine `name`: its speed there is recorded."""
        machine = self.state["machines"][name]
        group = self.state["groups"][key]
        seconds = found.get("seconds") or 0
        if not found.get("member_steps"):
            return  # (the progress written when the group was given, not a round)
        if seconds < 180:
            # A sitting cut short (the machine's job was stopped and started again, as for the
            # switch to shared cards on 23 November) is mostly start-up: it made the H100 look
            # half as fast as an A100. Too short to tell a speed from.
            return
        if name not in group.get("compiled_on", []):
            # the first round here includes compiling the group's shape (60 to 165 s)
            # (at most 4 tenths of the round: taking off 120 s made a round of 150 s look more
            # than 3 times as fast as it was, and so the fast cards faster than they are)
            seconds = seconds - min(0.8 * fleet_plan.MOVE_SECONDS, 0.4 * seconds)
        # The groups that shared the card: those training now (the groups it holds that only
        # wait did not share it; counting them would make a machine look up to twice as fast).
        sharing = min(self.workers_of(name), max(1, training))
        shape = fleet_plan.shape_of(self.groups[int(key)])
        whole = self.speeds.record(name, machine["card"], shape, found.get("member_steps"),
                                   seconds, sharing)
        group.setdefault("compiled_on", [])
        if name not in group["compiled_on"]:
            group["compiled_on"].append(name)
        group.pop("training_since", None)
        if whole:
            self.say(f"machine {name} ({machine['card']}): group {key} (shape {shape}) at "
                     f"{whole:,.0f} member-steps a second for the whole card")

    def give_up(self, name, why):
        """The machine is lost: its groups go back to the pool, and it is stopped."""
        self.say(f"machine {name} given up ({why}); its groups go to other machines")
        self.release(name)
        self.colab.stop(name)
        del self.state["machines"][name]
        self.save()

    def release(self, name):
        """The groups of machine `name` go back to the pool, from their checkpoint here."""
        for group in self.state["groups"].values():
            if group.get("machine") == name:
                group["machine"] = None
                group["step"] = group.get("saved_step", 0)
                group.pop("training_since", None)
            # A move up to or from this machine will not happen now (its group is free, or the
            # machine it was going to is gone): the planner decides again.
            if (group.get("parking") or {}).get("from") == name or \
                    (group.get("parking") or {}).get("to") == name:
                group.pop("parking", None)

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

    # --- planning: who trains what (v3/fleet_plan.py does the arithmetic) ---------------------

    def plan_machine(self, name, card, ready=0.0):
        return {"card": card, "ready": ready,
                "rate": lambda shape: self.speeds.rate(card, shape, name)}

    def survival(self):
        """The share of members expected to go on after a look: measured from the groups that
        have passed looks (from 3 groups on), else the guess."""
        shares = []
        for key, group in self.state["groups"].items():
            configs = self.groups[int(key)]
            reached, members = group.get("reached") or 0, group.get("members")
            first = configs[0].get("first_look")
            if members is None or not first or reached < first:
                continue
            looks = math.log2(reached / first) + 1
            shares.append((members / len(configs)) ** (1 / looks))
        if len(shares) < 3:
            return fleet_plan.SURVIVAL
        return min(0.95, max(0.2, sum(shares) / len(shares)))

    def model(self, answers):
        """The machines and groups as the planner sees them (see v3/fleet_plan.py)."""
        now = time.time()
        machines = {name: self.plan_machine(name, self.state["machines"][name]["card"])
                    for name in answers}
        survival = self.survival()
        finished = self.finished_names()
        groups, where, movable = {}, {}, []
        for index in self.open_groups():
            group = self.state["groups"].setdefault(str(index), {})
            holder = group.get("machine")
            if holder is None and self.state["failures"].get(str(index), 0) >= 3:
                continue
            if holder is not None and holder not in answers:
                continue  # on a machine that is leaving or did not answer: it stays there
            configs = self.groups[index]
            shape = fleet_plan.shape_of(configs)
            members = group.get("members")
            if members is None:
                members = sum(1 for c in configs if c["name"] not in finished)
            training = holder is not None and index in answers[holder]["taken"]
            waiting = holder is not None and index in answers[holder]["waiting"]
            reached = (group.get("reached") or 0) if holder else (group.get("step") or 0)
            done_in_round = 0.0
            compiled = set(group.get("compiled_on", []))
            if training:
                compiled.add(holder)
                since = group.setdefault("training_since", now)
                sharing = max(1, min(self.workers_of(holder), len(answers[holder]["taken"])))
                done_in_round = machines[holder]["rate"](shape) / sharing * (now - since)
            else:
                group.pop("training_since", None)
            groups[index] = {"work": fleet_plan.work_left(
                                 configs[0], reached, members,
                                 fleet_plan.survival_guess(configs[0], survival), done_in_round),
                             "shape": shape, "compiled_on": compiled}
            where[index] = holder
            # Moved only between rounds, and only with its checkpoint already here. A group that
            # is being parked for a move up is left to that move (settle_parking).
            pinned = group.get("pinned_until", 0) > now  # (just put on a free worker: it stays)
            if holder is None or (waiting and not pinned and not group.get("parking")
                                  and (reached == 0 or group.get("saved_step") == reached)):
                movable.append(index)
        return machines, groups, where, movable

    def replan(self, answers):
        """Hand out the free groups and make the moves that bring the end of the stage earlier
        (v3/fleet_plan.py); stop a slow machine in the middle of its rounds only when the stage
        then ends clearly earlier. Returns the model and the plan (for asking for machines)."""
        if not answers:
            return None
        machines, groups, where, movable = self.model(answers)
        start = fleet_plan.improve(machines, groups, where, [])  # free groups only
        plan = fleet_plan.improve(machines, groups, start, movable)
        ends = fleet_plan.ends_of(machines, groups, plan)
        end = fleet_plan.stage_end(ends)
        leaving = any(m.get("leaving") for m in self.state["machines"].values())
        # A slow machine stopped in the middle of its rounds (it saves first; its groups go on
        # elsewhere from there): only if the stage then ends clearly earlier. Only the slowest
        # machine with groups in the middle of a round is considered, one at a time.
        if not leaving and len(machines) > 1:
            busy = [n for n in machines if machines[n]["card"] not in FAST_CARDS
                    and any(h == n and i not in movable for i, h in plan.items())]
            for name in sorted(busy, key=lambda n: self.speeds.card_factor(
                    machines[n]["card"]))[:1]:
                trial = fleet_plan.without(machines, groups, plan, movable, name)
                if not trial:
                    continue
                gain = end - fleet_plan.stage_end(trial[1])
                if gain > max(600.0, 0.15 * end):
                    self.say(f"machine {name} ({machines[name]['card']}): stopped in the middle "
                             f"of its rounds (they are saved first); the stage should end about "
                             f"{gain / 60:.0f} minutes earlier with its groups elsewhere")
                    self.colab.execute(name, with_values(SIGNAL, WHAT="stop"))
                    self.state["machines"][name]["leaving"] = leaving = True
                    machines = {n: m for n, m in machines.items() if n != name}
                    # Its groups between rounds move now; the others when it has saved them.
                    plan = {i: (name if plan[i] == name and i not in movable else n)
                            for i, n in trial[0].items()}
                    ends, end = trial[1], fleet_plan.stage_end(trial[1])
        # Cost as a tie-breaker: a machine whose groups are all between rounds is given back if
        # the stage does not end later without it (the most expensive per speed first).
        by_cost = sorted(machines, key=lambda n: -fleet_plan.UNITS_AN_HOUR.get(
            machines[n]["card"], 7.0) / self.speeds.card_factor(machines[n]["card"]))
        for name in by_cost if not leaving else []:
            mine = [i for i, n in plan.items() if n == name]
            if machines[name]["card"] in FAST_CARDS:
                continue  # a fast card is kept
            if len(machines) < 2 or any(i not in movable for i in mine):
                continue
            trial = fleet_plan.without(machines, groups, plan, movable, name)
            latest = max(ends, key=ends.get)
            # Not later, and with room to spare on the others (the estimates are rough).
            if trial and fleet_plan.stage_end(trial[1]) <= end + 1.0 and all(
                    e <= 0.85 * end for n, e in trial[1].items() if n != latest):
                if mine:
                    self.say(f"machine {name}: its groups go to the others (the stage does not "
                             f"end later, and it costs less)")
                plan = trial[0]
                machines = {n: m for n, m in machines.items() if n != name}
                ends, end = trial[1], fleet_plan.stage_end(trial[1])
                break
        self.carry_out(plan, where, answers)
        self.estimate = end
        return machines, groups, plan, movable

    def fill_free_workers(self, answers):
        """A fast machine whose workers share the card takes, on its free workers, groups that
        only wait in another machine's queue (the ones with the most work first).

        The planner counts a machine's speed as fixed, so it sees no gain in this. But a shared
        card trains more in total the more groups it has at once (measured on the H100 on
        23 November: 11,632 model-steps a second with 4 programs, 15,088 with 16), and a group
        that waits in a queue trains nowhere. A group moved this way stays where it is for 20
        minutes, so that the planner does not send it straight back."""
        machines, groups, where, movable = self.model(answers)
        waiting = sorted(((groups[i]["work"], i) for i in movable
                          if where[i] is not None and i in answers[where[i]]["waiting"]), reverse=True)
        plan = dict(where)
        fastest_first = sorted(answers, key=lambda n: -self.speeds.card_factor(
            self.state["machines"][n]["card"]))
        for name in fastest_first:
            machine = self.state["machines"][name]
            if machine["card"] not in self.shared_cards or machine.get("leaving"):
                continue
            free = self.free_workers(name, answers)
            for work, index in list(waiting):
                if free <= 0:
                    break
                holder = where[index]
                if holder == name:
                    continue
                # A machine as fast as this one, with a worker for every group in its queue,
                # starts the group itself within seconds: moving it would only cost an upload
                # and a compile (and two such cards would pass groups back and forth).
                if (self.free_workers(holder, answers) >= 0
                        and self.speeds.card_factor(self.state["machines"][holder]["card"])
                        >= self.speeds.card_factor(machine["card"])):
                    continue
                plan[index] = name
                waiting.remove((work, index))
                free -= 1
        moved = [i for i in plan if plan[i] != where[i]]
        if not moved:
            return
        self.say(f"free workers on shared cards take {len(moved)} groups that were waiting elsewhere")
        self.carry_out(plan, where, answers)
        for index in moved:
            group = self.state["groups"][str(index)]
            if group.get("machine") == plan[index]:
                group["pinned_until"] = time.time() + 1200
        self.save()

    # --- move up: a group in the middle of a round goes to a faster machine -------------------
    #
    # How a move up goes (one group g, from the slower machine S to the faster machine F):
    #   1. move_up decides it, notes {"from": S, "to": F, "at": now} as g's "parking" in the state
    #      file (saved), then writes the signal out/park/<g> on S.
    #   2. On S the worker training g sees the signal within seconds, saves g (progress and the
    #      hand-off checkpoint as at the end of any round), moves g's task file to out/parked/ and
    #      removes the signal. (If g's round ended before that, g waits in out/tasks/ with the
    #      signal; the next worker that looks for work parks it, or settle_parking takes it.)
    #   3. The next looks bring g's checkpoint home. Once the step of the checkpoint here equals
    #      the step g reached on S, settle_parking takes g's task file off S (a rename that can
    #      succeed only once), notes g as free, and gives it to F if F still has a free worker
    #      (else the planner hands it out like any free group).
    # A group is never on two machines: S keeps it (in the state file) until its file is off S.

    def incoming(self, name):
        """Groups being parked elsewhere to come to machine `name` (a worker is kept for each)."""
        return sum(1 for g in self.state["groups"].values()
                   if (g.get("parking") or {}).get("to") == name)

    def free_workers(self, name, answers):
        """Workers of a machine with nothing to do: its workers, less the groups it trains, the
        groups waiting in its queue and the groups on their way to it by a move up."""
        answer = answers[name]
        return (self.workers_of(name) - len(answer["taken"]) - len(answer["waiting"])
                - self.incoming(name))

    def settle_parking(self, answers):
        """Carry on the moves up that are under way (see above), and clear up what is left of
        moves that did not happen. Every case is safe in any order of events: see the report in
        v3/tests/test_v3_fleet_move_up.py."""
        for name, answer in answers.items():
            machine = self.state["machines"].get(name)
            if machine is None or machine.get("leaving"):
                continue  # (a leaving machine's groups are all freed when it has stopped)
            signals = set(answer.get("park_signals", []))
            parked = set(answer.get("parked", []))
            # 1. Groups parked on this machine.
            for index in sorted(parked):
                group = self.state["groups"].setdefault(str(index), {})
                if group.get("machine") not in (None, name) or self.group_done(index):
                    # An old copy: the group is on another machine now (or done). Removed, so
                    # that nothing ever trains it here again.
                    self.colab.execute(name, with_values(TAKE_PARKED, NUMBER=index,
                                                         WHERE=("parked",)))
                    continue
                if group.get("machine") is None:
                    group["machine"] = name  # (as reconcile does for groups found training)
                self.hand_on(name, index, answer, answers, was_parked=True)
            # 2. Moves up from this machine whose group is not parked (yet).
            for key, group in list(self.state["groups"].items()):
                record = group.get("parking")
                index = int(key)
                if not record or record.get("from") != name or index in parked:
                    continue
                if group.get("machine") != name or self.group_done(index) or \
                        answer["progress"].get(key, {}).get("done"):
                    # It ended (or failed, or went elsewhere) before it could be parked.
                    group.pop("parking", None)
                    self.say(f"group {index}: the move up from {name} is dropped (the group "
                             f"ended its training there or left it first)")
                    continue
                if answer.get("asked_at", 0) < record.get("at", 0):
                    continue  # an answer asked for before the signal was written: wait
                if index in answer["waiting"]:
                    # Its round ended before a worker saw the signal: take it from the queue.
                    self.hand_on(name, index, answer, answers, was_parked=False)
                elif index in answer["taken"]:
                    if index not in signals:
                        # The signal never arrived (the request failed, or this program was
                        # cut off before it sent it): the move is dropped; it may be chosen again.
                        group.pop("parking", None)
                        self.say(f"group {index}: the park signal did not reach {name}; the move "
                                 f"up is dropped")
                    # (else: the worker parks it within seconds; the next look sees it)
                else:
                    # Not on the machine at all: it was taken off already (this program was cut
                    # off after that, before it noted it). It goes on from its checkpoint here.
                    # Only when it was missing in two answers in a row (look_at counts them): a
                    # look can miss a group in the instant a worker puts it back in the queue, and
                    # freeing it then would let two machines train it.
                    if group.get("absent", 0) >= 2:
                        self.free_after_parking(name, index, answer, answers)
            # 3. Signals that no planned move stands for any more (the group ended, or the move was
            #    dropped): removed, so that they cannot park the group later by surprise.
            planned = {int(k) for k, g in self.state["groups"].items()
                       if (g.get("parking") or {}).get("from") == name}
            stale = sorted(signals - planned)
            if stale:
                self.colab.execute(name, with_values(UNPARK, NUMBERS=stale))
        self.save()

    def hand_on(self, name, index, answer, answers, was_parked):
        """Group `index` is parked on machine `name` (or waits there with a park signal): once its
        checkpoint is home, take it off the machine and give it on."""
        group = self.state["groups"][str(index)]
        reported = answer["progress"].get(str(index), {}).get("step") or 0
        if reported and group.get("saved_step") != reported:
            # The checkpoint of the step it reached is not here yet. It normally comes with the
            # next look or the one after. If it has not come after RESEND_SECONDS, it is sent
            # again (it may have come once while this program was cut off before noting it, and
            # is then never packed again). Not more often: sending again dates the file now, and
            # a look packs only files older than 5 seconds, so asking at every look would keep
            # it from ever coming.
            now = time.time()
            if was_parked and now - group.setdefault("parked_since", now) > PARK_WAIT_SECONDS:
                # Still not here after many looks (e.g. its hand-off file is missing): rather
                # than leave the group parked for ever, it goes back into the slower machine's
                # queue and trains on there, from its newest checkpoint, which is there.
                worked, printed = self.colab.execute(name, with_values(UNPARK_BACK, NUMBER=index))
                if worked and "BACK IN THE QUEUE" in printed:
                    group.pop("parking", None)
                    group.pop("parked_since", None)
                    group.pop("resent_at", None)
                    self.say(f"group {index}: its checkpoint did not come home from {name} in "
                             f"{PARK_WAIT_SECONDS / 60:.0f} minutes; the move up is dropped and it "
                             f"trains on there")
                return
            since = max(group.setdefault("parked_since", now), group.get("resent_at", 0))
            if now - since > RESEND_SECONDS and not answer.get("incomplete"):
                self.colab.execute(name, with_values(RESEND, NUMBER=index))
                group["resent_at"] = now
            return
        # (Only from where the answer saw it: a parked group only from parked/, a waiting one
        # only from the queue, so that what is taken is what the answer's step stands for.)
        worked, printed = self.colab.execute(name, with_values(
            TAKE_PARKED, NUMBER=index, WHERE=("parked",) if was_parked else ("tasks",)))
        if not worked:
            return  # tried again at the next look
        if "PARKED TAKEN" not in printed:
            # Not there any more: a worker took it from the queue meanwhile (it parks it), or it
            # went back into the queue (UNPARK_BACK) after this answer was asked, or this program
            # took it off before and was cut off before noting it. Nothing is freed on this
            # answer alone: a group missing from the machine in two answers in a row is freed by
            # settle_parking (or by look_at), so it can never train on two machines.
            return
        if answer and index in answer["waiting"]:
            answer["waiting"].remove(index)
        self.free_after_parking(name, index, answer, answers)

    def free_after_parking(self, name, index, answer, answers):
        """The group's file is off machine `name` and its checkpoint is here: it is free, and goes
        to the machine of its move up if that one still has a free worker."""
        group = self.state["groups"][str(index)]
        record = group.pop("parking", None) or {}
        reported = answer["progress"].get(str(index), {}).get("step") or 0
        group["machine"] = None
        group["step"] = group.get("saved_step", 0) if reported else 0
        group.pop("training_since", None)
        group.pop("parked_since", None)
        group.pop("resent_at", None)
        group.pop("absent", None)
        self.save()  # (free before it is given: a cut-off now loses nothing)
        target = record.get("to")
        usable = (target in answers and target in self.state["machines"]
                  and not self.state["machines"][target].get("leaving"))
        if usable and self.free_workers(target, answers) > 0:
            self.say(f"group {index}: saved on {name} at step {group['step']:,}; it goes on on "
                     f"{target}")
            self.give(target, [index])
            if group.get("machine") == target:
                answers[target]["waiting"].append(index)
                group["pinned_until"] = time.time() + PIN_SECONDS
                return
        self.say(f"group {index}: saved on {name} at step {group['step']:,}; "
                 + (f"{target} has no free worker any more, so " if target else "")
                 + "it is handed out like any free group")

    def move_up_rate(self, name, shape, high):
        """The speed of machine `name` for a group of `shape` (member-steps a second for the whole
        card) as a move up counts it, or None while its card has no measured speed at all (then
        the speed is only a guess, and no move is decided on it). Of the speed for this shape and
        the card's typical speed, the higher (`high`, for the machine the group would leave) or
        the lower (for the machine it would go to): so one odd reading cannot cause a move."""
        card = self.state["machines"][name]["card"]
        if not self.speeds.measured.get(card):
            return None
        for_shape = self.speeds.rate(card, shape, name)
        typical = fleet_plan.a100_guess(shape) * self.speeds.card_factor(card)
        return max(for_shape, typical) if high else min(for_shape, typical)

    def move_up(self, answers):
        """Choose the groups in the middle of a round that go to a faster machine with a free
        worker (the rule at the top of this file), and start their moves. Largest gain first."""
        now = time.time()
        live = {n: a for n, a in answers.items() if n in self.state["machines"]
                and not self.state["machines"][n].get("leaving")}
        if len(live) < 2:
            return
        _, groups, where, _ = self.model(live)

        def choose(skip, reserved=None):
            """The moves, chosen greedily; machines in `skip` give no groups, and `reserved`
            ({machine: workers}) are workers already promised to the groups of a machine that
            leaves as a whole."""
            free = {n: self.free_workers(n, live) - (reserved or {}).get(n, 0) for n in live}
            busy = {n: len(live[n]["taken"]) + self.incoming(n) for n in live}
            options = []
            for source in live:
                if source in skip:
                    continue
                for index in live[source]["taken"]:
                    group = self.state["groups"].get(str(index), {})
                    if where.get(index) != source or group.get("parking") \
                            or group.get("pinned_until", 0) > now:
                        continue
                    options.append((source, index))
            chosen = []
            while True:
                best = None
                for source, index in options:
                    shape, work = groups[index]["shape"], groups[index]["work"]
                    here = self.move_up_rate(source, shape, high=True)
                    if here is None:
                        continue  # no measured speed for this card yet: only a guess
                    stay = work / (here / max(1, busy[source]))
                    for target in live:
                        if target == source or free[target] <= 0:
                            continue
                        there = self.move_up_rate(target, shape, high=False)
                        if there is None or there <= here:
                            continue  # not faster for this shape
                        go = MOVE_UP_SECONDS + work / (there / (busy[target] + 1))
                        gain = stay - go
                        if gain > MOVE_UP_MARGIN and go < MOVE_UP_SHARE * stay and (
                                best is None or gain > best[0]):
                            best = (gain, source, index, target, stay, go)
                if best is None:
                    return chosen
                gain, source, index, target, stay, go = best
                chosen.append(best)
                options.remove((source, index))
                # Count the move: one worker less on the target, one group more training there,
                # one less on the source (whose other groups then train faster).
                free[target] -= 1
                busy[target] += 1
                busy[source] -= 1

        # A machine whose job cannot park one group (started by an older machine_job.py): moved
        # only as a whole, when every group it trains would move up (and it has none waiting).
        # This holds for fast cards too (the owner's rule of 23 November: an A100 whose groups
        # all end sooner on an H100 gives them up, as the A100s released by hand that day).
        old_jobs = {n for n in live if not live[n].get("can_park")}
        first = choose(set())
        whole = []
        for name in sorted(old_jobs):
            mine = [c for c in first if c[1] == name]
            if mine and len(mine) == len(live[name]["taken"]) and not live[name]["waiting"]:
                whole.append((name, mine))
        reserved = {}
        for name, mine in whole:
            worked, printed = self.colab.execute(name, with_values(SIGNAL, WHAT="stop"))
            if worked and "SIGNALLED" in printed:
                self.state["machines"][name]["leaving"] = True
                # Its groups come free once it has saved them: the free workers they would take
                # are not given to other moves in this look.
                for c in mine:
                    reserved[c[3]] = reserved.get(c[3], 0) + 1
                soonest = max(c[0] for c in mine)
                self.say(f"machine {name} ({self.state['machines'][name]['card']}) cannot park "
                         f"one group, and all its groups {sorted(c[2] for c in mine)} should end "
                         f"sooner on faster machines (up to {soonest / 60:.0f} minutes): it saves "
                         f"them and is given back, and they go on elsewhere")
        for gain, source, index, target, stay, go in choose(old_jobs, reserved):
            group = self.state["groups"][str(index)]
            # Noted (and saved) before the signal is sent: a program cut off in between finds
            # the note without a signal and drops the move (settle_parking).
            group["parking"] = {"from": source, "to": target, "at": time.time()}
            for old in ("parked_since", "resent_at", "absent"):  # (left from an earlier move)
                group.pop(old, None)
            group["pinned_until"] = time.time() + PIN_SECONDS
            self.save()
            # (Written before the signal, so the log has it even if this program is cut off.)
            self.say(f"group {index} moves up from {source} "
                     f"({self.state['machines'][source]['card']}) to {target} "
                     f"({self.state['machines'][target]['card']}) in the middle of its round: it "
                     f"should end about {gain / 60:.0f} minutes sooner ({go / 60:.0f} instead of "
                     f"{stay / 60:.0f} minutes)")
            worked, printed = self.colab.execute(source, with_values(PARK, NUMBER=index))
            if not (worked and "PARK SIGNALLED" in printed):
                # (If it was written after all, the next look removes it as a stale signal.)
                group.pop("parking", None)
                group.pop("pinned_until", None)
                self.say(f"group {index}: the park signal could not be sent to {source}; the "
                         f"move up is dropped")
        self.save()

    def carry_out(self, plan, where, answers):
        """Make the moves of the plan: take each group back from its machine (only if no worker
        has taken it meanwhile), then give it to its new machine with its checkpoint."""
        leaving = {}
        for index, target in plan.items():
            if target != where[index] and where[index] is not None:
                leaving.setdefault(where[index], []).append(index)
        moved = set()
        for donor, numbers in leaving.items():
            worked, printed = self.colab.execute(donor, with_values(TAKE_BACK_MANY,
                                                                   NUMBERS=numbers))
            lines = [line for line in printed.splitlines() if line.startswith("TAKEN ")]
            if not worked or not lines:
                continue  # not taken back: they stay where they are
            for number in json.loads(lines[-1][len("TAKEN "):]):
                group = self.state["groups"][str(number)]
                group["machine"] = None
                group["step"] = group.get("saved_step", 0) if group.get("reached") else 0
                if number in answers[donor]["waiting"]:
                    answers[donor]["waiting"].remove(number)
                moved.add(number)
        share = {}
        for index, target in plan.items():
            if target is not None and (where[index] is None or index in moved):
                share.setdefault(target, []).append(index)
        for target, indices in share.items():
            sources = sorted({where[i] for i in indices if where[i] is not None})
            if sources:
                self.say(f"groups {sorted(i for i in indices if where[i])} move from "
                         f"{', '.join(sources)} to {target}")
            self.give(target, sorted(indices))
            answers[target]["waiting"] += [i for i in indices
                                           if self.state["groups"][str(i)].get("machine") == target]

    def ask_for_machines(self, model):
        """Ask Colab for a machine of each card type (fastest first) that would make the stage
        end clearly earlier, up to --machines. Getting the machine is the test."""
        pending = self.prober.pending_cards()
        count = len(self.state["machines"])
        if count > self.want or not self.open_groups():
            return
        slowest = min((self.speeds.card_factor(m["card"]) for m in self.state["machines"].values()),
                      default=0.0)
        # Every card type that would help is asked for at once (fastest first): getting the
        # machine is the test. If more are granted than there is room for, the fastest are kept.
        asked = []
        for card in self.cards:
            if card in pending:
                continue
            if count >= self.want and self.speeds.card_factor(card) <= slowest:
                continue  # full: only a faster card than the slowest machine is of use
            have = sum(1 for m in self.state["machines"].values() if m["card"] == card)
            refused_at = self.state.get("card_limit", {}).get(card)
            last = self.state["availability"].get(card, {})
            if (refused_at is not None and have >= refused_at
                    and time.time() - last.get("at", 0) < 1800):
                continue  # Colab's limit for this card type: asked again in half an hour
            if model is not None:
                machines, groups, plan, movable = model
                end = fleet_plan.stage_end(fleet_plan.ends_of(machines, groups, plan))
                trial_machines = dict(machines)
                trial_machines["(new)"] = self.plan_machine(None, card,
                                                            fleet_plan.NEW_MACHINE_SECONDS)
                trial = fleet_plan.improve(trial_machines, groups, plan, movable)
                best = fleet_plan.stage_end(fleet_plan.ends_of(trial_machines, groups, trial))
                # with the new machine, the slowest one may also give up its groups
                slow = sorted(machines, key=lambda n: self.speeds.card_factor(
                    machines[n]["card"]))[:1]
                for name in slow:
                    out = fleet_plan.without(trial_machines, groups, trial, movable, name)
                    if out:
                        best = min(best, fleet_plan.stage_end(out[1]))
                if end - best < max(self.ask_gain_seconds, self.ask_gain_share * end):
                    continue
            name = self.prober.ask(card)
            if name:  # written down until answered (in doubt if this program is cut off)
                self.state.setdefault("asking", {})[name] = card
            asked.append(card)
        if asked and asked != self.state.get("last_asked"):
            self.say(f"asking Colab for: {', '.join(asked)} (once each, at every look)")
        self.state["last_asked"] = asked
        self.save()

    # --- the main loop -------------------------------------------------------------------------

    def units_ok(self):
        found = self.colab.balance()
        if found is None:
            return True  # cannot tell; asked again next time
        left, rate = found
        # the units that the machines (and those we may still add) use before the next look
        extra = 7.0 * min(len(self.cards), max(0, self.want - len(self.state["machines"])))
        need = (rate + extra) * (3 * self.poll_seconds / 3600) + 1.0
        if left - need < self.reserve:
            self.say(f"compute units: {left:.1f} left, using {rate:.1f} an hour; the reserve of "
                     f"{self.reserve:.0f} would be reached: stopping everything cleanly")
            return False
        return True

    def stop_all(self, wait_minutes=8.0):
        """Ask every job to stop, bring home what they saved, stop the machines."""
        self.prober.close()  # a machine granted from now on is given back at once
        for name in list(self.state["machines"]):
            self.colab.execute(name, with_values(SIGNAL, WHAT="stop"))
        until = time.time() + wait_minutes * 60
        while self.state["machines"] and time.time() < until:
            for name in list(self.state["machines"]):
                answer = self.poll(name)
                if answer is not None and answer["alive"] == 0:
                    # A look packs only files older than 5 seconds, and the job writes its last
                    # checkpoints and results just before it ends: one more look brings them.
                    time.sleep(6)
                    answer = self.poll(name)
                # Stopped only when everything it saved is here (a machine that does not answer,
                # or whose download failed, is asked again until the time is up).
                if answer is not None and answer["alive"] == 0 and not answer.get("incomplete"):
                    self.note_progress(name, answer)
                    for group in self.state["groups"].values():
                        if group.get("machine") == name:
                            group["machine"] = None
                    self.colab.stop(name)
                    self.say(f"machine {name} stopped")
                    del self.state["machines"][name]
                    self.save()
                elif answer is None and self.colab.exists(name) is False:
                    self.release(name)  # gone already: nothing more can come from it
                    del self.state["machines"][name]
                    self.say(f"machine {name}: Colab says it is not there any more")
                    self.save()
            time.sleep(10)
        for name in list(self.state["machines"]):  # did not stop in time
            self.colab.stop(name)
            del self.state["machines"][name]
            self.say(f"machine {name} stopped (its job had not finished saving)")
        for group in self.state["groups"].values():
            group["machine"] = None
            group["step"] = group.get("saved_step", 0)
            group.pop("training_since", None)
            group.pop("parking", None)
        self.save()

    def look_at(self, name):
        """Ask one machine how far it is (and bring home what is new). Returns its answer when it
        can be given work, else None."""
        machine = self.state["machines"][name]
        on_its_way = name in self.fetching  # its last request has not come back: no second one
        fetched = self.fetched.pop(name, None)
        if fetched is not None:
            exists = fetched["exists"]
        else:
            exists = None if on_its_way else self.colab.exists(name)
        if exists is False:
            self.give_up(name, "Colab says it is not there any more")
            return None
        if on_its_way:
            answer = None
        elif not machine.get("started") and not self.start_job(name):
            # (Its job never started: without this it would cost units for ever, and when it is
            # the only machine no other would be asked for.)
            waited = (time.time() - machine.get("added", time.time())) / 60
            if waited > self.lost_minutes:
                self.give_up(name, f"its job could not be started in {waited:.0f} minutes")
            return None
        elif fetched is not None and fetched["polled"]:
            answer = fetched["answer"]
            self.apply_saved(answer)
        else:  # just started, or not asked beforehand
            answer = self.poll(name)
        if answer is None:
            silent = (time.time() - machine["last_answer"]) / 60
            fast = machine["card"] in FAST_CARDS
            if fast and silent > 0.6 * self.lost_minutes and not machine.get("repair_tried"):
                # The training job runs apart from the kernel that answers our requests, so a
                # restart of that kernel should leave the training alone.
                machine["repair_tried"] = True
                self.say(f"machine {name} ({machine['card']}): no answer for {silent:.0f} "
                         f"minutes; restarting its kernel before giving it up")
                self.colab.run("restart-kernel", "-s", name, machine=name, tries=1, timeout=120)
                self.save()
            if silent > self.lost_minutes * (2 if fast else 1):
                self.give_up(name, f"no answer for {silent:.0f} minutes")
            return None
        machine["last_answer"] = time.time()
        machine.pop("repair_tried", None)
        if machine.get("leaving"):
            # Stopping to give its groups to faster machines: once its job has ended (every
            # group saved), the last checkpoints come home and the machine is given back.
            self.note_progress(name, answer)
            if answer["alive"] == 0:
                time.sleep(6)
                answer = self.poll(name)
                if answer is None or answer.get("incomplete"):
                    return None  # its last checkpoints are not here yet: tried again next look
                self.note_progress(name, answer)
                self.release(name)
                self.colab.stop(name)
                del self.state["machines"][name]
                self.say(f"machine {name}: stopped and given back; its groups go on elsewhere")
                self.save()
            return None
        self.reconcile(name, answer)
        self.note_progress(name, answer)
        held = self.held_by(name)
        if answer["alive"] == 0 and held:
            self.say(f"machine {name}: its job is not running; starting it again")
            machine["started"] = False
            return None
        machine["waiting"] = len(answer["waiting"])
        # A group this program counts as the machine's, but which the machine neither has
        # waiting nor trains, and has not finished, twice in a row: it was taken back while this
        # program was cut off (e.g. stopped in the middle of a move). It goes back to the pool.
        # (A parked group is still on the machine: it is waiting to be taken off it.)
        listed = set(answer["waiting"]) | set(answer["taken"]) | set(answer.get("parked", []))
        for index in held:
            group = self.state["groups"][str(index)]
            progress = answer["progress"].get(str(index), {})
            if index in listed or progress.get("done") or self.group_done(index):
                group.pop("absent", None)
                continue
            group["absent"] = group.get("absent", 0) + 1
            if (group.get("parking") or {}).get("from") == name:
                continue  # a group being moved up from here: settle_parking frees it (and gives it on)
            if group["absent"] >= 2:
                self.say(f"group {index}: not on {name} any more (taken back before a restart?); "
                         f"it goes to a machine again from its checkpoint here")
                group["machine"] = None
                group["step"] = group.get("saved_step", 0)
                group.pop("absent", None)
        held = self.held_by(name)
        for index in answer["taken"]:  # its shape is compiled there now
            group = self.state["groups"].setdefault(str(index), {})
            group.setdefault("compiled_on", [])
            if name not in group["compiled_on"]:
                group["compiled_on"].append(name)
        done_here = sum(1 for i in held if self.group_done(i))
        self.say(f"machine {name} ({machine['card']}): {len(answer['waiting'])} waiting, "
                 f"{len(answer['taken'])} training, card {answer['card']}; {done_here} of its "
                 f"{len(held)} groups done")
        return answer

    def step(self):
        """One look at everything. Returns False when the stage is done (or must stop)."""
        if not self.units_ok():
            self.stop_all()
            return False
        if not self.open_groups():
            self.say("every run of the stage is here")
            if self.keep_machines:
                # Fast cards are hard to get again: the machines stay up (idle, their jobs
                # waiting for groups) so that the next stage can take them over at once.
                self.say(f"the {len(self.state['machines'])} machines are kept for the next stage "
                         f"(they cost units until a launcher takes them over or stops them)")
                self.save()
            else:
                self.stop_all(wait_minutes=1)
            return False
        if not self.free_groups() and not any(self.held_by(n) for n in self.state["machines"]):
            failed = [i for i in self.open_groups() if self.state["failures"].get(str(i), 0) >= 3]
            self.say(f"groups {failed} failed three times each and nothing else is left: "
                     f"stopping (see the workers' logs)")
            self.stop_all(wait_minutes=1)
            return False
        if self.looks % 5 == 0:  # (every few minutes: one request to Colab)
            self.find_lost_machines()
        self.looks += 1
        self.take_answers()  # machines granted since the last look (and lost ones found)
        self.fetch_all()  # every machine is asked at the same time
        answers = {}
        for name in list(self.state["machines"]):
            answer = self.look_at(name)
            if answer is not None:
                answers[name] = answer
        for name, answer in answers.items():  # groups whose runs are all here are let go
            for index in self.held_by(name):
                if self.group_done(index) and index not in answer["waiting"] + answer["taken"]:
                    self.state["groups"][str(index)]["machine"] = None
        # Moves up under way: a parked group whose checkpoint is home goes to its faster machine
        # (or becomes free, and the planner just below hands it out).
        self.settle_parking(answers)
        model = self.replan(answers)
        if answers and self.shared_cards:
            self.fill_free_workers(answers)
        # Groups in the middle of a round move to faster machines with free workers.
        self.move_up(answers)
        if model is not None or not self.state["machines"]:  # no machine at all: ask for any
            self.ask_for_machines(model)
        # Colab grants a free card in about 3 seconds: such a machine starts work in this look.
        if self.prober.pending:
            self.prober.wait(self.grant_wait)
        new = self.take_answers()
        if new:
            for name in new:
                answer = self.look_at(name)
                if answer is not None:
                    answers[name] = answer
            self.replan(answers)
        # A machine with nothing left to do is given back at once (unless a machine is stopping
        # to hand its groups over: they are coming).
        coming = any(m.get("leaving") for m in self.state["machines"].values())
        for name in list(answers):
            machine = self.state["machines"].get(name)
            # (A machine that a group is being parked for is not idle: the group is coming.)
            idle = (bool(machine) and not self.held_by(name) and not self.free_groups()
                    and not self.incoming(name))
            if machine:
                machine["idle_looks"] = machine.get("idle_looks", 0) + 1 if idle else 0
            if idle and machine["card"] in FAST_CARDS and machine["idle_looks"] < 5:
                continue  # a fast card waits a few looks: work may still come to it
            if machine and not coming and idle:
                self.colab.execute(name, with_values(SIGNAL, WHAT="close"))
                self.colab.stop(name)
                del self.state["machines"][name]
                self.say(f"machine {name}: nothing left to do; stopped")
        self.save()
        finished = self.finished_names()
        total = sum(len(g) for g in self.groups)
        here = sum(1 for g in self.groups for c in g if c["name"] in finished)
        cards = ", ".join(sorted(m["card"] for m in self.state["machines"].values()))
        estimate = (f"; the stage should end in about {self.estimate / 3600:.1f} h"
                    if self.estimate else "")
        self.say(f"stage {self.stage}: {here} of {total} runs here; {len(self.open_groups())} "
                 f"groups open; {len(self.state['machines'])} machines ({cards}){estimate}")
        return True

    def run(self):
        self.say(f"fleet for stage {self.stage}: {len(self.groups)} groups, up to {self.want} "
                 f"machines ({', '.join(self.cards)}), reserve {self.reserve:.0f} units; "
                 f"{len(self.state['machines'])} machines taken over from the state file")
        unknown = colab_gpu.unknown_machines(self.colab.tool)
        if unknown:
            self.say(f"WARNING: Colab has machines that no state file here knows: {unknown}. "
                     f"One may have been granted to a request that was cut off (it costs units "
                     f"and does nothing); see `colab sessions`.")
        try:
            while self.step():
                time.sleep(self.poll_seconds)
        finally:
            self.prober.close()  # a machine granted after the end is given back
            self.save()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a stage on several Colab machines.")
    parser.add_argument("stage")
    parser.add_argument("--machines", type=int, default=12,
                        help="at most this many machines (Colab's own limit may be lower)")
    parser.add_argument("--reserve", type=float, default=100.0,
                        help="compute units that are never used (default 100)")
    parser.add_argument("--workers", type=int, default=2, help="groups at once on a machine")
    parser.add_argument("--cards", default=",".join(colab_gpu.ORDER),
                        help="card types, fastest first")
    parser.add_argument("--lost-minutes", type=float, default=15.0)
    parser.add_argument("--poll-seconds", type=float, default=40.0)
    parser.add_argument("--prefix", default="air")
    parser.add_argument("--ask-gain-minutes", type=float, default=5.0,
                        help="ask for a card type when one more machine of it ends the stage "
                             "at least this many minutes earlier ...")
    parser.add_argument("--ask-gain-share", type=float, default=0.05,
                        help="... and by at least this share of the time left")
    parser.add_argument("--shared-cards", default="",
                        help="cards whose workers share the card through NVIDIA's MPS, with the "
                             "number of workers, e.g. G4=8,A100=3")
    parser.add_argument("--keep-machines", action="store_true",
                        help="at the end of the stage leave the machines running, for the next stage")
    parser.add_argument("--release", default="",
                        help="machines (names, with commas) that save their groups and are given "
                             "back at once; their groups go on on the other machines. For the end "
                             "of a stage, when faster machines have free workers.")
    parser.add_argument("--stop", action="store_true",
                        help="stop every machine of the fleet cleanly (they save first) and end")
    args = parser.parse_args(argv)
    from v3.orchestrate import stage_groups
    fleet = Fleet(args.stage, stage_groups(args.stage), Colab(), machines=args.machines,
                  reserve=args.reserve, workers=args.workers, cards=args.cards.split(","),
                  lost_minutes=args.lost_minutes, poll_seconds=args.poll_seconds,
                  prefix=args.prefix, ask_gain_minutes=args.ask_gain_minutes,
                  ask_gain_share=args.ask_gain_share,
                  shared_cards={part.split("=")[0]: int(part.split("=")[1])
                                for part in args.shared_cards.split(",") if part},
                  keep_machines=args.keep_machines)
    if args.stop:
        fleet.say("asked to stop every machine")
        fleet.stop_all()
        return
    for name in [n for n in args.release.split(",") if n]:
        machine = fleet.state["machines"].get(name)
        if machine is None or machine.get("leaving"):
            continue
        # The same clean stop the launcher uses itself for a slow machine: the workers save,
        # the checkpoints come home, the machine is given back, its groups go on elsewhere.
        worked, printed = fleet.colab.execute(name, with_values(SIGNAL, WHAT="stop"))
        if worked and "SIGNALLED" in printed:
            machine["leaving"] = True
            fleet.say(f"machine {name} ({machine['card']}): asked to save and stop; its groups go "
                      f"to machines with free workers")
        else:
            fleet.say(f"machine {name}: the request to stop did not get through; it trains on")
    fleet.save()
    try:
        fleet.run()
    except KeyboardInterrupt:
        fleet.say("stopped by hand; the machines go on. Start this program again to go on "
                  "watching them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
