"""
v3/orchestrate.py - run a stage of the third experiment on Kaggle's free graphics cards, from the
owner's computer (v3/PLAN.md, Part B.3).

    .venv/bin/python -m v3.orchestrate practice     # the practice runs, then the rate choice
    .venv/bin/python -m v3.orchestrate counted      # the counted runs (needs v3/settings.json)
    .venv/bin/python -m v3.orchestrate rate_check_third   # after ALL counted runs have ended:
    .venv/bin/python -m v3.orchestrate rate_check_tenth   # the best seeds at 1/3 and 1/10 rate
    .venv/bin/python -m v3.orchestrate status       # what has been done so far

It can be stopped (Ctrl+C) and started again with the same command at any time: everything it
knows is kept in logs/v3/state.json, and nothing is lost or done twice.

What it does
------------
1. The stage's groups (v3/grid.py) are cut into sessions. A session gets as many groups as it
   can probably finish in its time (from --speed, member-steps per second on one card, and two
   cards per session), but at least two per card, so that no card stands idle.
2. Sessions are sent to Kaggle, named air-v3-<stage>-<number>, as many at once as Kaggle
   accepts. When Kaggle refuses one, it waits and tries again later.
3. Before a session is sent, `kaggle quota` is read. A session never asks for more time than the
   weekly hours left, minus half an hour, minus the time asked by the sessions still running.
   When there is not enough left for a useful session, it stops and says so.
4. Running sessions are watched. When one ends, its output is downloaded into
   runs_v3/_kaggle/<session>/ (kept, with the checkpoints), and the summaries and model.pt of its
   finished runs are copied into runs_v3/<stage>/<name>/.
5. Groups that are not finished are sent again in a later session, with the earlier session that
   holds their checkpoint attached as a "kernel source", so they go on from the checkpoint.
   A group that fails three times without any progress is given up and reported (and tried
   again when the program is started again).
6. `practice` ends by running v3/choose_rates.py, which writes v3/settings.json.

A plain log is written to logs/v3/orchestrate.log. The Kaggle user name is read from
~/.kaggle/username_for_scripts and never written to the log.
"""

import argparse
import base64
import fcntl
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from v3 import grid

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = Path.home() / ".local" / "bin" / "kaggle"
USER_FILE = Path.home() / ".kaggle" / "username_for_scripts"

DEFAULT_HOURS = {"counted": 11.0, "practice": 3.0, "test": 0.1, "rate_check_third": 11.0,
                 "rate_check_tenth": 11.0, "bridge": 3.0}
STAGES = ["practice", "counted", "rate_check_third", "rate_check_tenth", "bridge", "test"]
QUOTA_MARGIN_HOURS = 0.5  # never plan to use the last half hour of the weekly quota
MIN_SESSION_HOURS = 0.5  # a session shorter than this is not worth starting (not for "test")
TEST_MIN_HOURS = 0.08  # the same for the short "test" sessions (shorter ones train nothing)
JOB_RESERVE_MINUTES = 12  # the job inside a session stops this long before the time asked
CARDS = 2  # graphics cards in one session
DEFAULT_SPEED = 700.0  # member-steps per second on one T4: measured for 24 members on 23
# November (logs/v3/speed_report.md) from 478 (64 columns) to 1,244 (8 columns); about 775 on
# average over the column counts, so 700 is a little cautious
COUNTED_STEPS_GUESS = 160000  # expected steps of a counted run (they end at 40k ... 640k)
SPEED_FILL = 0.8  # plan sessions to be full to this share only
MAX_FAILURES = 3  # a group failing this often without progress is given up
MAX_SOURCES = 10  # at most this many earlier sessions attached to one session
MAX_DOWNLOAD_TRIES = 10  # downloads of an ended session tried before it is taken as it is
GROUPS_PER_CARD = 2  # a session gets at least this many groups per card (when there are so
# many waiting), so that a card whose group finishes early takes another instead of standing idle
POLL_SECONDS = 60
# Kaggle's queue took about 45 minutes on 22 November. A session still waiting (or that Kaggle
# cannot report on) this long after it was sent is reported once in the log.
LONG_WAIT_HOURS = 6

PACK_FOLDERS = ("v2", "v3")

HEAD = '''# Made by v3/orchestrate.py. Unpack the code, then run this session's groups.
import base64, io, json, os, sys, tarfile, time
STARTED = time.time()
CODE = os.environ.get("AIR_CODE_DIR", "/tmp/air_code")
os.makedirs(CODE, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PACKED))) as packed:
    if hasattr(tarfile, "data_filter"):
        packed.extractall(CODE, filter="data")
    else:
        packed.extractall(CODE)
sys.path.insert(0, CODE)
from v3.session_job import run_session, card_count
if REQUIRE_CARDS and not card_count() and not os.environ.get("AIR_ALLOW_NO_CARD"):
    raise SystemExit("No graphics card in this session; stopping so no time is wasted.")
run_session(json.loads(GROUPS), STAGE, BUDGET_MINUTES,
            input_root=os.environ.get("AIR_KAGGLE_INPUT", "/kaggle/input"),
            working=os.environ.get("AIR_KAGGLE_WORKING", "/kaggle/working"), started=STARTED,
            end_reserve=END_RESERVE, min_start=MIN_START)
'''


# ------------------------------------------------------------------------------------------------
# The groups of a stage
# ------------------------------------------------------------------------------------------------


def counted_summaries(runs_root):
    """{name: summary} of the counted runs downloaded so far (runs_v3/counted)."""
    found = {}
    for path in (Path(runs_root) / "counted").glob("*/summary.json"):
        found[path.parent.name] = json.loads(path.read_text())
    return found


def stage_groups(stage, runs_root=None):
    """The groups of a stage, in the order they are to be run. The rate-check stages need the
    counted results (in runs_root/counted) and refuse to start before every counted run has
    ended."""
    if stage in grid.RATE_CHECKS:
        if not grid.SETTINGS_FILE.exists():
            raise SystemExit("v3/settings.json is missing: run the practice stage first.")
        try:
            configs = grid.rate_check(grid.read_settings(grid.SETTINGS_FILE),
                                      counted_summaries(runs_root or ROOT / "runs_v3"),
                                      grid.RATE_CHECKS[stage])
        except ValueError as error:
            raise SystemExit(f"{stage} cannot start: {error}. Finish the counted stage first.")
        return grid.groups(configs)
    if stage == "practice":
        return grid.groups(grid.practice())
    if stage == "counted":
        if not grid.SETTINGS_FILE.exists():
            raise SystemExit("v3/settings.json is missing: run the practice stage first "
                             "(it ends by choosing the rates).")
        return grid.groups(grid.counted(grid.read_settings(grid.SETTINGS_FILE)))
    if stage == "bridge":  # the bridging runs (v3/grid.py, bridge)
        if not grid.SETTINGS_FILE.exists():
            raise SystemExit("v3/settings.json is missing: run the practice stage first.")
        return grid.groups(grid.bridge(grid.read_settings(grid.SETTINGS_FILE)))
    if stage == "test":
        return grid.groups(grid.test_stage())
    raise SystemExit(f"unknown stage {stage!r}")


def group_member_steps(configs):
    """The expected member-steps of a group (steps of one member, added up)."""
    each = [COUNTED_STEPS_GUESS if c.get("length_rule") == "doubling" and c["steps"] >= 640000
            else c["steps"] for c in configs]
    return sum(each)


def group_signature(configs):
    return [c["name"] for c in configs]


# ------------------------------------------------------------------------------------------------
# The Kaggle tool
# ------------------------------------------------------------------------------------------------


class Kaggle:
    """The few Kaggle commands used here. `tool` is the path of the command-line tool (a pretend
    tool in the tests)."""

    def __init__(self, tool=KAGGLE, user=None):
        self.tool = str(tool)
        self.user = user or USER_FILE.read_text().strip()

    def run(self, *arguments, timeout=600):
        try:
            done = subprocess.run([self.tool, *arguments], capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return 1, "the Kaggle tool did not answer in time"
        text = (done.stdout + done.stderr).strip()
        return done.returncode, self.hide(text)

    def hide(self, text):
        """The text with the user name replaced, so it never reaches a log."""
        return text.replace(self.user, "<user>") if self.user else text

    def ref(self, name):
        return f"{self.user}/{name}"

    def quota_hours(self):
        """Remaining weekly hours of graphics cards, or None when it cannot be read."""
        code, text = self.run("quota", timeout=120)
        for line in text.splitlines():
            parts = line.split()
            if parts and parts[0] == "GPU" and len(parts) >= 3:
                try:
                    return float(parts[2].rstrip("h"))
                except ValueError:
                    return None
        return None

    def push(self, folder, seconds):
        """Send a session. Returns (accepted, text)."""
        code, text = self.run("kernels", "push", "-p", str(folder), "--accelerator",
                              "NvidiaTeslaT4", "-t", str(int(seconds)), timeout=600)
        accepted = code == 0 and "successfully pushed" in text.lower()
        return accepted, text

    def status(self, name):
        """"running", "queued", "complete", "error", "cancelled" or "unknown"."""
        code, text = self.run("kernels", "status", self.ref(name), timeout=120)
        # Only the status itself is read ('... has status "KernelWorkerStatus.RUNNING"'). A
        # message such as "404 Client Error" or "429 Too Many Requests" is NOT the session
        # having ended with an error: the session may still be running, so it is "unknown".
        found = re.search(r'has status "([^"]*)"', text)
        if code != 0 or found is None:
            return "unknown"
        upper = found.group(1).upper()
        for word, meaning in (("COMPLETE", "complete"), ("ERROR", "error"),
                              ("CANCEL", "cancelled"), ("RUNNING", "running"),
                              ("QUEUED", "queued")):
            if word in upper:
                return meaning
        return "unknown"

    def output(self, name, folder):
        code, text = self.run("kernels", "output", self.ref(name), "-p", str(folder), "-o",
                              timeout=3600)
        return code == 0, text


# ------------------------------------------------------------------------------------------------
# The orchestrator
# ------------------------------------------------------------------------------------------------


def packed_code():
    """The folders v2 and v3 (programs and settings, not tests or reports), as one text."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for folder in PACK_FOLDERS:
            for path in sorted((ROOT / folder).rglob("*")):
                relative = path.relative_to(ROOT)
                skip = {"tests", "report", "__pycache__"} & set(relative.parts)
                if path.is_file() and path.suffix in (".py", ".json") and not skip:
                    tar.add(path, arcname=str(relative))
    return base64.b64encode(buffer.getvalue()).decode()


class Orchestrator:
    def __init__(self, stage, kaggle, runs_root, logs_dir, hours, speed, poll_seconds,
                 max_sessions=0, quiet=False, sleep=time.sleep, stop_after_sessions=0):
        self.stage = stage
        self.kaggle = kaggle
        self.runs_root = Path(runs_root)
        self.logs_dir = Path(logs_dir)
        self.hours = hours
        self.speed = speed
        self.poll_seconds = poll_seconds
        self.max_sessions = max_sessions
        self.quiet = quiet
        self.sleep = sleep
        self.stop_after_sessions = stop_after_sessions  # for tests: send at most this many
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.logs_dir / "state.json"
        self.log_path = self.logs_dir / "orchestrate.log"
        self.state = self.load_state()

    # --- state and log -------------------------------------------------------------------------

    def load_state(self):
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"sessions": {}, "stages": {}}

    def save_state(self):
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=1))
        os.replace(temporary, self.state_path)

    def log(self, text):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {self.kaggle.hide(text)}"
        with open(self.log_path, "a") as file:
            file.write(line + "\n")
        if not self.quiet:
            print(line, flush=True)

    def stage_state(self, groups):
        """The bookkeeping of this stage's groups, made on first use and checked after."""
        stages = self.state["stages"]
        if self.stage not in stages:
            stages[self.stage] = {"next_number": 1, "groups": [
                {"names": group_signature(g), "source": None, "failures": 0, "given_up": False}
                for g in groups]}
            self.save_state()
        known = stages[self.stage]
        # Groups given up in an earlier start get one more chance when the owner starts the
        # program again (after reading why they failed), so nothing has to be edited by hand.
        again = [i for i, g in enumerate(known["groups"]) if g["given_up"]]
        if again:
            for i in again:
                known["groups"][i].update(given_up=False, failures=0)
            self.save_state()
            self.log(f"groups given up before ({', '.join(map(str, again))}) are tried again")
        if [g["names"] for g in known["groups"]] != [group_signature(g) for g in groups]:
            raise SystemExit(f"The groups of stage {self.stage} are not those in "
                             f"{self.state_path} (was v3/settings.json changed?). Stopping.")
        return known

    # --- what is where -------------------------------------------------------------------------

    def finished_names(self):
        stage_dir = self.runs_root / self.stage
        if not stage_dir.exists():
            return set()
        return {p.parent.name for p in stage_dir.glob("*/summary.json")}

    def running_sessions(self):
        return {name: s for name, s in self.state["sessions"].items()
                if s["stage"] == self.stage and s["state"] in ("pushing", "running")}

    def sessions_sent(self):
        return sum(1 for s in self.state["sessions"].values() if s["stage"] == self.stage
                   and s["state"] != "not sent")

    def all_running(self):
        return {name: s for name, s in self.state["sessions"].items()
                if s["state"] in ("pushing", "running")}

    def group_status(self, groups, known):
        """For every group: "done", "running", "given up" or "waiting"."""
        done_names = self.finished_names()
        busy = {index for s in self.running_sessions().values() for index in s["groups"]}
        result = []
        for index, configs in enumerate(groups):
            if all(c["name"] in done_names for c in configs):
                result.append("done")
            elif index in busy:
                result.append("running")
            elif known["groups"][index]["given_up"]:
                result.append("given up")
            else:
                result.append("waiting")
        return result

    # --- watching and downloading --------------------------------------------------------------

    def refresh(self, groups, known):
        """Look at every running session of this stage; download those that have ended."""
        for name, session in list(self.running_sessions().items()):
            status = self.kaggle.status(name)
            if session["state"] == "pushing":
                # The program stopped while sending this session: did Kaggle take it?
                if status == "unknown":
                    self.log(f"{name}: was never accepted; its groups wait for another session")
                    session["state"] = "not sent"
                else:
                    session["state"] = "running"
                self.save_state()
                continue
            if status in ("queued", "unknown") and not session.get("warned_waiting") and \
                    time.time() - session["pushed"] > LONG_WAIT_HOURS * 3600:
                # A session that never starts would hold its groups (and its hours of quota)
                # for ever. It is NOT sent again by itself: should it start after all, the same
                # groups would be trained twice. The owner is told what to do instead.
                session["warned_waiting"] = True
                self.save_state()
                self.log(f"{name}: still {status} {LONG_WAIT_HOURS} hours after it was sent. "
                         f"If it is stuck, cancel it on kaggle.com; this program then sends "
                         f"its groups again in a new session.")
            if status in ("running", "queued", "unknown"):
                continue
            self.collect(name, session, status, groups, known)

    def collect(self, name, session, status, groups, known):
        """Download a session that has ended, take its results, and note its checkpoints."""
        target = self.runs_root / "_kaggle" / name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        ok, text = self.kaggle.output(name, target)
        if not ok:
            # Tried again at the next look. After MAX_DOWNLOAD_TRIES the session is taken as
            # it is (perhaps it left no output), so its groups do not wait for ever.
            session["download_failures"] = session.get("download_failures", 0) + 1
            self.save_state()
            self.log(f"{name}: ended ({status}) but the download failed "
                     f"({session['download_failures']} of {MAX_DOWNLOAD_TRIES} tries): "
                     f"{text.splitlines()[-1] if text else ''}")
            if session["download_failures"] < MAX_DOWNLOAD_TRIES:
                return
            status = "no output"
        out = next((p.parent for p in target.rglob("session.json")), None)
        report = {}
        if out is not None and (out / "status.json").exists():
            report = json.loads((out / "status.json").read_text())
        done_before = self.finished_names()
        copied = 0
        if out is not None and (out / self.stage).exists():
            for summary in sorted((out / self.stage).glob("*/summary.json")):
                destination = self.runs_root / self.stage / summary.parent.name
                if (destination / "summary.json").exists():
                    continue
                if not (summary.parent / "model.pt").exists():
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary.parent / "model.pt", destination / "model.pt")
                shutil.copy2(summary, destination / "summary.json")  # last: the run is complete
                copied += 1
        failed = {f["group"] for f in report.get("failed_groups", [])}
        # What the session says about each unfinished group: was it started, and how far did
        # it get (the step)?
        unfinished = {g["group"]: g for g in report.get("unfinished_groups", [])}
        done_names = self.finished_names()
        session_broke = status != "complete" or not report
        # Progress of each group in THIS session: new runs finished, or a step further than
        # before. (A checkpoint alone is no sign: the session copies the earlier one in.)
        progress = {}
        for local, index in enumerate(session["groups"]):
            configs = groups[index]
            entry = known["groups"][index]
            newly_done = any(c["name"] in done_names and c["name"] not in done_before
                             for c in configs)
            step = (unfinished.get(local) or {}).get("step")
            progress[local] = newly_done or (step is not None and step > entry.get("step", 0))
            if step is not None:
                entry["step"] = max(step, entry.get("step", 0))
        nothing_moved = not any(progress.values())
        for local, index in enumerate(session["groups"]):
            configs = groups[index]
            entry = known["groups"][index]
            if all(c["name"] in done_names for c in configs):
                continue
            checkpoint = None
            if out is not None:
                from_names = self.checkpoint_folder(configs)
                path = out / "_groups" / from_names / "checkpoint.pt"
                checkpoint = path if path.exists() else None
            if checkpoint is not None:
                entry["source"] = name  # the next session goes on from this checkpoint
            # A failure: the group's training raised an error; or it got nowhere in a session
            # that broke down, or in a session in which nothing at all moved (for example its
            # workers died at once). A group that simply was not reached (or was taken just
            # before the end and stopped at once) in a session where others made progress is
            # no failure. Without this a group that never moves would be sent again and again
            # until the quota is gone.
            if progress[local]:
                entry["failures"] = 0  # "three times without any progress" counts again
            elif local in failed or session_broke or nothing_moved:
                entry["failures"] += 1
                if entry["failures"] >= MAX_FAILURES:
                    entry["given_up"] = True
                    self.log(f"group {index} failed {entry['failures']} times without any "
                             f"progress; given up")
            if local in failed and progress[local]:
                entry["failures"] += 1  # an error counts even after some progress
                if entry["failures"] >= MAX_FAILURES:
                    entry["given_up"] = True
                    self.log(f"group {index} failed {entry['failures']} times; given up")
        session["state"] = "finished" if status == "complete" and report else "failed"
        session["ended"] = time.time()
        session["report"] = {"minutes": report.get("minutes"), "speed": report.get("speed"),
                             "finished": len(report.get("finished", [])),
                             "failed_groups": [f["error"].strip().splitlines()[-1]
                                               for f in report.get("failed_groups", [])]}
        self.save_state()
        speeds = ", ".join(f"{d} {v['per_second']:,.0f}" for d, v in
                           sorted((report.get("speed") or {}).items()))
        self.log(f"{name}: {status}; {copied} new runs downloaded; "
                 f"{report.get('minutes', 0) or 0:.0f} minutes; member-steps per second: "
                 f"{speeds or 'none measured'}")
        for error in session["report"]["failed_groups"]:
            self.log(f"{name}: a group failed: {error}")

    @staticmethod
    def checkpoint_folder(configs):
        """The folder name the group trainer gives the group's checkpoint (v2/batch.py
        group_name). Computed here without torch, by the same rule."""
        import hashlib
        first = configs[0]
        names = "\n".join(sorted(config["name"] for config in configs))
        code = hashlib.sha256(names.encode()).hexdigest()
        rates = {config["lr"] for config in configs}
        rate = first["lr"] if len(rates) == 1 else "mixed"
        table = first.get("table", "trained")
        return (f"k{first['kinds']}_b{first['body']}_{table}_lr{rate}"
                f"_steps{first['steps']}_{len(configs)}members_{code[:10]}")

    # --- sending -------------------------------------------------------------------------------

    def hours_available(self):
        """Hours a new session may ask for: the quota left, minus the margin, minus the hours
        asked by sessions still running. None when the quota cannot be read."""
        remaining = self.kaggle.quota_hours()
        if remaining is None:
            return None, None
        reserved = sum(s["hours"] for s in self.all_running().values())
        return remaining - QUOTA_MARGIN_HOURS - reserved, remaining

    def choose_groups(self, waiting, groups, known, hours):
        """The waiting groups for one session: in order, as many as two cards can probably
        finish in the session's time (but at least GROUPS_PER_CARD per card), with at most
        MAX_SOURCES earlier sessions attached."""
        seconds = max(60.0, hours * 3600 - JOB_RESERVE_MINUTES * 60) * SPEED_FILL
        per_card = [0.0] * CARDS
        chosen, sources = [], set()
        for index in waiting:
            source = known["groups"][index]["source"]
            if source and source not in sources and len(sources) >= MAX_SOURCES:
                continue
            need = group_member_steps(groups[index]) / self.speed
            card = per_card.index(min(per_card))
            if chosen and per_card[card] + need > seconds:
                if len(chosen) >= CARDS * GROUPS_PER_CARD:
                    break
            per_card[card] += need
            chosen.append(index)
            if source:
                sources.add(source)
        return chosen, sorted(sources)

    def write_session(self, name, indexes, groups, sources, hours):
        folder = self.logs_dir / "sessions" / name
        folder.mkdir(parents=True, exist_ok=True)
        budget = max(1.0, hours * 60 - JOB_RESERVE_MINUTES)
        if self.stage == "test":
            budget = max(1.0, hours * 60 - 2.5)  # short test sessions
        session_groups = [groups[i] for i in indexes]
        script = (f'PACKED = "{packed_code()}"\n'
                  f"GROUPS = {json.dumps(json.dumps(session_groups))}\n"
                  f"STAGE = {self.stage!r}\nBUDGET_MINUTES = {budget!r}\n"
                  f"REQUIRE_CARDS = True\n"
                  # Seconds kept free at the end, and the least time to start a group with:
                  # smaller for the short test sessions.
                  f"END_RESERVE = {20 if self.stage == 'test' else 180}\n"
                  f"MIN_START = {10 if self.stage == 'test' else 120}\n" + HEAD)
        (folder / "script.py").write_text(script)
        (folder / "kernel-metadata.json").write_text(json.dumps({
            "id": self.kaggle.ref(name), "title": name.replace("-", " "),
            "code_file": "script.py", "language": "python", "kernel_type": "script",
            "is_private": "true", "enable_gpu": "true", "enable_internet": "false",
            "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [self.kaggle.ref(s) for s in sources]}))
        return folder

    def send_more(self, groups, known):
        """Send sessions for waiting groups until Kaggle refuses or nothing is left. Returns
        "sent", "refused", "quota", "nothing" or "limit"."""
        result = "nothing"
        while True:
            statuses = self.group_status(groups, known)
            waiting = [i for i, s in enumerate(statuses) if s == "waiting"]
            if not waiting:
                return result
            if self.max_sessions and len(self.all_running()) >= self.max_sessions:
                return "limit"
            if self.stop_after_sessions and self.sessions_sent() >= self.stop_after_sessions:
                return "limit"
            hours, remaining = self.hours_available()
            if hours is None:
                self.log("could not read the quota (kaggle quota); will try again later")
                return "refused"
            smallest = TEST_MIN_HOURS if self.stage == "test" else MIN_SESSION_HOURS
            hours = min(self.hours, hours)
            if hours < smallest:
                self.log(f"weekly quota: {remaining:.2f} hours left; not enough for another "
                         f"session (half an hour is kept free, and running sessions may need "
                         f"their time)")
                return "quota"
            indexes, sources = self.choose_groups(waiting, groups, known, hours)
            number = known["next_number"]
            name = f"air-v3-{self.stage}-{number}"
            folder = self.write_session(name, indexes, groups, sources, hours)
            # Noted before sending: if this program stops during the push, the next start asks
            # Kaggle whether the session exists.
            self.state["sessions"][name] = {"stage": self.stage, "number": number,
                                            "groups": indexes, "sources": sources,
                                            "hours": hours, "state": "pushing",
                                            "pushed": time.time()}
            known["next_number"] = number + 1
            self.save_state()
            accepted, text = self.kaggle.push(folder, hours * 3600)
            if not accepted and self.kaggle.status(name) != "unknown":
                # The tool gave no clear answer (for example it did not answer in time), but
                # Kaggle has the session: it runs and uses quota, so it is watched like any other.
                self.log(f"{name}: the Kaggle tool did not confirm the push, but the session "
                         f"exists; it is watched as a running session")
                accepted = True
            if not accepted:
                # Its number is NOT used again: should Kaggle have taken it after all, a later
                # push under the same name would replace that session and its output.
                del self.state["sessions"][name]
                self.save_state()
                last = text.splitlines()[-1] if text else "no answer"
                self.log(f"Kaggle did not take {name} now ({last}); will try again later")
                return "refused"
            self.state["sessions"][name]["state"] = "running"
            self.save_state()
            self.log(f"{name}: sent with {len(indexes)} groups ({hours:.2f} hours asked; "
                     f"{remaining:.2f} hours of quota left before it); earlier sessions "
                     f"attached: {', '.join(sources) or 'none'}")
            result = "sent"

    # --- the main loop -------------------------------------------------------------------------

    def run(self):
        # Only one orchestrator at a time may use logs/v3: two would both send sessions for
        # the same waiting groups. The lock is let go when this program ends, however it ends.
        self.lock = open(self.logs_dir / "orchestrate.lock", "w")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"Another orchestrator is already running with {self.logs_dir}; "
                             f"stop it first (or just watch it with the status command).")
        try:
            return self.run_locked()
        finally:
            self.lock.close()  # lets go of the lock

    def run_locked(self):
        smallest = TEST_MIN_HOURS if self.stage == "test" else MIN_SESSION_HOURS
        if self.hours < smallest:
            raise SystemExit(f"--hours {self.hours} is less than the shortest session worth "
                             f"starting ({smallest} hours)")
        groups = stage_groups(self.stage, self.runs_root)
        known = self.stage_state(groups)
        self.log(f"stage {self.stage}: {sum(len(g) for g in groups)} runs in {len(groups)} "
                 f"groups; sessions of {self.hours} hours; speed {self.speed:,.0f} member-steps "
                 f"per second per card")
        refused_alone = 0
        while True:
            self.refresh(groups, known)
            statuses = self.group_status(groups, known)
            if all(s in ("done", "given up") for s in statuses) and not self.running_sessions():
                given_up = statuses.count("given up")
                self.log(f"stage {self.stage}: finished; {statuses.count('done')} groups done, "
                         f"{given_up} given up")
                return "done" if not given_up else "given up"
            sent = self.send_more(groups, known)
            if sent == "limit" and self.stop_after_sessions and not self.running_sessions() \
                    and self.sessions_sent() >= self.stop_after_sessions:
                self.log(f"stopping: {self.stop_after_sessions} sessions were sent "
                         f"(--stop-after-sessions)")
                return "stopped"
            if sent == "quota" and not self.running_sessions():
                self.log("stopping: the weekly quota is used up. Start the same command again "
                         "after the quota is renewed; nothing is lost.")
                return "quota"
            if sent == "refused" and not self.all_running():
                refused_alone += 1
                if refused_alone >= 5:
                    self.log("stopping: Kaggle refused five times with no session running; "
                             "see the messages above.")
                    return "refused"
            else:
                refused_alone = 0
            self.sleep(self.poll_seconds)


def show_status(kaggle, runs_root, logs_dir):
    """Print what has been done so far, for every stage that has been started."""
    state_path = Path(logs_dir) / "state.json"
    if not state_path.exists():
        print("nothing started yet")
        return
    state = json.loads(state_path.read_text())
    remaining = kaggle.quota_hours()
    print(f"weekly quota left: {remaining:.2f} hours" if remaining is not None
          else "weekly quota: could not be read")
    for stage, known in state["stages"].items():
        groups = len(known["groups"])
        runs = sum(len(g["names"]) for g in known["groups"])
        stage_dir = Path(runs_root) / stage
        done = len(list(stage_dir.glob("*/summary.json"))) if stage_dir.exists() else 0
        given_up = sum(g["given_up"] for g in known["groups"])
        print(f"{stage}: {done} of {runs} runs downloaded ({groups} groups; {given_up} given up)")
        for name, session in state["sessions"].items():
            if session["stage"] != stage:
                continue
            line = f"  {name}: {session['state']}, {len(session['groups'])} groups, " \
                   f"{session['hours']:.2f} hours asked"
            if session["state"] in ("running", "pushing"):
                line += f", started {(time.time() - session['pushed']) / 60:.0f} minutes ago"
            report = session.get("report") or {}
            if report.get("speed"):
                line += ", member-steps per second: " + ", ".join(
                    f"{d} {v['per_second']:,.0f}" for d, v in sorted(report["speed"].items()))
            print(line)


def main(argv=None, kaggle=None, sleep=time.sleep):
    parser = argparse.ArgumentParser(description="Run a stage of the third experiment on Kaggle.")
    parser.add_argument("what", choices=STAGES + ["status"])
    parser.add_argument("--hours", type=float, default=None,
                        help="session time asked (default 3 for practice, 11 for counted)")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                        help="member-steps per second on one card, to plan the sessions")
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    parser.add_argument("--max-sessions", type=int, default=0,
                        help="at most this many sessions at once (0: as many as Kaggle takes)")
    parser.add_argument("--runs-root", default=str(ROOT / "runs_v3"))
    parser.add_argument("--logs-dir", default=str(ROOT / "logs" / "v3"))
    parser.add_argument("--kaggle", default=str(KAGGLE), help="the Kaggle command-line tool")
    parser.add_argument("--stop-after-sessions", type=int, default=0,
                        help="for tests: stop when this many sessions have been sent and ended")
    args = parser.parse_args(argv)
    kaggle = kaggle or Kaggle(args.kaggle)
    if args.what == "status":
        show_status(kaggle, args.runs_root, args.logs_dir)
        return None
    hours = args.hours if args.hours is not None else DEFAULT_HOURS[args.what]
    orchestrator = Orchestrator(args.what, kaggle, args.runs_root, args.logs_dir, hours,
                                args.speed, args.poll_seconds, args.max_sessions, sleep=sleep,
                                stop_after_sessions=args.stop_after_sessions)
    result = orchestrator.run()
    if args.what == "practice" and result == "done":
        from v3 import choose_rates
        orchestrator.log("practice finished: choosing the learning rates")
        choose_rates.main(["--practice-dir", str(Path(args.runs_root) / "practice")])
    return result


if __name__ == "__main__":
    main()
