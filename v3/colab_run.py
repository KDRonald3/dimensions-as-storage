"""Run a stage of the third experiment on one Colab machine and bring the results home.

    .venv/bin/python -m v3.colab_run practice --session air1 --gpu A100
    .venv/bin/python -m v3.colab_run counted --session air2 --gpu A100 --share 1/3

It uses Google's Colab command-line tool (~/.local/bin/colab, signed in). What it does:

1. Takes the groups of the stage (the same list and order as v3/orchestrate.py), leaves out
   the groups whose runs are already in runs_v3/<stage>/, and, with --share K/N, keeps only
   every N-th group starting at K, so that N machines can each do their own part.
2. Makes sure the Colab machine exists (creates it if not), sends the programs, and starts the
   training there as a job of its own, so it keeps going whatever happens to this computer.
   On the machine, several groups train at the same time on the one large card.
3. Every few minutes it asks the machine how far it is, downloads the runs that have ended
   since last time into runs_v3/<stage>/, and writes a line to its log.
4. When the job on the machine has ended and nothing is missing, it stops the machine, so no
   compute units are used any more.

It can be stopped and started again: it finds the job still running, or sends what is missing.
"""

import argparse
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from v2.kaggle_push import packed_v2
from v3.orchestrate import stage_groups

ROOT = Path(__file__).resolve().parent.parent
COLAB = Path.home() / ".local" / "bin" / "colab"

# Runs on the Colab machine: unpack the programs and start the training as its own job.
START = r'''
import base64, io, json, os, subprocess, sys, tarfile
os.makedirs("/content/air_code", exist_ok=True)
os.makedirs("/content/air_run", exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PACKED))) as packed:
    packed.extractall("/content/air_code")
open("/content/air_run/groups.json", "w").write(GROUPS)
open("/content/air_run/start.py", "w").write("""
import json, sys
sys.path.insert(0, "/content/air_code")
from v3.session_job import run_session
run_session(json.load(open("/content/air_run/groups.json")), %r, %r,
            input_root="/content/air_in", working="/content/air_run")
""" % (STAGE, BUDGET_MINUTES))
alive = subprocess.run("pgrep -f '[a]ir_run/start.py' | wc -l", shell=True, capture_output=True, text=True).stdout.strip()
if alive != "0":
    print("ALREADY RUNNING")
else:
    for old in ("/content/air_run/out/status.json",):
        if os.path.exists(old):
            os.remove(old)
    environment = dict(os.environ, AIR_WORKERS_PER_CARD=str(WORKERS), AIR_THREADS_PER_WORKER="1",
                       PYTHONPATH="/content/air_code")
    subprocess.Popen([sys.executable, "/content/air_run/start.py"], cwd="/content/air_code",
                     env=environment, stdout=open("/content/air_run/job.log", "a"),
                     stderr=subprocess.STDOUT, start_new_session=True)
    print("STARTED")
'''

# Runs on the Colab machine: say how far the job is and pack the runs that ended since last time.
POLL = r'''
import glob, os, subprocess, tarfile, time
out = "/content/air_run/out"
marker = "/content/air_run/synced.txt"
last = float(open(marker).read()) if os.path.exists(marker) else 0.0
now = time.time() - 5
new = [p for p in glob.glob(f"{out}/{STAGE}/*/*")
       if last <= os.path.getmtime(p) < now and os.path.exists(os.path.join(os.path.dirname(p), "summary.json"))]
with tarfile.open("/content/air_run/new.tar", "w") as tar:
    for p in new:
        tar.add(p, arcname=os.path.relpath(p, out))
open(marker + ".next", "w").write(repr(now))
print("NEWFILES", len(new))
print("FINISHED", len(glob.glob(f"{out}/{STAGE}/*/summary.json")))
print("ALIVE", subprocess.run("pgrep -f '[a]ir_run/start.py' | wc -l", shell=True, capture_output=True, text=True).stdout.strip())
print("ENDED", int(os.path.exists(f"{out}/status.json")))
card = subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip()
print("CARD", card)
for log in sorted(glob.glob(f"{out}/logs/worker*.log")):
    lines = open(log).read().strip().splitlines()
    print("WORKER", os.path.basename(log)[:-4], "|", lines[-1][:140] if lines else "")
'''

COMMIT = 'import os\nos.replace("/content/air_run/synced.txt.next", "/content/air_run/synced.txt")\nprint("COMMITTED")\n'


def colab(*arguments, text=None, timeout=900):
    done = subprocess.run([str(COLAB), *arguments], input=text, capture_output=True, text=True,
                          timeout=timeout)
    return done.returncode, done.stdout + done.stderr


def execute(session, code, timeout=900):
    """Run Python on the Colab machine and return what it printed."""
    # --timeout is the longest silence the tool allows. With 0 (no limit) a finished request
    # sometimes never returned, so a limit of two minutes is used; our requests take seconds.
    return colab("exec", "-s", session, "--timeout", "120", text=code, timeout=timeout)[1]


def finished_names(stage):
    return {p.parent.name for p in (ROOT / "runs_v3" / stage).glob("*/summary.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--session", required=True, help="a name for the Colab machine")
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--workers", type=int, default=8, help="groups trained at once on the card")
    parser.add_argument("--share", default="0/1", help="K/N: every N-th group starting at K")
    parser.add_argument("--groups", default="", help="A-B: only the groups numbered A to B in the stage's list")
    parser.add_argument("--hours", type=float, default=23.0, help="time budget of the job")
    parser.add_argument("--poll-seconds", type=float, default=180.0)
    args = parser.parse_args()
    part, parts = (int(x) for x in args.share.split("/"))
    first, last = (int(x) for x in args.groups.split("-")) if args.groups else (0, 10 ** 9)
    log_path = ROOT / "logs" / "v3" / f"colab_{args.stage}_{args.session}.log"
    share = args.share + (f", groups {args.groups}" if args.groups else "")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def say(text):
        line = f"{time.strftime('%F %T')}  {text}"
        print(line, flush=True)
        with open(log_path, "a") as log:
            log.write(line + "\n")

    def waiting_groups():
        done = finished_names(args.stage)
        return [group for index, group in enumerate(stage_groups(args.stage))
                if index % parts == part and first <= index <= last
                and not all(c["name"] in done for c in group)]

    def sync():
        """Ask the machine how far it is and download what is new. Returns the answers."""
        printed = execute(args.session, f"STAGE = {args.stage!r}\n" + POLL)
        answers = {line.split()[0]: line.split(None, 1)[1] if " " in line else ""
                   for line in printed.splitlines() if line.split() and line.split()[0] in
                   ("NEWFILES", "FINISHED", "ALIVE", "ENDED", "CARD")}
        answers["WORKERS"] = [line for line in printed.splitlines() if line.startswith("WORKER")]
        if not {"NEWFILES", "ALIVE", "ENDED"} <= set(answers):
            return None
        if int(answers["NEWFILES"]):
            local = ROOT / "logs" / "v3" / f"colab_new_{args.session}.tar"
            code, text = colab("download", "-s", args.session, "/content/air_run/new.tar", str(local))
            if code != 0 or not local.exists():
                say(f"download failed, will try again: {text.strip()[-200:]}")
                return answers
            with tarfile.open(local) as tar:
                tar.extractall(ROOT / "runs_v3")
            local.unlink()
        if "COMMITTED" not in execute(args.session, COMMIT):
            say("could not confirm the download on the machine; the same files come again next time")
        return answers

    todo = waiting_groups()
    if not todo:
        say(f"stage {args.stage}, share {share}: every run is already here; nothing to do")
        return
    runs = sum(len(group) for group in todo)
    say(f"stage {args.stage}, share {share}: {len(todo)} groups ({runs} runs) to train on "
        f"Colab machine '{args.session}' ({args.gpu}), {args.workers} groups at a time")

    if "not found" in colab("status", "-s", args.session)[1]:
        code, text = colab("new", "-s", args.session, "--gpu", args.gpu, timeout=600)
        if code != 0 or "READY" not in text:
            say(f"Colab did not give a machine: {text.strip()[-300:]}")
            sys.exit(1)
        say(f"machine '{args.session}' created")

    job = (f'PACKED = "{packed_v2()}"\nGROUPS = {json.dumps(json.dumps(todo))}\n'
           f"STAGE = {args.stage!r}\nBUDGET_MINUTES = {args.hours * 60!r}\nWORKERS = {args.workers!r}\n" + START)
    printed = execute(args.session, job)
    if "STARTED" in printed:
        say("the job has been started on the machine")
    elif "ALREADY RUNNING" in printed:
        say("a job is already running on the machine; watching it")
    else:
        say(f"the job could not be started: {printed.strip()[-400:]}")
        sys.exit(1)

    failures = 0
    while True:
        time.sleep(args.poll_seconds)
        try:
            answers = sync()
        except subprocess.TimeoutExpired:
            answers = None
        if answers is None:
            failures += 1
            say(f"the machine did not answer (try {failures})")
            if failures >= 10:
                say("giving up after 10 tries without an answer; the machine may be gone. "
                    "Start this program again to send what is missing to a new machine.")
                sys.exit(1)
            continue
        failures = 0
        here = len(finished_names(args.stage))
        say(f"runs ended on the machine {answers.get('FINISHED')}, here {here}; card {answers.get('CARD')}")
        if answers["ALIVE"] == "0":
            break
    sync()  # once more, for runs that ended in the last seconds
    left = waiting_groups()
    if left:
        say(f"the job on the machine has ended but {len(left)} groups are not finished "
            f"(time budget used or an error; see /content/air_run/job.log on the machine). "
            f"The machine is left running; start this program again to go on.")
        sys.exit(2)
    say("every run of this share has ended and is here")
    colab("stop", "-s", args.session)
    say(f"machine '{args.session}' stopped")


if __name__ == "__main__":
    main()
