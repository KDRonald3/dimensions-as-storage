"""Get Colab machines with the fastest graphics cards that are free RIGHT NOW.

    .venv/bin/python -m v3.colab_gpu grab --count 2 --prefix air     # get 2 machines
    .venv/bin/python -m v3.colab_gpu grab --count 1 --dry            # only look, keep nothing

Whether a card type is free can only be found out by asking for a machine with it, and the
answer can change within minutes. So this program does not test first and launch later. It asks
for every card type at the same moment, waits a short time for the answers, keeps the fastest
machines that were granted, and gives the others back at once. Getting the machine IS the test,
so the choice is never based on old information.

The Colab tool keeps a lock on its file of sessions while it asks for a machine, so requests
that share that file wait for each other. So every request here gets a state file of its own
(--config, in ~/.config/colab-cli/fleet/<name>.json). Every later command for that machine
must name the same file: `colab --config FILE exec -s NAME ...` (the fleet does this). The
practice watchers, which use the usual file, are not held up either.

A request that has not answered within the wait is not killed (a machine could then be
granted with nobody knowing): it is left to finish, and a machine it gets is given back at once.

It prints one line of JSON: the machines kept (session name, card, state file), and how long it
took.
"""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

COLAB = Path.home() / ".local" / "bin" / "colab"
CONFIGS = Path.home() / ".config" / "colab-cli" / "fleet"

# Fastest first. H100 and A100 are measured classes; the place of G4 is an assumption until it
# has been timed with our trainer.
ORDER = ["H100", "G4", "A100", "L4", "T4"]


def config_of(name):
    """The state file of the machine `name` (see the top)."""
    return CONFIGS / f"{name}.json"


def ask(name, card, colab=COLAB):
    CONFIGS.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen([str(colab), "--config", str(config_of(name)), "new", "-s", name,
                             "--gpu", card], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)


def stop(names, colab=COLAB):
    waiting = [subprocess.Popen([str(colab), "--config", str(config_of(name)), "stop", "-s", name],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL) for name in names]
    for process in waiting:
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()


def give_back_when_answered(late, colab=COLAB, wait_seconds=600):
    """For requests that had not answered in time: wait for their answer and give back any
    machine they got. Runs in the background (a thread); returns the thread."""
    def work():
        for name, (card, process) in late.items():
            try:
                text = process.communicate(timeout=wait_seconds)[0]
            except subprocess.TimeoutExpired:
                process.kill()
                text = ""
            if "READY" in text:
                stop([name], colab)
    thread = threading.Thread(target=work, daemon=False)
    thread.start()
    return thread


def grab(count, prefix, wait_seconds=10.0, cards=ORDER, dry=False, colab=COLAB):
    """Ask for `count` machines of every card type at once; keep the `count` fastest granted."""
    started = time.time()
    stamp = time.strftime("%H%M%S")
    asked = {}
    for card in cards:
        for number in range(count):
            name = f"{prefix}-{card.lower()}-{stamp}-{number}"
            asked[name] = (card, ask(name, card, colab))
    granted, refused = [], {}
    deadline = started + wait_seconds
    pending = dict(asked)
    while pending and time.time() < deadline:
        for name, (card, process) in list(pending.items()):
            if process.poll() is None:
                continue
            text = process.stdout.read()
            if "READY" in text:
                granted.append((cards.index(card), name, card, round(time.time() - started, 1)))
            else:
                refused[card] = (text.strip().splitlines() or ["no answer"])[-1][-80:]
            del pending[name]
        # Stop waiting once enough machines are granted and no faster type is still unanswered.
        granted.sort()
        if len(granted) >= count:
            slowest_kept = granted[count - 1][0]
            if not any(cards.index(card) < slowest_kept for card, _ in pending.values()):
                break
        time.sleep(0.3)
    granted.sort()
    keep = [] if dry else granted[:count]
    kept_names = {k[1] for k in keep}
    stop([name for _, name, _, _ in granted if name not in kept_names], colab)
    give_back_when_answered(pending, colab)
    return {"kept": [{"session": name, "card": card, "config": str(config_of(name)),
                      "answered_after_seconds": seconds} for _, name, card, seconds in keep],
            "granted": [f"{card} after {seconds}s" for _, _, card, seconds in granted],
            "refused": refused, "not_answered_in_time": len(pending),
            "seconds": round(time.time() - started, 1)}


# ------------------------------------------------------------------------------------------------
# One request per card type, without the tool's hidden retries (used by the fleet)
# ------------------------------------------------------------------------------------------------
#
# What was found on 23 November (in ~/.config/colab-cli/colab.log, all morning):
#   - Colab grants a free card in 2.5 to 5 seconds (once 18 s).
#   - When no card of a type is free, Colab answers "503 Service Unavailable", always after
#     30.2 seconds (Colab itself waits that long before saying no).
#   - When the account may not have another machine, Colab answers "412 Precondition Failed"
#     within 3 to 8 seconds.
#   - The tool hides the 503: it waits 10, 20, 40 s and asks again, 4 times in all, so
#     `colab new` for a card type that is not free takes about 3 minutes.
# The tool reads the number of retries from the setting COLAB_CLI_ASSIGN_RETRIES (default 3).
# With 0 it asks once and reports the answer: a granted machine is written into the state file
# named by --config exactly as always, so `colab exec`, `download` and `stop` work on it as on
# any other. So a look at every card type takes about 30 seconds (the time Colab needs to say
# no) instead of minutes, and a card that is free is known (and ours) after about 3 seconds.

def outcome_of(text):
    """What the tool's answer to `colab new` means."""
    lowered = text.lower()
    if "session ready" in lowered or "READY" in text:
        return "granted"
    if "precondition failed" in lowered or "allocation refused" in lowered or " 412" in text:
        return "limit"  # too many machines at once, or a usage limit: not "no card free"
    if "service unavailable" in lowered or " 503" in text:
        return "no card free"
    if "rejected accelerator" in lowered:
        return "not allowed"
    return "error"


class Prober:
    """Asks Colab once for a machine of each card type, in the background, and hands over the
    answers as they come. A granted machine is the caller's from then on (it must use it or
    stop it)."""

    def __init__(self, prefix, colab=COLAB):
        self.prefix, self.colab = prefix, colab
        self.pending = {}  # session name -> (card, process, started)
        self.counter = 0
        self.found = []  # machines found running that no launcher knew (see colab_fleet)

    def pending_cards(self):
        return [card for card, _, _ in self.pending.values()]

    def ask(self, card):
        """Start one request for a machine with this card (unless one is already waiting)."""
        if card in self.pending_cards():
            return None
        CONFIGS.mkdir(parents=True, exist_ok=True)
        self.counter += 1
        name = f"{self.prefix}-{card.lower()}-{time.strftime('%H%M%S')}-{self.counter}"
        # No hidden retries; and the tool waits up to 240 s for Colab's answer (its default is
        # 120 s, after which it gives up while Colab may still grant the machine to nobody).
        environment = dict(os.environ, COLAB_CLI_ASSIGN_RETRIES="0",
                           COLAB_CLI_REQUEST_TIMEOUT="240")
        process = subprocess.Popen([str(self.colab), "--config", str(config_of(name)), "new",
                                    "-s", name, "--gpu", card], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   env=environment)
        self.pending[name] = (card, process, time.time())
        return name

    def answers(self):
        """The requests that have been answered since the last call:
        [{"session", "card", "outcome", "seconds", "text"}]."""
        done, self.found = self.found, []
        for name, (card, process, started) in list(self.pending.items()):
            if process.poll() is None:
                if time.time() - started < 300:
                    continue
                # Never seen (Colab answers within about 30 s), but a request must not hang for
                # ever. It is not killed: a machine it gets later is given back.
                give_back_when_answered({name: (card, process)}, self.colab)
                done.append({"session": name, "card": card, "outcome": "no answer",
                             "seconds": round(time.time() - started, 1), "text": ""})
                del self.pending[name]
                continue
            text = process.stdout.read()
            done.append({"session": name, "card": card, "outcome": outcome_of(text),
                         "seconds": round(time.time() - started, 1),
                         "text": (text.strip().splitlines() or [""])[-1][-160:]})
            del self.pending[name]
        return done

    def wait(self, seconds):
        """Wait until every request has been answered (or `seconds` have passed)."""
        until = time.time() + seconds
        while self.pending and time.time() < until:
            if all(p.poll() is not None for _, p, _ in self.pending.values()):
                break
            time.sleep(0.2)

    def close(self):
        """Requests still waiting are left to finish; a machine they get is given back."""
        if self.pending:
            give_back_when_answered({n: (c, p) for n, (c, p, _) in self.pending.items()},
                                    self.colab)
        self.pending = {}


def unknown_machines(colab=COLAB):
    """Machines Colab has for this account that no state file on this computer knows (for
    example one granted to a request that was cut off). Returns ["<endpoint> (<card>)"], or None
    when Colab could not be asked. Such a machine may also be one opened in a browser."""
    found = server_machines(colab)
    if found is None:
        return None
    known = known_sessions()
    return [f"{endpoint} ({card})" for endpoint, card in found.items() if endpoint not in known]


def server_machines(colab=COLAB):
    """{endpoint: card} of every machine Colab runs for this account now, or None when Colab
    could not be asked. (Asked with an empty state file of its own, so that the tool does not
    rewrite anybody's file of sessions.)"""
    CONFIGS.mkdir(parents=True, exist_ok=True)
    listing = CONFIGS / "_listing.json"
    listing.unlink(missing_ok=True)
    try:
        done = subprocess.run([str(colab), "--config", str(listing), "sessions"],
                              capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if done.returncode != 0:
        return None
    found = {}
    for line in done.stdout.splitlines():
        if line.startswith("[") and "] " in line:
            endpoint = line.split("] ", 1)[1].split(" ", 1)[0]
            card = line.split("Hardware: ", 1)[1].split(" ", 1)[0] if "Hardware: " in line else "?"
            found[endpoint] = card
    return found


def known_sessions():
    """{endpoint: (session name, seconds since its state file was written)} of every machine
    written in a state file of the Colab tool on this computer (the fleet's and the usual one)."""
    known = {}
    # Every project that uses the account keeps its state files in a folder of its own beside
    # the fleet's (~/.config/colab-cli/brain/ for the owner's brain project, since 23 November
    # 2026). They are read too, so that a machine of another project is known by its name and is
    # never taken for a lost machine of this fleet.
    others = [f for f in CONFIGS.parent.glob("*/*.json") if f.parent != CONFIGS]
    for path in list(CONFIGS.glob("*.json")) + others + [CONFIGS.parent / "sessions.json"]:
        try:
            age = time.time() - path.stat().st_mtime
            for name, session in json.loads(path.read_text()).items():
                known[session["endpoint"]] = (name, age)
        except (OSError, ValueError, AttributeError, KeyError, TypeError):
            continue
    return known


def take_in_endpoint(name, endpoint):
    """Make a machine that Colab runs but no state file here knows usable under the name `name`:
    it is written into a state file of its own with only its address. The tool fills in the
    rest from Colab before its first use (a session without an expiry time is refreshed), so
    `colab --config FILE exec/stop -s NAME` then work on it as on any other."""
    CONFIGS.mkdir(parents=True, exist_ok=True)
    config_of(name).write_text(json.dumps({name: {"name": name, "token": "", "url": "",
                                                  "endpoint": endpoint}}))


def look(cards, prefix, keep=0, colab=COLAB):
    """Ask once for every card type at the same time; keep the `keep` fastest granted and give
    the others back. Returns what happened and how long it took."""
    started = time.time()
    prober = Prober(prefix, colab)
    for card in cards:
        prober.ask(card)
    prober.wait(120)
    found = prober.answers()
    prober.close()
    granted = sorted((a for a in found if a["outcome"] == "granted"),
                     key=lambda a: cards.index(a["card"]))
    stop([a["session"] for a in granted[keep:]], colab)
    return {"kept": [a["session"] for a in granted[:keep]],
            "answers": [{k: a[k] for k in ("card", "outcome", "seconds")} for a in found],
            "seconds": round(time.time() - started, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["grab", "look"])
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--prefix", default="air")
    parser.add_argument("--wait-seconds", type=float, default=10.0)
    parser.add_argument("--cards", default=",".join(ORDER))
    parser.add_argument("--dry", action="store_true", help="look only: give every machine back")
    args = parser.parse_args()
    if args.what == "look":  # one request per card type, no retries; keeps --count machines
        print(json.dumps(look(args.cards.split(","), args.prefix,
                              keep=0 if args.dry else args.count)), flush=True)
        return
    print(json.dumps(grab(args.count, args.prefix, args.wait_seconds, args.cards.split(","),
                          args.dry)), flush=True)


if __name__ == "__main__":
    main()
