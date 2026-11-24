"""A pretend Colab command-line tool, for the tests of v3/colab_fleet.py.

It understands what the fleet uses: new, exec, upload, download, stop, status, usage (with the
global option --config, which it ignores). Its state is the JSON file named by the environment
variable FAKE_COLAB_STATE:

    machines   {name: {"card": .., "running": true/false}}
    refuse     card types it does not grant ("no card free"; a test may change it at any time)
    limit      (optional) how many machines may run at once ("precondition failed" beyond)
    gone       machines that have disappeared (status: not found; exec fails)
    balance    compute units left; rate: units an hour
    log        every command, in order

A machine is a folder next to the state file: machines/<name>. In code sent by exec, and in
the paths of upload and download, "/content" is replaced by that folder. The code is run on
this computer, so the job of v3/machine_job.py really trains there (on the processor).
"""

import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

STATE = Path(os.environ["FAKE_COLAB_STATE"])


def change(work):
    """Read, change and write the state, holding a lock (several commands run at once)."""
    with open(STATE.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(STATE.read_text())
        result = work(state)
        STATE.write_text(json.dumps(state, indent=1))
        return result


def folder(name):
    return STATE.parent / "machines" / name


def mapped(path, name):
    return path.replace("/content", str(folder(name)))


def main(argv):
    if argv[:1] == ["--config"]:
        argv = argv[2:]
    command, rest = argv[0], argv[1:]
    name = rest[rest.index("-s") + 1] if "-s" in rest else None
    change(lambda s: s.setdefault("log", []).append([command, name]))
    state = json.loads(STATE.read_text())
    there = (name in state["machines"] and state["machines"][name]["running"]
             and name not in state.get("gone", []))
    if command == "new":
        card = rest[rest.index("--gpu") + 1]
        # (The real tool asks again after a 503 unless COLAB_CLI_ASSIGN_RETRIES=0; the fleet
        # always sets 0. Here the answer is always given at once.)
        running = [n for n, m in state["machines"].items()
                   if m["running"] and n not in state.get("gone", [])]
        if state.get("limit") is not None and len(running) >= state["limit"]:
            print("[colab] Allocation refused (precondition failed). This can mean too many "
                  "active sessions, ...")
            return 1
        if card in state.get("refuse", []):
            print("colab_cli.client.ColabRequestError: Failed to issue request POST "
                  "https://colab.research.google.com/tun/m/assign: Service Unavailable")
            return 1
        folder(name).mkdir(parents=True, exist_ok=True)
        change(lambda s: s["machines"].__setitem__(name, {"card": card, "running": True}))
        print(f"Session {name} READY")
        return 0
    if command == "usage":
        print(f"Current balance: {state['balance']} compute units\n"
              f"Usage rate: {state.get('rate', 0)}/hr")
        return 0
    if command == "status":
        print(f"[{name}] {state['machines'][name]['card']}" if there
              else f"[colab] Session '{name}' not found.")
        return 0
    if command == "stop":
        subprocess.run(["pkill", "-f", str(folder(name))])
        if name in state["machines"]:
            change(lambda s: s["machines"][name].__setitem__("running", False))
        return 0
    if not there:
        print("Error: Connection was lost")
        return 1
    if command == "exec":
        code = mapped(sys.stdin.read(), name)
        script = folder(name) / f"exec_{os.getpid()}.py"
        script.write_text(code)
        done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                              cwd=STATE.parent)
        script.unlink()
        print(done.stdout + done.stderr)
        return done.returncode
    if command == "upload":
        local, remote = rest[-2], mapped(rest[-1], name)
        Path(remote).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, remote)
        return 0
    if command == "download":
        remote, local = mapped(rest[-2], name), rest[-1]
        if not Path(remote).exists():
            print("Error: no such file")
            return 1
        shutil.copy2(remote, local)
        return 0
    print(f"fake colab: unknown command {command}")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
