"""
v3/modal_profile.py - run v3/profile_speed.py on one T4 card on Modal, and print what it printed.

    ~/.local/bin/modal run v3/modal_profile.py                       # all three shapes
    ~/.local/bin/modal run v3/modal_profile.py --args "--table"      # with torch.profiler
    ~/.local/bin/modal run v3/modal_profile.py --args "--plain ;; --decompose"   # two in one run

Uses `modal run` only: nothing stays deployed or running afterwards. A run stops after at most
5 minutes.
"""

import shlex
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent.parent

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.15.0", "numpy")
         .add_local_dir(HERE / "v2", "/root/air/v2", ignore=["tests", "report", "writeup",
                                                              "__pycache__"])
         .add_local_dir(HERE / "v3", "/root/air/v3", ignore=["tests", "__pycache__"]))
app = modal.App("air-v3-profile", image=image)


@app.function(gpu="T4", timeout=300)
def profile(args: str = "") -> str:
    import subprocess
    printed = ""
    for one in args.split(";;"):  # several measurements in one run, separated by ";;"
        done = subprocess.run(["python", "-m", "v3.profile_speed", *shlex.split(one)],
                              cwd="/root/air", capture_output=True, text=True)
        printed += done.stdout + done.stderr[-1500:]
    return printed


@app.local_entrypoint()
def main(args: str = ""):
    print(profile.remote(args))
