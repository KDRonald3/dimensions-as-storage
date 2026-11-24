"""
v3/modal_fleet_profile.py - run v3/profile_fleet.py (several measurements) on one A100 on Modal.

    ~/.local/bin/modal run v3/modal_fleet_profile.py --args "--groups 8,4,128x1 ;; --groups 8,4,128x4"

`modal run` only: nothing stays deployed. A run stops after at most 6 minutes.
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
app = modal.App("air-v3-fleet-profile", image=image)


@app.function(gpu="A100-40GB", timeout=360)
def profile(args: str = "") -> str:
    import subprocess
    import time
    started = time.time()
    printed = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True,
                             capture_output=True, text=True).stdout
    for one in args.split(";;"):  # several measurements, one program each
        if time.time() - started > 300:
            printed += f"SKIPPED (no time left): {one}\n"
            continue
        # "A && B": A and B as separate programs at the same time (sharing the card).
        programs = [subprocess.Popen(["python", "-m", "v3.profile_fleet", *shlex.split(part)],
                                     cwd="/root/air", stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True)
                    for part in one.split("&&")]
        for program in programs:
            try:
                out, err = program.communicate(timeout=max(10, 345 - (time.time() - started)))
                printed += out + (err[-1200:] if program.returncode else "")
            except subprocess.TimeoutExpired:
                program.kill()
                printed += f"TIMED OUT: {one}\n"
    return printed


@app.local_entrypoint()
def main(args: str = ""):
    print(profile.remote(args))
