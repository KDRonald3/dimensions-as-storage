"""
v3/modal_card_check.py - run the reviewer's card check (v3/card_check.py) on one A100 on Modal.

    ~/.local/bin/modal run v3/modal_card_check.py --columns 32,8

`modal run` only: nothing stays deployed. A run stops after at most 6 minutes. What the check
prints is shown as it comes (so a run cut short still shows how far it got).
"""

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent.parent

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.15.0", "numpy")
         .add_local_dir(HERE / "v2", "/root/air/v2", ignore=["tests", "report", "writeup",
                                                              "__pycache__"])
         .add_local_dir(HERE / "v3", "/root/air/v3", ignore=["tests", "__pycache__"]))
app = modal.App("air-v3-card-check", image=image)


@app.function(gpu="A100-40GB", timeout=360)
def check(columns: str = "32") -> None:
    """Runs the check once for every width in `columns` (for example "32,8"). Only the check's
    own lines are shown (the compiler's long tuning messages go to a file, shown on an error)."""
    import os
    import subprocess
    import time
    started = time.time()
    for width in columns.split(","):
        environment = dict(os.environ, CHECK_COLUMNS=width)
        with open("/tmp/stderr.txt", "w") as errors:
            done = subprocess.run(["python", "-m", "v3.card_check"], cwd="/root/air",
                                  env=environment, stderr=errors,
                                  timeout=max(10, 345 - (time.time() - started)))
        print(f"columns {width}: exit code {done.returncode}", flush=True)
        if done.returncode not in (0, 1):
            print(open("/tmp/stderr.txt").read()[-2000:], flush=True)


@app.local_entrypoint()
def main(columns: str = "32"):
    check.remote(columns)
