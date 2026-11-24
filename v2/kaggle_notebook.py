"""
v2/kaggle_notebook.py - train one stage of the second experiment on a free Kaggle graphics card.

Paste this whole file into one cell of a Kaggle notebook (or push it as the notebook's script).
Settings of the notebook: Accelerator "GPU T4 x2" or "GPU P100"; Internet may stay off.
The v2 folder must be there, either
  - as a dataset added to the notebook (Add Input -> your dataset that holds the folder v2), or
  - as files uploaded to /kaggle/working/v2.
See v2/BATCH.md for the steps.

Results are written to /kaggle/working/runs_v2_batch/STAGE/NAME/ (summary.json, model.pt), and
packed into /kaggle/working/runs_v2_batch.zip at the end for downloading.

Picking up again: Kaggle stops a session after 12 hours. Save the notebook's output ("Save
Version"), add that output to the next version as an input, and run again: any runs_v2_batch
folder found in the inputs is copied here first, finished runs are skipped and a stopped group
goes on from its checkpoint.
"""

# ---- Settings: change these -------------------------------------------------------------------
STAGE = "main8"
ONLY = ""  # for example "body=8+16" to train only some runs; "" for the whole stage
MAX_MEMBERS = 0  # split groups larger than this (0 = no limit); lower it if memory runs out
# -----------------------------------------------------------------------------------------------

import shutil
import sys
import time
from pathlib import Path

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
RESULTS = WORKING / "runs_v2_batch"


def find_code():
    """The folder that holds v2/ (uploaded files first, then attached datasets)."""
    if (WORKING / "v2" / "batch.py").exists():
        return WORKING
    for found in sorted(INPUT.glob("**/v2/batch.py")):
        # The inputs cannot be written to; copy the code next to the results.
        target = WORKING / "code"
        if not (target / "v2").exists():
            shutil.copytree(found.parent, target / "v2",
                            ignore=shutil.ignore_patterns("__pycache__", "report", "tests"))
        return target
    raise SystemExit("The folder v2 was not found. Add it as a dataset or upload it to "
                     "/kaggle/working/v2 (see v2/BATCH.md).")


def bring_back_earlier_results():
    """Copy results of an earlier version of this notebook (added as an input) into working."""
    for earlier in sorted(INPUT.glob("**/runs_v2_batch")):
        if earlier.is_dir():
            print(f"copying earlier results from {earlier}", flush=True)
            shutil.copytree(earlier, RESULTS, dirs_exist_ok=True)


code = find_code()
sys.path.insert(0, str(code))
bring_back_earlier_results()

import torch  # noqa: E402

print(f"torch {torch.__version__}, graphics card: "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("No graphics card. In the notebook settings, choose a GPU accelerator.")

from v2.batch_cli import main  # noqa: E402

arguments = [STAGE, "--out-root", str(RESULTS), "--device", "cuda"]
if ONLY:
    arguments += ["--only", ONLY]
if MAX_MEMBERS:
    arguments += ["--max-members", str(MAX_MEMBERS)]
started = time.perf_counter()
main(arguments)  # prints one line per finished run and the speed of every group
print(f"stage {STAGE} done in {(time.perf_counter() - started) / 3600:.2f} hours", flush=True)

shutil.make_archive(str(RESULTS), "zip", root_dir=RESULTS)
print(f"results packed into {RESULTS}.zip", flush=True)
