"""Send a job to Kaggle's free graphics cards and bring back what it printed.

    .venv/bin/python -m v2.kaggle_push JOB_FILE --name air-timing

A Kaggle "script" is one file. So this program packs the whole v2 folder into that one file,
followed by the job (ordinary Python that runs on Kaggle after v2 has been unpacked into the
working folder). It then sends the file with the Kaggle tool, waits until the session has ended
(the session, and with it the use of the weekly quota, stops when the script ends), downloads
the log and prints it.

The Kaggle user name is read from ~/.kaggle/username_for_scripts; the sign-in itself is the
Kaggle tool's own (~/.kaggle).
"""

import argparse
import base64
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = Path.home() / ".local" / "bin" / "kaggle"
WORK = Path("/tmp/air_kaggle")

HEAD = '''# Made by v2/kaggle_push.py. First unpack the v2 folder, then run the job below.
import base64, io, os, sys, tarfile
os.chdir("/kaggle/working")
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PACKED))).extractall(".")
sys.path.insert(0, "/kaggle/working")
'''


def packed_v2():
    """The v2 folder (programs, settings and tests, not the report) as one text string."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for path in sorted(list((ROOT / "v2").rglob("*")) + list((ROOT / "v3").rglob("*"))):
            relative = path.relative_to(ROOT)
            if path.is_file() and path.suffix in (".py", ".json") and "report" not in relative.parts \
                    and "__pycache__" not in relative.parts:
                tar.add(path, arcname=str(relative))
    return base64.b64encode(buffer.getvalue()).decode()


def kaggle(*arguments, timeout=300):
    done = subprocess.run([str(KAGGLE), *arguments], capture_output=True, text=True, timeout=timeout)
    return (done.stdout + done.stderr).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", help="a Python file to run on Kaggle after v2 is unpacked")
    parser.add_argument("--name", required=True, help="short name of the Kaggle script, like air-timing")
    parser.add_argument("--wait-minutes", type=int, default=60)
    args = parser.parse_args()

    user = (Path.home() / ".kaggle" / "username_for_scripts").read_text().strip()
    folder = WORK / args.name
    folder.mkdir(parents=True, exist_ok=True)
    script = f'PACKED = "{packed_v2()}"\n' + HEAD + "\n" + Path(args.job).read_text()
    (folder / "script.py").write_text(script)
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{args.name}", "title": args.name.replace("-", " "), "code_file": "script.py",
        "language": "python", "kernel_type": "script", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "false",
        "dataset_sources": [], "competition_sources": [], "kernel_sources": []}))
    print(kaggle("kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4")
          .replace(user, "<user>").splitlines()[-1], flush=True)

    started = time.time()
    status = ""
    while time.time() - started < args.wait_minutes * 60:
        time.sleep(20)
        status = kaggle("kernels", "status", f"{user}/{args.name}").replace(user, "<user>")
        if any(word in status for word in ("COMPLETE", "ERROR", "CANCEL")):
            break
    print(f"{status.splitlines()[-1] if status else 'no status'}  "
          f"(after {(time.time() - started) / 60:.1f} minutes)", flush=True)

    out = folder / "out"
    out.mkdir(exist_ok=True)
    kaggle("kernels", "output", f"{user}/{args.name}", "-p", str(out), timeout=600)
    for log in out.glob("*.log"):
        try:
            lines = [item.get("data", "") for item in json.loads(log.read_text())]
        except ValueError:
            lines = log.read_text().splitlines(keepends=True)
        text = "".join(lines)
        (folder / "log.txt").write_text(text)
        print(f"log saved to {folder / 'log.txt'} ({len(text):,} characters)")
    if not any(word in status for word in ("COMPLETE", "ERROR", "CANCEL")):
        sys.exit("The session had not ended when the wait was over. Check it on kaggle.com.")


if __name__ == "__main__":
    main()
