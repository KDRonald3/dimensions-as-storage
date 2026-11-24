# Running the second experiment on Modal

Run all commands from the project folder. `modal` is at `~/.local/bin/modal`.

## One-time set-up (you do this)

    ~/.local/bin/modal setup

This opens a browser to sign in. It creates `~/.modal.toml`.

## Start a stage

Use `--detach`. The runs then keep going on Modal even if you close the
terminal or shut the computer.

    modal run --detach v2/modal_app.py::main --stage smoke
    modal run --detach v2/modal_app.py::main --stage pilot
    modal run --detach v2/modal_app.py::main --stage main8
    modal run --detach v2/modal_app.py::main --stage bits16
    modal run --detach v2/modal_app.py::main --stage bits32
    modal run --detach v2/modal_app.py::main --stage long

Try only the first 3 runs of a stage:

    modal run --detach v2/modal_app.py::main --stage pilot --limit 3

What it does:

- Runs that already have a `summary.json` on Modal are skipped. So you can
  start a stage again after a failure and it only does the missing runs.
- At most 100 runs go at the same time. Each gets 2 cores and 2 GiB.
- A line is printed as each run ends: name, facts score, minutes.
- At the end you get the number of failures and a cost estimate.
- If the terminal closed, the runs still finish and save on Modal. Only the
  printing stops. Start the same command again to see which are done (finished
  runs are skipped), or just fetch.

A failed run is tried 3 times in all (first try plus 2 retries). A retry
continues from the last checkpoint (saved every 1,000 steps, and the volume is
saved every 3 minutes).

## Fetch the results

    modal run v2/modal_app.py::fetch --stage pilot

This copies `summary.json` and `model.pt` of every run into
`runs_v2/<stage>/<name>/`. Files already there are not downloaded again.
It also prints the cost estimate.

## Look at things

    modal volume ls air-v2-runs /pilot
    modal app list

Results on Modal are in the volume `air-v2-runs`, at `/<stage>/<name>/`.

## Cost estimate

Per run: minutes x 60 x (2 x $0.0000131 + 2 x $0.00000222) per second.
That is about $0.0000306 per second, or $0.0018 per minute. It counts
training time only.
