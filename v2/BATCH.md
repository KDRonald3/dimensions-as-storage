# The group trainer: many models at once on one graphics card

## What it is

`v2/run.py` trains one model at a time on the processor. Our models are tiny, so one of
them leaves a graphics card almost idle. The group trainer (`v2/batch.py`) trains many models
side by side in one pass instead.

- A **group** is a set of runs that differ only in the number of creatures, the word use and
  the seed. Everything else (kinds, body width, learning rate, steps, ...) is the same.
  In `main8`, that gives 5 groups of 40 runs (one per body width).
- Each model in the group starts from exactly the same numbers as in `run.py`, and is trained
  by the same rules. The models never mix: each one learns only from its own sentences.
- The training sentences are made on the graphics card (`v2/batch_text.py`). They follow the
  same rules and chances as the sentences of `v2/lang.py`, but they are not the same sentences.
- Each run ends with the same files as from `run.py`: `summary.json` and `model.pt`. The
  summary has one more entry, `"trainer": "batch"`. The analysis can read them unchanged.
- The minutes and steps per second in a summary are those of the whole group.
- Every 2,000 steps the group is saved. A group that is started again goes on from there.

## Run it on your computer (small checks only)

    .venv/bin/python -m v2.batch_cli smoke --device cpu

## Run it on Kaggle (free graphics card)

1. Make a Kaggle dataset from the folder `v2` (kaggle.com, Datasets, New Dataset, upload the
   folder). Leave out `v2/report` and `v2/tests`, which are not needed.
2. Make a new notebook. In its settings, choose the accelerator "GPU P100" or "GPU T4 x2".
3. Add Input: your dataset.
4. Paste the whole of `v2/kaggle_notebook.py` into one cell. At the top, set `STAGE`
   (for example `"main8"`).
5. Click "Save Version" and choose "Save & Run All". This runs in the background, even with
   the browser closed. A session may last at most 12 hours.
6. When it is done, open the version, go to Output, and download `runs_v2_batch.zip`.
7. If it was stopped before the end: add that version's output as an input of the notebook
   and save a new version. Finished runs are skipped and the stopped group goes on.

With "T4 x2" only one of the two cards is used.

## Run it on Modal with a graphics card

`v2/modal_app.py` uses the processor only and was not changed. A graphics card on Modal needs
a function like this one (not written as a file, and not tried):

    gpu_image = (modal.Image.debian_slim(python_version="3.12")
                 .pip_install("torch==2.15.0", "numpy==2.5.5")   # the build with CUDA
                 .add_local_python_source("v2"))

    @app.function(image=gpu_image, gpu="T4", volumes={"/runs": volume}, timeout=12 * 60 * 60)
    def train_stage_on_gpu(stage: str):
        from v2.batch_cli import main
        main([stage, "--out-root", "/runs/batch", "--device", "cuda"])
        volume.commit()

Started with `modal run --detach FILE::train_stage_on_gpu --stage main8`. Saving the volume
every few minutes (as `modal_app.py` does) would keep the checkpoints if the machine stops.
A graphics card costs money on Modal; try a short stage first.

## What was tested, and what was not

Tested on the processor (`v2/tests/test_v2_batch.py`, 9 tests, about one minute):

- Three models trained together for 40 steps on the same sentences as three separate trainings
  by `run.py` end with the same numbers (within 0.00001), with the word table trained and
  frozen. One exception: the bias of the attention "keys". It has no effect on what a model
  computes, so it only moves by rounding noise, in `run.py` as much as here. The test checks
  that it stays close and that the model's scores still agree.
- The sentence maker: 20,480 sentences for each of 8 settings (16 and 256 creatures, equal and
  Zipf use, 4 and 16 kinds) are all true, never name a held-back pair, are never longer than 13
  words, and have the same shares as those of `lang.py` (sentence types, list lengths, kinds,
  values, f, and how often each creature is named).
- A group of 16 and 64 creatures learns all facts (100) in 600 steps.
- The summary has exactly the entries of a `run.py` summary, plus `"trainer"`.
- A group stopped after a checkpoint and started again ends exactly as one that never stopped.
- The sentences of a step are the same when made twice.

Not tested:

- **Nothing was run on a graphics card.** The speed there is not known.
- Whether Kaggle's version of PyTorch still supports the P100 card (newer versions dropped
  older cards; the T4 is supported).
- How many models fit on one card at once. If memory runs out, lower `MAX_MEMBERS`.
- Exact picking up again on a graphics card (it is exact on the processor). It should hold
  when the same card type is used again.
- A run picked up on a different kind of machine gets different sentences from then on (same
  rules and chances, so the result is just as valid, but not identical).
