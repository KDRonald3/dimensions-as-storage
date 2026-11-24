# Dimensions as storage

Do small transformers store facts in the space their words live in?

**Preliminary results: https://kdronald3.github.io/dimensions-as-storage/**

This repository holds everything behind that page: the code, the plan written before each
experiment, the result of every training run, and the reports.

## The question

A language model is given a word as a row of numbers (its embedding). How much a model can
remember about its words might depend on how many numbers that row has (the dimensions of the
space the words live in), or simply on how many parameters the model has in total. The
experiments here train thousands of very small transformers on made-up languages whose only
content is random facts, so that each language is a known number of bits to store, and vary the
two sizes separately.

So far: at the same parameter count, the model with more dimensions knew more facts in every
comparison that was not already at 100. The results page shows the pictures and the limits of
the work.

## What is here

| Path | What it is |
|---|---|
| `docs/` | The results page (served by GitHub Pages). |
| `v3/` | The third experiment: 2,880 models over 960 settings, and 45 size-matched models. Plan: `v3/PLAN.md`. Report: `v3/writeup/report.pdf` (written before the size-matched runs). The runs are defined in `v3/grid.py`. |
| `v2/` | The second experiment, and the code the third reuses: the model (`v2/net.py`), the language (`v2/lang.py`), one run (`v2/run.py`) and the trainer that trains many small models together on one card (`v2/batch.py`). Plan: `v2/PLAN.md`. Report: `v2/writeup/report.pdf`. |
| `*.py` at the top, `report/`, `results/` | The first experiment. See `experiment1.md`. |
| `runs_v3/`, `runs_v2/`, `runs_more_exact/` | One `summary.json` for every training run: its settings, parameter counts, scores and learning curve. The trained weights are not included because of their size. |
| `tests/`, `v2/tests/`, `v3/tests/` | Tests. |

`v3/colab_fleet.py`, `v3/fleet_plan.py`, `v3/machine_job.py` and `v3/colab_gpu.py` are the
scheduler that ran the third experiment on rented Google Colab GPUs. They are specific to that
setup.

## Running it

Python 3.12 or newer with PyTorch, NumPy and Matplotlib (see `pyproject.toml`).

```
python -m pytest v3/tests/test_v3_bridge.py -q     # checks the size-matched runs' design
python -m v3.matched_pairs_picture pairs.png       # redraws the main figure from the run summaries
python -m v2.run --help                            # train one model
```

## Status

Preliminary. Small models (2 layers), a synthetic language, one learning rate, training capped
at 640,000 steps. The next two tests are named at the end of the results page.

## License

MIT. See `LICENSE`.
