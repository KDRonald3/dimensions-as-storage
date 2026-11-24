"""Tests of v2/grid.py: it needs only the standard library, and the stages are right."""

import json
import subprocess
import sys
from pathlib import Path

from v2 import grid

PROJECT = Path(__file__).resolve().parents[2]


def test_grid_imports_only_the_standard_library():
    check = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import v2.grid\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "outside = sorted(m for m in new if m != 'v2' and m not in sys.stdlib_module_names)\n"
        "print(outside)\n"
    )
    result = subprocess.run([sys.executable, "-c", check], cwd=PROJECT, capture_output=True,
                            text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_stages():
    main8 = grid.grid("main8")
    assert len(main8) == 200
    assert len({config["name"] for config in main8}) == 200
    assert all(config["kinds"] == 4 and config["threads"] == (1 if config["body"] <= 64 else 2) for config in main8)
    assert all(config["table"] == "trained" for config in main8)
    bits64 = grid.grid("bits64")
    assert len(bits64) == 36 and len({c["name"] for c in bits64}) == 36
    assert {c["kinds"] for c in bits64} == {32} and {c["seed"] for c in bits64} == {0, 1}
    for stage, kinds in (("bits16", 8), ("bits32", 16)):
        configs = grid.grid(stage)
        assert len(configs) == 120 and len({c["name"] for c in configs}) == 120
        assert {c["kinds"] for c in configs} == {kinds}
        assert {c["seed"] for c in configs} == {0, 1, 2}
        assert {c["table"] for c in configs} == {"trained"}
    frozen8 = grid.grid("frozen8")
    assert len(frozen8) == 200 and {c["table"] for c in frozen8} == {"frozen"}
    assert [c["name"] for c in frozen8] == [c["name"] + "_frozen" for c in main8]
    assert [{k: v for k, v in c.items() if k not in ("name", "table")} for c in frozen8] == \
        [{k: v for k, v in c.items() if k not in ("name", "table")} for c in main8]
    pilot = grid.grid("pilot")
    assert len(pilot) == 21 and len({c["name"] for c in pilot}) == 21
    assert {c["seed"] for c in pilot} == {100}
    smoke = grid.grid("smoke")
    assert smoke == [{"creatures": 16, "kinds": 4, "usage": "equal", "body": 32, "seed": 100,
                      "steps": 300, "lr": 0.01, "threads": 1, "table": "trained",
                      "name": "c16_b32_equal_k4_s100"}]
    long_runs = grid.grid("long")
    assert len(long_runs) == 8 and all(c["name"].endswith("_long") and c["seed"] == 0 for c in long_runs)
    settings = grid.settings()
    assert {c["steps"] for c in main8} == {settings["steps"]}
    assert {c["lr"] for c in main8} == {settings["lr"]}


def test_command_line_prints_json():
    result = subprocess.run([sys.executable, "-m", "v2.grid", "main8"], cwd=PROJECT,
                            capture_output=True, text=True, check=True)
    assert len(json.loads(result.stdout)) == 200
