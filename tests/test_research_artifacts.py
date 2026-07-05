"""Small end-to-end checks for remote benchmark generation."""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import oracle_rollout


REPO = Path(__file__).resolve().parents[1]
BENCHMARKS = REPO / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import _bench_common as bench_common  # noqa: E402
from _bench_common import write_csv  # noqa: E402


def test_python_rollout_benchmark_baseline_matches_oracle(pendulum):
    state = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    controls = torch.tensor([[[0.3], [-0.2]]], dtype=torch.float64)
    actual = bench_common.python_rollout(pendulum, state, controls)
    expected = oracle_rollout(pendulum, state[0].numpy(), controls[0].numpy())
    torch.testing.assert_close(actual[0], torch.from_numpy(expected))


def test_writer_emits_lf_csv_and_hashed_manifest(tmp_path):
    path = tmp_path / "samples.csv"
    write_csv(path, [{"model": "pendulum", "elapsed_ms": 1.5}], seed=7)

    data = path.read_bytes()
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert b"\r" not in data
    assert manifest["data_file"] == path.name
    assert manifest["data_sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["seed"] == 7
    assert manifest["git_commit"]
    assert {"python", "packages", "os", "cpu", "command"} <= manifest.keys()


@pytest.mark.parametrize(
    ("suffix", "rows"),
    [("", [{"value": 1}]), (".json", [{"value": 1}]), (".CSV", [{"value": 1}]), (".csv", [])],
)
def test_writer_requires_nonempty_csv_output(tmp_path, suffix, rows):
    path = tmp_path / f"samples{suffix}"
    with pytest.raises(ValueError):
        write_csv(path, rows)


@pytest.mark.parametrize(
    ("script", "backend"),
    [("bench_rollout.py", "python_ref"), ("bench_backward.py", "python_vjp")],
)
def test_cpu_runner_writes_samples_and_manifest(tmp_path, script, backend):
    output = tmp_path / f"{Path(script).stem}.csv"
    subprocess.run(
        [
            sys.executable,
            str(BENCHMARKS / script),
            "--backend", backend,
            "--model", "dm_pendulum",
            "--batch", "1",
            "--horizon", "1",
            "--repeat", "2",
            "--warmup", "1",
            "--out", str(output),
        ],
        cwd=REPO,
        check=True,
        timeout=120,
    )
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert [row["sample"] for row in rows] == ["0", "1"]
    assert all(float(row["elapsed_ms"]) > 0 for row in rows)
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert manifest["data_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_correctness_runner_records_both_epsilons(tmp_path):
    output = tmp_path / "gradient.csv"
    subprocess.run(
        [
            sys.executable,
            str(BENCHMARKS / "compare_finite_difference.py"),
            "--model", "dm_pendulum",
            "--horizon", "1",
            "--reference-eps", "1e-6",
            "--seed", "0",
            "--out", str(output),
        ],
        cwd=REPO,
        check=True,
        timeout=120,
    )
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {float(row["reference_eps"]) for row in rows} == {1e-6}
    assert {float(row["transition_eps"]) for row in rows} == {1e-8}
    transition = np.array([float(row["transition_fd_vjp"]) for row in rows])
    reference = np.array([float(row["whole_rollout_fd"]) for row in rows])
    assert np.linalg.norm(transition - reference) / np.linalg.norm(reference) < 1e-6


def test_generated_results_are_ignored():
    ignored = subprocess.run(
        ["git", "check-ignore", "benchmarks/results/probe.csv"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert ignored.strip()
