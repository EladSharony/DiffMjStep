"""Small integrity checks for generated benchmark figures."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest


BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import _plot_style as style  # noqa: E402
import plot_accuracy  # noqa: E402
import plot_dm_control  # noqa: E402
import plot_integrator_ablation  # noqa: E402


def _artifact(path: Path, rows: list[dict], **metadata) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "data_file": path.name,
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **metadata,
    }
    path.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(monkeypatch, module, *arguments):
    captured = {}
    monkeypatch.setattr(sys, "argv", [str(module.__file__), *map(str, arguments)])
    monkeypatch.setattr(module, "save", lambda fig, _path: captured.setdefault("fig", fig))
    module.main()
    return captured["fig"]


def test_reader_rejects_a_manifest_checksum_mismatch(tmp_path):
    path = _artifact(tmp_path / "rows.csv", [{"value": 1}])
    path.write_text("value\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        style.read_rows(path)


def test_hero_rejects_duplicate_raw_sample_identity(tmp_path, monkeypatch):
    common = {
        "model": "dm_pendulum",
        "batch": 1,
        "horizon": 32,
        "dtype": "float64",
        "dof": 1,
        "sample": 0,
    }
    forward = _artifact(
        tmp_path / "forward.csv",
        [
            {"backend": "python_ref", **common, "elapsed_ms": 2.0},
            {"backend": "python_ref", **common, "elapsed_ms": 2.1},
            {"backend": "mujoco_rollout", **common, "elapsed_ms": 1.0},
        ],
        cpu="Test CPU",
    )
    backward = _artifact(
        tmp_path / "backward.csv",
        [
            {"backend": "python_ref", **common, "elapsed_ms": 3.0},
            {"backend": "cpp_vjp", **common, "elapsed_ms": 1.5},
        ],
        cpu="Test CPU",
    )

    with pytest.raises(ValueError, match="duplicate"):
        _run(
            monkeypatch,
            plot_dm_control,
            "--forward",
            forward,
            "--backward",
            backward,
            "--batch",
            1,
        )


def test_accuracy_labels_reference_and_transition_epsilons(tmp_path, monkeypatch):
    data = _artifact(
        tmp_path / "components.csv",
        [{
            "model": "dm_pendulum", "seed": 0, "horizon": 1,
            "reference_eps": 1e-6, "transition_eps": 1e-8,
            "component": 0, "transition_fd_vjp": 1.0, "whole_rollout_fd": 1.0,
        }],
    )

    fig = _run(monkeypatch, plot_accuracy, "--data", data)

    subtitle = fig.axes[0].get_title()
    assert "reference eps = 1e-06" in subtitle
    assert "transition eps = 1e-08" in subtitle


def test_integrator_discloses_zero_error_plotting_floor(tmp_path, monkeypatch):
    data = _artifact(
        tmp_path / "integrator.csv",
        [
            {"model": "dm_pendulum", "dof": 1, "integrator": integrator,
             "dt": 0.01, "horizon": 5, "reference_eps": 1e-6,
             "transition_eps": 1e-8, "grad_rel_error": 0.0}
            for integrator in ("Euler", "implicit", "implicitfast")
        ],
    )

    fig = _run(monkeypatch, plot_integrator_ablation, "--data", data)

    text = " ".join(item.get_text() for item in fig.axes[0].texts)
    assert "zeros shown at 1e-16" in text
    subtitle = fig.axes[0].get_title()
    assert "reference eps = 1e-06" in subtitle
    assert "transition eps = 1e-08" in subtitle
