"""Write compact reproducibility metadata beside benchmark CSV files."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_METADATA_TIMEOUT = 5
_RESERVED_FIELDS = frozenset(
    {
        "schema_version",
        "command",
        "cwd",
        "git_commit",
        "git_dirty",
        "timestamp_utc",
        "python",
        "packages",
        "os",
        "cpu",
        "compiler",
        "data_file",
        "data_sha256",
    }
)


def _csv_path(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix != ".csv":
        raise ValueError(f"benchmark output must use a .csv suffix: {path}")
    return path


def _validate_experiment(experiment: dict[str, Any]) -> None:
    collisions = sorted(_RESERVED_FIELDS.intersection(experiment))
    if collisions:
        raise ValueError(f"reserved manifest fields: {', '.join(collisions)}")
    json.dumps(experiment, allow_nan=False, sort_keys=True)


def _output(command: list[str]) -> str | None:
    try:
        return (
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=_METADATA_TIMEOUT,
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cpu() -> str:
    if sys.platform == "linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        identity = _output(["sysctl", "-n", "machdep.cpu.brand_string"])
        if identity:
            return identity
    return platform.processor() or platform.machine()


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            timeout=_METADATA_TIMEOUT,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return bool(result.stdout)


def _context() -> dict[str, Any]:
    compiler = _output([*(shlex.split(os.environ.get("CXX", "")) or ["c++"]), "--version"])
    return {
        "schema_version": 1,
        "command": shlex.join(["python", *sys.argv]),
        "cwd": str(Path.cwd()),
        "git_commit": _output(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"]),
        "git_dirty": _git_dirty(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "packages": {
            name: _version(name)
            for name in ("diffmjstep", "torch", "mujoco", "numpy", "dm-control")
        },
        "os": platform.platform(),
        "cpu": _cpu(),
        "compiler": compiler.splitlines()[0] if compiler else None,
    }


def _write_csv_and_manifest(
    csv_path: str | Path, csv_bytes: bytes, **experiment: Any
) -> Path:
    csv_path = _csv_path(csv_path)
    _validate_experiment(experiment)
    manifest_path = csv_path.with_suffix(".json")
    payload = {
        **experiment,
        **_context(),
        "data_file": csv_path.name,
        "data_sha256": hashlib.sha256(csv_bytes).hexdigest(),
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(csv_bytes)
    manifest_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path
