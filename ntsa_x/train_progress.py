"""Optional, exact completed-training-step telemetry for native NTSA-X trainers.

Per-event file writes are deliberately infrequent and default to disabled.
The event stream is observational only: it never mutates weights, changes
validation or selects model checkpoints.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path


def _write(path: Path, message: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n")


def open_progress(path: str | Path | None, *, total_steps: int) -> Path | None:
    """Create a new progress JSONL stream without overwriting prior evidence."""
    if path is None:
        return None
    if not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError("Expected positive total optimizer updates")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8"):
        pass
    emit_progress(target, "train_started", step=0, total_steps=total_steps, epoch=0)
    return target


def emit_progress(path: Path | None, kind: str, *, step: int,
                  total_steps: int, epoch: int, validation_metric: float | None = None,
                  every: int = 8) -> None:
    """Report only work that really occurred and finite validation results."""
    if path is None:
        return
    if (kind not in ("train_started", "optimizer_step", "epoch_complete") or
            type(step) is not int or type(total_steps) is not int or
            not 0 <= step <= total_steps or total_steps < 1 or
            type(epoch) is not int or epoch < 0 or
            type(every) is not int or not 1 <= every <= 10000):
        raise ValueError("Invalid training telemetry event")
    if kind == "optimizer_step" and step % every and step != total_steps:
        return
    if kind == "epoch_complete":
        if (type(validation_metric) not in (int, float) or
                not math.isfinite(float(validation_metric))):
            raise ValueError("Epoch needs finite measured held-out validation metric")
    message = {"schema": "kindred-native-training-progress-v1",
               "type": kind, "step": step, "total_steps": total_steps,
               "epoch": epoch, "timestamp": datetime.now(timezone.utc).isoformat()}
    if kind == "epoch_complete":
        message["validation_metric"] = float(validation_metric)
    _write(path, message)
