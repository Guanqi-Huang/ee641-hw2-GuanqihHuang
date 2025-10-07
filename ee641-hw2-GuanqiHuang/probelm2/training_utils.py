# problem2/training_utils.py
# KL and temperature annealing schedules, logging helpers.
# ASCII only to avoid unicode parse errors.

from __future__ import annotations
import math
import json
from pathlib import Path
from typing import Dict, Any

def kl_annealing_schedule(epoch: int, method: str = "cyclical",
                          total_epochs: int = 100, cycles: int = 4,
                          max_beta: float = 1.0) -> float:
    """
    Returns beta
    'linear': linearly increases from 0 to max_beta over total_epochs.
    'cyclical': cycles warm-up ramps, helpful to reduce posterior collapse.
    """
    if method == "linear":
        t = min(epoch / max(1, total_epochs), 1.0)
        return max_beta * t
    if method == "cyclical":
        cycle_len = max(1, total_epochs // cycles)
        pos = epoch % cycle_len
        return max_beta * min(pos / cycle_len, 1.0)
    # fallback
    return max_beta

def temperature_annealing_schedule(epoch: int,
                          start_temp: float = 1.5,
                          end_temp: float = 0.5,
                          total_epochs: int = 100) -> float:
    if total_epochs <= 1:
        return end_temp
    r = epoch / (total_epochs - 1)
    return end_temp + (start_temp - end_temp) * (1.0 - r)**2

def save_json(data: Dict[str, Any], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
