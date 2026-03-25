"""Helpers for compatibility with existing multicore `.npz` result layout."""

from __future__ import annotations

import numpy as np


def key_prefix(p_value: float, run_idx: int) -> str:
    """Return legacy key prefix used by production data files."""
    return f"p_{p_value}_run_{run_idx}"


def pack_run_result(p_value: float, run_idx: int, lambdas: np.ndarray, vectors: np.ndarray) -> dict:
    """Pack one run using the exact historical key naming convention."""
    prefix = key_prefix(p_value, run_idx)
    return {
        f"{prefix}_lambdas": lambdas,
        f"{prefix}_vectors": vectors,
        f"{prefix}_p_value": p_value,
    }

