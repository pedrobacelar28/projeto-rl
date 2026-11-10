"""Torch device helpers."""

from __future__ import annotations

import os

import torch


def get_torch_device() -> torch.device:
    requested = os.environ.get("PROJETO_RL_DEVICE", "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("PROJETO_RL_DEVICE requests CUDA but torch.cuda.is_available() is false")
        return torch.device("cuda")
    raise ValueError("PROJETO_RL_DEVICE must be one of auto, cpu, cuda, or gpu")


def as_device_string(device: torch.device | str) -> str:
    return str(device)
