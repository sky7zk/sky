"""Shared checkpoint loading and inference helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def load_inference_weights(model, state: dict, *, use_raw: bool = False) -> str:
    """Load EMA weights by default and return the selected weight variant."""

    if "model" not in state:
        model.load_state_dict(state)
        return "standalone"
    if state.get("weights_variant") == "ema_export":
        model.load_state_dict(state["model"])
        return "ema_export"
    if not use_raw and "ema" in state:
        model.load_state_dict(state["ema"]["shadow"])
        return "ema"
    model.load_state_dict(state["model"])
    return "raw"


def load_checkpoint_for_inference(
    model,
    checkpoint: str | Path,
    *,
    device: str = "cuda",
    use_raw: bool = False,
) -> tuple[dict, str]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    variant = load_inference_weights(model, state, use_raw=use_raw)
    return state, variant


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
