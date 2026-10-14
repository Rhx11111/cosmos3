# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Load a diffusers-layout Cosmos3 checkpoint into an unsharded network."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from cosmos_framework.inference.model import (
    _diffusers_files_to_keys,
    _diffusers_to_net_key,
    _diffusers_weight_map,
    _is_diffusers_model_weight_path,
)

# Task-specialized checkpoints omit the generative heads they do not use; a
# checkpoint without any tensor of a head leaves that head at its initialization,
# the same tolerance the DCP load planner applies.
_OPTIONAL_HEAD_PREFIXES = (
    ("action2llm.", "llm2action.", "action_modality_embed"),
    ("sound2llm.", "llm2sound.", "sound_modality_embed"),
)
_VISUAL_PREFIX = "language_model.visual."
_VISION_ENCODER_DIR = "vision_encoder/"


def load_diffusers_net_weights(net: torch.nn.Module, checkpoint_dir: Path) -> None:
    """Copy every mapped tensor of a diffusers checkpoint into ``net`` in place.

    Reads the safetensors shards directly, so it needs neither a process group nor
    the torch 2.8 DCP Hugging Face storage reader.

    Args:
        net: Unsharded ``Cosmos3VFMNetwork`` whose state dict receives the weights.
        checkpoint_dir: Diffusers layout directory with ``model_index.json`` and the
            root or ``transformer/`` safetensors index.

    Raises:
        KeyError: A model tensor in the checkpoint has no network counterpart.
        ValueError: A tensor shape disagrees with the network or a required tensor is missing.
    """
    checkpoint_dir = Path(checkpoint_dir)
    state = net.state_dict()
    weight_map = _diffusers_weight_map(checkpoint_dir)
    has_vision_weights = any(path.startswith(_VISION_ENCODER_DIR) for path in weight_map.values())
    has_visual_target = any(key.startswith(_VISUAL_PREFIX) for key in state)
    loaded: set[str] = set()
    for relative_path, keys in sorted(_diffusers_files_to_keys(weight_map).items()):
        with safe_open(str(checkpoint_dir / relative_path), framework="pt", device="cpu") as shard:
            for key in sorted(keys):
                net_key = _diffusers_to_net_key(key, relative_path)
                if net_key is None:
                    if _is_diffusers_model_weight_path(relative_path) and not (
                        relative_path.startswith(_VISION_ENCODER_DIR) and not has_visual_target
                    ):
                        raise KeyError(f"Diffusers model key {key!r} from {relative_path!r} has no Cosmos3 mapping.")
                    continue
                target = state.get(net_key)
                if target is None:
                    continue
                if net_key in loaded:
                    raise KeyError(f"Multiple diffusers keys map to target model key {net_key!r}.")
                tensor = shard.get_tensor(key)
                if tuple(tensor.shape) != tuple(target.shape):
                    raise ValueError(
                        f"{key!r} has shape {tuple(tensor.shape)} but {net_key!r} expects {tuple(target.shape)}."
                    )
                with torch.no_grad():
                    target.copy_(tensor.to(dtype=target.dtype))
                loaded.add(net_key)

    missing = set(state) - loaded
    if not has_vision_weights:
        missing = {key for key in missing if not key.startswith(_VISUAL_PREFIX)}
    for prefixes in _OPTIONAL_HEAD_PREFIXES:
        if not any(key.startswith(prefixes) for key in loaded):
            missing = {key for key in missing if not key.startswith(prefixes)}
    if missing:
        raise ValueError(
            f"Diffusers checkpoint at {checkpoint_dir} did not provide {len(missing)} required model "
            f"tensor(s). First up to 10: {sorted(missing)[:10]}"
        )
