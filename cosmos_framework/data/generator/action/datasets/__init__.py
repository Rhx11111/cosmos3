# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Action dataset wrappers for Cosmos Action.

All concrete datasets inherit from :class:`ActionBaseDataset` and expose a
``load_action_stats()`` classmethod for retrieving pre-computed normalization
statistics without instantiating the dataset.
"""

import importlib

# Imported on first access: most of these datasets need the optional ``lerobot`` package,
# which importing a sibling module (for example ``action_sft_dataset``) must not require.
_LAZY_EXPORTS = {
    "AgiBotWorldBetaLeRobotDataset": "agibotworld_beta_lerobot_dataset",
    "ActionBaseDataset": "base_dataset",
    "BridgeOrigLeRobotDataset": "bridge_orig_lerobot_dataset",
    "DROIDMergedLeRobotDataset": "droid_merged_lerobot_dataset",
    "DROIDLeRobotDataset": "droid_lerobot_dataset",
    "FractalLeRobotDataset": "fractal_lerobot_dataset",
    "HumanHandPoseLeRobotDataset": "human_hand_pose_lerobot_dataset",
    "RoboCasaLeRobotDataset": "robocasa_lerobot_dataset",
    "LIBEROLeRobotDataset": "libero_lerobot_dataset",
    "RoboMINDFrankaDataset": "robomind_franka_dataset",
    "RoboMINDURDataset": "robomind_ur_dataset",
    "UMILeRobotDataset": "umi_lerobot_dataset",
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{_LAZY_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


__all__ = [
    "ActionBaseDataset",
    "AgiBotWorldBetaLeRobotDataset",
    "BridgeOrigLeRobotDataset",
    "DROIDLeRobotDataset",
    "DROIDMergedLeRobotDataset",
    "FractalLeRobotDataset",
    "HumanHandPoseLeRobotDataset",
    "RoboCasaLeRobotDataset",
    "LIBEROLeRobotDataset",
    "RoboMINDFrankaDataset",
    "RoboMINDURDataset",
    "UMILeRobotDataset",
]
