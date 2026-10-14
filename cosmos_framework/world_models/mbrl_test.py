# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

from cosmos_framework.world_models.contract import Cosmos3WorldModelContract
from cosmos_framework.world_models.mbrl import edge_model_config


def test_relative_asset_paths_resolve_against_the_caller(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = edge_model_config("Cosmos3-Edge", Path("vae/Wan2.2_VAE.pth"), Cosmos3WorldModelContract())["config"]

    assert config["tokenizer"]["vae_path"] == str((tmp_path / "vae" / "Wan2.2_VAE.pth").resolve())
    assert config["vlm_config"]["tokenizer"]["tokenizer_type"] == str((tmp_path / "Cosmos3-Edge").resolve())
