# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Attention Subpackage:
Unified implementation for all Attention implementations.

Flash Attention v3 (flash3) Backend
"""

import importlib

import torch

from cosmos_framework.model.attention.utils.safe_ops import log
from cosmos_framework.model.attention.utils.version import version_at_least

# Flash Attention v3 builds with the same interface, in order of preference: NVIDIA's
# cosmos-dependencies build, then the upstream build on the PyTorch wheel index (abi3, so it
# also serves PyTorch releases the NVIDIA build does not cover).
FLASH3_PACKAGES = (("flash_attn_3_nv", "1.0.3"), ("flash_attn_3", "3.0.1"))


def find_flash3_package() -> str | None:
    """
    Returns the name of the first usable Flash Attention v3 package, or None.
    Requirements are:
        * Presence of CUDA Runtime (via PyTorch)
        * Presence of a Flash Attention v3 package, meeting its minimum version requirement

    This check guards imports / dependencies on the Flash Attention package.
    """
    if not torch.cuda.is_available():
        log.debug("Flash Attention v3 is not supported because PyTorch did not detect CUDA runtime.")
        return None

    for name, min_version in FLASH3_PACKAGES:
        try:
            package = importlib.import_module(name)
        except ImportError:
            log.debug(f"Flash Attention v3 package {name!r} was not found.")
            continue
        except Exception as e:
            log.debug(f"Importing the Flash Attention v3 package {name!r} failed: {e}")
            continue
        if not version_at_least(package.__version__, min_version):
            log.debug(
                f"Flash Attention v3 ({name!r}) build is not supported; minimum required version is "
                f"{min_version}, got {package.__version__}."
            )
            continue
        return name

    log.debug("Flash Attention v3 is not supported because no supported package was found.")
    return None


FLASH3_PACKAGE = find_flash3_package()
FLASH3_SUPPORTED = FLASH3_PACKAGE is not None


if FLASH3_SUPPORTED:
    from cosmos_framework.model.attention.flash3.functions import flash3_attention

else:
    from cosmos_framework.model.attention.flash3.stubs import flash3_attention

__all__ = ["flash3_attention", "FLASH3_PACKAGE", "FLASH3_SUPPORTED"]
