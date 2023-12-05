from functools import lru_cache as _lru_cache

import torch
from ...library import Library as _Library

__all__ = ["is_built", "is_available", "is_macos13_or_newer"]


def is_built() -> bool:
    r"""Returns whether PyTorch is built with APU support. Note that this
    doesn't necessarily mean APU is available; just that if this PyTorch
    binary were run a machine with working APU drivers and devices, we
    would be able to use it."""
    return True
    #GW return torch._C._has_mps


@_lru_cache
def is_available() -> bool:
    r"""Returns a bool indicating if APU is currently available."""
    return True
    #GW return torch._C._mps_is_available()


_lib = None


def _init():
    r"""Register prims as implementation of var_mean and group_norm"""
    global _lib
    if is_built() is False or _lib is not None:
        return
    from ..._decomp.decompositions import (
        native_group_norm_backward as _native_group_norm_backward,
    )
    from ..._refs import native_group_norm as _native_group_norm, var_mean as _var_mean

    print("PTORCH", "getting aten lib")
    _lib = _Library("aten", "IMPL")
    print("PYTORCH", "replacing stuff")
    _lib.impl("var_mean.correction", _var_mean, "APU")
    _lib.impl("native_group_norm", _native_group_norm, "APU")
    _lib.impl("native_group_norm_backward", _native_group_norm_backward, "APU")
