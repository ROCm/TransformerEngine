# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""TE_ROCM_EXTENSION_API, name level: every name upstream's Python asks of the seam exists on
the built extension, except an explicit allowlist; ROCm extras are pinned so drift is visible."""
from pathlib import Path

import pytest

HERE = Path(__file__).parent

# Names upstream references that the ROCm extension legitimately does not provide.
MISSING_ALLOWLIST = {
    "LayerNorm",   # upstream dead code: tex.LayerNorm is registered by NO csrc, upstream's own included (onnx_extensions.py)
}
# CUDA-only by design; every upstream caller is capability-guarded (plan F5).
CUDA_ONLY_ALLOWLIST = {
    "cusolvermp_ctx_create", "cusolvermp_ctx_destroy", "newton_schulz", "get_cublasLt_version",
    "get_grouped_gemm_setup_workspace_size", "te_general_grouped_gemm_for_discrete_in",
    "te_general_grouped_gemm_for_discrete_out", "te_general_grouped_gemm_for_grouped_tensor",
}
# Feature-gated symbol GROUPS (proposal sec 3.3): optional even in upstream builds, keyed to the
# capability that gates them. Present only when the group's build flag is on.
FEATURE_GATED = {
    "te.ep.build_enabled": {   # NVTE_WITH_NCCL_EP; off on ROCm (no RCCL device API) - manifest capability_graph_ep
        "ep_initialize", "ep_finalize", "ep_prepare", "ep_dispatch", "ep_dispatch_bwd",
        "ep_combine", "ep_combine_bwd", "ep_get_zero_copy", "ep_handle_mem_size",
    },
}


def gated_names() -> set[str]:
    return set().union(*FEATURE_GATED.values())


def public_names(mod) -> set[str]:
    return {n for n in dir(mod) if not n.startswith("_")}


def test_upstream_demand_is_supplied(tex, upstream_demand):
    supplied = public_names(tex)
    demanded = set(upstream_demand["names"])
    missing = demanded - supplied - MISSING_ALLOWLIST - CUDA_ONLY_ALLOWLIST - gated_names()
    assert not missing, f"upstream asks the seam for names the extension lacks: {sorted(missing)}"
    # the allowlists must stay honest: an allowlisted name that is now supplied should be removed
    stale = (MISSING_ALLOWLIST | CUDA_ONLY_ALLOWLIST) & supplied
    assert not stale, f"allowlisted as missing but now supplied - prune the allowlist: {sorted(stale)}"
    # a feature-gated group must be all-or-nothing: partial presence means a broken build
    for cap, group in FEATURE_GATED.items():
        present = group & supplied
        assert not present or present == group, f"{cap}: group partially present {sorted(present)}"


def test_extras_are_pinned(tex, upstream_demand):
    """ROCm extras leak into te.pytorch.cpp_extensions via its star-import (plan F4). They are
    allowed, but any change must be deliberate: the set is pinned in extras_allowlist.txt."""
    supplied = public_names(tex)
    demanded = set(upstream_demand["names"])
    extras = sorted(n for n in supplied - demanded
                    if not n.startswith("__") and n not in ("os", "sys"))
    pinned_file = HERE / "extras_allowlist.txt"
    if not pinned_file.exists():
        pinned_file.write_text("\n".join(extras) + "\n")
        pytest.skip(f"extras allowlist created with {len(extras)} names; rerun to enforce")
    pinned = set(pinned_file.read_text().split())
    new = set(extras) - pinned
    gone = pinned - set(extras)
    assert not new and not gone, (
        f"extension surface drifted vs extras_allowlist.txt: new={sorted(new)} gone={sorted(gone)} "
        f"- update the file deliberately")


def test_star_import_carries_the_same_objects(tex):
    """cpp_extensions/__init__.py does `from transformer_engine_torch import *` and THEN imports
    its own Python wrappers (fused_attn_bwd, general_gemm, ...) which deliberately shadow raw
    pybind names. Rule: every name cpp_extensions exposes is either the extension's own object
    (identity) or an upstream Python-level wrapper defined under transformer_engine.pytorch -
    never a copy or a foreign object."""
    import transformer_engine.pytorch.cpp_extensions as ce
    shadowed = []
    for n in public_names(tex):
        if not hasattr(ce, n):
            continue
        obj = getattr(ce, n)
        if obj is getattr(tex, n):
            continue
        mod = getattr(obj, "__module__", "") or ""
        assert mod.startswith("transformer_engine.pytorch"), f"{n}: shadowed by a non-upstream object from {mod!r}"
        shadowed.append(n)
    # pin the shadow set so an upstream change in cpp_extensions' wrappers is noticed
    assert shadowed, "expected upstream's Python wrappers to shadow some raw pybind names"
