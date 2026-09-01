# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""S7.1 seam conformance: the handler dict merges over registrations() before any
ffi.register_ffi_target call, and an empty dict changes nothing."""
import subprocess
import sys

def test_seam_override_reaches_registration():
    code = r"""
import os
# bootstrap with no framework so the package init does NOT import transformer_engine.jax
# before the spy is installed (registration is an import-time event).
os.environ["NVTE_FRAMEWORK"] = "none"
import unittest.mock as m
import transformer_engine.te_rocm.jax_handlers as jh
jh.register_override("te_dequantize_ffi", "OVERRIDE-SENTINEL")
calls = {}
def spy(name, value, platform=None):
    calls[name] = value
import jax
with m.patch.object(jax.ffi, "register_ffi_target", side_effect=spy):
    import transformer_engine.jax  # triggers the registration loop
assert calls, "registration loop never ran under the spy"
assert calls.get("te_dequantize_ffi") == "OVERRIDE-SENTINEL", (
    "override did not reach register_ffi_target: %r" % (calls.get("te_dequantize_ffi"),))
others = [k for k, v in calls.items() if v == "OVERRIDE-SENTINEL" and k != "te_dequantize_ffi"]
assert not others, "override leaked onto other names: %r" % others
print("SEAM-OK", len(calls))
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**__import__("os").environ, "NVTE_FRAMEWORK": "none"})
    assert "SEAM-OK" in r.stdout, f"stdout={r.stdout[-500:]} stderr={r.stderr[-800:]}"
