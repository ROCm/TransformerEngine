# This file was modified for portability to AMDGPU
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for Triton-based custom calls in TE JAX."""

import jax
import jax.numpy as jnp
import pytest

from utils import assert_allclose, pytest_parametrize_wrapper, require_triton_or_skip_test_file

require_triton_or_skip_test_file()

# require_triton_or_skip_test_file only checks the JAX version, so guard triton itself.
pytest.importorskip("triton")

import triton
import triton.language as tl

from transformer_engine.jax.cpp_extensions.base import BasePrimitive, register_primitive
from transformer_engine.jax.triton_extensions import triton_call_lowering

# Gluon support is optional and warp-size explicit. HAS_GLUON gates the Gluon
# test below so the Triton test still runs when Gluon is unavailable.
from typing import TYPE_CHECKING

GLUON_SKIP_REASON = ""

if TYPE_CHECKING:
    # Resolve Gluon symbols for the type checker; the runtime guard is below.
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    HAS_GLUON = True
    WARP_SIZE = 64
else:
    try:
        from packaging.version import Version

        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl

        WARP_SIZE = triton.runtime.driver.active.get_current_target().warp_size
        # Gluon-on-ROCm needs the make_ir fix shipped after Triton 3.4.0. Compare
        # the base release so a "+rocm.git" local segment can't slip past.
        HAS_GLUON = Version(triton.__version__).release > (3, 4, 0)
        if not HAS_GLUON:
            GLUON_SKIP_REASON = f"Gluon needs Triton > 3.4.0, found {triton.__version__}"
    except (ImportError, AttributeError, RuntimeError) as e:
        # Anything else is a real breakage, not a missing Gluon.
        gluon = gl = None
        WARP_SIZE = None
        HAS_GLUON = False
        GLUON_SKIP_REASON = f"Gluon unavailable: {type(e).__name__}: {e}"

requires_gluon = pytest.mark.skipif(not HAS_GLUON, reason=GLUON_SKIP_REASON or "Gluon unavailable")


@pytest.fixture(autouse=True, scope="module")
def init():
    """WAR for CUDA uninitialize error"""
    _ = jnp.zeros(0)
    yield


@pytest.mark.triton
class TestTritonBinding:
    """Test Triton binding primitive."""

    # Define autotuned Triton kernel
    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}),  # Uses defaults: num_warps=4, num_stages=3
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8),  # Custom num_warps
        ],
        key=["n_elements"],  # Autotune based on input size
    )
    @triton.jit
    def amax_kernel(
        x_ptr,
        amax_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute amax using Triton with autotuning."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        abs_x = tl.abs(x)
        block_max = tl.max(abs_x)

        tl.atomic_max(amax_ptr, block_max)

    # Define test primitive
    class AmaxTritonPrimitive(BasePrimitive):
        """Test primitive using Triton kernel."""

        name = "te_amax_triton_test"
        multiple_results = False
        impl_static_args = ()

        @staticmethod
        def abstract(x_aval):
            return jax.core.ShapedArray((1,), jnp.float32)

        @staticmethod
        def impl(x):
            assert TestTritonBinding.AmaxTritonPrimitive.inner_primitive is not None
            return TestTritonBinding.AmaxTritonPrimitive.inner_primitive.bind(x)

        @staticmethod
        def lowering(ctx, x):
            """MLIR lowering using Triton kernel."""
            n_elements = 1
            for dim in ctx.avals_in[0].shape:
                n_elements *= dim

            # For autotuned kernels, use the minimum BLOCK_SIZE from configs
            # to ensure all elements are processed by all configs
            block_size = min(
                config.kwargs.get("BLOCK_SIZE") for config in TestTritonBinding.amax_kernel.configs
            )
            grid = (triton.cdiv(n_elements, block_size),)

            return triton_call_lowering(
                ctx,
                TestTritonBinding.amax_kernel,  # Autotuned kernel
                x,
                grid=grid,
                constexprs={"n_elements": n_elements},
                # BLOCK_SIZE comes from autotuner config, not passed here
            )

    register_primitive(AmaxTritonPrimitive)

    @staticmethod
    def _triton_amax(x: jnp.ndarray) -> jnp.ndarray:
        """Compute amax using Triton kernel."""
        return TestTritonBinding.AmaxTritonPrimitive.outer_primitive.bind(x)

    @pytest_parametrize_wrapper("shape", [(1024, 1024)])
    @pytest_parametrize_wrapper("dtype", [jnp.bfloat16])
    def test_triton_amax(self, shape, dtype):
        """Test Triton amax with JIT."""
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, shape, dtype)

        expected = jnp.max(jnp.abs(x), keepdims=False).astype(jnp.float32)
        jitted_amax = jax.jit(self._triton_amax)
        result = jitted_amax(x)

        assert_allclose(result, expected, dtype=jnp.float32)


# Gluon binding tests. Drive a @gluon.jit kernel through the same
# triton_call_lowering bridge as the Triton test above.
if HAS_GLUON:
    # BLOCK_SIZE must divide NUM_WARPS * WARP_SIZE so the 1-D layout tiles exactly.
    BLOCK_SIZE = 1024
    NUM_WARPS = 4

    # Elementwise out = x * 2, shared by the jit and autotuned tests. Gluon is
    # layout-explicit: `arange` needs a BlockedLayout whose warps_per_cta matches
    # the launch num_warps.
    @gluon.jit
    def _double_kernel(
        x_ptr,
        out_ptr,
        n_elements: gl.constexpr,
        BLOCK_SIZE: gl.constexpr,
        NUM_WARPS: gl.constexpr,
        WARP_SIZE: gl.constexpr,
    ):
        """Multiply each element by 2 using Gluon."""
        SIZE_PER_THREAD: gl.constexpr = BLOCK_SIZE // (NUM_WARPS * WARP_SIZE)
        layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[SIZE_PER_THREAD],
            threads_per_warp=[WARP_SIZE],
            warps_per_cta=[NUM_WARPS],
            order=[0],
        )

        pid = gl.program_id(0)
        offsets = pid * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
        mask = offsets < n_elements
        x = gl.load(x_ptr + offsets, mask)
        gl.store(out_ptr + offsets, x * 2.0, mask)

else:
    _double_kernel = None


@pytest.mark.triton
@requires_gluon
class TestGluonBinding:
    """Test Gluon binding primitive through the Triton custom-call bridge."""

    # Define test primitive
    class DoubleGluonPrimitive(BasePrimitive):
        """Test primitive using a Gluon kernel."""

        name = "te_double_gluon_test"
        multiple_results = False
        impl_static_args = ()

        @staticmethod
        def abstract(x_aval):
            return jax.core.ShapedArray(x_aval.shape, x_aval.dtype)

        @staticmethod
        def impl(x):
            assert TestGluonBinding.DoubleGluonPrimitive.inner_primitive is not None
            return TestGluonBinding.DoubleGluonPrimitive.inner_primitive.bind(x)

        @staticmethod
        def lowering(ctx, x):
            """MLIR lowering using the Gluon kernel."""
            n_elements = 1
            for dim in ctx.avals_in[0].shape:
                n_elements *= dim

            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

            return triton_call_lowering(
                ctx,
                _double_kernel,
                x,
                grid=grid,
                constexprs={
                    "n_elements": n_elements,
                    "BLOCK_SIZE": BLOCK_SIZE,
                    "NUM_WARPS": NUM_WARPS,
                    "WARP_SIZE": WARP_SIZE,
                },
                num_warps=NUM_WARPS,  # must match warps_per_cta in the layout
            )

    register_primitive(DoubleGluonPrimitive)

    @staticmethod
    def _gluon_double(x: jnp.ndarray) -> jnp.ndarray:
        """Double each element using the Gluon kernel."""
        return TestGluonBinding.DoubleGluonPrimitive.outer_primitive.bind(x)

    @pytest_parametrize_wrapper("shape", [(1024, 1024), (1000, 1000)])
    @pytest_parametrize_wrapper("dtype", [jnp.float32])
    def test_gluon_double(self, shape, dtype):
        """Test the Gluon double kernel with JIT."""
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, shape, dtype)

        expected = (x * 2.0).astype(dtype)
        jitted_double = jax.jit(self._gluon_double)
        result = jitted_double(x)

        assert_allclose(result, expected, dtype=dtype)


@pytest.mark.triton
@requires_gluon
class TestGluonAutotunedBinding:
    """Autotuned Gluon binding; exercises the TritonAutotunedKernelCall path."""

    # Reuse the shared kernel; each config carries BLOCK_SIZE and NUM_WARPS as
    # constexprs (the layout needs them) with a matching launch num_warps.
    double_kernel_autotuned = (
        triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 256, "NUM_WARPS": 4}, num_warps=4),
                triton.Config({"BLOCK_SIZE": 1024, "NUM_WARPS": 8}, num_warps=8),
            ],
            key=["n_elements"],
        )(_double_kernel)
        if HAS_GLUON
        else None
    )

    class DoubleGluonAutotunedPrimitive(BasePrimitive):
        """Test primitive using an autotuned Gluon kernel."""

        name = "te_double_gluon_autotuned_test"
        multiple_results = False
        impl_static_args = ()

        @staticmethod
        def abstract(x_aval):
            return jax.core.ShapedArray(x_aval.shape, x_aval.dtype)

        @staticmethod
        def impl(x):
            prim = TestGluonAutotunedBinding.DoubleGluonAutotunedPrimitive
            assert prim.inner_primitive is not None
            return prim.inner_primitive.bind(x)

        @staticmethod
        def lowering(ctx, x):
            """MLIR lowering using the autotuned Gluon kernel."""
            n_elements = 1
            for dim in ctx.avals_in[0].shape:
                n_elements *= dim

            kernel = TestGluonAutotunedBinding.double_kernel_autotuned
            # Smallest BLOCK_SIZE so every config's grid covers all elements.
            block_size = min(c.kwargs.get("BLOCK_SIZE") for c in kernel.configs)
            grid = (triton.cdiv(n_elements, block_size),)

            return triton_call_lowering(
                ctx,
                kernel,
                x,
                grid=grid,
                # BLOCK_SIZE/NUM_WARPS come from each autotune config; only
                # n_elements and WARP_SIZE are passed here.
                constexprs={"n_elements": n_elements, "WARP_SIZE": WARP_SIZE},
            )

    register_primitive(DoubleGluonAutotunedPrimitive)

    @staticmethod
    def _gluon_double_autotuned(x: jnp.ndarray) -> jnp.ndarray:
        """Double each element using the autotuned Gluon kernel."""
        prim = TestGluonAutotunedBinding.DoubleGluonAutotunedPrimitive
        return prim.outer_primitive.bind(x)

    @pytest_parametrize_wrapper("shape", [(1024, 1024), (1000, 1000)])
    @pytest_parametrize_wrapper("dtype", [jnp.float32])
    def test_gluon_double_autotuned(self, shape, dtype):
        """Test the autotuned Gluon double kernel with JIT."""
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, shape, dtype)

        expected = (x * 2.0).astype(dtype)
        result = jax.jit(self._gluon_double_autotuned)(x)

        assert_allclose(result, expected, dtype=dtype)
