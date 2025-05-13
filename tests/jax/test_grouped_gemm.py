import pytest
import jax
import jax.numpy as jnp
import numpy as np
from transformer_engine.jax import cpp_extensions as tex
import math
from typing import Dict, Optional, Union, List, Tuple

# Define DType type for type annotations
DType = jnp.dtype

def assert_allclose(a, b, dtype: Union[DType, "TEDType", np.dtype], reference_value: float = 1.0) -> bool:
    """Assert that two arrays are close enough with proper dtype handling"""
    # Get tolerances based on dtype
    tols = dtype_tols(dtype, reference_value)
    rtol, atol = tols["rtol"], tols["atol"]

    # Convert both arrays to float32 for comparison to avoid dtype issues
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)

    # Let pytest handle the assertion and reporting
    np.testing.assert_allclose(a_np, b_np, rtol=rtol, atol=atol)
    return True

# Enum for Transformer Engine data types
class TEDType:
    kByte = 0
    kInt32 = 1
    kInt64 = 2
    kFloat32 = 3
    kFloat16 = 4
    kBFloat16 = 5
    kFloat8E4M3 = 6
    kFloat8E5M2 = 7

def dtype_tols(
    dtype: Union[DType, TEDType, np.dtype],
    reference_value: float = 1.0,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
) -> Dict[str, float]:
    """Expected numerical tolerance for a data type.

    Args:
      dtype: data type.
      reference_value: reference value (default: 1).
      rtol: override for relative tolerance estimate
      atol: override for absolute tolerance estimate

    Returns:
      Dictionary with "rtol" and "atol" as keys
    """

    # Return immediately if tolerances are fully specified
    if rtol is not None and atol is not None:
        return {"rtol": rtol, "atol": atol}

    # Convert to JAX dtype if needed
    if isinstance(dtype, TEDType):
        dtype = {
            TEDType.kByte: jnp.uint8,
            TEDType.kInt32: jnp.int32,
            TEDType.kInt64: jnp.int64,
            TEDType.kFloat32: jnp.float32,
            TEDType.kFloat16: jnp.float16,
            TEDType.kBFloat16: jnp.bfloat16,
            TEDType.kFloat8E4M3: jnp.float8_e4m3fn,
            TEDType.kFloat8E5M2: jnp.float8_e5m2,
        }[dtype]
    elif isinstance(dtype, np.dtype):
        dtype = jnp.dtype(dtype)

    # Expect bit-wise accuracy for integer dtypes
    if not jnp.issubdtype(dtype, jnp.floating):
        if rtol is None:
            rtol = 0.0
        if atol is None:
            atol = 0.0
        return {"rtol": rtol, "atol": atol}

    # Estimate floating-point error
    finfo = jnp.finfo(dtype)
    eps_relaxed = math.pow(finfo.eps, 2 / 3)
    with jax.default_device(jax.devices("cpu")[0]):
        if isinstance(reference_value, (float, int)):
            reference_value = jnp.array(reference_value, dtype=dtype)
        else:
            reference_value = reference_value.astype(dtype)
        spacing_high = jnp.nextafter(reference_value, finfo.max) - reference_value
        spacing_low = reference_value - jnp.nextafter(reference_value, finfo.min)
        ulp = max(spacing_high.item(), spacing_low.item())
    if rtol is None:
        rtol = eps_relaxed
    if atol is None:
        atol = max(ulp, eps_relaxed)
    return {"rtol": rtol, "atol": atol}

class TestGroupedGemm:
    @staticmethod
    def _ref_grouped_gemm_with_jnp_dot(lhs_list, rhs_list, contracting_dims_list):
        """Reference implementation of grouped GEMM using JAX's dot_general."""
        ref_out_list = []
        for lhs, rhs, contracting_dims in zip(lhs_list, rhs_list, contracting_dims_list):
            dim_nums = (contracting_dims, ((), ()))
            ref_out_list.append(jax.lax.dot_general(lhs, rhs, dim_nums))
        return ref_out_list

    @staticmethod
    def _generate_grouped_gemm_input(dtype, shape_list, layout_list):
        """Generate inputs for grouped GEMM tests."""
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, len(shape_list) * 2)

        lhs_list, rhs_list, contracting_dims_list = [], [], []
        for i, ((m, n, k), data_layout) in enumerate(zip(shape_list, layout_list)):
            lhs = jax.random.uniform(
                subkeys[2 * i],
                (m if data_layout[0] == "N" else k, k if data_layout[0] == "N" else m),
                dtype=dtype,
            )
            rhs = jax.random.uniform(
                subkeys[2 * i + 1],
                (k if data_layout[1] == "N" else n, n if data_layout[1] == "N" else k),
                dtype=dtype,
            )
            lhs_contracting_dim = (1,) if data_layout[0] == "N" else (0,)
            rhs_contracting_dim = (0,) if data_layout[1] == "N" else (1,)
            contracting_dims = (lhs_contracting_dim, rhs_contracting_dim)

            lhs_list.append(lhs)
            rhs_list.append(rhs)
            contracting_dims_list.append(contracting_dims)

        return lhs_list, rhs_list, contracting_dims_list

    def run_grouped_gemm_test(self, dtype, layout_list, shape_list=None):
        """Run grouped GEMM test with different configurations."""
        # Default shapes if none provided
        if shape_list is None:
            # Format: (m, n, k) for each GEMM in the group
            shape_list = [(128, 64, 32), (256, 128, 64), (512, 256, 128), (1024, 512, 256)]

        # Ensure layout_list matches shape_list length
        if len(layout_list) != len(shape_list):
            if len(layout_list) == 1:
                # Expand single layout to match shape_list length
                layout_list = layout_list * len(shape_list)
            else:
                raise ValueError(f"Layout list length ({len(layout_list)}) must match shape list length ({len(shape_list)})")

        lhs_list, rhs_list, contracting_dims_list = self._generate_grouped_gemm_input(
            dtype, shape_list, layout_list
        )

        # Run reference implementation
        ref_out = self._ref_grouped_gemm_with_jnp_dot(lhs_list, rhs_list, contracting_dims_list)

        # Run transformer_engine implementation
        primitive_out = tex.grouped_gemm(lhs_list, rhs_list, contracting_dims_list)

        # Verify results
        for i in range(len(shape_list)):
            # Use max value from output as reference for tolerance calculation
            max_val = float(np.max(np.abs(ref_out[i])))
            assert_allclose(primitive_out[i], ref_out[i], dtype=dtype, reference_value=max_val)

        return True

# Pytest fixture for test instance
@pytest.fixture
def test_instance():
    """Fixture to create a TestGroupedGemm instance."""
    return TestGroupedGemm()

# Parameterized tests for different configurations
@pytest.mark.parametrize("dtype,layouts", [
    (jnp.bfloat16, ["NN", "TN", "NT", "TT"]),
    (jnp.float16, ["NN", "TN", "NT", "TT"])
])
def test_grouped_gemm_multi_layout(test_instance, dtype, layouts):
    """Test grouped GEMM with multiple layouts."""
    test_instance.run_grouped_gemm_test(dtype, layouts)

@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_grouped_gemm_single_matrix(test_instance, dtype):
    """Test grouped GEMM with a single matrix."""
    single_shape = [(512, 256, 128)]
    single_layout = ["NN"]
    test_instance.run_grouped_gemm_test(dtype, single_layout, single_shape)

# More specialized test cases
@pytest.mark.parametrize("dtype,layout,shape", [
    (jnp.bfloat16, ["NN"], [(512, 256, 128)]),
    (jnp.float16, ["NN"], [(512, 256, 128)]),
    (jnp.bfloat16, ["TN"], [(256, 128, 64)]),
    (jnp.float16, ["NT"], [(128, 64, 32)])
])
def test_grouped_gemm_specific_configs(test_instance, dtype, layout, shape):
    """Test grouped GEMM with specific configurations."""
    test_instance.run_grouped_gemm_test(dtype, layout, shape)

# Test shapes with different sizes
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_grouped_gemm_varying_shapes(test_instance, dtype):
    """Test grouped GEMM with varying matrix shapes."""
    shapes = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]
    layouts = ["NN", "TN", "NT", "TT"]
    test_instance.run_grouped_gemm_test(dtype, layouts, shapes)
