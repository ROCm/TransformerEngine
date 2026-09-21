# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""ROCm fused comm+GEMM overlap tests"""
import os
import subprocess
from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions as tex

from torch.utils.cpp_extension import IS_HIP_EXTENSION
from transformer_engine.pytorch.utils import get_device_compute_capability


if torch.cuda.device_count() < 2:
    pytest.skip("Comm+GEMM overlap requires at least 2 GPUs.", allow_module_level=True)

fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)

RNG_SEED: int = 42
SEQ_LENGTH: int = 1024
BATCH_SIZE: int = 2
NUM_HEADS: int = 32
HEAD_DIM: int = 48

TEST_ROOT = Path(__file__).parent.resolve()


def _fused_shape_ok(nprocs: int) -> bool:
    """The minimum both backends need: a 256-aligned per-rank chunk and a 256-aligned N."""
    return (SEQ_LENGTH * BATCH_SIZE) % (256 * nprocs) == 0 and (NUM_HEADS * HEAD_DIM) % 256 == 0


FUSED_PROC_COUNTS = [n for n in (4, 8) if n <= torch.cuda.device_count() and _fused_shape_ok(n)]

fused_available = (
    IS_HIP_EXTENSION and get_device_compute_capability() == (9, 5) and len(FUSED_PROC_COUNTS) > 0
)
reason_for_no_fused = (
    "Fused comm+GEMM overlap requires a gfx950 device, tp_size in (4, 8) and a 256-aligned "
    "per-rank chunk."
)

# Both GEMM layouts the fused AG backend implements; NN is selected with the harness' --dgrad.
FUSED_LAYOUTS = ("TN", "NN")
FUSED_QUANTIZATIONS = ("none", "mxfp8")


def _fused_launch_cmd(nprocs: int):
    """Same form as LAUNCH_CMD, but at a rank count the fused tests choose."""
    if tex.ubuf_built_with_mpi():
        return ["mpirun", "-np", str(nprocs), "--oversubscribe", "--quiet", "python3"]
    return ["torchrun", f"--nproc_per_node={nprocs}"]


def _run_fused_ag(
    nprocs,
    bulk=False,
    quantization="none",
    layout="TN",
    num_heads=NUM_HEADS,
    head_dim=HEAD_DIM,
    extra_args=(),
    env_extra=None,
):
    """Run the AG overlap harness with the fused backend, returning the completed process."""
    test_cmd = _fused_launch_cmd(nprocs) + [
        str(TEST_ROOT / "run_gemm_with_overlap.py"),
        "--check-numerics",
        f"--seed={RNG_SEED}",
        f"--seq-length={SEQ_LENGTH}",
        f"--batch-size={BATCH_SIZE}",
        f"--num-heads={num_heads}",
        f"--head-dim={head_dim}",
        "--comm-type=AG",
        "--fused",
    ]
    test_cmd += list(extra_args)
    # The bulk harness pins its GEMM to NN regardless, so the layout only applies to the p2p path.
    test_cmd += (
        ["--bulk-overlap", f"--quantization={quantization}"]
        if bulk
        else ["--p2p", f"--quantization={quantization}"] + (["--dgrad"] if layout == "NN" else [])
    )
    env = os.environ.copy()
    env.update(env_extra or {})
    return subprocess.run(test_cmd, env=env, capture_output=True, check=False)


ELIGIBLE_OUT_FEATURES_PER_RANK = 1536
INELIGIBLE_OUT_FEATURES_PER_RANK = 1568
UNALIGNED_SEQ_LENGTH = 1152

# K = num_heads * head_dim; the kernel only gathers scales in-flight at K >= 16384.
INTERLEAVE_NUM_HEADS: int = 128
INTERLEAVE_HEAD_DIM: int = 128
# 1x hidden, not the harness' 4x default, which would only grow the weight.
INTERLEAVE_FFN_HIDDEN: int = INTERLEAVE_NUM_HEADS * INTERLEAVE_HEAD_DIM


def _run_fused_layer(nprocs, extra_args, seq_length=SEQ_LENGTH, quantization="none"):
    """Run the layer harness on a column-parallel LayerNormLinear with the fused backend live."""
    test_cmd = (
        _fused_launch_cmd(nprocs)
        + [
            str(TEST_ROOT / "run_layer_with_overlap.py"),
            f"--seed={RNG_SEED}",
            f"--seq-length={seq_length}",
            f"--batch-size={BATCH_SIZE}",
            f"--num-heads={NUM_HEADS}",
            f"--head-dim={HEAD_DIM}",
            f"--layer-type={te.LayerNormLinear.__name__}",
            "--linear-parallel-mode=column",
            "--num-layers=1",
            "--use-bf16-params",
            # TODO: Add bias support
            "--no-bias",
        ]
        + extra_args
    )
    if quantization != "none":
        test_cmd += ["--fp8", f"--quantization={quantization}"]
    env = os.environ.copy()
    env["PYTORCH_JIT"] = "0"
    env["NVTE_TORCH_COMPILE"] = "0"
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    env["NVTE_RS_DIAG"] = "1"
    return subprocess.run(test_cmd, env=env, capture_output=True, check=False)


def _reported_names(stdout, prefix):
    """The layer name sets the harness printed under `prefix`."""
    for line in stdout.decode().splitlines():
        if prefix in line:
            return set(line.split(prefix, 1)[1].split())
    return None


def _output_hashes(stdout):
    """The per-rank digests the harness printed, in rank order."""
    prefix = "OUTPUT HASH: "
    return [ln.split(prefix, 1)[1].strip() for ln in stdout.decode().splitlines() if prefix in ln]


def _assert_bulk_rs_precision(result, quantization):
    stderr = result.stderr.decode()
    ran_mxfp8 = "[RS_DIAG] bulk launched mxfp8" in stderr
    ran_bf16 = "[RS_DIAG] bulk launched bf16" in stderr
    if quantization == "mxfp8":
        assert ran_mxfp8, f"the bulk RS MXFP8 kernel never ran\n{stderr}"
        assert not ran_bf16, f"the bulk RS fell back to the bf16 kernel\n{stderr}"
    else:
        assert ran_bf16, f"the bulk RS bf16 kernel never ran\n{stderr}"
        assert not ran_mxfp8, f"a bf16 call ran the MXFP8 bulk RS kernel\n{stderr}"


def _assert_numerics_passed(result):
    stdout, stderr = result.stdout.decode(), result.stderr.decode()
    assert result.returncode == 0, f"non-zero exit\n{stderr}"
    assert "NUMERICAL CHECK FAILED" not in stderr, stderr
    assert "NUMERICAL CHECK PASSED" in stdout, stdout


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("layout", FUSED_LAYOUTS)
@pytest.mark.parametrize("quantization", FUSED_QUANTIZATIONS)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_ag_overlap(nprocs, quantization, layout):
    """An aligned shape: the fused backend runs and the result is correct."""
    if quantization == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    _assert_numerics_passed(_run_fused_ag(nprocs, quantization=quantization, layout=layout))


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("layout", FUSED_LAYOUTS)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_ag_overlap_mxfp8_interleaved_scales(nprocs, layout):
    """K >= 16384 packs the peers' scales inside the gatherer instead of via gather_scales."""
    if not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    result = _run_fused_ag(
        nprocs,
        quantization="mxfp8",
        layout=layout,
        num_heads=INTERLEAVE_NUM_HEADS,
        head_dim=INTERLEAVE_HEAD_DIM,
        extra_args=[f"--ffn-hidden-size={INTERLEAVE_FFN_HIDDEN}"],
        env_extra={"NVTE_AG_DIAG": "1"},
    )
    _assert_numerics_passed(result)
    assert "UB SCALE CHECK PASSED" in result.stdout.decode(), result.stdout.decode()
    # A threshold change would otherwise leave the shapes above covering the other path.
    stderr = result.stderr.decode()
    assert "scales=interleaved" in stderr, stderr


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_ag_overlap_rejects_fp8(nprocs):
    """Delayed-scaling FP8 is outside the backend; only MXFP8 1D scaling dispatches."""
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)
    result = _run_fused_ag(nprocs, quantization="fp8")
    assert result.returncode != 0, "fused AG+GEMM accepted a delayed-scaling FP8 operand"
    assert "fused AG+GEMM failed to launch" in result.stderr.decode(), result.stderr.decode()


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_ag_overlap_is_deterministic(nprocs):
    """Bitwise reproducibility across runs"""
    first = _run_fused_ag(nprocs)
    _assert_numerics_passed(first)
    second = _run_fused_ag(nprocs)
    _assert_numerics_passed(second)

    first_hashes, second_hashes = _output_hashes(first.stdout), _output_hashes(second.stdout)
    assert first_hashes, f"harness printed no output hash\n{first.stdout.decode()}"
    assert first_hashes == second_hashes, "two identical runs produced different outputs"


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("quantization", FUSED_QUANTIZATIONS)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_bulk_ag_overlap(nprocs, quantization):
    """The bulk all-gather that rides in an unrelated GEMM's grid."""
    if quantization == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    _assert_numerics_passed(_run_fused_ag(nprocs, bulk=True, quantization=quantization))


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("quantization", FUSED_QUANTIZATIONS)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_layer_bulk_dgrad(nprocs, quantization):
    """A column-parallel layer whose dgrad dimensions clear the fused contract."""
    if quantization == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    result = _run_fused_layer(
        nprocs,
        [f"--out-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"],
        quantization=quantization,
    )
    _assert_numerics_passed(result)
    fused = _reported_names(result.stdout, "UB FUSED NAMES: ")
    assert fused is not None, f"harness printed no fused name set\n{result.stdout.decode()}"
    assert "qkv_dgrad" in fused, fused
    eligible = _reported_names(result.stdout, "UB BULK ELIGIBLE: ")
    assert eligible is not None, f"harness printed no eligibility set\n{result.stdout.decode()}"
    assert "qkv_dgrad" in eligible, eligible


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("quantization", FUSED_QUANTIZATIONS)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_layer_bulk_wgrad(nprocs, quantization):
    """The bulk reduce-scatter that carries qkv's dgrad behind the wgrad GEMM."""
    if quantization == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    result = _run_fused_layer(
        nprocs,
        [f"--out-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"],
        quantization=quantization,
    )
    _assert_numerics_passed(result)
    fused = _reported_names(result.stdout, "UB FUSED NAMES: ")
    assert fused is not None, f"harness printed no fused name set\n{result.stdout.decode()}"
    assert "qkv_wgrad" in fused, fused
    eligible = _reported_names(result.stdout, "UB BULK ELIGIBLE: ")
    assert eligible is not None, f"harness printed no eligibility set\n{result.stdout.decode()}"
    assert "qkv_wgrad" in eligible, eligible
    _assert_bulk_rs_precision(result, quantization)


def _run_fused_row_parallel_layer(nprocs, extra_args, seq_length=SEQ_LENGTH, quantization="none"):
    """Run the layer harness on a row-parallel Linear: RS in proj_fprop, AG in proj_dgrad."""
    test_cmd = (
        _fused_launch_cmd(nprocs)
        + [
            str(TEST_ROOT / "run_layer_with_overlap.py"),
            f"--seed={RNG_SEED}",
            f"--seq-length={seq_length}",
            f"--batch-size={BATCH_SIZE}",
            f"--num-heads={NUM_HEADS}",
            f"--head-dim={HEAD_DIM}",
            f"--layer-type={te.Linear.__name__}",
            "--linear-parallel-mode=row",
            "--num-layers=1",
            "--use-bf16-params",
        ]
        + extra_args
    )
    # "none" deliberately omits --fp8: that flag is what puts the UB in FP8 quantization mode and
    # makes recipe.mxfp8() true, so passing it unconditionally would hide the bf16 path entirely.
    if quantization != "none":
        test_cmd += ["--fp8", f"--quantization={quantization}"]
    env = os.environ.copy()
    env["PYTORCH_JIT"] = "0"
    env["NVTE_TORCH_COMPILE"] = "0"
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    env["NVTE_RS_DIAG"] = "1"
    return subprocess.run(test_cmd, env=env, capture_output=True, check=False)


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("quantization", FUSED_QUANTIZATIONS)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_rs_overlap(nprocs, quantization):
    """bf16 or mxfp8 at an aligned shape: the fused reduce-scatter runs and the result is correct."""
    if quantization == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    result = _run_fused_row_parallel_layer(
        nprocs,
        [f"--in-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"],
        quantization=quantization,
    )
    _assert_numerics_passed(result)
    stderr = result.stderr.decode()
    assert "failed to launch" not in stderr, stderr
    disabled = _reported_names(result.stdout, "UB DISABLED NAMES: ")
    assert disabled is not None, f"harness printed no disabled name set\n{result.stdout.decode()}"
    assert "proj_fprop" not in disabled, disabled
    ran_mxfp8 = "[RS_DIAG] launched mxfp8" in stderr
    ran_bf16 = "[RS_DIAG] launched bf16" in stderr
    if quantization == "mxfp8":
        assert ran_mxfp8, f"the fused RS MXFP8 kernel never ran\n{stderr}"
        assert not ran_bf16, f"the fused RS fell back to the bf16 kernel\n{stderr}"
    else:
        assert ran_bf16, f"the fused RS bf16 kernel never ran\n{stderr}"
        assert not ran_mxfp8, f"a bf16 call ran the MXFP8 kernel\n{stderr}"


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_ag_overlap_row_parallel_mxfp8(nprocs):
    """Backward gathers dY twice: row-scaled for dgrad, column-scaled for wgrad."""
    if not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    result = _run_fused_row_parallel_layer(
        nprocs,
        [f"--in-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"],
        quantization="mxfp8",
    )
    _assert_numerics_passed(result)
    stderr = result.stderr.decode()
    # The second gather used to be routed to CommOverlapP2PBase::bulk_overlap_external_ag, which
    # is a stub on every backend.
    assert "Operation not supported" not in stderr, stderr
    assert "failed to launch" not in stderr, stderr


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_rs_overlap_rejects_delayed_scaling(nprocs):
    """Delayed-scaling FP8 is outside the backend -- only MXFP8 1D scaling dispatches -- so a
    row-parallel Linear on that recipe must fall back cleanly rather than reach the kernel."""
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)
    result = _run_fused_row_parallel_layer(
        nprocs,
        [f"--in-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"],
        quantization="fp8_delayed_scaling",
    )
    stderr = result.stderr.decode()
    assert "non-bf16 operand" not in stderr, f"a non-bf16 operand reached the kernel\n{stderr}"
    assert "only supports MXFP8_1D_SCALING" not in stderr, f"the kernel was reached\n{stderr}"
    _assert_numerics_passed(result)


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_rs_declines_unaligned_region(nprocs):
    """tokens %% (tp * BLOCK_ROW) != 0 has no whole band, so setup must decline."""
    result = _run_fused_row_parallel_layer(
        nprocs,
        [f"--in-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"],
        seq_length=UNALIGNED_SEQ_LENGTH,
    )
    _assert_numerics_passed(result)
    disabled = _reported_names(result.stdout, "UB DISABLED NAMES: ")
    assert disabled is not None, f"harness printed no disabled name set\n{result.stdout.decode()}"
    assert "proj_fprop" in disabled, disabled


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_rs_declines_ineligible_k(nprocs):
    """A K the fused kernels cannot serve falls back without erroring."""
    result = _run_fused_row_parallel_layer(
        nprocs, [f"--in-features={INELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"]
    )
    _assert_numerics_passed(result)
    stderr = result.stderr.decode()
    assert "ineligible shape" not in stderr, stderr
    assert "failed to launch" not in stderr, stderr


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_rs_overlap_is_deterministic(nprocs):
    """Two runs at the same seed agree: the collective's arrival order must not reach the output."""
    args = [f"--in-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"]
    first, second = (_run_fused_row_parallel_layer(nprocs, args) for _ in range(2))
    _assert_numerics_passed(first)
    _assert_numerics_passed(second)
    first_hashes, second_hashes = _output_hashes(first.stdout), _output_hashes(second.stdout)
    assert first_hashes, f"harness printed no output hash\n{first.stdout.decode()}"
    assert first_hashes == second_hashes, "two identical runs produced different outputs"


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_layer_declines_ineligible_k(nprocs):
    """A K the fused kernels cannot serve has to fall back to no overlap."""
    result = _run_fused_layer(
        nprocs, [f"--out-features={INELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"]
    )
    _assert_numerics_passed(result)
    stderr = result.stderr.decode()
    assert "ineligible shape" not in stderr, stderr
    assert "failed to launch" not in stderr, stderr
    fused = _reported_names(result.stdout, "UB FUSED NAMES: ")
    disabled = _reported_names(result.stdout, "UB DISABLED NAMES: ")
    assert fused is not None, f"harness printed no fused name set\n{result.stdout.decode()}"
    assert "qkv_dgrad" in fused, fused
    assert disabled is not None and "qkv_dgrad" not in disabled, disabled
    eligible = _reported_names(result.stdout, "UB BULK ELIGIBLE: ")
    assert eligible is not None, f"harness printed no eligibility set\n{result.stdout.decode()}"
    assert "qkv_dgrad" not in eligible, eligible


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_layer_declines_unaligned_region(nprocs):
    """A Userbuffers region the fused backend cannot serve declines at setup."""
    result = _run_fused_layer(
        nprocs,
        [f"--out-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"],
        seq_length=UNALIGNED_SEQ_LENGTH,
    )
    _assert_numerics_passed(result)
    fused = _reported_names(result.stdout, "UB FUSED NAMES: ")
    disabled = _reported_names(result.stdout, "UB DISABLED NAMES: ")
    assert fused == set(), f"expected no fused communicators, got {fused}"
    assert disabled is not None, f"harness printed no disabled name set\n{result.stdout.decode()}"
    assert {"qkv_fprop", "qkv_dgrad", "qkv_wgrad"} <= disabled, disabled
