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
    """The minimum both backends need: a 256-aligned per-rank chunk and a 256-aligned N.

    The reduce-scatter backend is stricter still -- its combine is specialised on 8 ranks -- but
    that is asserted per test rather than hidden in collection, so a tp=4 decline stays visible.
    """
    return (SEQ_LENGTH * BATCH_SIZE) % (256 * nprocs) == 0 and (NUM_HEADS * HEAD_DIM) % 256 == 0


FUSED_PROC_COUNTS = [n for n in (4, 8) if n <= torch.cuda.device_count() and _fused_shape_ok(n)]

fused_available = (
    IS_HIP_EXTENSION and get_device_compute_capability() == (9, 5) and len(FUSED_PROC_COUNTS) > 0
)
reason_for_no_fused = (
    "Fused comm+GEMM overlap requires a gfx950 device, tp_size in (4, 8) and a 256-aligned "
    "per-rank chunk."
)


def _fused_launch_cmd(nprocs: int):
    """Same form as LAUNCH_CMD, but at a rank count the fused tests choose."""
    if tex.ubuf_built_with_mpi():
        return ["mpirun", "-np", str(nprocs), "--oversubscribe", "--quiet", "python3"]
    return ["torchrun", f"--nproc_per_node={nprocs}"]


def _run_fused_ag(nprocs, bulk=False, quantization="none"):
    """Run the AG overlap harness with the fused backend, returning the completed process."""
    test_cmd = _fused_launch_cmd(nprocs) + [
        str(TEST_ROOT / "run_gemm_with_overlap.py"),
        "--check-numerics",
        f"--seed={RNG_SEED}",
        f"--seq-length={SEQ_LENGTH}",
        f"--batch-size={BATCH_SIZE}",
        f"--num-heads={NUM_HEADS}",
        f"--head-dim={HEAD_DIM}",
        "--comm-type=AG",
        "--fused",
    ]
    test_cmd += ["--bulk-overlap"] if bulk else ["--p2p", f"--quantization={quantization}"]
    return subprocess.run(test_cmd, env=os.environ, capture_output=True, check=False)


ELIGIBLE_OUT_FEATURES_PER_RANK = 1536
INELIGIBLE_OUT_FEATURES_PER_RANK = 1568
UNALIGNED_SEQ_LENGTH = 1152


def _run_fused_layer(nprocs, extra_args, seq_length=SEQ_LENGTH):
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
        ]
        + extra_args
    )
    env = os.environ.copy()
    env["PYTORCH_JIT"] = "0"
    env["NVTE_TORCH_COMPILE"] = "0"
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    return subprocess.run(test_cmd, env=env, capture_output=True, check=False)


def _reported_names(stdout, prefix):
    """The layer name sets the harness printed under `prefix`."""
    for line in stdout.decode().splitlines():
        if prefix in line:
            return set(line.split(prefix, 1)[1].split())
    return None


def _assert_numerics_passed(result):
    stdout, stderr = result.stdout.decode(), result.stderr.decode()
    assert result.returncode == 0, f"non-zero exit\n{stderr}"
    assert "NUMERICAL CHECK FAILED" not in stderr, stderr
    assert "NUMERICAL CHECK PASSED" in stdout, stdout


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_ag_overlap_bf16(nprocs):
    """bf16 at an aligned shape: the fused backend runs and the result is correct."""
    _assert_numerics_passed(_run_fused_ag(nprocs))


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
@pytest.mark.parametrize("quantization", ("fp8", "mxfp8"))
def test_fused_ag_overlap_rejects_non_bf16(quantization, nprocs):
    """Non-bf16 is currently outside the backend."""
    if quantization == "fp8" and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if quantization == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    result = _run_fused_ag(nprocs, quantization=quantization)
    assert result.returncode != 0, "fused AG+GEMM accepted a non-bf16 operand"
    assert "non-bf16 operand" in result.stderr.decode(), result.stderr.decode()


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_ag_overlap_is_deterministic(nprocs):
    """Bitwise reproducibility across runs"""
    first = _run_fused_ag(nprocs)
    _assert_numerics_passed(first)
    second = _run_fused_ag(nprocs)
    _assert_numerics_passed(second)

    def _hashes(out):
        prefix = "OUTPUT HASH: "
        return [ln.split(prefix, 1)[1].strip() for ln in out.decode().splitlines() if prefix in ln]

    first_hashes, second_hashes = _hashes(first.stdout), _hashes(second.stdout)
    assert first_hashes, f"harness printed no output hash\n{first.stdout.decode()}"
    assert first_hashes == second_hashes, "two identical runs produced different outputs"


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_bulk_ag_overlap_bf16(nprocs):
    """The bulk all-gather that rides in an unrelated GEMM's grid."""
    _assert_numerics_passed(_run_fused_ag(nprocs, bulk=True))


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_layer_bulk_dgrad_bf16(nprocs):
    """A column-parallel layer whose dgrad dimensions clear the fused contract."""
    result = _run_fused_layer(nprocs, [f"--out-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"])
    _assert_numerics_passed(result)
    fused = _reported_names(result.stdout, "UB FUSED NAMES: ")
    assert fused is not None, f"harness printed no fused name set\n{result.stdout.decode()}"
    assert "qkv_dgrad" in fused, fused
    eligible = _reported_names(result.stdout, "UB BULK ELIGIBLE: ")
    assert eligible is not None, f"harness printed no eligibility set\n{result.stdout.decode()}"
    assert "qkv_dgrad" in eligible, eligible


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_layer_bulk_wgrad_bf16(nprocs):
    """The backward reduce-scatter hidden behind the wgrad GEMM, on the fused method.

    wgrad is in the default fused layer set on gfx950, so this needs no extra configuration.
    """
    result = _run_fused_layer(nprocs, [f"--out-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"])
    _assert_numerics_passed(result)
    fused = _reported_names(result.stdout, "UB FUSED NAMES: ")
    assert fused is not None, f"harness printed no fused name set\n{result.stdout.decode()}"
    assert "qkv_wgrad" in fused, fused
    eligible = _reported_names(result.stdout, "UB BULK ELIGIBLE: ")
    assert eligible is not None, f"harness printed no eligibility set\n{result.stdout.decode()}"
    assert "qkv_wgrad" in eligible, eligible


def _run_fused_rs_layer(nprocs, extra_args, seq_length=SEQ_LENGTH):
    """Run the layer harness on a ROW-parallel Linear, which is the fused reduce-scatter's path."""
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
    env = os.environ.copy()
    env["PYTORCH_JIT"] = "0"
    env["NVTE_TORCH_COMPILE"] = "0"
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    return subprocess.run(test_cmd, env=env, capture_output=True, check=False)


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_rs_overlap_bf16(nprocs):
    """bf16 at an aligned shape: the fused reduce-scatter runs and the result is correct.

    The combine is specialised on 8 ranks, so below that the backend must decline and fall back
    rather than fail -- both outcomes are correct, and which one applies is asserted on nprocs.
    """
    result = _run_fused_rs_layer(
        nprocs, [f"--in-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"]
    )
    _assert_numerics_passed(result)
    stderr = result.stderr.decode()
    assert "failed to launch" not in stderr, stderr
    disabled = _reported_names(result.stdout, "UB DISABLED NAMES: ")
    assert disabled is not None, f"harness printed no disabled name set\n{result.stdout.decode()}"
    if nprocs == 8:
        assert "proj_fprop" not in disabled, disabled
    else:
        assert "proj_fprop" in disabled, f"tp={nprocs} must decline the fused RS, got {disabled}"


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
@pytest.mark.parametrize("quantization", ("fp8_delayed_scaling", "mxfp8"))
def test_fused_rs_overlap_rejects_non_bf16(quantization, nprocs):
    """Non-bf16 is outside the backend, and setup must decline rather than reach the kernel.

    The AG sibling asserts a non-zero exit because its region carries the operand dtype straight
    into the kernel guard. The RS path is gated earlier -- _fused_rs_ub_supported rejects any
    non-bf16 region at setup -- so the layer is disabled and the run stays correct. Either way the
    kernel must never see a non-bf16 operand, which is what the stderr assertion pins.
    """
    if quantization.startswith("fp8") and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if quantization == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    result = _run_fused_rs_layer(
        nprocs,
        [
            f"--in-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}",
            f"--quantization={quantization}",
        ],
    )
    stderr = result.stderr.decode()
    assert "non-bf16 operand" not in stderr, f"a non-bf16 operand reached the kernel\n{stderr}"
    if result.returncode == 0:
        disabled = _reported_names(result.stdout, "UB DISABLED NAMES: ")
        assert disabled is not None, f"harness printed no disabled set\n{result.stdout.decode()}"
        assert "proj_fprop" in disabled, f"non-bf16 was not declined at setup: {disabled}"


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_rs_declines_unaligned_region(nprocs):
    """tokens %% (tp * BLOCK_ROW) != 0 has no whole band, so setup must decline."""
    result = _run_fused_rs_layer(
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
    result = _run_fused_rs_layer(
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
    first, second = (_run_fused_rs_layer(nprocs, args) for _ in range(2))
    _assert_numerics_passed(first)
    _assert_numerics_passed(second)


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
