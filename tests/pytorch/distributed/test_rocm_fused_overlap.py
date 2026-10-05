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
            # TODO: Add bias support
            "--no-bias",
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


def _output_hashes(stdout):
    """The per-rank digests the harness printed, in rank order."""
    prefix = "OUTPUT HASH: "
    return [ln.split(prefix, 1)[1].strip() for ln in stdout.decode().splitlines() if prefix in ln]


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
def test_fused_ag_overlap_rejects_fp8(nprocs):
    """Per-tensor FP8 is outside the backend (MXFP8 is covered by KOSMOS, below)."""
    if not fp8_available:
        pytest.skip(reason_for_no_fp8)
    result = _run_fused_ag(nprocs, quantization="fp8")
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

    first_hashes, second_hashes = _output_hashes(first.stdout), _output_hashes(second.stdout)
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
    result = _run_fused_layer(nprocs, [f"--out-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"])
    _assert_numerics_passed(result)
    fused = _reported_names(result.stdout, "UB FUSED NAMES: ")
    assert fused is not None, f"harness printed no fused name set\n{result.stdout.decode()}"
    assert "qkv_wgrad" in fused, fused
    eligible = _reported_names(result.stdout, "UB BULK ELIGIBLE: ")
    assert eligible is not None, f"harness printed no eligibility set\n{result.stdout.decode()}"
    assert "qkv_wgrad" in eligible, eligible


def _run_fused_rs_layer(nprocs, extra_args, seq_length=SEQ_LENGTH):
    """Run the layer harness on a row-parallel Linear, which is the fused reduce-scatter's path."""
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
    """bf16 at an aligned shape: the fused reduce-scatter runs and the result is correct."""
    result = _run_fused_rs_layer(
        nprocs, [f"--in-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}"]
    )
    _assert_numerics_passed(result)
    stderr = result.stderr.decode()
    assert "failed to launch" not in stderr, stderr
    disabled = _reported_names(result.stdout, "UB DISABLED NAMES: ")
    assert disabled is not None, f"harness printed no disabled name set\n{result.stdout.decode()}"
    assert "proj_fprop" not in disabled, disabled


@pytest.mark.skipif(not fused_available, reason=reason_for_no_fused)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
@pytest.mark.parametrize("quantization", ("fp8_delayed_scaling", "mxfp8"))
def test_fused_rs_overlap_rejects_non_bf16(quantization, nprocs):
    """A quantized row-parallel Linear must fall back cleanly instead of reaching the bf16 kernel."""
    if quantization.startswith("fp8") and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if quantization == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    result = _run_fused_rs_layer(
        nprocs,
        [
            f"--in-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}",
            "--fp8",
            f"--quantization={quantization}",
        ],
    )
    stderr = result.stderr.decode()
    assert "non-bf16 operand" not in stderr, f"a non-bf16 operand reached the kernel\n{stderr}"
    _assert_numerics_passed(result)


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


# ---------------------------------------------------------------------------------------------
# MXFP8 on the KOSMOS backend
# ---------------------------------------------------------------------------------------------


def _kosmos_built() -> bool:
    """Whether this TransformerEngine links the KOSMOS comm+GEMM backend."""
    from transformer_engine.common import _get_shared_object_file

    with open(_get_shared_object_file("core"), "rb") as lib:
        return b"[KOSMOS] ub" in lib.read()


mxfp8_kosmos_available = fused_available and mxfp8_available and _kosmos_built()
reason_for_no_mxfp8_kosmos = (
    "MXFP8 fused overlaps need a gfx950 device, NVTE_ROCM_ENABLE_MXFP8=1 and TransformerEngine "
    "built with KOSMOS (NVTE_KOSMOS_ROOT)."
)
# The fused row-parallel dgrad+wgrad runs at TP 8 here (TP 4 config-3 fp32 dW is under KOSMOS
# investigation).
DGRAD_WGRAD_PROC_COUNTS = [n for n in FUSED_PROC_COUNTS if n == 8]

MXFP8_ARGS = ["--fp8", "--quantization=mxfp8", "--poison", "--fp32-truth"]


def _run_mxfp8_layer(nprocs, layer_args, num_heads=NUM_HEADS, head_dim=HEAD_DIM):
    """Run the layer harness in MXFP8 against the non-overlapped TE MXFP8 reference."""
    test_cmd = (
        _fused_launch_cmd(nprocs)
        + [
            str(TEST_ROOT / "run_layer_with_overlap.py"),
            f"--seed={RNG_SEED}",
            f"--seq-length={SEQ_LENGTH}",
            f"--batch-size={BATCH_SIZE}",
            f"--num-heads={num_heads}",
            f"--head-dim={head_dim}",
            "--num-layers=1",
            "--use-bf16-params",
        ]
        + MXFP8_ARGS
        + layer_args
    )
    env = os.environ.copy()
    env["PYTORCH_JIT"] = "0"
    env["NVTE_TORCH_COMPILE"] = "0"
    env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    env["NVTE_ROCM_ENABLE_MXFP8"] = "1"
    env["NVTE_KOSMOS_LOG"] = "1"
    result = subprocess.run(test_cmd, env=env, capture_output=True, check=False)
    _report(result)
    return result


def _report(result):
    """Echo the backend evidence (shown by pytest -rP): KOSMOS lines, GEMM counts, checks on rank 0."""
    keys = ("[KOSMOS]", "GEMM CALLS", "REPEAT", "UB FUSED DGRAD+WGRAD", "OUTPUT HASH")
    for ln in result.stdout.decode().splitlines():
        if ln.startswith("[KOSMOS]") or (
            ln.startswith("[rank0]")
            and (any(k in ln for k in keys) or "ACCURACY" in ln or "NUMERICAL CHECK" in ln)
        ):
            print(ln)
    for ln in result.stderr.decode().splitlines():
        if "FAILED" in ln or "Error" in ln:
            print("stderr:", ln)


def _kosmos_lines(result):
    return [ln for ln in result.stdout.decode().splitlines() if ln.startswith("[KOSMOS]")]


def _served(result, op, detail=None):
    """Whether a KOSMOS log line reports `op` served in MXFP8 (with `detail` after the "--")."""
    prefix = f" {op} mxfp8: KOSMOS"
    for ln in _kosmos_lines(result):
        if ln.startswith("[KOSMOS] ub ") and prefix in ln:
            if detail is None or ln.endswith(f"-- {detail}"):
                return True
    return False


def _decision(result, ub_name, op):
    """The Python gate's outcome for `ub_name`/`op` in MXFP8: "fused" or "declined -- reason"."""
    prefix = f"[KOSMOS] py {ub_name} {op} mxfp8: "
    outcomes = [ln[len(prefix) :] for ln in _kosmos_lines(result) if ln.startswith(prefix)]
    assert len(outcomes) == 1, f"expected one decision for {ub_name} {op}, got {outcomes}"
    return outcomes[0]


def _gemm_calls(result):
    """GEMMs by layout (and fused dgrad+wgrad calls) in the first overlapped step."""
    for ln in result.stdout.decode().splitlines():
        if "GEMM CALLS: " in ln:
            pairs = ln.split("GEMM CALLS: ", 1)[1].split()
            return {k: int(v) for k, v in (p.split("=") for p in pairs)}
    raise AssertionError(f"harness printed no GEMM call counts\n{result.stdout.decode()}")


def _assert_repeat_identical(result):
    assert "REPEAT 2: identical" in result.stdout.decode(), result.stdout.decode()


@pytest.mark.skipif(not mxfp8_kosmos_available, reason=reason_for_no_mxfp8_kosmos)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_fused_ag_overlap_mxfp8(nprocs):
    """GEMM-level MXFP8 AG+GEMM (TN): KOSMOS gathers the fp8 data and scales in place."""
    env = os.environ.copy()
    env["NVTE_ROCM_ENABLE_MXFP8"] = "1"
    env["NVTE_KOSMOS_LOG"] = "1"
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
        "--p2p",
        "--quantization=mxfp8",
    ]
    result = subprocess.run(test_cmd, env=env, capture_output=True, check=False)
    _report(result)
    _assert_numerics_passed(result)
    assert _served(result, "AG+GEMM"), _kosmos_lines(result)


@pytest.mark.skipif(not mxfp8_kosmos_available, reason=reason_for_no_mxfp8_kosmos)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
@pytest.mark.parametrize("layer_type", ("LayerNormLinear", "Linear"))
def test_mxfp8_column_parallel(layer_type, nprocs):
    """Column-parallel MXFP8: AG+GEMM fprop, bulk AG under the dgrad, bulk RS under the wgrad."""
    result = _run_mxfp8_layer(
        nprocs,
        [
            f"--layer-type={layer_type}",
            "--linear-parallel-mode=column",
            f"--out-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}",
            "--no-bias",
            "--repeat=2",
        ],
    )
    _assert_numerics_passed(result)
    _assert_repeat_identical(result)
    assert _decision(result, "qkv_fprop", "AG+GEMM") == "fused"
    assert _decision(result, "qkv_dgrad", "bulk AG") == "fused"
    assert _decision(result, "qkv_wgrad", "bulk RS") == "fused"
    for op in ("AG+GEMM", "bulk AG", "bulk RS"):
        assert _served(result, op), (op, _kosmos_lines(result))
    assert _gemm_calls(result) == {"TN": 1, "NN": 1, "NT": 1}


@pytest.mark.skipif(not mxfp8_kosmos_available, reason=reason_for_no_mxfp8_kosmos)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_mxfp8_column_parallel_fuse_wgrad_accumulation(nprocs):
    """Bulk RS wgrad into the fp32 main_grad: overwritten on microbatch 1, accumulated on 2."""
    result = _run_mxfp8_layer(
        nprocs,
        [
            "--layer-type=LayerNormLinear",
            "--linear-parallel-mode=column",
            f"--out-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}",
            "--no-bias",
            "--fuse-wgrad-accumulation",
            "--microbatches=2",
            "--repeat=2",
        ],
    )
    _assert_numerics_passed(result)
    _assert_repeat_identical(result)
    assert _served(result, "bulk RS", "fp32 D"), _kosmos_lines(result)
    assert _served(result, "bulk RS", "fp32 D +="), _kosmos_lines(result)


def _row_parallel_args(nprocs, extra=()):
    return [
        "--layer-type=Linear",
        "--linear-parallel-mode=row",
        f"--in-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}",
    ] + list(extra)


def _assert_fused_dgrad_wgrad(result, ub_name="proj", calls=None):
    """The row-parallel backward ran as one fused call: no NN dgrad GEMM, no NT wgrad GEMM."""
    assert _decision(result, f"{ub_name}_dgrad", "AG dgrad+wgrad") == "fused"
    assert f"UB FUSED DGRAD+WGRAD: {ub_name}" in result.stdout.decode()
    assert _served(result, "AG dgrad+wgrad"), _kosmos_lines(result)
    # The dgrad+wgrad buffer served nothing else (no separate AG+GEMM dgrad on it).
    ub_lines = [ln.split() for ln in _kosmos_lines(result) if ln.startswith("[KOSMOS] ub ")]
    fused_ubs = {w[2] for w in ub_lines if "dgrad+wgrad" in w}
    assert not any(w[2] in fused_ubs and "dgrad+wgrad" not in w for w in ub_lines), ub_lines
    assert _gemm_calls(result) == (calls or {"TN": 1, "FUSED_DGRAD_WGRAD": 1})


@pytest.mark.skipif(not mxfp8_kosmos_available, reason=reason_for_no_mxfp8_kosmos)
@pytest.mark.parametrize("nprocs", DGRAD_WGRAD_PROC_COUNTS)
def test_mxfp8_row_parallel_fused_dgrad_wgrad(nprocs):
    """Row-parallel MXFP8: GEMM+RS fprop and the fused dY all-gather + dgrad + wgrad backward."""
    result = _run_mxfp8_layer(nprocs, _row_parallel_args(nprocs, ["--repeat=2"]))
    _assert_numerics_passed(result)
    _assert_repeat_identical(result)
    assert _decision(result, "proj_fprop", "GEMM+RS") == "fused"
    assert _served(result, "GEMM+RS"), _kosmos_lines(result)
    assert _served(result, "AG dgrad+wgrad", "bf16 dW"), _kosmos_lines(result)
    _assert_fused_dgrad_wgrad(result)


@pytest.mark.skipif(not mxfp8_kosmos_available, reason=reason_for_no_mxfp8_kosmos)
@pytest.mark.parametrize("nprocs", DGRAD_WGRAD_PROC_COUNTS)
def test_mxfp8_row_parallel_is_deterministic(nprocs):
    """Two runs at the same seed agree bit for bit."""
    first, second = (_run_mxfp8_layer(nprocs, _row_parallel_args(nprocs)) for _ in range(2))
    _assert_numerics_passed(first)
    _assert_numerics_passed(second)
    first_hashes, second_hashes = _output_hashes(first.stdout), _output_hashes(second.stdout)
    assert first_hashes, f"harness printed no output hash\n{first.stdout.decode()}"
    assert first_hashes == second_hashes, "two identical runs produced different outputs"


@pytest.mark.skipif(not mxfp8_kosmos_available, reason=reason_for_no_mxfp8_kosmos)
@pytest.mark.parametrize("nprocs", DGRAD_WGRAD_PROC_COUNTS)
@pytest.mark.parametrize("overwrite", (False, True))
def test_mxfp8_row_parallel_fuse_wgrad_accumulation(overwrite, nprocs):
    """The fused dgrad+wgrad writes the fp32 main_grad (microbatch 1) and accumulates (2), unless
    the weight overwrites main_grad."""
    extra = ["--fuse-wgrad-accumulation", "--microbatches=2", "--repeat=2"]
    if overwrite:
        extra.append("--overwrite-main-grad")
    result = _run_mxfp8_layer(nprocs, _row_parallel_args(nprocs, extra))
    _assert_numerics_passed(result)
    _assert_repeat_identical(result)
    _assert_fused_dgrad_wgrad(result, calls={"TN": 2, "FUSED_DGRAD_WGRAD": 2})
    assert _served(result, "AG dgrad+wgrad", "fp32 dW"), _kosmos_lines(result)
    assert _served(result, "AG dgrad+wgrad", "fp32 dW +=") != overwrite, _kosmos_lines(result)


@pytest.mark.skipif(not mxfp8_kosmos_available, reason=reason_for_no_mxfp8_kosmos)
@pytest.mark.parametrize("nprocs", DGRAD_WGRAD_PROC_COUNTS)
@pytest.mark.parametrize(
    "case,reason",
    (
        ("hybrid", "gradients not E4M3 (HYBRID recipe)"),
        ("delay_wgrad", "delay_wgrad_compute"),
        ("shape", "shape"),
    ),
)
def test_mxfp8_row_parallel_declines(case, reason, nprocs):
    """Cases outside the fused dgrad+wgrad take the non-overlapped backward and stay correct."""
    args = _row_parallel_args(nprocs)
    if case == "hybrid":
        args.append("--fp8-format=hybrid")
    elif case == "delay_wgrad":
        args.append("--delay-wgrad-compute")
    else:
        # k_local % 256 != 0 (still a multiple of 128, so the forward GEMM+RS stays fused)
        args[-1] = f"--in-features={1408 * nprocs}"
    result = _run_mxfp8_layer(nprocs, args)
    _assert_numerics_passed(result)
    assert _decision(result, "proj_dgrad", "AG dgrad+wgrad").startswith(f"declined -- {reason}")
    assert not _served(result, "AG dgrad+wgrad"), _kosmos_lines(result)
    assert "UB FUSED DGRAD+WGRAD: proj" not in result.stdout.decode()
    calls = _gemm_calls(result)
    assert calls.get("FUSED_DGRAD_WGRAD", 0) == 0 and calls["NN"] == 1 and calls["NT"] == 1, calls


@pytest.mark.skipif(not mxfp8_kosmos_available, reason=reason_for_no_mxfp8_kosmos)
@pytest.mark.parametrize("nprocs", FUSED_PROC_COUNTS)
def test_mxfp8_column_parallel_hybrid_declines_backward(nprocs):
    """HYBRID: E4M3 forward stays fused, the E5M2-gradient bulk overlaps decline."""
    result = _run_mxfp8_layer(
        nprocs,
        [
            "--layer-type=LayerNormLinear",
            "--linear-parallel-mode=column",
            f"--out-features={ELIGIBLE_OUT_FEATURES_PER_RANK * nprocs}",
            "--no-bias",
            "--fp8-format=hybrid",
        ],
    )
    _assert_numerics_passed(result)
    assert _decision(result, "qkv_fprop", "AG+GEMM") == "fused"
    gradients = "declined -- gradients not E4M3 (HYBRID recipe)"
    assert _decision(result, "qkv_dgrad", "bulk AG") == gradients
    assert _decision(result, "qkv_wgrad", "bulk RS") == gradients
    assert not _served(result, "bulk AG") and not _served(result, "bulk RS"), _kosmos_lines(result)


@pytest.mark.skipif(not mxfp8_kosmos_available, reason=reason_for_no_mxfp8_kosmos)
@pytest.mark.parametrize("nprocs", DGRAD_WGRAD_PROC_COUNTS)
@pytest.mark.parametrize("fuse_wgrad_accumulation", (False, True))
def test_mxfp8_layernorm_mlp(fuse_wgrad_accumulation, nprocs):
    """LayerNormMLP: fc1 AG+GEMM and bulk AG/RS, fc2 GEMM+RS and the fused fc2 dgrad+wgrad."""
    extra = ["--layer-type=LayerNormMLP", "--no-bias", "--repeat=2"]
    if fuse_wgrad_accumulation:
        extra += ["--fuse-wgrad-accumulation", "--microbatches=2"]
    result = _run_mxfp8_layer(nprocs, extra)
    _assert_numerics_passed(result)
    _assert_repeat_identical(result)
    assert _decision(result, "fc1_fprop", "AG+GEMM") == "fused"
    assert _decision(result, "fc2_fprop", "GEMM+RS") == "fused"
    assert _decision(result, "fc1_dgrad", "bulk AG") == "fused"
    assert _decision(result, "fc1_wgrad", "bulk RS") == "fused"
    for op in ("AG+GEMM", "GEMM+RS", "bulk AG", "bulk RS"):
        assert _served(result, op), (op, _kosmos_lines(result))
    mb = 2 if fuse_wgrad_accumulation else 1
    _assert_fused_dgrad_wgrad(
        result, "fc2", {"TN": 2 * mb, "NN": mb, "NT": mb, "FUSED_DGRAD_WGRAD": mb}
    )
    if fuse_wgrad_accumulation:
        assert _served(result, "AG dgrad+wgrad", "fp32 dW +="), _kosmos_lines(result)
        assert _served(result, "bulk RS", "fp32 D +="), _kosmos_lines(result)


@pytest.mark.skipif(not mxfp8_kosmos_available, reason=reason_for_no_mxfp8_kosmos)
@pytest.mark.parametrize("nprocs", DGRAD_WGRAD_PROC_COUNTS)
def test_mxfp8_layernorm_mlp_with_bias(nprocs):
    """A bias epilogue keeps fc1's forward off the backend; the fc2 backward stays fused."""
    result = _run_mxfp8_layer(nprocs, ["--layer-type=LayerNormMLP"])
    _assert_numerics_passed(result)
    assert _decision(result, "fc1_fprop", "AG+GEMM") == "declined -- bias epilogue"
    assert _decision(result, "fc2_dgrad", "AG dgrad+wgrad") == "fused"
    assert _served(result, "AG dgrad+wgrad"), _kosmos_lines(result)


@pytest.mark.skipif(not mxfp8_kosmos_available, reason=reason_for_no_mxfp8_kosmos)
@pytest.mark.parametrize("nprocs", DGRAD_WGRAD_PROC_COUNTS)
@pytest.mark.parametrize("fuse_wgrad_accumulation", (False, True))
def test_mxfp8_transformer_layer(fuse_wgrad_accumulation, nprocs):
    """A TransformerLayer whose every TP GEMM runs on KOSMOS in MXFP8."""
    extra = ["--layer-type=TransformerLayer", "--no-bias"]
    if fuse_wgrad_accumulation:
        extra += ["--fuse-wgrad-accumulation", "--microbatches=2"]
    else:
        extra += ["--repeat=2"]
    # head_dim 128 makes the per-rank qkv width a multiple of 256.
    result = _run_mxfp8_layer(nprocs, extra, head_dim=128)
    _assert_numerics_passed(result)
    for name, op in (
        ("qkv_fprop", "AG+GEMM"),
        ("qkv_dgrad", "bulk AG"),
        ("qkv_wgrad", "bulk RS"),
        ("proj_fprop", "GEMM+RS"),
        ("proj_dgrad", "AG dgrad+wgrad"),
        ("fc1_fprop", "AG+GEMM"),
        ("fc1_dgrad", "bulk AG"),
        ("fc1_wgrad", "bulk RS"),
        ("fc2_fprop", "GEMM+RS"),
        ("fc2_dgrad", "AG dgrad+wgrad"),
    ):
        assert _decision(result, name, op) == "fused", (name, op)
    for op in ("AG+GEMM", "GEMM+RS", "bulk AG", "bulk RS", "AG dgrad+wgrad"):
        assert _served(result, op), (op, _kosmos_lines(result))
    assert "UB FUSED DGRAD+WGRAD: fc2 proj" in result.stdout.decode()
    mb = 2 if fuse_wgrad_accumulation else 1
    expected = {"TN": 4 * mb, "NN": 2 * mb, "NT": 2 * mb, "FUSED_DGRAD_WGRAD": 2 * mb}
    assert _gemm_calls(result) == expected
    if not fuse_wgrad_accumulation:
        _assert_repeat_identical(result)
