"""
Standalone correctness + performance verification for PR #519
cast_transpose: bf16 -> fp8e4m3, shape [2048, 12288], delayed scaling.

Compares the C++/HIP kernel (tex.quantize) against a PyTorch reference.
Validates both rowwise (cast) and columnwise (transpose) outputs.
"""
import torch
import numpy as np
import time

from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.utils import get_torch_float8_e4m3_type

# ── Config ──────────────────────────────────────────────────────────────
ROWS, COLS = 2048, 12288
IN_DTYPE = torch.bfloat16
FP8_DTYPE = tex.DType.kFloat8E4M3
NUM_WARMUP = 5
NUM_ITERS = 100

torch_fp8 = get_torch_float8_e4m3_type()
fp8_max = torch.finfo(torch_fp8).max

# ── Input data (deterministic) ──────────────────────────────────────────
rng = np.random.default_rng(np.random.MT19937(12345))
data = rng.uniform(-2.0, 1.0, (ROWS, COLS)).astype(np.float32)
input_tensor = torch.tensor(data, device="cuda").to(IN_DTYPE)

# ── Delayed-scaling quantizer ──────────────────────────────────────────
scale = torch.tensor([1.5], dtype=torch.float32, device="cuda")
amax = torch.zeros(1, dtype=torch.float32, device="cuda")
quantizer = Float8Quantizer(scale=scale.clone(), amax=amax.clone(), fp8_dtype=FP8_DTYPE)

# ── Run the C++/HIP kernel (PR #519 path) ─────────────────────────────
result = tex.quantize(input_tensor, quantizer)
torch.cuda.synchronize()

# Extract outputs
cast_out = result._data.view(torch_fp8)          # [2048, 12288] fp8
transpose_out = result._transpose.view(torch_fp8) # [12288, 2048] fp8

# ── PyTorch reference ──────────────────────────────────────────────────
ref_fp32 = input_tensor.float() * scale.item()
ref_clamped = ref_fp32.clamp(-fp8_max, fp8_max)
ref_cast = ref_clamped.to(torch_fp8)
ref_transpose = ref_cast.T.contiguous()

# ── Compare rowwise (cast) output ──────────────────────────────────────
cast_f32 = cast_out.float()
ref_cast_f32 = ref_cast.float()
cast_diff = (cast_f32 - ref_cast_f32).abs()
cast_mismatch = (cast_diff > 0).sum().item()
cast_max_diff = cast_diff.max().item()
total = ROWS * COLS

print("=" * 70)
print(f"Cast Transpose Correctness: bf16 -> fp8e4m3, [{ROWS}, {COLS}]")
print(f"  Delayed scaling, scale = {scale.item()}")
print("=" * 70)
print(f"\n[Rowwise cast output]")
print(f"  Shape:      {cast_out.shape}")
print(f"  Mismatches: {cast_mismatch} / {total}")
print(f"  Max diff:   {cast_max_diff}")

if cast_mismatch > 0:
    idx = torch.where(cast_diff > 0)
    print(f"  First 5 mismatches:")
    for i in range(min(5, cast_mismatch)):
        r, c = idx[0][i].item(), idx[1][i].item()
        print(f"    [{r},{c}]: kernel={cast_f32[r,c].item():.6f}, ref={ref_cast_f32[r,c].item():.6f}, "
              f"input={input_tensor[r,c].float().item():.6f}")

# ── Compare transpose output ──────────────────────────────────────────
trans_f32 = transpose_out.float()
ref_trans_f32 = ref_transpose.float()
trans_diff = (trans_f32 - ref_trans_f32).abs()
trans_mismatch = (trans_diff > 0).sum().item()
trans_max_diff = trans_diff.max().item()

print(f"\n[Transpose output]")
print(f"  Shape:      {transpose_out.shape}")
print(f"  Mismatches: {trans_mismatch} / {total}")
print(f"  Max diff:   {trans_max_diff}")

if trans_mismatch > 0:
    idx = torch.where(trans_diff > 0)
    print(f"  First 5 mismatches:")
    for i in range(min(5, trans_mismatch)):
        r, c = idx[0][i].item(), idx[1][i].item()
        print(f"    [{r},{c}]: kernel={trans_f32[r,c].item():.6f}, ref={ref_trans_f32[r,c].item():.6f}")

# ── Verify amax ──────────────────────────────────────────────────────
kernel_amax = result._get_quantizer().amax.item()
ref_amax = input_tensor.float().abs().max().item()
print(f"\n[Amax]")
print(f"  Kernel: {kernel_amax:.6f}")
print(f"  Ref:    {ref_amax:.6f}")
print(f"  Match:  {abs(kernel_amax - ref_amax) < 1e-5}")

# ── Performance: GPU-timed iterations ──────────────────────────────────
print(f"\n{'=' * 70}")
print(f"Performance ({NUM_ITERS} iterations, {NUM_WARMUP} warmup)")
print("=" * 70)

# Warmup
for _ in range(NUM_WARMUP):
    q = Float8Quantizer(scale=scale.clone(), amax=torch.zeros(1, dtype=torch.float32, device="cuda"), fp8_dtype=FP8_DTYPE)
    tex.quantize(input_tensor, q)
torch.cuda.synchronize()

# Timed runs
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

times = []
for _ in range(NUM_ITERS):
    q = Float8Quantizer(scale=scale.clone(), amax=torch.zeros(1, dtype=torch.float32, device="cuda"), fp8_dtype=FP8_DTYPE)
    start_event.record()
    tex.quantize(input_tensor, q)
    end_event.record()
    torch.cuda.synchronize()
    times.append(start_event.elapsed_time(end_event))

times = np.array(times)
avg_ms = np.mean(times)
median_ms = np.median(times)
min_ms = np.min(times)

# Convert ms to us for display
avg_us = avg_ms * 1000
median_us = median_ms * 1000
min_us = min_ms * 1000

# Bandwidth: bf16 read (2B) + fp8 write rowwise (1B) + fp8 write transpose (1B) = 4 bytes/elem
bytes_total = ROWS * COLS * 4
bw_avg = bytes_total / (avg_ms * 1e-3) / (1024**4)  # TiB/s
bw_median = bytes_total / (median_ms * 1e-3) / (1024**4)
bw_min_time = bytes_total / (min_ms * 1e-3) / (1024**4)

print(f"  Avg time:    {avg_us:.1f} us  ({bw_avg:.3f} TiB/s)")
print(f"  Median time: {median_us:.1f} us  ({bw_median:.3f} TiB/s)")
print(f"  Min time:    {min_us:.1f} us  ({bw_min_time:.3f} TiB/s)")

# ── Final verdict ──────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
if cast_mismatch == 0 and trans_mismatch == 0:
    print("PASS: All outputs match reference exactly.")
else:
    print(f"FAIL: {cast_mismatch} cast mismatches, {trans_mismatch} transpose mismatches.")
print("=" * 70)
