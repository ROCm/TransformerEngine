import os
import time
import torch
import transformer_engine.pytorch as te

torch.manual_seed(0)

os.environ["NVTE_USE_CK_GROUPED_GEMM"] = "1"
os.environ["NVTE_CK_GROUPED_GEMM_WARN_FALLBACK"] = "1"

device = "cuda"
dtype  = torch.bfloat16

E = 4
K = 1024
N = 2048
m_splits = [128, 64, 0, 256]
M_total = sum(m_splits)

x = torch.randn(M_total, K, device=device, dtype=dtype)

# Timing helper
def bench_cuda(fn, warmup=20, iters=100, name=""):
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) * 1000.0 / iters
    if name:
        print(f"{name}: {avg_ms:.3f} ms (avg over {iters} runs, {warmup} warmup)")
    return avg_ms

# TE GroupedLinear
glinear = te.GroupedLinear(E, K, N, bias=False).to(device=device, dtype=dtype)

def te_run():
    return glinear(x, m_splits=m_splits)

te_ms = bench_cuda(te_run, warmup=20, iters=100, name="TE GroupedLinear")

# Grab weights for reference path
Ws = [getattr(glinear, f"weight{e}") for e in range(E)]  # each [N, K]
W = torch.stack(Ws, dim=0)                               # [E, N, K]
assert W.shape == (E, N, K), f"Unexpected weight shape: {W.shape}"

# Torch reference (group loop)
offsets = []
off = 0
for m in m_splits:
    offsets.append(off)
    off += m

y_ref_buf = torch.empty((M_total, N), device=device, dtype=dtype)

def torch_run():
    # Fill the preallocated buffer
    for e, m in enumerate(m_splits):
        if m == 0:
            continue
        o = offsets[e]
        y_ref_buf[o:o+m].copy_(x[o:o+m] @ W[e].transpose(0, 1))
    return y_ref_buf

torch_ms = bench_cuda(torch_run, warmup=20, iters=100, name="Torch loop (prealloc out)")

# Compare outputs
y_te = te_run()
y_ref = torch_run().clone()

diff = (y_te.float() - y_ref.float())
max_abs = diff.abs().max().item()
rel = (diff.abs() / (y_ref.float().abs() + 1e-6)).max().item()

print(f"\nErrors:")
print(f"  {y_te.shape=}, {y_ref.shape=}")
print("  max_abs_err:", max_abs)
print("  max_rel_err:", rel)

torch.testing.assert_close(y_te.float(), y_ref.float(), rtol=3e-2, atol=3e-2)

print(f"\nTiming:")
print(f"  TE avg:    {te_ms:.3f} ms")
print(f"  Torch avg: {torch_ms:.3f} ms")
print(f"  Speedup:   {torch_ms/te_ms:.2f}x (Torch / TE)")
