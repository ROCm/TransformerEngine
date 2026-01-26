import os, torch
import transformer_engine.pytorch as te
from time import time

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

# TE
start = time()

glinear = te.GroupedLinear(E, K, N, bias=False).to(device=device, dtype=dtype)
y_te = glinear(x, m_splits=m_splits)
print("TE time:   ", time()-start)


Ws = []
for e in range(E):
    w = getattr(glinear, f"weight{e}")  # expect [N, K]
    Ws.append(w)
W = torch.stack(Ws, dim=0)              # [E, N, K]
assert W.shape == (E, N, K), f"Unexpected weight shape: {W.shape}"


# Torch
start = time()

ys = []
offset = 0
for e, m in enumerate(m_splits):
    if m == 0:
        continue
    x_e = x[offset:offset+m]          # [m, K]
    y_e = x_e @ W[e].transpose(0, 1)  # [m, N]
    ys.append(y_e)
    offset += m

y_ref = torch.cat(ys, dim=0)
print("Torch time:", time()-start)

# Compare
diff = (y_te.float() - y_ref.float())
max_abs = diff.abs().max().item()
rel = (diff.abs() / (y_ref.float().abs() + 1e-6)).max().item()

print(f"{y_te.shape=}, {y_ref.shape=}")
print("max_abs_err:", max_abs)
print("max_rel_err:", rel)

torch.testing.assert_close(y_te.float(), y_ref.float(), rtol=3e-2, atol=3e-2)
