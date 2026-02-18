# Forward Pass (fprop) Analysis for Triton MXFP8 Implementation

## Overview
Forward pass computes: `Y = X @ W^T` where:
- X (input): `[batch, in_features]`
- W (weight): `[out_features, in_features]`
- Y (output): `[batch, out_features]`

## Concrete Example Dimensions
- Weight: `W[1024, 768]` (out_features=1024, in_features=768)
- Input: `X[batch, 768]`
- Output: `Y[batch, 1024]`

---

## 1. The Computation (Row-Major Perspective)

### What we want to compute:
```
Y[batch, 1024] = X[batch, 768] @ W^T[768, 1024]
```

### In Triton (row-major), this is directly:
```python
Y = matmul(X, W.T)  # Row-major computation
```

---

## 2. GEMM Call Analysis

### From the codebase (`linear.py`):
```python
gemm_out = general_gemm(
    weightmat,      # W[1024, 768]
    inputmat_total, # X[batch, 768]
    layout="TN",    # transA=True, transB=False
    ...
)
```

### What this means:
- First operand (A): `W[1024, 768]` with `transA=True` → computes `W^T`
- Second operand (B): `X[batch, 768]` with `transB=False` → uses `X` as-is
- Result: `Y = W^T @ X` in BLAS column-major
- When read as row-major: `Y = X @ W^T` ✓

---

## 3. Triton Requirements (Row-Major)

### For computing `Y = X @ W^T`:

**First operand (X):**
- Needs: `[batch, 768]`
- Scale needs: `[batch, 768/32] = [batch, 24]`
- Meaning: Each input row has 24 scale blocks along in_features

**Second operand (W^T):**
- Needs: `[768, 1024]` (transposed from W)
- Scale needs: `[768/32, 1024] = [24, 1024]`
- Meaning: Each output column has 24 scale blocks along in_features

---

## 4. MXFP8 Data Selection for Triton

### For Input (X):
**Shape:** `[batch, 768]` with `transA=False`

**Selection:** Use X rowwise
- Data: `[batch, 768]` ✓
- Scales: `[batch, 24]` ✓
- **Perfect match!** Rowwise quantization gives exactly what we need.

### For Weight (W):
**Shape:** `[1024, 768]` with `transA=True` (need `W^T[768, 1024]`)

**Option 1: W rowwise (doesn't work)**
- Data: `[1024, 768]`
- After transpose: `[768, 1024]` ✓
- Scales: `[1024, 24]`
- After transpose: `[24, 1024]`? No! Transposing data doesn't correctly transpose scales
- ✗ Scale layout is wrong

**Option 2: W columnwise (WORKS!)**
- Stored as: `[768, 1024]` (already transposed in storage!)
- Scales: `[768/32, 1024] = [24, 1024]`
- **This is exactly W^T with the right scale layout!**
- ✓ Perfect match without any additional transpose

---

## 5. The Complete Forward Pass Solution

### Data Selection:
```python
# For fprop: Y = X @ W^T
# Layout: TN (transA=True, transB=False)

# Input X (transB=False):
X_data = X._rowwise_data         # [batch, 768]
X_scale = X._rowwise_scale_inv   # [batch, 24]

# Weight W (transA=True):
W_data = W._columnwise_data      # [768, 1024] (stored as W^T)
W_scale = W._columnwise_scale_inv # [24, 1024]

# Direct computation in Triton:
Y = tl.dot_scaled(
    X_data, X_scale,  # [batch, 768] with scales [batch, 24]
    W_data, W_scale,  # [768, 1024] with scales [24, 1024]
)
```

### Why this works:
1. **Input uses rowwise:** Scales along in_features dimension (768)
2. **Weight uses columnwise:** Already stored as W^T with correct scale layout
3. **Both accumulate along in_features:** The 768 dimension with 24 blocks
4. **Scales align perfectly:** Both have 24 scale blocks along the reduction dimension

---

## 6. Comparison with BLAS/C++ Implementation

### C++ Selection (from the document):
- Weight: `transA=True` → uses **rowwise**
- Input: `transB=False` → uses **rowwise**

### Triton Selection (our analysis):
- Input: `transB=False` → uses **rowwise** (same as C++)
- Weight: `transA=True` → uses **columnwise** (different from C++!)

### Why the difference?
- **C++ (column-major):** Needs to convert everything to TN layout
- **Triton (row-major):** Can directly use the natural layout
- **Weight columnwise:** Is already stored as W^T, perfect for Triton!

---

## 7. Memory Access Pattern

### Input (X rowwise):
```
X[batch, 768] with scales[batch, 24]:

Row 0: [block0(32) | block1(32) | ... | block23(32)]
       scale[0,0]   scale[0,1]   ...   scale[0,23]

Row 1: [block0(32) | block1(32) | ... | block23(32)]
       scale[1,0]   scale[1,1]   ...   scale[1,23]
```

### Weight (W columnwise = W^T):
```
W^T[768, 1024] with scales[24, 1024]:

        col0    col1    ...  col1023
block0  [...]   [...]   ...  [...]    (elements 0-31 of each column)
scale:  s[0,0]  s[0,1]  ...  s[0,1023]

block1  [...]   [...]   ...  [...]    (elements 32-63 of each column)
scale:  s[1,0]  s[1,1]  ...  s[1,1023]

...

block23 [...]   [...]   ...  [...]    (elements 736-767 of each column)
scale:  s[23,0] s[23,1] ...  s[23,1023]
```

---

## 8. Key Insights

1. **Weight columnwise is magic:** It's already stored as W^T with perfect scale layout
2. **No transpose needed:** Unlike BLAS which forces TN, Triton can use natural layouts
3. **Scales align perfectly:** Both operands have 24 blocks along the 768 dimension
4. **Memory efficient:** No data movement, just use the right pre-stored format

---

## 9. Summary Table

| Aspect | Input (X) | Weight (W) |
|--------|-----------|------------|
| **Original shape** | `[batch, 768]` | `[1024, 768]` |
| **Transpose needed** | No | Yes (need W^T) |
| **MXFP8 format** | Rowwise | Columnwise |
| **Actual data** | `[batch, 768]` | `[768, 1024]` (stored as W^T) |
| **Scale shape** | `[batch, 24]` | `[24, 1024]` |
| **Scale meaning** | 24 blocks per row | 24 blocks per column |
| **Accumulation dim** | in_features (768) | in_features (768) |

This shows that for fprop with Triton, we should:
- Use **rowwise** for inputs (same as C++)
- Use **columnwise** for weights (different from C++ which uses rowwise)

The columnwise weight storage naturally gives us W^T with the correct scale layout for `tl.dot_scaled`!