# Verifying Triton Logic After Operand Swap

## BLAS to Triton Conversion

When converting from BLAS (column-major) to Triton (row-major), we:
1. Swap operands: B becomes first, A becomes second
2. Apply appropriate transposes in Triton kernel

## Case Analysis

### 1. fprop: gemm(W, X, "TN")
**BLAS computes (column-major)**: W^T @ X → result is Y^T
**Row-major interpretation**: Y = X @ W^T

**After operand swap for Triton**:
- First operand: X (was B)
- Second operand: W (was A)
- Need to compute: X @ W^T

**Data selection**:
- W (transA=T): uses rowwise → shape [1024, 768]
- X (transB=N): uses rowwise → shape [128, 768]

**In Triton kernel**:
- X: [128, 768] (no transpose needed)
- W: [1024, 768] → needs transpose to [768, 1024]
- Compute: X[128,768] @ W^T[768,1024] = Y[128,1024] ✓

### 2. dgrad: gemm(W, dY, "NN")
**BLAS computes (column-major)**: W @ dY → result is dX^T
**Row-major interpretation**: dX = dY @ W

**After operand swap for Triton**:
- First operand: dY (was B)
- Second operand: W (was A)
- Need to compute: dY @ W

**Data selection**:
- W (transA=N): uses columnwise → shape [1024, 768]
- dY (transB=N): uses rowwise → shape [128, 1024]

**In Triton kernel**:
- dY: [128, 1024] (no transpose needed)
- W: [1024, 768] (no transpose needed)
- Compute: dY[128,1024] @ W[1024,768] = dX[128,768] ✓

### 3. wgrad: gemm(X, dY, "NT")
**BLAS computes (column-major)**: X @ dY^T → result is dW^T
**Row-major interpretation**: dW = dY^T @ X

**After operand swap for Triton**:
- First operand: dY (was B)
- Second operand: X (was A)
- Need to compute: dY^T @ X

**Data selection**:
- X (transA=N): uses columnwise → shape [128, 768]
- dY (transB=T): uses columnwise → shape [128, 1024]

**In Triton kernel**:
- dY: [128, 1024] → needs transpose to [1024, 128]
- X: [128, 768] (no transpose needed)
- Compute: dY^T[1024,128] @ X[128,768] = dW[1024,768] ✓

## The Issue

After swapping, Triton needs to know when to transpose:
- For fprop: second operand (W) needs transpose
- For dgrad: no transposes needed
- For wgrad: first operand (dY) needs transpose

But wait, the current code applies transpose based on BLAS flags to the wrong operands after swapping!

## Correct Logic

After swapping operands, the transpose flags should also be swapped:
- Original BLAS: transA applies to A, transB applies to B
- After swap: transB applies to first operand (was B), transA applies to second operand (was A)

So in the swapped code:
- First operand uses transB flag
- Second operand uses transA flag