# Correct Triton Mapping for MXFP8

## The Problem

The current MXFP8 implementation doesn't swap operands like regular FP8 does. This is incorrect because we need to match BLAS behavior.

## Understanding the Conversion

### Column-Major (BLAS) vs Row-Major (Triton)

When BLAS computes in column-major: `C = op(A) @ op(B)`
The same matrix in row-major appears as: `C^T`

To get the same result in row-major, we need to compute:
`C^T = op(B)^T @ op(A)^T`

But since we want C (not C^T), we need:
`C = (op(B)^T @ op(A)^T)^T = op(A)^T^T @ op(B)^T^T = op(A) @ op(B)`

Wait, that's not right. Let me think again...

Actually, the key insight is:
- A matrix stored row-major appears transposed to column-major
- So a row-major `A[m,n]` appears as `A^T[n,m]` to column-major

## Correct Mapping

### Case 1: fprop
**BLAS Call**: `gemm(W, X, layout="TN", M=1024, N=batch, K=768)`
- First arg W: `[1024, 768]` with transA=T
- Second arg X: `[batch, 768]` with transB=N

**BLAS Computation (column-major view)**:
- W stored row-major `[1024, 768]` → BLAS sees `W^T[768, 1024]`
- X stored row-major `[batch, 768]` → BLAS sees `X^T[768, batch]`
- With transA=T: op(W^T) = W^T^T = W
- With transB=N: op(X^T) = X^T
- Result: `W @ X^T` → stored as `(W @ X^T)^T = X @ W^T` in row-major

**For Triton (row-major)**:
To compute `X @ W^T`:
- Need first operand: X `[batch, 768]`
- Need second operand: W^T `[768, 1024]`
- So we swap operands: A=X, B=W
- And transpose flags: transA=False (X as-is), transB=True (W needs transpose)

### Case 2: dgrad
**BLAS Call**: `gemm(W, dY, layout="NN", M=768, N=batch, K=1024)`
- First arg W: `[1024, 768]` with transA=N
- Second arg dY: `[batch, 1024]` with transB=N

**BLAS Computation (column-major view)**:
- W stored row-major `[1024, 768]` → BLAS sees `W^T[768, 1024]`
- dY stored row-major `[batch, 1024]` → BLAS sees `dY^T[1024, batch]`
- With transA=N: op(W^T) = W^T
- With transB=N: op(dY^T) = dY^T
- Result: `W^T @ dY^T` → stored as `(W^T @ dY^T)^T = dY @ W` in row-major

**For Triton (row-major)**:
To compute `dY @ W`:
- Need first operand: dY `[batch, 1024]`
- Need second operand: W `[1024, 768]`
- So we swap operands: A=dY, B=W
- And transpose flags: transA=False, transB=False (both as-is)

### Case 3: wgrad
**BLAS Call**: `gemm(X, dY, layout="NT", M=768, N=1024, K=batch)`
- First arg X: `[batch, 768]` with transA=N
- Second arg dY: `[batch, 1024]` with transB=T

**BLAS Computation (column-major view)**:
- X stored row-major `[batch, 768]` → BLAS sees `X^T[768, batch]`
- dY stored row-major `[batch, 1024]` → BLAS sees `dY^T[1024, batch]`
- With transA=N: op(X^T) = X^T
- With transB=T: op(dY^T) = dY^T^T = dY
- Result: `X^T @ dY` → stored as `(X^T @ dY)^T = dY^T @ X` in row-major

**For Triton (row-major)**:
To compute `dY^T @ X`:
- Need first operand: dY^T `[1024, batch]`
- Need second operand: X `[batch, 768]`
- So we swap operands: A=dY, B=X
- And transpose flags: transA=True (dY needs transpose), transB=False (X as-is)

## Summary: Triton Should Use

| BLAS Call | BLAS transA/B | Triton A | Triton B | Triton transA | Triton transB |
|-----------|---------------|----------|----------|---------------|---------------|
| gemm(W, X, "TN") | T, N | X | W | False | True |
| gemm(W, dY, "NN") | N, N | dY | W | False | False |
| gemm(X, dY, "NT") | N, T | dY | X | True | False |

## Key Pattern

For Triton:
1. **Swap the operands**: Second BLAS arg becomes first Triton arg
2. **Swap and invert transpose flags**:
   - Triton transA = BLAS transB
   - Triton transB = BLAS transA

## MXFP8 Selection Based on Triton Flags

After swapping, for Triton's flags:

| Operation | Triton transA | Triton transB | A Selection | B Selection |
|-----------|---------------|---------------|-------------|-------------|
| fprop | False | True | X: rowwise | W: needs transpose |
| dgrad | False | False | dY: rowwise | W: columnwise |
| wgrad | True | False | dY: needs transpose | X: columnwise |

For transpose cases, we need the format that gives correct scales after logical transpose.