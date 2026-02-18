# Triton API Analysis - Matching BLAS Calls

## Key Context
- **BLAS**: Column-major, row-major data appears transposed
- **Triton**: Row-major, data appears as-is
- Same API calls but different interpretations!

## API Call Analysis

### 1. Forward Pass (fprop)

**BLAS Call**: `general_gemm(W, X, layout="TN")`
- W: `[1024, 768]` (out_features, in_features)
- X: `[batch, 768]` (batch, in_features)
- transA=T, transB=N

**What BLAS sees (column-major)**:
- W appears as `W^T[768, 1024]`
- X appears as `X^T[768, batch]`
- Computes: `W^T^T @ X^T = W @ X^T`
- Result appears as `Y^T[1024, batch]`
- When read row-major: `Y[batch, 1024]` ✓

**What Triton sees (row-major)**:
- W is `[1024, 768]`
- X is `[batch, 768]`
- With transA=T, transB=N: computes `W^T @ X`
- But wait! This gives `[768, 1024] @ [batch, 768]` - dimension mismatch!

**The Issue**: In Triton, we need to swap operands!
- Triton should compute: `X @ W^T`
- So we need: A=X, B=W with transA=False, transB=True

### 2. Backward dgrad

**BLAS Call**: `general_gemm(W, dY, layout="NN")`
- W: `[1024, 768]`
- dY: `[batch, 1024]`
- transA=N, transB=N

**What BLAS sees (column-major)**:
- W appears as `W^T[768, 1024]`
- dY appears as `dY^T[1024, batch]`
- Computes: `W^T @ dY^T`
- Result appears as `dX^T[768, batch]`
- When read row-major: `dX[batch, 768]` ✓

**What Triton sees (row-major)**:
- W is `[1024, 768]`
- dY is `[batch, 1024]`
- With transA=N, transB=N: computes `W @ dY`
- This gives `[1024, 768] @ [batch, 1024]` - dimension mismatch!

**The Issue**: Need to swap operands!
- Triton should compute: `dY @ W`
- So we need: A=dY, B=W with transA=False, transB=False

### 3. Backward wgrad

**BLAS Call**: `general_gemm(X, dY, layout="NT")`
- X: `[batch, 768]`
- dY: `[batch, 1024]`
- transA=N, transB=T

**What BLAS sees (column-major)**:
- X appears as `X^T[768, batch]`
- dY appears as `dY^T[1024, batch]`
- Computes: `X^T @ dY^T^T = X^T @ dY`
- Result appears as `dW^T[768, 1024]`
- When read row-major: `dW[1024, 768]` ✓

**What Triton sees (row-major)**:
- X is `[batch, 768]`
- dY is `[batch, 1024]`
- With transA=N, transB=T: computes `X @ dY^T`
- This gives `[batch, 768] @ [1024, batch]` - dimension mismatch!

**The Issue**: Need to swap AND adjust transposes!
- Triton should compute: `dY^T @ X`
- So we need: A=dY, B=X with transA=True, transB=False

## Operand Swapping Pattern

For row-major Triton to match column-major BLAS results:

| BLAS Call | BLAS Layout | Triton Needs | Triton Call |
|-----------|-------------|--------------|-------------|
| gemm(A, B, "TN") | A^T @ B | B @ A^T | gemm(B, A, "NT") |
| gemm(A, B, "NN") | A @ B | B^T @ A^T → swap | gemm(B, A, "NN") |
| gemm(A, B, "NT") | A @ B^T | B^T @ A → swap+flip | gemm(B, A, "TN") |
| gemm(A, B, "TT") | A^T @ B^T | B^T @ A | gemm(B, A, "TT") |

Wait, let me reconsider. Actually I think the issue is simpler...

## The Real Issue: Operand Order

When BLAS (column-major) computes `C = A @ B`:
- The result C in column-major is equivalent to `C^T = B^T @ A^T` in row-major

So for Triton (row-major) to get the same result:
- We need to swap operands: compute `B @ A` instead of `A @ B`
- But we DON'T flip the transpose flags

Let me recalculate...

## Correct Mapping

| Operation | BLAS Call | BLAS Computes | Triton Should Compute | Triton Call |
|-----------|-----------|---------------|----------------------|-------------|
| fprop | gemm(W, X, "TN") | W^T @ X (col-major) | X @ W^T (row-major) | gemm(X, W, "NT") |
| dgrad | gemm(W, dY, "NN") | W @ dY (col-major) | dY @ W (row-major) | gemm(dY, W, "NN") |
| wgrad | gemm(X, dY, "NT") | X @ dY^T (col-major) | dY^T @ X (row-major) | gemm(dY, X, "TN") |

## Key Insight

The Triton implementation needs to:
1. **Swap the operands** (B, A instead of A, B)
2. **Keep the same transpose flags** but applied to swapped operands

So if BLAS calls gemm(A, B, "TN"):
- Triton should call with (B, A) and same flags "TN"
- But this means transB for BLAS becomes transA for Triton!

Actually, wait... I think I'm overcomplicating. Let me check the actual code to see how it handles this.