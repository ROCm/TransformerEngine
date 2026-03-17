# AITER Fused RoPE Dispatch for FusedRoPEFunc

## Modified Files

### `transformer_engine/pytorch/attention/rope.py`
- Added optional import of `aiter.ops.rope.{rope_fwd, rope_bwd}` with broad exception handling and user-visible warning on failure.
- Introduced `NVTE_USE_AITER_FUSED_ROPE` env var (default `"1"`) to enable/disable the AITER path at module load time.
- Added `FusedRoPEFunc._can_use_aiter()` guard: dispatches only for sbhd format, non-interleaved, cp_size==1, no cu_seqlens/start_positions.
- Modified `FusedRoPEFunc.forward` and `backward` to conditionally call AITER kernels or fall back to TE-native `tex.fused_rope_{forward,backward}`.
- No changes to `RotaryPositionEmbedding`, unfused paths, or `FusedQKVRoPEFunc`.

## New Files

None.

## Test

Pending Docker image rebuild and e2e validation on target hardware with `NVTE_USE_AITER_FUSED_ROPE=1`.
