# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Tests for MORI Expert Parallelism integration in tealite.

This test suite validates the MORI EP integration at two levels:

1. **Unit tests** (no GPU / no MORI required): Test config validation, API
   surface, initialization guards, and error handling using mocks.

2. **Multi-GPU integration tests** (require MORI + multiple AMD GPUs): Test
   actual dispatch/combine with real MORI kernels. Skipped automatically when
   MORI is not installed or insufficient GPUs are available.

Run unit tests (always works):
    pytest tests/pytorch/test_lite_mori_ep.py -v -k "unit"

Run integration tests (requires MORI + multi-GPU):
    pytest tests/pytorch/test_lite_mori_ep.py -v -k "integration"
"""

import os
import sys
import pytest
from unittest import mock
from unittest.mock import MagicMock, patch

import torch

os.environ["NVTE_LITE"] = "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mori_installed():
    try:
        import mori  # noqa: F401
        return True
    except ImportError:
        return False


def _gpu_count():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


skip_no_mori = pytest.mark.skipif(
    not _mori_installed(), reason="MORI not installed"
)
skip_insufficient_gpus = pytest.mark.skipif(
    _gpu_count() < 2, reason=f"Need >=2 GPUs, found {_gpu_count()}"
)


# ---------------------------------------------------------------------------
# Unit Tests -- no MORI or GPU required
# ---------------------------------------------------------------------------

class TestMoriEPAvailability:
    """Test availability detection and import guards."""

    def test_mori_ep_available_returns_bool(self):
        from transformer_engine.pytorch._lite.mori_ep import mori_ep_available
        result = mori_ep_available()
        assert isinstance(result, bool)

    def test_mori_ep_available_reflects_import(self):
        """mori_ep_available() should return True iff mori is importable."""
        from transformer_engine.pytorch._lite.mori_ep import mori_ep_available
        expected = _mori_installed()
        assert mori_ep_available() == expected


class TestMoriEPInitGuards:
    """Test that init_mori_ep enforces prerequisites."""

    def test_init_without_mori_raises(self):
        """init_mori_ep raises RuntimeError when MORI is not installed."""
        import transformer_engine.pytorch._lite.mori_ep as mod

        with mock.patch.object(mod, "_mori_available", False), \
             mock.patch.object(mod, "_mori", None):
            with pytest.raises(RuntimeError, match="MORI is not installed"):
                # Reset cached state so it re-checks
                orig = mod._mori_available
                mod._mori_available = False
                try:
                    mod.init_mori_ep()
                finally:
                    mod._mori_available = orig

    def test_init_without_dist_raises(self):
        """init_mori_ep raises RuntimeError if torch.distributed not initialized."""
        import transformer_engine.pytorch._lite.mori_ep as mod

        mock_mori = MagicMock()
        with mock.patch.object(mod, "_mori_available", True), \
             mock.patch.object(mod, "_mori", mock_mori), \
             mock.patch.object(mod, "_mori_shmem_initialized", False), \
             mock.patch("torch.distributed.is_initialized", return_value=False):
            with pytest.raises(RuntimeError, match="torch.distributed must be initialized"):
                mod.init_mori_ep()

    def test_init_idempotent(self):
        """Calling init_mori_ep when already initialized is a no-op."""
        import transformer_engine.pytorch._lite.mori_ep as mod

        with mock.patch.object(mod, "_mori_shmem_initialized", True):
            # Should not raise or call anything
            mod.init_mori_ep()

    def test_finalize_when_not_initialized_is_noop(self):
        """finalize_mori_ep when not initialized should be a no-op."""
        import transformer_engine.pytorch._lite.mori_ep as mod

        with mock.patch.object(mod, "_mori_shmem_initialized", False):
            mod.finalize_mori_ep()  # should not raise


class TestMoriExpertParallelConfig:
    """Test MoriExpertParallel config validation (mocked MORI)."""

    def _make_ep(self, **kwargs):
        """Create a MoriExpertParallel with mocked MORI backend."""
        import transformer_engine.pytorch._lite.mori_ep as mod

        mock_mori = MagicMock()
        # Mock the kernel type enum
        mock_kt = MagicMock()
        for name in ["IntraNode", "InterNode", "InterNodeV1", "InterNodeV1LL", "AsyncLL"]:
            setattr(mock_kt, name, name)
        mock_mori.ops.EpDispatchCombineKernelType = mock_kt
        mock_mori.ops.EpDispatchCombineConfig = MagicMock()
        mock_mori.ops.EpDispatchCombineOp = MagicMock()

        defaults = dict(
            rank=0,
            world_size=8,
            hidden_dim=7168,
            num_experts_per_rank=32,
            num_experts_per_token=8,
            max_num_inp_token_per_rank=4096,
        )
        defaults.update(kwargs)

        with mock.patch.object(mod, "_mori_available", True), \
             mock.patch.object(mod, "_mori", mock_mori), \
             mock.patch.object(mod, "_mori_shmem_initialized", True):
            ep = mod.MoriExpertParallel(**defaults)

        return ep, mock_mori

    def test_default_construction(self):
        ep, mock_mori = self._make_ep()
        assert ep.rank == 0
        assert ep.world_size == 8
        assert ep.hidden_dim == 7168
        assert ep.num_experts_per_rank == 32
        assert ep.num_experts_per_token == 8
        assert ep.num_experts == 256  # 32 * 8

        # Verify MORI config was created with correct params
        mock_mori.ops.EpDispatchCombineConfig.assert_called_once()
        call_kwargs = mock_mori.ops.EpDispatchCombineConfig.call_args
        assert call_kwargs.kwargs["rank"] == 0
        assert call_kwargs.kwargs["world_size"] == 8
        assert call_kwargs.kwargs["hidden_dim"] == 7168

    def test_invalid_kernel_type_raises(self):
        import transformer_engine.pytorch._lite.mori_ep as mod

        mock_mori = MagicMock()
        with mock.patch.object(mod, "_mori_available", True), \
             mock.patch.object(mod, "_mori", mock_mori), \
             mock.patch.object(mod, "_mori_shmem_initialized", True):
            with pytest.raises(ValueError, match="Unknown kernel_type"):
                mod.MoriExpertParallel(
                    rank=0, world_size=8, hidden_dim=128,
                    num_experts_per_rank=4, num_experts_per_token=2,
                    max_num_inp_token_per_rank=64,
                    kernel_type="nonexistent",
                )

    @pytest.mark.parametrize("kernel_type,expected", [
        ("intra_node", "IntraNode"),
        ("inter_node", "InterNode"),
        ("inter_node_v1", "InterNodeV1"),
        ("inter_node_v1_ll", "InterNodeV1LL"),
        ("async_ll", "AsyncLL"),
    ])
    def test_kernel_type_mapping(self, kernel_type, expected):
        ep, mock_mori = self._make_ep(kernel_type=kernel_type)
        call_kwargs = mock_mori.ops.EpDispatchCombineConfig.call_args.kwargs
        assert call_kwargs["kernel_type"] == expected

    def test_not_initialized_raises(self):
        """Creating MoriExpertParallel without init raises RuntimeError."""
        import transformer_engine.pytorch._lite.mori_ep as mod

        mock_mori = MagicMock()
        with mock.patch.object(mod, "_mori_available", True), \
             mock.patch.object(mod, "_mori", mock_mori), \
             mock.patch.object(mod, "_mori_shmem_initialized", False):
            with pytest.raises(RuntimeError, match="MORI shmem not initialized"):
                mod.MoriExpertParallel(
                    rank=0, world_size=8, hidden_dim=128,
                    num_experts_per_rank=4, num_experts_per_token=2,
                    max_num_inp_token_per_rank=64,
                )

    def test_fp8_direct_cast_sets_external_buf(self):
        ep, mock_mori = self._make_ep(quant_type="fp8_direct_cast")
        call_kwargs = mock_mori.ops.EpDispatchCombineConfig.call_args.kwargs
        assert call_kwargs["use_external_inp_buf"] is True

    def test_no_quant_unsets_external_buf(self):
        ep, mock_mori = self._make_ep(quant_type="none")
        call_kwargs = mock_mori.ops.EpDispatchCombineConfig.call_args.kwargs
        assert call_kwargs["use_external_inp_buf"] is False


class TestMoriExpertParallelDispatchCombine:
    """Test dispatch/combine API surface with mocked MORI backend."""

    def _make_ep_with_mock_op(self):
        """Create EP with a fully mocked dispatch/combine operator."""
        import transformer_engine.pytorch._lite.mori_ep as mod

        mock_mori = MagicMock()
        mock_kt = MagicMock()
        mock_kt.IntraNode = "IntraNode"
        mock_mori.ops.EpDispatchCombineKernelType = mock_kt
        mock_mori.ops.EpDispatchCombineConfig = MagicMock()

        # Mock the op returned by EpDispatchCombineOp()
        mock_op = MagicMock()
        mock_mori.ops.EpDispatchCombineOp.return_value = mock_op

        with mock.patch.object(mod, "_mori_available", True), \
             mock.patch.object(mod, "_mori", mock_mori), \
             mock.patch.object(mod, "_mori_shmem_initialized", True):
            ep = mod.MoriExpertParallel(
                rank=0, world_size=2,
                hidden_dim=128,
                num_experts_per_rank=4,
                num_experts_per_token=2,
                max_num_inp_token_per_rank=32,
            )

        return ep, mock_op

    @patch("torch.cuda.synchronize")
    def test_dispatch_calls_mori_op(self, mock_sync):
        ep, mock_op = self._make_ep_with_mock_op()

        # Set up mock return for dispatch
        num_recv = 10
        hidden_dim = 128
        topk = 2
        mock_op.dispatch.return_value = (
            torch.zeros(num_recv, hidden_dim),       # out tokens
            torch.zeros(num_recv, topk),              # out weights
            None,                                     # out scales
            torch.zeros(num_recv, topk, dtype=torch.int32),  # out indices
            torch.tensor([num_recv], dtype=torch.int32),     # total_recv
        )

        input_t = torch.randn(8, hidden_dim)
        weights = torch.rand(8, topk)
        indices = torch.randint(0, 8, (8, topk), dtype=torch.int32)

        recv_tokens, recv_weights, recv_indices, n_recv = ep.dispatch(
            input_t, weights, indices,
        )

        mock_op.dispatch.assert_called_once()
        assert n_recv == num_recv
        assert recv_tokens.shape == (num_recv, hidden_dim)
        mock_sync.assert_called()

    @patch("torch.cuda.synchronize")
    def test_combine_calls_mori_op(self, mock_sync):
        ep, mock_op = self._make_ep_with_mock_op()

        hidden_dim = 128
        topk = 2
        max_tokens = 32
        mock_op.combine.return_value = (
            torch.zeros(max_tokens, hidden_dim),  # output
            torch.zeros(max_tokens, topk),        # output_weights
        )

        expert_out = torch.randn(10, hidden_dim)
        weights = torch.rand(10, topk)
        indices = torch.randint(0, 8, (10, topk), dtype=torch.int32)

        output, output_weights = ep.combine(expert_out, weights, indices)

        mock_op.combine.assert_called_once()
        assert output.shape == (max_tokens, hidden_dim)
        mock_sync.assert_called()

    def test_reset_calls_mori_op(self):
        ep, mock_op = self._make_ep_with_mock_op()
        ep.reset()
        mock_op.reset.assert_called_once()

    @patch("torch.cuda.synchronize")
    def test_dispatch_casts_int64_indices_to_int32(self, mock_sync):
        ep, mock_op = self._make_ep_with_mock_op()

        hidden_dim = 128
        topk = 2
        mock_op.dispatch.return_value = (
            torch.zeros(5, hidden_dim),
            torch.zeros(5, topk),
            None,
            torch.zeros(5, topk, dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
        )

        input_t = torch.randn(8, hidden_dim)
        weights = torch.rand(8, topk)
        indices_int64 = torch.randint(0, 8, (8, topk), dtype=torch.int64)

        ep.dispatch(input_t, weights, indices_int64)

        # Check that the indices passed to MORI are int32
        call_args = mock_op.dispatch.call_args
        actual_indices = call_args.args[3]  # 4th positional arg
        assert actual_indices.dtype == torch.int32

    @patch("torch.cuda.synchronize")
    def test_dispatch_and_combine_full_cycle(self, mock_sync):
        """Test the convenience dispatch_and_combine method."""
        ep, mock_op = self._make_ep_with_mock_op()

        hidden_dim = 128
        topk = 2
        num_recv = 6

        mock_op.dispatch.return_value = (
            torch.randn(num_recv, hidden_dim),
            torch.rand(num_recv, topk),
            None,
            torch.randint(0, 8, (num_recv, topk), dtype=torch.int32),
            torch.tensor([num_recv], dtype=torch.int32),
        )
        mock_op.combine.return_value = (
            torch.randn(32, hidden_dim),
            torch.rand(32, topk),
        )

        # Expert function: identity
        def expert_fn(tokens, indices, n):
            return tokens

        input_t = torch.randn(8, hidden_dim)
        weights = torch.rand(8, topk)
        indices = torch.randint(0, 8, (8, topk), dtype=torch.int32)

        output, output_weights = ep.dispatch_and_combine(
            input_t, weights, indices, expert_fn,
        )

        mock_op.dispatch.assert_called_once()
        mock_op.combine.assert_called_once()
        mock_op.reset.assert_called_once()
        assert output.shape[1] == hidden_dim


class TestMaskToIndex:
    """Test mask-map ↔ index-map routing conversion."""

    def test_basic_conversion(self):
        from transformer_engine.pytorch._lite.mori_ep import mask_to_index

        # token 0 → experts 1,3; token 1 → experts 0,2
        mask = torch.tensor([[0, 1, 0, 1],
                             [1, 0, 1, 0]], dtype=torch.int32)
        probs = torch.tensor([[0.0, 0.3, 0.0, 0.7],
                              [0.5, 0.0, 0.5, 0.0]], dtype=torch.float32)

        indices, weights = mask_to_index(mask, probs)

        assert indices.shape == (2, 2)
        assert indices.dtype == torch.int32
        assert weights.shape == (2, 2)

        # nonzero produces sorted columns, so expert order is ascending
        assert indices[0].tolist() == [1, 3]
        assert indices[1].tolist() == [0, 2]
        assert torch.allclose(weights[0], torch.tensor([0.3, 0.7]))
        assert torch.allclose(weights[1], torch.tensor([0.5, 0.5]))

    def test_no_probs_gives_uniform_weights(self):
        from transformer_engine.pytorch._lite.mori_ep import mask_to_index

        mask = torch.tensor([[1, 0, 1],
                             [0, 1, 1]], dtype=torch.int32)
        indices, weights = mask_to_index(mask, probs=None)

        assert indices[0].tolist() == [0, 2]
        assert indices[1].tolist() == [1, 2]
        assert torch.all(weights == 1.0)

    def test_single_expert_per_token(self):
        from transformer_engine.pytorch._lite.mori_ep import mask_to_index

        mask = torch.tensor([[0, 0, 1],
                             [1, 0, 0],
                             [0, 1, 0]], dtype=torch.int32)
        indices, weights = mask_to_index(mask)

        assert indices.shape == (3, 1)
        assert indices.flatten().tolist() == [2, 0, 1]

    def test_empty_input(self):
        from transformer_engine.pytorch._lite.mori_ep import mask_to_index

        mask = torch.zeros(0, 4, dtype=torch.int32)
        indices, weights = mask_to_index(mask)
        assert indices.shape[0] == 0
        assert weights.shape[0] == 0

    @pytest.mark.parametrize("topk", [1, 2, 4, 8])
    def test_various_topk(self, topk):
        from transformer_engine.pytorch._lite.mori_ep import mask_to_index

        num_tokens, num_experts = 16, 32
        mask = torch.zeros(num_tokens, num_experts, dtype=torch.int32)
        for i in range(num_tokens):
            chosen = torch.randperm(num_experts)[:topk].sort().values
            mask[i, chosen] = 1

        indices, weights = mask_to_index(mask)
        assert indices.shape == (num_tokens, topk)
        # Each row's experts should match the mask
        for i in range(num_tokens):
            expected = mask[i].nonzero(as_tuple=False).flatten().to(torch.int32)
            assert torch.equal(indices[i], expected)


class TestIndexToMask:
    """Test index-map → mask-map conversion."""

    def test_basic_conversion(self):
        from transformer_engine.pytorch._lite.mori_ep import index_to_mask

        indices = torch.tensor([[1, 3], [0, 2]], dtype=torch.int32)
        weights = torch.tensor([[0.3, 0.7], [0.5, 0.5]], dtype=torch.float32)

        mask, probs = index_to_mask(indices, num_experts=4, weights=weights)

        assert mask.shape == (2, 4)
        assert mask.dtype == torch.int32
        assert mask[0].tolist() == [0, 1, 0, 1]
        assert mask[1].tolist() == [1, 0, 1, 0]
        assert probs is not None
        assert torch.allclose(probs[0], torch.tensor([0.0, 0.3, 0.0, 0.7]))
        assert torch.allclose(probs[1], torch.tensor([0.5, 0.0, 0.5, 0.0]))

    def test_no_weights(self):
        from transformer_engine.pytorch._lite.mori_ep import index_to_mask

        indices = torch.tensor([[2], [0]], dtype=torch.int32)
        mask, probs = index_to_mask(indices, num_experts=3)

        assert mask[0].tolist() == [0, 0, 1]
        assert mask[1].tolist() == [1, 0, 0]
        assert probs is None


class TestRoundtrip:
    """Test mask→index→mask and index→mask→index round-trips."""

    def test_mask_index_mask(self):
        from transformer_engine.pytorch._lite.mori_ep import mask_to_index, index_to_mask

        num_tokens, num_experts, topk = 8, 16, 3
        mask_orig = torch.zeros(num_tokens, num_experts, dtype=torch.int32)
        probs_orig = torch.zeros(num_tokens, num_experts, dtype=torch.float32)
        for i in range(num_tokens):
            chosen = torch.randperm(num_experts)[:topk].sort().values
            mask_orig[i, chosen] = 1
            probs_orig[i, chosen] = torch.rand(topk)

        indices, weights = mask_to_index(mask_orig, probs_orig)
        mask_rt, probs_rt = index_to_mask(indices, num_experts, weights)

        assert torch.equal(mask_orig, mask_rt)
        assert torch.allclose(probs_orig, probs_rt)

    def test_index_mask_index(self):
        from transformer_engine.pytorch._lite.mori_ep import mask_to_index, index_to_mask

        num_tokens, num_experts, topk = 8, 16, 2
        # Generate sorted indices (mask_to_index produces sorted output)
        indices_orig = torch.stack([
            torch.randperm(num_experts)[:topk].sort().values
            for _ in range(num_tokens)
        ]).to(torch.int32)
        weights_orig = torch.rand(num_tokens, topk)

        mask, probs = index_to_mask(indices_orig, num_experts, weights_orig)
        indices_rt, weights_rt = mask_to_index(mask, probs)

        assert torch.equal(indices_orig, indices_rt)
        assert torch.allclose(weights_orig, weights_rt)


class TestExportedSymbols:
    """Verify symbols are exported from _lite/__init__.py."""

    def test_mori_ep_symbols_exported(self):
        from transformer_engine.pytorch._lite import (
            mori_ep_available,
            init_mori_ep,
            finalize_mori_ep,
            is_mori_ep_initialized,
            MoriExpertParallel,
        )
        assert callable(mori_ep_available)
        assert callable(init_mori_ep)
        assert callable(finalize_mori_ep)
        assert callable(is_mori_ep_initialized)
        assert callable(MoriExpertParallel)

    def test_autograd_symbols_exported(self):
        from transformer_engine.pytorch._lite import (
            MoriEPDispatch,
            MoriEPCombine,
        )
        assert callable(MoriEPDispatch.apply)
        assert callable(MoriEPCombine.apply)

    def test_routing_conversion_symbols_exported(self):
        from transformer_engine.pytorch._lite import mask_to_index, index_to_mask
        assert callable(mask_to_index)
        assert callable(index_to_mask)

    def test_std_moe_symbols_exported(self):
        from transformer_engine.pytorch._lite import (
            MoriEPDispatchStdMoE,
            MoriEPCombineStdMoE,
        )
        assert callable(MoriEPDispatchStdMoE.apply)
        assert callable(MoriEPCombineStdMoE.apply)


# ---------------------------------------------------------------------------
# Autograd Unit Tests
# ---------------------------------------------------------------------------

class TestMoriEPCycleState:
    """Test the shared cycle state object."""

    def _make_ep_and_state(self):
        import transformer_engine.pytorch._lite.mori_ep as mod

        mock_mori = MagicMock()
        mock_kt = MagicMock()
        mock_kt.IntraNode = "IntraNode"
        mock_mori.ops.EpDispatchCombineKernelType = mock_kt
        mock_mori.ops.EpDispatchCombineConfig = MagicMock()
        mock_mori.ops.EpDispatchCombineOp.return_value = MagicMock()

        with mock.patch.object(mod, "_mori_available", True), \
             mock.patch.object(mod, "_mori", mock_mori), \
             mock.patch.object(mod, "_mori_shmem_initialized", True):
            ep = mod.MoriExpertParallel(
                rank=0, world_size=2, hidden_dim=64,
                num_experts_per_rank=4, num_experts_per_token=2,
                max_num_inp_token_per_rank=16,
            )
            state = ep.new_cycle()

        return ep, state

    def test_new_cycle_returns_state(self):
        from transformer_engine.pytorch._lite.mori_ep import _MoriEPCycleState
        ep, state = self._make_ep_and_state()
        assert isinstance(state, _MoriEPCycleState)
        assert state.ep is ep

    def test_initial_state_is_empty(self):
        ep, state = self._make_ep_and_state()
        assert state.fwd_weights is None
        assert state.fwd_indices is None
        assert state.fwd_num_input == 0
        assert state.fwd_num_recv == 0
        assert state.bwd_recv_weights is None
        assert state.bwd_recv_indices is None
        assert state.bwd_num_recv == 0


class TestMoriEPDispatchAutograd:
    """Test MoriEPDispatch autograd function with mocked MORI backend."""

    def _make_ep_and_mock(self):
        import transformer_engine.pytorch._lite.mori_ep as mod

        mock_mori = MagicMock()
        mock_kt = MagicMock()
        mock_kt.IntraNode = "IntraNode"
        mock_mori.ops.EpDispatchCombineKernelType = mock_kt
        mock_mori.ops.EpDispatchCombineConfig = MagicMock()

        mock_op = MagicMock()
        mock_mori.ops.EpDispatchCombineOp.return_value = mock_op

        with mock.patch.object(mod, "_mori_available", True), \
             mock.patch.object(mod, "_mori", mock_mori), \
             mock.patch.object(mod, "_mori_shmem_initialized", True):
            ep = mod.MoriExpertParallel(
                rank=0, world_size=2, hidden_dim=64,
                num_experts_per_rank=4, num_experts_per_token=2,
                max_num_inp_token_per_rank=16,
            )

        return ep, mock_op

    @patch("torch.cuda.synchronize")
    def test_dispatch_forward_saves_state(self, mock_sync):
        from transformer_engine.pytorch._lite.mori_ep import MoriEPDispatch

        ep, mock_op = self._make_ep_and_mock()
        hidden_dim, topk, num_recv = 64, 2, 5

        mock_op.dispatch.return_value = (
            torch.randn(num_recv, hidden_dim),
            torch.rand(num_recv, topk),
            None,
            torch.randint(0, 8, (num_recv, topk), dtype=torch.int32),
            torch.tensor([num_recv], dtype=torch.int32),
        )

        state = ep.new_cycle()
        input_t = torch.randn(4, hidden_dim, requires_grad=True)
        weights = torch.rand(4, topk)
        indices = torch.randint(0, 8, (4, topk), dtype=torch.int32)

        recv, recv_w, recv_idx = MoriEPDispatch.apply(
            input_t, weights, indices, state,
        )

        # Check state was populated
        assert state.fwd_num_input == 4
        assert state.fwd_num_recv == num_recv
        assert state.fwd_weights is not None
        assert state.fwd_indices is not None
        assert recv.shape == (num_recv, hidden_dim)
        assert recv_w.shape == (num_recv, topk)

    @patch("torch.cuda.synchronize")
    def test_dispatch_backward_calls_combine(self, mock_sync):
        """Dispatch backward should call MORI combine to reverse the communication."""
        from transformer_engine.pytorch._lite.mori_ep import MoriEPDispatch

        ep, mock_op = self._make_ep_and_mock()
        hidden_dim, topk = 64, 2
        num_recv = 5
        num_input = 4

        mock_op.dispatch.return_value = (
            torch.randn(num_recv, hidden_dim),
            torch.rand(num_recv, topk),
            None,
            torch.randint(0, 8, (num_recv, topk), dtype=torch.int32),
            torch.tensor([num_recv], dtype=torch.int32),
        )
        mock_op.combine.return_value = (
            torch.randn(16, hidden_dim),  # max_tokens=16
            torch.rand(16, topk),
        )

        state = ep.new_cycle()
        input_t = torch.randn(num_input, hidden_dim, requires_grad=True)
        weights = torch.rand(num_input, topk)
        indices = torch.randint(0, 8, (num_input, topk), dtype=torch.int32)

        recv, recv_w, recv_idx = MoriEPDispatch.apply(
            input_t, weights, indices, state,
        )

        # Simulate backward: set bwd state as if combine.backward ran first
        state.bwd_recv_weights = torch.rand(num_recv, topk)
        state.bwd_recv_indices = torch.randint(0, 8, (num_recv, topk), dtype=torch.int32)
        state.bwd_num_recv = num_recv

        # Trigger backward
        loss = recv.sum()
        loss.backward()

        # Verify combine was called in backward
        mock_op.combine.assert_called_once()
        mock_op.reset.assert_called_once()
        assert input_t.grad is not None
        assert input_t.grad.shape == (num_input, hidden_dim)


class TestMoriEPCombineAutograd:
    """Test MoriEPCombine autograd function with mocked MORI backend."""

    def _make_ep_and_mock(self):
        import transformer_engine.pytorch._lite.mori_ep as mod

        mock_mori = MagicMock()
        mock_kt = MagicMock()
        mock_kt.IntraNode = "IntraNode"
        mock_mori.ops.EpDispatchCombineKernelType = mock_kt
        mock_mori.ops.EpDispatchCombineConfig = MagicMock()

        mock_op = MagicMock()
        mock_mori.ops.EpDispatchCombineOp.return_value = mock_op

        with mock.patch.object(mod, "_mori_available", True), \
             mock.patch.object(mod, "_mori", mock_mori), \
             mock.patch.object(mod, "_mori_shmem_initialized", True):
            ep = mod.MoriExpertParallel(
                rank=0, world_size=2, hidden_dim=64,
                num_experts_per_rank=4, num_experts_per_token=2,
                max_num_inp_token_per_rank=16,
            )

        return ep, mock_op

    @patch("torch.cuda.synchronize")
    def test_combine_forward_resets_op(self, mock_sync):
        """Combine forward should reset the MORI op after combining."""
        from transformer_engine.pytorch._lite.mori_ep import MoriEPCombine

        ep, mock_op = self._make_ep_and_mock()
        hidden_dim, topk = 64, 2
        num_input = 4

        mock_op.combine.return_value = (
            torch.randn(16, hidden_dim),
            torch.rand(16, topk),
        )

        state = ep.new_cycle()
        state.fwd_num_input = num_input  # as if dispatch already ran

        expert_out = torch.randn(5, hidden_dim, requires_grad=True)
        recv_w = torch.rand(5, topk)
        recv_idx = torch.randint(0, 8, (5, topk), dtype=torch.int32)

        output, output_w = MoriEPCombine.apply(expert_out, recv_w, recv_idx, state)

        mock_op.combine.assert_called_once()
        mock_op.reset.assert_called_once()
        assert output.shape == (num_input, hidden_dim)

    @patch("torch.cuda.synchronize")
    def test_combine_backward_dispatches_gradients(self, mock_sync):
        """Combine backward should dispatch gradients to expert ranks."""
        from transformer_engine.pytorch._lite.mori_ep import MoriEPCombine

        ep, mock_op = self._make_ep_and_mock()
        hidden_dim, topk = 64, 2
        num_input = 4
        num_expert_tokens = 5  # tokens this rank received from dispatch

        mock_op.combine.return_value = (
            torch.randn(16, hidden_dim),
            torch.rand(16, topk),
        )
        # Mock the backward dispatch call -- in a real scenario, the backward
        # dispatch uses the same routing as forward, so the number of received
        # gradient tokens matches the forward expert_output count.
        mock_op.dispatch.return_value = (
            torch.randn(num_expert_tokens, hidden_dim),
            torch.rand(num_expert_tokens, topk),
            None,
            torch.randint(0, 8, (num_expert_tokens, topk), dtype=torch.int32),
            torch.tensor([num_expert_tokens], dtype=torch.int32),
        )

        state = ep.new_cycle()
        state.fwd_num_input = num_input
        state.fwd_weights = torch.rand(num_input, topk)
        state.fwd_indices = torch.randint(0, 8, (num_input, topk), dtype=torch.int32)

        expert_out = torch.randn(num_expert_tokens, hidden_dim, requires_grad=True)
        recv_w = torch.rand(num_expert_tokens, topk)
        recv_idx = torch.randint(0, 8, (num_expert_tokens, topk), dtype=torch.int32)

        output, output_w = MoriEPCombine.apply(expert_out, recv_w, recv_idx, state)

        # Reset mock counters (combine was called in forward, reset was called)
        mock_op.reset.reset_mock()

        loss = output.sum()
        loss.backward()

        # Verify dispatch was called in backward for gradient communication
        assert mock_op.dispatch.call_count == 1
        assert expert_out.grad is not None
        assert expert_out.grad.shape == (num_expert_tokens, hidden_dim)
        # Verify backward state was saved
        assert state.bwd_num_recv == num_expert_tokens


class TestMoriEPFullAutogradCycle:
    """Test a full forward+backward cycle through dispatch → expert → combine."""

    def _make_ep_and_mock(self):
        import transformer_engine.pytorch._lite.mori_ep as mod

        mock_mori = MagicMock()
        mock_kt = MagicMock()
        mock_kt.IntraNode = "IntraNode"
        mock_mori.ops.EpDispatchCombineKernelType = mock_kt
        mock_mori.ops.EpDispatchCombineConfig = MagicMock()

        mock_op = MagicMock()
        mock_mori.ops.EpDispatchCombineOp.return_value = mock_op

        with mock.patch.object(mod, "_mori_available", True), \
             mock.patch.object(mod, "_mori", mock_mori), \
             mock.patch.object(mod, "_mori_shmem_initialized", True):
            ep = mod.MoriExpertParallel(
                rank=0, world_size=2, hidden_dim=64,
                num_experts_per_rank=4, num_experts_per_token=2,
                max_num_inp_token_per_rank=16,
            )

        return ep, mock_op

    @patch("torch.cuda.synchronize")
    def test_full_forward_backward_cycle(self, mock_sync):
        """Test complete dispatch → expert → combine → backward flow."""
        from transformer_engine.pytorch._lite.mori_ep import (
            MoriEPDispatch, MoriEPCombine,
        )

        ep, mock_op = self._make_ep_and_mock()
        hidden_dim, topk = 64, 2
        num_input = 4
        num_recv = 6

        # Forward dispatch mock
        fwd_recv_tokens = torch.randn(num_recv, hidden_dim)
        fwd_recv_weights = torch.rand(num_recv, topk)
        fwd_recv_indices = torch.randint(0, 8, (num_recv, topk), dtype=torch.int32)

        mock_op.dispatch.return_value = (
            fwd_recv_tokens,
            fwd_recv_weights,
            None,
            fwd_recv_indices,
            torch.tensor([num_recv], dtype=torch.int32),
        )

        # Forward combine mock
        mock_op.combine.return_value = (
            torch.randn(16, hidden_dim),
            torch.rand(16, topk),
        )

        # --- Forward ---
        state = ep.new_cycle()
        input_t = torch.randn(num_input, hidden_dim, requires_grad=True)
        weights = torch.rand(num_input, topk)
        indices = torch.randint(0, 8, (num_input, topk), dtype=torch.int32)

        # Step 1: Dispatch
        recv, recv_w, recv_idx = MoriEPDispatch.apply(
            input_t, weights, indices, state,
        )

        # Step 2: Expert computation (simple linear, differentiable)
        expert_weight = torch.randn(hidden_dim, hidden_dim, requires_grad=True)
        expert_out = recv @ expert_weight

        # Step 3: Combine
        output, output_w = MoriEPCombine.apply(
            expert_out, recv_w, recv_idx, state,
        )

        # Verify forward calls
        assert mock_op.dispatch.call_count == 1  # forward dispatch
        assert mock_op.combine.call_count == 1   # forward combine
        assert mock_op.reset.call_count == 1     # forward reset

        # --- Backward ---
        # Set up backward mocks (dispatch in combine.bwd, combine in dispatch.bwd)
        bwd_recv = torch.randn(num_recv, hidden_dim)
        mock_op.dispatch.return_value = (
            bwd_recv,
            torch.rand(num_recv, topk),
            None,
            torch.randint(0, 8, (num_recv, topk), dtype=torch.int32),
            torch.tensor([num_recv], dtype=torch.int32),
        )
        mock_op.combine.return_value = (
            torch.randn(16, hidden_dim),
            torch.rand(16, topk),
        )

        loss = output.sum()
        loss.backward()

        # Verify backward calls
        assert mock_op.dispatch.call_count == 2  # fwd dispatch + bwd dispatch (in combine.bwd)
        assert mock_op.combine.call_count == 2   # fwd combine + bwd combine (in dispatch.bwd)
        assert mock_op.reset.call_count == 2     # fwd reset + bwd reset

        # Verify gradients exist
        assert input_t.grad is not None
        assert input_t.grad.shape == (num_input, hidden_dim)
        assert expert_weight.grad is not None
        assert expert_weight.grad.shape == (hidden_dim, hidden_dim)


# ---------------------------------------------------------------------------
# Multi-GPU Integration Tests -- require MORI + AMD GPUs
# ---------------------------------------------------------------------------

def _run_ep_worker(rank, world_size, hidden_dim, num_experts_per_rank,
                   num_experts_per_token, max_tokens, results_dict):
    """Worker function for multi-GPU dispatch/combine test."""
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "1G")

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    # Register world group for MORI
    world_group = dist.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)

    from transformer_engine.pytorch._lite.mori_ep import (
        init_mori_ep, finalize_mori_ep, MoriExpertParallel,
    )

    try:
        init_mori_ep()

        ep = MoriExpertParallel(
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_dim,
            num_experts_per_rank=num_experts_per_rank,
            num_experts_per_token=num_experts_per_token,
            max_num_inp_token_per_rank=max_tokens,
            dtype=torch.bfloat16,
            kernel_type="intra_node",
        )

        # Generate test data
        torch.manual_seed(42 + rank)
        num_tokens = max_tokens // 2  # use half capacity
        total_experts = num_experts_per_rank * world_size
        input_tokens = torch.randn(
            num_tokens, hidden_dim, dtype=torch.bfloat16, device=device,
        )
        weights = torch.rand(
            num_tokens, num_experts_per_token, dtype=torch.float32, device=device,
        )
        indices = torch.stack([
            torch.randperm(total_experts, device=device)[:num_experts_per_token]
            for _ in range(num_tokens)
        ]).to(torch.int32)

        # Dispatch
        recv_tokens, recv_weights, recv_indices, num_recv = ep.dispatch(
            input_tokens, weights, indices,
        )

        # Simple expert function: identity (pass through)
        expert_output = recv_tokens[:num_recv].clone().to(torch.bfloat16)

        # Combine
        output, output_weights = ep.combine(
            expert_output, recv_weights, recv_indices,
        )
        ep.reset()

        # Verify basic properties
        results_dict[rank] = {
            "num_recv": num_recv,
            "output_shape": tuple(output.shape),
            "num_input_tokens": num_tokens,
            "success": True,
        }

    except Exception as e:
        results_dict[rank] = {
            "success": False,
            "error": str(e),
        }

    finally:
        finalize_mori_ep()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Standard MoE layout tests
# ---------------------------------------------------------------------------

class TestStdMoEDispatchCombine:
    """Test standard MoE per-expert layout dispatch/combine with mocked MORI."""

    def _make_ep_and_mock(self):
        import transformer_engine.pytorch._lite.mori_ep as mod

        mock_mori = MagicMock()
        mock_kt = MagicMock()
        mock_kt.IntraNode = "IntraNode"
        mock_mori.ops.EpDispatchCombineKernelType = mock_kt
        mock_mori.ops.EpDispatchCombineConfig = MagicMock()

        mock_op = MagicMock()
        mock_mori.ops.EpDispatchCombineOp.return_value = mock_op

        with mock.patch.object(mod, "_mori_available", True), \
             mock.patch.object(mod, "_mori", mock_mori), \
             mock.patch.object(mod, "_mori_shmem_initialized", True):
            ep = mod.MoriExpertParallel(
                rank=0, world_size=2, hidden_dim=64,
                num_experts_per_rank=4, num_experts_per_token=2,
                max_num_inp_token_per_rank=16,
                kernel_type="intra_node",
            )

        return ep, mock_op

    @patch("torch.cuda.synchronize")
    def test_dispatch_standard_moe_returns_per_expert_layout(self, mock_sync):
        ep, mock_op = self._make_ep_and_mock()
        hidden_dim = 64
        num_experts = 4
        max_tpe = 32  # max_tokens_per_expert = world_size * max_tokens

        mock_op.dispatch_standard_moe.return_value = (
            torch.randn(num_experts, max_tpe, hidden_dim),  # packed_tokens
            torch.tensor([5, 3, 7, 2], dtype=torch.int32),  # recv_count
            torch.zeros(num_experts, max_tpe, dtype=torch.int32),  # src_info
            torch.empty(0),  # layout_range
        )

        input_t = torch.randn(8, hidden_dim)
        weights = torch.rand(8, 2)
        indices = torch.randint(0, 8, (8, 2), dtype=torch.int32)

        packed, recv_count, src_info = ep.dispatch_standard_moe(
            input_t, weights, indices,
        )

        mock_op.dispatch_standard_moe.assert_called_once()
        assert packed.shape == (num_experts, max_tpe, hidden_dim)
        assert recv_count.shape == (num_experts,)
        assert recv_count.tolist() == [5, 3, 7, 2]

    @patch("torch.cuda.synchronize")
    def test_combine_standard_moe(self, mock_sync):
        ep, mock_op = self._make_ep_and_mock()
        hidden_dim = 64
        num_experts = 4
        max_tpe = 32

        mock_op.combine_standard_moe.return_value = (
            torch.randn(16, hidden_dim),  # output
            None,
        )

        expert_out = torch.randn(num_experts, max_tpe, hidden_dim)
        weights = torch.rand(8, 2)
        indices = torch.randint(0, 8, (8, 2), dtype=torch.int32)

        output, output_w = ep.combine_standard_moe(
            expert_out, weights, indices,
        )

        mock_op.combine_standard_moe.assert_called_once()
        assert output.shape[1] == hidden_dim
        assert output_w is None

    @patch("torch.cuda.synchronize")
    def test_convert_dispatch_to_standard(self, mock_sync):
        ep, mock_op = self._make_ep_and_mock()
        hidden_dim = 64
        num_experts = 4
        max_tpe = 32

        mock_op.convert_dispatch_output.return_value = (
            torch.randn(num_experts, max_tpe, hidden_dim),
            torch.tensor([3, 4, 2, 1], dtype=torch.int32),
            torch.zeros(num_experts, max_tpe, dtype=torch.int32),
            torch.empty(0),
        )

        dispatch_tokens = torch.randn(20, hidden_dim)
        dispatch_indices = torch.randint(0, 8, (20, 2), dtype=torch.int32)

        packed, recv_count, src_info = ep.convert_dispatch_to_standard(
            dispatch_tokens, dispatch_indices,
        )

        mock_op.convert_dispatch_output.assert_called_once()
        assert packed.shape == (num_experts, max_tpe, hidden_dim)

    @patch("torch.cuda.synchronize")
    def test_convert_standard_to_combine_input(self, mock_sync):
        ep, mock_op = self._make_ep_and_mock()
        hidden_dim = 64
        num_experts = 4
        max_tpe = 32
        max_recv = 20

        mock_op.convert_combine_input.return_value = torch.randn(max_recv, hidden_dim)

        packed = torch.randn(num_experts, max_tpe, hidden_dim)
        src_info = torch.zeros(num_experts, max_tpe, dtype=torch.int32)

        flat = ep.convert_standard_to_combine_input(packed, src_info)

        mock_op.convert_combine_input.assert_called_once()
        assert flat.shape == (max_recv, hidden_dim)


class TestStdMoECycleState:
    """Test standard MoE cycle state creation."""

    def _make_ep(self):
        import transformer_engine.pytorch._lite.mori_ep as mod

        mock_mori = MagicMock()
        mock_kt = MagicMock()
        mock_kt.IntraNode = "IntraNode"
        mock_mori.ops.EpDispatchCombineKernelType = mock_kt
        mock_mori.ops.EpDispatchCombineConfig = MagicMock()
        mock_mori.ops.EpDispatchCombineOp.return_value = MagicMock()

        with mock.patch.object(mod, "_mori_available", True), \
             mock.patch.object(mod, "_mori", mock_mori), \
             mock.patch.object(mod, "_mori_shmem_initialized", True):
            ep = mod.MoriExpertParallel(
                rank=0, world_size=2, hidden_dim=64,
                num_experts_per_rank=4, num_experts_per_token=2,
                max_num_inp_token_per_rank=16,
            )
        return ep

    def test_new_std_moe_cycle(self):
        from transformer_engine.pytorch._lite.mori_ep import _MoriStdMoECycleState
        ep = self._make_ep()
        state = ep.new_std_moe_cycle()
        assert isinstance(state, _MoriStdMoECycleState)
        assert state.ep is ep
        assert state.fwd_recv_count is None
        assert state.fwd_src_info is None


class TestStdMoEAutograd:
    """Test standard MoE autograd dispatch/combine cycle."""

    def _make_ep_and_mock(self):
        import transformer_engine.pytorch._lite.mori_ep as mod

        mock_mori = MagicMock()
        mock_kt = MagicMock()
        mock_kt.IntraNode = "IntraNode"
        mock_mori.ops.EpDispatchCombineKernelType = mock_kt
        mock_mori.ops.EpDispatchCombineConfig = MagicMock()

        mock_op = MagicMock()
        mock_mori.ops.EpDispatchCombineOp.return_value = mock_op

        with mock.patch.object(mod, "_mori_available", True), \
             mock.patch.object(mod, "_mori", mock_mori), \
             mock.patch.object(mod, "_mori_shmem_initialized", True):
            ep = mod.MoriExpertParallel(
                rank=0, world_size=2, hidden_dim=64,
                num_experts_per_rank=4, num_experts_per_token=2,
                max_num_inp_token_per_rank=16,
            )

        return ep, mock_op

    @patch("torch.cuda.synchronize")
    def test_full_std_moe_forward_backward(self, mock_sync):
        """Test dispatch_standard_moe → expert → combine_standard_moe → backward."""
        from transformer_engine.pytorch._lite.mori_ep import (
            MoriEPDispatchStdMoE, MoriEPCombineStdMoE,
        )

        ep, mock_op = self._make_ep_and_mock()
        hidden_dim = 64
        topk = 2
        num_input = 4
        num_experts = 4
        max_tpe = 32

        # Forward dispatch mock
        mock_op.dispatch_standard_moe.return_value = (
            torch.randn(num_experts, max_tpe, hidden_dim),
            torch.tensor([3, 2, 4, 1], dtype=torch.int32),
            torch.zeros(num_experts, max_tpe, dtype=torch.int32),
            torch.empty(0),
        )

        # Forward combine mock
        mock_op.combine_standard_moe.return_value = (
            torch.randn(16, hidden_dim),
            None,
        )

        # --- Forward ---
        state = ep.new_std_moe_cycle()
        input_t = torch.randn(num_input, hidden_dim, requires_grad=True)
        weights = torch.rand(num_input, topk)
        indices = torch.randint(0, 8, (num_input, topk), dtype=torch.int32)

        packed, recv_count, src_info = MoriEPDispatchStdMoE.apply(
            input_t, weights, indices, state,
        )

        assert packed.shape == (num_experts, max_tpe, hidden_dim)
        assert state.fwd_recv_count is not None

        # Expert computation: simple per-expert linear
        expert_weight = torch.randn(hidden_dim, hidden_dim, requires_grad=True)
        expert_out = torch.einsum("eth,hd->etd", packed.float(), expert_weight.float())
        expert_out = expert_out.to(packed.dtype)

        output, _ = MoriEPCombineStdMoE.apply(
            expert_out, weights, indices, state,
        )

        # Verify forward calls
        assert mock_op.dispatch_standard_moe.call_count == 1
        assert mock_op.combine_standard_moe.call_count == 1
        assert mock_op.reset.call_count == 1

        # --- Backward ---
        # Backward mocks: combine.bwd calls dispatch_standard_moe,
        # dispatch.bwd calls combine_standard_moe
        mock_op.dispatch_standard_moe.return_value = (
            torch.randn(num_experts, max_tpe, hidden_dim),
            torch.tensor([3, 2, 4, 1], dtype=torch.int32),
            torch.zeros(num_experts, max_tpe, dtype=torch.int32),
            torch.empty(0),
        )
        mock_op.combine_standard_moe.return_value = (
            torch.randn(16, hidden_dim),
            None,
        )

        loss = output.sum()
        loss.backward()

        # Verify backward calls
        assert mock_op.dispatch_standard_moe.call_count == 2  # fwd + bwd
        assert mock_op.combine_standard_moe.call_count == 2  # fwd + bwd
        assert mock_op.reset.call_count == 2  # fwd + bwd

        assert input_t.grad is not None
        assert input_t.grad.shape == (num_input, hidden_dim)
        assert expert_weight.grad is not None

    @patch("torch.cuda.synchronize")
    def test_dispatch_std_moe_saves_state(self, mock_sync):
        from transformer_engine.pytorch._lite.mori_ep import MoriEPDispatchStdMoE

        ep, mock_op = self._make_ep_and_mock()
        hidden_dim, topk, num_experts, max_tpe = 64, 2, 4, 32

        mock_op.dispatch_standard_moe.return_value = (
            torch.randn(num_experts, max_tpe, hidden_dim),
            torch.tensor([2, 3, 1, 4], dtype=torch.int32),
            torch.ones(num_experts, max_tpe, dtype=torch.int32),
            torch.empty(0),
        )

        state = ep.new_std_moe_cycle()
        input_t = torch.randn(6, hidden_dim, requires_grad=True)
        weights = torch.rand(6, topk)
        indices = torch.randint(0, 8, (6, topk), dtype=torch.int32)

        MoriEPDispatchStdMoE.apply(input_t, weights, indices, state)

        assert state.fwd_num_input == 6
        assert state.fwd_weights is not None
        assert state.fwd_indices is not None
        assert state.fwd_recv_count is not None
        assert state.fwd_src_info is not None


@skip_no_mori
@skip_insufficient_gpus
class TestMoriEPIntegration:
    """Integration tests requiring MORI and multiple GPUs."""

    def test_dispatch_combine_basic(self):
        """Test basic dispatch/combine round-trip across GPUs."""
        import torch.multiprocessing as mp

        world_size = min(_gpu_count(), 8)
        hidden_dim = 256
        num_experts_per_rank = 4
        num_experts_per_token = 2
        max_tokens = 64

        manager = mp.Manager()
        results = manager.dict()

        mp.spawn(
            _run_ep_worker,
            args=(world_size, hidden_dim, num_experts_per_rank,
                  num_experts_per_token, max_tokens, results),
            nprocs=world_size,
            join=True,
        )

        for rank in range(world_size):
            assert rank in results, f"Rank {rank} did not report results"
            r = results[rank]
            assert r["success"], f"Rank {rank} failed: {r.get('error', 'unknown')}"
            assert r["num_recv"] >= 0, f"Rank {rank}: invalid num_recv"
            assert r["output_shape"][1] == hidden_dim


def _run_ep_roundtrip_verify(rank, world_size, hidden_dim, num_experts_per_rank,
                              num_experts_per_token, max_tokens, results_dict):
    """Worker that verifies dispatch/combine preserves token data."""
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "1G")

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    world_group = dist.group.WORLD
    torch._C._distributed_c10d._register_process_group("default", world_group)

    from transformer_engine.pytorch._lite.mori_ep import (
        init_mori_ep, finalize_mori_ep, MoriExpertParallel,
    )

    try:
        init_mori_ep()

        ep = MoriExpertParallel(
            rank=rank,
            world_size=world_size,
            hidden_dim=hidden_dim,
            num_experts_per_rank=num_experts_per_rank,
            num_experts_per_token=num_experts_per_token,
            max_num_inp_token_per_rank=max_tokens,
            dtype=torch.bfloat16,
            kernel_type="intra_node",
        )

        # Fixed seed for reproducibility across runs
        torch.manual_seed(123 + rank)
        num_tokens = 16
        total_experts = num_experts_per_rank * world_size

        input_tokens = torch.randn(
            num_tokens, hidden_dim, dtype=torch.bfloat16, device=device,
        )
        weights = torch.ones(
            num_tokens, num_experts_per_token, dtype=torch.float32, device=device,
        )
        indices = torch.stack([
            torch.randperm(total_experts, device=device)[:num_experts_per_token]
            for _ in range(num_tokens)
        ]).to(torch.int32)

        # Dispatch
        recv_tokens, recv_weights, recv_indices, num_recv = ep.dispatch(
            input_tokens, weights, indices,
        )

        # Identity expert: just pass tokens through
        expert_output = recv_tokens[:num_recv].clone().to(torch.bfloat16)

        # Combine
        output, _ = ep.combine(expert_output, recv_weights, recv_indices)

        # After identity expert with weights=1.0, each token's output should
        # equal input * (number of unique PEs the token was sent to).
        # This matches the MORI test pattern.
        for i in range(num_tokens):
            pes = set()
            for idx in indices[i].cpu().tolist():
                pes.add(idx // num_experts_per_rank)
            unique_pes = len(pes)

            expected = (input_tokens[i].float() * unique_pes).to(torch.bfloat16)
            got = output[i]
            match = torch.allclose(got.float(), expected.float(), atol=1e-2, rtol=1e-2)
            if not match:
                results_dict[rank] = {
                    "success": False,
                    "error": (
                        f"Token {i}: expected scale {unique_pes}, "
                        f"max_diff={torch.abs(got.float() - expected.float()).max().item()}"
                    ),
                }
                ep.reset()
                return

        ep.reset()
        results_dict[rank] = {"success": True, "num_tokens_verified": num_tokens}

    except Exception as e:
        results_dict[rank] = {"success": False, "error": str(e)}

    finally:
        finalize_mori_ep()
        dist.destroy_process_group()


@skip_no_mori
@skip_insufficient_gpus
class TestMoriEPRoundtrip:
    """Verify data integrity through dispatch/combine round-trip."""

    def test_identity_expert_roundtrip(self):
        """With identity expert and uniform weights, output = input * num_unique_pes."""
        import torch.multiprocessing as mp

        world_size = min(_gpu_count(), 8)
        hidden_dim = 128
        num_experts_per_rank = 4
        num_experts_per_token = 2
        max_tokens = 64

        manager = mp.Manager()
        results = manager.dict()

        mp.spawn(
            _run_ep_roundtrip_verify,
            args=(world_size, hidden_dim, num_experts_per_rank,
                  num_experts_per_token, max_tokens, results),
            nprocs=world_size,
            join=True,
        )

        for rank in range(world_size):
            assert rank in results, f"Rank {rank} did not report results"
            r = results[rank]
            assert r["success"], f"Rank {rank} failed: {r.get('error', 'unknown')}"
