# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Communication overlap stubs.

In lite mode, comm-overlap is not available. These stubs provide the API
surface so that imports succeed, but raise NotImplementedError if actually used.
Use torch.distributed for communication instead.
"""

from .enums import CommOverlapCore


class CommOverlapBase(CommOverlapCore):
    """Stub for CommOverlapBase."""
    pass


class CommOverlapP2PBase(CommOverlapCore):
    """Stub for CommOverlapP2PBase."""
    pass


class CommOverlapHelper:
    """Stub for CommOverlapHelper."""
    def __init__(self, *args, **kwargs):
        pass


class CommOverlap(CommOverlapBase):
    """Stub for CommOverlap."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CommOverlap is not available in lite mode. "
            "Use torch.distributed for communication."
        )

    def copy_into_buffer(self, *args, **kwargs):
        raise NotImplementedError

    def get_buffer(self, *args, **kwargs):
        raise NotImplementedError

    def get_communication_stream(self, *args, **kwargs):
        raise NotImplementedError


class CommOverlapP2P(CommOverlapP2PBase):
    """Stub for CommOverlapP2P."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CommOverlapP2P is not available in lite mode. "
            "Use torch.distributed for communication."
        )

    def copy_into_buffer(self, *args, **kwargs):
        raise NotImplementedError

    def get_buffer(self, *args, **kwargs):
        raise NotImplementedError

    def get_communication_stream(self, *args, **kwargs):
        raise NotImplementedError


def bulk_overlap_ag_with_external_gemm(*args, **kwargs):
    """Stub: Bulk overlap AG with external GEMM."""
    raise NotImplementedError("Communication overlap not available in lite mode.")


def init_nvshmem_backend(*args, **kwargs):
    """Stub: Initialize NVSHMEM/ROCSHMEM backend."""
    raise NotImplementedError("NVSHMEM/ROCSHMEM not available in lite mode.")


def create_nvshmem_tensor(*args, **kwargs):
    """Stub: Create NVSHMEM/ROCSHMEM tensor."""
    raise NotImplementedError("NVSHMEM/ROCSHMEM not available in lite mode.")


def nvshmem_send_on_current_stream(*args, **kwargs):
    """Stub: NVSHMEM send."""
    raise NotImplementedError("NVSHMEM/ROCSHMEM not available in lite mode.")


def nvshmem_wait_on_current_stream(*args, **kwargs):
    """Stub: NVSHMEM wait."""
    raise NotImplementedError("NVSHMEM/ROCSHMEM not available in lite mode.")


def nvshmem_finalize(*args, **kwargs):
    """Stub: NVSHMEM finalize."""
    raise NotImplementedError("NVSHMEM/ROCSHMEM not available in lite mode.")


def device_supports_multicast(device_id=-1):
    """Stub: Check multicast support."""
    return False


def get_stream_priority_range(device_id=-1):
    """Stub: Get stream priority range."""
    return (0, 0)


def ubuf_built_with_mpi():
    """Stub: Check if userbuffers built with MPI."""
    return False
