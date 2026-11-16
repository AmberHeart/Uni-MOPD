# Copyright (c) 2026, Uni-MOPD contributors.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, MOPD contributors.
"""CPU tensor transport for hidden replay inside the trusted student mesh.

The server HTTP boundary is JSON/safetensors. Within the native student mesh,
large hidden buffers use Gloo tensor operations, not object collectives that
pickle/copy the entire multi-gigabyte response on every receiver.
"""
import torch
import torch.distributed as dist

from .remote_teacher_protocol import require


def scatter_hidden_buffers(buffers, *, expected_elements, dp_rank, leader_ranks,
                           tp_root, tp_group, control_group):
    """One variable-length BF16 buffer per DP replica, replicated within TP.

    All ranks call on their main thread. Rank zero provides CPU flat buffers in
    DP order. Every rank validates its expected shape before any payload moves.
    Transport uses byte views, keeping BF16 bit patterns and avoiding backend
    dtype-conversion behavior. No padding to the longest replica is needed.
    """
    rank = dist.get_rank()
    lengths = torch.zeros(len(leader_ranks), dtype=torch.int64)
    if rank == 0:
        require(len(buffers) == len(leader_ranks), 'Hidden buffers do not cover the DP mesh')
        for index, buffer in enumerate(buffers):
            require(buffer.ndim == 1 and buffer.dtype == torch.bfloat16
                    and buffer.device.type == 'cpu' and buffer.is_contiguous(),
                    'Hidden transport requires flat contiguous CPU BF16')
            lengths[index] = buffer.numel()
    dist.broadcast(lengths, src=0, group=control_group)
    mismatch = torch.tensor(int(int(lengths[dp_rank]) != expected_elements), dtype=torch.int64)
    dist.all_reduce(mismatch, op=dist.ReduceOp.MAX, group=control_group)
    require(not mismatch.item(), 'Hidden redistribution length differs from the bound local inputs')
    flat = buffers[0] if rank == 0 else torch.empty(expected_elements, dtype=torch.bfloat16)
    pending = []
    if rank == 0:
        for index, leader in enumerate(leader_ranks[1:], 1):
            if buffers[index].numel():
                pending.append(dist.isend(buffers[index].view(torch.uint8), dst=leader, group=control_group))
    elif rank == tp_root and flat.numel():
        dist.recv(flat.view(torch.uint8), src=0, group=control_group)
    for request in pending:
        request.wait()
    if flat.numel():
        dist.broadcast(flat.view(torch.uint8), src=tp_root, group=tp_group)
    return flat
