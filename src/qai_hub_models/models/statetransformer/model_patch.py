# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import types
from collections.abc import Callable
from typing import Any

import torch
from transformers import masking_utils


def custom_one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    """
    Creates a one-hot encoded tensor from indices.

    Parameters
    ----------
    tensor
        Tensor containing indices to be one-hot encoded.
    num_classes
        Total number of classes. Defaults to -1.

    Returns
    -------
    result : torch.Tensor
        One-hot encoded tensor with shape (*tensor.shape, num_classes).
    """
    if num_classes == -1:
        num_classes = int(tensor.max().item()) + 1
    shape = (*tensor.shape, num_classes)
    one_hot = torch.zeros(shape, device=tensor.device)
    one_hot.scatter_(-1, tensor.unsqueeze(-1), 1.0)
    return one_hot


def dense_moe_forward(
    self: Any, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Export-friendly replacement for MixtralSparseMoeBlock.forward.

    Upstream loops over `expert_hit`, the nonzero rows of the routing mask, so
    the set of experts evaluated depends on the input values. torch.export
    cannot guard on that, and torch.jit.trace silently bakes in whichever
    experts the tracing input happened to route to.

    Running every expert and weighting by the (zero-padded) gate matrix is
    mathematically identical, since a token's gate for an unselected expert
    is exactly zero.

    Parameters
    ----------
    self
        The MixtralSparseMoeBlock instance.
    hidden_states
        Shape (batch, sequence, hidden_dim).

    Returns
    -------
    final_hidden_states : torch.Tensor
        Shape (batch, sequence, hidden_dim).
    router_logits : torch.Tensor
        Shape (batch * sequence, num_experts).
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    routing_weights = torch.nn.functional.softmax(
        router_logits, dim=1, dtype=torch.float
    )
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    gates = torch.zeros(
        hidden_states.shape[0],
        self.num_experts,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    ).scatter(1, selected_experts, routing_weights)

    final_hidden_states = torch.zeros_like(hidden_states)
    for expert_idx, expert_layer in enumerate(self.experts):
        final_hidden_states = (
            final_hidden_states
            + expert_layer(hidden_states) * gates[:, expert_idx, None]
        )

    return (
        final_hidden_states.reshape(batch_size, sequence_length, hidden_dim),
        router_logits,
    )


def patch_mixtral_moe(model: torch.nn.Module) -> None:
    """Swap every MixtralSparseMoeBlock onto the dense forward above."""
    for module in model.modules():
        if type(module).__name__ == "MixtralSparseMoeBlock":
            module.forward = types.MethodType(dense_moe_forward, module)


def broadcast_for_bhqkv(mask_function: Callable, bh_indices: bool = True) -> Callable:
    """
    vmap-free replacement for transformers.masking_utils._vmap_for_bhqkv.

    The mask functions are elementwise over the (batch, head, q, kv) index
    grid, so nesting torch.vmap is equivalent to calling them with broadcast
    aranges. vmap leaves a `lazy_load_decompositions` node in the exported
    graph, which torch.export.save cannot serialize.
    """

    def inner(
        batch_idx: torch.Tensor | None,
        head_idx: torch.Tensor | None,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        if not bh_indices:
            mask = mask_function(batch_idx, head_idx, q_idx[:, None], kv_idx[None, :])
            return mask.expand(q_idx.shape[0], kv_idx.shape[0])
        assert batch_idx is not None and head_idx is not None
        mask = mask_function(
            batch_idx[:, None, None, None],
            head_idx[None, :, None, None],
            q_idx[None, None, :, None],
            kv_idx[None, None, None, :],
        )
        return mask.expand(
            batch_idx.shape[0], head_idx.shape[0], q_idx.shape[0], kv_idx.shape[0]
        )

    return inner


def patch_attention_mask_vmap() -> None:
    """Route transformers' 4D mask construction off torch.vmap."""
    masking_utils._vmap_for_bhqkv = broadcast_for_bhqkv
