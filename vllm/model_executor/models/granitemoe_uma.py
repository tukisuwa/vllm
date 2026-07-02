# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for Granite MoE checkpoints."""

from dataclasses import dataclass
from typing import Any

from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
    TensorCatalog,
    WeightPlan,
    WeightPlanEntry,
    build_auto_weight_plan_for_module,
)

from .routed_moe_uma import (
    RoutedMoeEntry,
    RoutedMoeSourcePlan,
    load_routed_moe_weights_from_source,
    routed_entry_requires_local_read,
)
from .utils import PPMissingLayer


@dataclass(frozen=True)
class GraniteMoeSourcePlan:
    auto_plan: WeightPlan
    routed_entries: tuple[RoutedMoeEntry, ...]


class _GraniteSourceMapper:
    def _map_name_with_shard(self, name: str) -> tuple[str, str | None] | None:
        stacked = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        for param_name, weight_name, shard_id in stacked:
            if weight_name in name:
                return name.replace(weight_name, param_name, 1), shard_id
        return name, None


def _full_tail(rank: int) -> tuple[slice, ...]:
    return tuple(slice(None) for _ in range(max(rank - 2, 0)))


def _parse_layer_id(name: str) -> int | None:
    parts = name.split(".")
    for idx in range(len(parts) - 1):
        if parts[idx] == "layers" and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    return None


def _resolve_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Granite MoE UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"Granite MoE UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return None
    block_sparse_moe = getattr(layer, "block_sparse_moe", None)
    routed_experts = getattr(block_sparse_moe, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "Granite MoE UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return routed_experts


def _get_routed_experts_for_layer(model: nn.Module, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id)


def _make_granite_expert_entries(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    num_experts: int,
    routed_prefix: str,
) -> list[RoutedMoeEntry]:
    entries: list[RoutedMoeEntry] = []
    for name in catalog.names():
        layer_id = _parse_layer_id(name)
        if layer_id is None:
            continue

        if name.endswith(".block_sparse_moe.input_linear.weight"):
            meta = catalog.get(name)
            if len(meta.shape) < 2 or meta.shape[0] != num_experts:
                raise RuntimeError(
                    f"Granite input_linear tensor has unsupported shape: {name} "
                    f"{meta.shape}"
                )
            fused_dim = meta.shape[1]
            if fused_dim % 2 != 0:
                raise RuntimeError(
                    "Granite input_linear tensor cannot split w1/w3 evenly: "
                    f"{name} {meta.shape}"
                )
            half = fused_dim // 2
            tail = _full_tail(len(meta.shape))
            routed_experts = _resolve_routed_experts_for_layer(model, layer_id)
            for expert_id in range(num_experts):
                w13_name = f"{routed_prefix}w13_weight"
                w1_weight_name = f"{routed_experts.layer_name}.{w13_name}"
                entries.append(
                    RoutedMoeEntry(
                        checkpoint_name=name,
                        layer_id=layer_id,
                        expert_id=expert_id,
                        param_name=w13_name,
                        shard_id="w1",
                        local_required=routed_entry_requires_local_read(
                            routed_experts, expert_id, w1_weight_name
                        ),
                        source_slices=(expert_id, slice(0, half), *tail),
                    )
                )
                entries.append(
                    RoutedMoeEntry(
                        checkpoint_name=name,
                        layer_id=layer_id,
                        expert_id=expert_id,
                        param_name=w13_name,
                        shard_id="w3",
                        local_required=routed_entry_requires_local_read(
                            routed_experts, expert_id, w1_weight_name
                        ),
                        source_slices=(expert_id, slice(half, fused_dim), *tail),
                    )
                )
        elif name.endswith(".block_sparse_moe.output_linear.weight"):
            meta = catalog.get(name)
            if len(meta.shape) < 1 or meta.shape[0] != num_experts:
                raise RuntimeError(
                    f"Granite output_linear tensor has unsupported shape: {name} "
                    f"{meta.shape}"
                )
            tail = tuple(slice(None) for _ in range(len(meta.shape) - 1))
            routed_experts = _resolve_routed_experts_for_layer(model, layer_id)
            for expert_id in range(num_experts):
                w2_name = f"{routed_prefix}w2_weight"
                w2_weight_name = f"{routed_experts.layer_name}.{w2_name}"
                entries.append(
                    RoutedMoeEntry(
                        checkpoint_name=name,
                        layer_id=layer_id,
                        expert_id=expert_id,
                        param_name=w2_name,
                        shard_id="w2",
                        local_required=routed_entry_requires_local_read(
                            routed_experts, expert_id, w2_weight_name
                        ),
                        source_slices=(expert_id, *tail),
                    )
                )
    return entries


def build_granite_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    num_experts: int,
    routed_prefix: str = "",
    skip_prefixes: list[str] | None = None,
) -> GraniteMoeSourcePlan:
    routed_entries = _make_granite_expert_entries(
        model,
        catalog,
        num_experts=num_experts,
        routed_prefix=routed_prefix,
    )
    auto_plan = build_auto_weight_plan_for_module(
        model,
        catalog,
        mapper=_GraniteSourceMapper(),
        skip_prefixes=skip_prefixes,
        skip_substrs=[
            ".block_sparse_moe.input_linear.",
            ".block_sparse_moe.output_linear.",
        ],
    )

    routed_source_names = {entry.checkpoint_name for entry in routed_entries}
    rewritten_entries: list[WeightPlanEntry] = []
    for entry in auto_plan:
        if entry.checkpoint_name in routed_source_names:
            continue
        if entry.checkpoint_name.endswith(".block_sparse_moe.router.layer.weight"):
            rewritten_entries.append(
                WeightPlanEntry(
                    checkpoint_name=entry.checkpoint_name,
                    target_name=entry.target_name.replace(
                        ".block_sparse_moe.router.layer.weight",
                        ".block_sparse_moe.gate.weight",
                    ),
                    ignore_missing=entry.ignore_missing,
                )
            )
            continue
        rewritten_entries.append(entry)

    return GraniteMoeSourcePlan(
        auto_plan=WeightPlan(tuple(rewritten_entries)),
        routed_entries=tuple(routed_entries),
    )


def load_granite_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: GraniteMoeSourcePlan,
) -> set[str]:
    source_plan = RoutedMoeSourcePlan(plan.auto_plan, plan.routed_entries)
    return load_routed_moe_weights_from_source(
        model,
        source,
        source_plan,
        family_name="Granite MoE",
        get_routed_experts=_get_routed_experts_for_layer,
    )
