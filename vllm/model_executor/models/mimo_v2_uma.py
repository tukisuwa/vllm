# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for MiMoV2 models."""

from typing import Any

import torch
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
    execute_weight_plan,
)
from vllm.model_executor.model_loader.weight_plan import (
    TensorCatalog,
    WeightPlan,
    WeightPlanEntry,
    build_auto_weight_plan_for_module,
)
from vllm.model_executor.models.utils import WeightsMapper

from .routed_moe_uma import (
    RoutedExpertPattern,
    RoutedExpertsResolution,
    RoutedMoeEntry,
    RoutedProjectionMap,
    RoutedProjectionRule,
    SliceRule,
    SliceRuleEntry,
    StackedProjectionMap,
    StackedProjectionRule,
    build_routed_moe_weight_plan,
    load_routed_moe_weights_from_source,
)
from .utils import PPMissingLayer


MimoV2MoeRoutedEntry = RoutedMoeEntry

_MIMO_V2_STACKED_PROJECTIONS = StackedProjectionMap(
    (
        StackedProjectionRule(".q_proj", ".qkv_proj", "q"),
        StackedProjectionRule(".k_proj", ".qkv_proj", "k"),
        StackedProjectionRule(".v_proj", ".qkv_proj", "v"),
        StackedProjectionRule(".gate_proj", ".gate_up_proj", 0),
        StackedProjectionRule(".up_proj", ".gate_up_proj", 1),
    )
)
_MIMO_V2_ROUTED_EXPERT_PATTERN = RoutedExpertPattern(
    module_path=("mlp", "experts"),
    projections=("gate_proj", "down_proj", "up_proj"),
)
_MIMO_V2_ROUTED_PROJECTION_MAP = RoutedProjectionMap(
    (
        RoutedProjectionRule("gate_proj", "w13", "w1"),
        RoutedProjectionRule("up_proj", "w13", "w3"),
        RoutedProjectionRule("down_proj", "w2", "w2"),
    )
)


def _mimo_v2_weight_mapper() -> WeightsMapper:
    return _MIMO_V2_STACKED_PROJECTIONS.as_weights_mapper()


def _mimo_v2_name_transform(name: str) -> tuple[str, None] | None:
    if (
        "rotary_emb.inv_freq" in name
        or "rotary_emb.cos_cached" in name
        or "rotary_emb.sin_cached" in name
        or "mtp" in name
    ):
        return None
    return name, None


def _parse_mimo_v2_routed_expert_name(
    name: str,
) -> tuple[int, int, str, str] | None:
    return _MIMO_V2_ROUTED_EXPERT_PATTERN.parse(name)


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("MiMoV2 UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"MiMoV2 UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    mlp = getattr(layer, "mlp", None)
    routed_experts = getattr(mlp, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "MiMoV2 UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def _reject_unsupported_fp8_qkv(catalog: TensorCatalog) -> None:
    unsupported = [
        name
        for name in catalog.names()
        if name.endswith("qkv_proj.weight_scale_inv")
        or (
            name.endswith("qkv_proj.weight")
            and "self_attn.qkv_proj.weight" in name
            and "q_proj" not in name
            and catalog.get(name).dtype == torch.float8_e4m3fn
        )
    ]
    if unsupported:
        raise RuntimeError(
            "MiMoV2 UMA plan does not yet support Pro-format fused FP8 qkv_proj "
            "pairs because they require paired tensor dequantize/reorder/requantize "
            f"before loading: {unsupported[:3]!r}"
        )


def _apply_attention_sink_slices(
    catalog: TensorCatalog,
    plan: WeightPlan,
) -> WeightPlan:
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return plan

    entries: list[WeightPlanEntry] = []
    for entry in plan.entries:
        if entry.required and "attention_sink_bias" in entry.target_name:
            # MiMoV2 stores one sink per total attention head. Each TP rank owns
            # a contiguous head range, matching the regular loader's narrow().
            record = catalog.get(entry.checkpoint_name)
            if len(record.shape) != 1 or record.shape[0] % tp_size != 0:
                raise RuntimeError(
                    "MiMoV2 UMA plan cannot shard attention_sink_bias with "
                    f"shape {record.shape} across TP size {tp_size}: "
                    f"{entry.checkpoint_name}"
                )
            heads_per_rank = record.shape[0] // tp_size
            head_start = tp_rank * heads_per_rank
            sliced_entry = SliceRule(
                entry.checkpoint_name,
                (
                    SliceRuleEntry(
                        entry.target_name,
                        (slice(head_start, head_start + heads_per_rank),),
                        entry.shard_id,
                    ),
                ),
            ).to_weight_plan_entries()[0]
            entries.append(
                WeightPlanEntry(
                    checkpoint_name=sliced_entry.checkpoint_name,
                    target_name=sliced_entry.target_name,
                    required=entry.required,
                    source_slices=sliced_entry.source_slices,
                    shard_id=sliced_entry.shard_id,
                    transform_ops=entry.transform_ops,
                    ignore_missing=entry.ignore_missing,
                )
            )
        else:
            entries.append(entry)
    return WeightPlan(tuple(entries))


def build_mimo_v2_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
) -> WeightPlan:
    _reject_unsupported_fp8_qkv(catalog)
    plan = build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="MiMoV2",
        parse_name=_parse_mimo_v2_routed_expert_name,
        map_projection=_MIMO_V2_ROUTED_PROJECTION_MAP.map,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=_mimo_v2_weight_mapper(),
        name_transform=_mimo_v2_name_transform,
    )
    return _apply_attention_sink_slices(catalog, plan)


def build_mimo_v2_flash_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
) -> WeightPlan:
    return build_auto_weight_plan_for_module(
        model,
        catalog,
        mapper=_mimo_v2_weight_mapper(),
        name_transform=_mimo_v2_name_transform,
    )


def load_mimo_v2_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: WeightPlan,
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
    )


def load_mimo_v2_flash_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: WeightPlan,
) -> set[str]:
    return execute_weight_plan(model, source, plan)
