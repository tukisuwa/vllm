# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for Param2MoE models."""

from typing import Any

from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
)
from vllm.model_executor.model_loader.weight_plan import (
    TensorCatalog,
    TransformOp,
    WeightPlan,
    WeightPlanEntry,
)
from vllm.model_executor.models.utils import WeightsMapper

from .routed_moe_uma import (
    NameRewriteRule,
    NameRewriter,
    RoutedExpertPattern,
    RoutedExpertsResolution,
    RoutedMoeEntry,
    RoutedProjectionMap,
    RoutedProjectionRule,
    build_routed_moe_weight_plan,
    load_routed_moe_weights_from_source,
)
from .utils import PPMissingLayer


Param2MoeRoutedEntry = RoutedMoeEntry
Param2MoeSourcePlan = WeightPlan

_PARAM2MOE_NAME_REWRITER = NameRewriter(
    (
        NameRewriteRule("model.word_embeddings.", "model.embed_tokens."),
        NameRewriteRule(".attention.query_key_value.", ".self_attn.qkv_proj."),
        NameRewriteRule(".attention.dense.", ".self_attn.o_proj."),
        NameRewriteRule(".attention.query_layernorm.", ".self_attn.q_layernorm."),
        NameRewriteRule(".attention.key_layernorm.", ".self_attn.k_layernorm."),
        NameRewriteRule(".attention.", ".self_attn."),
    )
)
_PARAM2MOE_ROUTED_EXPERT_PATTERN = RoutedExpertPattern(
    module_path=("mlp", "experts"),
    projections=("gate_proj", "down_proj", "up_proj"),
)
_PARAM2MOE_ROUTED_PROJECTION_MAP = RoutedProjectionMap(
    (
        RoutedProjectionRule("gate_proj", "w13", "w1"),
        RoutedProjectionRule("up_proj", "w13", "w3"),
        RoutedProjectionRule("down_proj", "w2", "w2"),
    )
)


def _param2moe_name_transform(name: str):
    name = _PARAM2MOE_NAME_REWRITER.apply(name)
    if name.endswith(".mlp.gate.expert_bias"):
        name = name.replace(
            ".mlp.gate.expert_bias",
            ".mlp.gate.e_score_correction_bias",
        )
        return name, (TransformOp("zero_mean"),)
    return name, None


def _param2moe_weight_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )


def _parse_param2moe_routed_expert_name(
    name: str,
) -> tuple[int, int, str, str] | None:
    transformed = _param2moe_name_transform(name)
    if transformed is None:
        return None
    return _PARAM2MOE_ROUTED_EXPERT_PATTERN.parse(transformed[0])


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Param2MoE UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"Param2MoE UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    mlp = getattr(layer, "mlp", None)
    routed_experts = getattr(mlp, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "Param2MoE UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def _qkv_split_entries(model: nn.Module, catalog: TensorCatalog) -> tuple[
    list[WeightPlanEntry],
    set[str],
]:
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim or (config.hidden_size // num_heads)
    q_split = num_heads * head_dim
    kv_split = num_kv_heads * head_dim
    entries: list[WeightPlanEntry] = []
    names: set[str] = set()
    for checkpoint_name in catalog.names():
        transformed = _param2moe_name_transform(checkpoint_name)
        if transformed is None:
            continue
        target_name = transformed[0]
        if not target_name.endswith(".self_attn.qkv_proj.weight"):
            continue
        record = catalog.get(checkpoint_name)
        if len(record.shape) < 1 or record.shape[0] != q_split + 2 * kv_split:
            raise RuntimeError(
                "Param2MoE UMA plan cannot split fused qkv tensor "
                f"{checkpoint_name}: shape={record.shape}, "
                f"expected first dim {q_split + 2 * kv_split}"
            )
        names.add(checkpoint_name)
        rest = (slice(None),) * (len(record.shape) - 1)
        entries.extend(
            [
                WeightPlanEntry(
                    checkpoint_name=checkpoint_name,
                    target_name=target_name,
                    source_slices=(slice(0, q_split), *rest),
                    shard_id="q",
                ),
                WeightPlanEntry(
                    checkpoint_name=checkpoint_name,
                    target_name=target_name,
                    source_slices=(slice(q_split, q_split + kv_split), *rest),
                    shard_id="k",
                ),
                WeightPlanEntry(
                    checkpoint_name=checkpoint_name,
                    target_name=target_name,
                    source_slices=(
                        slice(q_split + kv_split, q_split + 2 * kv_split),
                        *rest,
                    ),
                    shard_id="v",
                ),
            ]
        )
    return entries, names


def build_param2moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
) -> Param2MoeSourcePlan:
    qkv_entries, qkv_names = _qkv_split_entries(model, catalog)
    plan = build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="Param2MoE",
        parse_name=_parse_param2moe_routed_expert_name,
        map_projection=_PARAM2MOE_ROUTED_PROJECTION_MAP.map,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=_param2moe_weight_mapper(),
        name_transform=_param2moe_name_transform,
        skip_prefixes=(["lm_head."] if model.tie_word_embeddings else None),
    )
    entries = [
        entry for entry in plan.entries
        if entry.checkpoint_name not in qkv_names
    ]
    return WeightPlan(tuple(entries) + tuple(qkv_entries))


def load_param2moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: Param2MoeSourcePlan,
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
    )
