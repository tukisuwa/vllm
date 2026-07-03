# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for HunYuan v1 models."""

from dataclasses import dataclass
from typing import Any

from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
    execute_weight_plan,
)
from vllm.model_executor.model_loader.weight_plan import (
    TensorCatalog,
    WeightPlan,
    WeightPlanEntry,
)

from .hunyuan_v1 import _get_cla_factor, _is_moe
from .routed_moe_uma import (
    NameRewriteRule,
    NameRewriter,
    RoutedExpertPattern,
    RoutedExpertsResolution,
    RoutedMoeEntry,
    RoutedProjectionMap,
    RoutedProjectionRule,
    SliceRule,
    SliceRuleEntry,
    StackedProjectionMap,
    StackedProjectionRule,
    build_interleaved_row_gather_segments,
    build_routed_moe_weight_plan,
)
from .utils import PPMissingLayer

HunyuanV1RoutedEntry = RoutedMoeEntry


@dataclass(frozen=True)
class HunyuanV1SourcePlan:
    weight_plan: WeightPlan

_HUNYUAN_NAME_REWRITER = NameRewriter(
    (
        NameRewriteRule(".gate_proj_bias", ".gate_proj.bias"),
        NameRewriteRule(".up_proj_bias", ".up_proj.bias"),
        NameRewriteRule(".mlp.gate.wg.", ".mlp.gate."),
    )
)
_HUNYUAN_STACKED_PROJECTIONS = StackedProjectionMap(
    (
        StackedProjectionRule(".q_proj", ".qkv_proj", "q"),
        StackedProjectionRule(".k_proj", ".qkv_proj", "k"),
        StackedProjectionRule(".v_proj", ".qkv_proj", "v"),
        StackedProjectionRule(".gate_proj", ".gate_up_proj", 0),
        StackedProjectionRule(".up_proj", ".gate_up_proj", 1),
    )
)
_HUNYUAN_ROUTED_EXPERT_PATTERN = RoutedExpertPattern(
    module_path=("mlp", "experts"),
    projections=("gate_proj", "down_proj", "up_proj"),
)
_HUNYUAN_ROUTED_PROJECTION_MAP = RoutedProjectionMap(
    (
        RoutedProjectionRule("gate_proj", "w13", "w1"),
        RoutedProjectionRule("up_proj", "w13", "w3"),
        RoutedProjectionRule("down_proj", "w2", "w2"),
    )
)


def _hunyuan_name_transform(name: str):
    name = _HUNYUAN_NAME_REWRITER.apply(name)
    return name, None


def _parse_hunyuan_routed_expert_name(
    name: str,
) -> tuple[int, int, str, str] | None:
    transformed = _hunyuan_name_transform(name)
    if transformed is None:
        return None
    return _HUNYUAN_ROUTED_EXPERT_PATTERN.parse(transformed[0])


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    if not _is_moe(model.config):
        raise RuntimeError(
            "HunYuan UMA plan matched routed expert tensors for a dense model"
        )
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("HunYuan UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"HunYuan UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    mlp = getattr(layer, "mlp", None)
    routed_experts = getattr(mlp, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "HunYuan UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def _is_cross_layer_q_proj(model: Any, name: str) -> bool:
    cla_factor = _get_cla_factor(model.config)
    if cla_factor <= 1 or ".q_proj" not in name:
        return False
    parts = name.split(".")
    for idx in range(len(parts) - 1):
        if parts[idx] == "layers" and parts[idx + 1].isdigit():
            return int(parts[idx + 1]) % cla_factor != 0
    return False


def _collect_fused_plan_entries(
    model: nn.Module,
    catalog: TensorCatalog,
) -> tuple[list[WeightPlanEntry], set[str]]:
    entries: list[WeightPlanEntry] = []
    skip_auto: set[str] = set()
    for checkpoint_name in catalog.names():
        transformed = _hunyuan_name_transform(checkpoint_name)
        if transformed is None:
            continue
        name = transformed[0]
        if ".gate_and_up_proj." in name:
            record = catalog.get(checkpoint_name)
            if len(record.shape) < 1 or record.shape[0] % 2 != 0:
                raise RuntimeError(
                    "HunYuan UMA plan cannot split gate_and_up tensor "
                    f"{checkpoint_name}: shape={record.shape}"
                )
            half = record.shape[0] // 2
            target_name = name.replace(".gate_and_up_proj.", ".gate_up_proj.")
            rest = (slice(None),) * (len(record.shape) - 1)
            entries.extend(
                SliceRule(
                    checkpoint_name,
                    (
                        SliceRuleEntry(
                            target_name,
                            (slice(0, half), *rest),
                            1,
                        ),
                        SliceRuleEntry(
                            target_name,
                            (slice(half, record.shape[0]), *rest),
                            0,
                        ),
                    ),
                ).to_weight_plan_entries()
            )
            skip_auto.add(checkpoint_name)
            continue
        if ".qkv_proj." in name:
            record = catalog.get(checkpoint_name)
            num_heads = model.config.num_attention_heads
            num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
            head_dim = getattr(
                model.config,
                "head_dim",
                getattr(
                    model.config,
                    "attention_head_dim",
                    model.config.hidden_size // num_heads,
                ),
            )
            expected = (num_heads + num_kv_heads * 2) * head_dim
            if len(record.shape) < 2 or record.shape[0] != expected:
                raise RuntimeError(
                    "HunYuan UMA plan cannot split fused qkv tensor "
                    f"{checkpoint_name}: shape={record.shape}, "
                    f"expected first dim {expected}"
                )
            group_count = num_kv_heads
            group_rows = (num_heads // num_kv_heads + 2) * head_dim
            q_rows = (num_heads // num_kv_heads) * head_dim
            kv_rows = head_dim
            extra_dims = len(record.shape) - 1
            rest_shape = tuple(record.shape[1:])
            entries.extend(
                SliceRule(
                    checkpoint_name,
                    (
                        SliceRuleEntry(
                            name,
                            read_segments=build_interleaved_row_gather_segments(
                                group_count=group_count,
                                group_rows=group_rows,
                                block_start=0,
                                block_rows=q_rows,
                                extra_dims=extra_dims,
                            ),
                            staging_shape=(num_heads * head_dim, *rest_shape),
                            shard_id="q",
                        ),
                        SliceRuleEntry(
                            name,
                            read_segments=build_interleaved_row_gather_segments(
                                group_count=group_count,
                                group_rows=group_rows,
                                block_start=q_rows,
                                block_rows=kv_rows,
                                extra_dims=extra_dims,
                            ),
                            staging_shape=(num_kv_heads * head_dim, *rest_shape),
                            shard_id="k",
                        ),
                        SliceRuleEntry(
                            name,
                            read_segments=build_interleaved_row_gather_segments(
                                group_count=group_count,
                                group_rows=group_rows,
                                block_start=q_rows + kv_rows,
                                block_rows=kv_rows,
                                extra_dims=extra_dims,
                            ),
                            staging_shape=(num_kv_heads * head_dim, *rest_shape),
                            shard_id="v",
                        ),
                    ),
                ).to_weight_plan_entries()
            )
            skip_auto.add(checkpoint_name)
    return entries, skip_auto


def build_hunyuan_v1_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
) -> HunyuanV1SourcePlan:
    fused_entries, skip_auto = _collect_fused_plan_entries(
        model,
        catalog,
    )
    skip_prefixes = ["lm_head."] if model.config.tie_word_embeddings else None
    weight_plan = build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="HunYuan",
        parse_name=_parse_hunyuan_routed_expert_name,
        map_projection=_HUNYUAN_ROUTED_PROJECTION_MAP.map,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=_HUNYUAN_STACKED_PROJECTIONS.as_weights_mapper(),
        name_transform=_hunyuan_name_transform,
        skip_prefixes=skip_prefixes,
        skip_predicate=lambda name: (
            name in skip_auto
            or ".gate_and_up_proj." in name
            or ".qkv_proj." in name
            or _is_cross_layer_q_proj(model, name)
        ),
    )
    entries = [
        entry
        for entry in weight_plan.entries
        if entry.checkpoint_name not in skip_auto
    ]
    return HunyuanV1SourcePlan(
        weight_plan=WeightPlan(tuple(entries) + tuple(fused_entries)),
    )


def load_hunyuan_v1_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: HunyuanV1SourcePlan,
) -> set[str]:
    return execute_weight_plan(model, source, plan.weight_plan)
