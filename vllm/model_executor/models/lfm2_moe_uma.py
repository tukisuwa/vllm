# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for LFM2 MoE models."""

from typing import Any

from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
)
from vllm.model_executor.model_loader.weight_plan import (
    TensorCatalog,
    WeightPlan,
)

from .routed_moe_uma import (
    RoutedExpertPattern,
    RoutedExpertsResolution,
    RoutedMoeEntry,
    RoutedProjectionMap,
    RoutedProjectionRule,
    build_routed_moe_weight_plan,
    load_routed_moe_weights_from_source,
)
from .utils import PPMissingLayer, WeightsMapper


Lfm2MoeRoutedEntry = RoutedMoeEntry
Lfm2MoeSourcePlan = WeightPlan

_LFM2_ROUTED_EXPERT_PATTERN = RoutedExpertPattern(
    module_path=("feed_forward", "experts"),
    projections=("w1", "w2", "w3"),
)
_LFM2_ROUTED_PROJECTION_MAP = RoutedProjectionMap(
    (
        RoutedProjectionRule("w1", "w13", "w1"),
        RoutedProjectionRule("w3", "w13", "w3"),
        RoutedProjectionRule("w2", "w2", "w2"),
    )
)


def _lfm2_moe_name_transform(name: str) -> tuple[str, None]:
    if "expert_bias" in name:
        name = name.replace("expert_bias", "gate.e_score_correction_bias")
    return name, None


def _lfm2_moe_weight_mapper(base_mapper: WeightsMapper) -> WeightsMapper:
    return base_mapper | WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".w1": (".w13", 0),
            ".w3": (".w13", 1),
        }
    )


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("LFM2 MoE UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"LFM2 MoE UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    feed_forward = getattr(layer, "feed_forward", None)
    routed_experts = getattr(feed_forward, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "LFM2 MoE UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def build_lfm2_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    mapper: WeightsMapper,
    skip_prefixes: list[str] | None = None,
) -> Lfm2MoeSourcePlan:
    return build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="LFM2 MoE",
        parse_name=_LFM2_ROUTED_EXPERT_PATTERN.parse,
        map_projection=_LFM2_ROUTED_PROJECTION_MAP.map,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".feed_forward.experts.",
        mapper=_lfm2_moe_weight_mapper(mapper),
        name_transform=_lfm2_moe_name_transform,
        skip_prefixes=skip_prefixes,
    )


def load_lfm2_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: Lfm2MoeSourcePlan,
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
    )
