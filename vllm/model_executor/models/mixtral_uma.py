# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for Mixtral-style routed MoE models."""

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
from .utils import PPMissingLayer


MixtralMoeRoutedEntry = RoutedMoeEntry
MixtralMoeSourcePlan = WeightPlan


_MIXTRAL_ROUTED_EXPERT_PATTERN = RoutedExpertPattern(
    module_path=("block_sparse_moe", "experts"),
    projections=("w1", "w2", "w3"),
)
_MIXTRAL_ROUTED_PROJECTION_MAP = RoutedProjectionMap(
    (
        RoutedProjectionRule("w1", "w13", "w1"),
        RoutedProjectionRule("w3", "w13", "w3"),
        RoutedProjectionRule("w2", "w2", "w2"),
    )
)


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Mixtral MoE UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"Mixtral MoE UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    block_sparse_moe = getattr(layer, "block_sparse_moe", None)
    routed_experts = getattr(block_sparse_moe, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "Mixtral MoE UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def build_mixtral_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    family_name: str = "Mixtral MoE",
    mapper: object | None = None,
    skip_prefixes: list[str] | None = None,
    skip_substrs: list[str] | None = None,
) -> MixtralMoeSourcePlan:
    return build_routed_moe_weight_plan(
        model,
        catalog,
        family_name=family_name,
        parse_name=_MIXTRAL_ROUTED_EXPERT_PATTERN.parse,
        map_projection=_MIXTRAL_ROUTED_PROJECTION_MAP.map,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".block_sparse_moe.experts.",
        mapper=mapper,
        skip_prefixes=skip_prefixes,
        skip_substrs=skip_substrs,
    )


def load_mixtral_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: MixtralMoeSourcePlan,
    *,
    family_name: str = "Mixtral MoE",
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
    )
