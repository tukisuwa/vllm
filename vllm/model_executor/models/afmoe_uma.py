# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for AFMoE."""

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


AfmoeMoeRoutedEntry = RoutedMoeEntry
AfmoeMoeSourcePlan = WeightPlan


_AFMOE_ROUTED_EXPERT_PATTERN = RoutedExpertPattern(
    module_path=("mlp", "experts"),
    projections=("gate_proj", "down_proj", "up_proj"),
)
_AFMOE_ROUTED_PROJECTION_MAP = RoutedProjectionMap(
    (
        RoutedProjectionRule("gate_proj", "w13", "w1"),
        RoutedProjectionRule("up_proj", "w13", "w3"),
        RoutedProjectionRule("down_proj", "w2", "w2"),
    )
)


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("AFMoE UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"AFMoE UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    if getattr(layer, "moe_enabled", True) is False:
        raise RuntimeError(
            "AFMoE UMA plan matched a routed expert tensor for dense "
            f"layer {layer_id}"
        )
    mlp = getattr(layer, "mlp", None)
    routed_experts = getattr(mlp, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "AFMoE UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def build_afmoe_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    mapper: WeightsMapper,
) -> AfmoeMoeSourcePlan:
    return build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="AFMoE",
        parse_name=_AFMOE_ROUTED_EXPERT_PATTERN.parse,
        map_projection=_AFMOE_ROUTED_PROJECTION_MAP.map,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=mapper,
    )


def load_afmoe_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: AfmoeMoeSourcePlan,
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
    )
