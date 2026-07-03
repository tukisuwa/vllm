# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for Qwen-family routed MoE models."""

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


QwenMoeRoutedEntry = RoutedMoeEntry
QwenMoeSourcePlan = WeightPlan


_QWEN_ROUTED_EXPERT_PATTERN = RoutedExpertPattern(
    module_path=("mlp", "experts"),
    projections=("gate_proj", "down_proj", "up_proj"),
)
_QWEN_ROUTED_PROJECTION_MAP = RoutedProjectionMap(
    (
        RoutedProjectionRule("gate_proj", "w13", "w1"),
        RoutedProjectionRule("up_proj", "w13", "w3"),
        RoutedProjectionRule("down_proj", "w2", "w2"),
    )
)


def _get_model_layers(model: Any) -> Any | None:
    direct_layers = getattr(getattr(model, "model", None), "layers", None)
    if direct_layers is not None:
        return direct_layers
    language_model = getattr(model, "language_model", None)
    return getattr(getattr(language_model, "model", None), "layers", None)


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = _get_model_layers(model)
    if layers is None:
        raise RuntimeError("Qwen MoE UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"Qwen MoE UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    mlp = getattr(layer, "mlp", None)
    if not hasattr(mlp, "experts"):
        raise RuntimeError(
            "Qwen MoE UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but that model layer has no experts module"
        )
    routed_experts = getattr(mlp.experts, "routed_experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "Qwen MoE UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no RoutedExperts weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def build_qwen_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    family_name: str = "Qwen MoE",
    mapper: object | None = None,
    skip_prefixes: list[str] | None = None,
    skip_substrs: list[str] | None = None,
) -> QwenMoeSourcePlan:
    return build_routed_moe_weight_plan(
        model,
        catalog,
        family_name=family_name,
        parse_name=_QWEN_ROUTED_EXPERT_PATTERN.parse,
        map_projection=_QWEN_ROUTED_PROJECTION_MAP.map,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=mapper,
        skip_prefixes=skip_prefixes,
        skip_substrs=skip_substrs,
    )


def load_qwen_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: QwenMoeSourcePlan,
    *,
    family_name: str = "Qwen MoE",
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
    )
