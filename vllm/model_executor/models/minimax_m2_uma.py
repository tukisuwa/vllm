# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for MiniMaxM2 MoE."""

from dataclasses import replace
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
    NameRewriteRule,
    NameRewriter,
    RoutedExpertPattern,
    RoutedExpertsResolution,
    RoutedProjectionMap,
    RoutedProjectionRule,
    build_routed_moe_weight_plan,
    load_routed_moe_weights_from_source,
)
from .utils import PPMissingLayer, WeightsMapper


MiniMaxM2MoeSourcePlan = WeightPlan

_MINIMAX_M2_NAME_REWRITER = NameRewriter(
    (NameRewriteRule("model.", ""),)
)
_MINIMAX_M2_ROUTED_EXPERT_PATTERN = RoutedExpertPattern(
    module_path=("mlp", "experts"),
    projections=("w1", "w2", "w3"),
)
_MINIMAX_M2_ROUTED_PROJECTION_MAP = RoutedProjectionMap(
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
        raise RuntimeError("MiniMaxM2 UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"MiniMaxM2 UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    moe = getattr(layer, "block_sparse_moe", None)
    routed_experts = getattr(moe, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "MiniMaxM2 UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def build_minimax_m2_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    mapper: WeightsMapper,
) -> MiniMaxM2MoeSourcePlan:
    skip_prefixes = None
    num_mtp = getattr(model.config, "num_mtp_modules", 0)
    if num_mtp:
        base = model.config.num_hidden_layers
        skip_prefixes = [f"layers.{base + i}." for i in range(num_mtp)]

    def name_transform(name: str):
        return _MINIMAX_M2_NAME_REWRITER.apply(name), None

    plan = build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="MiniMaxM2 MoE",
        parse_name=lambda name: _MINIMAX_M2_ROUTED_EXPERT_PATTERN.parse(
            _MINIMAX_M2_NAME_REWRITER.apply(name)
        ),
        map_projection=_MINIMAX_M2_ROUTED_PROJECTION_MAP.map,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=mapper,
        name_transform=name_transform,
        skip_prefixes=skip_prefixes,
    )

    entries = []
    for entry in plan:
        if entry.expert_id is None and entry.checkpoint_name.startswith("model."):
            entries.append(replace(entry, target_name=f"model.{entry.target_name}"))
        else:
            entries.append(entry)
    return WeightPlan(tuple(entries))


def load_minimax_m2_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: MiniMaxM2MoeSourcePlan,
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
    )
