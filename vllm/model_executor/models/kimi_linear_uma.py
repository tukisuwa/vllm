# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for Kimi Linear MoE models."""

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


KimiLinearMoeRoutedEntry = RoutedMoeEntry
KimiLinearMoeSourcePlan = WeightPlan

_KIMI_LINEAR_ROUTED_EXPERT_PATTERN = RoutedExpertPattern(
    module_path=("block_sparse_moe", "experts"),
    projections=("w1", "w2", "w3"),
)
_KIMI_LINEAR_ROUTED_PROJECTION_MAP = RoutedProjectionMap(
    (
        RoutedProjectionRule("w1", "w13", "w1"),
        RoutedProjectionRule("w3", "w13", "w3"),
        RoutedProjectionRule("w2", "w2", "w2"),
    )
)


class _KimiLinearSourceMapper:
    mapper = WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def _map_name_with_shard(self, name: str) -> tuple[str, str | int | None] | None:
        return self.mapper._map_name_with_shard(name)


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Kimi Linear MoE UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"Kimi Linear MoE UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    block_sparse_moe = getattr(layer, "block_sparse_moe", None)
    routed_experts = getattr(block_sparse_moe, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "Kimi Linear MoE UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def _get_spec_layer_idx_from_weight_name(config: Any, weight_name: str) -> int | None:
    num_nextn = getattr(config, "num_nextn_predict_layers", 0)
    if num_nextn and num_nextn > 0:
        layer_idx = config.num_hidden_layers
        for idx in range(num_nextn):
            if weight_name.startswith(f"model.layers.{layer_idx + idx}."):
                return layer_idx + idx
    return None


def build_kimi_linear_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
) -> KimiLinearMoeSourcePlan:
    def skip_predicate(name: str) -> bool:
        return _get_spec_layer_idx_from_weight_name(model.config, name) is not None

    return build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="Kimi Linear MoE",
        parse_name=_KIMI_LINEAR_ROUTED_EXPERT_PATTERN.parse,
        map_projection=_KIMI_LINEAR_ROUTED_PROJECTION_MAP.map,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".block_sparse_moe.experts.",
        mapper=_KimiLinearSourceMapper(),
        skip_prefixes=(["lm_head."] if model.config.tie_word_embeddings else None),
        skip_predicate=skip_predicate,
    )


def load_kimi_linear_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: KimiLinearMoeSourcePlan,
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
    )
