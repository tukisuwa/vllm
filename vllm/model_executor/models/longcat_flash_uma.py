# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for Longcat Flash models."""

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
    StackedProjectionMap,
    StackedProjectionRule,
    build_routed_moe_weight_plan,
    load_routed_moe_weights_from_source,
)
from .utils import PPMissingLayer


LongcatFlashMoeRoutedEntry = RoutedMoeEntry
LongcatFlashSourcePlan = WeightPlan

_LONGCAT_FLASH_STACKED_PROJECTIONS = StackedProjectionMap(
    (
        StackedProjectionRule("q_a_proj", "fused_qkv_a_proj", 0),
        StackedProjectionRule("kv_a_proj_with_mqa", "fused_qkv_a_proj", 1),
        StackedProjectionRule(".gate_proj", ".gate_up_proj", 0),
        StackedProjectionRule(".up_proj", ".gate_up_proj", 1),
    )
)
_LONGCAT_FLASH_ROUTED_EXPERT_PATTERN = RoutedExpertPattern(
    module_path=("mlp", "experts"),
    projections=("gate_proj", "down_proj", "up_proj"),
)
_LONGCAT_FLASH_ROUTED_PROJECTION_MAP = RoutedProjectionMap(
    (
        RoutedProjectionRule("gate_proj", "w13", "w1"),
        RoutedProjectionRule("up_proj", "w13", "w3"),
        RoutedProjectionRule("down_proj", "w2", "w2"),
    )
)


def _longcat_flash_name_transform(name: str) -> tuple[str, None] | None:
    if "rotary_emb.inv_freq" in name or ".mtp." in name:
        return None
    return name, None


class _LongcatFlashWeightsMapper:

    def __init__(self, model: nn.Module):
        self._params = dict(model.named_parameters())
        self._mapper = _LONGCAT_FLASH_STACKED_PROJECTIONS.as_weights_mapper()

    def _map_name_with_shard(self, name: str):
        # Longcat has a routed `mlp` and dense `mlps`. The ordinary loader
        # intentionally applies gate/up stacking only to `mlps`, not `mlp`.
        if "mlp" in name and "mlps" not in name:
            return name, None
        mapped = self._mapper._map_name_with_shard(name)
        if mapped is None:
            return None
        target_name, shard_id = mapped
        if target_name.endswith((".bias", "_bias")) and target_name not in self._params:
            return target_name, shard_id
        return target_name, shard_id


def _parse_longcat_routed_expert_name(
    name: str,
) -> tuple[int, int, str, str] | None:
    return _LONGCAT_FLASH_ROUTED_EXPERT_PATTERN.parse(name)


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Longcat Flash UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"Longcat Flash UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    mlp = getattr(layer, "mlp", None)
    routed_experts = getattr(mlp, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "Longcat Flash UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def build_longcat_flash_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
) -> LongcatFlashSourcePlan:
    return build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="Longcat Flash",
        parse_name=_parse_longcat_routed_expert_name,
        map_projection=_LONGCAT_FLASH_ROUTED_PROJECTION_MAP.map,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=_LongcatFlashWeightsMapper(model),
        name_transform=_longcat_flash_name_transform,
        ignore_unexpected_suffixes=[".kv_scale"],
    )


def load_longcat_flash_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: LongcatFlashSourcePlan,
) -> set[str]:
    loaded = load_routed_moe_weights_from_source(
        model,
        source,
        plan,
    )
    finalize = getattr(model, "_finalize_mla_weights", None)
    if not callable(finalize):
        finalize = getattr(getattr(model, "model", None), "_finalize_mla_weights", None)
    if not callable(finalize):
        raise RuntimeError("Longcat Flash UMA loader could not finalize MLA weights")
    finalize()
    return loaded
