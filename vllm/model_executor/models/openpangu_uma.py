# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for OpenPangu models."""

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


OpenPanguRoutedEntry = RoutedMoeEntry
OpenPanguSourcePlan = WeightPlan

_OPENPANGU_STANDARD_STACKED_PROJECTIONS = StackedProjectionMap(
    (
        StackedProjectionRule(".q_proj", ".qkv_proj", "q"),
        StackedProjectionRule(".k_proj", ".qkv_proj", "k"),
        StackedProjectionRule(".v_proj", ".qkv_proj", "v"),
        StackedProjectionRule(".gate_proj", ".gate_up_proj", 0),
        StackedProjectionRule(".up_proj", ".gate_up_proj", 1),
    )
)
_OPENPANGU_MLA_STACKED_PROJECTIONS = StackedProjectionMap(
    (
        StackedProjectionRule(".q_a_proj", ".fused_qkv_a_proj", 0),
        StackedProjectionRule(".kv_a_proj_with_mqa", ".fused_qkv_a_proj", 1),
    )
)
_OPENPANGU_ROUTED_EXPERT_PATTERN = RoutedExpertPattern(
    module_path=("mlp", "experts"),
    projections=("gate_proj", "down_proj", "up_proj"),
)
_OPENPANGU_ROUTED_PROJECTION_MAP = RoutedProjectionMap(
    (
        RoutedProjectionRule("gate_proj", "w13", "w1"),
        RoutedProjectionRule("up_proj", "w13", "w3"),
        RoutedProjectionRule("down_proj", "w2", "w2"),
    )
)


def _openpangu_name_transform(name: str) -> tuple[str, None] | None:
    if name.endswith("e_score_correction_bias"):
        name = name.replace(
            "e_score_correction_bias",
            "gate.e_score_correction_bias",
        )
    return name, None


class _OpenPanguWeightsMapper:

    def __init__(self, model: nn.Module):
        self._params = dict(model.named_parameters())
        self._mapper = _OPENPANGU_STANDARD_STACKED_PROJECTIONS.as_weights_mapper()
        if getattr(model, "fuse_qkv_a_proj", False):
            self._mapper = (
                self._mapper
                | _OPENPANGU_MLA_STACKED_PROJECTIONS.as_weights_mapper()
            )

    def _map_name_with_shard(self, name: str):
        mapped = self._mapper._map_name_with_shard(name)
        if mapped is None:
            return None
        target_name, shard_id = mapped
        if target_name.endswith(".bias") and target_name not in self._params:
            return target_name, shard_id
        return target_name, shard_id


def _parse_openpangu_routed_expert_name(
    name: str,
) -> tuple[int, int, str, str] | None:
    transformed = _openpangu_name_transform(name)
    if transformed is None:
        return None
    return _OPENPANGU_ROUTED_EXPERT_PATTERN.parse(transformed[0])


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("OpenPangu UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"OpenPangu UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    mlp = getattr(layer, "mlp", None)
    routed_experts = getattr(mlp, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "OpenPangu UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def _should_skip_openpangu_name(model: nn.Module, name: str) -> bool:
    if "rotary_emb.inv_freq" in name:
        return True
    config = model.config
    if config.tie_word_embeddings and "lm_head.weight" in name:
        return True
    if (
        "layers" in name
        and getattr(config, "num_nextn_predict_layers", 0) > 0
        and hasattr(config, "num_hidden_layers")
    ):
        try:
            layer_idx = int(name.split("layers.")[-1].split(".")[0])
        except (IndexError, ValueError):
            return False
        mtp_idx = layer_idx - config.num_hidden_layers
        if 0 <= mtp_idx < config.num_nextn_predict_layers:
            return True
    return False


def build_openpangu_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
) -> OpenPanguSourcePlan:
    return build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="OpenPangu",
        parse_name=_parse_openpangu_routed_expert_name,
        map_projection=_OPENPANGU_ROUTED_PROJECTION_MAP.map,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=_OpenPanguWeightsMapper(model),
        name_transform=_openpangu_name_transform,
        skip_predicate=lambda name: _should_skip_openpangu_name(model, name),
    )


def load_openpangu_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: OpenPanguSourcePlan,
) -> set[str]:
    loaded = load_routed_moe_weights_from_source(
        model,
        source,
        plan,
    )
    post_weight_load = getattr(getattr(model, "model", None), "post_weight_load", None)
    if callable(post_weight_load):
        post_weight_load()
    return loaded
