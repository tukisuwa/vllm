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
    RoutedExpertsResolution,
    RoutedMoeEntry,
    build_routed_moe_weight_plan,
    load_routed_moe_weights_from_source,
)
from .utils import PPMissingLayer, WeightsMapper


Lfm2MoeRoutedEntry = RoutedMoeEntry
Lfm2MoeSourcePlan = WeightPlan


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


def _parse_lfm2_routed_expert_name(
    name: str,
) -> tuple[int, int, str, str] | None:
    parts = name.split(".")
    for idx in range(len(parts) - 6):
        if parts[idx] != "layers":
            continue
        if (
            not parts[idx + 1].isdigit()
            or parts[idx + 2] != "feed_forward"
            or parts[idx + 3] != "experts"
            or not parts[idx + 4].isdigit()
        ):
            continue
        proj_name = parts[idx + 5]
        if proj_name not in ("w1", "w2", "w3"):
            continue
        suffix = ".".join(parts[idx + 6 :])
        if not suffix:
            return None
        return int(parts[idx + 1]), int(parts[idx + 4]), proj_name, suffix
    return None


def _routed_param_for_projection(
    proj_name: str,
    suffix: str,
) -> tuple[str, str]:
    if proj_name == "w1":
        return f"w13_{suffix}", "w1"
    if proj_name == "w3":
        return f"w13_{suffix}", "w3"
    if proj_name == "w2":
        return f"w2_{suffix}", "w2"
    raise ValueError(f"Unsupported LFM2 MoE expert projection {proj_name!r}")


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
        parse_name=_parse_lfm2_routed_expert_name,
        map_projection=_routed_param_for_projection,
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
        family_name="LFM2 MoE",
        get_routed_experts=_get_routed_experts_for_layer,
    )
