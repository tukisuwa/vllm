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
    RoutedExpertsResolution,
    RoutedMoeSourcePlan,
    build_routed_moe_weight_plan,
    load_routed_moe_weights_from_source,
)
from .utils import PPMissingLayer, WeightsMapper


MiniMaxM2MoeSourcePlan = RoutedMoeSourcePlan


def _parse_minimax_m2_routed_expert_name(
    name: str,
) -> tuple[int, int, str, str] | None:
    parts = name.split(".")
    for idx in range(len(parts) - 6):
        if parts[idx] != "layers":
            continue
        if (
            not parts[idx + 1].isdigit()
            or parts[idx + 2] != "mlp"
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


def _routed_param_for_projection(proj_name: str, suffix: str) -> tuple[str, str]:
    if proj_name == "w1":
        return f"w13_{suffix}", "w1"
    if proj_name == "w3":
        return f"w13_{suffix}", "w3"
    if proj_name == "w2":
        return f"w2_{suffix}", "w2"
    raise ValueError(f"Unsupported MiniMaxM2 routed expert projection {proj_name!r}")


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
        if name.startswith("model."):
            return name[len("model.") :], None
        return name, None

    plan = build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="MiniMaxM2 MoE",
        parse_name=_parse_minimax_m2_routed_expert_name,
        map_projection=_routed_param_for_projection,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=mapper,
        name_transform=name_transform,
        skip_prefixes=skip_prefixes,
    )

    entries = []
    for entry in plan.auto_plan:
        if entry.checkpoint_name.startswith("model."):
            entries.append(replace(entry, target_name=f"model.{entry.target_name}"))
        else:
            entries.append(entry)
    return MiniMaxM2MoeSourcePlan(WeightPlan(tuple(entries)), plan.routed_entries)


def load_minimax_m2_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: MiniMaxM2MoeSourcePlan,
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
        family_name="MiniMaxM2 MoE",
        get_routed_experts=_get_routed_experts_for_layer,
    )
