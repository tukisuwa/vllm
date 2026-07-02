# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for Ernie 4.5 routed MoE models."""

from collections.abc import Callable
from typing import Any

import torch
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


Ernie45MoeRoutedEntry = RoutedMoeEntry
Ernie45MoeSourcePlan = WeightPlan


class _Ernie45MoeSourceMapper:
    mapper = WeightsMapper(
        orig_to_new_stacked={
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
        },
    )

    def _map_name_with_shard(self, name: str) -> tuple[str, str | int | None] | None:
        return self.mapper._map_name_with_shard(name)


def _parse_ernie45_routed_expert_name(
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
        if proj_name not in ("gate_proj", "down_proj", "up_proj"):
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
    if proj_name == "gate_proj":
        return f"w13_{suffix}", "w1"
    if proj_name == "up_proj":
        return f"w13_{suffix}", "w3"
    if proj_name == "down_proj":
        return f"w2_{suffix}", "w2"
    raise ValueError(f"Unsupported Ernie 4.5 routed expert projection {proj_name!r}")


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Ernie 4.5 MoE UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"Ernie 4.5 MoE UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    mlp = getattr(layer, "mlp", None)
    routed_experts = getattr(mlp, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "Ernie 4.5 MoE UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def _squeeze_first_dim(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.squeeze(0)


def _name_transform(
    name: str,
) -> tuple[str, Callable[[torch.Tensor], torch.Tensor] | None] | None:
    if "e_score_correction_bias" in name:
        return name.replace("moe_statics", "gate"), _squeeze_first_dim
    return name, None


def _skip_predicate(model: nn.Module, name: str) -> bool:
    config = getattr(model, "config", None)
    if bool(getattr(config, "tie_word_embeddings", False)) and name.endswith(
        "lm_head.weight"
    ):
        return True
    return "mtp" in name


def build_ernie45_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
) -> Ernie45MoeSourcePlan:
    return build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="Ernie 4.5 MoE",
        parse_name=_parse_ernie45_routed_expert_name,
        map_projection=_routed_param_for_projection,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=_Ernie45MoeSourceMapper(),
        name_transform=_name_transform,
        skip_prefixes=(["lm_head."] if model.config.tie_word_embeddings else None),
        skip_predicate=lambda name: _skip_predicate(model, name),
    )


def load_ernie45_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: Ernie45MoeSourcePlan,
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
        family_name="Ernie 4.5 MoE",
        get_routed_experts=_get_routed_experts_for_layer,
    )
