# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for Bailing-style routed MoE models."""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
)
from vllm.model_executor.model_loader.weight_plan import (
    TransformOp,
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


BailingMoeRoutedEntry = RoutedMoeEntry
BailingMoeSourcePlan = WeightPlan
NameTransform = Callable[
    [str], tuple[str, tuple[TransformOp, ...] | None] | None
]


class _BailingSourceMapper:
    mapper = WeightsMapper(
        orig_to_new_stacked={
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
        },
    )

    def _map_name_with_shard(self, name: str) -> tuple[str, str | int | None] | None:
        return self.mapper._map_name_with_shard(name)


def _parse_bailing_routed_expert_name(
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
    raise ValueError(f"Unsupported Bailing routed expert projection {proj_name!r}")


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Bailing MoE UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"Bailing MoE UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    mlp = getattr(layer, "mlp", None)
    routed_experts = getattr(mlp, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "Bailing MoE UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def _compose_name_transform(
    first: NameTransform | None,
    second: NameTransform,
) -> NameTransform:
    def transform(
        name: str,
    ) -> tuple[str, tuple[TransformOp, ...] | None] | None:
        first_result = first(name) if first is not None else (name, None)
        if first_result is None:
            return None
        intermediate_name, first_ops = first_result
        second_result = second(intermediate_name)
        if second_result is None:
            return None
        final_name, second_ops = second_result
        # Composition of named ops is tuple concatenation, applied in order.
        return final_name, (*(first_ops or ()), *(second_ops or ())) or None

    return transform


def _bailing_name_transform(model: nn.Module) -> NameTransform:
    config = getattr(model, "config", None)
    norm_head = bool(getattr(config, "norm_head", False))

    def transform(
        name: str,
    ) -> tuple[str, tuple[TransformOp, ...] | None] | None:
        if norm_head and "lm_head.weight" in name:
            return name, (TransformOp("l2_normalize"),)
        return name, None

    return transform


def build_bailing_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    extra_name_transform: NameTransform | None = None,
) -> BailingMoeSourcePlan:
    name_transform = _compose_name_transform(
        extra_name_transform,
        _bailing_name_transform(model),
    )
    return build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="Bailing MoE",
        parse_name=_parse_bailing_routed_expert_name,
        map_projection=_routed_param_for_projection,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=_BailingSourceMapper(),
        name_transform=name_transform,
        skip_prefixes=(["lm_head."] if model.tie_word_embeddings else None),
    )


def load_bailing_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: BailingMoeSourcePlan,
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
    )
