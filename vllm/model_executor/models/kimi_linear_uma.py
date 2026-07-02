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
    RoutedExpertsResolution,
    RoutedMoeEntry,
    build_routed_moe_weight_plan,
    load_routed_moe_weights_from_source,
)
from .utils import PPMissingLayer, WeightsMapper


KimiLinearMoeRoutedEntry = RoutedMoeEntry
KimiLinearMoeSourcePlan = WeightPlan


class _KimiLinearSourceMapper:
    mapper = WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )

    def _map_name_with_shard(self, name: str) -> tuple[str, str | int | None] | None:
        return self.mapper._map_name_with_shard(name)


def _parse_kimi_linear_routed_expert_name(
    name: str,
) -> tuple[int, int, str, str] | None:
    parts = name.split(".")
    for idx in range(len(parts) - 6):
        if parts[idx] != "layers":
            continue
        if (
            not parts[idx + 1].isdigit()
            or parts[idx + 2] != "block_sparse_moe"
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
    raise ValueError(f"Unsupported Kimi Linear routed expert projection {proj_name!r}")


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
        parse_name=_parse_kimi_linear_routed_expert_name,
        map_projection=_routed_param_for_projection,
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
        family_name="Kimi Linear MoE",
        get_routed_experts=_get_routed_experts_for_layer,
    )
