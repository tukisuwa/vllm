# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for DeepSeek V2/V3-style MoE models."""

from collections.abc import Callable
from typing import Any

from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
    TensorCatalog,
)

from .routed_moe_uma import (
    RoutedExpertsResolution,
    RoutedMoeSourcePlan,
    build_routed_moe_weight_plan,
    load_routed_moe_weights_from_source,
)
from .utils import PPMissingLayer


class _DeepseekSourceMapper:
    def __init__(self, model: nn.Module):
        self._params = dict(model.named_parameters())
        self._mappings: list[tuple[str, str, int | str]] = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        if getattr(model, "use_mha", False):
            self._mappings.extend(
                [
                    ("qkv_proj", "q_proj", "q"),
                    ("qkv_proj", "k_proj", "k"),
                    ("qkv_proj", "v_proj", "v"),
                ]
            )
        else:
            self._mappings.extend(
                [
                    ("fused_qkv_a_proj", "q_a_proj", 0),
                    ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
                ]
            )

    def _map_name_with_shard(self, name: str) -> tuple[str, int | str | None] | None:
        for param_name, weight_name, shard_id in self._mappings:
            if weight_name not in name:
                continue
            mapped_name = name.replace(weight_name, param_name, 1)
            if (
                param_name == "fused_qkv_a_proj"
                and mapped_name not in self._params
            ):
                continue
            if mapped_name.endswith(".bias") and mapped_name not in self._params:
                return mapped_name, shard_id
            return mapped_name, shard_id
        return name, None


def _parse_deepseek_routed_expert_name(
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


def _routed_param_for_projection(proj_name: str, suffix: str) -> tuple[str, str]:
    if proj_name == "gate_proj":
        return f"w13_{suffix}", "w1"
    if proj_name == "up_proj":
        return f"w13_{suffix}", "w3"
    if proj_name == "down_proj":
        return f"w2_{suffix}", "w2"
    raise ValueError(f"Unsupported DeepSeek routed expert projection {proj_name!r}")


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("DeepSeek UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"DeepSeek UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    mlp = getattr(layer, "mlp", None)
    routed_experts = getattr(mlp, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "DeepSeek UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def build_deepseek_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    skip_prefixes: list[str] | None = None,
    skip_predicate: Callable[[str], bool] | None = None,
) -> RoutedMoeSourcePlan:
    # The legacy DeepSeek loader has a special split path for fused AITER
    # shared experts. Keep the first UMA hook conservative until that transform
    # is represented as a model-side WeightPlan operation.
    if any("mlp.shared_experts." in name for name in catalog.names()):
        raise RuntimeError(
            "DeepSeek UMA WeightSource hook does not yet support "
            "mlp.shared_experts tensors"
        )

    indexer_present_prefixes = {
        name.rsplit(".indexer.", 1)[0]
        for name, _param in model.named_parameters()
        if ".indexer." in name
    }

    def combined_skip_predicate(name: str) -> bool:
        if skip_predicate is not None and skip_predicate(name):
            return True
        if ".indexer." in name:
            return name.rsplit(".indexer.", 1)[0] not in indexer_present_prefixes
        if "indexer.wk." in name:
            raise RuntimeError(
                "DeepSeek UMA WeightSource hook does not yet support FP8 "
                "indexer.wk fusion"
            )
        return False

    return build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="DeepSeek MoE",
        parse_name=_parse_deepseek_routed_expert_name,
        map_projection=_routed_param_for_projection,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=_DeepseekSourceMapper(model),
        skip_prefixes=skip_prefixes,
        skip_predicate=combined_skip_predicate,
    )


def load_deepseek_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: RoutedMoeSourcePlan,
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
        family_name="DeepSeek MoE",
        get_routed_experts=_get_routed_experts_for_layer,
    )
