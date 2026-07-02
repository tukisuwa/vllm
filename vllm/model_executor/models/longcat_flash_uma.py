# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for Longcat Flash models."""

from typing import Any

from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
    TensorCatalog,
)
from vllm.model_executor.models.utils import WeightsMapper

from .routed_moe_uma import (
    RoutedExpertsResolution,
    RoutedMoeEntry,
    RoutedMoeSourcePlan,
    build_routed_moe_weight_plan,
    load_routed_moe_weights_from_source,
)
from .utils import PPMissingLayer


LongcatFlashMoeRoutedEntry = RoutedMoeEntry
LongcatFlashSourcePlan = RoutedMoeSourcePlan


def _longcat_flash_name_transform(name: str) -> tuple[str, None] | None:
    if "rotary_emb.inv_freq" in name or ".mtp." in name:
        return None
    return name, None


class _LongcatFlashWeightsMapper:

    def __init__(self, model: nn.Module):
        self._params = dict(model.named_parameters())
        self._mapper = WeightsMapper(
            orig_to_new_stacked={
                "q_a_proj": ("fused_qkv_a_proj", 0),
                "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
                ".gate_proj": (".gate_up_proj", 0),
                ".up_proj": (".gate_up_proj", 1),
            }
        )

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
    raise ValueError(f"Unsupported Longcat Flash expert projection {proj_name!r}")


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
        map_projection=_routed_param_for_projection,
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
        family_name="Longcat Flash",
        get_routed_experts=_get_routed_experts_for_layer,
    )
    finalize = getattr(model, "_finalize_mla_weights", None)
    if not callable(finalize):
        finalize = getattr(getattr(model, "model", None), "_finalize_mla_weights", None)
    if not callable(finalize):
        raise RuntimeError("Longcat Flash UMA loader could not finalize MLA weights")
    finalize()
    return loaded
