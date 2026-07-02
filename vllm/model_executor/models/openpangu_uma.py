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
from vllm.model_executor.models.utils import WeightsMapper

from .routed_moe_uma import (
    RoutedExpertsResolution,
    RoutedMoeEntry,
    build_routed_moe_weight_plan,
    load_routed_moe_weights_from_source,
)
from .utils import PPMissingLayer


OpenPanguRoutedEntry = RoutedMoeEntry
OpenPanguSourcePlan = WeightPlan


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
        orig_to_new_stacked = {
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
        if getattr(model, "fuse_qkv_a_proj", False):
            orig_to_new_stacked.update(
                {
                    ".q_a_proj": (".fused_qkv_a_proj", 0),
                    ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
                }
            )
        self._mapper = WeightsMapper(orig_to_new_stacked=orig_to_new_stacked)

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
    name = transformed[0]
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
    raise ValueError(f"Unsupported OpenPangu expert projection {proj_name!r}")


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
        map_projection=_routed_param_for_projection,
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
