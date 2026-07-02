# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for Param2MoE models."""

from typing import Any

import torch
from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
)
from vllm.model_executor.model_loader.weight_plan import (
    TensorCatalog,
    WeightPlan,
    WeightPlanEntry,
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


Param2MoeRoutedEntry = RoutedMoeEntry
Param2MoeSourcePlan = RoutedMoeSourcePlan


def _zero_mean_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor - tensor.mean()


def _param2moe_name_transform(name: str):
    name = name.replace("model.word_embeddings.", "model.embed_tokens.")
    name = name.replace(".attention.query_key_value.", ".self_attn.qkv_proj.")
    name = name.replace(".attention.dense.", ".self_attn.o_proj.")
    name = name.replace(".attention.query_layernorm.", ".self_attn.q_layernorm.")
    name = name.replace(".attention.key_layernorm.", ".self_attn.k_layernorm.")
    name = name.replace(".attention.", ".self_attn.")
    if name.endswith(".mlp.gate.expert_bias"):
        name = name.replace(
            ".mlp.gate.expert_bias",
            ".mlp.gate.e_score_correction_bias",
        )
        return name, _zero_mean_tensor
    return name, None


def _param2moe_weight_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )


def _parse_param2moe_routed_expert_name(
    name: str,
) -> tuple[int, int, str, str] | None:
    transformed = _param2moe_name_transform(name)
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
    raise ValueError(f"Unsupported Param2MoE expert projection {proj_name!r}")


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Param2MoE UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"Param2MoE UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    mlp = getattr(layer, "mlp", None)
    routed_experts = getattr(mlp, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "Param2MoE UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def _qkv_split_entries(model: nn.Module, catalog: TensorCatalog) -> tuple[
    list[WeightPlanEntry],
    set[str],
]:
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim or (config.hidden_size // num_heads)
    q_split = num_heads * head_dim
    kv_split = num_kv_heads * head_dim
    entries: list[WeightPlanEntry] = []
    names: set[str] = set()
    for checkpoint_name in catalog.names():
        transformed = _param2moe_name_transform(checkpoint_name)
        if transformed is None:
            continue
        target_name = transformed[0]
        if not target_name.endswith(".self_attn.qkv_proj.weight"):
            continue
        record = catalog.get(checkpoint_name)
        if len(record.shape) < 1 or record.shape[0] != q_split + 2 * kv_split:
            raise RuntimeError(
                "Param2MoE UMA plan cannot split fused qkv tensor "
                f"{checkpoint_name}: shape={record.shape}, "
                f"expected first dim {q_split + 2 * kv_split}"
            )
        names.add(checkpoint_name)
        rest = (slice(None),) * (len(record.shape) - 1)
        entries.extend(
            [
                WeightPlanEntry(
                    checkpoint_name=checkpoint_name,
                    target_name=target_name,
                    source_slices=(slice(0, q_split), *rest),
                    shard_id="q",
                ),
                WeightPlanEntry(
                    checkpoint_name=checkpoint_name,
                    target_name=target_name,
                    source_slices=(slice(q_split, q_split + kv_split), *rest),
                    shard_id="k",
                ),
                WeightPlanEntry(
                    checkpoint_name=checkpoint_name,
                    target_name=target_name,
                    source_slices=(
                        slice(q_split + kv_split, q_split + 2 * kv_split),
                        *rest,
                    ),
                    shard_id="v",
                ),
            ]
        )
    return entries, names


def build_param2moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
) -> Param2MoeSourcePlan:
    qkv_entries, qkv_names = _qkv_split_entries(model, catalog)
    plan = build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="Param2MoE",
        parse_name=_parse_param2moe_routed_expert_name,
        map_projection=_routed_param_for_projection,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=_param2moe_weight_mapper(),
        name_transform=_param2moe_name_transform,
        skip_prefixes=(["lm_head."] if model.tie_word_embeddings else None),
    )
    auto_entries = [
        entry for entry in plan.auto_plan.entries
        if entry.checkpoint_name not in qkv_names
    ]
    return Param2MoeSourcePlan(
        auto_plan=WeightPlan(tuple(auto_entries) + tuple(qkv_entries)),
        routed_entries=plan.routed_entries,
    )


def load_param2moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: Param2MoeSourcePlan,
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
        family_name="Param2MoE",
        get_routed_experts=_get_routed_experts_for_layer,
    )
