# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for Qwen-family routed MoE models."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
    TensorCatalog,
    WeightPlan,
    build_auto_weight_plan_for_module,
    execute_weight_plan,
)

from .utils import PPMissingLayer


@dataclass(frozen=True)
class QwenMoeRoutedEntry:
    checkpoint_name: str
    layer_id: int
    expert_id: int
    param_name: str
    shard_id: str
    local_required: bool


@dataclass(frozen=True)
class QwenMoeSourcePlan:
    auto_plan: WeightPlan
    routed_entries: tuple[QwenMoeRoutedEntry, ...]


def _parse_routed_expert_name(
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
    raise ValueError(f"Unsupported routed expert projection {proj_name!r}")


def _get_model_layers(model: Any) -> Any | None:
    direct_layers = getattr(getattr(model, "model", None), "layers", None)
    if direct_layers is not None:
        return direct_layers
    language_model = getattr(model, "language_model", None)
    return getattr(getattr(language_model, "model", None), "layers", None)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    layers = _get_model_layers(model)
    if layers is None or layer_id < 0 or layer_id >= len(layers):
        return None
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return None
    mlp = getattr(layer, "mlp", None)
    if not hasattr(mlp, "experts"):
        return None
    routed_experts = getattr(mlp.experts, "routed_experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        return None
    return routed_experts


def _routed_entry_requires_local_read(
    routed_experts: Any,
    expert_id: int,
    weight_name: str,
) -> bool:
    map_global = getattr(
        routed_experts, "_map_global_expert_id_to_local_expert_id", None
    )
    if not callable(map_global):
        return True
    if map_global(expert_id) != -1:
        return True
    quant_method = getattr(routed_experts, "quant_method", None)
    use_global_sf = (
        getattr(quant_method, "use_global_sf", False) and "input_scale" in weight_name
    )
    return bool(use_global_sf)


def build_qwen_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    mapper: object | None = None,
    skip_prefixes: list[str] | None = None,
    skip_substrs: list[str] | None = None,
) -> QwenMoeSourcePlan:
    routed_entries: list[QwenMoeRoutedEntry] = []
    for name in catalog.names():
        parsed = _parse_routed_expert_name(name)
        if parsed is None:
            continue
        layer_id, expert_id, proj_name, suffix = parsed
        routed_experts = _get_routed_experts_for_layer(model, layer_id)
        if routed_experts is None:
            routed_entries.append(
                QwenMoeRoutedEntry(
                    checkpoint_name=name,
                    layer_id=layer_id,
                    expert_id=expert_id,
                    param_name="",
                    shard_id="",
                    local_required=False,
                )
            )
            continue
        param_name, shard_id = _routed_param_for_projection(proj_name, suffix)
        weight_name = f"{routed_experts.layer_name}.{param_name}"
        routed_entries.append(
            QwenMoeRoutedEntry(
                checkpoint_name=name,
                layer_id=layer_id,
                expert_id=expert_id,
                param_name=param_name,
                shard_id=shard_id,
                local_required=_routed_entry_requires_local_read(
                    routed_experts, expert_id, weight_name
                ),
            )
        )

    auto_skip_substrs = [*(skip_substrs or []), ".mlp.experts."]
    auto_plan = build_auto_weight_plan_for_module(
        model,
        catalog,
        mapper=mapper,
        skip_prefixes=skip_prefixes,
        skip_substrs=auto_skip_substrs,
    )
    routed_names = {entry.checkpoint_name for entry in routed_entries}
    auto_plan = WeightPlan(
        tuple(entry for entry in auto_plan if entry.checkpoint_name not in routed_names)
    )
    return QwenMoeSourcePlan(auto_plan, tuple(routed_entries))


def load_qwen_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: QwenMoeSourcePlan,
) -> set[str]:
    loaded = execute_weight_plan(model, source, plan.auto_plan)
    for entry in plan.routed_entries:
        if not entry.local_required:
            source.skip(entry.checkpoint_name, "non-local routed expert")
            continue
        routed_experts = _get_routed_experts_for_layer(model, entry.layer_id)
        if routed_experts is None:
            raise RuntimeError(
                "Qwen MoE UMA plan requires routed experts for "
                f"{entry.checkpoint_name}, but none were found"
            )
        if not hasattr(routed_experts, entry.param_name):
            raise RuntimeError(
                "Qwen MoE UMA plan target parameter "
                f"{entry.param_name!r} does not exist for {entry.checkpoint_name}"
            )
        tensor = source.read_full_cpu(entry.checkpoint_name)
        weight_name = f"{routed_experts.layer_name}.{entry.param_name}"
        success = routed_experts.weight_loader(
            param=getattr(routed_experts, entry.param_name),
            loaded_weight=tensor,
            weight_name=weight_name,
            shard_id=entry.shard_id,
            expert_id=entry.expert_id,
            return_success=True,
        )
        if not success:
            raise RuntimeError(
                "Qwen MoE routed expert weight_loader refused a tensor "
                "that the UMA plan marked local: "
                f"{entry.checkpoint_name}"
            )
        loaded.add(f"{routed_experts.layer_name}.{entry.param_name}")
    return loaded
