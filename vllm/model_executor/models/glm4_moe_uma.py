# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for GLM4 MoE models."""

from collections.abc import Callable
from typing import Any

from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
)
from vllm.model_executor.model_loader.weight_plan import (
    TensorCatalog,
    WeightPlan,
)

from .routed_moe_uma import (
    RoutedExpertPattern,
    RoutedExpertsResolution,
    RoutedMoeEntry,
    RoutedProjectionMap,
    RoutedProjectionRule,
    StackedProjectionMap,
    StackedProjectionRule,
    build_routed_moe_weight_plan,
    load_routed_moe_weights_from_source,
    routed_entry_requires_local_read,
)
from .utils import PPMissingLayer


Glm4MoeRoutedEntry = RoutedMoeEntry
Glm4MoeSourcePlan = WeightPlan

_GLM4_STANDARD_STACKED_PROJECTIONS = StackedProjectionMap(
    (
        StackedProjectionRule(".q_proj", ".qkv_proj", "q"),
        StackedProjectionRule(".k_proj", ".qkv_proj", "k"),
        StackedProjectionRule(".v_proj", ".qkv_proj", "v"),
        StackedProjectionRule(".gate_proj", ".gate_up_proj", 0),
        StackedProjectionRule(".up_proj", ".gate_up_proj", 1),
    )
)
_GLM4_MLA_STACKED_PROJECTIONS = StackedProjectionMap(
    (
        StackedProjectionRule(".q_a_proj", ".fused_qkv_a_proj", 0),
        StackedProjectionRule(".kv_a_proj_with_mqa", ".fused_qkv_a_proj", 1),
    )
)
_GLM4_ROUTED_EXPERT_PATTERN = RoutedExpertPattern(
    module_path=("mlp", "experts"),
    projections=("gate_proj", "down_proj", "up_proj"),
)
_GLM4_ROUTED_PROJECTION_MAP = RoutedProjectionMap(
    (
        RoutedProjectionRule("gate_proj", "w13", "w1"),
        RoutedProjectionRule("up_proj", "w13", "w3"),
        RoutedProjectionRule("down_proj", "w2", "w2"),
    )
)


class _Glm4MoeSourceMapper:
    def __init__(self, *, include_mla: bool = False) -> None:
        self.mapper = _GLM4_STANDARD_STACKED_PROJECTIONS.as_weights_mapper()
        if include_mla:
            self.mapper = self.mapper | _GLM4_MLA_STACKED_PROJECTIONS.as_weights_mapper()

    def _map_name_with_shard(self, name: str) -> tuple[str, str | int | None] | None:
        return self.mapper._map_name_with_shard(name)


def _parse_glm4_moe_routed_expert_name(
    name: str,
) -> tuple[int, int, str, str] | None:
    return _GLM4_ROUTED_EXPERT_PATTERN.parse(name)


def _parse_glm4_moe_shared_expert_name(
    name: str,
) -> tuple[int, str, str] | None:
    parts = name.split(".")
    for idx in range(len(parts) - 5):
        if parts[idx] != "layers":
            continue
        if (
            not parts[idx + 1].isdigit()
            or parts[idx + 2] != "mlp"
            or parts[idx + 3] != "shared_experts"
        ):
            continue
        proj_name = parts[idx + 4]
        if proj_name not in ("gate_proj", "down_proj", "up_proj"):
            continue
        suffix = ".".join(parts[idx + 5 :])
        if not suffix:
            return None
        return int(parts[idx + 1]), proj_name, suffix
    return None


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("GLM4 MoE UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"GLM4 MoE UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    mlp = getattr(layer, "mlp", None)
    routed_experts = getattr(mlp, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "GLM4 MoE UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def _build_fused_shared_expert_entries(
    model: nn.Module,
    catalog: TensorCatalog,
    skip_predicate: Callable[[str], bool] | None,
) -> list[RoutedMoeEntry]:
    if not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
        return []
    n_routed_experts = getattr(getattr(model, "config", None), "n_routed_experts", None)
    n_shared_experts = getattr(getattr(model, "config", None), "n_shared_experts", None)
    if n_routed_experts is None or n_shared_experts is None:
        raise RuntimeError(
            "GLM4 MoE UMA FSE plan requires config.n_routed_experts and "
            "config.n_shared_experts"
        )
    if n_shared_experts <= 0:
        return []

    entries: list[RoutedMoeEntry] = []
    for name in catalog.names():
        if skip_predicate is not None and skip_predicate(name):
            continue
        parsed = _parse_glm4_moe_shared_expert_name(name)
        if parsed is None:
            continue
        layer_id, proj_name, suffix = parsed
        resolution = _resolve_routed_experts_for_layer(model, layer_id)
        routed_experts = resolution.routed_experts
        if routed_experts is None:
            entries.append(
                RoutedMoeEntry(
                    checkpoint_name=name,
                    layer_id=layer_id,
                    expert_id=0,
                    param_name="",
                    shard_id="",
                    local_required=False,
                    skip_reason=resolution.skip_reason,
                )
            )
            continue

        record = catalog.get(name)
        split_dim = 1 if proj_name == "down_proj" and len(record.shape) > 1 else 0
        total = record.shape[split_dim]
        if total % n_shared_experts != 0:
            raise RuntimeError(
                f"GLM4 MoE FSE shared expert tensor {name} dimension {total} is "
                f"not divisible by n_shared_experts={n_shared_experts}"
            )
        chunk_size = total // n_shared_experts
        param_name, shard_id = _GLM4_ROUTED_PROJECTION_MAP.map(proj_name, suffix)
        weight_name = f"{routed_experts.layer_name}.{param_name}"
        for shared_idx in range(n_shared_experts):
            source_slices: list[slice | int] = [
                slice(None) for _ in range(len(record.shape))
            ]
            source_slices[split_dim] = slice(
                shared_idx * chunk_size,
                (shared_idx + 1) * chunk_size,
            )
            expert_id = n_routed_experts + shared_idx
            entries.append(
                RoutedMoeEntry(
                    checkpoint_name=name,
                    layer_id=layer_id,
                    expert_id=expert_id,
                    param_name=param_name,
                    shard_id=shard_id,
                    local_required=routed_entry_requires_local_read(
                        routed_experts,
                        expert_id,
                        weight_name,
                    ),
                    source_slices=tuple(source_slices),
                )
            )
    return entries


def build_glm4_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    family_name: str = "GLM4 MoE",
    skip_predicate: Callable[[str], bool] | None = None,
    include_mla: bool = False,
) -> Glm4MoeSourcePlan:
    def parse_name(name: str):
        if skip_predicate is not None and skip_predicate(name):
            return None
        return _parse_glm4_moe_routed_expert_name(name)

    shared_expert_entries = _build_fused_shared_expert_entries(
        model,
        catalog,
        skip_predicate,
    )

    return build_routed_moe_weight_plan(
        model,
        catalog,
        family_name=family_name,
        parse_name=parse_name,
        map_projection=_GLM4_ROUTED_PROJECTION_MAP.map,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".mlp.experts.",
        mapper=_Glm4MoeSourceMapper(include_mla=include_mla),
        skip_predicate=skip_predicate,
        extra_routed_entries=shared_expert_entries,
    )


def load_glm4_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: Glm4MoeSourcePlan,
    *,
    family_name: str = "GLM4 MoE",
) -> set[str]:
    return load_routed_moe_weights_from_source(
        model,
        source,
        plan,
    )
