# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared UMA-safe WeightSource helpers for routed MoE model families."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
    TensorCatalog,
    WeightPlan,
    build_auto_weight_plan_for_module,
    execute_weight_plan,
)


@dataclass(frozen=True)
class RoutedExpertsResolution:
    routed_experts: Any | None
    skip_reason: str | None = None


@dataclass(frozen=True)
class RoutedMoeEntry:
    checkpoint_name: str
    layer_id: int
    expert_id: int
    param_name: str
    shard_id: str
    local_required: bool
    skip_reason: str | None = None
    source_slices: tuple[slice | int, ...] | None = None


@dataclass(frozen=True)
class RoutedMoeSourcePlan:
    auto_plan: WeightPlan
    routed_entries: tuple[RoutedMoeEntry, ...]


RoutedNameParser = Callable[[str], tuple[int, int, str, str] | None]
RoutedProjectionMapper = Callable[[str, str], tuple[str, str]]
RoutedExpertResolver = Callable[[nn.Module, int], RoutedExpertsResolution]


def routed_entry_requires_local_read(
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


def build_routed_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    family_name: str,
    parse_name: RoutedNameParser,
    map_projection: RoutedProjectionMapper,
    resolve_routed_experts: RoutedExpertResolver,
    auto_skip_substr: str,
    mapper: object | None = None,
    name_transform: (
        Callable[[str], tuple[str, Callable[[Any], Any] | None] | None] | None
    ) = None,
    skip_prefixes: list[str] | None = None,
    skip_substrs: list[str] | None = None,
    skip_predicate: Callable[[str], bool] | None = None,
    ignore_unexpected_suffixes: list[str] | None = None,
    extra_routed_entries: list[RoutedMoeEntry] | None = None,
) -> RoutedMoeSourcePlan:
    routed_entries: list[RoutedMoeEntry] = []
    for name in catalog.names():
        parsed = parse_name(name)
        if parsed is None:
            continue
        layer_id, expert_id, proj_name, suffix = parsed
        resolution = resolve_routed_experts(model, layer_id)
        routed_experts = resolution.routed_experts
        if routed_experts is None:
            routed_entries.append(
                RoutedMoeEntry(
                    checkpoint_name=name,
                    layer_id=layer_id,
                    expert_id=expert_id,
                    param_name="",
                    shard_id="",
                    local_required=False,
                    skip_reason=resolution.skip_reason,
                )
            )
            continue
        param_name, shard_id = map_projection(proj_name, suffix)
        weight_name = f"{routed_experts.layer_name}.{param_name}"
        routed_entries.append(
            RoutedMoeEntry(
                checkpoint_name=name,
                layer_id=layer_id,
                expert_id=expert_id,
                param_name=param_name,
                shard_id=shard_id,
                local_required=routed_entry_requires_local_read(
                    routed_experts, expert_id, weight_name
                ),
            )
        )
    if extra_routed_entries:
        routed_entries.extend(extra_routed_entries)

    auto_skip_substrs = [*(skip_substrs or []), auto_skip_substr]
    auto_plan = build_auto_weight_plan_for_module(
        model,
        catalog,
        mapper=mapper,
        name_transform=name_transform,
        skip_prefixes=skip_prefixes,
        skip_substrs=auto_skip_substrs,
        skip_predicate=skip_predicate,
        ignore_unexpected_suffixes=ignore_unexpected_suffixes,
    )
    routed_names = {entry.checkpoint_name for entry in routed_entries}
    auto_plan = WeightPlan(
        tuple(entry for entry in auto_plan if entry.checkpoint_name not in routed_names)
    )
    return RoutedMoeSourcePlan(auto_plan, tuple(routed_entries))


def load_routed_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: RoutedMoeSourcePlan,
    *,
    family_name: str,
    get_routed_experts: Callable[[nn.Module, int], Any | None],
) -> set[str]:
    loaded = execute_weight_plan(model, source, plan.auto_plan)
    for entry in plan.routed_entries:
        if not entry.local_required:
            source.skip(
                entry.checkpoint_name,
                entry.skip_reason or "non-local routed expert",
            )
            continue
        routed_experts = get_routed_experts(model, entry.layer_id)
        if routed_experts is None:
            raise RuntimeError(
                f"{family_name} UMA plan requires routed experts for "
                f"{entry.checkpoint_name}, but none were found"
            )
        if not hasattr(routed_experts, entry.param_name):
            raise RuntimeError(
                f"{family_name} UMA plan target parameter "
                f"{entry.param_name!r} does not exist for {entry.checkpoint_name}"
            )
        if entry.source_slices is None:
            tensor = source.read_full_cpu(entry.checkpoint_name)
        else:
            tensor = source.read_slice_cpu(
                entry.checkpoint_name,
                entry.source_slices,
            )
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
                f"{family_name} routed expert weight_loader refused a tensor "
                "that the UMA plan marked local: "
                f"{entry.checkpoint_name}"
            )
        loaded.add(f"{routed_experts.layer_name}.{entry.param_name}")
    return loaded
