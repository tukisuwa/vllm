# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared UMA-safe WeightSource helpers for routed MoE model families."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
    execute_weight_plan,
)
from vllm.model_executor.model_loader.weight_plan import (
    TensorCatalog,
    TransformOp,
    WeightPlan,
    WeightPlanEntry,
    build_auto_weight_plan_for_module,
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
class RoutedExpertPattern:
    """Declarative parser for common per-expert checkpoint names.

    The pattern matches:
    ``... layers.<layer_id>.<module_path>.<expert_id>.<projection>.<suffix>``.
    Prefixes before ``layers`` are allowed so wrappers can reuse the same spec.
    """

    module_path: tuple[str, ...]
    projections: tuple[str, ...]
    layer_token: str = "layers"

    def parse(self, name: str) -> tuple[int, int, str, str] | None:
        parts = name.split(".")
        min_remaining = 3 + len(self.module_path)
        for idx in range(len(parts) - min_remaining):
            if parts[idx] != self.layer_token:
                continue
            layer_id_idx = idx + 1
            module_start = idx + 2
            expert_id_idx = module_start + len(self.module_path)
            projection_idx = expert_id_idx + 1
            suffix_start = projection_idx + 1
            if not parts[layer_id_idx].isdigit():
                continue
            if tuple(parts[module_start:expert_id_idx]) != self.module_path:
                continue
            if not parts[expert_id_idx].isdigit():
                continue
            projection = parts[projection_idx]
            if projection not in self.projections:
                continue
            suffix = ".".join(parts[suffix_start:])
            if not suffix:
                return None
            return (
                int(parts[layer_id_idx]),
                int(parts[expert_id_idx]),
                projection,
                suffix,
            )
        return None


@dataclass(frozen=True)
class RoutedProjectionRule:
    projection: str
    param_prefix: str
    shard_id: str


@dataclass(frozen=True)
class RoutedProjectionMap:
    rules: tuple[RoutedProjectionRule, ...]

    def map(self, projection: str, suffix: str) -> tuple[str, str]:
        for rule in self.rules:
            if rule.projection == projection:
                return f"{rule.param_prefix}_{suffix}", rule.shard_id
        raise ValueError(f"Unsupported routed expert projection {projection!r}")


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
        Callable[[str], tuple[str, tuple[TransformOp, ...] | None] | None] | None
    ) = None,
    skip_prefixes: list[str] | None = None,
    skip_substrs: list[str] | None = None,
    skip_predicate: Callable[[str], bool] | None = None,
    ignore_unexpected_suffixes: list[str] | None = None,
    extra_routed_entries: list[RoutedMoeEntry] | None = None,
) -> WeightPlan:
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
    return routed_moe_entries_to_weight_plan(
        model,
        auto_plan,
        tuple(routed_entries),
        family_name=family_name,
        get_routed_experts=lambda m, layer_id: resolve_routed_experts(
            m,
            layer_id,
        ).routed_experts,
    )


def _named_module_paths_by_id(model: nn.Module) -> dict[int, tuple[str, ...]]:
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return {}
    try:
        modules = named_modules(remove_duplicate=False)
    except TypeError:
        modules = named_modules()
    paths: dict[int, list[str]] = {}
    for name, module in modules:
        paths.setdefault(id(module), []).append(name)
    return {module_id: tuple(names) for module_id, names in paths.items()}


def _select_routed_experts_module_path(
    routed_experts: Any,
    module_paths: dict[int, tuple[str, ...]],
) -> str | None:
    paths = module_paths.get(id(routed_experts))
    if not paths:
        return None
    layer_name = getattr(routed_experts, "layer_name", None)
    if isinstance(layer_name, str):
        suffixes = (layer_name, layer_name.removeprefix("model."))
        for path in paths:
            if path in suffixes or any(
                path.endswith(f".{suffix}") for suffix in suffixes
            ):
                return path
    return paths[0]


def routed_moe_entries_to_weight_plan(
    model: nn.Module,
    auto_plan: WeightPlan,
    routed_entries: tuple[RoutedMoeEntry, ...],
    *,
    family_name: str,
    get_routed_experts: Callable[[nn.Module, int], Any | None],
) -> WeightPlan:
    """Fold routed entries into one executable `WeightPlan`.

    Routed expert payloads become first-class plan entries, so
    `summarize_weight_plan` accounts for them and a single executor performs
    the whole load instead of a side loop issuing its own reads.
    """

    module_paths: dict[int, tuple[str, ...]] | None = None
    entries: list[WeightPlanEntry] = list(auto_plan.entries)
    for entry in routed_entries:
        if not entry.local_required:
            entries.append(
                WeightPlanEntry(
                    checkpoint_name=entry.checkpoint_name,
                    target_name=entry.checkpoint_name,
                    required=False,
                    shard_id=entry.shard_id,
                    expert_id=entry.expert_id,
                    weight_name=entry.param_name,
                    source_slices=entry.source_slices,
                    skip_reason=entry.skip_reason or "non-local routed expert",
                )
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
        if module_paths is None:
            module_paths = _named_module_paths_by_id(model)
        module_path = _select_routed_experts_module_path(
            routed_experts,
            module_paths,
        )
        if module_path is None:
            raise RuntimeError(
                f"{family_name} routed expert module for "
                f"{entry.checkpoint_name} is not a registered model module"
            )
        target_name = f"{module_path}.{entry.param_name}"
        entries.append(
            WeightPlanEntry(
                checkpoint_name=entry.checkpoint_name,
                target_name=target_name,
                source_slices=entry.source_slices,
                shard_id=entry.shard_id,
                expert_id=entry.expert_id,
                weight_name=f"{routed_experts.layer_name}.{entry.param_name}",
                loader_target_name=module_path,
            )
        )
    return WeightPlan(tuple(entries))


def load_routed_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: WeightPlan,
) -> set[str]:
    return execute_weight_plan(model, source, plan)
