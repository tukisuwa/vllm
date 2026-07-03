# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for Llama4 models."""

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
    WeightPlanReadSegment,
)

from .llama4 import Llama4MoE
from .routed_moe_uma import (
    RoutedExpertPattern,
    RoutedExpertsResolution,
    RoutedMoeEntry,
    RoutedProjectionMap,
    RoutedProjectionRule,
    SliceRule,
    SliceRuleEntry,
    StackedProjectionMap,
    StackedProjectionRule,
    build_routed_moe_weight_plan,
)
from .utils import PPMissingLayer


Llama4RoutedEntry = RoutedMoeEntry

Llama4SourcePlan = WeightPlan

_LLAMA4_STACKED_PROJECTIONS = StackedProjectionMap(
    (
        StackedProjectionRule(".q_proj", ".qkv_proj", "q"),
        StackedProjectionRule(".k_proj", ".qkv_proj", "k"),
        StackedProjectionRule(".v_proj", ".qkv_proj", "v"),
        StackedProjectionRule(".gate_proj", ".gate_up_proj", 0),
        StackedProjectionRule(".up_proj", ".gate_up_proj", 1),
    )
)
_LLAMA4_ROUTED_EXPERT_PATTERN = RoutedExpertPattern(
    module_path=("feed_forward", "experts"),
    projections=("gate_proj", "down_proj", "up_proj"),
)
_LLAMA4_ROUTED_PROJECTION_MAP = RoutedProjectionMap(
    (
        RoutedProjectionRule("gate_proj", "w13", "w1"),
        RoutedProjectionRule("up_proj", "w13", "w3"),
        RoutedProjectionRule("down_proj", "w2", "w2"),
    )
)


def _llama4_name_transform(
    model: Any,
    catalog: TensorCatalog,
    checkpoint_name: str,
) -> tuple[str, tuple[TransformOp, ...] | None]:
    modules = checkpoint_name.split(".")
    leaf = modules[-1]
    is_weight = leaf in ("weight", "weight_packed")
    is_weight_scale = leaf == "weight_scale" and catalog.numel(checkpoint_name) > 1
    is_k_proj = "wk" in modules or "k_proj" in modules
    is_q_proj = "wq" in modules or "q_proj" in modules
    if not ((is_weight or is_weight_scale) and (is_k_proj or is_q_proj)):
        return checkpoint_name, None
    n_heads = (
        model.config.num_key_value_heads
        if is_k_proj
        else model.config.num_attention_heads
    )
    return checkpoint_name, (TransformOp("qk_rope_permute", (n_heads,)),)


def _parse_llama4_routed_expert_name(
    name: str,
) -> tuple[int, int, str, str] | None:
    return _LLAMA4_ROUTED_EXPERT_PATTERN.parse(name)


def _resolve_routed_experts_for_layer(
    model: Any,
    layer_id: int,
) -> RoutedExpertsResolution:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Llama4 UMA plan could not find model layers")
    if layer_id < 0 or layer_id >= len(layers):
        raise RuntimeError(
            f"Llama4 UMA plan found checkpoint layer {layer_id}, "
            f"but model has {len(layers)} layers"
        )
    layer = layers[layer_id]
    if isinstance(layer, PPMissingLayer):
        return RoutedExpertsResolution(None, "pipeline-missing routed expert layer")
    feed_forward = getattr(layer, "feed_forward", None)
    if not isinstance(feed_forward, Llama4MoE):
        raise RuntimeError(
            "Llama4 UMA plan matched routed expert tensor for dense "
            f"layer {layer_id}"
        )
    routed_experts = getattr(feed_forward, "experts", None)
    if routed_experts is None or not hasattr(routed_experts, "weight_loader"):
        raise RuntimeError(
            "Llama4 UMA plan matched a routed expert tensor for "
            f"layer {layer_id}, but no FusedMoE weight_loader was found"
        )
    return RoutedExpertsResolution(routed_experts)


def _get_routed_experts_for_layer(model: Any, layer_id: int) -> Any | None:
    return _resolve_routed_experts_for_layer(model, layer_id).routed_experts


def _local_expert_slice(routed_experts: Any) -> tuple[tuple[slice, ...] | None, int]:
    expert_map = getattr(routed_experts, "expert_map", None)
    if expert_map is None:
        return None, 0
    local = (expert_map != -1).nonzero().flatten().tolist()
    if not local:
        raise RuntimeError("Llama4 fused expert tensor has no local experts")
    start = int(local[0])
    stop = int(local[-1]) + 1
    if local != list(range(start, stop)):
        raise RuntimeError(
            "Llama4 UMA fused expert source slicing requires contiguous "
            f"local experts, got {local}"
        )
    return (slice(start, stop),), start


def _parse_fused_expert_name(name: str) -> tuple[int, str] | None:
    parts = name.split(".")
    for idx in range(len(parts) - 5):
        if parts[idx] != "layers":
            continue
        if (
            not parts[idx + 1].isdigit()
            or parts[idx + 2] != "feed_forward"
            or parts[idx + 3] != "experts"
        ):
            continue
        proj_name = parts[idx + 4]
        if proj_name not in ("gate_up_proj", "down_proj"):
            continue
        return int(parts[idx + 1]), proj_name
    return None


def _fused_expert_plan_entry(
    *,
    checkpoint_name: str,
    target_name: str,
    source_slices: tuple[slice | int, ...] | None = None,
    read_segments: tuple[WeightPlanReadSegment, ...] | None = None,
    staging_shape: tuple[int, ...] | None = None,
    shard_id: str,
    expert_id: int,
) -> WeightPlanEntry:
    return WeightPlanEntry(
        checkpoint_name=checkpoint_name,
        target_name=target_name,
        source_slices=source_slices,
        read_into_cpu=read_segments is not None,
        read_segments=read_segments,
        staging_shape=staging_shape,
        transform_ops=(TransformOp("transpose_last_two"),),
        shard_id=shard_id,
        expert_id=expert_id,
        weight_name=target_name,
    )


def _llama4_gate_up_segments(
    *,
    expert_axis: slice,
    record_shape: list[int],
    start: int,
    stop: int,
) -> tuple[tuple[WeightPlanReadSegment, ...], tuple[int, ...]]:
    expert_start, expert_stop, expert_step = expert_axis.indices(record_shape[0])
    if expert_step != 1:
        raise RuntimeError("Llama4 fused expert slices must be contiguous")
    expert_count = expert_stop - expert_start
    if expert_count <= 0:
        raise RuntimeError("Llama4 fused expert slice is empty")
    middle = record_shape[1]
    width = stop - start
    segments: list[WeightPlanReadSegment] = []
    for expert_offset, expert_idx in enumerate(range(expert_start, expert_stop)):
        for middle_idx in range(middle):
            segments.append(
                WeightPlanReadSegment(
                    (expert_idx, middle_idx, slice(start, stop)),
                    (expert_offset, middle_idx, slice(0, width)),
                )
            )
    return tuple(segments), (expert_count, middle, width)


def _collect_fused_expert_entries(
    model: nn.Module,
    catalog: TensorCatalog,
) -> tuple[list[WeightPlanEntry], set[str]]:
    entries: list[WeightPlanEntry] = []
    names: set[str] = set()
    for checkpoint_name in catalog.names():
        parsed = _parse_fused_expert_name(checkpoint_name)
        if parsed is None:
            continue
        layer_id, proj_name = parsed
        routed_experts = _get_routed_experts_for_layer(model, layer_id)
        if routed_experts is None:
            raise RuntimeError(
                "Llama4 UMA plan matched fused expert tensor for "
                f"layer {layer_id}, but no local routed experts were found"
            )
        expert_slices, expert_id = _local_expert_slice(routed_experts)
        record = catalog.get(checkpoint_name)
        if len(record.shape) != 3:
            raise RuntimeError(
                "Llama4 UMA fused expert tensors must be 3D: "
                f"{checkpoint_name} shape={record.shape}"
            )
        expert_axis = expert_slices[0] if expert_slices is not None else slice(None)
        if proj_name == "gate_up_proj":
            if record.shape[-1] % 2 != 0:
                raise RuntimeError(
                    "Llama4 UMA fused gate_up tensor last dimension must be even: "
                    f"{checkpoint_name} shape={record.shape}"
                )
            half = record.shape[-1] // 2
            target_name = checkpoint_name.replace(
                ".experts.gate_up_proj.",
                ".experts.w13_",
            )
            for shard_id, start, stop in (
                ("w1", 0, half),
                ("w3", half, record.shape[-1]),
            ):
                segments, staging_shape = _llama4_gate_up_segments(
                    expert_axis=expert_axis,
                    record_shape=record.shape,
                    start=start,
                    stop=stop,
                )
                entries.extend(
                    _fused_expert_plan_entry(
                        checkpoint_name=checkpoint_name,
                        target_name=target_name,
                        source_slices=entry.source_slices,
                        read_segments=entry.read_segments,
                        staging_shape=entry.staging_shape,
                        shard_id=entry.shard_id,
                        expert_id=expert_id,
                    )
                    for entry in SliceRule(
                        checkpoint_name,
                        (
                            SliceRuleEntry(
                                target_name,
                                read_segments=segments,
                                staging_shape=staging_shape,
                                shard_id=shard_id,
                            ),
                        ),
                    ).to_weight_plan_entries()
                )
        else:
            target_name = checkpoint_name.replace(
                ".experts.down_proj.",
                ".experts.w2_",
            )
            entries.extend(
                _fused_expert_plan_entry(
                    checkpoint_name=checkpoint_name,
                    target_name=target_name,
                    source_slices=entry.source_slices,
                    shard_id=entry.shard_id,
                    expert_id=expert_id,
                )
                for entry in SliceRule(
                    checkpoint_name,
                    (
                        SliceRuleEntry(
                            target_name,
                            (expert_axis, slice(None), slice(None)),
                            "w2",
                        ),
                    ),
                ).to_weight_plan_entries()
            )
        names.add(checkpoint_name)
    return entries, names


def build_llama4_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
) -> Llama4SourcePlan:
    fused_entries, fused_names = _collect_fused_expert_entries(model, catalog)
    weight_plan = build_routed_moe_weight_plan(
        model,
        catalog,
        family_name="Llama4",
        parse_name=_parse_llama4_routed_expert_name,
        map_projection=_LLAMA4_ROUTED_PROJECTION_MAP.map,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".feed_forward.experts.",
        mapper=_LLAMA4_STACKED_PROJECTIONS.as_weights_mapper(),
        name_transform=lambda name: _llama4_name_transform(model, catalog, name),
        skip_prefixes=(["lm_head."] if model.config.tie_word_embeddings else None),
        skip_predicate=lambda name: name in fused_names,
    )
    entries = [
        entry
        for entry in weight_plan.entries
        if entry.checkpoint_name not in fused_names
    ]
    return WeightPlan(tuple(entries) + tuple(fused_entries))


def load_llama4_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: Llama4SourcePlan,
) -> set[str]:
    return execute_weight_plan(model, source, plan)
