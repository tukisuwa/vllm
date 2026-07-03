# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for Llama4 models."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
    _call_weight_loader,
    execute_weight_plan,
    _resolve_attr,
)
from vllm.model_executor.model_loader.weight_plan import (
    TensorCatalog,
    TransformOp,
    WeightPlan,
)
from vllm.model_executor.models.utils import WeightsMapper

from .llama4 import Llama4MoE
from .routed_moe_uma import (
    RoutedExpertsResolution,
    RoutedMoeEntry,
    build_routed_moe_weight_plan,
)
from .utils import PPMissingLayer


Llama4RoutedEntry = RoutedMoeEntry


@dataclass(frozen=True)
class Llama4FusedExpertEntry:
    checkpoint_name: str
    layer_id: int
    target_name: str
    shard_id: str
    source_slices: tuple[slice | int, ...] | None
    expert_id: int
    kind: str


@dataclass(frozen=True)
class Llama4SourcePlan:
    weight_plan: WeightPlan
    fused_expert_entries: tuple[Llama4FusedExpertEntry, ...]


def _llama4_weight_mapper() -> WeightsMapper:
    return WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
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
    parts = name.split(".")
    for idx in range(len(parts) - 6):
        if parts[idx] != "layers":
            continue
        if (
            not parts[idx + 1].isdigit()
            or parts[idx + 2] != "feed_forward"
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
    raise ValueError(f"Unsupported Llama4 expert projection {proj_name!r}")


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


def _collect_fused_expert_entries(
    model: nn.Module,
    catalog: TensorCatalog,
) -> tuple[list[Llama4FusedExpertEntry], set[str]]:
    entries: list[Llama4FusedExpertEntry] = []
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
        tail = (slice(None),) * (len(record.shape) - 1)
        source_slices = None
        if expert_slices is not None:
            source_slices = (*expert_slices, *tail)
        if proj_name == "gate_up_proj":
            entries.extend(
                [
                    Llama4FusedExpertEntry(
                        checkpoint_name=checkpoint_name,
                        layer_id=layer_id,
                        target_name=checkpoint_name.replace(
                            ".experts.gate_up_proj.",
                            ".experts.w13_",
                        ),
                        shard_id="w1",
                        source_slices=source_slices,
                        expert_id=expert_id,
                        kind="gate_up",
                    ),
                    Llama4FusedExpertEntry(
                        checkpoint_name=checkpoint_name,
                        layer_id=layer_id,
                        target_name=checkpoint_name.replace(
                            ".experts.gate_up_proj.",
                            ".experts.w13_",
                        ),
                        shard_id="w3",
                        source_slices=source_slices,
                        expert_id=expert_id,
                        kind="gate_up",
                    ),
                ]
            )
        else:
            entries.append(
                Llama4FusedExpertEntry(
                    checkpoint_name=checkpoint_name,
                    layer_id=layer_id,
                    target_name=checkpoint_name.replace(
                        ".experts.down_proj.",
                        ".experts.w2_",
                    ),
                    shard_id="w2",
                    source_slices=source_slices,
                    expert_id=expert_id,
                    kind="down",
                )
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
        map_projection=_routed_param_for_projection,
        resolve_routed_experts=_resolve_routed_experts_for_layer,
        auto_skip_substr=".feed_forward.experts.",
        mapper=_llama4_weight_mapper(),
        name_transform=lambda name: _llama4_name_transform(model, catalog, name),
        skip_prefixes=(["lm_head."] if model.config.tie_word_embeddings else None),
        skip_predicate=lambda name: name in fused_names,
    )
    entries = [
        entry
        for entry in weight_plan.entries
        if entry.checkpoint_name not in fused_names
    ]
    return Llama4SourcePlan(
        weight_plan=WeightPlan(tuple(entries)),
        fused_expert_entries=tuple(fused_entries),
    )


def _read_fused_expert_tensor(
    source: ODirectSafetensorsWeightSource,
    entry: Llama4FusedExpertEntry,
) -> torch.Tensor:
    if entry.source_slices is None:
        return source.read_full_cpu(entry.checkpoint_name)
    return source.read_slice_cpu(entry.checkpoint_name, entry.source_slices)


def _dispatch_fused_expert_entry(
    model: nn.Module,
    entry: Llama4FusedExpertEntry,
    loaded_tensor: torch.Tensor,
) -> str:
    param = _resolve_attr(model, entry.target_name)
    weight_loader = getattr(param, "weight_loader", None)
    if not callable(weight_loader):
        raise RuntimeError(
            f"Llama4 UMA fused expert target {entry.target_name!r} "
            "has no weight_loader"
        )
    tensor = loaded_tensor
    if tensor.ndim == 3:
        tensor = tensor.transpose(-1, -2)
        if entry.kind == "gate_up":
            shard_idx = 0 if entry.shard_id == "w1" else 1
            tensor = tensor.chunk(2, dim=-2)[shard_idx]
    _call_weight_loader(
        weight_loader,
        param,
        tensor,
        source_is_sharded=False,
        kwargs={
            "weight_name": entry.target_name,
            "shard_id": entry.shard_id,
            "expert_id": entry.expert_id,
        },
        entry_name=entry.checkpoint_name,
    )
    return entry.target_name


def _load_fused_expert_entries(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    entries: tuple[Llama4FusedExpertEntry, ...],
) -> set[str]:
    loaded: set[str] = set()
    last_key: tuple[str, str] | None = None
    last_tensor: torch.Tensor | None = None
    for entry in entries:
        key = (entry.checkpoint_name, repr(entry.source_slices))
        if key != last_key:
            last_tensor = _read_fused_expert_tensor(source, entry)
            last_key = key
        assert last_tensor is not None
        loaded.add(_dispatch_fused_expert_entry(model, entry, last_tensor))
    return loaded


def load_llama4_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: Llama4SourcePlan,
) -> set[str]:
    loaded = execute_weight_plan(model, source, plan.weight_plan)
    loaded.update(_load_fused_expert_entries(model, source, plan.fused_expert_entries))
    return loaded
