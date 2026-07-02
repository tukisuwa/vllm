# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMA-safe WeightSource helpers for DeepSeek V2/V3-style MoE models."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
    TensorCatalog,
)

from .routed_moe_uma import (
    RoutedMoeEntry,
    RoutedExpertsResolution,
    RoutedMoeSourcePlan,
    build_routed_moe_weight_plan,
    load_routed_moe_weights_from_source,
    routed_entry_requires_local_read,
)
from .utils import PPMissingLayer


@dataclass(frozen=True)
class DeepseekFp8IndexerWkEntry:
    weight_name: str
    scale_name: str
    target_name: str


@dataclass(frozen=True)
class DeepseekMoeSourcePlan:
    routed_plan: RoutedMoeSourcePlan
    fp8_indexer_wk_entries: tuple[DeepseekFp8IndexerWkEntry, ...] = ()


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


def _parse_deepseek_shared_expert_name(
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


def _deepseek_shared_expert_target_exists(
    mapper: _DeepseekSourceMapper,
    name: str,
) -> bool:
    mapped = mapper._map_name_with_shard(name)
    return mapped is not None and mapped[0] in mapper._params


def _build_deepseek_shared_expert_entries(
    model: nn.Module,
    catalog: TensorCatalog,
    mapper: _DeepseekSourceMapper,
) -> list[RoutedMoeEntry]:
    entries: list[RoutedMoeEntry] = []
    n_routed_experts = getattr(getattr(model, "config", None), "n_routed_experts", None)
    n_shared_experts = getattr(getattr(model, "config", None), "n_shared_experts", None)
    for name in catalog.names():
        parsed = _parse_deepseek_shared_expert_name(name)
        if parsed is None:
            continue
        if _deepseek_shared_expert_target_exists(mapper, name):
            continue
        if n_routed_experts is None or n_shared_experts is None:
            raise RuntimeError(
                "DeepSeek UMA plan found shared_experts tensors but model config "
                "does not define n_routed_experts/n_shared_experts"
            )
        if n_shared_experts <= 0:
            raise RuntimeError(
                "DeepSeek UMA plan found shared_experts tensors but "
                f"n_shared_experts={n_shared_experts}"
            )
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
                f"DeepSeek shared expert tensor {name} dimension {total} is not "
                f"divisible by n_shared_experts={n_shared_experts}"
            )
        chunk_size = total // n_shared_experts
        param_name, shard_id = _routed_param_for_projection(proj_name, suffix)
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


def _build_deepseek_fp8_indexer_wk_entries(
    model: nn.Module,
    catalog: TensorCatalog,
) -> list[DeepseekFp8IndexerWkEntry]:
    params = dict(model.named_parameters())
    indexer_present_prefixes = {
        name.rsplit(".indexer.", 1)[0]
        for name in params
        if ".indexer." in name
    }
    entries: list[DeepseekFp8IndexerWkEntry] = []
    for name in catalog.names():
        if "indexer.wk." not in name or "wk_weights" in name:
            continue
        if not name.endswith(".weight"):
            continue
        record = catalog.get(name)
        if record.dtype is not torch.float8_e4m3fn:
            continue
        layer_prefix = name.rsplit(".wk.", 1)[0]
        if layer_prefix.rsplit(".indexer", 1)[0] not in indexer_present_prefixes:
            continue
        scale_name = f"{name}_scale_inv"
        if not catalog.has(scale_name):
            raise RuntimeError(
                "DeepSeek UMA WeightSource hook found FP8 indexer.wk weight "
                f"{name}, but missing scale tensor {scale_name}"
            )
        target_name = f"{layer_prefix}.wk_weights_proj.weight"
        if target_name not in params:
            raise RuntimeError(
                "DeepSeek UMA WeightSource hook found FP8 indexer.wk weight "
                f"{name}, but target {target_name} is absent"
            )
        entries.append(
            DeepseekFp8IndexerWkEntry(
                weight_name=name,
                scale_name=scale_name,
                target_name=target_name,
            )
        )
    return entries


def build_deepseek_moe_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    skip_prefixes: list[str] | None = None,
    skip_predicate: Callable[[str], bool] | None = None,
) -> DeepseekMoeSourcePlan:
    mapper = _DeepseekSourceMapper(model)
    shared_expert_entries = _build_deepseek_shared_expert_entries(
        model,
        catalog,
        mapper,
    )
    fp8_indexer_wk_entries = _build_deepseek_fp8_indexer_wk_entries(model, catalog)
    fp8_indexer_names = {
        name
        for entry in fp8_indexer_wk_entries
        for name in (entry.weight_name, entry.scale_name)
    }

    indexer_present_prefixes = {
        name.rsplit(".indexer.", 1)[0]
        for name, _param in model.named_parameters()
        if ".indexer." in name
    }

    def combined_skip_predicate(name: str) -> bool:
        if skip_predicate is not None and skip_predicate(name):
            return True
        if "indexer.wk." in name:
            if name in fp8_indexer_names:
                return True
            if "weight_scale_inv" in name:
                return True
        if ".indexer." in name:
            return name.rsplit(".indexer.", 1)[0] not in indexer_present_prefixes
        return False

    return DeepseekMoeSourcePlan(
        routed_plan=build_routed_moe_weight_plan(
            model,
            catalog,
            family_name="DeepSeek MoE",
            parse_name=_parse_deepseek_routed_expert_name,
            map_projection=_routed_param_for_projection,
            resolve_routed_experts=_resolve_routed_experts_for_layer,
            auto_skip_substr=".mlp.experts.",
            mapper=mapper,
            skip_prefixes=skip_prefixes,
            skip_predicate=combined_skip_predicate,
            extra_routed_entries=shared_expert_entries,
        ),
        fp8_indexer_wk_entries=tuple(fp8_indexer_wk_entries),
    )


def load_deepseek_moe_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: DeepseekMoeSourcePlan,
) -> set[str]:
    loaded = load_routed_moe_weights_from_source(
        model,
        source,
        plan.routed_plan,
        family_name="DeepSeek MoE",
        get_routed_experts=_get_routed_experts_for_layer,
    )
    params = dict(model.named_parameters())
    for entry in plan.fp8_indexer_wk_entries:
        weight_fp8 = source.read_full_cpu(entry.weight_name)
        scale_inv = source.read_full_cpu(entry.scale_name)
        if weight_fp8.ndim != 2 or scale_inv.ndim != 2:
            raise RuntimeError(
                "DeepSeek UMA FP8 indexer.wk fusion requires 2D weight and "
                f"scale tensors: {entry.weight_name}, {entry.scale_name}"
            )
        if weight_fp8.shape[1] % scale_inv.shape[1] != 0:
            raise RuntimeError(
                "DeepSeek UMA FP8 indexer.wk fusion cannot infer block size: "
                f"weight_shape={list(weight_fp8.shape)}, "
                f"scale_shape={list(scale_inv.shape)}"
            )
        block_size = weight_fp8.shape[1] // scale_inv.shape[1]
        weight_bf16 = scaled_dequantize(
            weight_fp8,
            scale_inv,
            group_shape=GroupShape(block_size, block_size),
            out_dtype=torch.bfloat16,
        )
        param = params[entry.target_name]
        param.weight_loader(param, weight_bf16, 0)
        loaded.add(entry.target_name)
    return loaded
