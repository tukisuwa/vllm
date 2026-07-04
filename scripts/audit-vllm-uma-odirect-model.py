#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata-only audit for uma_odirect_safetensors model directories.

This script intentionally reads safetensors headers only.  It uses the same
TensorCatalog validation as the UMA O_DIRECT loader, then prints a compact
summary that is useful before deciding whether a model should get a
model-side WeightSource hook.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from torch import nn

from vllm.model_executor.model_loader.weight_plan import (
    WeightPlanEntry,
    validate_weight_plan_read_segments,
)
from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    TensorCatalog,
    TensorMeta,
)


DEFAULT_METADATA_LIMIT_MIB = 16
DEFAULT_TOP_N = 20
DEFAULT_PLAN_EXAMPLES = 20


def _format_gib(nbytes: int) -> str:
    return f"{nbytes / (1024**3):.3f}GiB"


def _find_safetensors(model_dir: Path) -> list[str]:
    files = sorted(str(path) for path in model_dir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"No .safetensors files found in {model_dir}")
    return files


def _read_architectures(model_dir: Path) -> list[str]:
    data = _read_config(model_dir)
    architectures = data.get("architectures")
    if not isinstance(architectures, list):
        return []
    return [item for item in architectures if isinstance(item, str)]


def _read_config(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    if not config_path.exists() or os.path.islink(config_path):
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _classify_name(name: str) -> str:
    if ".mlp.experts." in name:
        return "per_expert_moe"
    if ".indexer.wk." in name and "weight_scale_inv" in name:
        return "deepseek_fp8_indexer_wk_scale"
    if ".indexer.wk." in name:
        return "deepseek_indexer_wk"
    if ".indexer." in name:
        return "deepseek_indexer"
    if ".block_sparse_moe.input_linear." in name:
        return "granite_input_linear"
    if ".block_sparse_moe.output_linear." in name:
        return "granite_output_linear"
    if ".block_sparse_moe.router." in name or ".block_sparse_moe.gate." in name:
        return "granite_router"
    if ".shared_experts." in name:
        return "shared_experts"
    if ".q_proj." in name or ".k_proj." in name or ".v_proj." in name:
        return "split_qkv"
    if ".qkv_proj." in name:
        return "packed_qkv"
    if ".gate_proj." in name or ".up_proj." in name:
        return "split_gate_up"
    if ".gate_up_proj." in name:
        return "packed_gate_up"
    if name.endswith(".lm_head.weight"):
        return "lm_head"
    if "embed_tokens" in name:
        return "embedding"
    return "other"


def _shape_text(record: TensorMeta) -> str:
    return _shape_dims_text(record.shape)


def _shape_dims_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def _config_section(config: dict[str, object]) -> dict[str, object]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config
    return config


def _config_value(config: dict[str, object], key: str) -> object:
    text_config = _config_section(config)
    if key in text_config:
        return text_config[key]
    return config.get(key)


def _config_int(config: dict[str, object], key: str) -> int:
    value = _config_value(config, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"config.{key} must be an integer for llama4 plan audit")
    return value


def _config_bool(config: dict[str, object], key: str, *, default: bool) -> bool:
    value = _config_value(config, key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise SystemExit(f"config.{key} must be a boolean for llama4 plan audit")
    return value


def _infer_layer_count(catalog: TensorCatalog, config: dict[str, object]) -> int:
    configured = _config_value(config, "num_hidden_layers")
    layer_count = 0
    if configured is not None:
        if isinstance(configured, bool) or not isinstance(configured, int):
            raise SystemExit("config.num_hidden_layers must be an integer")
        layer_count = configured
    max_layer = -1
    marker = ".layers."
    for name in catalog.names():
        idx = name.find(marker)
        if idx < 0:
            continue
        suffix = name[idx + len(marker) :]
        layer_token = suffix.split(".", 1)[0]
        if layer_token.isdigit():
            max_layer = max(max_layer, int(layer_token))
    return max(layer_count, max_layer + 1)


class _AuditLlama4Experts(nn.Module):
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_name = f"model.layers.{layer_id}.feed_forward.experts"
        self.w13_weight = object()
        self.w2_weight = object()
        self.quant_method = object()
        self.expert_map = None

    def weight_loader(self, **_kwargs: object) -> bool:
        return True


class _AuditLlama4MoE(nn.Module):
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.experts = _AuditLlama4Experts(layer_id)


class _AuditLlama4Layer(nn.Module):
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.feed_forward = _AuditLlama4MoE(layer_id)


class _AuditLlama4InnerModel(nn.Module):
    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_AuditLlama4Layer(layer_id) for layer_id in range(layer_count)]
        )


class _AuditLlama4Model(nn.Module):
    def __init__(self, config: dict[str, object], catalog: TensorCatalog) -> None:
        super().__init__()
        layer_count = _infer_layer_count(catalog, config)
        if layer_count <= 0:
            raise SystemExit("Could not infer any Llama4 layers from config/catalog")
        self.config = SimpleNamespace(
            tie_word_embeddings=_config_bool(
                config, "tie_word_embeddings", default=False
            ),
            num_attention_heads=_config_int(config, "num_attention_heads"),
            num_key_value_heads=_config_int(config, "num_key_value_heads"),
        )
        self.model = _AuditLlama4InnerModel(layer_count)


def _entry_staging_bytes(record: TensorMeta, entry: WeightPlanEntry) -> int:
    if entry.staging_shape is None:
        return 0
    elements = math.prod(entry.staging_shape)
    source_elements = math.prod(record.shape)
    if source_elements <= 0:
        return 0
    item_size = record.size // source_elements
    return elements * item_size


def _print_llama4_plan_audit(
    catalog: TensorCatalog,
    config: dict[str, object],
    *,
    examples: int,
) -> None:
    if not config:
        raise SystemExit("config.json is required for --uma-plan llama4")
    from vllm.model_executor.models import llama4_uma

    original_llama4_moe = llama4_uma.Llama4MoE
    llama4_uma.Llama4MoE = _AuditLlama4MoE
    try:
        model = _AuditLlama4Model(config, catalog)
        plan = llama4_uma.build_llama4_weight_plan(model, catalog)
    finally:
        llama4_uma.Llama4MoE = original_llama4_moe

    entries = list(plan.entries)
    segment_entries = [
        entry for entry in entries if entry.read_segments is not None
    ]
    source_slice_entries = [
        entry for entry in entries if entry.source_slices is not None
    ]
    segment_errors: list[str] = []
    segment_count = 0
    staging_bytes = 0
    fused_gate_up_entries = 0
    checkpoint_counts: Counter[str] = Counter()
    shard_counts: Counter[str] = Counter()
    for entry in segment_entries:
        record = catalog.get(entry.checkpoint_name)
        try:
            validate_weight_plan_read_segments(record, entry)
        except Exception as exc:
            segment_errors.append(f"{entry.checkpoint_name}: {exc}")
            continue
        segment_count += len(entry.read_segments or ())
        staging_bytes += _entry_staging_bytes(record, entry)
        checkpoint_counts[entry.checkpoint_name] += 1
        shard_counts[str(entry.shard_id)] += 1
        if ".feed_forward.experts.gate_up_proj." in entry.checkpoint_name:
            fused_gate_up_entries += 1

    print(
        "uma_plan"
        " family=llama4"
        f" entries={len(entries)}"
        f" read_segment_entries={len(segment_entries)}"
        f" source_slice_entries={len(source_slice_entries)}"
        f" read_segments={segment_count}"
        f" staging_payload={_format_gib(staging_bytes)}"
        f" fused_gate_up_read_segment_entries={fused_gate_up_entries}"
    )
    for shard_id, count in sorted(shard_counts.items()):
        print(f"uma_plan_shard shard_id={shard_id} read_segment_entries={count}")
    print(
        "uma_plan_validation"
        f" read_segments={'ok' if not segment_errors else 'failed'}"
        f" errors={len(segment_errors)}"
    )
    for error in segment_errors[:examples]:
        print(f"uma_plan_error {error}")
    for checkpoint_name, count in checkpoint_counts.most_common(examples):
        record = catalog.get(checkpoint_name)
        print(
            "uma_plan_checkpoint"
            f" entries={count}"
            f" dtype={record.dtype}"
            f" source_shape={_shape_text(record)}"
            f" payload={_format_gib(record.size)}"
            f" name={checkpoint_name}"
        )
    for entry in segment_entries[:examples]:
        record = catalog.get(entry.checkpoint_name)
        print(
            "uma_plan_read_segment"
            f" segments={len(entry.read_segments or ())}"
            f" staging_shape={_shape_dims_text(entry.staging_shape or ())}"
            f" shard_id={entry.shard_id}"
            f" expert_id={entry.expert_id}"
            f" target={entry.target_name}"
            f" source_shape={_shape_text(record)}"
            f" checkpoint={entry.checkpoint_name}"
        )


def _print_top(records: list[TensorMeta], *, top_n: int, pattern: str | None) -> None:
    if pattern:
        records = [
            record for record in records if fnmatch.fnmatch(record.name, pattern)
        ]
    print(f"top_tensors count={min(top_n, len(records))} pattern={pattern or '*'}")
    for record in sorted(records, key=lambda item: item.size, reverse=True)[:top_n]:
        print(
            "top_tensor"
            f" size={_format_gib(record.size)}"
            f" dtype={record.dtype}"
            f" shape={_shape_text(record)}"
            f" class={_classify_name(record.name)}"
            f" name={record.name}"
        )


def _print_uma_hints(
    architectures: list[str],
    class_counts: Counter[str],
) -> None:
    arch_text = ",".join(architectures) if architectures else "unknown"
    print(f"uma_hint architectures={arch_text}")
    if any("DeepseekV2" in arch or "DeepseekV3" in arch for arch in architectures):
        print("uma_hint direct_plan=deepseek_v2_v3 status=implemented")
    if class_counts["per_expert_moe"]:
        print(
            "uma_hint moe=per_expert"
            f" tensors={class_counts['per_expert_moe']}"
            " note=model hook should skip non-local experts before payload reads"
        )
    if class_counts["shared_experts"]:
        print(
            "uma_hint moe=shared_experts"
            f" tensors={class_counts['shared_experts']}"
            " note=DeepSeek fusion path should use source slices"
        )
    if class_counts["deepseek_indexer_wk"] or class_counts["deepseek_fp8_indexer_wk_scale"]:
        print(
            "uma_hint deepseek_indexer_wk"
            f" weights={class_counts['deepseek_indexer_wk']}"
            f" scales={class_counts['deepseek_fp8_indexer_wk_scale']}"
            " note=FP8 WK requires weight/scale pairing before fused load"
        )
    if class_counts["granite_input_linear"] or class_counts["granite_output_linear"]:
        print(
            "uma_hint granite_fused_experts"
            f" input_linear={class_counts['granite_input_linear']}"
            f" output_linear={class_counts['granite_output_linear']}"
            " note=model hook should read local expert slices only"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit local safetensors metadata for UMA O_DIRECT loading. "
            "No tensor payload bytes are read."
        )
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--metadata-limit-mib",
        type=int,
        default=DEFAULT_METADATA_LIMIT_MIB,
        help="Maximum safetensors metadata size per file.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of largest tensors to print.",
    )
    parser.add_argument(
        "--pattern",
        help="Optional fnmatch pattern for top tensor output.",
    )
    parser.add_argument(
        "--uma-plan",
        choices=("llama4",),
        help=(
            "Build a metadata-only model WeightPlan and validate generated "
            "read_segments. No tensor payload bytes are read."
        ),
    )
    parser.add_argument(
        "--plan-examples",
        type=int,
        default=DEFAULT_PLAN_EXAMPLES,
        help="Maximum UMA plan examples/errors to print.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"Model path is not a directory: {model_dir}")
    if args.metadata_limit_mib <= 0:
        raise SystemExit("--metadata-limit-mib must be positive")
    if args.top < 0:
        raise SystemExit("--top must be non-negative")
    if args.plan_examples < 0:
        raise SystemExit("--plan-examples must be non-negative")

    files = _find_safetensors(model_dir)
    catalog = TensorCatalog.from_safetensors_files(
        files,
        metadata_limit_bytes=args.metadata_limit_mib * 1024 * 1024,
    )
    records = list(catalog.records())
    architectures = _read_architectures(model_dir)
    class_counts: Counter[str] = Counter()
    class_bytes: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    file_bytes: Counter[str] = Counter()

    for record in records:
        kind = _classify_name(record.name)
        class_counts[kind] += 1
        class_bytes[kind] += record.size
        dtype = str(record.dtype)
        dtype_counts[dtype] += 1
        dtype_bytes[dtype] += record.size
        file_bytes[record.file_path] += record.size

    print(f"model_dir={model_dir}")
    if architectures:
        print(f"architectures={','.join(architectures)}")
    print(
        f"files={len(files)} tensors={len(records)} "
        f"payload={_format_gib(catalog.total_bytes())}"
    )
    for path in files:
        print(f"file path={path} payload={_format_gib(file_bytes[path])}")
    for kind, count in sorted(class_counts.items()):
        print(
            "class"
            f" name={kind}"
            f" tensors={count}"
            f" payload={_format_gib(class_bytes[kind])}"
        )
    for dtype, count in sorted(dtype_counts.items()):
        print(
            "dtype"
            f" name={dtype}"
            f" tensors={count}"
            f" payload={_format_gib(dtype_bytes[dtype])}"
        )
    _print_uma_hints(architectures, class_counts)
    if args.top:
        _print_top(records, top_n=args.top, pattern=args.pattern)
    if args.uma_plan == "llama4":
        _print_llama4_plan_audit(
            catalog,
            _read_config(model_dir),
            examples=args.plan_examples,
        )


if __name__ == "__main__":
    main()
