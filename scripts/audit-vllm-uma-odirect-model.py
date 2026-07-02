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
import os
from collections import Counter
from pathlib import Path

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    TensorCatalog,
    TensorMeta,
)


DEFAULT_METADATA_LIMIT_MIB = 16
DEFAULT_TOP_N = 20


def _format_gib(nbytes: int) -> str:
    return f"{nbytes / (1024**3):.3f}GiB"


def _find_safetensors(model_dir: Path) -> list[str]:
    files = sorted(str(path) for path in model_dir.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"No .safetensors files found in {model_dir}")
    for path in files:
        if os.path.islink(path):
            raise SystemExit(f"Refusing symlinked weight file: {path}")
    return files


def _read_architectures(model_dir: Path) -> list[str]:
    config_path = model_dir / "config.json"
    if not config_path.exists() or os.path.islink(config_path):
        return []
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    architectures = data.get("architectures")
    if not isinstance(architectures, list):
        return []
    return [item for item in architectures if isinstance(item, str)]


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
    return "x".join(str(dim) for dim in record.shape)


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
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"Model path is not a directory: {model_dir}")
    if args.metadata_limit_mib <= 0:
        raise SystemExit("--metadata-limit-mib must be positive")
    if args.top < 0:
        raise SystemExit("--top must be non-negative")

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


if __name__ == "__main__":
    main()
