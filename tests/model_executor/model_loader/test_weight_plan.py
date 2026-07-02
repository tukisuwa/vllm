# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.model_loader.weight_plan import (
    TensorCatalog,
    TensorMeta,
    WeightPlan,
    WeightPlanEntry,
    build_auto_weight_plan_from_catalog,
    summarize_weight_plan,
)


def test_weight_plan_summary_without_odirect_loader():
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", "full", torch.float32, [4], 0, 16),
            TensorMeta("model.safetensors", "rows", torch.float32, [4, 2], 16, 32),
            TensorMeta("model.safetensors", "skip", torch.float32, [2], 48, 8),
        ]
    )
    plan = WeightPlan(
        (
            WeightPlanEntry("full", "full_param"),
            WeightPlanEntry(
                "rows",
                "row_param",
                source_slices=(slice(1, 3), slice(None)),
            ),
            WeightPlanEntry("skip", "unused", required=False),
        )
    )

    summary = summarize_weight_plan(catalog, plan)

    assert summary.entries == 3
    assert summary.required_entries == 2
    assert summary.skipped_entries == 1
    assert summary.full_payload_bytes == 16
    assert summary.sliced_payload_bytes == 16
    assert summary.skipped_payload_bytes == 8
    assert summary.total_read_payload_bytes == 32


def test_auto_weight_plan_from_catalog_without_odirect_loader():
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", "keep.weight", torch.float32, [1], 0, 4),
            TensorMeta("model.safetensors", "skip.weight", torch.float32, [1], 4, 4),
        ]
    )

    plan = build_auto_weight_plan_from_catalog(
        catalog,
        skip_prefixes=["skip."],
    )

    assert plan.entries == (
        WeightPlanEntry("keep.weight", "keep.weight"),
        WeightPlanEntry("skip.weight", "skip.weight", required=False),
    )
