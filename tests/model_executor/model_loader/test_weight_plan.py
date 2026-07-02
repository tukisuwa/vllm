# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.model_loader.weight_plan import (
    TensorCatalog,
    TensorMeta,
    WeightPlan,
    WeightPlanEntry,
    build_auto_weight_plan_from_catalog,
    resolve_weight_plan_source_hooks,
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


def test_resolve_weight_plan_source_hooks_requires_complete_contract():
    class NoHooks:
        pass

    class BuilderOnly:
        def build_weight_plan(self, catalog):
            return catalog

    class CompleteHooks:
        def build_weight_plan(self, catalog):
            return catalog

        def load_weights_from_source(self, source, plan):
            return {"loaded"}

    assert resolve_weight_plan_source_hooks(NoHooks()) is None

    with pytest.raises(RuntimeError, match="must implement both"):
        resolve_weight_plan_source_hooks(BuilderOnly())

    hooks = resolve_weight_plan_source_hooks(CompleteHooks())
    assert hooks is not None
    build_weight_plan, load_weights_from_source = hooks
    assert build_weight_plan("catalog") == "catalog"
    assert load_weights_from_source("source", "plan") == {"loaded"}
