# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import pytest
import torch

from vllm.model_executor.model_loader.weight_plan import (
    ExecutorCapability,
    TensorCatalog,
    TensorMeta,
    WeightPlan,
    WeightPlanEntry,
    build_auto_weight_plan_from_catalog,
    resolve_weight_plan,
    resolve_weight_plan_source_hooks,
    summarize_weight_plan,
    verify_loaded_weights,
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


def test_resolve_weight_plan_fills_tp_source_slices():
    param = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    param.output_dim = 0
    param.tp_rank = 1
    param.tp_size = 2
    model = types.SimpleNamespace(w=param)
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", "w", torch.float32, [8, 4], 0, 128),
        ]
    )
    plan = WeightPlan((WeightPlanEntry("w", "w"),))

    resolved = resolve_weight_plan(model, catalog, plan)

    entry = resolved.entries[0]
    assert entry.source_slices == (slice(4, 8), slice(None, None))
    assert entry.source_is_sharded is True

    summary = summarize_weight_plan(catalog, resolved)
    assert summary.sliced_payload_bytes == 64
    assert summary.full_payload_bytes == 0


def test_resolve_weight_plan_leaves_expert_and_explicit_entries_untouched():
    param = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    param.output_dim = 0
    param.tp_rank = 1
    param.tp_size = 2
    model = types.SimpleNamespace(w=param)
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", "w", torch.float32, [8, 4], 0, 128),
        ]
    )
    expert_entry = WeightPlanEntry("w", "w", expert_id=3)
    explicit_entry = WeightPlanEntry(
        "w",
        "w",
        source_slices=(slice(0, 4), slice(None)),
    )
    plan = WeightPlan((expert_entry, explicit_entry))

    resolved = resolve_weight_plan(model, catalog, plan)

    assert resolved.entries[0] is expert_entry
    assert resolved.entries[1] is explicit_entry


def test_verify_loaded_weights_fails_closed_on_missing_parameters():
    model = torch.nn.Linear(2, 2, bias=True)

    with pytest.raises(RuntimeError, match="bias"):
        verify_loaded_weights(model, {"weight"})

    verify_loaded_weights(model, {"weight", "bias"})


def test_verify_loaded_weights_exempts_postprocess_quant_modules():
    model = torch.nn.Linear(2, 2, bias=True)
    model.quant_method = types.SimpleNamespace(
        process_weights_after_loading=lambda module: None,
    )

    verify_loaded_weights(model, set())


def test_executor_capability_uma_odirect_defaults_fail_closed():
    capability = ExecutorCapability.uma_odirect(max_staging_bytes=1024)

    assert capability.supports_partial_read is True
    assert capability.supports_strided_read is True
    assert capability.requires_alignment is True
    assert capability.allows_mmap is False
    assert capability.supports_full_tensor_fallback is False
    assert capability.fail_closed is True
    assert capability.max_staging_bytes == 1024
