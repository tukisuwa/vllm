# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for folding routed MoE entries into a single executable WeightPlan."""

import types

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    execute_weight_plan,
)
from vllm.model_executor.model_loader.weight_plan import (
    TensorCatalog,
    TensorMeta,
    WeightPlan,
)
from vllm.model_executor.models.routed_moe_uma import (
    RoutedMoeEntry,
    RoutedMoeSourcePlan,
    routed_moe_source_plan_to_weight_plan,
)


class _Experts(nn.Module):
    def __init__(self, *, accept: bool = True):
        super().__init__()
        self.layer_name = "experts"
        self.w13_weight = nn.Parameter(torch.zeros(2, 4), requires_grad=False)
        self.w13_weight.weight_loader = self.weight_loader
        self._accept = accept
        self.seen: list[tuple[str, str, int]] = []

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        self.seen.append((weight_name, shard_id, expert_id))
        if not self._accept:
            return False if return_success else None
        param.data.copy_(loaded_weight)
        return True if return_success else None


class _Model(nn.Module):
    def __init__(self, *, accept: bool = True):
        super().__init__()
        self.experts = _Experts(accept=accept)


class _FakeSource:
    def __init__(self, catalog, tensors):
        self.catalog = catalog
        self._tensors = tensors
        self.skipped: list[tuple[str, str]] = []

    def read_full_cpu(self, name):
        return self._tensors[name]

    def read_slice_cpu(self, name, source_slices):
        return self._tensors[name][source_slices]

    def skip(self, name, reason):
        self.skipped.append((name, reason))


def _catalog():
    return TensorCatalog(
        [
            TensorMeta(
                "f", "layers.0.experts.0.gate.weight", torch.float32, [2, 4], 0, 32
            ),
            TensorMeta(
                "f", "layers.0.experts.1.gate.weight", torch.float32, [2, 4], 32, 32
            ),
        ]
    )


def _routed_entries():
    return (
        RoutedMoeEntry(
            checkpoint_name="layers.0.experts.0.gate.weight",
            layer_id=0,
            expert_id=0,
            param_name="w13_weight",
            shard_id="w1",
            local_required=True,
        ),
        RoutedMoeEntry(
            checkpoint_name="layers.0.experts.1.gate.weight",
            layer_id=0,
            expert_id=1,
            param_name="w13_weight",
            shard_id="w1",
            local_required=False,
            skip_reason="non-local expert 1",
        ),
    )


def test_routed_plan_folds_into_single_weight_plan_and_executes():
    model = _Model()
    source_plan = RoutedMoeSourcePlan(WeightPlan(()), _routed_entries())

    weight_plan = routed_moe_source_plan_to_weight_plan(
        model,
        source_plan,
        family_name="Test",
        get_routed_experts=lambda m, layer_id: m.experts,
    )

    assert [entry.required for entry in weight_plan.entries] == [True, False]
    assert weight_plan.entries[0].target_name == "experts.w13_weight"
    assert weight_plan.entries[0].weight_name == "experts.w13_weight"
    assert weight_plan.entries[1].skip_reason == "non-local expert 1"

    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    source = _FakeSource(_catalog(), {"layers.0.experts.0.gate.weight": tensor})

    loaded = execute_weight_plan(model, source, weight_plan)

    assert loaded == {"experts.w13_weight"}
    assert source.skipped == [
        ("layers.0.experts.1.gate.weight", "non-local expert 1")
    ]
    assert model.experts.seen == [("experts.w13_weight", "w1", 0)]
    assert torch.equal(model.experts.w13_weight.data, tensor)


def test_refused_local_expert_fails_closed():
    model = _Model(accept=False)
    source_plan = RoutedMoeSourcePlan(WeightPlan(()), _routed_entries())
    weight_plan = routed_moe_source_plan_to_weight_plan(
        model,
        source_plan,
        family_name="Test",
        get_routed_experts=lambda m, layer_id: m.experts,
    )
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    source = _FakeSource(_catalog(), {"layers.0.experts.0.gate.weight": tensor})

    with pytest.raises(RuntimeError, match="refused a tensor"):
        execute_weight_plan(model, source, weight_plan)


def test_unregistered_routed_parameter_fails_closed():
    model = _Model()
    rogue = types.SimpleNamespace(
        layer_name="experts",
        w13_weight=nn.Parameter(torch.zeros(2, 4), requires_grad=False),
    )

    source_plan = RoutedMoeSourcePlan(WeightPlan(()), _routed_entries()[:1])
    with pytest.raises(RuntimeError, match="not a registered model parameter"):
        routed_moe_source_plan_to_weight_plan(
            model,
            source_plan,
            family_name="Test",
            get_routed_experts=lambda m, layer_id: rogue,
        )
