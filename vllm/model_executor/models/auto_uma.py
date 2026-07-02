# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared model-side helpers for AutoWeightsLoader-style UMA hooks."""

from collections.abc import Callable

from torch import nn

from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
    execute_weight_plan,
)
from vllm.model_executor.model_loader.weight_plan import (
    TensorCatalog,
    TransformOp,
    WeightPlan,
    build_auto_weight_plan_for_module,
)


def build_auto_uma_weight_plan(
    model: nn.Module,
    catalog: TensorCatalog,
    *,
    mapper: object | None = None,
    name_transform: (
        Callable[
            [str], tuple[str, tuple[TransformOp, ...] | None] | None
        ]
        | None
    ) = None,
    skip_prefixes: list[str] | None = None,
    skip_substrs: list[str] | None = None,
    skip_predicate: Callable[[str], bool] | None = None,
    ignore_unexpected_suffixes: list[str] | None = None,
) -> WeightPlan:
    return build_auto_weight_plan_for_module(
        model,
        catalog,
        mapper=mapper,
        name_transform=name_transform,
        skip_prefixes=skip_prefixes,
        skip_substrs=skip_substrs,
        skip_predicate=skip_predicate,
        ignore_unexpected_suffixes=ignore_unexpected_suffixes,
    )


def load_auto_uma_weights_from_source(
    model: nn.Module,
    source: ODirectSafetensorsWeightSource,
    plan: WeightPlan,
) -> set[str]:
    return execute_weight_plan(model, source, plan)
