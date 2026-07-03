# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import types

import pytest
import torch

from vllm.model_executor.model_loader.weight_plan import (
    ExecutorCapability,
    TensorCatalog,
    TensorMeta,
    TransformOp,
    WeightPlan,
    WeightPlanEntry,
    WeightPlanReadSegment,
    apply_transform_ops,
    build_auto_weight_plan_from_catalog,
    register_weight_transform,
    resolve_weight_plan,
    resolve_weight_plan_source_hooks,
    schedule_weight_plan_reads,
    summarize_weight_plan,
    transform_ops_extra_staging_factor,
    verify_loaded_weights,
)


def test_tensor_meta_shape_is_immutable_tuple():
    meta = TensorMeta("model.safetensors", "w", torch.float32, [2, 3], 0, 24)

    assert meta.shape == (2, 3)
    assert isinstance(meta.shape, tuple)


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


def test_resolve_weight_plan_fails_closed_on_shard_size_mapping_error():
    class Owner:
        def _get_shard_size_mapping(self, _shard_id):
            raise RuntimeError("mapping changed")

        def weight_loader(self, *_args, **_kwargs):
            pass

    param = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    param.output_dim = 0
    param.tp_rank = 1
    param.tp_size = 2
    param.weight_loader = Owner().weight_loader
    model = types.SimpleNamespace(w=param)
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", "w", torch.float32, [8, 4], 0, 128),
        ]
    )
    plan = WeightPlan((WeightPlanEntry("w", "w", shard_id="q"),))

    with pytest.raises(RuntimeError, match="refusing to silently fall back"):
        resolve_weight_plan(model, catalog, plan)


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


def test_executor_capability_aligned_direct_io_defaults_fail_closed():
    capability = ExecutorCapability.for_aligned_direct_io(max_staging_bytes=1024)

    assert capability.supports_partial_read is True
    assert capability.supports_strided_read is True
    assert capability.requires_alignment is True
    assert capability.allows_mmap is False
    assert capability.supports_full_tensor_fallback is False
    assert capability.fail_closed is True
    assert capability.max_staging_bytes == 1024


def test_schedule_weight_plan_reads_orders_required_entries_by_file_offset():
    catalog = TensorCatalog(
        [
            TensorMeta("f", "a", torch.uint8, [4], 0, 4),
            TensorMeta("f", "b", torch.uint8, [4], 4096, 4),
            TensorMeta("f", "c", torch.uint8, [4], 8192, 4),
            TensorMeta("f", "pad", torch.uint8, [4096], 12288, 4096),
        ]
    )
    plan = WeightPlan(
        (
            WeightPlanEntry("c", "c"),
            WeightPlanEntry("a", "a"),
            WeightPlanEntry("skip", "skip", required=False),
            WeightPlanEntry("b", "b"),
        )
    )

    schedule = schedule_weight_plan_reads(
        catalog,
        plan,
        chunk_size=4096,
        window_size=8192,
        alignment=4096,
    )

    assert [entry.checkpoint_name for entry in schedule.plan.entries] == [
        "a",
        "b",
        "c",
        "skip",
    ]
    assert schedule.summary.required_entries == 3
    assert schedule.summary.read_ranges == 3
    assert schedule.summary.expected_window_loads == 2
    assert schedule.summary.expected_window_hits == 3
    assert schedule.summary.expected_bytes_read == 16 * 1024
    assert schedule.summary.payload_bytes == 12


def test_schedule_weight_plan_reads_estimates_large_direct_reads():
    catalog = TensorCatalog(
        [
            TensorMeta("f", "large", torch.uint8, [12288], 0, 12288),
        ]
    )
    plan = WeightPlan((WeightPlanEntry("large", "large"),))

    schedule = schedule_weight_plan_reads(
        catalog,
        plan,
        chunk_size=4096,
        window_size=4096,
        alignment=4096,
    )

    assert schedule.summary.read_ranges == 1
    assert schedule.summary.expected_direct_reads == 3
    assert schedule.summary.expected_window_loads == 0
    assert schedule.summary.expected_bytes_read == 12288
    assert schedule.summary.read_amplification == 1.0


def test_schedule_weight_plan_reads_estimates_strided_ranges_without_expansion():
    catalog = TensorCatalog(
        [
            TensorMeta("f", "cols", torch.uint8, [4, 4], 0, 16),
        ]
    )
    plan = WeightPlan(
        (
            WeightPlanEntry(
                "cols",
                "cols",
                source_slices=(slice(None), slice(0, 1)),
            ),
        )
    )

    schedule = schedule_weight_plan_reads(
        catalog,
        plan,
        chunk_size=4,
        window_size=8,
        alignment=1,
    )

    assert schedule.summary.read_ranges == 4
    assert schedule.summary.payload_bytes == 4
    assert schedule.summary.expected_window_loads == 2
    assert schedule.summary.expected_window_hits == 4


def test_schedule_weight_plan_reads_coalesces_segment_ranges_across_entries():
    row_bytes = 4096
    catalog = TensorCatalog(
        [
            TensorMeta(
                "model.safetensors",
                "qkv",
                torch.float32,
                [8, row_bytes // 4],
                0,
                8 * row_bytes,
            ),
        ]
    )
    plan = WeightPlan(
        (
            WeightPlanEntry(
                "qkv",
                "qkv_proj",
                read_into_cpu=True,
                staging_shape=(2, row_bytes // 4),
                read_segments=(
                    WeightPlanReadSegment(
                        (slice(0, 1), slice(None)),
                        (slice(0, 1), slice(None)),
                    ),
                    WeightPlanReadSegment(
                        (slice(4, 5), slice(None)),
                        (slice(1, 2), slice(None)),
                    ),
                ),
                shard_id="q",
            ),
            WeightPlanEntry(
                "qkv",
                "qkv_proj",
                read_into_cpu=True,
                staging_shape=(2, row_bytes // 4),
                read_segments=(
                    WeightPlanReadSegment(
                        (slice(2, 3), slice(None)),
                        (slice(0, 1), slice(None)),
                    ),
                    WeightPlanReadSegment(
                        (slice(6, 7), slice(None)),
                        (slice(1, 2), slice(None)),
                    ),
                ),
                shard_id="k",
            ),
            WeightPlanEntry(
                "qkv",
                "qkv_proj",
                read_into_cpu=True,
                staging_shape=(2, row_bytes // 4),
                read_segments=(
                    WeightPlanReadSegment(
                        (slice(3, 4), slice(None)),
                        (slice(0, 1), slice(None)),
                    ),
                    WeightPlanReadSegment(
                        (slice(7, 8), slice(None)),
                        (slice(1, 2), slice(None)),
                    ),
                ),
                shard_id="v",
            ),
        )
    )

    schedule = schedule_weight_plan_reads(
        catalog,
        plan,
        chunk_size=row_bytes,
        window_size=row_bytes * 2,
        alignment=4096,
    )

    assert schedule.summary.expected_window_loads == 4
    assert schedule.summary.expected_window_hits == 6


def test_register_weight_transform_is_idempotent_and_fails_on_conflict():
    def op_a(tensor):
        return tensor

    register_weight_transform("test_op_a", op_a, extra_staging_factor=0.0)
    register_weight_transform("test_op_a", op_a, extra_staging_factor=0.0)

    def op_b(tensor):
        return tensor

    with pytest.raises(RuntimeError, match="already registered"):
        register_weight_transform("test_op_a", op_b, extra_staging_factor=0.0)


def test_apply_transform_ops_chains_in_order_with_args():
    tensor = torch.ones(1, 4)

    result = apply_transform_ops(
        (
            TransformOp("squeeze", (0,)),
            TransformOp("zero_mean"),
        ),
        tensor,
    )

    assert result.shape == (4,)
    assert torch.allclose(result, torch.zeros(4))


def test_builtin_transform_ops_match_family_semantics():
    tensor = torch.tensor([[3.0, 4.0]])

    normalized = apply_transform_ops((TransformOp("l2_normalize", (1,)),), tensor)
    assert torch.allclose(normalized, torch.tensor([[0.6, 0.8]]))

    empty = apply_transform_ops((TransformOp("zero_mean"),), torch.empty(0))
    assert empty.numel() == 0

    qk = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    assert torch.equal(
        apply_transform_ops((TransformOp("qk_rope_permute", (2,)),), qk),
        qk.view(2, 1, 2, 4).transpose(1, 2).reshape(4, 4),
    )

    qk_scale = torch.arange(4, dtype=torch.float32)
    assert torch.equal(
        apply_transform_ops((TransformOp("qk_rope_permute", (2,)),), qk_scale),
        qk_scale.unsqueeze(-1)
        .view(2, 1, 2, 1)
        .transpose(1, 2)
        .reshape(4, 1)
        .squeeze(-1),
    )
    assert torch.equal(
        apply_transform_ops((TransformOp("qk_rope_permute_2d", (2,)),), qk_scale),
        qk_scale.view(2, 1, 2, 1).transpose(1, 2).reshape(4, 1),
    )

    patch = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    patch_reshaped = apply_transform_ops(
        (TransformOp("patch_embedding_reshape", (2, 3)),),
        patch,
    )
    assert patch_reshaped.shape == (2, 3, 2, 2)
    assert patch_reshaped[0, :, 0, 0].tolist() == [0.0, 1.0, 2.0]


def test_summarize_weight_plan_fails_closed_on_unknown_transform_op():
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", "w", torch.float32, [4], 0, 16),
        ]
    )
    plan = WeightPlan(
        (
            WeightPlanEntry(
                "w",
                "w",
                transform_ops=(TransformOp("does_not_exist"),),
            ),
        )
    )

    with pytest.raises(RuntimeError, match="Unknown weight transform op"):
        summarize_weight_plan(catalog, plan)


def test_summarize_weight_plan_accounts_transform_staging():
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", "big", torch.float32, [8], 0, 32),
            TensorMeta("model.safetensors", "small", torch.float32, [2], 32, 8),
        ]
    )
    plan = WeightPlan(
        (
            WeightPlanEntry(
                "big",
                "big",
                transform_ops=(TransformOp("zero_mean"),),
            ),
            WeightPlanEntry(
                "small",
                "small",
                transform_ops=(
                    TransformOp("zero_mean"),
                    TransformOp("l2_normalize"),
                ),
            ),
        )
    )

    summary = summarize_weight_plan(catalog, plan)

    # big: 32 bytes x factor 1.0; small: 8 bytes x factor 2.0 -> peak is big.
    assert summary.peak_transform_staging_bytes == 32
    assert transform_ops_extra_staging_factor(
        (TransformOp("zero_mean"), TransformOp("l2_normalize"))
    ) == 2.0


def test_summarize_weight_plan_ignores_transform_ops_on_skipped_entries():
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", "w", torch.float32, [4], 0, 16),
        ]
    )
    plan = WeightPlan(
        (
            WeightPlanEntry(
                "w",
                "w",
                required=False,
                transform_ops=(TransformOp("does_not_exist"),),
            ),
        )
    )

    summary = summarize_weight_plan(catalog, plan)
    assert summary.peak_transform_staging_bytes == 0
