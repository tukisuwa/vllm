# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Golden snapshots for representative UMA WeightPlan builders."""

import torch
from torch import nn

from vllm.model_executor.model_loader.weight_plan import (
    TensorCatalog,
    TensorMeta,
    TransformOp,
    WeightPlan,
    WeightPlanEntry,
    WeightPlanReadSegment,
)
from vllm.model_executor.models import bagel, llama4_uma, mistral, qwen3_moe


def _slice_json(item: slice | int):
    if isinstance(item, int):
        return item
    return [item.start, item.stop, item.step]


def _slices_json(slices: tuple[slice | int, ...] | None):
    if slices is None:
        return None
    return [_slice_json(item) for item in slices]


def _transform_ops_json(ops: tuple[TransformOp, ...]):
    return [{"op": op.op, "args": list(op.args)} for op in ops]


def _read_segments_json(segments: tuple[WeightPlanReadSegment, ...] | None):
    if segments is None:
        return None
    return [
        {
            "source_slices": _slices_json(segment.source_slices),
            "target_slices": _slices_json(segment.target_slices),
        }
        for segment in segments
    ]


def _entry_json(entry: WeightPlanEntry):
    data = {
        "checkpoint_name": entry.checkpoint_name,
        "target_name": entry.target_name,
    }
    if not entry.required:
        data["required"] = False
    if entry.source_slices is not None:
        data["source_slices"] = _slices_json(entry.source_slices)
    if entry.target_slices is not None:
        data["target_slices"] = _slices_json(entry.target_slices)
    if entry.read_segments is not None:
        data["read_segments"] = _read_segments_json(entry.read_segments)
    if entry.staging_shape is not None:
        data["staging_shape"] = list(entry.staging_shape)
    if entry.transform_ops:
        data["transform_ops"] = _transform_ops_json(entry.transform_ops)
    if entry.source_is_sharded:
        data["source_is_sharded"] = True
    if entry.read_into_cpu:
        data["read_into_cpu"] = True
    if entry.shard_id is not None:
        data["shard_id"] = entry.shard_id
    if entry.expert_id is not None:
        data["expert_id"] = entry.expert_id
    if entry.weight_name is not None:
        data["weight_name"] = entry.weight_name
    if entry.loader_target_name is not None:
        data["loader_target_name"] = entry.loader_target_name
    if entry.ignore_missing:
        data["ignore_missing"] = True
    if entry.skip_reason is not None:
        data["skip_reason"] = entry.skip_reason
    return data


def _plan_json(plan: WeightPlan):
    return [_entry_json(entry) for entry in plan.entries]


def _fused_expert_entries_json(entries):
    return [
        {
            "checkpoint_name": entry.checkpoint_name,
            "layer_id": entry.layer_id,
            "target_name": entry.target_name,
            "shard_id": entry.shard_id,
            "source_slices": _slices_json(entry.source_slices),
            "expert_id": entry.expert_id,
            "kind": entry.kind,
        }
        for entry in entries
    ]


def test_qwen3_moe_weight_plan_golden():
    names = [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
    ]
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", names[0], torch.float32, [1], 0, 4),
            TensorMeta("model.safetensors", names[1], torch.float32, [1], 4, 4),
        ]
    )

    class FakeRoutedExperts(nn.Module):
        layer_name = "model.layers.0.mlp.experts.routed_experts"
        w13_weight = object()
        quant_method = object()

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return 0 if expert_id == 0 else -1

        def weight_loader(self, **_kwargs):
            return True

    class FakeExperts(nn.Module):
        def __init__(self, routed_experts):
            super().__init__()
            self.routed_experts = routed_experts

    class FakeMLP:
        def __init__(self, routed_experts):
            self.experts = FakeExperts(routed_experts)

    class FakeLayer:
        def __init__(self, routed_experts):
            self.mlp = FakeMLP(routed_experts)

    class FakeInnerModel:
        def __init__(self, routed_experts):
            self.layers = [FakeLayer(routed_experts)]

    class FakeConfig:
        tie_word_embeddings = False

    class FakeQwen3Moe(qwen3_moe.Qwen3MoeForCausalLM):
        hf_to_vllm_mapper = None

        def __init__(self):
            nn.Module.__init__(self)
            self.config = FakeConfig()
            self.routed_experts = FakeRoutedExperts()
            self.model = FakeInnerModel(self.routed_experts)

    assert _plan_json(FakeQwen3Moe().build_weight_plan(catalog)) == [
        {
            "checkpoint_name": names[0],
            "target_name": "routed_experts.w13_weight",
            "shard_id": "w1",
            "expert_id": 0,
            "weight_name": (
                "model.layers.0.mlp.experts.routed_experts.w13_weight"
            ),
            "loader_target_name": "routed_experts",
        },
        {
            "checkpoint_name": names[1],
            "target_name": names[1],
            "required": False,
            "shard_id": "w1",
            "expert_id": 1,
            "weight_name": "w13_weight",
            "skip_reason": "non-local routed expert",
        },
    ]


def test_mistral_weight_plan_golden():
    catalog = TensorCatalog(
        [
            TensorMeta(
                "model.safetensors",
                "layers.0.attention.wq.weight",
                torch.float32,
                [4, 4],
                0,
                64,
            ),
            TensorMeta(
                "model.safetensors",
                "output.weight",
                torch.float32,
                [1],
                64,
                4,
            ),
        ]
    )

    class FakeConfig:
        tie_word_embeddings = True
        head_dim = 2
        hidden_size = 4
        num_attention_heads = 2
        num_key_value_heads = 2

    class FakeMistral:
        config = FakeConfig()
        hf_to_vllm_mapper = mistral.MistralForCausalLM.hf_to_vllm_mapper
        mistral_mapping = mistral.MistralForCausalLM.mistral_mapping
        _remap_mistral_name = mistral.MistralForCausalLM._remap_mistral_name
        _mistral_source_name_transform = (
            mistral.MistralForCausalLM._mistral_source_name_transform
        )

        def children(self):
            return []

    plan = mistral.MistralForCausalLM.build_weight_plan(FakeMistral(), catalog)

    assert _plan_json(plan) == [
        {
            "checkpoint_name": "layers.0.attention.wq.weight",
            "target_name": "model.layers.0.self_attn.qkv_proj.weight",
            "transform_ops": [{"op": "qk_rope_permute", "args": [2]}],
            "shard_id": "q",
        },
        {
            "checkpoint_name": "output.weight",
            "target_name": "lm_head.weight",
            "required": False,
        },
    ]


def test_bagel_weight_plan_golden():
    catalog = TensorCatalog(
        [
            TensorMeta(
                "model.safetensors",
                "vit_model.patch_embedding.weight",
                torch.float32,
                [2, 12],
                0,
                96,
            ),
            TensorMeta(
                "model.safetensors",
                "moe_gen.experts.0.w1.weight",
                torch.float32,
                [1],
                96,
                4,
            ),
            TensorMeta(
                "model.safetensors",
                "vit_pos_embed.pos_embed.weight",
                torch.float32,
                [1],
                100,
                4,
            ),
        ]
    )

    class FakeVitConfig:
        patch_size = 2
        num_channels = 3

    class FakeConfig:
        vit_config = FakeVitConfig()

    class FakePatchEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 3, 2, 2))

    class FakeVitModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embedding = FakePatchEmbedding()

    class FakeBagel(bagel.BagelForConditionalGeneration):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = FakeConfig()
            self.vit_model = FakeVitModel()
            self.hf_to_vllm_mapper = bagel.WeightsMapper(
                orig_to_new_prefix={"vit_model.": "vit_model."}
            )

    assert _plan_json(FakeBagel().build_weight_plan(catalog)) == [
        {
            "checkpoint_name": "vit_model.patch_embedding.weight",
            "target_name": "vit_model.patch_embedding.weight",
            "transform_ops": [
                {"op": "patch_embedding_reshape", "args": [2, 3]}
            ],
        },
        {
            "checkpoint_name": "moe_gen.experts.0.w1.weight",
            "target_name": "moe_gen.experts.0.w1.weight",
            "required": False,
        },
        {
            "checkpoint_name": "vit_pos_embed.pos_embed.weight",
            "target_name": "vit_pos_embed.pos_embed.weight",
            "required": False,
        },
    ]


def test_llama4_weight_plan_golden(monkeypatch):
    local_name = "model.layers.0.feed_forward.experts.1.gate_proj.weight"
    remote_name = "model.layers.0.feed_forward.experts.3.down_proj.weight"
    fused_gate_up = "model.layers.0.feed_forward.experts.gate_up_proj.weight"
    q_name = "model.layers.0.self_attn.q_proj.weight"
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", local_name, torch.float32, [1], 0, 4),
            TensorMeta("model.safetensors", remote_name, torch.float32, [1], 4, 4),
            TensorMeta(
                "model.safetensors",
                fused_gate_up,
                torch.float32,
                [4, 2, 6],
                8,
                192,
            ),
            TensorMeta("model.safetensors", q_name, torch.float32, [4, 1], 200, 16),
        ]
    )

    class FakeConfig:
        tie_word_embeddings = False
        num_attention_heads = 2
        num_key_value_heads = 1

    class FakeRoutedExperts(nn.Module):
        layer_name = "model.layers.0.feed_forward.experts"
        w13_weight = object()
        w2_weight = object()
        quant_method = object()
        expert_map = torch.tensor([-1, 0, 1, -1])

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return 0 if expert_id == 1 else -1

        def weight_loader(self, **_kwargs):
            return True

    class FakeMoE:
        def __init__(self, experts):
            self.experts = experts

    monkeypatch.setattr(llama4_uma, "Llama4MoE", FakeMoE)

    class FakeLayer:
        def __init__(self, experts):
            self.feed_forward = FakeMoE(experts)

    class FakeLayers:
        def __init__(self, layer):
            self._layers = [layer]

        def __len__(self):
            return len(self._layers)

        def __getitem__(self, idx):
            return self._layers[idx]

    class FakeInnerModel:
        def __init__(self, experts):
            self.layers = FakeLayers(FakeLayer(experts))

    class FakeOuter(nn.Module):
        config = FakeConfig()

        def __init__(self):
            super().__init__()
            self.routed_experts = FakeRoutedExperts()
            self.model = FakeInnerModel(self.routed_experts)

        def children(self):
            return []

    plan = llama4_uma.build_llama4_weight_plan(FakeOuter(), catalog)

    assert {
        "weight_plan": _plan_json(plan.weight_plan),
        "fused_expert_entries": _fused_expert_entries_json(
            plan.fused_expert_entries
        ),
    } == {
        "weight_plan": [
            {
                "checkpoint_name": q_name,
                "target_name": "model.layers.0.self_attn.qkv_proj.weight",
                "transform_ops": [{"op": "qk_rope_permute", "args": [2]}],
                "shard_id": "q",
            },
            {
                "checkpoint_name": local_name,
                "target_name": "routed_experts.w13_weight",
                "shard_id": "w1",
                "expert_id": 1,
                "weight_name": "model.layers.0.feed_forward.experts.w13_weight",
                "loader_target_name": "routed_experts",
            },
            {
                "checkpoint_name": remote_name,
                "target_name": remote_name,
                "required": False,
                "shard_id": "w2",
                "expert_id": 3,
                "weight_name": "w2_weight",
                "skip_reason": "non-local routed expert",
            },
        ],
        "fused_expert_entries": [
            {
                "checkpoint_name": fused_gate_up,
                "layer_id": 0,
                "target_name": "model.layers.0.feed_forward.experts.w13_weight",
                "shard_id": "w1",
                "source_slices": [[1, 3, None], [None, None, None], [None, None, None]],
                "expert_id": 1,
                "kind": "gate_up",
            },
            {
                "checkpoint_name": fused_gate_up,
                "layer_id": 0,
                "target_name": "model.layers.0.feed_forward.experts.w13_weight",
                "shard_id": "w3",
                "source_slices": [[1, 3, None], [None, None, None], [None, None, None]],
                "expert_id": 1,
                "kind": "gate_up",
            },
        ],
    }
