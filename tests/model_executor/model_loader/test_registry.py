# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader, register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.uma_safetensors_loader import (
    UmaSafetensorsModelLoader,
)
from vllm.model_executor.model_loader.uma_odirect_safetensors_loader import (
    ODirectSafetensorsWeightSource,
    TensorCatalog,
    TensorMeta,
    UmaODirectSafetensorsModelLoader,
    WeightPlan,
    WeightPlanEntry,
    build_auto_weight_plan_from_catalog,
    execute_weight_plan,
)
from vllm.model_executor.models import (
    commandr,
    cohere2_moe,
    bloom,
    deepseek_v2,
    exaone,
    falcon,
    gemma,
    gemma2,
    gemma3,
    gpt_bigcode,
    granitemoe,
    granitemoehybrid,
    granitemoeshared,
    internlm2,
    llama,
    mixtral,
    mistral,
    nemotron,
    olmo,
    olmo2,
    opt,
    olmoe,
    phimoe,
    phi,
    qwen2,
    qwen2_moe,
    qwen3,
    qwen3_5,
    qwen3_moe,
    qwen3_next,
    starcoder2,
)
from vllm.model_executor.models.utils import PPMissingLayer, WeightsMapper


@register_model_loader("custom_load_format")
class CustomModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass


def _write_safetensors(path, metadata: dict, payload: bytes) -> None:
    metadata_raw = json.dumps(metadata).encode("utf-8")
    path.write_bytes(len(metadata_raw).to_bytes(8, "little") + metadata_raw + payload)


def test_register_model_loader():
    load_config = LoadConfig(load_format="custom_load_format")
    assert isinstance(get_model_loader(load_config), CustomModelLoader)


def test_invalid_model_loader():
    with pytest.raises(ValueError):

        @register_model_loader("invalid_load_format")
        class InValidModelLoader:
            pass


def test_default_loader_rejects_zero_num_threads():
    # num_threads=0 used to fail late in ThreadPoolExecutor ("max_workers must be > 0").
    with pytest.raises(ValueError, match="num_threads"):
        DefaultModelLoader(
            LoadConfig(
                model_loader_extra_config={
                    "enable_multithread_load": True,
                    "num_threads": 0,
                }
            )
        )


def test_default_loader_rejects_multithread_with_non_lazy_strategy():
    # The multi-thread loader ignores safetensors_load_strategy; reject the
    # combination instead of silently dropping the requested strategy.
    with pytest.raises(ValueError, match="does not support"):
        DefaultModelLoader(
            LoadConfig(
                safetensors_load_strategy="torchao",
                model_loader_extra_config={"enable_multithread_load": True},
            )
        )


def test_default_loader_explicit_safetensors_does_not_misread_pt(tmp_path):
    # Explicit safetensors must not fall back to a .pt and open it as safetensors.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        loader._prepare_weights(
            str(tmp_path),
            None,
            None,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )


def test_default_loader_hf_still_falls_back_to_pt(tmp_path):
    # Control: load_format="hf" still picks up .pt weights via fallback.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="hf"))
    _, files, use_safetensors = loader._prepare_weights(
        str(tmp_path),
        None,
        None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )
    assert use_safetensors is False
    assert any(f.endswith("model.pt") for f in files)


def test_uma_safetensors_registered():
    load_config = LoadConfig(load_format="uma_safetensors")
    assert isinstance(get_model_loader(load_config), UmaSafetensorsModelLoader)


def test_uma_odirect_safetensors_registered():
    load_config = LoadConfig(load_format="uma_odirect_safetensors")
    assert isinstance(get_model_loader(load_config), UmaODirectSafetensorsModelLoader)


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"typo_key": 1}, "Unexpected extra config"),
        ({"distributed": "yes"}, "distributed must be a bool"),
        ({"allow_hf_download": "yes"}, "allow_hf_download must be a bool"),
        ({"concurrency": 0}, "concurrency must be a positive integer"),
        ({"memory_limit": -1}, "memory_limit must be a positive integer"),
        ({"min_available_gib": -1}, "min_available_gib must be a non-negative"),
        ({"psi_gate_seconds": -1}, "psi_gate_seconds must be a non-negative"),
        ({"max_swap_gib": -1}, "max_swap_gib must be a non-negative"),
    ],
)
def test_uma_safetensors_rejects_invalid_extra_config(extra, match):
    with pytest.raises(ValueError, match=match):
        UmaSafetensorsModelLoader(
            LoadConfig(
                load_format="uma_safetensors",
                model_loader_extra_config=extra,
            )
        )


def test_uma_safetensors_rejects_eager_strategy():
    with pytest.raises(ValueError, match="does not support"):
        UmaSafetensorsModelLoader(
            LoadConfig(
                load_format="uma_safetensors",
                safetensors_load_strategy="eager",
            )
        )


def test_uma_safetensors_rejects_implicit_hf_download():
    loader = UmaSafetensorsModelLoader(LoadConfig(load_format="uma_safetensors"))
    with pytest.raises(RuntimeError, match="refuses implicit Hugging Face"):
        loader._prepare_weights("org/model", None)


def test_uma_safetensors_prepares_local_safetensors(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\x00\x00\x00\x00")
    loader = UmaSafetensorsModelLoader(LoadConfig(load_format="uma_safetensors"))
    assert loader._prepare_weights(str(tmp_path), None) == [
        str(tmp_path / "model.safetensors")
    ]


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"typo_key": 1}, "Unexpected extra config"),
        ({"alignment": 0}, "alignment must be a positive integer"),
        ({"chunk_size": 0}, "chunk_size must be a positive integer"),
        ({"window_size": 0}, "window_size must be a positive integer"),
        ({"alignment": 3000}, "alignment must be a power of two"),
        (
            {"chunk_size": 8 * 1024 * 1024, "window_size": 4 * 1024 * 1024},
            "window_size must be greater than or equal to chunk_size",
        ),
        (
            {"chunk_size": 8 * 1024 * 1024, "gate_interval_mib": 4},
            "chunk_size must be less than or equal to gate_interval_mib",
        ),
        (
            {
                "chunk_size": 4 * 1024 * 1024,
                "window_size": 8 * 1024 * 1024,
                "gate_interval_mib": 4,
            },
            "window_size must be less than or equal to gate_interval_mib",
        ),
        ({"min_available_gib": -1}, "min_available_gib must be a non-negative"),
        ({"psi_gate_seconds": -1}, "psi_gate_seconds must be a non-negative"),
        ({"max_swap_gib": -1}, "max_swap_gib must be a non-negative"),
        (
            {"allocation_gate_min_mib": -1},
            "allocation_gate_min_mib must be a non-negative",
        ),
        ({"direct_per_expert_moe": True}, "Unexpected extra config"),
        ({"direct_qwen35_moe": True}, "Unexpected extra config"),
    ],
)
def test_uma_odirect_safetensors_rejects_invalid_extra_config(extra, match):
    with pytest.raises(ValueError, match=match):
        UmaODirectSafetensorsModelLoader(
            LoadConfig(
                load_format="uma_odirect_safetensors",
                model_loader_extra_config=extra,
            )
        )


def test_uma_odirect_safetensors_rejects_eager_strategy():
    with pytest.raises(ValueError, match="does not support"):
        UmaODirectSafetensorsModelLoader(
            LoadConfig(
                load_format="uma_odirect_safetensors",
                safetensors_load_strategy="eager",
            )
        )


def test_uma_odirect_safetensors_rejects_implicit_hf_download():
    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    with pytest.raises(RuntimeError, match="only accepts a local"):
        loader._prepare_files("org/model")


def test_uma_odirect_safetensors_rejects_large_metadata(tmp_path):
    path = tmp_path / "model.safetensors"
    metadata_size = 2 * 1024 * 1024
    with open(path, "wb") as f:
        f.write(metadata_size.to_bytes(8, "little"))
        f.truncate(8 + metadata_size)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(
            load_format="uma_odirect_safetensors",
            model_loader_extra_config={"metadata_limit_mib": 1},
        )
    )
    with pytest.raises(RuntimeError, match="metadata too large"):
        loader._read_records([str(path)])


def test_uma_odirect_safetensors_rejects_overlapping_ranges(tmp_path):
    metadata = {
        "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "b": {"dtype": "F32", "shape": [1], "data_offsets": [2, 6]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 6)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    with pytest.raises(RuntimeError, match="Overlapping safetensors"):
        loader._read_records([str(path)])


def test_uma_odirect_safetensors_rejects_duplicate_tensor_names(tmp_path):
    metadata = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    path_a = tmp_path / "a.safetensors"
    path_b = tmp_path / "b.safetensors"
    _write_safetensors(path_a, metadata, b"\0" * 4)
    _write_safetensors(path_b, metadata, b"\0" * 4)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    with pytest.raises(RuntimeError, match="Duplicate safetensors tensor name"):
        loader._read_records([str(path_a), str(path_b)])


def test_uma_odirect_safetensors_rejects_symlink(tmp_path):
    target = tmp_path / "target.safetensors"
    target.write_bytes(b"\0" * 8)
    link = tmp_path / "model.safetensors"
    link.symlink_to(target)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    with pytest.raises(RuntimeError, match="Refusing symlinked"):
        loader._prepare_files(str(tmp_path))


@pytest.mark.parametrize(
    "metadata, match",
    [
        ({"a": {"shape": [1], "data_offsets": [0, 4]}}, "Missing safetensors"),
        ({"a": {"dtype": "F32", "shape": [1]}}, "Missing safetensors"),
        ({"a": {"dtype": "F32", "data_offsets": [0, 4]}}, "Missing safetensors"),
        (
            {"a": {"dtype": "F32", "shape": [1], "data_offsets": "0,4"}},
            "Invalid safetensors data_offsets",
        ),
        (
            {"a": {"dtype": "F32", "shape": "1", "data_offsets": [0, 4]}},
            "Invalid safetensors shape",
        ),
        (
            {"a": {"dtype": "F32", "shape": [1], "data_offsets": [-1, 4]}},
            "Invalid safetensors byte range",
        ),
        (
            {"a": {"dtype": "F32", "shape": [1], "data_offsets": [4, 0]}},
            "Invalid safetensors byte range",
        ),
    ],
)
def test_uma_odirect_safetensors_rejects_malformed_metadata(
    tmp_path, metadata, match
):
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 4)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    with pytest.raises(RuntimeError, match=match):
        loader._read_records([str(path)])


def test_uma_odirect_safetensors_allows_end_at_file_boundary(tmp_path):
    metadata = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 4)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    records = loader._read_records([str(path)])
    assert len(records) == 1
    assert records[0].name == "a"


def test_uma_odirect_tensor_catalog_lookup(tmp_path):
    metadata = {
        "b": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 8)

    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    assert catalog.names() == ("a", "b")
    assert catalog.has("a")
    assert not catalog.has("missing")
    assert catalog.get("a").offset < catalog.get("b").offset
    assert catalog.total_bytes() == 8


def test_uma_odirect_weight_source_builds_catalog_without_payload_read(
    tmp_path, monkeypatch
):
    metadata = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 4)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))

    assert source.files == [str(path)]
    assert source.catalog.names() == ("a",)
    assert source.catalog.get("a").size == 4


def test_uma_odirect_weight_source_read_full_cpu(tmp_path, monkeypatch):
    metadata = {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 8)

    class FakeODirectFile:
        window_size = 64 * 1024 * 1024

        def __init__(self, *_args):
            self.direct_reads = 0
            self.window_loads = 0
            self.window_hits = 0
            self.bytes_read = 0
            self.bytes_copied = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            pass

        def read_record_into_tensor(self, tensor, _offset, size, gate=None):
            self.direct_reads += 1
            self.bytes_read += size
            self.bytes_copied += size
            tensor.fill_(3)
            if gate is not None:
                gate(size)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.uma_odirect_safetensors_loader."
        "_ODirectFile",
        FakeODirectFile,
    )

    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))
    tensor = source.read_full_cpu("a")

    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float32
    assert tensor.tolist() == [3.0, 3.0]
    stats = source.stats_snapshot()
    assert stats["files_opened"] == 1
    assert stats["tensors_read"] == 1
    assert stats["tensors_read_full"] == 1
    assert stats["tensors_read_sliced"] == 0
    assert stats["bytes_read"] == 8
    assert stats["bytes_copied"] == 8
    assert stats["bytes_tensor_payload"] == 8
    assert stats["bytes_full_tensor_payload"] == 8
    assert stats["bytes_sliced_tensor_payload"] == 0
    assert stats["bytes_skipped_payload"] == 0


def test_uma_odirect_weight_source_iter_full_tensors_updates_source_stats(
    tmp_path, monkeypatch
):
    metadata = {
        "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 12)

    class FakeODirectFile:
        window_size = 64 * 1024 * 1024

        def __init__(self, *_args):
            self.direct_reads = 0
            self.window_loads = 0
            self.window_hits = 0
            self.bytes_read = 0
            self.bytes_copied = 0
            self.closed = False

        def read_record_into_tensor(self, tensor, _offset, size, gate=None):
            self.direct_reads += 1
            self.bytes_read += size
            self.bytes_copied += size
            tensor.fill_(size)
            if gate is not None:
                gate(size)

        def close(self):
            self.closed = True

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.uma_odirect_safetensors_loader."
        "_ODirectFile",
        FakeODirectFile,
    )

    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))
    items = list(source.iter_full_tensors())

    assert [name for name, _tensor in items] == ["a", "b"]
    assert items[0][1].tolist() == [4.0]
    assert items[1][1].tolist() == [8.0, 8.0]
    stats = source.stats_snapshot()
    assert stats["files_opened"] == 1
    assert stats["tensors_read"] == 2
    assert stats["tensors_read_full"] == 2
    assert stats["tensors_read_sliced"] == 0
    assert stats["direct_reads"] == 2
    assert stats["bytes_read"] == 12
    assert stats["bytes_copied"] == 12
    assert stats["bytes_tensor_payload"] == 12
    assert stats["bytes_full_tensor_payload"] == 12
    assert stats["bytes_sliced_tensor_payload"] == 0


def test_uma_odirect_weight_source_read_contiguous_slice_cpu(tmp_path, monkeypatch):
    metadata = {"a": {"dtype": "F32", "shape": [4, 3], "data_offsets": [0, 48]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 48)
    calls = []

    class FakeODirectFile:
        window_size = 64 * 1024 * 1024

        def __init__(self, *_args):
            self.direct_reads = 0
            self.window_loads = 0
            self.window_hits = 0
            self.bytes_read = 0
            self.bytes_copied = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            pass

        def read_record_into_tensor(self, tensor, offset, size, gate=None):
            calls.append((offset, size, tuple(tensor.shape)))
            self.direct_reads += 1
            self.bytes_read += size
            self.bytes_copied += size
            tensor.fill_(7)
            if gate is not None:
                gate(size)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.uma_odirect_safetensors_loader."
        "_ODirectFile",
        FakeODirectFile,
    )

    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))
    tensor = source.read_slice_cpu("a", (slice(1, 3), slice(None)))

    record = source.catalog.get("a")
    assert calls == [(record.offset + 12, 24, (2, 3))]
    assert tensor.tolist() == [[7.0, 7.0, 7.0], [7.0, 7.0, 7.0]]
    assert source.stats_snapshot()["bytes_tensor_payload"] == 24
    assert source.stats_snapshot()["tensors_read_sliced"] == 1
    assert source.stats_snapshot()["bytes_sliced_tensor_payload"] == 24


def test_uma_odirect_weight_source_read_into_cpu_full_and_slices(
    tmp_path, monkeypatch
):
    metadata = {
        "full": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "rows": {"dtype": "F32", "shape": [4, 3], "data_offsets": [8, 56]},
        "cols": {"dtype": "F32", "shape": [2, 4], "data_offsets": [56, 88]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 88)
    calls = []

    class FakeODirectFile:
        window_size = 64 * 1024 * 1024

        def __init__(self, *_args):
            self.direct_reads = 0
            self.window_loads = 0
            self.window_hits = 0
            self.bytes_read = 0
            self.bytes_copied = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            pass

        def read_record_into_tensor(self, tensor, offset, size, gate=None):
            calls.append((offset, size, tuple(tensor.shape)))
            self.direct_reads += 1
            self.bytes_read += size
            self.bytes_copied += size
            tensor.fill_(len(calls))
            if gate is not None:
                gate(size)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.uma_odirect_safetensors_loader."
        "_ODirectFile",
        FakeODirectFile,
    )

    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))
    full = torch.empty(2)
    rows = torch.empty(2, 3)
    cols = torch.empty(2, 2)

    source.read_into_cpu("full", full)
    source.read_into_cpu("rows", rows, source_slices=(slice(1, 3), slice(None)))
    source.read_into_cpu("cols", cols, source_slices=(slice(None), slice(2, 4)))

    full_record = source.catalog.get("full")
    rows_record = source.catalog.get("rows")
    cols_record = source.catalog.get("cols")
    assert calls == [
        (full_record.offset, 8, (2,)),
        (rows_record.offset + 12, 24, (2, 3)),
        (cols_record.offset + 8, 8, (2,)),
        (cols_record.offset + 24, 8, (2,)),
    ]
    assert full.tolist() == [1.0, 1.0]
    assert rows.tolist() == [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    assert cols.tolist() == [[3.0, 3.0], [4.0, 4.0]]
    stats = source.stats_snapshot()
    assert stats["tensors_read"] == 3
    assert stats["tensors_read_full"] == 1
    assert stats["tensors_read_sliced"] == 2
    assert stats["bytes_tensor_payload"] == 48
    assert stats["bytes_full_tensor_payload"] == 8
    assert stats["bytes_sliced_tensor_payload"] == 40


def test_uma_odirect_weight_source_read_into_cpu_target_slices(
    tmp_path, monkeypatch
):
    metadata = {
        "full": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        "rows": {"dtype": "F32", "shape": [4, 2], "data_offsets": [16, 48]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 48)
    calls = []

    class FakeODirectFile:
        window_size = 64 * 1024 * 1024

        def __init__(self, *_args):
            self.direct_reads = 0
            self.window_loads = 0
            self.window_hits = 0
            self.bytes_read = 0
            self.bytes_copied = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            pass

        def read_record_into_tensor(self, tensor, offset, size, gate=None):
            calls.append((offset, size, tuple(tensor.shape)))
            self.direct_reads += 1
            self.bytes_read += size
            self.bytes_copied += size
            tensor.fill_(len(calls))
            if gate is not None:
                gate(size)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.uma_odirect_safetensors_loader."
        "_ODirectFile",
        FakeODirectFile,
    )

    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))
    dst = torch.zeros(4, 2)
    source.read_into_cpu("full", dst, target_slices=(slice(1, 3), slice(None)))
    source.read_into_cpu(
        "rows",
        dst,
        source_slices=(slice(2, 4), slice(None)),
        target_slices=(slice(0, 2), slice(None)),
    )

    full_record = source.catalog.get("full")
    rows_record = source.catalog.get("rows")
    assert calls == [
        (full_record.offset, 16, (2, 2)),
        (rows_record.offset + 16, 16, (2, 2)),
    ]
    assert dst.tolist() == [
        [2.0, 2.0],
        [2.0, 2.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ]

    with pytest.raises(RuntimeError, match="target slice .*contiguous"):
        source.read_into_cpu(
            "full",
            torch.zeros(2, 4),
            target_slices=(slice(None), slice(1, 3)),
        )


def test_uma_odirect_weight_source_empty_cpu_uses_source_shape(tmp_path, monkeypatch):
    metadata = {"a": {"dtype": "F32", "shape": [2, 4], "data_offsets": [0, 32]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 32)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))

    full = source.empty_cpu("a")
    sliced = source.empty_cpu("a", source_slices=(slice(None), slice(2, 4)))

    assert full.shape == (2, 4)
    assert sliced.shape == (2, 2)
    assert full.dtype == torch.float32
    assert sliced.dtype == torch.float32
    assert source.stats_snapshot()["tensors_read"] == 0


def test_uma_odirect_weight_source_read_into_cpu_rejects_bad_dst(
    tmp_path, monkeypatch
):
    metadata = {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 8)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))

    with pytest.raises(RuntimeError, match="dtype mismatch"):
        source.read_into_cpu("a", torch.empty(2, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        source.read_into_cpu("a", torch.empty(3))


def test_uma_odirect_weight_source_rejects_noncontiguous_slice(tmp_path, monkeypatch):
    metadata = {"a": {"dtype": "F32", "shape": [4, 3, 2], "data_offsets": [0, 96]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 96)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))

    with pytest.raises(ValueError, match="not contiguous"):
        source.read_slice_cpu("a", (slice(1, 3), slice(1, 3), slice(None)))


def test_uma_odirect_weight_source_rejects_stepped_slice(tmp_path, monkeypatch):
    metadata = {"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 16)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))

    with pytest.raises(ValueError, match="step=1"):
        source.read_slice_cpu("a", (slice(None, None, 2),))


def test_uma_odirect_load_weights_uses_model_source_hook(tmp_path, monkeypatch):
    metadata = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 4)

    class FakeODirectFile:
        window_size = 64 * 1024 * 1024

        def __init__(self, *_args):
            self.direct_reads = 0
            self.window_loads = 0
            self.window_hits = 0
            self.bytes_read = 0
            self.bytes_copied = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            pass

        def read_record_into_tensor(self, tensor, _offset, size, gate=None):
            self.direct_reads += 1
            self.bytes_read += size
            self.bytes_copied += size
            tensor.fill_(5)
            if gate is not None:
                gate(size)

    class FakeModel:
        def __init__(self):
            self.plan_names = ()
            self.loaded = None

        def build_weight_plan(self, catalog):
            self.plan_names = catalog.names()
            return ["a"]

        def load_weights_from_source(self, source, plan):
            assert plan == ["a"]
            self.loaded = source.read_full_cpu("a")

        def load_weights(self, _weights):
            raise AssertionError("compatibility iterator path should not be used")

    class FakeModelConfig:
        model = str(tmp_path)
        model_weights = None

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.uma_odirect_safetensors_loader."
        "_ODirectFile",
        FakeODirectFile,
    )

    model = FakeModel()
    loader.load_weights(model, FakeModelConfig())

    assert model.plan_names == ("a",)
    assert model.loaded is not None
    assert model.loaded.tolist() == [5.0]


def test_uma_odirect_execute_weight_plan_reads_full_and_slice(tmp_path, monkeypatch):
    metadata = {
        "full": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "rows": {"dtype": "F32", "shape": [4, 2], "data_offsets": [4, 36]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 36)
    calls = []

    class FakeODirectFile:
        window_size = 64 * 1024 * 1024

        def __init__(self, *_args):
            self.direct_reads = 0
            self.window_loads = 0
            self.window_hits = 0
            self.bytes_read = 0
            self.bytes_copied = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            pass

        def read_record_into_tensor(self, tensor, offset, size, gate=None):
            calls.append((offset, size, tuple(tensor.shape)))
            self.direct_reads += 1
            self.bytes_read += size
            self.bytes_copied += size
            tensor.fill_(len(calls))
            if gate is not None:
                gate(size)

    class FakeParam:
        def __init__(self):
            self.loaded = []

        def weight_loader(self, param, tensor, **kwargs):
            assert param is self
            self.loaded.append((tensor.clone(), kwargs))

    class FakeModel:
        def __init__(self):
            self.full_param = FakeParam()
            self.slice_param = FakeParam()

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.uma_odirect_safetensors_loader."
        "_ODirectFile",
        FakeODirectFile,
    )
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))
    model = FakeModel()
    plan = WeightPlan(
        (
            WeightPlanEntry("full", "full_param"),
            WeightPlanEntry(
                "rows",
                "slice_param",
                source_slices=(slice(1, 3), slice(None)),
                shard_id="rows",
            ),
        )
    )

    loaded = execute_weight_plan(model, source, plan)

    rows_record = source.catalog.get("rows")
    assert loaded == {"full_param", "slice_param"}
    assert calls == [
        (source.catalog.get("full").offset, 4, (1,)),
        (rows_record.offset + 8, 16, (2, 2)),
    ]
    assert model.full_param.loaded[0][0].tolist() == [1.0]
    assert model.slice_param.loaded[0][0].tolist() == [[2.0, 2.0], [2.0, 2.0]]
    assert model.slice_param.loaded[0][1] == {"shard_id": "rows"}


def test_uma_odirect_execute_weight_plan_can_read_into_cpu(tmp_path, monkeypatch):
    metadata = {
        "full": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "rows": {"dtype": "F32", "shape": [4, 2], "data_offsets": [8, 40]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 40)
    calls = []

    class FakeODirectFile:
        window_size = 64 * 1024 * 1024

        def __init__(self, *_args):
            self.direct_reads = 0
            self.window_loads = 0
            self.window_hits = 0
            self.bytes_read = 0
            self.bytes_copied = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            pass

        def read_record_into_tensor(self, tensor, offset, size, gate=None):
            calls.append((offset, size, tuple(tensor.shape)))
            self.direct_reads += 1
            self.bytes_read += size
            self.bytes_copied += size
            tensor.fill_(len(calls))
            if gate is not None:
                gate(size)

    class FakeParam:
        def __init__(self):
            self.loaded = []

        def weight_loader(self, param, tensor, **kwargs):
            assert param is self
            self.loaded.append((tensor.clone(), kwargs))

    class FakeModel:
        def __init__(self):
            self.full_param = FakeParam()
            self.slice_param = FakeParam()

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.uma_odirect_safetensors_loader."
        "_ODirectFile",
        FakeODirectFile,
    )
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))
    model = FakeModel()
    plan = WeightPlan(
        (
            WeightPlanEntry("full", "full_param", read_into_cpu=True),
            WeightPlanEntry(
                "rows",
                "slice_param",
                source_slices=(slice(1, 3), slice(None)),
                read_into_cpu=True,
            ),
        )
    )

    loaded = execute_weight_plan(model, source, plan)

    rows_record = source.catalog.get("rows")
    assert loaded == {"full_param", "slice_param"}
    assert calls == [
        (source.catalog.get("full").offset, 8, (2,)),
        (rows_record.offset + 8, 16, (2, 2)),
    ]
    assert model.full_param.loaded[0][0].tolist() == [1.0, 1.0]
    assert model.slice_param.loaded[0][0].tolist() == [[2.0, 2.0], [2.0, 2.0]]
    stats = source.stats_snapshot()
    assert stats["tensors_read_full"] == 1
    assert stats["tensors_read_sliced"] == 1


def test_uma_odirect_execute_weight_plan_applies_transform(tmp_path, monkeypatch):
    metadata = {
        "weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 8)

    class FakeODirectFile:
        window_size = 64 * 1024 * 1024

        def __init__(self, *_args):
            self.direct_reads = 0
            self.window_loads = 0
            self.window_hits = 0
            self.bytes_read = 0
            self.bytes_copied = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            pass

        def read_record_into_tensor(self, tensor, _offset, size, gate=None):
            self.direct_reads += 1
            self.bytes_read += size
            self.bytes_copied += size
            tensor.copy_(torch.tensor([1.0, 2.0]))
            if gate is not None:
                gate(size)

    class FakeParam:
        def __init__(self):
            self.loaded = None

        def weight_loader(self, param, tensor, **kwargs):
            assert param is self
            assert kwargs == {}
            self.loaded = tensor.clone()

    class FakeModel:
        def __init__(self):
            self.param = FakeParam()

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.uma_odirect_safetensors_loader."
        "_ODirectFile",
        FakeODirectFile,
    )
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))
    model = FakeModel()
    plan = WeightPlan(
        (
            WeightPlanEntry(
                "weight",
                "param",
                transform=lambda tensor: tensor.flip(0),
            ),
        )
    )

    loaded = execute_weight_plan(model, source, plan)

    assert loaded == {"param"}
    assert model.param.loaded.tolist() == [2.0, 1.0]


def test_uma_odirect_execute_weight_plan_infers_output_tp_slice(
    tmp_path, monkeypatch
):
    metadata = {
        "rows": {"dtype": "F32", "shape": [4, 3], "data_offsets": [0, 48]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 48)
    calls = []

    class FakeODirectFile:
        window_size = 64 * 1024 * 1024

        def __init__(self, *_args):
            self.direct_reads = 0
            self.window_loads = 0
            self.window_hits = 0
            self.bytes_read = 0
            self.bytes_copied = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            pass

        def read_record_into_tensor(self, tensor, offset, size, gate=None):
            calls.append((offset, size, tuple(tensor.shape)))
            self.direct_reads += 1
            self.bytes_read += size
            self.bytes_copied += size
            tensor.fill_(9)
            if gate is not None:
                gate(size)

    class FakeParam:
        output_dim = 0
        tp_rank = 1
        tp_size = 2

        def __init__(self):
            self.data = torch.empty(2, 3)
            self.loaded = []

        def weight_loader(self, param, tensor, **kwargs):
            assert param is self
            assert getattr(self, "is_sharded_weight", False) is True
            self.loaded.append((tensor.clone(), kwargs))

    class FakeModel:
        def __init__(self):
            self.param = FakeParam()

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.uma_odirect_safetensors_loader."
        "_ODirectFile",
        FakeODirectFile,
    )
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))
    model = FakeModel()
    plan = WeightPlan((WeightPlanEntry("rows", "param"),))

    loaded = execute_weight_plan(model, source, plan)

    rows_record = source.catalog.get("rows")
    assert loaded == {"param"}
    assert calls == [(rows_record.offset + 24, 24, (2, 3))]
    assert not hasattr(model.param, "is_sharded_weight")
    assert model.param.loaded[0][0].tolist() == [
        [9.0, 9.0, 9.0],
        [9.0, 9.0, 9.0],
    ]
    stats = source.stats_snapshot()
    assert stats["tensors_read_sliced"] == 1
    assert stats["bytes_sliced_tensor_payload"] == 24
    assert stats["tensors_read_full"] == 0


def test_uma_odirect_execute_weight_plan_infers_shard_id_output_tp_slice(
    tmp_path, monkeypatch
):
    metadata = {
        "q": {"dtype": "F32", "shape": [4, 3], "data_offsets": [0, 48]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 48)
    calls = []

    class FakeODirectFile:
        window_size = 64 * 1024 * 1024

        def __init__(self, *_args):
            self.direct_reads = 0
            self.window_loads = 0
            self.window_hits = 0
            self.bytes_read = 0
            self.bytes_copied = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            pass

        def read_record_into_tensor(self, tensor, offset, size, gate=None):
            calls.append((offset, size, tuple(tensor.shape)))
            self.direct_reads += 1
            self.bytes_read += size
            self.bytes_copied += size
            tensor.fill_(7)
            if gate is not None:
                gate(size)

    class FakeQKVLayer:
        def __init__(self):
            self.loaded = []

        def _get_shard_size_mapping(self, shard_id):
            return {"q": 2}.get(shard_id)

        def weight_loader(self, param, tensor, shard_id=None, **kwargs):
            assert shard_id == "q"
            assert kwargs == {}
            assert getattr(param, "is_sharded_weight", False) is True
            self.loaded.append((tensor.clone(), shard_id))

    class FakeParam:
        output_dim = 0
        tp_rank = 1
        tp_size = 2

        def __init__(self):
            self.data = torch.empty(6, 3)
            self.layer = FakeQKVLayer()
            self.weight_loader = self.layer.weight_loader

    class FakeModel:
        def __init__(self):
            self.param = FakeParam()

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.uma_odirect_safetensors_loader."
        "_ODirectFile",
        FakeODirectFile,
    )
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))
    model = FakeModel()
    plan = WeightPlan((WeightPlanEntry("q", "param", shard_id="q"),))

    loaded = execute_weight_plan(model, source, plan)

    q_record = source.catalog.get("q")
    assert loaded == {"param"}
    assert calls == [(q_record.offset + 24, 24, (2, 3))]
    assert not hasattr(model.param, "is_sharded_weight")
    assert model.param.layer.loaded[0][0].tolist() == [
        [7.0, 7.0, 7.0],
        [7.0, 7.0, 7.0],
    ]
    assert model.param.layer.loaded[0][1] == "q"
    stats = source.stats_snapshot()
    assert stats["tensors_read_sliced"] == 1
    assert stats["bytes_sliced_tensor_payload"] == 24
    assert stats["tensors_read_full"] == 0


def test_uma_odirect_execute_weight_plan_infers_input_tp_strided_slice(
    tmp_path, monkeypatch
):
    metadata = {
        "cols": {"dtype": "F32", "shape": [2, 4], "data_offsets": [0, 32]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 32)
    calls = []

    class FakeODirectFile:
        window_size = 64 * 1024 * 1024

        def __init__(self, *_args):
            self.direct_reads = 0
            self.window_loads = 0
            self.window_hits = 0
            self.bytes_read = 0
            self.bytes_copied = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            pass

        def read_record_into_tensor(self, tensor, offset, size, gate=None):
            calls.append((offset, size, tuple(tensor.shape)))
            self.direct_reads += 1
            self.bytes_read += size
            self.bytes_copied += size
            tensor.fill_(3 + len(calls))
            if gate is not None:
                gate(size)

    class FakeParam:
        input_dim = 1
        tp_rank = 1
        tp_size = 2

        def __init__(self):
            self.data = torch.empty(2, 2)
            self.loaded = []

        def weight_loader(self, param, tensor, **kwargs):
            assert param is self
            assert kwargs == {}
            assert getattr(self, "is_sharded_weight", False) is True
            self.loaded.append(tensor.clone())

    class FakeModel:
        def __init__(self):
            self.param = FakeParam()

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.uma_odirect_safetensors_loader."
        "_ODirectFile",
        FakeODirectFile,
    )
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))
    model = FakeModel()
    plan = WeightPlan((WeightPlanEntry("cols", "param"),))

    loaded = execute_weight_plan(model, source, plan)

    cols_record = source.catalog.get("cols")
    assert loaded == {"param"}
    assert calls == [
        (cols_record.offset + 8, 8, (2,)),
        (cols_record.offset + 24, 8, (2,)),
    ]
    assert not hasattr(model.param, "is_sharded_weight")
    assert model.param.loaded[0].tolist() == [[4.0, 4.0], [5.0, 5.0]]
    stats = source.stats_snapshot()
    assert stats["tensors_read_sliced"] == 1
    assert stats["bytes_sliced_tensor_payload"] == 16
    assert stats["tensors_read_full"] == 0


def test_uma_odirect_execute_weight_plan_skips_not_required(tmp_path, monkeypatch):
    metadata = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 4)

    class FakeModel:
        pass

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))
    plan = WeightPlan((WeightPlanEntry("a", "missing", required=False),))

    assert execute_weight_plan(FakeModel(), source, plan) == set()
    stats = source.stats_snapshot()
    assert stats["tensors_skipped"] == 1
    assert stats["bytes_skipped_payload"] == 4


def test_uma_odirect_execute_weight_plan_skips_absent_not_required(
    tmp_path, monkeypatch
):
    metadata = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 4)

    class FakeModel:
        pass

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))
    plan = WeightPlan((WeightPlanEntry("missing", "missing", required=False),))

    assert execute_weight_plan(FakeModel(), source, plan) == set()
    stats = source.stats_snapshot()
    assert stats["tensors_skipped"] == 1
    assert stats["bytes_skipped_payload"] == 0


def test_uma_odirect_execute_weight_plan_ignores_missing_target_before_read():
    class FakeSource:
        def __init__(self):
            self.skipped = []
            self.reads = 0

        def skip(self, name, reason):
            self.skipped.append((name, reason))

        def read_full_cpu(self, _name):
            self.reads += 1
            raise AssertionError("missing ignored target should not be read")

    plan = WeightPlan(
        (
            WeightPlanEntry(
                "bias",
                "missing.bias",
                ignore_missing=True,
            ),
        )
    )
    source = FakeSource()

    assert execute_weight_plan(object(), source, plan) == set()
    assert source.reads == 0
    assert source.skipped == [("bias", "weight plan target is ignored")]


def test_uma_odirect_execute_weight_plan_missing_target_fails_before_read():
    class FakeSource:
        def __init__(self):
            self.reads = 0

        def read_full_cpu(self, _name):
            self.reads += 1
            raise AssertionError("missing target should fail before read")

    source = FakeSource()
    plan = WeightPlan((WeightPlanEntry("weight", "missing.weight"),))

    with pytest.raises(RuntimeError, match="Cannot resolve"):
        execute_weight_plan(object(), source, plan)
    assert source.reads == 0


def test_uma_odirect_build_auto_weight_plan_from_catalog(tmp_path):
    metadata = {
        "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
        "model.layers.0.rotary_emb.inv_freq": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [8, 12],
        },
        "drop_me.weight": {"dtype": "F32", "shape": [1], "data_offsets": [12, 16]},
        "spec_layer.weight": {"dtype": "F32", "shape": [1], "data_offsets": [16, 20]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 20)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeMapper:
        def _map_name_with_shard(self, name):
            if name == "drop_me.weight":
                return None
            if name.endswith("q_proj.weight"):
                return name.replace("q_proj", "qkv_proj"), "q"
            return name, None

    plan = build_auto_weight_plan_from_catalog(
        catalog,
        mapper=FakeMapper(),
        skip_prefixes=["lm_head."],
        skip_predicate=lambda name: name.startswith("spec_layer."),
    )
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["lm_head.weight"].required is False
    assert entries["model.layers.0.rotary_emb.inv_freq"].required is False
    assert entries["drop_me.weight"].required is False
    assert entries["spec_layer.weight"].required is False
    q_proj = entries["model.layers.0.self_attn.q_proj.weight"]
    assert q_proj.required is True
    assert q_proj.target_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert q_proj.shard_id == "q"


def test_uma_odirect_build_auto_weight_plan_marks_ignored_suffix(tmp_path):
    metadata = {
        "linear.bias": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 4)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    plan = build_auto_weight_plan_from_catalog(
        catalog,
        ignore_unexpected_suffixes=[".bias"],
    )

    assert plan.entries[0].checkpoint_name == "linear.bias"
    assert plan.entries[0].ignore_missing is True


def test_qwen3_build_weight_plan_uses_catalog_mapper_and_tie_skip(tmp_path):
    metadata = {
        "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 8)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeConfig:
        tie_word_embeddings = True

    class FakeQwen3:
        config = FakeConfig()
        hf_to_vllm_mapper = qwen3.Qwen3ForCausalLM.hf_to_vllm_mapper

        def children(self):
            return []

    plan = qwen3.Qwen3ForCausalLM.build_weight_plan(FakeQwen3(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["lm_head.weight"].required is False
    q_proj = entries["model.layers.0.self_attn.q_proj.weight"]
    assert q_proj.required is True
    assert q_proj.target_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert q_proj.shard_id == "q"


def test_qwen3_build_weight_plan_uses_quant_cache_mapper_and_ignore_suffixes(
    tmp_path,
):
    metadata = {
        "cache_scale": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "ignored_quant.ignored": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 8)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeQuantConfig:
        _ignore_unexpected_suffixes = [".ignored"]

        def get_cache_scale_mapper(self):
            return WeightsMapper(
                orig_to_new_substr={"cache_scale": "model.layers.0.cache_scale"}
            )

    class FakeConfig:
        tie_word_embeddings = False

    class FakeChild:
        quant_config = FakeQuantConfig()

    class FakeQwen3:
        config = FakeConfig()
        hf_to_vllm_mapper = qwen3.Qwen3ForCausalLM.hf_to_vllm_mapper

        def children(self):
            return [FakeChild()]

    plan = qwen3.Qwen3ForCausalLM.build_weight_plan(FakeQwen3(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["cache_scale"].target_name == "model.layers.0.cache_scale"
    assert entries["ignored_quant.ignored"].target_name == "ignored_quant.ignored"
    assert entries["ignored_quant.ignored"].ignore_missing is True


def test_qwen3_load_weights_from_source_delegates_to_executor(monkeypatch):
    calls = []

    def fake_load(model, source, plan):
        calls.append((model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(qwen3, "load_auto_uma_weights_from_source", fake_load)
    model = object()
    source = object()
    plan = WeightPlan(())

    loaded = qwen3.Qwen3ForCausalLM.load_weights_from_source(model, source, plan)

    assert loaded == {"loaded"}
    assert calls == [(model, source, plan)]


def test_qwen2_build_weight_plan_uses_catalog_mapper_and_tie_skip(tmp_path):
    metadata = {
        "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
        "model.layers.0.mlp.gate_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [8, 12],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 12)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeConfig:
        tie_word_embeddings = True

    class FakeQwen2:
        config = FakeConfig()
        hf_to_vllm_mapper = qwen2.Qwen2ForCausalLM.hf_to_vllm_mapper

        def children(self):
            return []

    plan = qwen2.Qwen2ForCausalLM.build_weight_plan(FakeQwen2(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["lm_head.weight"].required is False
    q_proj = entries["model.layers.0.self_attn.q_proj.weight"]
    assert q_proj.target_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert q_proj.shard_id == "q"
    gate_proj = entries["model.layers.0.mlp.gate_proj.weight"]
    assert gate_proj.target_name == "model.layers.0.mlp.gate_up_proj.weight"
    assert gate_proj.shard_id == 0


def test_qwen2_load_weights_from_source_delegates_to_executor(monkeypatch):
    calls = []

    def fake_load(model, source, plan):
        calls.append((model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(qwen2, "load_auto_uma_weights_from_source", fake_load)
    model = object()
    source = object()
    plan = WeightPlan(())

    loaded = qwen2.Qwen2ForCausalLM.load_weights_from_source(model, source, plan)

    assert loaded == {"loaded"}
    assert calls == [(model, source, plan)]


def test_llama_build_weight_plan_uses_catalog_mapper_and_tie_skip(tmp_path):
    metadata = {
        "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
        "model.layers.0.mlp.gate_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [8, 12],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 12)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeConfig:
        tie_word_embeddings = True

    class FakeLlama:
        config = FakeConfig()
        hf_to_vllm_mapper = llama.LlamaForCausalLM.hf_to_vllm_mapper

        def children(self):
            return []

    plan = llama.LlamaForCausalLM.build_weight_plan(FakeLlama(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["lm_head.weight"].required is False
    q_proj = entries["model.layers.0.self_attn.q_proj.weight"]
    assert q_proj.target_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert q_proj.shard_id == "q"
    gate_proj = entries["model.layers.0.mlp.gate_proj.weight"]
    assert gate_proj.target_name == "model.layers.0.mlp.gate_up_proj.weight"
    assert gate_proj.shard_id == 0


def test_llama_build_weight_plan_uses_quant_cache_mapper_and_ignore_suffixes(
    tmp_path,
):
    metadata = {
        "cache_scale": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "ignored_quant.ignored": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 8)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeQuantConfig:
        _ignore_unexpected_suffixes = [".ignored"]

        def get_cache_scale_mapper(self):
            return WeightsMapper(
                orig_to_new_substr={"cache_scale": "model.layers.0.cache_scale"}
            )

    class FakeConfig:
        tie_word_embeddings = False

    class FakeChild:
        quant_config = FakeQuantConfig()

    class FakeLlama:
        config = FakeConfig()
        hf_to_vllm_mapper = llama.LlamaForCausalLM.hf_to_vllm_mapper

        def children(self):
            return [FakeChild()]

    plan = llama.LlamaForCausalLM.build_weight_plan(FakeLlama(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["cache_scale"].target_name == "model.layers.0.cache_scale"
    assert entries["ignored_quant.ignored"].ignore_missing is True


def test_llama_load_weights_from_source_delegates_to_executor(monkeypatch):
    calls = []

    def fake_load(model, source, plan):
        calls.append((model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(llama, "load_auto_uma_weights_from_source", fake_load)
    model = object()
    source = object()
    plan = WeightPlan(())

    loaded = llama.LlamaForCausalLM.load_weights_from_source(model, source, plan)

    assert loaded == {"loaded"}
    assert calls == [(model, source, plan)]


def test_olmo2_build_weight_plan_uses_catalog_mapper_and_tie_skip(tmp_path):
    metadata = {
        "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
        "model.layers.0.mlp.up_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [8, 12],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 12)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeConfig:
        tie_word_embeddings = True

    class FakeOlmo2:
        config = FakeConfig()
        hf_to_vllm_mapper = olmo2.Olmo2ForCausalLM.hf_to_vllm_mapper

        def children(self):
            return []

    plan = olmo2.Olmo2ForCausalLM.build_weight_plan(FakeOlmo2(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["lm_head.weight"].required is False
    q_proj = entries["model.layers.0.self_attn.q_proj.weight"]
    assert q_proj.target_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert q_proj.shard_id == "q"
    up_proj = entries["model.layers.0.mlp.up_proj.weight"]
    assert up_proj.target_name == "model.layers.0.mlp.gate_up_proj.weight"
    assert up_proj.shard_id == 1


def test_olmo2_load_weights_from_source_delegates_to_executor(monkeypatch):
    calls = []

    def fake_load(model, source, plan):
        calls.append((model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(olmo2, "load_auto_uma_weights_from_source", fake_load)
    model = object()
    source = object()
    plan = WeightPlan(())

    loaded = olmo2.Olmo2ForCausalLM.load_weights_from_source(model, source, plan)

    assert loaded == {"loaded"}
    assert calls == [(model, source, plan)]


@pytest.mark.parametrize(
    "model_cls, module, fused_source_name, shard_id",
    [
        (olmo.OlmoForCausalLM, olmo, "model.layers.0.mlp.gate_proj.weight", 0),
        (exaone.ExaoneForCausalLM, exaone, "model.layers.0.mlp.c_fc_0.weight", 0),
    ],
)
def test_olmo_exaone_dense_hooks_use_mapper_and_tie_skip(
    tmp_path, model_cls, module, fused_source_name, shard_id
):
    metadata = {
        "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
        fused_source_name: {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [8, 12],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 12)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeConfig:
        tie_word_embeddings = True

    class FakeModel:
        config = FakeConfig()
        hf_to_vllm_mapper = model_cls.hf_to_vllm_mapper

        def children(self):
            return []

    plan = model_cls.build_weight_plan(FakeModel(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["lm_head.weight"].required is False
    q_proj = entries["model.layers.0.self_attn.q_proj.weight"]
    assert q_proj.target_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert q_proj.shard_id == "q"
    gate_proj = entries[fused_source_name]
    assert gate_proj.target_name == "model.layers.0.mlp.gate_up_proj.weight"
    assert gate_proj.shard_id == shard_id


@pytest.mark.parametrize(
    "model_cls, module",
    [
        (olmo.OlmoForCausalLM, olmo),
        (exaone.ExaoneForCausalLM, exaone),
        (nemotron.NemotronForCausalLM, nemotron),
    ],
)
def test_more_dense_load_weights_from_source_delegates_to_executor(
    monkeypatch, model_cls, module
):
    calls = []

    def fake_load(model, source, plan):
        calls.append((model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(module, "load_auto_uma_weights_from_source", fake_load)
    model = object()
    source = object()
    plan = WeightPlan(())

    loaded = model_cls.load_weights_from_source(model, source, plan)

    assert loaded == {"loaded"}
    assert calls == [(model, source, plan)]


def test_nemotron_build_weight_plan_uses_qkv_mapper_without_lm_head_skip(tmp_path):
    metadata = {
        "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "model.layers.0.self_attn.k_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
        "model.layers.0.mlp.up_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [8, 12],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 12)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeNemotron:
        hf_to_vllm_mapper = nemotron.NemotronForCausalLM.hf_to_vllm_mapper

        def children(self):
            return []

    plan = nemotron.NemotronForCausalLM.build_weight_plan(FakeNemotron(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["lm_head.weight"].required is True
    k_proj = entries["model.layers.0.self_attn.k_proj.weight"]
    assert k_proj.target_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert k_proj.shard_id == "k"
    assert (
        entries["model.layers.0.mlp.up_proj.weight"].target_name
        == "model.layers.0.mlp.up_proj.weight"
    )


def test_commandr_build_weight_plan_uses_catalog_mapper_and_static_skips(tmp_path):
    metadata = {
        "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "model.rotary_emb.inv_freq": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [8, 12],
        },
        "model.layers.0.mlp.up_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [12, 16],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 16)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeCommandR:
        hf_to_vllm_mapper = commandr.CohereForCausalLM.hf_to_vllm_mapper

        def children(self):
            return []

    plan = commandr.CohereForCausalLM.build_weight_plan(FakeCommandR(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["lm_head.weight"].required is False
    assert entries["model.rotary_emb.inv_freq"].required is False
    q_proj = entries["model.layers.0.self_attn.q_proj.weight"]
    assert q_proj.target_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert q_proj.shard_id == "q"
    up_proj = entries["model.layers.0.mlp.up_proj.weight"]
    assert up_proj.target_name == "model.layers.0.mlp.gate_up_proj.weight"
    assert up_proj.shard_id == 1


def test_commandr_load_weights_from_source_delegates_to_executor(monkeypatch):
    calls = []

    def fake_load(model, source, plan):
        calls.append((model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(commandr, "load_auto_uma_weights_from_source", fake_load)
    model = object()
    source = object()
    plan = WeightPlan(())

    loaded = commandr.CohereForCausalLM.load_weights_from_source(model, source, plan)

    assert loaded == {"loaded"}
    assert calls == [(model, source, plan)]


@pytest.mark.parametrize(
    "model_cls",
    [
        gemma.GemmaForCausalLM,
        gemma2.Gemma2ForCausalLM,
        gemma3.Gemma3ForCausalLM,
    ],
)
def test_gemma_dense_hooks_use_mapper_and_tie_skip(tmp_path, model_cls):
    metadata = {
        "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
        "model.layers.0.mlp.up_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [8, 12],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 12)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeConfig:
        tie_word_embeddings = True

    class FakeGemma:
        config = FakeConfig()
        hf_to_vllm_mapper = model_cls.hf_to_vllm_mapper

        def children(self):
            return []

    plan = model_cls.build_weight_plan(FakeGemma(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["lm_head.weight"].required is False
    q_proj = entries["model.layers.0.self_attn.q_proj.weight"]
    assert q_proj.target_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert q_proj.shard_id == "q"
    up_proj = entries["model.layers.0.mlp.up_proj.weight"]
    assert up_proj.target_name == "model.layers.0.mlp.gate_up_proj.weight"
    assert up_proj.shard_id == 1


@pytest.mark.parametrize(
    "model_cls, module",
    [
        (gemma.GemmaForCausalLM, gemma),
        (gemma2.Gemma2ForCausalLM, gemma2),
        (gemma3.Gemma3ForCausalLM, gemma3),
    ],
)
def test_gemma_dense_load_weights_from_source_delegates_to_executor(
    monkeypatch, model_cls, module
):
    calls = []

    def fake_load(model, source, plan):
        calls.append((model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(module, "load_auto_uma_weights_from_source", fake_load)
    model = object()
    source = object()
    plan = WeightPlan(())

    loaded = model_cls.load_weights_from_source(model, source, plan)

    assert loaded == {"loaded"}
    assert calls == [(model, source, plan)]


def test_internlm2_build_weight_plan_uses_mapper_and_tie_skip(tmp_path):
    metadata = {
        "output.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "model.layers.0.feed_forward.w1.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
        "model.layers.0.feed_forward.w3.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [8, 12],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 12)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeConfig:
        tie_word_embeddings = True

    class FakeInternLM2:
        config = FakeConfig()
        hf_to_vllm_mapper = internlm2.InternLM2ForCausalLM.hf_to_vllm_mapper

        def children(self):
            return []

    plan = internlm2.InternLM2ForCausalLM.build_weight_plan(FakeInternLM2(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["output.weight"].required is False
    w1 = entries["model.layers.0.feed_forward.w1.weight"]
    assert w1.target_name == "model.layers.0.feed_forward.gate_up_proj.weight"
    assert w1.shard_id == 0
    w3 = entries["model.layers.0.feed_forward.w3.weight"]
    assert w3.target_name == "model.layers.0.feed_forward.gate_up_proj.weight"
    assert w3.shard_id == 1


def test_internlm2_load_weights_from_source_delegates_to_executor(monkeypatch):
    calls = []

    def fake_load(model, source, plan):
        calls.append((model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(internlm2, "load_auto_uma_weights_from_source", fake_load)
    model = object()
    source = object()
    plan = WeightPlan(())

    loaded = internlm2.InternLM2ForCausalLM.load_weights_from_source(
        model, source, plan
    )

    assert loaded == {"loaded"}
    assert calls == [(model, source, plan)]


@pytest.mark.parametrize(
    "model_cls",
    [
        phi.PhiForCausalLM,
        starcoder2.Starcoder2ForCausalLM,
    ],
)
def test_phi_starcoder2_dense_hooks_use_qkv_mapper(tmp_path, model_cls):
    metadata = {
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        },
        "model.layers.0.self_attn.k_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
        "model.layers.0.self_attn.v_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [8, 12],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 12)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeConfig:
        tie_word_embeddings = False

    class FakeModel:
        config = FakeConfig()
        hf_to_vllm_mapper = model_cls.hf_to_vllm_mapper

        def children(self):
            return []

    plan = model_cls.build_weight_plan(FakeModel(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    for source_name, shard_id in [
        ("model.layers.0.self_attn.q_proj.weight", "q"),
        ("model.layers.0.self_attn.k_proj.weight", "k"),
        ("model.layers.0.self_attn.v_proj.weight", "v"),
    ]:
        entry = entries[source_name]
        assert entry.target_name == "model.layers.0.self_attn.qkv_proj.weight"
        assert entry.shard_id == shard_id


def test_starcoder2_build_weight_plan_skips_tied_lm_head(tmp_path):
    metadata = {
        "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 4)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeConfig:
        tie_word_embeddings = True

    class FakeStarcoder2:
        config = FakeConfig()
        hf_to_vllm_mapper = starcoder2.Starcoder2ForCausalLM.hf_to_vllm_mapper

        def children(self):
            return []

    plan = starcoder2.Starcoder2ForCausalLM.build_weight_plan(
        FakeStarcoder2(), catalog
    )

    assert plan.entries[0].checkpoint_name == "lm_head.weight"
    assert plan.entries[0].required is False


@pytest.mark.parametrize(
    "model_cls, module",
    [
        (phi.PhiForCausalLM, phi),
        (starcoder2.Starcoder2ForCausalLM, starcoder2),
    ],
)
def test_phi_starcoder2_load_weights_from_source_delegates_to_executor(
    monkeypatch, model_cls, module
):
    calls = []

    def fake_load(model, source, plan):
        calls.append((model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(module, "load_auto_uma_weights_from_source", fake_load)
    model = object()
    source = object()
    plan = WeightPlan(())

    loaded = model_cls.load_weights_from_source(model, source, plan)

    assert loaded == {"loaded"}
    assert calls == [(model, source, plan)]


def test_falcon_build_weight_plan_uses_tie_skip(tmp_path):
    metadata = {
        "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "transformer.h.0.self_attention.query_key_value.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 8)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeConfig:
        tie_word_embeddings = True

    class FakeFalcon:
        config = FakeConfig()

        def children(self):
            return []

    plan = falcon.FalconForCausalLM.build_weight_plan(FakeFalcon(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["lm_head.weight"].required is False
    qkv = entries["transformer.h.0.self_attention.query_key_value.weight"]
    assert qkv.required is True
    assert qkv.target_name == "transformer.h.0.self_attention.query_key_value.weight"


def test_falcon_load_weights_from_source_delegates_to_executor(monkeypatch):
    calls = []

    def fake_load(model, source, plan):
        calls.append((model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(falcon, "load_auto_uma_weights_from_source", fake_load)
    model = object()
    source = object()
    plan = WeightPlan(())

    loaded = falcon.FalconForCausalLM.load_weights_from_source(model, source, plan)

    assert loaded == {"loaded"}
    assert calls == [(model, source, plan)]


def test_mistral_build_weight_plan_remaps_names_and_permute_transform(tmp_path):
    metadata = {
        "layers.0.attention.wq.weight": {
            "dtype": "F32",
            "shape": [4, 4],
            "data_offsets": [0, 64],
        },
        "output.weight": {"dtype": "F32", "shape": [1], "data_offsets": [64, 68]},
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 68)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
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
        _permute_mistral_weight = mistral.MistralForCausalLM._permute_mistral_weight
        _remap_mistral_name = mistral.MistralForCausalLM._remap_mistral_name
        _mistral_source_name_transform = (
            mistral.MistralForCausalLM._mistral_source_name_transform
        )

        def children(self):
            return []

    fake = FakeMistral()
    plan = mistral.MistralForCausalLM.build_weight_plan(fake, catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    wq = entries["layers.0.attention.wq.weight"]
    assert wq.target_name == "model.layers.0.self_attn.qkv_proj.weight"
    assert wq.shard_id == "q"
    assert wq.transform is not None
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    assert torch.equal(
        wq.transform(tensor),
        fake._permute_mistral_weight(tensor, 2, 4),
    )
    output = entries["output.weight"]
    assert output.required is False
    assert output.target_name == "lm_head.weight"


def test_mistral_load_weights_from_source_delegates_to_executor(monkeypatch):
    calls = []

    def fake_load(model, source, plan):
        calls.append((model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(mistral, "load_auto_uma_weights_from_source", fake_load)
    model = object()
    source = object()
    plan = WeightPlan(())

    loaded = mistral.MistralForCausalLM.load_weights_from_source(model, source, plan)

    assert loaded == {"loaded"}
    assert calls == [(model, source, plan)]


@pytest.mark.parametrize(
    "model_cls, module, skip_prefix",
    [
        (gpt_bigcode.GPTBigCodeForCausalLM, gpt_bigcode, "lm_head."),
        (opt.OPTForCausalLM, opt, "lm_head.weight"),
    ],
)
def test_gpt_bigcode_opt_dense_hooks_use_tie_skip(
    tmp_path, model_cls, module, skip_prefix
):
    metadata = {
        "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "model.layers.0.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 8)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeConfig:
        tie_word_embeddings = True

    class FakeModel:
        config = FakeConfig()
        hf_to_vllm_mapper = getattr(model_cls, "hf_to_vllm_mapper", None)

        def children(self):
            return []

    plan = model_cls.build_weight_plan(FakeModel(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["lm_head.weight"].required is False
    assert entries["model.layers.0.weight"].required is True
    assert skip_prefix


def test_bloom_build_weight_plan_adds_transformer_prefix_and_tie_skip(tmp_path):
    metadata = {
        "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "h.0.self_attention.query_key_value.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
        "transformer.word_embeddings.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [8, 12],
        },
    }
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 12)
    catalog = TensorCatalog.from_safetensors_files(
        [str(path)],
        metadata_limit_bytes=1024 * 1024,
    )

    class FakeBloom:
        def children(self):
            return []

    plan = bloom.BloomForCausalLM.build_weight_plan(FakeBloom(), catalog)
    entries = {entry.checkpoint_name: entry for entry in plan.entries}

    assert entries["lm_head.weight"].target_name == "transformer.lm_head.weight"
    assert entries["h.0.self_attention.query_key_value.weight"].target_name == (
        "transformer.h.0.self_attention.query_key_value.weight"
    )
    assert entries["transformer.word_embeddings.weight"].target_name == (
        "transformer.word_embeddings.weight"
    )


@pytest.mark.parametrize(
    "model_cls, module",
    [
        (gpt_bigcode.GPTBigCodeForCausalLM, gpt_bigcode),
        (opt.OPTForCausalLM, opt),
        (bloom.BloomForCausalLM, bloom),
    ],
)
def test_more_dense_compat_load_weights_from_source_delegates_to_executor(
    monkeypatch, model_cls, module
):
    calls = []

    def fake_load(model, source, plan):
        calls.append((model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(module, "load_auto_uma_weights_from_source", fake_load)
    model = object()
    source = object()
    plan = WeightPlan(())

    loaded = model_cls.load_weights_from_source(model, source, plan)

    assert loaded == {"loaded"}
    assert calls == [(model, source, plan)]


def test_qwen3_moe_source_plan_skips_nonlocal_experts_before_read():
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

    class FakeRoutedExperts:
        layer_name = "model.layers.0.mlp.experts.routed_experts"
        w13_weight = object()
        quant_method = object()

        def __init__(self):
            self.calls = []

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return 0 if expert_id == 0 else -1

        def weight_loader(self, **kwargs):
            self.calls.append(kwargs)
            return True

    class FakeExperts:
        def __init__(self, routed_experts):
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

    class FakeSource:
        def __init__(self, catalog):
            self.catalog = catalog
            self.reads = []
            self.skips = []

        def read_full_cpu(self, name):
            self.reads.append(name)
            return torch.ones(1)

        def skip(self, name, reason):
            self.skips.append((name, reason))

    model = FakeQwen3Moe()
    source = FakeSource(catalog)

    plan = model.build_weight_plan(catalog)
    assert len(plan.auto_plan.entries) == 0
    assert [entry.local_required for entry in plan.routed_entries] == [True, False]

    loaded = model.load_weights_from_source(source, plan)

    assert source.reads == [names[0]]
    assert source.skips == [(names[1], "non-local routed expert")]
    assert loaded == {"model.layers.0.mlp.experts.routed_experts.w13_weight"}
    assert model.routed_experts.calls[0]["expert_id"] == 0
    assert model.routed_experts.calls[0]["shard_id"] == "w1"


def test_mixtral_moe_source_plan_skips_nonlocal_experts_before_read():
    names = [
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.block_sparse_moe.experts.1.w1.weight",
    ]
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", names[0], torch.float32, [1], 0, 4),
            TensorMeta("model.safetensors", names[1], torch.float32, [1], 4, 4),
        ]
    )

    class FakeRoutedExperts:
        layer_name = "model.layers.0.block_sparse_moe.experts"
        w13_weight = object()
        quant_method = object()

        def __init__(self):
            self.calls = []

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return 0 if expert_id == 0 else -1

        def weight_loader(self, **kwargs):
            self.calls.append(kwargs)
            return True

    class FakeBlockSparseMoe:
        def __init__(self, routed_experts):
            self.experts = routed_experts

    class FakeLayer:
        def __init__(self, routed_experts):
            self.block_sparse_moe = FakeBlockSparseMoe(routed_experts)

    class FakeInnerModel:
        def __init__(self, routed_experts):
            self.layers = [FakeLayer(routed_experts)]

    class FakeConfig:
        tie_word_embeddings = False

    class FakeMixtral(mixtral.MixtralForCausalLM):
        hf_to_vllm_mapper = None

        def __init__(self):
            nn.Module.__init__(self)
            self.config = FakeConfig()
            self.routed_experts = FakeRoutedExperts()
            self.model = FakeInnerModel(self.routed_experts)

    class FakeSource:
        def __init__(self, catalog):
            self.catalog = catalog
            self.reads = []
            self.skips = []

        def read_full_cpu(self, name):
            self.reads.append(name)
            return torch.ones(1)

        def skip(self, name, reason):
            self.skips.append((name, reason))

    model = FakeMixtral()
    source = FakeSource(catalog)

    plan = model.build_weight_plan(catalog)
    assert len(plan.auto_plan.entries) == 0
    assert [entry.local_required for entry in plan.routed_entries] == [True, False]

    loaded = model.load_weights_from_source(source, plan)

    assert source.reads == [names[0]]
    assert source.skips == [(names[1], "non-local routed expert")]
    assert loaded == {"model.layers.0.block_sparse_moe.experts.w13_weight"}
    assert model.routed_experts.calls[0]["expert_id"] == 0
    assert model.routed_experts.calls[0]["shard_id"] == "w1"


def test_deepseek_moe_source_plan_skips_nonlocal_experts_before_read():
    names = [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.4.self_attn.indexer.wq.weight",
        "model.layers.2.input_layernorm.weight",
    ]
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", names[0], torch.float32, [1], 0, 4),
            TensorMeta("model.safetensors", names[1], torch.float32, [1], 4, 4),
            TensorMeta("model.safetensors", names[2], torch.float32, [1], 8, 4),
            TensorMeta("model.safetensors", names[3], torch.float32, [1], 12, 4),
        ]
    )

    class FakeRoutedExperts:
        layer_name = "model.layers.0.mlp.experts"
        w13_weight = object()
        quant_method = object()

        def __init__(self):
            self.calls = []

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return 0 if expert_id == 0 else -1

        def weight_loader(self, **kwargs):
            self.calls.append(kwargs)
            return True

    class FakeMLP:
        def __init__(self, routed_experts=None):
            self.experts = routed_experts

    class FakeLayer(nn.Module):
        def __init__(self, routed_experts=None):
            super().__init__()
            self.mlp = FakeMLP(routed_experts)
            self.input_layernorm = nn.LayerNorm(1)

    class FakeInnerModel(nn.Module):
        def __init__(self, routed_experts):
            super().__init__()
            self.layers = nn.ModuleList([
                FakeLayer(routed_experts),
                FakeLayer(),
                FakeLayer(),
                FakeLayer(),
                FakeLayer(),
            ])

    class FakeConfig:
        tie_word_embeddings = False
        num_hidden_layers = 3
        num_nextn_predict_layers = 1

    class FakeDeepseek(deepseek_v2.DeepseekV2ForCausalLM):
        use_mha = True

        def __init__(self):
            nn.Module.__init__(self)
            self.config = FakeConfig()
            self.routed_experts = FakeRoutedExperts()
            self.model = FakeInnerModel(self.routed_experts)

    class FakeSource:
        def __init__(self, catalog):
            self.catalog = catalog
            self.reads = []
            self.skips = []

        def read_full_cpu(self, name):
            self.reads.append(name)
            return torch.ones(1)

        def skip(self, name, reason):
            self.skips.append((name, reason))

    model = FakeDeepseek()
    source = FakeSource(catalog)

    plan = model.build_weight_plan(catalog)
    assert [entry.local_required for entry in plan.routed_entries] == [True, False]
    assert {
        entry.checkpoint_name
        for entry in plan.auto_plan.entries
        if not entry.required
    } == {names[2]}

    loaded = model.load_weights_from_source(source, plan)

    assert source.reads == [names[3], names[0]]
    assert source.skips == [
        (names[2], "weight plan marked not required"),
        (names[1], "non-local routed expert"),
    ]
    assert "model.layers.2.input_layernorm.weight" in loaded
    assert "model.layers.0.mlp.experts.w13_weight" in loaded
    assert model.routed_experts.calls[0]["expert_id"] == 0
    assert model.routed_experts.calls[0]["shard_id"] == "w1"


def test_deepseek_moe_source_plan_slices_shared_expert_fusion_before_read():
    name = "model.layers.0.mlp.shared_experts.gate_proj.weight"
    catalog = TensorCatalog(
        [TensorMeta("model.safetensors", name, torch.float32, [4, 1], 0, 16)]
    )

    class FakeRoutedExperts:
        layer_name = "model.layers.0.mlp.experts"
        w13_weight = object()
        quant_method = object()

        def __init__(self):
            self.calls = []

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return 0 if expert_id in (2, 3) else -1

        def weight_loader(self, **kwargs):
            self.calls.append(kwargs)
            return True

    class FakeMLP:
        def __init__(self, routed_experts):
            self.experts = routed_experts

    class FakeLayer(nn.Module):
        def __init__(self, routed_experts):
            super().__init__()
            self.mlp = FakeMLP(routed_experts)

    class FakeInnerModel(nn.Module):
        def __init__(self, routed_experts):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(routed_experts)])

    class FakeConfig:
        tie_word_embeddings = False
        num_nextn_predict_layers = 0
        n_routed_experts = 2
        n_shared_experts = 2

    class FakeDeepseek(deepseek_v2.DeepseekV2ForCausalLM):
        use_mha = True

        def __init__(self):
            nn.Module.__init__(self)
            self.config = FakeConfig()
            self.routed_experts = FakeRoutedExperts()
            self.model = FakeInnerModel(self.routed_experts)

    class FakeSource:
        def __init__(self, catalog):
            self.catalog = catalog
            self.slices = []

        def read_full_cpu(self, name):
            raise AssertionError(f"unexpected full read for {name}")

        def read_slice_cpu(self, name, source_slices):
            self.slices.append((name, source_slices))
            return torch.ones(2, 1)

        def skip(self, name, reason):
            raise AssertionError(f"unexpected skip for {name}: {reason}")

    model = FakeDeepseek()
    source = FakeSource(catalog)
    plan = model.build_weight_plan(catalog)

    assert len(plan.auto_plan.entries) == 0
    assert [
        (entry.expert_id, entry.source_slices) for entry in plan.routed_entries
    ] == [
        (2, (slice(0, 2), slice(None))),
        (3, (slice(2, 4), slice(None))),
    ]

    loaded = model.load_weights_from_source(source, plan)

    assert source.slices == [
        (name, (slice(0, 2), slice(None))),
        (name, (slice(2, 4), slice(None))),
    ]
    assert [call["expert_id"] for call in model.routed_experts.calls] == [2, 3]
    assert [call["shard_id"] for call in model.routed_experts.calls] == ["w1", "w1"]
    assert loaded == {"model.layers.0.mlp.experts.w13_weight"}


def test_granite_moe_source_plan_slices_fused_expert_tensors_before_read():
    names = [
        "model.layers.0.block_sparse_moe.input_linear.weight",
        "model.layers.0.block_sparse_moe.output_linear.weight",
        "model.layers.0.block_sparse_moe.router.layer.weight",
    ]
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", names[0], torch.float32, [2, 4, 3], 0, 96),
            TensorMeta(
                "model.safetensors", names[1], torch.float32, [2, 3, 2], 96, 48
            ),
            TensorMeta(
                "model.safetensors", names[2], torch.float32, [2, 1], 144, 8
            ),
        ]
    )

    class FakeRoutedExperts:
        layer_name = "model.layers.0.block_sparse_moe.experts"
        w13_weight = object()
        w2_weight = object()
        quant_method = object()

        def __init__(self):
            self.calls = []

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return 0 if expert_id == 0 else -1

        def weight_loader(self, **kwargs):
            self.calls.append(kwargs)
            return True

    class FakeBlockSparseMoe(nn.Module):
        def __init__(self, routed_experts):
            super().__init__()
            self.experts = routed_experts
            self.gate = nn.Linear(1, 2, bias=False)

    class FakeLayer(nn.Module):
        def __init__(self, routed_experts):
            super().__init__()
            self.block_sparse_moe = FakeBlockSparseMoe(routed_experts)

    class FakeInnerModel(nn.Module):
        def __init__(self, routed_experts):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(routed_experts)])

    class FakeConfig:
        tie_word_embeddings = False
        num_local_experts = 2

    class FakeGranite(granitemoe.GraniteMoeForCausalLM):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = FakeConfig()
            self.routed_experts = FakeRoutedExperts()
            self.model = FakeInnerModel(self.routed_experts)

    class FakeSource:
        def __init__(self, catalog):
            self.catalog = catalog
            self.full_reads = []
            self.slice_reads = []
            self.skips = []

        def read_full_cpu(self, name):
            self.full_reads.append(name)
            return torch.ones(catalog.get(name).shape)

        def read_slice_cpu(self, name, source_slices):
            self.slice_reads.append((name, source_slices))
            return torch.ones(1)

        def skip(self, name, reason):
            self.skips.append((name, reason))

    model = FakeGranite()
    source = FakeSource(catalog)
    plan = model.build_weight_plan(catalog)

    assert [entry.local_required for entry in plan.routed_entries] == [
        True,
        True,
        False,
        False,
        True,
        False,
    ]

    loaded = model.load_weights_from_source(source, plan)

    assert source.full_reads == [names[2]]
    assert source.slice_reads == [
        (names[0], (0, slice(0, 2), slice(None))),
        (names[0], (0, slice(2, 4), slice(None))),
        (names[1], (0, slice(None), slice(None))),
    ]
    assert source.skips == [
        (names[0], "non-local routed expert"),
        (names[0], "non-local routed expert"),
        (names[1], "non-local routed expert"),
    ]
    assert "model.layers.0.block_sparse_moe.gate.weight" in loaded
    assert "model.layers.0.block_sparse_moe.experts.w13_weight" in loaded
    assert "model.layers.0.block_sparse_moe.experts.w2_weight" in loaded
    assert [call["shard_id"] for call in model.routed_experts.calls] == [
        "w1",
        "w3",
        "w2",
    ]


def test_granitemoe_shared_source_hook_uses_routed_experts_prefix(monkeypatch):
    calls = []

    def fake_build(model, catalog, **kwargs):
        calls.append(("build", model, catalog, kwargs))
        return "plan"

    def fake_load(model, source, plan):
        calls.append(("load", model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(granitemoeshared, "build_granite_moe_weight_plan", fake_build)
    monkeypatch.setattr(
        granitemoeshared,
        "load_granite_moe_weights_from_source",
        fake_load,
    )

    class FakeConfig:
        tie_word_embeddings = False
        num_local_experts = 2

    class FakeGraniteShared(granitemoeshared.GraniteMoeSharedForCausalLM):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = FakeConfig()

    model = FakeGraniteShared()
    catalog = object()
    source = object()

    assert model.build_weight_plan(catalog) == "plan"
    assert model.load_weights_from_source(source, "plan") == {"loaded"}
    assert calls[0][3]["routed_prefix"] == "routed_experts_"
    assert calls[1] == ("load", model, source, "plan")


def test_granitemoe_hybrid_source_plan_slices_weight_scales_before_read():
    names = [
        "model.layers.0.block_sparse_moe.input_linear.weight_scale",
        "model.layers.0.block_sparse_moe.output_linear.weight_scale",
        "model.layers.0.mixer.A_log",
    ]
    catalog = TensorCatalog(
        [
            TensorMeta("model.safetensors", names[0], torch.float32, [2, 4], 0, 32),
            TensorMeta("model.safetensors", names[1], torch.float32, [2, 3], 32, 24),
            TensorMeta("model.safetensors", names[2], torch.float32, [1], 56, 4),
        ]
    )

    class FakeRoutedExperts:
        layer_name = "model.layers.0.block_sparse_moe.experts"
        routed_experts_w13_weight_scale = object()
        routed_experts_w2_weight_scale = object()
        quant_method = object()

        def __init__(self):
            self.calls = []

        def _map_global_expert_id_to_local_expert_id(self, expert_id):
            return 0 if expert_id == 0 else -1

        def weight_loader(self, **kwargs):
            self.calls.append(kwargs)
            return True

    class FakeBlockSparseMoe(nn.Module):
        def __init__(self, routed_experts):
            super().__init__()
            self.experts = routed_experts

    class FakeMixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.A = nn.Parameter(torch.zeros(1))

    class FakeLayer(nn.Module):
        def __init__(self, routed_experts):
            super().__init__()
            self.block_sparse_moe = FakeBlockSparseMoe(routed_experts)
            self.mixer = FakeMixer()

    class FakeInnerModel(nn.Module):
        def __init__(self, routed_experts):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(routed_experts)])

    class FakeConfig:
        tie_word_embeddings = False
        num_local_experts = 2

    class FakeGraniteHybrid(granitemoehybrid.GraniteMoeHybridForCausalLM):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = FakeConfig()
            self.routed_experts = FakeRoutedExperts()
            self.model = FakeInnerModel(self.routed_experts)

    class FakeSource:
        def __init__(self, catalog):
            self.catalog = catalog
            self.full_reads = []
            self.slice_reads = []
            self.skips = []

        def read_full_cpu(self, name):
            self.full_reads.append(name)
            return torch.ones(self.catalog.get(name).shape)

        def read_slice_cpu(self, name, source_slices):
            self.slice_reads.append((name, source_slices))
            return torch.ones(1)

        def skip(self, name, reason):
            self.skips.append((name, reason))

    model = FakeGraniteHybrid()
    source = FakeSource(catalog)
    plan = model.build_weight_plan(catalog)

    assert [entry.local_required for entry in plan.routed_entries] == [
        True,
        True,
        False,
        False,
        True,
        False,
    ]

    loaded = model.load_weights_from_source(source, plan)

    assert source.full_reads == [names[2]]
    assert source.slice_reads == [
        (names[0], (0, slice(0, 2))),
        (names[0], (0, slice(2, 4))),
        (names[1], (0, slice(None))),
    ]
    assert source.skips == [
        (names[0], "non-local routed expert"),
        (names[0], "non-local routed expert"),
        (names[1], "non-local routed expert"),
    ]
    assert "model.layers.0.mixer.A" in loaded
    assert (
        "model.layers.0.block_sparse_moe.experts."
        "routed_experts_w13_weight_scale"
    ) in loaded
    assert (
        "model.layers.0.block_sparse_moe.experts."
        "routed_experts_w2_weight_scale"
    ) in loaded
    assert [call["shard_id"] for call in model.routed_experts.calls] == [
        "w1",
        "w3",
        "w2",
    ]


def test_granitemoe_hybrid_source_hook_uses_hybrid_options(monkeypatch):
    calls = []

    def fake_build(model, catalog, **kwargs):
        calls.append(("build", model, catalog, kwargs))
        return "plan"

    def fake_load(model, source, plan):
        calls.append(("load", model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(granitemoehybrid, "build_granite_moe_weight_plan", fake_build)
    monkeypatch.setattr(
        granitemoehybrid,
        "load_granite_moe_weights_from_source",
        fake_load,
    )

    class FakeConfig:
        tie_word_embeddings = False
        num_local_experts = 2

    class FakeGraniteHybrid(granitemoehybrid.GraniteMoeHybridForCausalLM):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = FakeConfig()

    model = FakeGraniteHybrid()
    catalog = object()
    source = object()

    assert model.build_weight_plan(catalog) == "plan"
    assert model.load_weights_from_source(source, "plan") == {"loaded"}
    assert calls[0][3]["routed_prefix"] == "routed_experts_"
    assert calls[0][3]["include_weight_scales"] is True
    assert calls[0][3]["map_a_log"] is True
    assert calls[1] == ("load", model, source, "plan")


@pytest.mark.parametrize(
    "model_cls, module",
    [
        (qwen2_moe.Qwen2MoeForCausalLM, qwen2_moe),
        (qwen3_moe.Qwen3MoeForCausalLM, qwen3_moe),
        (qwen3_next.Qwen3NextForCausalLM, qwen3_next),
        (qwen3_5.Qwen3_5MoeForCausalLM, qwen3_5),
        (qwen3_5.Qwen3_5MoeForConditionalGeneration, qwen3_5),
    ],
)
def test_qwen_moe_source_hook_delegates_to_shared_helper(
    monkeypatch, model_cls, module
):
    calls = []

    def fake_build(model, catalog, **kwargs):
        calls.append(("build", model, catalog, kwargs))
        return "plan"

    def fake_load(model, source, plan):
        calls.append(("load", model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(module, "build_qwen_moe_weight_plan", fake_build)
    monkeypatch.setattr(module, "load_qwen_moe_weights_from_source", fake_load)

    class FakeModel(model_cls):
        config = type("FakeConfig", (), {"tie_word_embeddings": False})()

        def __init__(self):
            nn.Module.__init__(self)

        def _uma_weight_mapper(self):
            return None

    model = FakeModel()
    catalog = object()
    source = object()

    assert model.build_weight_plan(catalog) == "plan"
    assert model.load_weights_from_source(source, "plan") == {"loaded"}
    assert calls[0][0] == "build"
    assert calls[0][1] is model
    assert calls[0][2] is catalog
    assert calls[1] == ("load", model, source, "plan")


@pytest.mark.parametrize(
    "model_cls, module, family_name, skip_prefixes",
    [
        (olmoe.OlmoeForCausalLM, olmoe, "OLMoE", None),
        (
            cohere2_moe.Cohere2MoeForCausalLM,
            cohere2_moe,
            "Cohere2 MoE",
            ["lm_head."],
        ),
    ],
)
def test_mlp_experts_moe_source_hook_passes_family_options(
    monkeypatch, model_cls, module, family_name, skip_prefixes
):
    calls = []

    def fake_build(model, catalog, **kwargs):
        calls.append(("build", model, catalog, kwargs))
        return "plan"

    def fake_load(model, source, plan, **kwargs):
        calls.append(("load", model, source, plan, kwargs))
        return {"loaded"}

    monkeypatch.setattr(module, "build_qwen_moe_weight_plan", fake_build)
    monkeypatch.setattr(module, "load_qwen_moe_weights_from_source", fake_load)

    class FakeModel(model_cls):
        config = type("FakeConfig", (), {"tie_word_embeddings": False})()

        def __init__(self):
            nn.Module.__init__(self)

    model = FakeModel()
    catalog = object()
    source = object()

    assert model.build_weight_plan(catalog) == "plan"
    assert model.load_weights_from_source(source, "plan") == {"loaded"}
    assert calls[0][3]["family_name"] == family_name
    assert calls[0][3]["mapper"] is model_cls.hf_to_vllm_mapper
    assert calls[0][3].get("skip_prefixes") == skip_prefixes
    assert calls[1] == ("load", model, source, "plan",
                        {"family_name": family_name})


def test_qwen2_moe_source_hook_passes_hf_mapper_and_tie_skip(monkeypatch):
    calls = []

    def fake_build(model, catalog, **kwargs):
        calls.append(("build", model, catalog, kwargs))
        return "plan"

    def fake_load(model, source, plan):
        calls.append(("load", model, source, plan))
        return {"loaded"}

    monkeypatch.setattr(qwen2_moe, "build_qwen_moe_weight_plan", fake_build)
    monkeypatch.setattr(qwen2_moe, "load_qwen_moe_weights_from_source", fake_load)

    class FakeConfig:
        tie_word_embeddings = True

    class FakeQwen2Moe(qwen2_moe.Qwen2MoeForCausalLM):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = FakeConfig()

    model = FakeQwen2Moe()
    catalog = object()
    source = object()

    assert model.build_weight_plan(catalog) == "plan"
    assert model.load_weights_from_source(source, "plan") == {"loaded"}
    assert calls[0][3]["mapper"] is qwen2_moe.Qwen2MoeForCausalLM.hf_to_vllm_mapper
    assert calls[0][3]["skip_prefixes"] == ["lm_head."]
    assert calls[1] == ("load", model, source, "plan")


def test_phimoe_source_hook_passes_mixtral_family_options(monkeypatch):
    calls = []

    def fake_build(model, catalog, **kwargs):
        calls.append(("build", model, catalog, kwargs))
        return "plan"

    def fake_load(model, source, plan, **kwargs):
        calls.append(("load", model, source, plan, kwargs))
        return {"loaded"}

    monkeypatch.setattr(phimoe, "build_mixtral_moe_weight_plan", fake_build)
    monkeypatch.setattr(phimoe, "load_mixtral_moe_weights_from_source", fake_load)

    class FakePhiMoE(phimoe.PhiMoEForCausalLM):
        def __init__(self):
            nn.Module.__init__(self)

    model = FakePhiMoE()
    catalog = object()
    source = object()

    assert model.build_weight_plan(catalog) == "plan"
    assert model.load_weights_from_source(source, "plan") == {"loaded"}
    assert calls[0][3]["family_name"] == "PhiMoE"
    assert calls[0][3]["mapper"] is phimoe.PhiMoEForCausalLM.hf_to_vllm_mapper
    assert calls[1] == ("load", model, source, "plan",
                        {"family_name": "PhiMoE"})


def test_qwen_moe_source_plan_handles_nested_language_model_before_read():
    name = "model.language_model.model.layers.0.mlp.experts.1.up_proj.weight"
    catalog = TensorCatalog(
        [TensorMeta("model.safetensors", name, torch.float32, [1], 0, 4)]
    )

    class FakeRoutedExperts:
        layer_name = "language_model.model.layers.0.mlp.experts.routed_experts"
        w13_weight = object()
        quant_method = object()

        def __init__(self):
            self.calls = []

        def _map_global_expert_id_to_local_expert_id(self, _expert_id):
            return -1

        def weight_loader(self, **kwargs):
            self.calls.append(kwargs)
            return True

    class FakeExperts:
        def __init__(self, routed_experts):
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

    class FakeLanguageModel:
        def __init__(self, routed_experts):
            self.model = FakeInnerModel(routed_experts)

    class FakeWrapper:
        def __init__(self):
            self.routed_experts = FakeRoutedExperts()
            self.language_model = FakeLanguageModel(self.routed_experts)

        def children(self):
            return []

    class FakeSource:
        def __init__(self, catalog):
            self.catalog = catalog
            self.reads = []
            self.skips = []

        def read_full_cpu(self, name):
            self.reads.append(name)
            return torch.ones(1)

        def skip(self, name, reason):
            self.skips.append((name, reason))

    model = FakeWrapper()
    source = FakeSource(catalog)
    plan = qwen3_5.build_qwen_moe_weight_plan(model, catalog)

    assert len(plan.auto_plan.entries) == 0
    assert [entry.local_required for entry in plan.routed_entries] == [False]
    assert qwen3_5.load_qwen_moe_weights_from_source(model, source, plan) == set()
    assert source.reads == []
    assert source.skips == [(name, "non-local routed expert")]


def test_qwen_moe_source_plan_rejects_missing_routed_experts():
    name = "model.layers.0.mlp.experts.0.down_proj.weight"
    catalog = TensorCatalog(
        [TensorMeta("model.safetensors", name, torch.float32, [1], 0, 4)]
    )

    class FakeDenseMLP:
        pass

    class FakeLayer:
        mlp = FakeDenseMLP()

    class FakeInnerModel:
        layers = [FakeLayer()]

    class FakeModel:
        model = FakeInnerModel()

        def children(self):
            return []

    with pytest.raises(RuntimeError, match="has no experts module"):
        qwen3_5.build_qwen_moe_weight_plan(FakeModel(), catalog)


def test_qwen_moe_source_plan_skips_pp_missing_layer_before_read():
    name = "model.layers.0.mlp.experts.0.down_proj.weight"
    catalog = TensorCatalog(
        [TensorMeta("model.safetensors", name, torch.float32, [1], 0, 4)]
    )

    class FakeInnerModel:
        layers = [PPMissingLayer()]

    class FakeModel:
        model = FakeInnerModel()

        def children(self):
            return []

    class FakeSource:
        def __init__(self):
            self.reads = []
            self.skips = []

        def read_full_cpu(self, name):
            self.reads.append(name)
            return torch.ones(1)

        def skip(self, name, reason):
            self.skips.append((name, reason))

    model = FakeModel()
    source = FakeSource()
    plan = qwen3_5.build_qwen_moe_weight_plan(model, catalog)

    assert [entry.local_required for entry in plan.routed_entries] == [False]
    assert qwen3_5.load_qwen_moe_weights_from_source(model, source, plan) == set()
    assert source.reads == []
    assert source.skips == [(name, "pipeline-missing routed expert layer")]


def test_uma_odirect_model_source_hook_requires_both_methods(tmp_path, monkeypatch):
    metadata = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 4)

    class IncompleteFakeModel:
        def build_weight_plan(self, _catalog):
            return []

    class FakeModelConfig:
        model = str(tmp_path)
        model_weights = None

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)

    with pytest.raises(RuntimeError, match="must implement both"):
        loader.load_weights(IncompleteFakeModel(), FakeModelConfig())


def test_uma_odirect_safetensors_staging_tensor_is_cpu(tmp_path, monkeypatch):
    metadata = {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 4)

    seen_device_types: list[str] = []
    original_read = UmaODirectSafetensorsModelLoader._read_records

    def fake_read_records(self, files):
        records = original_read(self, files)
        assert len(records) == 1
        return records

    def fake_read_record_into_tensor(self, tensor, offset, size, gate=None):
        seen_device_types.append(tensor.device.type)
        tensor.zero_()
        if gate is not None:
            gate(size)

    monkeypatch.setattr(
        UmaODirectSafetensorsModelLoader,
        "_read_records",
        fake_read_records,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.uma_odirect_safetensors_loader."
        "_ODirectFile.read_record_into_tensor",
        fake_read_record_into_tensor,
    )

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    tensors = list(loader._get_weights_iterator(str(tmp_path)))
    assert tensors[0][0] == "a"
    assert tensors[0][1].device.type == "cpu"
    assert seen_device_types == ["cpu"]
