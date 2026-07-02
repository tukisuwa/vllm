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
    UmaODirectSafetensorsModelLoader,
    WeightPlan,
    WeightPlanEntry,
    _Qwen35MoeDirectLoader,
    _TensorRecord,
    execute_weight_plan,
)


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
    assert stats["bytes_read"] == 8
    assert stats["bytes_copied"] == 8
    assert stats["bytes_tensor_payload"] == 8


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


def test_uma_odirect_weight_source_rejects_noncontiguous_slice(tmp_path, monkeypatch):
    metadata = {"a": {"dtype": "F32", "shape": [4, 3], "data_offsets": [0, 48]}}
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, metadata, b"\0" * 48)

    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(load_format="uma_odirect_safetensors")
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _phase: None)
    source = ODirectSafetensorsWeightSource(loader, str(tmp_path))

    with pytest.raises(ValueError, match="not contiguous"):
        source.read_slice_cpu("a", (slice(None), slice(1, 3)))


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
    assert source.stats_snapshot()["tensors_skipped"] == 1


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


def test_uma_odirect_safetensors_accepts_generic_direct_moe_flag():
    loader = UmaODirectSafetensorsModelLoader(
        LoadConfig(
            load_format="uma_odirect_safetensors",
            model_loader_extra_config={"direct_per_expert_moe": True},
        )
    )
    assert loader._direct_per_expert_moe is True


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


def test_direct_qwen35_moe_false_requires_nonlocal_expert():
    class FakeRoutedExperts:
        layer_name = "model.layers.0.mlp.experts.routed_experts"
        w13_weight_packed = object()

        def __init__(self, local_expert: bool):
            self.local_expert = local_expert

        def weight_loader(self, **_kwargs):
            return False

        def _map_global_expert_id_to_local_expert_id(self, _expert_id):
            return 0 if self.local_expert else -1

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

    class FakeModel:
        def __init__(self, routed_experts):
            self.language_model = FakeLanguageModel(routed_experts)

    record = _TensorRecord(
        file_path="model.safetensors",
        name="model.language_model.layers.0.mlp.experts.0.gate_proj.weight_packed",
        dtype=torch.float32,
        shape=[1],
        offset=0,
        size=4,
    )

    local_loader = _Qwen35MoeDirectLoader(FakeModel(FakeRoutedExperts(True)))
    with pytest.raises(RuntimeError, match="False for a local"):
        local_loader(record, torch.zeros(1))

    nonlocal_loader = _Qwen35MoeDirectLoader(FakeModel(FakeRoutedExperts(False)))
    assert nonlocal_loader(record, torch.zeros(1)) is True
    assert nonlocal_loader.skipped_not_local == 1
