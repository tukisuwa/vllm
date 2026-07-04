# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the O_DIRECT source read-window handle cache."""

import errno
import json
import os
import socket
import threading
import time
import types

import pytest
import torch

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import uma_odirect_safetensors_loader as L


class _FakeODirectFile:
    def __init__(self, path, chunk_size, alignment, window_size):
        self.path = path
        self.fd = 1
        self.closed = False
        self.direct_reads = 1
        self.window_loads = 1
        self.window_hits = 0
        self.bytes_read = 100
        self.bytes_copied = 10

    def close(self):
        self.closed = True
        self.fd = -1


def _make_source(monkeypatch):
    monkeypatch.setattr(L, "_ODirectFile", _FakeODirectFile)
    source = object.__new__(L.ODirectSafetensorsWeightSource)
    source._loader = types.SimpleNamespace(
        _chunk_size=4096,
        _alignment=4096,
        _window_size=8192,
    )
    source._stats = L._SourceReadStats()
    source._bytes_since_gate = 0
    source._open_file_handle = None
    source._open_file_path = None
    source._expected_read_summary = None
    return source


def _write_single_tensor_safetensors(path, name: str, tensor: torch.Tensor) -> None:
    _write_tensors_safetensors(path, {name: tensor})


def _write_tensors_safetensors(path, tensors: dict[str, torch.Tensor]) -> None:
    metadata = {}
    payloads = []
    offset = 0
    for name, tensor in tensors.items():
        payload = tensor.contiguous().numpy().tobytes()
        metadata[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    metadata_raw = json.dumps(metadata).encode("utf-8")
    path.write_bytes(
        len(metadata_raw).to_bytes(8, "little") + metadata_raw + b"".join(payloads)
    )


def _real_source(tmp_path, monkeypatch, name: str, tensor: torch.Tensor):
    return _real_source_many(tmp_path, monkeypatch, {name: tensor})


def _real_source_many(tmp_path, monkeypatch, tensors: dict[str, torch.Tensor]):
    if not hasattr(os, "O_DIRECT"):
        pytest.skip("O_DIRECT is not available on this platform")
    _write_tensors_safetensors(tmp_path / "model.safetensors", tensors)
    loader = L.UmaODirectSafetensorsModelLoader(
        LoadConfig(
            load_format="uma_odirect_safetensors",
            model_loader_extra_config={
                "chunk_size": 4096,
                "window_size": 4096,
                "gate_interval_mib": 1,
                "remote_batch_payload_mib": 2,
            },
        )
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _reason: None)
    return L.ODirectSafetensorsWeightSource(loader, str(tmp_path))


def _remote_read_or_skip(fn):
    try:
        return fn()
    except RuntimeError as exc:
        if "OSError" in str(exc) and (
            "Invalid argument" in str(exc) or "Operation not supported" in str(exc)
        ):
            pytest.skip("test filesystem does not support O_DIRECT")
        raise


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_open_file_reuses_handle_for_same_path(monkeypatch):
    source = _make_source(monkeypatch)

    first = source._open_file("a.safetensors")
    second = source._open_file("a.safetensors")

    assert first is second
    assert first.closed is False
    assert source._stats.files_opened == 1
    # The handle stays open between reads, so per-file counters are folded
    # into the stats only once, at close time.
    assert source._stats.bytes_read == 0

    source.close_files()
    assert first.closed is True
    assert source._stats.bytes_read == 100


def test_open_file_switches_and_closes_previous_handle(monkeypatch):
    source = _make_source(monkeypatch)

    first = source._open_file("a.safetensors")
    second = source._open_file("b.safetensors")

    assert first.closed is True
    assert second.closed is False
    assert source._stats.files_opened == 2
    assert source._stats.bytes_read == 100  # first handle folded on switch

    source.close_files()
    assert second.closed is True
    assert source._stats.bytes_read == 200


def test_close_files_is_idempotent(monkeypatch):
    source = _make_source(monkeypatch)

    source.close_files()
    handle = source._open_file("a.safetensors")
    source.close_files()
    source.close_files()

    assert handle.closed is True
    assert source._stats.files_opened == 1
    assert source._stats.bytes_read == 100


def test_open_file_reopens_after_close(monkeypatch):
    source = _make_source(monkeypatch)

    first = source._open_file("a.safetensors")
    source.close_files()
    second = source._open_file("a.safetensors")

    assert first is not second
    assert second.closed is False
    assert source._stats.files_opened == 2


def test_read_segments_into_cpu_batches_forced_gates(monkeypatch):
    source = _make_source(monkeypatch)
    source.catalog = L.TensorCatalog(
        [
            L.TensorMeta("model.safetensors", "kv", torch.float32, [2, 2], 0, 16),
        ]
    )
    source._loader._gate_interval_bytes = 1024 * 1024
    gates = []
    source._loader._gate_memory = lambda reason: gates.append(reason)

    class FakeReadFile:
        def read_record_into_tensor(self, tensor, offset, size, gate=None):
            assert size == tensor.numel() * tensor.element_size()
            tensor.fill_(offset // 8)
            if gate is not None:
                gate(size)

    monkeypatch.setattr(source, "_open_file", lambda _path: FakeReadFile())
    dst = torch.empty((2, 2), dtype=torch.float32)

    source.read_segments_into_cpu(
        "kv",
        dst,
        (
            L.WeightPlanReadSegment(
                (slice(0, 1), slice(None)),
                (slice(0, 1), slice(None)),
            ),
            L.WeightPlanReadSegment(
                (slice(1, 2), slice(None)),
                (slice(1, 2), slice(None)),
            ),
        ),
    )

    assert gates == [
        "before reading kv[segments]",
        "after reading kv[segments]",
    ]
    assert dst.tolist() == [[0.0, 0.0], [1.0, 1.0]]
    assert source._stats.tensors_read == 2


def test_log_stats_warns_when_actual_reads_exceed_expected(caplog, monkeypatch):
    source = _make_source(monkeypatch)
    source._stats.bytes_read = 121
    source.set_expected_read_summary(
        L.ReadScheduleSummary(
            entries=1,
            required_entries=1,
            read_ranges=1,
            expected_direct_reads=1,
            expected_window_loads=1,
            expected_window_hits=1,
            expected_bytes_read=100,
            payload_bytes=100,
        )
    )

    caplog.set_level("WARNING")
    source.log_stats("test")

    assert "actual read amplification exceeded" in caplog.text
    assert "threshold=1.10x" in caplog.text


def test_real_odirect_file_reads_window_and_fails_closed_on_short_read(tmp_path):
    if not hasattr(os, "O_DIRECT"):
        pytest.skip("O_DIRECT is not available on this platform")

    path = tmp_path / "weights.bin"
    payload = bytes(range(256)) * 32
    path.write_bytes(payload)
    try:
        odirect = L._ODirectFile(
            str(path),
            chunk_size=4096,
            alignment=4096,
            window_size=8192,
        )
    except OSError as exc:
        if exc.errno in {errno.EINVAL, errno.EOPNOTSUPP}:
            pytest.skip("test filesystem does not support O_DIRECT")
        raise

    try:
        tensor = torch.empty(16, dtype=torch.uint8)
        odirect.read_record_into_tensor(tensor, 13, 16)
        assert bytes(tensor.tolist()) == payload[13:29]

        with pytest.raises(EOFError, match="short read did not cover"):
            odirect.read_record_into_tensor(torch.empty(16, dtype=torch.uint8), 8188, 16)
    finally:
        odirect.close()


def test_remote_odirect_weight_source_loopback_reads_real_odirect_payload(
    tmp_path,
    monkeypatch,
):
    name = "transformer.h.0.self_attention.key_value.weight"
    source_tensor = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    owner_source = _real_source(tmp_path, monkeypatch, name, source_tensor)
    token = "loopback-test-token"

    with L.RemoteODirectSafetensorsWeightSourceServer(
        owner_source,
        auth_token=token,
    ) as server:
        host, port = server.address
        remote = L.RemoteODirectSafetensorsWeightSource(
            owner_source.catalog,
            host=host,
            port=port,
            auth_token=token,
        )

        # The remote source intentionally omits the optional direct grouped
        # segment method; grouped segment reads are tunneled through read_many.
        assert getattr(remote, "read_segment_group_into_cpu", None) is None

        full = _remote_read_or_skip(lambda: remote.read_full_cpu(name))
        assert torch.equal(full, source_tensor)

        sliced = remote.read_slice_cpu(name, (slice(2, 5), slice(None)))
        assert torch.equal(sliced, source_tensor[2:5])

        dst = remote.empty_cpu_shape(name, (3, 4))
        remote.read_segments_into_cpu(
            name,
            dst,
            (
                L.WeightPlanReadSegment(
                    (slice(0, 1), slice(None)),
                    (slice(0, 1), slice(None)),
                ),
                L.WeightPlanReadSegment(
                    (slice(3, 4), slice(None)),
                    (slice(1, 2), slice(None)),
                ),
                L.WeightPlanReadSegment(
                    (slice(7, 8), slice(None)),
                    (slice(2, 3), slice(None)),
                ),
            ),
        )
        assert torch.equal(
            dst,
            torch.stack([source_tensor[0], source_tensor[3], source_tensor[7]]),
        )

        remote_stats = remote.stats_snapshot()
        assert remote_stats["tensors_read"] == 3
        assert remote_stats["tensors_read_full"] == 1
        assert remote_stats["tensors_read_sliced"] == 2
        assert remote_stats["remote_stream_bytes_recv"] == (
            source_tensor.numel() + sliced.numel() + dst.numel()
        ) * source_tensor.element_size()

        owner_stats = remote.owner_stats_snapshot()
        assert owner_stats["tensors_read"] == 5
        assert owner_stats["tensors_read_full"] == 1
        assert owner_stats["tensors_read_sliced"] == 4
        assert owner_stats["bytes_read"] > 0
        assert server.connections_accepted == 1


def test_remote_odirect_read_many_batches_real_odirect_payload(
    tmp_path,
    monkeypatch,
):
    tensors = {
        "a.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "b.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        "c.weight": torch.arange(8, dtype=torch.float32),
    }
    owner_source = _real_source_many(tmp_path, monkeypatch, tensors)
    token = "read-many-token"

    with L.RemoteODirectSafetensorsWeightSourceServer(
        owner_source,
        auth_token=token,
    ) as server:
        host, port = server.address
        remote = L.RemoteODirectSafetensorsWeightSource(
            owner_source.catalog,
            host=host,
            port=port,
            auth_token=token,
        )

        loaded = _remote_read_or_skip(
            lambda: remote.read_many_cpu(
                (
                    L._RemoteReadRequest("a.weight"),
                    L._RemoteReadRequest("b.weight", (slice(2, 5), slice(None))),
                    L._RemoteReadRequest("c.weight"),
                )
            )
        )

    assert len(loaded) == 3
    assert torch.equal(loaded[0], tensors["a.weight"])
    assert torch.equal(loaded[1], tensors["b.weight"][2:5])
    assert torch.equal(loaded[2], tensors["c.weight"])
    stats = remote.stats_snapshot()
    assert stats["batch_requests"] == 1
    assert stats["batch_tensors"] == 3
    assert stats["tensors_read"] == 3
    assert stats["tensors_read_full"] == 2
    assert stats["tensors_read_sliced"] == 1
    assert server.connections_accepted == 1


def test_remote_odirect_read_many_owner_rejects_oversized_batch(
    tmp_path,
    monkeypatch,
):
    name = "weight"
    owner_source = _real_source(
        tmp_path,
        monkeypatch,
        name,
        torch.arange(16, dtype=torch.float32),
    )
    with L.RemoteODirectSafetensorsWeightSourceServer(
        owner_source,
        auth_token="owner-token",
        max_batch_payload_bytes=8,
    ) as server:
        host, port = server.address
        remote = L.RemoteODirectSafetensorsWeightSource(
            owner_source.catalog,
            host=host,
            port=port,
            auth_token="owner-token",
        )
        with pytest.raises(RuntimeError, match="payload exceeds limit"):
            remote.read_many_cpu((L._RemoteReadRequest(name),))

    assert owner_source.stats_snapshot()["tensors_read"] == 0


def test_remote_odirect_weight_source_rejects_bad_auth(tmp_path, monkeypatch):
    name = "weight"
    owner_source = _real_source(
        tmp_path,
        monkeypatch,
        name,
        torch.arange(4, dtype=torch.float32),
    )
    with L.RemoteODirectSafetensorsWeightSourceServer(
        owner_source,
        auth_token="owner-token",
    ) as server:
        host, port = server.address
        with pytest.raises(RuntimeError, match="failed authentication"):
            L.RemoteODirectSafetensorsWeightSource(
                owner_source.catalog,
                host=host,
                port=port,
                auth_token="wrong-token",
            )


def test_remote_odirect_rejects_oversized_frame_before_payload_read():
    left, right = socket.socketpair()
    try:
        right.sendall((L._REMOTE_MAX_HEADER_BYTES + 1).to_bytes(8, "big"))
        with pytest.raises(RuntimeError, match="header is too large"):
            L._recv_frame(left)
    finally:
        left.close()
        right.close()

    left, right = socket.socketpair()
    try:
        header = b"{}"
        right.sendall(
            len(header).to_bytes(8, "big")
            + header
            + (1).to_bytes(8, "big")
        )
        with pytest.raises(RuntimeError, match="payload is too large"):
            L._recv_frame(left, max_payload_bytes=0)
    finally:
        left.close()
        right.close()


def test_remote_odirect_owner_source_calls_are_serialized():
    name = "weight"
    catalog = L.TensorCatalog(
        [
            L.TensorMeta("model.safetensors", name, torch.float32, (1,), 0, 4),
        ]
    )

    class SlowSource:
        def __init__(self):
            self.catalog = catalog
            self._loader = types.SimpleNamespace(
                _chunk_size=4096,
                _window_size=4096,
                _alignment=4096,
            )
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def read_full_cpu(self, requested_name):
            assert requested_name == name
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.1)
                return torch.tensor([1.0], dtype=torch.float32)
            finally:
                with self.lock:
                    self.active -= 1

    source = SlowSource()
    with L.RemoteODirectSafetensorsWeightSourceServer(
        source,
        auth_token="owner-token",
    ) as server:
        host, port = server.address
        remote = L.RemoteODirectSafetensorsWeightSource(
            catalog,
            host=host,
            port=port,
            auth_token="owner-token",
        )
        errors = []

        def read_remote():
            try:
                assert remote.read_full_cpu(name).item() == 1.0
            except Exception as exc:  # noqa: BLE001 - test captures thread errors.
                errors.append(exc)

        threads = [threading.Thread(target=read_remote) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert errors == []
    assert source.max_active == 1
    assert server.connections_accepted == 1


def test_remote_odirect_requires_owner_capability():
    name = "weight"
    catalog = L.TensorCatalog(
        [
            L.TensorMeta("model.safetensors", name, torch.float32, (1,), 0, 4),
        ]
    )

    class NoCapabilitySource:
        def __init__(self):
            self.catalog = catalog

    with L.RemoteODirectSafetensorsWeightSourceServer(
        NoCapabilitySource(),
        auth_token="owner-token",
    ) as server:
        host, port = server.address
        with pytest.raises(RuntimeError, match="loader capability"):
            L.RemoteODirectSafetensorsWeightSource(
                catalog,
                host=host,
                port=port,
                auth_token="owner-token",
            )


def test_execute_weight_plan_uses_remote_per_entry_segment_fallback(
    tmp_path,
    monkeypatch,
):
    name = "fused.key_value.weight"
    source_tensor = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    owner_source = _real_source(tmp_path, monkeypatch, name, source_tensor)
    segments = (
        L.WeightPlanReadSegment(
            (slice(1, 2), slice(None)),
            (slice(0, 1), slice(None)),
        ),
        L.WeightPlanReadSegment(
            (slice(6, 7), slice(None)),
            (slice(1, 2), slice(None)),
        ),
    )

    with L.RemoteODirectSafetensorsWeightSourceServer(
        owner_source,
        auth_token="owner-token",
    ) as server:
        host, port = server.address
        remote = L.RemoteODirectSafetensorsWeightSource(
            owner_source.catalog,
            host=host,
            port=port,
            auth_token="owner-token",
        )
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.empty(2, 4))
        plan = L.WeightPlan(
            (
                L.WeightPlanEntry(
                    checkpoint_name=name,
                    target_name="weight",
                    read_segments=segments,
                    staging_shape=(2, 4),
                    read_into_cpu=True,
                ),
            )
        )

        loaded = _remote_read_or_skip(lambda: L.execute_weight_plan(model, remote, plan))

    assert loaded == {"weight"}
    assert torch.equal(
        model.weight.detach(),
        torch.stack([source_tensor[1], source_tensor[6]]),
    )
    assert remote.stats_snapshot()["tensors_read_sliced"] == 1


def test_execute_weight_plan_batches_remote_segment_entries_by_source(
    tmp_path,
    monkeypatch,
):
    name = "fused.qkv.weight"
    source_tensor = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    owner_source = _real_source(tmp_path, monkeypatch, name, source_tensor)
    original_group_read = owner_source.read_segment_group_into_cpu
    group_calls = []

    def record_group_read(group_name, requests):
        group_calls.append((group_name, len(requests)))
        return original_group_read(group_name, requests)

    monkeypatch.setattr(owner_source, "read_segment_group_into_cpu", record_group_read)

    q_segments = (
        L.WeightPlanReadSegment(
            (slice(0, 1), slice(None)),
            (slice(0, 1), slice(None)),
        ),
        L.WeightPlanReadSegment(
            (slice(3, 4), slice(None)),
            (slice(1, 2), slice(None)),
        ),
    )
    k_segments = (
        L.WeightPlanReadSegment(
            (slice(1, 2), slice(None)),
            (slice(0, 1), slice(None)),
        ),
        L.WeightPlanReadSegment(
            (slice(4, 5), slice(None)),
            (slice(1, 2), slice(None)),
        ),
    )
    v_segments = (
        L.WeightPlanReadSegment(
            (slice(2, 3), slice(None)),
            (slice(0, 1), slice(None)),
        ),
        L.WeightPlanReadSegment(
            (slice(5, 6), slice(None)),
            (slice(1, 2), slice(None)),
        ),
    )

    with L.RemoteODirectSafetensorsWeightSourceServer(
        owner_source,
        auth_token="owner-token",
    ) as server:
        host, port = server.address
        remote = L.RemoteODirectSafetensorsWeightSource(
            owner_source.catalog,
            host=host,
            port=port,
            auth_token="owner-token",
        )
        model = torch.nn.Module()
        model.q = torch.nn.Parameter(torch.empty(2, 4))
        model.k = torch.nn.Parameter(torch.empty(2, 4))
        model.v = torch.nn.Parameter(torch.empty(2, 4))
        plan = L.WeightPlan(
            (
                L.WeightPlanEntry(
                    checkpoint_name=name,
                    target_name="q",
                    read_segments=q_segments,
                    staging_shape=(2, 4),
                    read_into_cpu=True,
                ),
                L.WeightPlanEntry(
                    checkpoint_name=name,
                    target_name="k",
                    read_segments=k_segments,
                    staging_shape=(2, 4),
                    read_into_cpu=True,
                ),
                L.WeightPlanEntry(
                    checkpoint_name=name,
                    target_name="v",
                    read_segments=v_segments,
                    staging_shape=(2, 4),
                    read_into_cpu=True,
                ),
            )
        )

        loaded = _remote_read_or_skip(lambda: L.execute_weight_plan(model, remote, plan))

    assert loaded == {"q", "k", "v"}
    assert torch.equal(model.q.detach(), torch.stack([source_tensor[0], source_tensor[3]]))
    assert torch.equal(model.k.detach(), torch.stack([source_tensor[1], source_tensor[4]]))
    assert torch.equal(model.v.detach(), torch.stack([source_tensor[2], source_tensor[5]]))
    assert group_calls == [(name, 3)]
    stats = remote.stats_snapshot()
    assert stats["batch_requests"] == 1
    assert stats["batch_tensors"] == 3
    assert stats["tensors_read_sliced"] == 3
    assert server.connections_accepted == 1


def test_execute_weight_plan_uses_remote_owner_capability_for_schedule(
    tmp_path,
    monkeypatch,
):
    name = "weight"
    tensor = torch.arange(4, dtype=torch.float32)
    owner_source = _real_source(tmp_path, monkeypatch, name, tensor)
    captured = {}
    original_schedule = L.schedule_weight_plan_reads

    def capture_schedule(catalog, plan, *, chunk_size, window_size, alignment):
        captured["chunk_size"] = chunk_size
        captured["window_size"] = window_size
        captured["alignment"] = alignment
        return original_schedule(
            catalog,
            plan,
            chunk_size=chunk_size,
            window_size=window_size,
            alignment=alignment,
        )

    monkeypatch.setattr(L, "schedule_weight_plan_reads", capture_schedule)

    with L.RemoteODirectSafetensorsWeightSourceServer(
        owner_source,
        auth_token="owner-token",
    ) as server:
        host, port = server.address
        remote = L.RemoteODirectSafetensorsWeightSource(
            owner_source.catalog,
            host=host,
            port=port,
            auth_token="owner-token",
        )
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.empty(4))
        loaded = _remote_read_or_skip(
            lambda: L.execute_weight_plan(
                model,
                remote,
                L.WeightPlan((L.WeightPlanEntry(name, "weight"),)),
            )
        )

    assert loaded == {"weight"}
    assert captured == {
        "chunk_size": 4096,
        "window_size": 4096,
        "alignment": 4096,
    }
    assert torch.equal(model.weight.detach(), tensor)


def test_execute_weight_plan_uses_remote_read_many_for_full_reads(
    tmp_path,
    monkeypatch,
):
    tensors = {
        "a.weight": torch.arange(4, dtype=torch.float32),
        "b.weight": torch.arange(4, 8, dtype=torch.float32),
        "c.weight": torch.arange(8, 12, dtype=torch.float32),
    }
    owner_source = _real_source_many(tmp_path, monkeypatch, tensors)

    with L.RemoteODirectSafetensorsWeightSourceServer(
        owner_source,
        auth_token="owner-token",
    ) as server:
        host, port = server.address
        remote = L.RemoteODirectSafetensorsWeightSource(
            owner_source.catalog,
            host=host,
            port=port,
            auth_token="owner-token",
        )
        model = torch.nn.Module()
        model.a = torch.nn.Parameter(torch.empty(4))
        model.b = torch.nn.Parameter(torch.empty(4))
        model.c = torch.nn.Parameter(torch.empty(4))
        plan = L.WeightPlan(
            (
                L.WeightPlanEntry("a.weight", "a"),
                L.WeightPlanEntry("b.weight", "b"),
                L.WeightPlanEntry("c.weight", "c"),
            )
        )

        loaded = _remote_read_or_skip(lambda: L.execute_weight_plan(model, remote, plan))

    assert loaded == {"a", "b", "c"}
    assert torch.equal(model.a.detach(), tensors["a.weight"])
    assert torch.equal(model.b.detach(), tensors["b.weight"])
    assert torch.equal(model.c.detach(), tensors["c.weight"])
    stats = remote.stats_snapshot()
    assert stats["batch_requests"] == 1
    assert stats["batch_tensors"] == 3
    assert stats["tensors_read"] == 3
    assert server.connections_accepted == 1


def test_execute_weight_plan_limits_remote_read_many_item_count(
    tmp_path,
    monkeypatch,
):
    tensors = {
        "a.weight": torch.arange(4, dtype=torch.float32),
        "b.weight": torch.arange(4, 8, dtype=torch.float32),
        "c.weight": torch.arange(8, 12, dtype=torch.float32),
    }
    owner_source = _real_source_many(tmp_path, monkeypatch, tensors)
    monkeypatch.setattr(L, "_REMOTE_MAX_BATCH_ITEMS", 2)

    with L.RemoteODirectSafetensorsWeightSourceServer(
        owner_source,
        auth_token="owner-token",
    ) as server:
        host, port = server.address
        remote = L.RemoteODirectSafetensorsWeightSource(
            owner_source.catalog,
            host=host,
            port=port,
            auth_token="owner-token",
        )
        model = torch.nn.Module()
        model.a = torch.nn.Parameter(torch.empty(4))
        model.b = torch.nn.Parameter(torch.empty(4))
        model.c = torch.nn.Parameter(torch.empty(4))
        plan = L.WeightPlan(
            (
                L.WeightPlanEntry("a.weight", "a"),
                L.WeightPlanEntry("b.weight", "b"),
                L.WeightPlanEntry("c.weight", "c"),
            )
        )

        loaded = _remote_read_or_skip(lambda: L.execute_weight_plan(model, remote, plan))

    assert loaded == {"a", "b", "c"}
    stats = remote.stats_snapshot()
    assert stats["batch_requests"] == 1
    assert stats["batch_tensors"] == 2
    assert stats["tensors_read"] == 3
    assert torch.equal(model.c.detach(), tensors["c.weight"])
    assert server.connections_accepted == 1


def test_uma_odirect_loader_env_wires_owner_and_remote_sources(
    tmp_path,
    monkeypatch,
):
    name = "weight"
    tensor = torch.arange(4, dtype=torch.float32)
    _write_single_tensor_safetensors(tmp_path / "model.safetensors", name, tensor)
    port = _free_tcp_port()
    token = "env-token"

    owner_loader = L.UmaODirectSafetensorsModelLoader(
        LoadConfig(
            load_format="uma_odirect_safetensors",
            model_loader_extra_config={
                "chunk_size": 4096,
                "window_size": 4096,
                "gate_interval_mib": 1,
            },
        )
    )
    monkeypatch.setattr(owner_loader, "_gate_memory", lambda _reason: None)
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_ROLE_ENV, "owner")
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_HOST_ENV, "127.0.0.1")
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_PORT_ENV, str(port))
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_TOKEN_ENV, token)

    owner_source = owner_loader._create_weight_source(str(tmp_path))
    try:
        assert isinstance(owner_source, L.ODirectSafetensorsWeightSource)
        assert owner_loader._remote_owner_server is not None

        remote_loader = L.UmaODirectSafetensorsModelLoader(
            LoadConfig(
                load_format="uma_odirect_safetensors",
                model_loader_extra_config={"remote_batch_payload_mib": 2},
            )
        )
        monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_ROLE_ENV, "remote")
        remote_source = remote_loader._create_weight_source(str(tmp_path / "missing"))
        assert isinstance(remote_source, L.RemoteODirectSafetensorsWeightSource)
        assert remote_source.catalog.names() == owner_source.catalog.names()
        assert remote_source.read_many_max_payload_bytes() == 2 * 1024 * 1024
        loaded = _remote_read_or_skip(lambda: remote_source.read_full_cpu(name))
        assert torch.equal(loaded, tensor)
        remote_loader.download_model(
            types.SimpleNamespace(model=str(tmp_path / "missing"))
        )
    finally:
        if owner_loader._remote_owner_server is not None:
            owner_loader._remote_owner_server.close()


def test_uma_odirect_owner_role_warns_when_host_defaults_to_loopback(
    tmp_path,
    monkeypatch,
    caplog,
):
    name = "weight"
    _write_single_tensor_safetensors(
        tmp_path / "model.safetensors",
        name,
        torch.arange(4, dtype=torch.float32),
    )
    loader = L.UmaODirectSafetensorsModelLoader(
        LoadConfig(
            load_format="uma_odirect_safetensors",
            model_loader_extra_config={
                "chunk_size": 4096,
                "window_size": 4096,
                "gate_interval_mib": 1,
            },
        )
    )
    monkeypatch.setattr(loader, "_gate_memory", lambda _reason: None)
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_ROLE_ENV, "owner")
    monkeypatch.setenv(
        L.UmaODirectSafetensorsModelLoader.REMOTE_PORT_ENV,
        str(_free_tcp_port()),
    )
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_TOKEN_ENV, "token")
    monkeypatch.delenv(L.UmaODirectSafetensorsModelLoader.REMOTE_HOST_ENV, raising=False)

    caplog.set_level("WARNING")
    source = loader._create_weight_source(str(tmp_path))
    try:
        assert isinstance(source, L.ODirectSafetensorsWeightSource)
        assert "binding to loopback 127.0.0.1" in caplog.text
    finally:
        if loader._remote_owner_server is not None:
            loader._remote_owner_server.close()


def test_uma_odirect_remote_port_offset(monkeypatch):
    loader = L.UmaODirectSafetensorsModelLoader(
        LoadConfig(
            load_format="uma_odirect_safetensors",
            model_loader_extra_config={},
        )
    )
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_PORT_ENV, "9000")
    monkeypatch.setenv(
        L.UmaODirectSafetensorsModelLoader.REMOTE_PORT_OFFSET_ENV,
        "3",
    )
    assert loader._remote_port() == 9003

    monkeypatch.setenv(
        L.UmaODirectSafetensorsModelLoader.REMOTE_PORT_OFFSET_ENV,
        "-1",
    )
    with pytest.raises(RuntimeError, match="must be non-negative"):
        loader._remote_port()

    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_PORT_ENV, "65535")
    monkeypatch.setenv(
        L.UmaODirectSafetensorsModelLoader.REMOTE_PORT_OFFSET_ENV,
        "1",
    )
    with pytest.raises(RuntimeError, match="out of range"):
        loader._remote_port()


def test_uma_odirect_remote_topology_manifest(monkeypatch, tmp_path):
    manifest = {
        "version": 1,
        "base_port": 9000,
        "owners": {"0": {"host": "192.0.2.10", "port_offset": 2}},
        "ranks": {
            "0": {"role": "owner", "owner": "0"},
            "1": {"role": "remote", "owner": "0"},
        },
    }
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loader = L.UmaODirectSafetensorsModelLoader(
        LoadConfig(
            load_format="uma_odirect_safetensors",
            model_loader_extra_config={},
        )
    )
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_TOPOLOGY_ENV, str(path))
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_RANK_ENV, "1")
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_TOKEN_ENV, "token")

    resolved = loader._remote_env()

    assert resolved.role == "remote"
    assert resolved.host == "192.0.2.10"
    assert resolved.port == 9002
    assert resolved.token == "token"


def test_uma_odirect_remote_topology_explicit_env_wins(monkeypatch, tmp_path):
    manifest = {
        "version": 1,
        "base_port": 9000,
        "owners": {"0": {"host": "192.0.2.10", "port_offset": 2}},
        "ranks": {"1": {"role": "remote", "owner": "0"}},
    }
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loader = L.UmaODirectSafetensorsModelLoader(
        LoadConfig(
            load_format="uma_odirect_safetensors",
            model_loader_extra_config={},
        )
    )
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_TOPOLOGY_ENV, str(path))
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_RANK_ENV, "1")
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_ROLE_ENV, "owner")
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_HOST_ENV, "127.0.0.1")
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_PORT_ENV, "9100")
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_TOKEN_ENV, "token")

    resolved = loader._remote_env()

    assert resolved.role == "owner"
    assert resolved.host == "127.0.0.1"
    assert resolved.port == 9100


def test_uma_odirect_remote_topology_missing_rank_fails(monkeypatch, tmp_path):
    manifest = {
        "version": 1,
        "base_port": 9000,
        "owners": {"0": {"host": "192.0.2.10", "port_offset": 0}},
        "ranks": {"0": {"role": "owner", "owner": "0"}},
    }
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loader = L.UmaODirectSafetensorsModelLoader(
        LoadConfig(
            load_format="uma_odirect_safetensors",
            model_loader_extra_config={},
        )
    )
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_TOPOLOGY_ENV, str(path))
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_RANK_ENV, "2")
    monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_TOKEN_ENV, "token")

    with pytest.raises(RuntimeError, match="has no rank entry"):
        loader._remote_env()
