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
    payload = tensor.contiguous().numpy().tobytes()
    metadata = {
        name: {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [0, len(payload)],
        },
    }
    metadata_raw = json.dumps(metadata).encode("utf-8")
    path.write_bytes(len(metadata_raw).to_bytes(8, "little") + metadata_raw + payload)


def _real_source(tmp_path, monkeypatch, name: str, tensor: torch.Tensor):
    if not hasattr(os, "O_DIRECT"):
        pytest.skip("O_DIRECT is not available on this platform")
    _write_single_tensor_safetensors(tmp_path / "model.safetensors", name, tensor)
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

        # Phase 1 intentionally omits the optional grouped segment method so
        # execute_weight_plan falls back to per-entry read_segments requests.
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
        remote = L.RemoteODirectSafetensorsWeightSource(
            owner_source.catalog,
            host=host,
            port=port,
            auth_token="wrong-token",
        )
        with pytest.raises(RuntimeError, match="failed authentication"):
            remote.read_full_cpu(name)


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
            LoadConfig(load_format="uma_odirect_safetensors")
        )
        monkeypatch.setenv(L.UmaODirectSafetensorsModelLoader.REMOTE_ROLE_ENV, "remote")
        remote_source = remote_loader._create_weight_source(str(tmp_path))
        assert isinstance(remote_source, L.RemoteODirectSafetensorsWeightSource)
        loaded = _remote_read_or_skip(lambda: remote_source.read_full_cpu(name))
        assert torch.equal(loaded, tensor)
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
