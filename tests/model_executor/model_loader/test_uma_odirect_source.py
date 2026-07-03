# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the O_DIRECT source read-window handle cache."""

import types

import torch

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
