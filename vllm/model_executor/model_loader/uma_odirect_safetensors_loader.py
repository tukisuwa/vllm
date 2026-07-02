# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Generator
from dataclasses import dataclass

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.posix_memalign.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
_LIBC.posix_memalign.restype = ctypes.c_int
_LIBC.free.argtypes = [ctypes.c_void_p]
_LIBC.free.restype = None
_LIBC.pread.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_longlong,
]
_LIBC.pread.restype = ctypes.c_ssize_t
_LIBC.memmove.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_LIBC.memmove.restype = ctypes.c_void_p


_DTYPE_MAP: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
_DTYPE_NBYTES: dict[torch.dtype, int] = {
    torch.bool: 1,
    torch.uint8: 1,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
}
if hasattr(torch, "float8_e4m3fn"):
    _DTYPE_MAP["F8_E4M3"] = torch.float8_e4m3fn
    _DTYPE_NBYTES[torch.float8_e4m3fn] = 1
if hasattr(torch, "float8_e5m2"):
    _DTYPE_MAP["F8_E5M2"] = torch.float8_e5m2
    _DTYPE_NBYTES[torch.float8_e5m2] = 1


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            key, raw_value = line.split(":", 1)
            parts = raw_value.strip().split()
            if not parts:
                continue
            value = int(parts[0])
            if len(parts) > 1 and parts[1] == "kB":
                value *= 1024
            values[key] = value
    return values


def _read_memory_psi_avg10() -> tuple[float, float]:
    some_avg10 = 0.0
    full_avg10 = 0.0
    with open("/proc/pressure/memory", encoding="utf-8") as f:
        for line in f:
            fields = dict(field.split("=", 1) for field in line.split()[1:])
            if line.startswith("some "):
                some_avg10 = float(fields["avg10"])
            elif line.startswith("full "):
                full_avg10 = float(fields["avg10"])
    return some_avg10, full_avg10


def _format_gib(value: float) -> str:
    return f"{value / 1024**3:.2f} GiB"


def _round_down(value: int, align: int) -> int:
    return value // align * align


def _round_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def _tensor_nbytes(shape: list[int], dtype: torch.dtype) -> int:
    elements = 1
    for dim in shape:
        elements *= dim
    return elements * _DTYPE_NBYTES[dtype]


def _row_major_strides(shape: list[int]) -> list[int]:
    strides: list[int] = []
    current = 1
    for dim in reversed(shape):
        strides.append(current)
        current *= dim
    return list(reversed(strides))


def _normalize_slice_selection(
    shape: list[int],
    selection: tuple[slice | int, ...],
) -> tuple[int, int, list[int]]:
    """Return byte-independent element offset/count/shape for a contiguous slice.

    Only selections representable as one contiguous row-major range are accepted.
    This is deliberately conservative: unsupported slices fail closed instead of
    reading a full tensor and slicing it in CPU RAM.
    """

    if len(selection) != len(shape):
        raise ValueError(
            f"Slice rank mismatch: got {len(selection)} indices for shape {shape}"
        )

    strides = _row_major_strides(shape)
    first_offset = 0
    last_offset = 0
    output_shape: list[int] = []
    selected_elements = 1

    for dim, stride, item in zip(shape, strides, selection):
        if isinstance(item, bool):
            raise ValueError(f"Boolean indices are not supported: {selection!r}")
        if isinstance(item, int):
            index = item + dim if item < 0 else item
            if index < 0 or index >= dim:
                raise IndexError(
                    f"Index {item} is out of bounds for dimension of size {dim}"
                )
            first_offset += index * stride
            last_offset += index * stride
            continue
        if not isinstance(item, slice):
            raise TypeError(f"Unsupported slice item {item!r}")
        if item.step not in (None, 1):
            raise ValueError(f"Only contiguous step=1 slices are supported: {item!r}")
        start, stop, _step = item.indices(dim)
        length = max(0, stop - start)
        output_shape.append(length)
        selected_elements *= length
        if length == 0:
            continue
        first_offset += start * stride
        last_offset += (stop - 1) * stride

    if selected_elements == 0:
        return first_offset, 0, output_shape

    span_elements = last_offset - first_offset + 1
    if span_elements != selected_elements:
        raise ValueError(
            "Slice is not contiguous in row-major storage and would require "
            f"multiple reads: shape={shape}, selection={selection!r}"
        )
    return first_offset, selected_elements, output_shape


@dataclass(frozen=True)
class TensorMeta:
    """Metadata-only view of one safetensors tensor payload.

    The UMA-safe loader should make skip/slice/placement decisions from this
    catalog before it reads payload bytes.  Keep the fields intentionally close
    to the existing loader internals while the current iterator path is still
    supported.
    """

    file_path: str
    name: str
    dtype: torch.dtype
    shape: list[int]
    offset: int
    size: int


_TensorRecord = TensorMeta


@dataclass(frozen=True)
class WeightPlanEntry:
    checkpoint_name: str
    target_name: str
    required: bool = True
    source_slices: tuple[slice | int, ...] | None = None
    source_is_sharded: bool = False
    shard_id: str | int | None = None
    expert_id: int | None = None
    weight_name: str | None = None
    ignore_missing: bool = False


@dataclass(frozen=True)
class WeightPlan:
    entries: tuple[WeightPlanEntry, ...]

    def __iter__(self):
        return iter(self.entries)


_ROTARY_EMBEDS_UNUSED_WEIGHTS = (
    "rotary_pos_emb.inv_freq",
    "rotary_emb.inv_freq",
    "rotary_emb.cos_cached",
    "rotary_emb.sin_cached",
)


def build_auto_weight_plan_from_catalog(
    catalog: "TensorCatalog",
    *,
    mapper: object | None = None,
    skip_prefixes: list[str] | None = None,
    skip_substrs: list[str] | None = None,
    ignore_unexpected_suffixes: list[str] | None = None,
) -> WeightPlan:
    """Build a name-mapped plan without reading tensor payloads.

    This mirrors the safe subset of AutoWeightsLoader's pre-read decisions:
    prefix/substr skips and optional WeightsMapper name/shard mapping.
    Resolution against actual module parameters is intentionally deferred to
    execute_weight_plan(), where missing targets fail closed.
    """

    prefixes = skip_prefixes or []
    substrs = [*(skip_substrs or []), *_ROTARY_EMBEDS_UNUSED_WEIGHTS]
    ignored_suffixes = ignore_unexpected_suffixes or []
    map_name_with_shard = getattr(mapper, "_map_name_with_shard", None)
    entries: list[WeightPlanEntry] = []
    for name in catalog.names():
        if any(name.startswith(prefix) for prefix in prefixes) or any(
            substr in name for substr in substrs
        ):
            entries.append(
                WeightPlanEntry(
                    checkpoint_name=name,
                    target_name=name,
                    required=False,
                )
            )
            continue

        target_name = name
        shard_id = None
        if callable(map_name_with_shard):
            mapped = map_name_with_shard(name)
            if mapped is None:
                entries.append(
                    WeightPlanEntry(
                        checkpoint_name=name,
                        target_name=name,
                        required=False,
                    )
                )
                continue
            target_name, shard_id = mapped

        entries.append(
            WeightPlanEntry(
                checkpoint_name=name,
                target_name=target_name,
                shard_id=shard_id,
                ignore_missing=any(
                    target_name.endswith(suffix) for suffix in ignored_suffixes
                ),
            )
        )
    return WeightPlan(tuple(entries))


def build_auto_weight_plan_for_module(
    module: nn.Module,
    catalog: "TensorCatalog",
    *,
    mapper: object | None = None,
    skip_prefixes: list[str] | None = None,
    skip_substrs: list[str] | None = None,
) -> WeightPlan:
    """Build an AutoWeightsLoader-like plan for a real vLLM module."""

    ignore_unexpected_suffixes = [".bias"]
    modules = (module, *module.children())
    iterator = (m.quant_config for m in modules if hasattr(m, "quant_config"))
    if quant_config := next(iterator, None):
        cache_scale_mapper = quant_config.get_cache_scale_mapper()
        if cache_scale_mapper is not None:
            mapper = (
                mapper | cache_scale_mapper
                if mapper is not None
                else cache_scale_mapper
            )
        ignore_unexpected_suffixes.extend(quant_config._ignore_unexpected_suffixes)

    return build_auto_weight_plan_from_catalog(
        catalog,
        mapper=mapper,
        skip_prefixes=skip_prefixes,
        skip_substrs=skip_substrs,
        ignore_unexpected_suffixes=ignore_unexpected_suffixes,
    )


def _resolve_attr(root: object, path: str) -> object:
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            raise RuntimeError(f"Cannot resolve weight plan target {path!r}")
        current = getattr(current, part)
    return current


def _infer_output_dim_source_slice(
    param: object,
    record: TensorMeta,
) -> tuple[slice | int, ...] | None:
    """Infer a safe source-side TP row slice for simple output-sharded params."""

    output_dim = getattr(param, "output_dim", None)
    if output_dim != 0:
        return None
    if getattr(param, "is_sharded_weight", False):
        return None
    if getattr(param, "use_bitsandbytes_4bit", False):
        return None
    if getattr(param, "packed_dim", None) == output_dim:
        return None

    param_data = getattr(param, "data", None)
    if param_data is None:
        return None
    param_shape = list(param_data.shape)
    if len(record.shape) != len(param_shape) or not param_shape:
        return None

    tp_rank = getattr(param, "tp_rank", None)
    tp_size = getattr(param, "tp_size", None)
    if not isinstance(tp_rank, int) or not isinstance(tp_size, int):
        return None
    if tp_size <= 1 or tp_rank < 0 or tp_rank >= tp_size:
        return None

    shard_size = param_shape[output_dim]
    if shard_size <= 0:
        return None
    if record.shape[output_dim] != shard_size * tp_size:
        return None
    for dim, (source_size, target_size) in enumerate(zip(record.shape, param_shape)):
        if dim == output_dim:
            continue
        if source_size != target_size:
            return None

    start = tp_rank * shard_size
    return (slice(start, start + shard_size), *([slice(None)] * (len(param_shape) - 1)))


def _call_weight_loader(
    weight_loader: Callable,
    param: object,
    tensor: torch.Tensor,
    *,
    source_is_sharded: bool,
    kwargs: dict[str, object],
) -> None:
    had_attr = hasattr(param, "is_sharded_weight")
    old_value = getattr(param, "is_sharded_weight", None)
    if source_is_sharded:
        setattr(param, "is_sharded_weight", True)
    try:
        weight_loader(param, tensor, **kwargs)
    finally:
        if source_is_sharded:
            if had_attr:
                setattr(param, "is_sharded_weight", old_value)
            else:
                try:
                    delattr(param, "is_sharded_weight")
                except AttributeError:
                    pass


def execute_weight_plan(
    model: nn.Module,
    source: "ODirectSafetensorsWeightSource",
    plan: WeightPlan,
) -> set[str]:
    """Execute a simple model-side WeightPlan with UMA-safe source reads."""

    loaded: set[str] = set()
    for entry in plan:
        if not entry.required:
            source.skip(entry.checkpoint_name, "weight plan marked not required")
            continue

        try:
            param = _resolve_attr(model, entry.target_name)
        except RuntimeError:
            if entry.ignore_missing:
                source.skip(entry.checkpoint_name, "weight plan target is ignored")
                continue
            raise

        source_slices = entry.source_slices
        source_is_sharded = entry.source_is_sharded
        if source_slices is None and entry.shard_id is None:
            source_slices = _infer_output_dim_source_slice(
                param,
                source.catalog.get(entry.checkpoint_name),
            )
            source_is_sharded = source_slices is not None

        if source_slices is None:
            tensor = source.read_full_cpu(entry.checkpoint_name)
        else:
            tensor = source.read_slice_cpu(entry.checkpoint_name, source_slices)

        weight_loader = getattr(param, "weight_loader", None)
        if not callable(weight_loader):
            raise RuntimeError(
                f"Weight plan target {entry.target_name!r} has no weight_loader"
            )

        kwargs = {}
        if entry.shard_id is not None:
            kwargs["shard_id"] = entry.shard_id
        if entry.expert_id is not None:
            kwargs["expert_id"] = entry.expert_id
        if entry.weight_name is not None:
            kwargs["weight_name"] = entry.weight_name
        _call_weight_loader(
            weight_loader,
            param,
            tensor,
            source_is_sharded=source_is_sharded,
            kwargs=kwargs,
        )
        loaded.add(entry.target_name)
    return loaded


class TensorCatalog:
    """Validated, metadata-only catalog of safetensors records."""

    def __init__(self, records: list[TensorMeta]) -> None:
        self._records = tuple(records)
        self._by_name = {record.name: record for record in records}
        if len(self._by_name) != len(records):
            raise RuntimeError("Duplicate tensor names in TensorCatalog")

    @classmethod
    def from_safetensors_files(
        cls,
        files: list[str],
        *,
        metadata_limit_bytes: int,
    ) -> "TensorCatalog":
        records: list[TensorMeta] = []
        seen_names: dict[str, str] = {}
        for path in files:
            file_size = os.path.getsize(path)
            with open(path, "rb", buffering=0) as f:
                raw_size = f.read(8)
                if len(raw_size) != 8:
                    raise RuntimeError(f"Invalid safetensors header in {path}")
                metadata_size = int.from_bytes(raw_size, "little")
                if metadata_size > metadata_limit_bytes:
                    raise RuntimeError(
                        f"Safetensors metadata too large in {path}: "
                        f"{metadata_size} bytes > {metadata_limit_bytes} bytes"
                    )
                if metadata_size > file_size - 8:
                    raise RuntimeError(
                        f"Invalid safetensors metadata size in {path}: "
                        f"{metadata_size} bytes exceeds file payload"
                    )
                metadata_raw = f.read(metadata_size)
                if len(metadata_raw) != metadata_size:
                    raise RuntimeError(f"Short safetensors metadata read in {path}")
            metadata = json.loads(metadata_raw)
            data_start = 8 + metadata_size
            file_ranges: list[tuple[int, int, str]] = []
            for name, info in metadata.items():
                if name == "__metadata__":
                    continue
                if name in seen_names:
                    raise RuntimeError(
                        f"Duplicate safetensors tensor name {name!r}: "
                        f"{seen_names[name]} and {path}"
                    )
                seen_names[name] = path
                if not isinstance(info, dict):
                    raise RuntimeError(f"Invalid safetensors metadata for {name}")
                try:
                    dtype_name = info["dtype"]
                    data_offsets = info["data_offsets"]
                    shape_raw = info["shape"]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Missing safetensors metadata key {exc.args[0]!r} for {name}"
                    ) from exc
                if dtype_name not in _DTYPE_MAP:
                    raise RuntimeError(
                        f"Unsupported safetensors dtype {dtype_name!r} for {name}"
                    )
                dtype = _DTYPE_MAP[dtype_name]
                if not isinstance(data_offsets, list) or len(data_offsets) != 2:
                    raise RuntimeError(
                        f"Invalid safetensors data_offsets for {name}: "
                        f"{data_offsets!r}"
                    )
                if not isinstance(shape_raw, list):
                    raise RuntimeError(
                        f"Invalid safetensors shape for {name}: {shape_raw!r}"
                    )
                start, end = data_offsets
                shape = list(shape_raw)
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or start < 0
                    or end < start
                    or data_start + end > file_size
                ):
                    raise RuntimeError(
                        f"Invalid safetensors byte range for {name}: "
                        f"start={start}, end={end}, file_size={file_size}, "
                        f"data_start={data_start}"
                    )
                if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
                    raise RuntimeError(f"Invalid safetensors shape for {name}: {shape}")
                size = end - start
                expected_size = _tensor_nbytes(shape, dtype)
                if expected_size != size:
                    raise RuntimeError(
                        f"Tensor size mismatch for {name}: metadata has {size} "
                        f"bytes, shape/dtype imply {expected_size} bytes"
                    )
                file_ranges.append((start, end, name))
                records.append(
                    TensorMeta(
                        file_path=path,
                        name=name,
                        dtype=dtype,
                        shape=shape,
                        offset=data_start + start,
                        size=size,
                    )
                )
            file_ranges.sort(key=lambda item: item[0])
            previous_end = 0
            previous_name = ""
            for start, _end, name in file_ranges:
                if start < previous_end:
                    raise RuntimeError(
                        f"Overlapping safetensors data ranges in {path}: "
                        f"{previous_name} ends at {previous_end}, {name} starts at {start}"
                    )
                previous_end = _end
                previous_name = name
        records.sort(key=lambda r: (r.file_path, r.offset))
        return cls(records)

    def records(self) -> tuple[TensorMeta, ...]:
        return self._records

    def names(self) -> tuple[str, ...]:
        return tuple(record.name for record in self._records)

    def has(self, name: str) -> bool:
        return name in self._by_name

    def get(self, name: str) -> TensorMeta:
        return self._by_name[name]

    def total_bytes(self) -> int:
        return sum(record.size for record in self._records)


@dataclass
class _SourceReadStats:
    files_opened: int = 0
    tensors_read: int = 0
    tensors_read_full: int = 0
    tensors_read_sliced: int = 0
    tensors_skipped: int = 0
    direct_reads: int = 0
    window_loads: int = 0
    window_hits: int = 0
    bytes_read: int = 0
    bytes_copied: int = 0
    bytes_tensor_payload: int = 0
    bytes_full_tensor_payload: int = 0
    bytes_sliced_tensor_payload: int = 0
    time_gate: float = 0.0
    time_alloc: float = 0.0
    time_read: float = 0.0

    def collect_file(self, file: "_ODirectFile") -> None:
        self.direct_reads += getattr(file, "direct_reads", 0)
        self.window_loads += getattr(file, "window_loads", 0)
        self.window_hits += getattr(file, "window_hits", 0)
        self.bytes_read += getattr(file, "bytes_read", 0)
        self.bytes_copied += getattr(file, "bytes_copied", 0)

    def log(self, label: str) -> None:
        logger.info(
            "uma_odirect_safetensors source stats (%s): files_opened=%d "
            "tensors_read=%d tensors_read_full=%d tensors_read_sliced=%d "
            "tensors_skipped=%d direct_reads=%d "
            "window_loads=%d window_hits=%d bytes_read=%s bytes_copied=%s "
            "tensor_payload=%s full_payload=%s sliced_payload=%s",
            label,
            self.files_opened,
            self.tensors_read,
            self.tensors_read_full,
            self.tensors_read_sliced,
            self.tensors_skipped,
            self.direct_reads,
            self.window_loads,
            self.window_hits,
            _format_gib(self.bytes_read),
            _format_gib(self.bytes_copied),
            _format_gib(self.bytes_tensor_payload),
            _format_gib(self.bytes_full_tensor_payload),
            _format_gib(self.bytes_sliced_tensor_payload),
        )
        logger.info(
            "uma_odirect_safetensors source timings (%s): gate=%.3fs "
            "alloc=%.3fs read_copy=%.3fs",
            label,
            self.time_gate,
            self.time_alloc,
            self.time_read,
        )

    def snapshot(self) -> dict[str, int | float]:
        return {
            "files_opened": self.files_opened,
            "tensors_read": self.tensors_read,
            "tensors_read_full": self.tensors_read_full,
            "tensors_read_sliced": self.tensors_read_sliced,
            "tensors_skipped": self.tensors_skipped,
            "direct_reads": self.direct_reads,
            "window_loads": self.window_loads,
            "window_hits": self.window_hits,
            "bytes_read": self.bytes_read,
            "bytes_copied": self.bytes_copied,
            "bytes_tensor_payload": self.bytes_tensor_payload,
            "bytes_full_tensor_payload": self.bytes_full_tensor_payload,
            "bytes_sliced_tensor_payload": self.bytes_sliced_tensor_payload,
            "time_gate": self.time_gate,
            "time_alloc": self.time_alloc,
            "time_read": self.time_read,
        }


class ODirectSafetensorsWeightSource:
    """Pull-oriented source for UMA-safe safetensors loading.

    Phase 1 keeps the existing full-tensor iterator behavior, but exposes a
    catalog-first object so model-side plans can inspect metadata before any
    payload bytes are read.
    """

    def __init__(
        self,
        loader: "UmaODirectSafetensorsModelLoader",
        model_or_path: str,
    ) -> None:
        self._loader = loader
        loader._gate_memory("preflight")
        self.files = loader._prepare_files(model_or_path)
        self.catalog = loader._build_catalog(self.files)
        self._stats = _SourceReadStats()
        self._bytes_since_gate = 0

    def iter_full_tensors(
        self,
        direct_consumer: Callable[[TensorMeta, torch.Tensor], bool] | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        yield from self._loader._iter_records(
            self.files,
            self.catalog,
            direct_consumer=direct_consumer,
        )

    def read_full_cpu(self, name: str) -> torch.Tensor:
        record = self.catalog.get(name)
        return self._read_record_cpu(record, sliced=False)

    def read_slice_cpu(
        self,
        name: str,
        source_slices: tuple[slice | int, ...],
    ) -> torch.Tensor:
        record = self.catalog.get(name)
        element_offset, element_count, output_shape = _normalize_slice_selection(
            record.shape,
            source_slices,
        )
        element_size = _DTYPE_NBYTES[record.dtype]
        slice_record = TensorMeta(
            file_path=record.file_path,
            name=f"{record.name}[slice]",
            dtype=record.dtype,
            shape=output_shape,
            offset=record.offset + element_offset * element_size,
            size=element_count * element_size,
        )
        return self._read_record_cpu(slice_record, sliced=True)

    def _read_record_cpu(self, record: TensorMeta, *, sliced: bool) -> torch.Tensor:
        self._maybe_gate(f"before reading {record.name}", force=True)
        with _ODirectFile(
            record.file_path,
            self._loader._chunk_size,
            self._loader._alignment,
            self._loader._window_size,
        ) as odirect_file:
            self._stats.files_opened += 1
            tensor, time_alloc, time_read = self._loader._read_record_tensor(
                record,
                odirect_file,
                self._note_loaded_bytes,
                gate_memory=lambda reason: self._maybe_gate(reason, force=True),
            )
            self._stats.collect_file(odirect_file)
        self._stats.tensors_read += 1
        self._stats.bytes_tensor_payload += record.size
        if sliced:
            self._stats.tensors_read_sliced += 1
            self._stats.bytes_sliced_tensor_payload += record.size
        else:
            self._stats.tensors_read_full += 1
            self._stats.bytes_full_tensor_payload += record.size
        self._stats.time_alloc += time_alloc
        self._stats.time_read += time_read
        self._maybe_gate(f"after reading {record.name}", force=True)
        return tensor

    def skip(self, name: str, reason: str) -> None:
        self._stats.tensors_skipped += 1
        if not self.catalog.has(name):
            logger.debug(
                "uma_odirect_safetensors skipping absent tensor %s reason=%s",
                name,
                reason,
            )
            return
        record = self.catalog.get(name)
        logger.debug(
            "uma_odirect_safetensors skipping tensor %s bytes=%s reason=%s",
            name,
            _format_gib(record.size),
            reason,
        )

    def log_stats(self, label: str) -> None:
        self._stats.log(label)

    def stats_snapshot(self) -> dict[str, int | float]:
        return self._stats.snapshot()

    def _maybe_gate(self, reason: str, force: bool = False) -> None:
        if (
            not force
            and self._loader._gate_interval_bytes > 0
            and self._bytes_since_gate < self._loader._gate_interval_bytes
        ):
            return
        t0 = time.perf_counter()
        self._loader._gate_memory(reason)
        self._stats.time_gate += time.perf_counter() - t0
        self._bytes_since_gate = 0

    def _note_loaded_bytes(self, nbytes: int) -> None:
        self._bytes_since_gate += nbytes
        self._maybe_gate(f"after {self._bytes_since_gate} loaded bytes")


class _ConsumerProfile:
    def __init__(self) -> None:
        self.seconds = 0.0
        self.count = 0
        self.bytes = 0
        self.by_key = defaultdict(lambda: [0, 0, 0.0])

    @staticmethod
    def _bucket(nbytes: int) -> str:
        if nbytes <= 4 * 1024:
            return "<=4KiB"
        if nbytes <= 64 * 1024:
            return "<=64KiB"
        if nbytes <= 1024 * 1024:
            return "<=1MiB"
        if nbytes <= 16 * 1024 * 1024:
            return "<=16MiB"
        return ">16MiB"

    @staticmethod
    def _name_group(name: str) -> str:
        parts = name.split(".")
        parts = ["#" if part.isdigit() else part for part in parts]
        if len(parts) >= 2 and parts[0] == "model" and parts[1] == "layers":
            return ".".join(parts[3:])
        return ".".join(parts[-3:])

    def record(self, record: _TensorRecord, elapsed: float) -> None:
        self.seconds += elapsed
        self.count += 1
        self.bytes += record.size
        key = (
            self._name_group(record.name),
            str(record.dtype),
            self._bucket(record.size),
        )
        item = self.by_key[key]
        item[0] += 1
        item[1] += record.size
        item[2] += elapsed

    def log(self) -> None:
        logger.info(
            "uma_odirect_safetensors consumer profile: total=%.3fs "
            "tensors=%d bytes=%s",
            self.seconds,
            self.count,
            _format_gib(self.bytes),
        )
        top_items = sorted(
            self.by_key.items(), key=lambda item: item[1][2], reverse=True
        )[:16]
        for (name_group, dtype, bucket), (count, nbytes, seconds) in top_items:
            logger.info(
                "uma_odirect_safetensors consumer profile item: "
                "seconds=%.3f count=%d bytes=%s name_group=%s dtype=%s bucket=%s",
                seconds,
                count,
                _format_gib(nbytes),
                name_group,
                dtype,
                bucket,
                )


class _PerExpertMoeDirectLoader:
    """Direct loader for per-expert routed MoE safetensors.

    This is deliberately narrow. It only consumes checkpoint names that match
    the common routed-expert layout:
    ``layers.<n>.mlp.experts.<expert>.{gate,up,down}_proj.*``.
    The actual copy/sharding semantics remain delegated to
    RoutedExperts.weight_loader().
    """

    _PROJ_TO_PARAM = {
        "gate_proj": ("w13_", "w1"),
        "down_proj": ("w2_", "w2"),
        "up_proj": ("w13_", "w3"),
    }
    _MOE_MARKER = ".mlp.experts."

    def __init__(self, model: nn.Module) -> None:
        self._routed_experts_by_layer = self._find_routed_experts(model)
        if not self._routed_experts_by_layer:
            raise RuntimeError(
                "direct_per_expert_moe was enabled, but no routed "
                "experts were found at language_model.model.layers[*].mlp.experts"
            )
        self.count = 0
        self.bytes = 0
        self.seconds = 0.0
        self.skipped_not_local = 0

    @staticmethod
    def _resolve_attr(root: object, path: str) -> object | None:
        current = root
        for part in path.split("."):
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
        return current

    @classmethod
    def _find_routed_experts(cls, model: nn.Module) -> dict[int, object]:
        layers = cls._resolve_attr(model, "language_model.model.layers")
        if layers is None:
            layers = cls._resolve_attr(model, "model.layers")
        if layers is None:
            return {}

        routed_by_layer: dict[int, object] = {}
        for layer_id, layer in enumerate(layers):
            experts = cls._resolve_attr(layer, "mlp.experts.routed_experts")
            if experts is None:
                continue
            if not hasattr(experts, "weight_loader"):
                continue
            routed_by_layer[layer_id] = experts
        return routed_by_layer

    @classmethod
    def _parse_name(cls, name: str) -> tuple[int, int, str, str] | None:
        parts = name.split(".")
        for idx in range(len(parts) - 6):
            if parts[idx] != "layers":
                continue
            if (
                not parts[idx + 1].isdigit()
                or parts[idx + 2] != "mlp"
                or parts[idx + 3] != "experts"
                or not parts[idx + 4].isdigit()
            ):
                continue
            proj_name = parts[idx + 5]
            if proj_name not in cls._PROJ_TO_PARAM:
                continue
            suffix = ".".join(parts[idx + 6 :])
            if not suffix:
                return None
            return int(parts[idx + 1]), int(parts[idx + 4]), proj_name, suffix
        return None

    def __call__(self, record: _TensorRecord, tensor: torch.Tensor) -> bool:
        parsed = self._parse_name(record.name)
        if parsed is None:
            if self._MOE_MARKER in record.name:
                raise RuntimeError(
                    "direct_per_expert_moe was enabled, but this MoE tensor "
                    "does not match the supported per-expert layout: "
                    f"{record.name}"
                )
            return False

        layer_id, expert_id, proj_name, suffix = parsed
        routed_experts = self._routed_experts_by_layer.get(layer_id)
        if routed_experts is None:
            raise RuntimeError(
                "direct_per_expert_moe matched a tensor for layer "
                f"{layer_id}, but no routed experts module exists: {record.name}"
            )

        param_prefix, shard_id = self._PROJ_TO_PARAM[proj_name]
        param_name = f"{param_prefix}{suffix}"
        if not hasattr(routed_experts, param_name):
            raise RuntimeError(
                "direct_per_expert_moe matched a tensor but the target parameter "
                f"{param_name!r} does not exist for {record.name}"
            )

        t0 = time.perf_counter()
        success = routed_experts.weight_loader(
            param=getattr(routed_experts, param_name),
            loaded_weight=tensor,
            weight_name=f"{routed_experts.layer_name}.{param_name}",
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=True,
        )
        self.seconds += time.perf_counter() - t0
        self.count += 1
        self.bytes += record.size
        if not success:
            map_global = getattr(
                routed_experts, "_map_global_expert_id_to_local_expert_id", None
            )
            if not callable(map_global) or map_global(expert_id) != -1:
                raise RuntimeError(
                    "direct_per_expert_moe routed expert weight_loader returned "
                    f"False for a local or unverifiable expert: {record.name}"
                )
            self.skipped_not_local += 1
        return True

    def log(self) -> None:
        logger.info(
            "uma_odirect_safetensors direct_per_expert_moe: tensors=%d "
            "bytes=%s seconds=%.3fs skipped_not_local=%d layers=%d",
            self.count,
            _format_gib(self.bytes),
            self.seconds,
            self.skipped_not_local,
            len(self._routed_experts_by_layer),
        )


_Qwen35MoeDirectLoader = _PerExpertMoeDirectLoader


class _AlignedBuffer:
    def __init__(self, size: int, alignment: int):
        self.size = size
        self.alignment = alignment
        self.ptr = ctypes.c_void_p()
        rc = _LIBC.posix_memalign(ctypes.byref(self.ptr), alignment, size)
        if rc != 0:
            raise OSError(rc, os.strerror(rc))

    def __del__(self) -> None:
        if getattr(self, "ptr", None) and self.ptr.value:
            _LIBC.free(self.ptr)
            self.ptr = ctypes.c_void_p()


class _ODirectFile:
    def __init__(
        self,
        path: str,
        chunk_size: int,
        alignment: int,
        window_size: int,
    ):
        flags = os.O_RDONLY
        if not hasattr(os, "O_DIRECT"):
            raise RuntimeError("O_DIRECT is not available on this platform")
        flags |= os.O_DIRECT
        self.path = path
        self.fd = os.open(path, flags)
        self.size = os.path.getsize(path)
        self.chunk_size = _round_up(chunk_size, alignment)
        self.alignment = alignment
        self.buffer = _AlignedBuffer(self.chunk_size + alignment, alignment)
        self.window_size = max(_round_up(window_size, alignment), self.chunk_size)
        self.window = _AlignedBuffer(self.window_size + alignment, alignment)
        self.window_start = 0
        self.window_valid = 0
        self.direct_reads = 0
        self.window_loads = 0
        self.window_hits = 0
        self.bytes_read = 0
        self.bytes_copied = 0

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "_ODirectFile":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def _pread(self, offset: int, size: int) -> int:
        return self._pread_into(self.buffer.ptr, offset, size)

    def _pread_into(self, ptr: ctypes.c_void_p, offset: int, size: int) -> int:
        ret = _LIBC.pread(self.fd, ptr, size, offset)
        if ret < 0:
            err = ctypes.get_errno()
            raise OSError(
                err,
                f"O_DIRECT pread failed for {self.path} at {offset} "
                f"size {size}: {os.strerror(err)}",
            )
        got = int(ret)
        self.direct_reads += 1
        self.bytes_read += got
        return got

    def _load_window(self, offset: int, size: int) -> None:
        read_start = _round_down(offset, self.alignment)
        required_size = _round_up((offset - read_start) + size, self.alignment)
        read_size = max(self.window_size, required_size)
        if read_size > self.window.size:
            raise RuntimeError(
                f"Internal O_DIRECT window too small: need {read_size}, "
                f"have {self.window.size}"
            )

        got = self._pread_into(self.window.ptr, read_start, read_size)
        if got == 0:
            raise EOFError(
                f"Unexpected EOF reading {self.path}: offset={read_start}, "
                f"size={read_size}"
            )
        if offset + size > read_start + got:
            raise EOFError(
                f"O_DIRECT window short read did not cover requested range for "
                f"{self.path}: requested={offset}+{size}, got={read_start}+{got}"
            )
        self.window_start = read_start
        self.window_valid = got
        self.window_loads += 1

    def _copy_from_window(self, tensor: torch.Tensor, offset: int, size: int) -> bool:
        window_end = self.window_start + self.window_valid
        if offset < self.window_start or offset + size > window_end:
            return False
        _LIBC.memmove(
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(self.window.ptr.value + offset - self.window_start),
            size,
        )
        self.window_hits += 1
        self.bytes_copied += size
        return True

    def read_record_into_tensor(
        self,
        tensor: torch.Tensor,
        offset: int,
        size: int,
        gate: Callable[[int], None] | None = None,
    ) -> None:
        if size <= self.window_size:
            if not self._copy_from_window(tensor, offset, size):
                self._load_window(offset, size)
                if not self._copy_from_window(tensor, offset, size):
                    raise RuntimeError(
                        f"Internal O_DIRECT window miss after loading {self.path}: "
                        f"offset={offset}, size={size}"
                    )
            return

        self.read_into_tensor(tensor, offset, size, gate=gate)

    def read_into_tensor(
        self,
        tensor: torch.Tensor,
        offset: int,
        size: int,
        gate: Callable[[int], None] | None = None,
    ) -> None:
        if size == 0:
            return
        if offset < 0 or size < 0 or offset + size > self.size:
            raise RuntimeError(
                f"Invalid tensor byte range for {self.path}: "
                f"offset={offset}, size={size}, file_size={self.size}"
            )

        dst_base = tensor.data_ptr()
        copied = 0
        while copied < size:
            wanted_offset = offset + copied
            wanted_size = min(size - copied, self.chunk_size)
            read_start = _round_down(wanted_offset, self.alignment)
            read_end = _round_up(wanted_offset + wanted_size, self.alignment)
            read_size = read_end - read_start
            if read_size > self.buffer.size:
                raise RuntimeError(
                    f"Internal O_DIRECT buffer too small: need {read_size}, "
                    f"have {self.buffer.size}"
                )

            got = self._pread(read_start, read_size)
            if got == 0:
                raise EOFError(
                    f"Unexpected EOF reading {self.path}: offset={read_start}, "
                    f"size={read_size}"
                )

            available_start = read_start
            available_end = read_start + got
            copy_start = max(wanted_offset, available_start)
            copy_end = min(wanted_offset + wanted_size, available_end)
            if copy_end <= copy_start:
                raise EOFError(
                    f"O_DIRECT short read did not cover requested range for "
                    f"{self.path}: requested={wanted_offset}+{wanted_size}, "
                    f"got={read_start}+{got}"
                )

            src_offset = copy_start - read_start
            dst_offset = copy_start - offset
            copy_size = copy_end - copy_start
            _LIBC.memmove(
                ctypes.c_void_p(dst_base + dst_offset),
                ctypes.c_void_p(self.buffer.ptr.value + src_offset),
                copy_size,
            )
            self.bytes_copied += copy_size
            copied = dst_offset + copy_size
            if gate is not None:
                gate(copy_size)


class UmaODirectSafetensorsModelLoader(BaseModelLoader):
    """Fail-closed local safetensors loader using Linux O_DIRECT reads.

    This is intentionally narrower than vLLM's general safetensors loaders:
    local paths only, no mmap, no implicit downloads, no buffered fallback, and
    memory/PSI gates between tensors.
    """

    DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024
    DEFAULT_WINDOW_SIZE = 64 * 1024 * 1024
    DEFAULT_ALIGNMENT = 4096
    DEFAULT_METADATA_LIMIT_MIB = 256
    DEFAULT_MIN_AVAILABLE_GIB = 20.0
    DEFAULT_PSI_GATE_SECONDS = 30.0
    DEFAULT_GATE_INTERVAL_MIB = 64
    DEFAULT_ALLOCATION_GATE_MIN_MIB = 16

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError(
                "model_loader_extra_config must be a dict for "
                f"{load_config.load_format}, got {type(extra_config).__name__}"
            )

        allowed_keys = {
            "alignment",
            "chunk_size",
            "window_size",
            "gate_interval_mib",
            "min_available_gib",
            "metadata_limit_mib",
            "allocation_gate_min_mib",
            "psi_gate_seconds",
            "max_swap_gib",
            "direct_per_expert_moe",
            "direct_qwen35_moe",
        }
        unexpected_keys = set(extra_config) - allowed_keys
        if unexpected_keys:
            raise ValueError(
                "Unexpected extra config keys for uma_odirect_safetensors: "
                f"{unexpected_keys}"
            )

        self._alignment = self._get_positive_int(
            extra_config, "alignment", self.DEFAULT_ALIGNMENT
        )
        self._chunk_size = self._get_positive_int(
            extra_config, "chunk_size", self.DEFAULT_CHUNK_SIZE
        )
        self._window_size = self._get_positive_int(
            extra_config, "window_size", self.DEFAULT_WINDOW_SIZE
        )
        self._gate_interval_bytes = (
            self._get_non_negative_float(
                extra_config, "gate_interval_mib", self.DEFAULT_GATE_INTERVAL_MIB
            )
            * 1024
            * 1024
        )
        self._min_available_gib = self._get_non_negative_float(
            extra_config, "min_available_gib", self.DEFAULT_MIN_AVAILABLE_GIB
        )
        self._psi_gate_seconds = self._get_non_negative_float(
            extra_config, "psi_gate_seconds", self.DEFAULT_PSI_GATE_SECONDS
        )
        self._max_swap_gib = self._get_non_negative_float(
            extra_config, "max_swap_gib", 0.0
        )
        self._metadata_limit_bytes = (
            self._get_positive_int(
                extra_config, "metadata_limit_mib", self.DEFAULT_METADATA_LIMIT_MIB
            )
            * 1024
            * 1024
        )
        self._allocation_gate_min_bytes = (
            self._get_non_negative_float(
                extra_config,
                "allocation_gate_min_mib",
                self.DEFAULT_ALLOCATION_GATE_MIN_MIB,
            )
            * 1024
            * 1024
        )
        direct_per_expert_moe = self._get_bool(
            extra_config, "direct_per_expert_moe", False
        )
        direct_qwen35_moe = self._get_bool(
            extra_config, "direct_qwen35_moe", False
        )
        self._direct_per_expert_moe = direct_per_expert_moe or direct_qwen35_moe

        if self._alignment & (self._alignment - 1) != 0:
            raise ValueError("alignment must be a power of two")
        if self._window_size < self._chunk_size:
            raise ValueError("window_size must be greater than or equal to chunk_size")
        if self._gate_interval_bytes > 0:
            if self._chunk_size > self._gate_interval_bytes:
                raise ValueError(
                    "chunk_size must be less than or equal to gate_interval_mib"
                )
            if self._window_size > self._gate_interval_bytes:
                raise ValueError(
                    "window_size must be less than or equal to gate_interval_mib"
                )
        if load_config.safetensors_load_strategy not in (None, "lazy"):
            raise ValueError(
                "uma_odirect_safetensors does not support "
                "safetensors_load_strategy="
                f"{load_config.safetensors_load_strategy!r}; it uses "
                "O_DIRECT and rejects eager/prefetch paths."
            )

    @staticmethod
    def _get_positive_int(config: dict, key: str, default: int) -> int:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer, got {value!r}")
        return value

    @staticmethod
    def _get_non_negative_float(config: dict, key: str, default: float) -> float:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{key} must be a non-negative number, got {value!r}")
        return float(value)

    @staticmethod
    def _get_bool(config: dict, key: str, default: bool) -> bool:
        value = config.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean, got {value!r}")
        return value

    def _gate_memory(self, phase: str) -> None:
        if sys.platform != "linux":
            raise RuntimeError("uma_odirect_safetensors is Linux-only")

        deadline = time.monotonic() + self._psi_gate_seconds
        while True:
            meminfo = _read_meminfo()
            available = meminfo.get("MemAvailable", 0)
            swap_used = meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0)
            some_avg10, full_avg10 = _read_memory_psi_avg10()
            min_available = self._min_available_gib * 1024**3
            max_swap = self._max_swap_gib * 1024**3

            if available < min_available:
                raise RuntimeError(
                    "uma_odirect_safetensors memory gate failed during "
                    f"{phase}: MemAvailable {_format_gib(available)} < "
                    f"{self._min_available_gib:.2f} GiB"
                )
            if swap_used > max_swap:
                raise RuntimeError(
                    "uma_odirect_safetensors swap gate failed during "
                    f"{phase}: swap used {_format_gib(swap_used)} > "
                    f"{self._max_swap_gib:.2f} GiB"
                )
            if some_avg10 == 0.0 and full_avg10 == 0.0:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "uma_odirect_safetensors PSI gate failed during "
                    f"{phase}: memory PSI avg10 some={some_avg10:.2f}, "
                    f"full={full_avg10:.2f}"
                )
            logger.warning(
                "uma_odirect_safetensors waiting for memory PSI to clear "
                "during %s: some=%.2f full=%.2f",
                phase,
                some_avg10,
                full_avg10,
            )
            time.sleep(1.0)

    def _prepare_files(self, model_name_or_path: str) -> list[str]:
        if not os.path.isdir(model_name_or_path):
            raise RuntimeError(
                "uma_odirect_safetensors only accepts a local safetensors "
                f"directory, got {model_name_or_path!r}"
            )

        files = [
            os.path.join(model_name_or_path, name)
            for name in sorted(os.listdir(model_name_or_path))
            if name.endswith(".safetensors")
        ]
        if not files:
            raise RuntimeError(
                f"Cannot find any safetensors model weights in {model_name_or_path}"
            )
        for path in files:
            if os.path.islink(path):
                raise RuntimeError(f"Refusing symlinked safetensors path: {path}")
            if not os.path.isfile(path):
                raise RuntimeError(f"Refusing non-file safetensors path: {path}")
        return files

    def _read_records(self, files: list[str]) -> list[_TensorRecord]:
        return list(self._build_catalog(files).records())

    def _build_catalog(self, files: list[str]) -> TensorCatalog:
        return TensorCatalog.from_safetensors_files(
            files,
            metadata_limit_bytes=self._metadata_limit_bytes,
        )

    def _get_weights_iterator(
        self,
        model_or_path: str,
        direct_consumer: Callable[[_TensorRecord, torch.Tensor], bool] | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        source = ODirectSafetensorsWeightSource(self, model_or_path)
        yield from source.iter_full_tensors(direct_consumer=direct_consumer)

    def _read_record_tensor(
        self,
        record: TensorMeta,
        odirect_file: _ODirectFile,
        note_loaded_bytes: Callable[[int], None],
        gate_memory: Callable[[str], None] | None = None,
    ) -> tuple[torch.Tensor, float, float]:
        gate_memory = gate_memory or self._gate_memory
        force_allocation_gate = record.size >= self._allocation_gate_min_bytes
        if force_allocation_gate:
            gate_memory(f"before allocating {record.name}")
        t0 = time.perf_counter()
        tensor = torch.empty(record.shape, dtype=record.dtype, device="cpu")
        time_alloc = time.perf_counter() - t0
        if force_allocation_gate:
            gate_memory(f"after allocating {record.name}")

        t0 = time.perf_counter()
        odirect_file.read_record_into_tensor(
            tensor,
            record.offset,
            record.size,
            gate=(
                note_loaded_bytes
                if record.size > odirect_file.window_size
                else None
            ),
        )
        time_read = time.perf_counter() - t0
        if record.size <= odirect_file.window_size:
            note_loaded_bytes(record.size)
        return tensor, time_alloc, time_read

    def _read_record_full_cpu(self, record: TensorMeta) -> torch.Tensor:
        bytes_since_gate = 0

        def note_loaded_bytes(nbytes: int) -> None:
            nonlocal bytes_since_gate
            bytes_since_gate += nbytes
            if (
                self._gate_interval_bytes > 0
                and bytes_since_gate >= self._gate_interval_bytes
            ):
                self._gate_memory(f"after {bytes_since_gate} loaded bytes")
                bytes_since_gate = 0

        self._gate_memory(f"before reading {record.name}")
        with _ODirectFile(
            record.file_path,
            self._chunk_size,
            self._alignment,
            self._window_size,
        ) as odirect_file:
            tensor, _time_alloc, _time_read = self._read_record_tensor(
                record,
                odirect_file,
                note_loaded_bytes,
            )
        self._gate_memory(f"after reading {record.name}")
        return tensor

    def _iter_records(
        self,
        files: list[str],
        catalog: TensorCatalog,
        direct_consumer: Callable[[TensorMeta, torch.Tensor], bool] | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        records = catalog.records()
        total_bytes = catalog.total_bytes()
        logger.info(
            "uma_odirect_safetensors using O_DIRECT: files=%d tensors=%d "
            "total=%s chunk_size=%d window_size=%d alignment=%d",
            len(files),
            len(records),
            _format_gib(total_bytes),
            self._chunk_size,
            self._window_size,
            self._alignment,
        )

        current_path: str | None = None
        odirect_file: _ODirectFile | None = None
        files_opened = 0
        direct_reads = 0
        window_loads = 0
        window_hits = 0
        bytes_read = 0
        bytes_copied = 0
        time_gate = 0.0
        time_alloc = 0.0
        time_read = 0.0
        time_consumer = 0.0
        bytes_since_gate = 0
        consumer_profile = (
            _ConsumerProfile()
            if os.environ.get("VLLM_UMA_LOAD_PROFILE", "").lower()
            in ("1", "true", "yes", "on")
            else None
        )

        def collect_stats(file: _ODirectFile) -> None:
            nonlocal direct_reads, window_loads, window_hits, bytes_read, bytes_copied
            direct_reads += file.direct_reads
            window_loads += file.window_loads
            window_hits += file.window_hits
            bytes_read += file.bytes_read
            bytes_copied += file.bytes_copied

        def maybe_gate(reason: str, force: bool = False) -> None:
            nonlocal bytes_since_gate, time_gate
            if (
                not force
                and self._gate_interval_bytes > 0
                and bytes_since_gate < self._gate_interval_bytes
            ):
                return
            t0 = time.perf_counter()
            self._gate_memory(reason)
            time_gate += time.perf_counter() - t0
            bytes_since_gate = 0

        def note_loaded_bytes(nbytes: int) -> None:
            nonlocal bytes_since_gate
            bytes_since_gate += nbytes
            maybe_gate(f"after {bytes_since_gate} loaded bytes")

        try:
            for record in records:
                if record.file_path != current_path:
                    if odirect_file is not None:
                        collect_stats(odirect_file)
                        odirect_file.close()
                    current_path = record.file_path
                    odirect_file = _ODirectFile(
                        current_path,
                        self._chunk_size,
                        self._alignment,
                        self._window_size,
                    )
                    files_opened += 1

                assert odirect_file is not None
                tensor, elapsed_alloc, elapsed_read = self._read_record_tensor(
                    record,
                    odirect_file,
                    note_loaded_bytes,
                    gate_memory=lambda reason: maybe_gate(reason, force=True),
                )
                time_alloc += elapsed_alloc
                time_read += elapsed_read

                t0 = time.perf_counter()
                if direct_consumer is not None and direct_consumer(record, tensor):
                    elapsed_consumer = time.perf_counter() - t0
                else:
                    yield record.name, tensor
                    elapsed_consumer = time.perf_counter() - t0
                time_consumer += elapsed_consumer
                if consumer_profile is not None:
                    consumer_profile.record(record, elapsed_consumer)
        finally:
            if sys.exc_info()[0] is None:
                maybe_gate("final", force=True)
            else:
                try:
                    maybe_gate("final", force=True)
                except Exception:
                    logger.warning(
                        "uma_odirect_safetensors final gate failed while another "
                        "load error was already being raised",
                        exc_info=True,
                    )
            if odirect_file is not None:
                collect_stats(odirect_file)
                odirect_file.close()
            logger.info(
                "uma_odirect_safetensors stats: files_opened=%d "
                "direct_reads=%d window_loads=%d window_hits=%d "
                "bytes_read=%s bytes_copied=%s",
                files_opened,
                direct_reads,
                window_loads,
                window_hits,
                _format_gib(bytes_read),
                _format_gib(bytes_copied),
            )
            logger.info(
                "uma_odirect_safetensors timings: gate=%.3fs alloc=%.3fs "
                "read_copy=%.3fs consumer=%.3fs",
                time_gate,
                time_alloc,
                time_read,
                time_consumer,
            )
            if consumer_profile is not None:
                consumer_profile.log()
            if direct_consumer is not None and hasattr(direct_consumer, "log"):
                direct_consumer.log()

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_files(model_config.model)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        source = ODirectSafetensorsWeightSource(self, model_weights)
        build_weight_plan = getattr(model, "build_weight_plan", None)
        load_weights_from_source = getattr(model, "load_weights_from_source", None)
        if callable(build_weight_plan) or callable(load_weights_from_source):
            if not callable(build_weight_plan) or not callable(load_weights_from_source):
                raise RuntimeError(
                    "Models using UMA-safe source loading must implement both "
                    "build_weight_plan(catalog) and "
                    "load_weights_from_source(source, plan)"
                )
            logger.info(
                "uma_odirect_safetensors using model WeightSource path: %s",
                type(model).__name__,
            )
            plan = build_weight_plan(source.catalog)
            try:
                load_weights_from_source(source, plan)
            finally:
                source.log_stats("model-source")
            return

        logger.info(
            "uma_odirect_safetensors using compatibility iterator path: %s",
            type(model).__name__,
        )
        direct_consumer = (
            _PerExpertMoeDirectLoader(model) if self._direct_per_expert_moe else None
        )
        model.load_weights(source.iter_full_tensors(direct_consumer=direct_consumer))
