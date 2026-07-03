# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import inspect
import math
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Generator
from dataclasses import dataclass, replace

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_plan import (
    _DTYPE_NBYTES,
    ExecutorCapability,
    TensorCatalog,
    TensorMeta,
    ReadScheduleSummary,
    TransformOp,
    WeightPlan,
    WeightPlanBuilder,
    WeightPlanEntry,
    WeightPlanExecutor,
    WeightPlanReadSegment,
    WeightPlanSourceModel,
    WeightPlanSummary,
    _normalize_single_dim_slice_selection,
    _normalize_slice_selection,
    _resolve_attr,
    _TensorRecord,
    _weight_plan_entry_target_shape,
    apply_transform_ops,
    build_auto_weight_plan_for_module,
    build_auto_weight_plan_from_catalog,
    register_weight_transform,
    resolve_weight_plan,
    resolve_weight_plan_source_hooks,
    schedule_weight_plan_reads,
    summarize_weight_plan,
    validate_weight_plan_read_segments,
    verify_loaded_weights,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)


__all__ = [
    "ODirectSafetensorsWeightSource",
    "ExecutorCapability",
    "TensorCatalog",
    "TensorMeta",
    "TransformOp",
    "UmaODirectSafetensorsModelLoader",
    "WeightPlan",
    "WeightPlanBuilder",
    "WeightPlanEntry",
    "WeightPlanExecutor",
    "WeightPlanReadSegment",
    "WeightPlanSourceModel",
    "WeightPlanSummary",
    "apply_transform_ops",
    "build_auto_weight_plan_for_module",
    "build_auto_weight_plan_from_catalog",
    "execute_weight_plan",
    "register_weight_transform",
    "resolve_weight_plan",
    "resolve_weight_plan_source_hooks",
    "schedule_weight_plan_reads",
    "summarize_weight_plan",
    "verify_loaded_weights",
]


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


def _select_contiguous_target_view(
    dst: torch.Tensor,
    target_slices: tuple[slice | int, ...] | None,
    name: str,
) -> torch.Tensor:
    if target_slices is None:
        return dst
    if len(target_slices) != dst.ndim:
        raise RuntimeError(
            f"read_into_cpu target slice rank mismatch for {name}: "
            f"got {len(target_slices)} indices for dst shape {list(dst.shape)}"
        )
    try:
        target = dst[target_slices]
    except Exception as exc:
        raise RuntimeError(
            f"read_into_cpu invalid target slice for {name}: {target_slices!r}"
        ) from exc
    if not isinstance(target, torch.Tensor):
        raise RuntimeError(f"read_into_cpu target slice for {name} is not a tensor")
    if not target.is_contiguous():
        raise RuntimeError(
            f"read_into_cpu target slice for {name} must be contiguous: "
            f"{target_slices!r}"
        )
    return target


def _source_tensor_shape(
    record: TensorMeta,
    source_slices: tuple[slice | int, ...] | None,
) -> list[int]:
    if source_slices is None:
        return list(record.shape)
    try:
        _element_offset, _element_count, output_shape = _normalize_slice_selection(
            record.shape,
            source_slices,
        )
        return output_shape
    except ValueError as exc:
        strided = _normalize_single_dim_slice_selection(record.shape, source_slices)
        if strided is None:
            raise exc
        return strided[4]


def _call_weight_loader(
    weight_loader: Callable,
    param: object,
    tensor: torch.Tensor,
    *,
    source_is_sharded: bool,
    kwargs: dict[str, object],
    entry_name: str,
) -> None:
    had_attr = hasattr(param, "is_sharded_weight")
    old_value = getattr(param, "is_sharded_weight", None)
    if source_is_sharded:
        setattr(param, "is_sharded_weight", True)
    try:
        call_kwargs = dict(kwargs)
        extra_args: list[object] = []
        expects_success = False
        if "shard_id" in call_kwargs or "expert_id" in call_kwargs:
            signature = inspect.signature(weight_loader)
            params = signature.parameters
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in params.values()
            )
            if (
                "shard_id" in call_kwargs
                and "shard_id" not in params
                and not accepts_kwargs
            ):
                # Several vLLM weight_loader_v2 implementations name this
                # argument loaded_shard_id and expect it positionally.  Keep the
                # plan IR field generic while preserving the layer call
                # convention.
                if "loaded_shard_id" in params:
                    extra_args.append(call_kwargs.pop("shard_id"))
            if (
                ("expert_id" in call_kwargs or "shard_id" in call_kwargs)
                and "return_success" in params
            ):
                # Some loaders report refusal through return_success.  The plan
                # already decided this entry is required, so a refusal means the
                # plan and the loader disagree and the load must fail closed.
                call_kwargs["return_success"] = True
                expects_success = True
        result = weight_loader(param, tensor, *extra_args, **call_kwargs)
        if expects_success and not result:
            raise RuntimeError(
                "weight_loader refused a tensor the weight plan marked "
                f"local and required: {entry_name}"
            )
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

    plan = resolve_weight_plan(model, source.catalog, plan)
    summary = summarize_weight_plan(source.catalog, plan)
    loader = getattr(source, "_loader", None)
    schedule = schedule_weight_plan_reads(
        source.catalog,
        plan,
        chunk_size=getattr(
            loader,
            "_chunk_size",
            UmaODirectSafetensorsModelLoader.DEFAULT_CHUNK_SIZE,
        ),
        window_size=getattr(
            loader,
            "_window_size",
            UmaODirectSafetensorsModelLoader.DEFAULT_WINDOW_SIZE,
        ),
        alignment=getattr(
            loader,
            "_alignment",
            UmaODirectSafetensorsModelLoader.DEFAULT_ALIGNMENT,
        ),
    )
    schedule_summary = schedule.summary
    set_expected_read_summary = getattr(source, "set_expected_read_summary", None)
    if callable(set_expected_read_summary):
        set_expected_read_summary(schedule_summary)
    logger.info(
        "uma_odirect_safetensors weight plan: entries=%d required=%d skipped=%d "
        "missing_skipped=%d full_reads=%d sliced_reads=%d read_into=%d "
        "full_payload=%s sliced_payload=%s read_into_payload=%s "
        "skipped_payload=%s total_read_payload=%s",
        summary.entries,
        summary.required_entries,
        summary.skipped_entries,
        summary.missing_skipped_entries,
        summary.full_read_entries,
        summary.sliced_read_entries,
        summary.read_into_entries,
        _format_gib(summary.full_payload_bytes),
        _format_gib(summary.sliced_payload_bytes),
        _format_gib(summary.read_into_payload_bytes),
        _format_gib(summary.skipped_payload_bytes),
        _format_gib(summary.total_read_payload_bytes),
    )
    logger.info(
        "uma_odirect_safetensors read schedule: entries=%d required=%d "
        "read_ranges=%d expected_direct_reads=%d expected_window_loads=%d "
        "expected_window_hits=%d expected_bytes_read=%s payload=%s "
        "expected_read_amplification=%.2fx",
        schedule_summary.entries,
        schedule_summary.required_entries,
        schedule_summary.read_ranges,
        schedule_summary.expected_direct_reads,
        schedule_summary.expected_window_loads,
        schedule_summary.expected_window_hits,
        _format_gib(schedule_summary.expected_bytes_read),
        _format_gib(schedule_summary.payload_bytes),
        schedule_summary.read_amplification,
    )

    loaded: set[str] = set()
    for entry in schedule.plan:
        if not entry.required:
            source.skip(
                entry.checkpoint_name,
                entry.skip_reason or "weight plan marked not required",
            )
            continue

        try:
            param = _resolve_attr(model, entry.target_name)
        except RuntimeError:
            if entry.ignore_missing:
                source.skip(entry.checkpoint_name, "weight plan target is ignored")
                continue
            raise

        weight_loader = getattr(param, "weight_loader", None)
        if not callable(weight_loader):
            if entry.loader_target_name is not None:
                loader_target = _resolve_attr(model, entry.loader_target_name)
                parent_loader = getattr(loader_target, "weight_loader", None)
                if callable(parent_loader):
                    def _parent_weight_loader(
                        param_arg: object,
                        tensor_arg: torch.Tensor,
                        *,
                        return_success: bool | None = None,
                        **kwargs: object,
                    ) -> object:
                        if return_success is not None:
                            kwargs["return_success"] = return_success
                        return parent_loader(
                            param=param_arg,
                            loaded_weight=tensor_arg,
                            **kwargs,
                        )

                    weight_loader = _parent_weight_loader
                else:
                    raise RuntimeError(
                        f"Weight plan loader target {entry.loader_target_name!r} "
                        "has no custom weight_loader"
                    )
            elif (
                entry.shard_id is not None
                or entry.expert_id is not None
                or entry.weight_name is not None
            ):
                raise RuntimeError(
                    f"Weight plan target {entry.target_name!r} requires loader "
                    "metadata but has no custom weight_loader"
                )
            else:
                weight_loader = default_weight_loader

        # The plan was resolved above; executing without further semantic
        # inference keeps the logged summary equal to the actual reads.
        source_slices = entry.source_slices
        source_is_sharded = entry.source_is_sharded
        record = source.catalog.get(entry.checkpoint_name)

        if entry.target_slices is not None and not entry.read_into_cpu:
            raise RuntimeError(
                "WeightPlanEntry.target_slices is only supported with "
                f"read_into_cpu=True: {entry.checkpoint_name}"
            )
        if entry.read_segments is not None:
            validate_weight_plan_read_segments(record, entry)

        if entry.read_segments is not None:
            empty_cpu_shape = getattr(source, "empty_cpu_shape", None)
            if not callable(empty_cpu_shape):
                raise RuntimeError(
                    "WeightSource does not support segmented CPU staging for "
                    f"{entry.checkpoint_name}"
                )
            tensor = empty_cpu_shape(entry.checkpoint_name, entry.staging_shape)
            read_segments_into_cpu = getattr(source, "read_segments_into_cpu", None)
            if callable(read_segments_into_cpu):
                read_segments_into_cpu(
                    entry.checkpoint_name,
                    tensor,
                    entry.read_segments,
                )
            else:
                for segment in entry.read_segments:
                    source_shape = _weight_plan_entry_target_shape(
                        record,
                        segment.source_slices,
                    )
                    target = _select_contiguous_target_view(
                        tensor,
                        segment.target_slices,
                        entry.checkpoint_name,
                    )
                    if tuple(target.shape) != source_shape:
                        raise RuntimeError(
                            "WeightPlanEntry.read_segments target shape mismatch for "
                            f"{entry.checkpoint_name}: target={list(target.shape)}, "
                            f"source_slice={list(source_shape)}"
                        )
                    source.read_into_cpu(
                        entry.checkpoint_name,
                        tensor,
                        source_slices=segment.source_slices,
                        target_slices=segment.target_slices,
                    )
        elif entry.read_into_cpu:
            tensor = source.empty_cpu(
                entry.checkpoint_name,
                source_slices=None
                if entry.target_slices is not None
                else source_slices,
            )
            source.read_into_cpu(
                entry.checkpoint_name,
                tensor,
                source_slices=source_slices,
                target_slices=entry.target_slices,
            )
        elif source_slices is None:
            tensor = source.read_full_cpu(entry.checkpoint_name)
        else:
            tensor = source.read_slice_cpu(entry.checkpoint_name, source_slices)
        if entry.transform_ops:
            tensor = apply_transform_ops(entry.transform_ops, tensor)

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
            entry_name=entry.checkpoint_name,
        )
        loaded.add(entry.target_name)
    return loaded


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
    bytes_skipped_payload: int = 0
    time_gate: float = 0.0
    time_alloc: float = 0.0
    time_read: float = 0.0
    time_consumer: float = 0.0

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
            "tensor_payload=%s full_payload=%s sliced_payload=%s "
            "skipped_payload=%s",
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
            _format_gib(self.bytes_skipped_payload),
        )
        logger.info(
            "uma_odirect_safetensors source timings (%s): gate=%.3fs "
            "alloc=%.3fs read_copy=%.3fs consumer=%.3fs",
            label,
            self.time_gate,
            self.time_alloc,
            self.time_read,
            self.time_consumer,
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
            "bytes_skipped_payload": self.bytes_skipped_payload,
            "time_gate": self.time_gate,
            "time_alloc": self.time_alloc,
            "time_read": self.time_read,
            "time_consumer": self.time_consumer,
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
        self._open_file_handle: "_ODirectFile | None" = None
        self._open_file_path: str | None = None
        self._expected_read_summary: ReadScheduleSummary | None = None

    def _open_file(self, path: str) -> "_ODirectFile":
        """Return an open O_DIRECT handle, keeping the most recent file open.

        The read window lives on the handle, so keeping one handle open across
        plan entries lets adjacent small tensors share window loads instead of
        paying one window-sized read per tensor.  Only the most recent file
        stays open, so fd count and window-buffer residency stay bounded at one.
        """

        handle = self._open_file_handle
        if handle is not None and self._open_file_path == path:
            return handle
        self.close_files()
        handle = _ODirectFile(
            path,
            self._loader._chunk_size,
            self._loader._alignment,
            self._loader._window_size,
        )
        self._open_file_handle = handle
        self._open_file_path = path
        self._stats.files_opened += 1
        return handle

    def close_files(self) -> None:
        """Close the cached O_DIRECT handle and fold its counters into stats."""

        handle = self._open_file_handle
        if handle is None:
            return
        self._open_file_handle = None
        self._open_file_path = None
        self._stats.collect_file(handle)
        close = getattr(handle, "close", None)
        if close is not None:
            close()

    def iter_full_tensors(
        self,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        records = self.catalog.records()
        logger.info(
            "uma_odirect_safetensors using O_DIRECT source iterator: files=%d "
            "tensors=%d total=%s chunk_size=%d window_size=%d alignment=%d",
            len(self.files),
            len(records),
            _format_gib(self.catalog.total_bytes()),
            self._loader._chunk_size,
            self._loader._window_size,
            self._loader._alignment,
        )
        consumer_profile = (
            _ConsumerProfile()
            if os.environ.get("VLLM_UMA_LOAD_PROFILE", "").lower()
            in ("1", "true", "yes", "on")
            else None
        )

        try:
            for record in records:
                odirect_file = self._open_file(record.file_path)
                tensor, time_alloc, time_read = self._read_record_tensor(
                    record,
                    odirect_file,
                )
                self._stats.tensors_read += 1
                self._stats.tensors_read_full += 1
                self._stats.bytes_tensor_payload += record.size
                self._stats.bytes_full_tensor_payload += record.size
                self._stats.time_alloc += time_alloc
                self._stats.time_read += time_read

                t0 = time.perf_counter()
                yield record.name, tensor
                elapsed_consumer = time.perf_counter() - t0
                self._stats.time_consumer += elapsed_consumer
                if consumer_profile is not None:
                    consumer_profile.record(record, elapsed_consumer)
        finally:
            if sys.exc_info()[0] is None:
                self._maybe_gate("final", force=True)
            else:
                try:
                    self._maybe_gate("final", force=True)
                except Exception:
                    logger.warning(
                        "uma_odirect_safetensors final source gate failed while "
                        "another load error was already being raised",
                        exc_info=True,
                    )
            self.close_files()
            if consumer_profile is not None:
                consumer_profile.log()

    def read_full_cpu(self, name: str) -> torch.Tensor:
        record = self.catalog.get(name)
        return self._read_record_cpu(record, sliced=False)

    def read_slice_cpu(
        self,
        name: str,
        source_slices: tuple[slice | int, ...],
    ) -> torch.Tensor:
        record = self.catalog.get(name)
        try:
            element_offset, element_count, output_shape = _normalize_slice_selection(
                record.shape,
                source_slices,
            )
        except ValueError as exc:
            strided = _normalize_single_dim_slice_selection(
                record.shape,
                source_slices,
            )
            if strided is None:
                raise exc
            return self._read_strided_slice_cpu(record, strided)

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

    def empty_cpu(
        self,
        name: str,
        *,
        source_slices: tuple[slice | int, ...] | None = None,
    ) -> torch.Tensor:
        record = self.catalog.get(name)
        shape = _source_tensor_shape(record, source_slices)
        return self.empty_cpu_shape(name, tuple(shape))

    def empty_cpu_shape(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        record = self.catalog.get(name)
        if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
            raise RuntimeError(f"Invalid CPU staging shape for {name}: {shape!r}")
        nbytes = math.prod(shape) * _DTYPE_NBYTES[record.dtype]
        force_allocation_gate = nbytes >= self._loader._allocation_gate_min_bytes
        if force_allocation_gate:
            self._maybe_gate(f"before allocating {name}", force=True)
        t0 = time.perf_counter()
        tensor = torch.empty(list(shape), dtype=record.dtype, device="cpu")
        self._stats.time_alloc += time.perf_counter() - t0
        if force_allocation_gate:
            self._maybe_gate(f"after allocating {name}", force=True)
        return tensor

    def read_into_cpu(
        self,
        name: str,
        dst: torch.Tensor,
        *,
        source_slices: tuple[slice | int, ...] | None = None,
        target_slices: tuple[slice | int, ...] | None = None,
        force_gate: bool = True,
    ) -> None:
        record = self.catalog.get(name)
        if dst.device.type != "cpu":
            raise RuntimeError(
                f"read_into_cpu requires a CPU destination for {name}, "
                f"got {dst.device}"
            )
        if dst.dtype != record.dtype:
            raise RuntimeError(
                f"read_into_cpu dtype mismatch for {name}: "
                f"dst={dst.dtype}, source={record.dtype}"
            )
        if not dst.is_contiguous():
            raise RuntimeError(f"read_into_cpu requires a contiguous dst for {name}")

        target = _select_contiguous_target_view(dst, target_slices, name)

        if source_slices is None:
            if tuple(target.shape) != record.shape:
                raise RuntimeError(
                    f"read_into_cpu shape mismatch for {name}: "
                    f"dst={list(target.shape)}, source={record.shape}"
                )
            self._read_record_into_cpu(
                record,
                target,
                sliced=False,
                force_gate=force_gate,
            )
            return

        try:
            element_offset, element_count, output_shape = _normalize_slice_selection(
                record.shape,
                source_slices,
            )
        except ValueError as exc:
            strided = _normalize_single_dim_slice_selection(
                record.shape,
                source_slices,
            )
            if strided is None:
                raise exc
            if list(target.shape) != strided[4]:
                raise RuntimeError(
                    f"read_into_cpu shape mismatch for {name}: "
                    f"dst={list(target.shape)}, source_slice={strided[4]}"
                )
            self._read_strided_slice_into_cpu(
                record,
                strided,
                target,
                force_gate=force_gate,
            )
            return

        if list(target.shape) != output_shape:
            raise RuntimeError(
                f"read_into_cpu shape mismatch for {name}: "
                f"dst={list(target.shape)}, source_slice={output_shape}"
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
        self._read_record_into_cpu(
            slice_record,
            target,
            sliced=True,
            force_gate=force_gate,
        )

    def read_segments_into_cpu(
        self,
        name: str,
        dst: torch.Tensor,
        segments: tuple[WeightPlanReadSegment, ...],
    ) -> None:
        self._maybe_gate(f"before reading {name}[segments]", force=True)
        for segment in segments:
            self.read_into_cpu(
                name,
                dst,
                source_slices=segment.source_slices,
                target_slices=segment.target_slices,
                force_gate=False,
            )
        self._maybe_gate(f"after reading {name}[segments]", force=True)

    def _read_strided_slice_cpu(
        self,
        record: TensorMeta,
        strided: tuple[int, int, int, int, list[int]],
    ) -> torch.Tensor:
        outer_count, source_dim, start, length, output_shape = strided
        element_size = _DTYPE_NBYTES[record.dtype]
        # Recompute inner_count from the sliced dimension instead of relying on
        # output rank equivalence. This keeps the byte math tied to the source
        # layout while the returned tensor uses output_shape.
        partial_dim = next(
            idx
            for idx, (src, out) in enumerate(zip(record.shape, output_shape))
            if src != out
        )
        inner_count = math.prod(record.shape[partial_dim + 1 :])
        segment_elements = length * inner_count
        segment_bytes = segment_elements * element_size
        if segment_bytes <= 0:
            raise RuntimeError(f"Invalid empty strided slice for {record.name}")

        total_elements = math.prod(output_shape)
        expected_elements = outer_count * segment_elements
        if total_elements != expected_elements:
            raise RuntimeError(
                f"Internal strided slice shape mismatch for {record.name}: "
                f"output={output_shape}, expected_elements={expected_elements}"
            )

        self._maybe_gate(f"before reading {record.name}[strided-slice]", force=True)
        t0 = time.perf_counter()
        tensor = torch.empty(output_shape, dtype=record.dtype, device="cpu")
        time_alloc = time.perf_counter() - t0
        self._maybe_gate(f"after allocating {record.name}[strided-slice]", force=True)

        flat = tensor.reshape(-1)
        t1 = time.perf_counter()
        odirect_file = self._open_file(record.file_path)
        for outer_idx in range(outer_count):
            source_element_offset = (
                outer_idx * source_dim * inner_count + start * inner_count
            )
            target_element_offset = outer_idx * segment_elements
            target_view = flat.narrow(0, target_element_offset, segment_elements)
            odirect_file.read_record_into_tensor(
                target_view,
                record.offset + source_element_offset * element_size,
                segment_bytes,
                gate=self._note_loaded_bytes,
            )
        time_read = time.perf_counter() - t1

        selected_bytes = total_elements * element_size
        self._stats.tensors_read += 1
        self._stats.tensors_read_sliced += 1
        self._stats.bytes_tensor_payload += selected_bytes
        self._stats.bytes_sliced_tensor_payload += selected_bytes
        self._stats.time_alloc += time_alloc
        self._stats.time_read += time_read
        self._maybe_gate(f"after reading {record.name}[strided-slice]", force=True)
        return tensor

    def _read_strided_slice_into_cpu(
        self,
        record: TensorMeta,
        strided: tuple[int, int, int, int, list[int]],
        dst: torch.Tensor,
        *,
        force_gate: bool = True,
    ) -> None:
        outer_count, source_dim, start, length, output_shape = strided
        element_size = _DTYPE_NBYTES[record.dtype]
        partial_dim = next(
            idx
            for idx, (src, out) in enumerate(zip(record.shape, output_shape))
            if src != out
        )
        inner_count = math.prod(record.shape[partial_dim + 1 :])
        segment_elements = length * inner_count
        segment_bytes = segment_elements * element_size
        if segment_bytes <= 0:
            raise RuntimeError(f"Invalid empty strided slice for {record.name}")

        total_elements = math.prod(output_shape)
        expected_elements = outer_count * segment_elements
        if total_elements != expected_elements:
            raise RuntimeError(
                f"Internal strided slice shape mismatch for {record.name}: "
                f"output={output_shape}, expected_elements={expected_elements}"
            )

        if force_gate:
            self._maybe_gate(f"before reading {record.name}[strided-slice]", force=True)
        flat = dst.reshape(-1)
        t0 = time.perf_counter()
        odirect_file = self._open_file(record.file_path)
        for outer_idx in range(outer_count):
            source_element_offset = (
                outer_idx * source_dim * inner_count + start * inner_count
            )
            target_element_offset = outer_idx * segment_elements
            target_view = flat.narrow(0, target_element_offset, segment_elements)
            odirect_file.read_record_into_tensor(
                target_view,
                record.offset + source_element_offset * element_size,
                segment_bytes,
                gate=self._note_loaded_bytes,
            )
        time_read = time.perf_counter() - t0

        selected_bytes = total_elements * element_size
        self._stats.tensors_read += 1
        self._stats.tensors_read_sliced += 1
        self._stats.bytes_tensor_payload += selected_bytes
        self._stats.bytes_sliced_tensor_payload += selected_bytes
        self._stats.time_read += time_read
        if force_gate:
            self._maybe_gate(f"after reading {record.name}[strided-slice]", force=True)

    def _read_record_cpu(self, record: TensorMeta, *, sliced: bool) -> torch.Tensor:
        self._maybe_gate(f"before reading {record.name}", force=True)
        odirect_file = self._open_file(record.file_path)
        tensor, time_alloc, time_read = self._read_record_tensor(
            record,
            odirect_file,
        )
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

    def _read_record_tensor(
        self,
        record: TensorMeta,
        odirect_file: "_ODirectFile",
    ) -> tuple[torch.Tensor, float, float]:
        force_allocation_gate = record.size >= self._loader._allocation_gate_min_bytes
        if force_allocation_gate:
            self._maybe_gate(f"before allocating {record.name}", force=True)
        t0 = time.perf_counter()
        tensor = torch.empty(record.shape, dtype=record.dtype, device="cpu")
        time_alloc = time.perf_counter() - t0
        if force_allocation_gate:
            self._maybe_gate(f"after allocating {record.name}", force=True)

        t0 = time.perf_counter()
        odirect_file.read_record_into_tensor(
            tensor,
            record.offset,
            record.size,
            gate=(
                self._note_loaded_bytes
                if record.size > odirect_file.window_size
                else None
            ),
        )
        time_read = time.perf_counter() - t0
        if record.size <= odirect_file.window_size:
            self._note_loaded_bytes(record.size)
        return tensor, time_alloc, time_read

    def _read_record_into_cpu(
        self,
        record: TensorMeta,
        dst: torch.Tensor,
        *,
        sliced: bool,
        force_gate: bool = True,
    ) -> None:
        if force_gate:
            self._maybe_gate(f"before reading {record.name}", force=True)
        if tuple(dst.shape) != record.shape:
            raise RuntimeError(
                f"Destination shape mismatch for {record.name}: "
                f"dst={list(dst.shape)}, source={record.shape}"
            )
        odirect_file = self._open_file(record.file_path)
        t0 = time.perf_counter()
        odirect_file.read_record_into_tensor(
            dst,
            record.offset,
            record.size,
            gate=self._note_loaded_bytes,
        )
        time_read = time.perf_counter() - t0
        self._stats.tensors_read += 1
        self._stats.bytes_tensor_payload += record.size
        if sliced:
            self._stats.tensors_read_sliced += 1
            self._stats.bytes_sliced_tensor_payload += record.size
        else:
            self._stats.tensors_read_full += 1
            self._stats.bytes_full_tensor_payload += record.size
        self._stats.time_read += time_read
        if force_gate:
            self._maybe_gate(f"after reading {record.name}", force=True)

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
        self._stats.bytes_skipped_payload += record.size
        logger.debug(
            "uma_odirect_safetensors skipping tensor %s bytes=%s reason=%s",
            name,
            _format_gib(record.size),
            reason,
        )

    def log_stats(self, label: str) -> None:
        self._stats.log(label)
        expected = self._expected_read_summary
        if expected is None or expected.expected_bytes_read <= 0:
            return
        threshold = expected.expected_bytes_read * 1.10
        if self._stats.bytes_read <= threshold:
            return
        logger.warning(
            "uma_odirect_safetensors actual read amplification exceeded "
            "schedule expectation (%s): actual_bytes_read=%s "
            "expected_bytes_read=%s ratio=%.2fx threshold=1.10x",
            label,
            _format_gib(self._stats.bytes_read),
            _format_gib(expected.expected_bytes_read),
            self._stats.bytes_read / expected.expected_bytes_read,
        )

    def set_expected_read_summary(self, summary: ReadScheduleSummary) -> None:
        self._expected_read_summary = summary

    def stats_snapshot(self) -> dict[str, int | float]:
        stats = replace(self._stats)
        if self._open_file_handle is not None:
            stats.collect_file(self._open_file_handle)
        return stats.snapshot()

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
            if gate is not None:
                gate(size)
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
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        source = ODirectSafetensorsWeightSource(self, model_or_path)
        yield from source.iter_full_tensors()

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_files(model_config.model)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        source = ODirectSafetensorsWeightSource(self, model_weights)
        source_hooks = resolve_weight_plan_source_hooks(model)
        if source_hooks is not None:
            build_weight_plan, load_weights_from_source = source_hooks
            logger.info(
                "uma_odirect_safetensors using model WeightSource path: %s",
                type(model).__name__,
            )
            plan = build_weight_plan(source.catalog)
            try:
                loaded_weights = load_weights_from_source(source, plan)
            finally:
                source.close_files()
                source.log_stats("model-source")
            if loaded_weights is None:
                raise RuntimeError(
                    "load_weights_from_source must return the loaded "
                    "parameter name set so UMA-safe loading can fail closed "
                    f"on uninitialized parameters: {type(model).__name__}"
                )
            verify_loaded_weights(model, loaded_weights)
            return

        logger.info(
            "uma_odirect_safetensors using compatibility iterator path: %s",
            type(model).__name__,
        )
        try:
            loaded_weights = model.load_weights(source.iter_full_tensors())
        finally:
            source.close_files()
            source.log_stats("compat-iterator")
        if loaded_weights is None:
            logger.warning(
                "uma_odirect_safetensors: model %s load_weights did not "
                "report a loaded set; skipping the fail-closed completeness "
                "check",
                type(model).__name__,
            )
            return
        verify_loaded_weights(model, loaded_weights)
