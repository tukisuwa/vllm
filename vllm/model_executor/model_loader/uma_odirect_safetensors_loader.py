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


@dataclass(frozen=True)
class _TensorRecord:
    file_path: str
    name: str
    dtype: torch.dtype
    shape: list[int]
    offset: int
    size: int


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
        records: list[_TensorRecord] = []
        seen_names: dict[str, str] = {}
        for path in files:
            file_size = os.path.getsize(path)
            with open(path, "rb", buffering=0) as f:
                raw_size = f.read(8)
                if len(raw_size) != 8:
                    raise RuntimeError(f"Invalid safetensors header in {path}")
                metadata_size = int.from_bytes(raw_size, "little")
                if metadata_size > self._metadata_limit_bytes:
                    raise RuntimeError(
                        f"Safetensors metadata too large in {path}: "
                        f"{metadata_size} bytes > {self._metadata_limit_bytes} bytes"
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
                if (
                    not isinstance(data_offsets, list)
                    or len(data_offsets) != 2
                ):
                    raise RuntimeError(
                        f"Invalid safetensors data_offsets for {name}: {data_offsets!r}"
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
                    _TensorRecord(
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
        return records

    def _get_weights_iterator(
        self,
        model_or_path: str,
        direct_consumer: Callable[[_TensorRecord, torch.Tensor], bool] | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        self._gate_memory("preflight")
        files = self._prepare_files(model_or_path)
        records = self._read_records(files)
        total_bytes = sum(record.size for record in records)
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

                force_allocation_gate = record.size >= self._allocation_gate_min_bytes
                maybe_gate(
                    f"before allocating {record.name}",
                    force=force_allocation_gate,
                )
                t0 = time.perf_counter()
                tensor = torch.empty(record.shape, dtype=record.dtype, device="cpu")
                time_alloc += time.perf_counter() - t0
                maybe_gate(
                    f"after allocating {record.name}",
                    force=force_allocation_gate,
                )

                assert odirect_file is not None
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
                time_read += time.perf_counter() - t0
                if record.size <= odirect_file.window_size:
                    note_loaded_bytes(record.size)

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
        direct_consumer = (
            _PerExpertMoeDirectLoader(model) if self._direct_per_expert_moe else None
        )
        model.load_weights(self._get_weights_iterator(model_weights, direct_consumer))
