# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import time
from typing import Any


def read_meminfo() -> dict[str, int]:
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


def read_memory_psi_avg10() -> tuple[float, float]:
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


def format_gib(value: float) -> str:
    return f"{value / 1024**3:.2f} GiB"


def gate_uma_memory(
    *,
    loader_label: str,
    phase: str,
    min_available_gib: float,
    psi_gate_seconds: float,
    max_swap_gib: float,
    logger: Any,
) -> None:
    if sys.platform != "linux":
        raise RuntimeError(f"{loader_label} is Linux-only")

    deadline = time.monotonic() + psi_gate_seconds
    while True:
        meminfo = read_meminfo()
        available = meminfo.get("MemAvailable", 0)
        swap_used = meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0)
        some_avg10, full_avg10 = read_memory_psi_avg10()
        min_available = min_available_gib * 1024**3
        max_swap = max_swap_gib * 1024**3

        if available < min_available:
            raise RuntimeError(
                f"{loader_label} memory gate failed during "
                f"{phase}: MemAvailable {format_gib(available)} < "
                f"{min_available_gib:.2f} GiB"
            )
        if swap_used > max_swap:
            raise RuntimeError(
                f"{loader_label} swap gate failed during "
                f"{phase}: swap used {format_gib(swap_used)} > "
                f"{max_swap_gib:.2f} GiB"
            )
        if some_avg10 == 0.0 and full_avg10 == 0.0:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"{loader_label} PSI gate failed during "
                f"{phase}: memory PSI avg10 some={some_avg10:.2f}, "
                f"full={full_avg10:.2f}"
            )
        logger.warning(
            "%s waiting for memory PSI to clear during %s: some=%.2f full=%.2f",
            loader_label,
            phase,
            some_avg10,
            full_avg10,
        )
        time.sleep(1.0)
