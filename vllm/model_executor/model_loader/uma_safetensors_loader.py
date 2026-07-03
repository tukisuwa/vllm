# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from glob import glob
from collections.abc import Generator

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader._uma_memory_gate import (
    format_gib,
    gate_uma_memory,
)
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    runai_safetensors_weights_iterator,
)
from vllm.transformers_utils.runai_utils import is_runai_obj_uri, list_safetensors

logger = init_logger(__name__)


class UmaSafetensorsModelLoader(BaseModelLoader):
    """Strict Run:ai safetensors loader wrapper for UMA experiments.

    This loader is intentionally conservative about loader selection and system
    gates. It still uses Run:ai Model Streamer as the tensor backend, so vLLM
    cannot pre-gate every internal streamer allocation or tensor clone.
    """

    DEFAULT_MEMORY_LIMIT = 1024**3
    DEFAULT_MIN_AVAILABLE_GIB = 20.0
    DEFAULT_PSI_GATE_SECONDS = 30.0

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError(
                "model_loader_extra_config must be a dict for "
                f"{load_config.load_format}, got {type(extra_config).__name__}"
            )

        allowed_keys = {
            "allow_hf_download",
            "allow_remote_paths",
            "concurrency",
            "distributed",
            "memory_limit",
            "min_available_gib",
            "psi_gate_seconds",
            "max_swap_gib",
        }
        unexpected_keys = set(extra_config) - allowed_keys
        if unexpected_keys:
            raise ValueError(
                "Unexpected extra config keys for uma_safetensors: "
                f"{unexpected_keys}"
            )

        self._is_distributed = self._get_bool(extra_config, "distributed", False)
        self._allow_hf_download = self._get_bool(
            extra_config, "allow_hf_download", False
        )
        self._allow_remote_paths = self._get_bool(
            extra_config, "allow_remote_paths", False
        )
        self._concurrency = self._get_positive_int(extra_config, "concurrency", 1)
        self._memory_limit = self._get_positive_int(
            extra_config, "memory_limit", self.DEFAULT_MEMORY_LIMIT
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

        if load_config.safetensors_load_strategy not in (None, "lazy"):
            raise ValueError(
                "uma_safetensors does not support safetensors_load_strategy="
                f"{load_config.safetensors_load_strategy!r}; it uses Run:ai "
                "Model Streamer and rejects eager/prefetch paths."
            )

        os.environ.update(
            {
                "RUNAI_STREAMER_CONCURRENCY": str(self._concurrency),
                "RUNAI_STREAMER_MEMORY_LIMIT": str(self._memory_limit),
            }
        )

    @staticmethod
    def _get_bool(config: dict, key: str, default: bool) -> bool:
        value = config.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a bool, got {value!r}")
        return value

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
        gate_uma_memory(
            loader_label="uma_safetensors",
            phase=phase,
            min_available_gib=self._min_available_gib,
            psi_gate_seconds=self._psi_gate_seconds,
            max_swap_gib=self._max_swap_gib,
            logger=logger,
        )

    def _prepare_weights(
        self, model_name_or_path: str, revision: str | None
    ) -> list[str]:
        is_object_storage_path = is_runai_obj_uri(model_name_or_path)
        is_local = os.path.isdir(model_name_or_path)
        if is_object_storage_path and not self._allow_remote_paths:
            raise RuntimeError(
                "uma_safetensors refuses object storage paths unless "
                "model_loader_extra_config.allow_remote_paths=true"
            )
        if not is_local and not is_object_storage_path and not self._allow_hf_download:
            raise RuntimeError(
                "uma_safetensors refuses implicit Hugging Face downloads. "
                "Use a local safetensors directory or set "
                "model_loader_extra_config.allow_hf_download=true."
            )

        safetensors_pattern = "*.safetensors"
        index_file = SAFE_WEIGHTS_INDEX_NAME
        hf_folder = (
            model_name_or_path
            if (is_local or is_object_storage_path)
            else download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                [safetensors_pattern],
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        )
        if is_local:
            hf_weights_files = sorted(glob(os.path.join(hf_folder, safetensors_pattern)))
        else:
            hf_weights_files = list_safetensors(path=hf_folder)

        if not is_local and not is_object_storage_path:
            download_safetensors_index_file_from_hf(
                model_name_or_path,
                index_file,
                cache_dir=self.load_config.download_dir,
                revision=revision,
            )

        if not hf_weights_files:
            raise RuntimeError(
                f"Cannot find any safetensors model weights with `{model_name_or_path}`"
            )

        total_size = "unknown"
        if is_local:
            total_bytes = sum(os.path.getsize(path) for path in hf_weights_files)
            total_size = f"{total_bytes / 1024**3:.2f} GiB"
        logger.info(
            "uma_safetensors using Run:ai streamer: files=%d total=%s "
            "concurrency=%d memory_limit=%s distributed=%s",
            len(hf_weights_files),
            total_size,
            self._concurrency,
            format_gib(self._memory_limit),
            self._is_distributed,
        )
        return hf_weights_files

    def _get_weights_iterator(
        self, model_or_path: str, revision: str | None
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        self._gate_memory("preflight")
        hf_weights_files = self._prepare_weights(model_or_path, revision)
        self._gate_memory("before stream")
        for name, tensor in runai_safetensors_weights_iterator(
            hf_weights_files,
            self.load_config.use_tqdm_on_load,
            self._is_distributed,
        ):
            self._gate_memory(f"tensor {name}")
            yield name, tensor
        self._gate_memory("after stream")

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model, model_config.revision)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        model_weights = model_config.model
        if model_weights_override := model_config.model_weights:
            model_weights = model_weights_override
        model.load_weights(
            self._get_weights_iterator(model_weights, model_config.revision)
        )
