# UMA Safe Loading Analysis

Date: 2026-07-01

This branch is for investigating how to make vLLM loading safer on UMA systems
such as DGX Spark, where CPU RAM and VRAM effectively share the same physical
memory pool.

The immediate goal is not to rewrite vLLM's model loading from scratch. vLLM
already has several fast safetensors loaders. The goal is to identify where a
fail-closed UMA-safe policy can be inserted without accidentally falling back to
page-cache-heavy or full-CPU-staging paths.

## Loader Entry Points

Primary files:

- `vllm/config/load.py`
  - Defines `LoadConfig`.
  - Holds `load_format`, `safetensors_load_strategy`,
    `safetensors_prefetch_*`, and `model_loader_extra_config`.
- `vllm/model_executor/model_loader/__init__.py`
  - Maps `load_format` to a `BaseModelLoader`.
  - `fastsafetensors`, `instanttensor`, and ordinary safetensors all use
    `DefaultModelLoader`.
  - `runai_streamer` uses `RunaiModelStreamerLoader`.
  - `runai_streamer_sharded` uses `ShardedStateLoader`.
- `vllm/model_executor/model_loader/base_loader.py`
  - Creates the model, calls `load_weights`, then post-processes weights.
- `vllm/model_executor/model_loader/default_loader.py`
  - Selects files.
  - Selects the actual weight iterator based on `load_format`.
- `vllm/model_executor/model_loader/weight_utils.py`
  - Implements ordinary safetensors, prefetch, eager, Run:ai iterator,
    fastsafetensors iterator, InstantTensor iterator, and torch load paths.

## Current Loader Behavior

### `load_format=auto`

`DefaultModelLoader._prepare_weights()` may resolve `auto` into an HF-style
loader path and allows both `*.safetensors` and `*.bin`.

UMA concern:

- Not fail-closed.
- May fall back to PyTorch `.bin` weights.
- Not suitable as an explicit safe-loader mode.

### `load_format=safetensors`

Uses `safetensors_weights_iterator()`.

Strategies:

- `None`: default lazy mmap-style `safe_open`; on NFS/Lustre, auto-prefetch may
  start if checkpoint size fits within 90% of available RAM.
- `lazy`: explicit lazy `safe_open`, no auto-prefetch.
- `eager`: reads the entire safetensors file into CPU memory via
  `load(f.read())`.
- `prefetch`: reads checkpoint files into OS page cache in a background thread.
- `torchao`: reconstructs torchao tensor subclasses from safetensors.

UMA concerns:

- `None` can auto-prefetch on network filesystems.
- `lazy` still uses file-backed pages and page cache.
- `eager` is explicit full-file CPU staging.
- `prefetch` intentionally fills page cache.

### `load_format=fastsafetensors`

Uses `fastsafetensors_weights_iterator()` via `DefaultModelLoader`.

Behavior:

- Creates a `ParallelLoader`.
- Uses current CUDA device.
- For tensor parallel size greater than 1, `nogds=True`.
- If GDS fails before yielding any tensor and `nogds` is false, it logs a
  warning and falls back to `nogds=True`.

UMA concerns:

- The current fallback from GDS to non-GDS is speed-friendly but not
  fail-closed for UMA safety.
- Need to understand whether `nogds=True` uses page cache and CPU staging.
- A safe mode would need to reject this fallback unless explicitly allowed.

### `load_format=instanttensor`

Uses `instanttensor_weights_iterator()` via `DefaultModelLoader`.

Behavior:

- Requires NVIDIA CUDA.
- Calls `instanttensor.safe_open(..., device=current_device,
  process_group=...)`.
- The docs describe distributed loading, pipelined prefetching, direct I/O, and
  GDS support when available.

UMA concerns:

- vLLM does not currently expose an explicit actual-path/fallback contract here.
- Need to verify whether InstantTensor can report direct/GDS vs fallback modes.

### `load_format=runai_streamer`

Uses `RunaiModelStreamerLoader`.

Behavior:

- Finds safetensors files from local FS, S3, GCS, Azure, or HF download.
- `model_loader_extra_config` accepts:
  - `distributed`
  - `concurrency`
  - `memory_limit`
- `concurrency` and `memory_limit` are applied through
  `RUNAI_STREAMER_CONCURRENCY` and `RUNAI_STREAMER_MEMORY_LIMIT`.
- `runai_safetensors_weights_iterator()` creates a `SafetensorsStreamer`,
  calls `stream_files`, then yields `tensor.clone()`.
- If `distributed=true` and platform is CUDA-like, streamer target device is
  `cuda:<current_device>`; otherwise it is `cpu`.

UMA concerns:

- This is the closest existing loader for controlled streaming.
- The final `tensor.clone()` may create an additional tensor copy.
- `memory_limit` limits the streamer's CPU buffer, but it is not a full system
  safety contract.
- No built-in MemAvailable / PSI gate.

### `load_format=runai_streamer_sharded`

Uses `ShardedStateLoader`.

Behavior:

- Expects rank-local checkpoint files:
  `model-rank-{rank}-part-{part}.safetensors`.
- This avoids every worker reading the whole checkpoint.
- If `runai_streamer_sharded`, it uses the Run:ai safetensors iterator.

UMA upside:

- This is likely important for large distributed vLLM/SGLang-style loads.
- Rank-local loading reduces read amplification and staging overlap.

UMA concerns:

- Requires pre-sharded checkpoint conversion.
- Still lacks MemAvailable / PSI / fallback policy.

## Likely Safe-Policy Insertion Points

### Low-risk wrapper approach

Add a policy layer around existing loaders:

- New `LoadConfig` fields or `model_loader_extra_config` keys:
  - `uma_safe`
  - `uma_min_available_gib`
  - `uma_psi_gate_seconds`
  - `uma_allow_fallback`
  - `uma_expected_loader`
- Preflight in loader constructor or before `_get_weights_iterator`.
- Gate between files / tensors in iterator wrappers.
- Validate log/actual loader path where possible.

Pros:

- Smallest initial patch.
- Reuses Run:ai / fastsafetensors / InstantTensor.

Cons:

- Cannot fully control external-library fallback unless the library exposes it.

### New loader format

Add a new load format:

```text
uma_safetensors
```

This could wrap `runai_streamer` first:

- Force safetensors-only.
- Force local model path initially.
- Force `concurrency=1` and low `memory_limit` unless explicitly overridden.
- Reject `auto`, `.bin`, `eager`, `prefetch`, mmap fallback, and unexpected
  loader paths.
- Record MemAvailable / PSI before and during load.

Pros:

- Explicit contract.
- Avoids changing normal vLLM behavior.

Cons:

- Still depends on Run:ai actual behavior unless deeper integration is added.

### Native O_DIRECT safetensors loader

The InstantTensor experiments showed that a fast external loader is not enough
for UMA safety:

- vLLM does not own the actual fallback decision. If the external loader
  changes from direct I/O to a buffered or mmap path, vLLM cannot reliably
  fail closed.
- In a two-node DGX Spark setup where the worker reads
  `192.168.100.11:/data/shared` over NFS, the worker's model reads still make
  the head node act as the storage server. The final weights are distributed,
  but the head node can still accumulate page cache.
- Even single-node direct-ish loaders can have extra cache/staging behavior
  that is difficult to prove from the vLLM side.

This branch therefore also adds:

```text
uma_odirect_safetensors
```

Initial contract:

- Local safetensors directory only.
- Linux only.
- No implicit Hugging Face download.
- No mmap, eager, prefetch, or buffered fallback.
- Read tensor payloads with `O_DIRECT`.
- Read only the safetensors header with ordinary buffered I/O.
- Gate `MemAvailable`, swap usage, and memory PSI before and between tensors.

Current limitations:

- It is a first safety-oriented prototype, not yet a fast loader.
- It stages one tensor at a time in a CPU tensor before vLLM's normal model
  weight loader consumes it.
- It does not yet do rank-local tensor filtering or checkpoint sharding.
- Many tiny tensors may cause inefficient repeated aligned reads. A production
  version should stream aligned file ranges and split tensors from a bounded
  ring buffer.

## Initial Recommendation

Start with an explicit `uma_safetensors` loader backed by Run:ai streamer:

1. Register `uma_safetensors` as a new load format.
2. Internally use Run:ai streamer with strict defaults:
   - `concurrency=1`
   - `memory_limit=1GiB`
   - `distributed=false` initially
3. Fail if safetensors files are missing or if any non-safetensors fallback is
   needed.
4. Add preflight checks:
   - local Linux only initially
   - MemAvailable threshold
   - swap not growing / ideally zero
   - memory PSI avg10 is zero before load
5. Add gate checks between yielded tensors:
   - MemAvailable threshold
   - memory PSI avg10 threshold
6. Emit explicit logs:
   - requested loader
   - actual loader
   - file count / total bytes
   - concurrency / memory_limit
   - pre/post MemAvailable and PSI

Future work:

- Determine whether `tensor.clone()` in Run:ai path can be avoided safely.
- Add fastsafetensors strict mode that refuses GDS to non-GDS fallback.
- Add InstantTensor strict mode if actual direct/GDS path can be queried.
- Add distributed/rank-local sharded safe path.

## Implemented Prototype

This branch now contains a first prototype of `load_format=uma_safetensors`.

Files:

- `vllm/model_executor/model_loader/uma_safetensors_loader.py`
- `vllm/model_executor/model_loader/__init__.py`
- `vllm/config/load.py`
- `tests/model_executor/model_loader/test_registry.py`

Contract:

- Safetensors only.
- Linux only.
- Local model directory by default.
- Hugging Face download is refused unless
  `model_loader_extra_config.allow_hf_download=true`.
- Object storage paths are refused unless
  `model_loader_extra_config.allow_remote_paths=true`.
- `safetensors_load_strategy=eager` and `prefetch` are rejected.
- Unexpected `model_loader_extra_config` keys are rejected.
- Memory gate checks:
  - `MemAvailable >= min_available_gib`, default `20`.
  - swap used <= `max_swap_gib`, default `0`.
  - `/proc/pressure/memory` `some/full avg10` must clear before load and
    between streamed tensors.
- Run:ai streamer defaults:
  - `concurrency=1`
  - `memory_limit=1GiB`
  - `distributed=false`

Example:

```bash
vllm serve /models/local-safetensors-model \
  --load-format uma_safetensors \
  --model-loader-extra-config \
    '{"concurrency":1,"memory_limit":1073741824,"min_available_gib":20}'
```

This is still a prototype. It establishes strict loader selection and system
gates, but it does not remove the `tensor.clone()` inside the existing Run:ai
iterator. It should not be treated as the same safety level as the native
`uma_odirect_safetensors` path because vLLM cannot pre-gate every internal
Run:ai allocation or clone.

## Implemented Native O_DIRECT Prototype

This branch also contains `load_format=uma_odirect_safetensors`, a native
safetensors reader for UMA experiments.

Files:

- `vllm/model_executor/model_loader/uma_odirect_safetensors_loader.py`
- `vllm/model_executor/model_loader/__init__.py`
- `vllm/config/load.py`
- `tests/model_executor/model_loader/test_registry.py`

Contract:

- Safetensors only.
- Linux only.
- Local model directory only.
- No mmap, eager, prefetch, Run:ai, InstantTensor, or buffered tensor payload
  fallback.
- The safetensors header is read with ordinary I/O.
- The ordinary header read is capped by `metadata_limit_mib` and validated
  against the file size before allocation.
- Tensor records are validated before payload reads. Invalid ranges,
  overlapping ranges, duplicate tensor names across shards, and symlinked
  safetensors files are rejected.
- Tensor payloads are read with `O_DIRECT` into a bounded aligned window.
- The loader gates `MemAvailable`, swap usage, and memory PSI before loading
  before tensor allocation, after tensor allocation, and after configurable
  read intervals.
- Invalid loader configuration is treated as an error rather than silently
  falling back to normal vLLM loaders.

Important `model_loader_extra_config` keys:

- `chunk_size`: aligned O_DIRECT read chunk size in bytes.
- `window_size`: bounded aligned read window in bytes. Adjacent small tensors
  can be served from this window without one direct read per tensor.
- `gate_interval_mib`: memory gate interval while reading tensor payloads.
  When non-zero, `chunk_size` and `window_size` must not exceed this bound.
- `allocation_gate_min_mib`: tensors at or above this size get forced
  allocation-before/after gates. Smaller tensors are covered by the byte
  interval gate to avoid per-tensor gate overhead.
- `min_available_gib`: minimum required `MemAvailable`.
- `max_swap_gib`: maximum allowed swap use.
- `psi_gate_seconds`: maximum allowed memory PSI `some/full avg10`.
- `metadata_limit_mib`: maximum safetensors metadata header size.

Routed MoE shortcuts are no longer configured in the storage loader.  Qwen,
Mixtral, DeepSeek, and Granite-style placement decisions are represented by
model-side WeightSource hooks so the loader remains model-family agnostic.

Example:

```bash
vllm serve /models/local-safetensors-model \
  --load-format uma_odirect_safetensors \
  --model-loader-extra-config \
    '{"chunk_size":8388608,"window_size":134217728,"gate_interval_mib":128,"min_available_gib":20,"max_swap_gib":0}'
```

### Model-Side Routed MoE Plans

Routed MoE models should implement `build_weight_plan(catalog)` and
`load_weights_from_source(source, plan)`.  The model plan:

- maps checkpoint expert tensors to the corresponding routed expert parameters
  before payload read
- delegates the actual sharding/copy behavior to the existing
  `RoutedExperts.weight_loader()`
- skips non-local experts before payload read when expert locality is known
- fails closed when a matched expert tensor cannot be represented safely

This keeps Qwen/Mixtral/DeepSeek/Granite naming rules out of the storage loader
and avoids accumulating model-family conditionals in
`uma_odirect_safetensors_loader.py`.

### Docker / Base Image Compatibility

The O_DIRECT loader is designed to be copied into an existing vLLM image with
minimal surface area. Do not blindly copy unrelated files from a newer source
tree into an older base image.

For the current DGX Spark test image:

- Copy `uma_odirect_safetensors_loader.py`, `uma_safetensors_loader.py`,
  `model_loader/__init__.py`, and `config/load.py`.
- Keep the base image's `RoutedExperts` implementation unless a specific
  compatibility fix is required.
- Do not copy `model_executor/models/utils.py` into the base image solely for
  profiling or convenience. Older images may have quantization config objects
  that do not match newer source-tree assumptions.

The public branch should keep the core loader self-contained. Model-specific
speedups should live in model-side WeightSource hooks rather than as
loader-side direct consumers.

### Current DGX Spark 35B Results

On the Qwen3.6-35B-A3B NVFP4 safetensors test model, the per-expert MoE direct
path loaded successfully with:

- Model load time before hardening: about `13.7s`.
- Model load time after stricter gates/range validation: about `22.3s`.
- `O_DIRECT` read/copy time: about `5.6s`.
- direct MoE consumer time: about `2.5-3.2s`.
- Local peak `used + buff/cache`: about `46.7GiB`.
- Memory PSI: `0`.
- Swap: `0`.

This is close to the llama.cpp GGUF O_DIRECT reference for the same 35B class
model, while keeping vLLM's model-side weight loading semantics intact. The
remaining gap is likely in generic tensor consumption, quantized tensor
post-processing, and vLLM startup overhead rather than raw file read speed.
