# UMA-safe WeightSource / WeightPlan design

Date: 2026-07-02

This document describes a deeper loader design for UMA-safe model loading in
vLLM.  It is based on the current `uma_odirect_safetensors` experiments and on
the limitations found while testing Qwen3.6-35B-A3B on DGX Spark.

## Problem

vLLM's current weight loading contract is mostly:

```text
storage loader -> Iterable[(checkpoint_name, full CPU torch.Tensor)]
               -> model.load_weights()
               -> model-specific shard/fuse/quant/MoE placement
```

This is simple and works well for many dGPU deployments, but it is the wrong
abstraction for UMA safety.

The loader must often materialize a complete CPU tensor before the model code
decides whether the tensor is:

- needed at all
- only partly needed for TP/PP/EP
- one shard of a fused parameter such as QKV or gate/up
- a routed expert that is not local to this rank
- a quantized payload tensor, scale tensor, or auxiliary metadata tensor
- a tensor that should be skipped because embeddings are tied or the layer is
  absent on this pipeline rank

For UMA systems, that ordering is dangerous.  CPU RAM, page cache, staging
buffers, CUDA allocations, and device-visible memory all share the same physical
pool.  The loader needs to avoid reading and staging data that the model will
later discard or slice.

The earlier `direct_per_expert_moe` optimization proved the issue: it improved
Qwen MoE loading by bypassing the normal full-tensor iterator for per-expert
weights, but it did so by teaching the loader about a specific MoE checkpoint
layout.  That was a local optimization, not a sound general design.  This
branch now keeps that knowledge in model-side WeightSource hooks instead of in
the storage loader.

## Current relevant code

Current loader entry points:

- `vllm/model_executor/model_loader/base_loader.py`
  - `BaseModelLoader.load_model()`
  - initializes the model and calls `self.load_weights(model, model_config)`
- `vllm/model_executor/model_loader/__init__.py`
  - maps `load_format` to a `BaseModelLoader`
- `vllm/model_executor/model_loader/uma_odirect_safetensors_loader.py`
  - current fail-closed local O_DIRECT safetensors loader
  - currently yields `(name, CPU tensor)` for the base path
  - does not contain model-family direct-loading branches; routed expert
    placement is handled by model-side WeightSource hooks

Current model-side loading:

- `vllm/model_executor/models/utils.py`
  - `AutoWeightsLoader`
  - recursively routes `(name, tensor)` to parameters/modules
  - applies mapper, skip prefixes/substrings, and `param.weight_loader`
- Representative model implementations:
  - `vllm/model_executor/models/llama.py`
  - `vllm/model_executor/models/qwen3.py`
  - `vllm/model_executor/models/qwen3_moe.py`
  - `vllm/model_executor/models/deepseek_v2.py`
  - `vllm/model_executor/models/granitemoe*.py`

The key design issue is that model-specific placement decisions live after full
CPU tensor materialization.

## Goal

Move weight loading from a push iterator of full tensors to a pull-based plan:

```text
safetensors metadata
  -> TensorCatalog
  -> model-specific WeightPlan
  -> UMA-safe WeightExecutor
  -> only required byte ranges are read and placed
```

The loader should not know whether a tensor is Qwen, DeepSeek, Mixtral, Granite,
or Llama.  The model should describe what it needs; the loader should provide
safe I/O and execution primitives.

## Non-goals

- Do not replace all vLLM model `load_weights()` implementations at once.
- Do not require every model to implement the new API immediately.
- Do not weaken the existing fail-closed behavior of `uma_odirect_safetensors`.
- Do not make UMA-safe loading depend on Docker, Run:ai, InstantTensor, or
  model-specific external services.

## Proposed abstractions

### TensorCatalog

Metadata-only view of checkpoint tensors.

```python
@dataclass(frozen=True)
class TensorMeta:
    name: str
    file_path: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    data_offset: int
    nbytes: int

class TensorCatalog:
    def names(self) -> Iterable[str]: ...
    def get(self, name: str) -> TensorMeta: ...
    def has(self, name: str) -> bool: ...
```

Properties:

- reads safetensors metadata only
- validates duplicate names, ranges, dtype, shape, overlap
- performs no tensor payload read
- has no model-specific logic

The current `_read_records()` in `uma_odirect_safetensors_loader.py` is already
close to this and should be split out.

### WeightSource

I/O primitive provider for a catalog.

```python
class WeightSource:
    catalog: TensorCatalog

    def read_full_cpu(self, name: str) -> torch.Tensor: ...

    def read_into_cpu(
        self,
        name: str,
        dst: torch.Tensor,
        *,
        source_slices: tuple[slice | int, ...] | None = None,
        target_slices: tuple[slice | int, ...] | None = None,
    ) -> None: ...

    def read_slice_cpu(
        self,
        name: str,
        source_slices: tuple[slice | int, ...],
    ) -> torch.Tensor: ...

    def empty_cpu_shape(
        self,
        name: str,
        shape: tuple[int, ...],
    ) -> torch.Tensor: ...

    def skip(self, name: str, reason: str) -> None: ...
```

Initial implementation can be conservative:

- only contiguous full-tensor reads are mandatory
- non-contiguous/sliced reads may initially read the minimal enclosing range or
  fall back to a CPU tensor slice if safety remains bounded
- O_DIRECT, PSI gate, swap gate, min-available gate, progress, and cancellation
  belong here

Longer-term implementation should support:

- direct contiguous byte-range reads
- row/column shard reads for TP
- local expert-only reads for EP
- destination-buffer reads when a parameter exposes a safe CPU/device staging
  target

### WeightPlan

Model-side declaration of required checkpoint reads and placements.

```python
@dataclass(frozen=True)
class WeightPlanReadSegment:
    source_slices: tuple[slice | int, ...]
    target_slices: tuple[slice | int, ...]

@dataclass(frozen=True)
class WeightPlanEntry:
    checkpoint_name: str
    target_name: str
    required: bool = True
    source_slices: tuple[slice | int, ...] | None = None
    target_slices: tuple[slice | int, ...] | None = None
    read_segments: tuple[WeightPlanReadSegment, ...] | None = None
    staging_shape: tuple[int, ...] | None = None
    shard_id: str | int | None = None
    expert_id: int | None = None
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    loader_kind: str = "default"
```

The plan is built by the model, because only the model knows:

- tied embeddings
- pipeline-missing layers
- tensor-parallel shard boundaries
- fused parameter layout
- MoE expert mapping
- quantization auxiliary tensors
- which missing/unexpected tensors are acceptable

Suggested model hook:

```python
class SupportsWeightSource:
    def build_weight_plan(self, catalog: TensorCatalog) -> WeightPlan: ...

    def load_weights_from_source(
        self,
        source: WeightSource,
        plan: WeightPlan,
    ) -> set[str]: ...
```

The hook should be optional.  If a model does not implement it, the loader falls
back to the existing safe `(name, CPU tensor)` iterator path.

### WeightExecutor

Executes a model plan through a source.

Responsibilities:

- enforce fail-closed policy
- decide when full CPU tensor fallback is allowed
- run gates before allocation/read/placement
- route to `param.weight_loader`
- preserve existing vLLM post-load processing
- collect timing and byte counters
- summarize full/sliced/read-into/skipped payload bytes from metadata before
  payload read

Conceptual execution:

```python
for entry in plan.entries:
    if not entry.required:
        source.skip(entry.checkpoint_name, "not required")
        continue

    meta = source.catalog.get(entry.checkpoint_name)
    param = resolve_target_param(model, entry.target_name)

    if entry.can_read_into_param_cpu_view:
        source.read_into_cpu(entry.checkpoint_name, cpu_view, ...)
        param.weight_loader(param, cpu_view, ...)
    elif entry.read_segments is not None:
        tensor = source.empty_cpu_shape(entry.checkpoint_name, entry.staging_shape)
        for segment in entry.read_segments:
            source.read_into_cpu(
                entry.checkpoint_name,
                tensor,
                source_slices=segment.source_slices,
                target_slices=segment.target_slices,
            )
        param.weight_loader(param, tensor, ...)
    elif entry.source_slices is not None:
        tensor = source.read_slice_cpu(entry.checkpoint_name, entry.source_slices)
        param.weight_loader(param, tensor, ...)
    else:
        tensor = source.read_full_cpu(entry.checkpoint_name)
        param.weight_loader(param, tensor, ...)
```

The first version can still call `weight_loader` with CPU tensors; the important
change is that the decision to skip/slice/fuse is made before payload read.

## Migration plan

### Phase 0: Keep current safe base path

Status: current `uma_odirect_safetensors`.

- O_DIRECT local safetensors reader
- metadata validation
- no mmap/eager/prefetch
- no buffered fallback
- memory/swap/PSI gates
- model-load-only tests on Qwen3.6-35B-A3B

### Phase 1: Extract TensorCatalog and WeightSource

Status: started on `uma-safe-weight-source`.

Implemented so far:

- `TensorMeta` / `TensorCatalog`
  - metadata-only safetensors catalog
  - duplicate name, dtype, shape, range, and overlap validation
  - lookup by name and stable file/offset ordering
- `ODirectSafetensorsWeightSource`
  - builds the catalog before payload reads
  - exposes `source.catalog`
  - keeps the compatibility full-tensor iterator
- optional model hook dispatch:
  - `build_weight_plan(catalog)`
  - `load_weights_from_source(source, plan)`
  - unsupported models still use `model.load_weights(iterator)`
- `source.read_full_cpu(name)`
  - uses the same O_DIRECT full-record read helper as the iterator path
  - keeps allocation/read memory gates on the pull path
- `source.read_slice_cpu(name, source_slices)`
  - supports row-major contiguous slices and simple single-dimension strided
    slices
  - rejects stepped or more complex non-contiguous slices instead of full-tensor
    fallback
  - records only the requested payload bytes in source stats
- `source.read_into_cpu(name, dst, source_slices=None, target_slices=None)`
  - reads full, contiguous slice, or supported single-dimension strided slice
    directly into a caller-provided contiguous CPU tensor
  - can place the read into a contiguous target slice of that CPU tensor; more
    complex non-contiguous target views fail closed
  - validates destination dtype, shape, device, and contiguity fail-closed
  - is implemented as a WeightSource primitive; model executors do not yet use
    it for parameter placement by default
- `source.empty_cpu(name, source_slices=None)`
  - allocates a correctly shaped CPU staging tensor under WeightSource
    allocation gates and timing stats
- `source.empty_cpu_shape(name, shape)`
  - allocates a caller-shaped CPU staging tensor under the same WeightSource
    allocation gates and timing stats
  - is used by segmented plan entries that assemble multiple source slices into
    one staging tensor before calling the existing parameter `weight_loader`
- source-level stats for model hook reads and compatibility iterator reads
  - files opened
  - tensors read/skipped
  - direct reads/window reads
  - bytes read/copied/payload/skipped
  - gate/allocation/read timing
- compatibility iterator path now uses `ODirectSafetensorsWeightSource`
  orchestration and logs the same source stats as model-side WeightPlan loads
- stable `source.stats_snapshot()` for tests and diagnostics
- minimal `WeightPlanEntry`, `WeightPlan`, and `execute_weight_plan(...)`
  - supports required/skipped entries
  - supports full CPU reads and contiguous source slices
  - can opt into `source.read_into_cpu(...)` CPU staging via
    `WeightPlanEntry.read_into_cpu`
  - routes tensors to a target parameter's `weight_loader`
- `build_auto_weight_plan_from_catalog(...)`
  - performs prefix/substr skips without reading payload bytes
  - supports a model-provided skip predicate for non-prefix decisions such as
    speculative layers or optional per-layer auxiliaries
  - applies `WeightsMapper`-style name and shard mapping before payload reads
  - can mark known ignorable missing suffixes such as `.bias`
  - keeps missing target detection fail-closed in the executor
- `build_auto_weight_plan_for_module(...)`
  - adds AutoWeightsLoader-compatible quant cache-scale mapper composition
  - adds module quant-config ignored suffixes
  - keeps model hooks small and consistent across dense model families
- `vllm/model_executor/models/auto_uma.py`
  - shared model-side helpers for dense AutoWeightsLoader-style source hooks
  - keeps model files from depending directly on generic executor internals
  - is used by Qwen2, Qwen3, Llama, Gemma, Gemma2, Gemma3, InternLM2, Phi,
    Starcoder2, Falcon, Mistral, GPTBigCode, OPT, BLOOM, OLMo, OLMo2,
    Nemotron, EXAONE, and Cohere
- shared routed-MoE model helper
  - keeps per-family parsing and module resolution in model-side files
  - centralizes local-expert skip decisions and existing FusedMoE/RoutedExperts
    `weight_loader` delegation
  - is used by Qwen-family MoE and Mixtral-style MoE hooks
- `execute_weight_plan(...)`
  - resolves target parameters before payload reads
  - skips `ignore_missing` entries before payload reads
  - fails closed before payload reads for unexpected missing targets
  - supports opt-in `read_into_cpu` entries for caller-controlled CPU staging;
    parameter/device direct placement is still future work
  - supports `WeightPlanEntry.target_slices` when paired with
    `read_into_cpu=True`, allowing a model hook to allocate one CPU staging
    tensor and fill only selected contiguous target views under WeightSource
    gates
  - supports model-provided tensor transforms after source read and before
    `weight_loader`, keeping special checkpoint transforms such as Mistral
    consolidated-checkpoint q/k permutation out of the storage loader
  - uses `source.empty_cpu(...)` for opt-in CPU staging so allocation gates stay
    under WeightSource control
  - uses `source.empty_cpu_shape(...)` and `WeightPlanReadSegment` for
    segmented CPU staging reads, allowing a model hook to assemble selected
    contiguous source slices into one CPU tensor without full checkpoint tensor
    materialization
  - still rejects `WeightPlanEntry.target_slices` without `read_into_cpu=True`
    because ordinary `weight_loader` calls do not provide a safe destination
    tensor contract
- tests for catalog lookup, source construction, full CPU reads, contiguous
  slice reads, source stats, optional model hook dispatch, name-only plan
  building, and basic plan execution

Remaining refactor without behavioral change:

- keep narrowing loader/source ownership boundaries where it removes duplicated
  I/O orchestration
- expand plan execution only where generic semantics are clear; model-specific
  transforms should stay in model-side plan code

Benefit:

- separates safe I/O from current iterator compatibility
- makes tests easier
- creates a stable API for plan experiments

### Phase 2: Add optional model hook

Status: initial hook and minimal generic executor implemented.

In `UmaODirectSafetensorsModelLoader.load_weights()`:

```python
if hasattr(model, "build_weight_plan") and hasattr(model, "load_weights_from_source"):
    catalog = source.catalog
    plan = model.build_weight_plan(catalog)
    model.load_weights_from_source(source, plan)
else:
    model.load_weights(source.iter_full_tensors())
```

Requirements:

- must be opt-in: implemented
- must log whether the model-source path or iterator path was used: implemented
- must fail closed if the model requested source loading but the source cannot
  provide a required operation: implemented for missing paired hooks,
  unsupported slices, missing targets, and missing `weight_loader`

Remaining:

- validate the current model-side `build_weight_plan(...)` hooks on more real
  checkpoints
- expand shared model-side helpers only where behavior is genuinely common
- add broader tests around actual vLLM parameter loaders

### Phase 3: Dense model prototype

Status: started with `Qwen2ForCausalLM`, `Qwen3ForCausalLM`,
`LlamaForCausalLM`, `GemmaForCausalLM`, `Gemma2ForCausalLM`,
`Gemma3ForCausalLM`, `InternLM2ForCausalLM`, `PhiForCausalLM`,
`Starcoder2ForCausalLM`, `FalconForCausalLM`, `MistralForCausalLM`,
`GPTBigCodeForCausalLM`, `OPTForCausalLM`, `BloomForCausalLM`,
`GPTJForCausalLM`, `MPTForCausalLM`, `OrionForCausalLM`,
`Step1ForCausalLM`, `ApertusForCausalLM`, `StablelmForCausalLM`,
`SolarForCausalLM`, `GPTNeoXForCausalLM`, `PersimmonForCausalLM`,
`GraniteForCausalLM`, `Jais2ForCausalLM`, `Exaone4ForCausalLM`,
`Plamo3ForCausalLM`, `ArceeForCausalLM`, `SeedOssForCausalLM`,
`HyperCLOVAXForCausalLM`, `Lfm2ForCausalLM`, `MiMoForCausalLM`,
`OlmoForCausalLM`, `Olmo2ForCausalLM`, `NemotronForCausalLM`,
`ExaoneForCausalLM`, `CohereForCausalLM`, `TeleChat2ForCausalLM`,
`FalconH1ForCausalLM`, `Zamba2ForCausalLM`, and `OuroForCausalLM`.

Start with Llama/Qwen dense, not MoE.

Why:

- fewer model-specific edge cases
- tests TP slicing and fused QKV/gate-up without expert mapping
- proves the plan interface without relying on Qwen35 MoE special cases

Target behavior:

- plan skips tied `lm_head`: implemented for Qwen2, Qwen3, Llama, Gemma,
  Gemma2, Gemma3, OLMo, OLMo2, and EXAONE where their existing loader skips it
- plan skips tied InternLM2 `output`: implemented where its existing loader
  skips it
- plan skips tied Starcoder2 `lm_head.weight`: implemented where its existing
  loader skips it
- plan skips tied Falcon `lm_head`: implemented where its existing loader skips it
- plan remaps Mistral consolidated-checkpoint names before payload read and
  attaches q/k permutation transforms to the affected entries
- plan remaps BLOOM checkpoint names with the existing `transformer.` prefix
  rule before payload read
- plan skips static non-payload entries such as Cohere `rotary_emb.inv_freq`
- plan uses segmented CPU staging reads for TeleChat2 `key_value` tensors,
  assembling K and V staging tensors from alternating source head blocks before
  delegating to the existing qkv parameter loader
- plan skips rotary/cache tensors: implemented through shared auto-plan helper
- dense model hooks use the shared `auto_uma` model-side helper instead of
  calling generic executor internals directly
- plan maps q/k/v into qkv placement before read: implemented through
  `hf_to_vllm_mapper` and `shard_id`
- plan maps gate/up into gate_up placement before read: implemented for Llama
- plan includes AutoWeightsLoader-compatible quant cache-scale mapper and
  ignored suffix handling for Qwen3 and Llama
- plan can read only local TP shard where possible: implemented for simple
  output-dimension row shards where the source slice is contiguous
- plan can also source-slice single `shard_id` fused output shards when the
  bound vLLM weight loader exposes a local shard-size mapping, covering QKV-like
  q/k/v and MergedColumn-like gate/up cases in the non-packed path
- plan can source-slice simple input-dimension TP shards with a single sliced
  dimension, using bounded strided O_DIRECT reads instead of materializing the
  full checkpoint tensor first
- packed, bitsandbytes, tuple/multi-shard fused entries, unknown fused loaders,
  and more complex non-contiguous slices still fall back to full tensor reads

Current limitations:

- The first Qwen3 hook still reads each required checkpoint tensor as a CPU
  tensor, then delegates to existing parameter `weight_loader`.
- It proves model-side planning and skip/map decisions before payload read, but
  only performs conservative TP-local source slicing for simple output-dimension
  shards and single-dimension input shards.
- Qwen3 MoE now has a model-side source hook; the older loader-side
  compatibility optimization has been removed so routed expert special cases
  remain model-side.

### Phase 4: Qwen MoE prototype

Status: started with `Qwen2MoeForCausalLM`, `Qwen3MoeForCausalLM`,
`Qwen3NextForCausalLM`, `Qwen3_5MoeForCausalLM`, and
`Qwen3_5MoeForConditionalGeneration`.

Replace `direct_per_expert_moe` with a model-side plan for Qwen MoE.

Target behavior:

- model plan declares local expert reads: implemented for Qwen-family MoE routed
  expert tensors
- non-local experts are skipped before payload read: implemented for Qwen-family
  MoE
- gate/up/down fused placements are described by plan entries: implemented for
  Qwen-family MoE per-expert gate/up/down naming
- quant payload and scale suffixes are handled by the model plan, not by loader:
  implemented for suffix-based Qwen-family MoE routed expert targets
- Qwen2 MoE reuses the Qwen-family helper while preserving its HF-to-vLLM
  mapper for QKV, dense MLP, and shared expert dense projections
- nested `language_model.model.layers` wrappers are supported for Qwen3.5 MoE
  conditional generation

This should preserve the observed Qwen35B performance while moving the
model-specific knowledge out of the loader.

Remaining work:

- validate Qwen3.5/Qwen3Next MoE and Qwen3.5 MoE conditional generation on real
  checkpoints; they share the model-side helper but have not been real-loaded
  yet
- add real-load verification that the model hook matches the previous
  Qwen35B performance and memory behavior

### Phase 5: Other MoE families

Add model-side plans only where needed:

- Mixtral: initial model-side source hook implemented for per-expert
  `block_sparse_moe.experts.<expert>.w{1,2,3}.*` tensors; it skips non-local
  experts before payload read and delegates placement to existing FusedMoE
  `weight_loader`. This now uses the shared routed-MoE helper rather than a
  second copy of the Qwen-specific plan executor.
- PhiMoE: uses the same Mixtral-style routed-MoE hook for
  `block_sparse_moe.experts.<expert>.w{1,2,3}.*` tensors, preserving the
  model's existing HF-to-vLLM mapper and FusedMoE placement path.
- OLMoE and Cohere2 MoE: use the Qwen-style routed-MoE helper for
  `mlp.experts.<expert>.{gate,up,down}_proj.*` tensors. Cohere2 keeps the
  existing `lm_head` skip behavior in the model hook.
- DeepSeek V2/V3 style MoE
  - initial conservative model-side hook implemented for routed
    `mlp.experts.<expert>.{gate,up,down}_proj.*` tensors and common packed
    dense mappings
  - skips non-local routed experts before payload read through the shared
    routed-MoE helper
  - skips speculative layers and absent per-layer indexer weights before
    payload read
  - shared-expert fusion is represented as source slices into appended expert
    slots, avoiding a full shared-expert tensor read before chunking
  - FP8 indexer WK fusion is handled as a DeepSeek-side deferred plan entry:
    the hook reads the FP8 WK tensor and its scale tensor through WeightSource,
    dequantizes to BF16, and loads shard 0 into `wk_weights_proj`
- Granite MoE variants
  - initial hook implemented for GraniteMoe and GraniteMoeShared checkpoint
    tensors that store all experts in `input_linear` / `output_linear`
  - reads expert-local `w1`, `w3`, and `w2` slices from those fused source
    tensors instead of materializing the full source tensor and splitting in
    Python
  - skips non-local expert slices before payload read when the FusedMoE expert
    map exposes locality
  - GraniteMoeHybrid has a separate hook for the same fused expert tensors,
    plus `weight_scale` slices and the existing `A_log` -> `A` Mamba mapping

Each model family should implement its own plan builder instead of adding
loader-side conditionals.

## Compatibility matrix

Expected current behavior:

| Model type | Base `uma_odirect_safetensors` | Direct plan path |
| --- | --- | --- |
| Dense safetensors | Should work if normal vLLM load works | Phase 3 hooks for Qwen2/Qwen3/Llama/Gemma/Gemma2/Gemma3/InternLM2/Phi/Starcoder2/Falcon/FalconH1/Mistral/GPTBigCode/OPT/BLOOM/GPT-J/MPT/Orion/Step1/Apertus/StableLM/Solar/GPT-NeoX/Persimmon/Granite/Jais2/EXAONE4/Plamo3/Arcee/SeedOss/HyperCLOVAX/LFM2/MiMo/OLMo/OLMo2/Nemotron/EXAONE/Cohere/TeleChat2/Zamba2/Ouro-style AutoWeightsLoader models |
| Sharded dense safetensors | Should work if no duplicate names | Phase 3 |
| Qwen2/Qwen3 / OLMoE / Cohere2 routed MoE | Base path should work if normal vLLM load works | Phase 4/5 model hook for `mlp.experts` gate/up/down tensors; loader-side direct path removed |
| Mixtral / PhiMoE routed MoE | Base path should work if normal vLLM load works | Phase 5 initial model hook for `block_sparse_moe.experts` w1/w2/w3 tensors |
| DeepSeek V2/V3 routed MoE | Base path should work if normal vLLM load works | Phase 5 hook for routed experts, shared-expert fusion source slices, and FP8 indexer WK fusion |
| Granite MoE / Granite MoE Shared | Base path should work if normal vLLM load works | Phase 5 initial model hook |
| Granite MoE Hybrid | Base path should work if normal vLLM load works | Phase 5 initial hook, including fused expert `weight_scale` and `A_log` mapping |
| Other routed MoE | Base path should work if normal vLLM load works | Not yet implemented |
| Non-safetensors | Not supported | Not supported |
| Remote HF path | Not supported by UMA-safe loader | Not supported |
| mmap/eager/prefetch | Rejected | Rejected |
| symlinked weights | Rejected | Rejected |

## Testing strategy

Static tests:

- malformed safetensors metadata
- duplicate names across shards
- symlink rejection
- overlapping ranges
- unsupported dtype
- zero-sized tensors
- `TensorCatalog` lookup and stable ordering

Model audit:

- `scripts/audit-vllm-uma-odirect-model.py <model-dir>`
- reads safetensors metadata only; no tensor payload bytes are read
- validates the same duplicate name, dtype, shape, range, and overlap rules as
  the UMA O_DIRECT loader
- prints total payload bytes, per-file payload bytes, largest tensors, dtype
  distribution, and coarse tensor-name classes such as per-expert MoE,
  Granite fused expert tensors, QKV, embeddings, and shared experts
- prints local `config.json` architectures when present, plus UMA hints for
  implemented direct-plan families such as DeepSeek V2/V3, shared experts,
  DeepSeek FP8 indexer WK pairs, and Granite fused experts
- should be run before adding a model to the compatibility list

Runtime tests:

- small dense model, single shard
- small dense model, multiple shards
- Qwen35B current NVFP4 model
- one non-Qwen MoE with direct path disabled
- later: one non-Qwen MoE with a model-side plan

Runtime metrics to record:

- model load time
- peak used
- peak buff/cache
- peak used + buff/cache
- min available
- swap
- memory PSI
- IO PSI
- bytes read
- bytes copied
- skipped bytes
- full-tensor fallback bytes
- slice/direct-read bytes
- full vs sliced tensor read counts and payload bytes from source stats
- metadata-only WeightPlan summary bytes before execution

## Design rules

- The loader must not contain model-family branches such as `if qwen` or
  `if deepseek`.
- The model must not perform raw file I/O.
- A requested safe source operation must fail closed if unsupported.
- Buffered fallback must be explicit and off by default.
- Skipped tensors should be skipped before payload read.
- Non-local expert tensors should be skipped before payload read where the model
  plan can determine locality.
- `read_full_cpu()` remains only for compatibility and should be measured.
- All source operations must honor the same memory/swap/PSI gates.

## Open questions

- Which model-side transforms should graduate from callables into named,
  inspectable plan operations?
- Can we expose destination parameter/device views for direct `read_into`
  without violating PyTorch storage assumptions?  CPU segmented staging is now
  implemented, but direct parameter/device placement is still open.
- Which tensor slices are contiguous enough to read directly from safetensors
  without reading an enclosing large range?
- How should online quantization and `uses_meta_device` models integrate?
- Should this live only in UMA loader first, or become a generic vLLM loading
  interface after the prototype?

## Near-term recommendation

Do not add more model-specific direct paths to
`uma_odirect_safetensors_loader.py`.

Instead:

1. Refactor current loader into `TensorCatalog` + `WeightSource`.
2. Keep the existing iterator path for safety and compatibility.
3. Add an optional model hook for `WeightPlan`.
4. Prototype dense Qwen/Llama.
5. Move the current Qwen MoE direct optimization into a model-side plan.
   Qwen3 MoE is started; validation and cleanup remain.

This addresses the root issue: new models should add model-side placement
plans, not loader-side special cases.
