# UMA-Safe Weight Source Fork

This fork carries an experimental vLLM loader path for UMA systems such as
DGX Spark, where CPU RAM and GPU-visible memory are effectively the same scarce
resource.  The goal is safe model weight loading first: fail closed before
payload reads, avoid mmap/page-cache amplification, account for expected bytes,
and keep memory/PSI gates in the hot path.

This is not an upstream vLLM feature.  Treat it as a fork-local experiment until
the remaining real-checkpoint validation is complete.

## Loader Format

Use `load_format="uma_odirect_safetensors"` to select the O_DIRECT safetensors
loader.  It is intentionally narrower than vLLM's general safetensors loaders:

- local safetensors directories only;
- Linux with `O_DIRECT` support;
- no mmap, eager, prefetch, or implicit Hugging Face download fallback;
- bounded O_DIRECT read windows and chunked direct reads;
- memory, swap, and memory PSI gates before and during load;
- expected-vs-actual read accounting with warning on drift.

The older `uma_safetensors` Run:ai-streamer wrapper remains present, but the
primary safe path for this fork is `uma_odirect_safetensors`.

## Safety Model

The fork follows fail-closed rules:

- unsupported filesystem or tensor metadata errors stop the load;
- unsupported `read_segments` shapes stop before values are dispatched;
- `read_segments` must cover the staging tensor exactly once, with no gaps or
  overlaps;
- tensor parallel slice inference raises if a private vLLM helper changes
  unexpectedly;
- routed experts and shard loaders that report refusal through
  `return_success` stop the load;
- manual DeepSeek FP8 indexer reads are included in expected-byte accounting.

The shared UMA memory gate lives in
`vllm/model_executor/model_loader/_uma_memory_gate.py` and is used by both UMA
loader variants.

## WeightPlan IR

The loader uses a serializable `WeightPlan` IR to separate model-family
semantics from executor reads:

- `TensorCatalog` records safetensors metadata without reading payloads.
- `WeightPlanEntry` records source tensor name, target parameter, shard/expert
  metadata, source slices, segmented reads, and transform ops.
- `WeightPlanReadSegment` describes byte-source slices into explicit CPU
  staging tensors.
- `TransformOp` names deterministic post-read tensor operations.
- `schedule_weight_plan_reads()` orders required reads, predicts O_DIRECT
  windows, and reports expected read amplification before execution.

Stage A fused-source coalescing is implemented without changing the IR:
consecutive segmented entries that share one checkpoint tensor are read as one
source group in source-offset order while `weight_loader` dispatch remains in
scheduled entry order.

## Model-Family Coverage

The routed MoE helper path covers the migrated `*_uma.py` families in this
branch, including Qwen-family MoE, Mixtral, DeepSeek, GLM4, LongCat, HunYuan,
Llama4, MiniMax M2, MiMoV2, Param2MoE, Granite MoE, Ernie 4.5 MoE, Bailing,
Sarvam, AFMoE, Kimi Linear, LFM2, Laguna, Jamba, Nemotron-H, EXAONE, HYV3, and
OpenPangu.

The highest-risk segmented families are covered by small real-safetensors
fixtures that exercise the actual `_ODirectFile` path:

- TeleChat2 interleaved `key_value.weight` -> K/V staging tensors;
- HunYuan interleaved fused QKV -> Q/K/V staging tensors across multiple KV
  groups;
- Llama4 fused `gate_up_proj` -> local expert w1/w3 staging tensors.

OpenPangu currently uses stacked/source-slice entries rather than
`read_segments`, so it is not part of the segmented fixture set.

## Validation Snapshot

As of 2026-07-04:

- targeted loader/WeightPlan tests: `291 passed`;
- `git diff --check` clean for the current branch;
- three standard DGX Spark model-load smokes passed earlier with
  expected bytes equal to actual bytes and no swap or memory PSI;
- tiny real-safetensors fixtures validate segmented byte boundaries and
  O_DIRECT window behavior.

## Known Gaps

The segmented families still need at least one real-checkpoint model-load smoke
when a suitable checkpoint is available.  The tiny fixtures validate byte
boundaries and O_DIRECT execution, but they do not prove full model
construction, quantization metadata, tokenizer/config compatibility, or
end-to-end generation.

FlashInfer/CUDA JIT memory spikes are a separate post-load issue on UMA
systems.  Loader validation should continue to use `model_load` stop mode until
JIT cache warming is controlled.

Upstream tracking is also a fork-maintenance task.  This branch intentionally
does not rebase on every upstream movement; private vLLM API dependencies such
as shard-size helper reflection must be rechecked whenever upstream is
advanced.

## Pointers

- Roadmap: `docs/uma-safe-weight-plan-ir-roadmap.md`
- Upstream-style RFC draft: `docs/uma-safe-weight-plan-upstream-rfc.md`
- Design background: `docs/uma-safe-weight-source-design.md`
