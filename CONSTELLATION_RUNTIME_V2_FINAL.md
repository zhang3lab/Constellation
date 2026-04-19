# Constellation Runtime v2

A minimal end-to-end runtime for DeepSeek-V3.1 MoE inference and chat serving. The current goal is correctness, clean runtime structure, and easy debugging, not peak performance.

## Current status

Validated end to end on real DeepSeek-V3.1 weights across both expert runtime and full-model runtime paths.

Working paths now include:
- node startup and GPU discovery
- control-plane inventory
- placement generation and broadcast
- expert weight preload
- resident expert upload to GPU
- single-expert inference
- top-k dispatch and weighted combine
- repeated-run stability validation
- full backbone preload for DeepSeek-V3.1
- prefill runtime
- single-token decode runtime
- sampling runtime
- generation runtime
- manual HF-vs-runtime compare for target layers
- OpenAI-style `/v1/chat/completions` API
- non-streaming chat responses
- streaming SSE chat responses

Latest validation status: **CHAT COMPLETIONS PATH WORKING END TO END**.

## System model

The system now has three major layers.

### 1. Control plane (Python)
Responsible for:
- reading config
- discovering expert nodes
- collecting inventory
- building placement
- preloading expert weights
- maintaining routing metadata

### 2. Full-model runtime (Python + Torch/CUDA)
Responsible for:
- loading non-MoE backbone tensors
- building runtime partition and MLA runtime
- prefill over full prompts
- single-token decode
- sampling
- chat/generation orchestration
- API serving

### 3. Expert node runtime (C++/CUDA)
Responsible for:
- enumerating local GPUs
- starting one worker per GPU
- exposing one control endpoint per node
- exposing one worker endpoint per GPU
- receiving weight-loading commands
- storing tensor metadata and bytes
- uploading resident experts to GPU memory
- executing expert inference kernels

## Ports and protocol

Each node exposes:
- one **control port** per node
- one **worker port** per GPU

Use them strictly as follows:
- **control port**: inventory, placement, weight loading, preload lifecycle
- **worker port**: inference only

`InferRequest` must never be sent to the control port.

## Core identifiers

The runtime uses `worker_id` as the stable GPU/worker identifier inside a node. The old `local_gpu_id` path has been removed.

Inventory and placement records include at least:
- `host`
- `control_port`
- `worker_id`
- `worker_port`
- `gpu_name`
- memory/status fields

## Tensor loading model

Each routed expert is loaded as six tensors:
- `w_up`
- `w_up_scale`
- `w_gate`
- `w_gate_scale`
- `w_down`
- `w_down_scale`

Weight loading uses a begin/chunk/end protocol.

### Tensor metadata

`LoadWeightsBegin` carries `TensorMeta`:
- `shape: vector<uint64_t>`
- `dtype: string`
- `row_block: uint32_t`
- `col_block: uint32_t`

This metadata is stored on the node and used to validate and interpret tensors. The node accepts normalized dtype names only:
- weight dtype: `float8_e4m3fn`
- scale dtype: `float32`

The current DeepSeek loaders return block size `128 x 128` as tensor metadata.

## Internal node data model

`HostTensorV2` stores:
- raw `bytes`
- `TensorMeta meta`
- `ready`

`ExpertTensorBundleV2` groups the six tensors for one expert.

`ExpertRegistryV2` owns:
- incoming tensors for experts being loaded
- resident per-worker uploaded storage

Once all six tensors are present and marked ready, the registry builds a validated expert view and uploads the expert to the assigned worker GPU.

## Shape and block handling

The node no longer hardcodes DeepSeek matrix sizes or block sizes inside the loader path.

`BuildExpertWeightsViewV2(...)` now:
- reads matrix shapes from tensor metadata
- reads block size from tensor metadata
- validates that `w_up` and `w_gate` have matching shape
- validates that `w_down` is dimensionally consistent with them
- validates that each scale tensor shape matches the implied block layout

## Runtime execution model

The runtime now has three phases.

### 1. Control phase
Performed before runtime use:
- discover nodes
- fetch inventory
- build placement
- send placement plan
- preload resident experts

### 2. Full-model runtime initialization
Performed once per runtime/session:
- preload non-MoE backbone tensors
- create backbone store
- build MLA runtime
- build frequency cache by device
- initialize paged attention cache managers

### 3. Runtime phase
Performed during inference/generation:
- run prompt prefill
- run token decode
- route sparse layers to top-k experts
- send `InferRequest` to selected worker ports
- combine weighted expert outputs
- sample next tokens
- build chat completion responses

Weight loading is not part of the inference session.

## Backbone runtime

The full-model runtime currently supports:
- embedding
- all 61 decoder layers
- first 3 dense layers
- remaining sparse layers
- final norm
- lm_head
- paged KV cache
- MLA runtime

Backbone preload is partitioned across GPUs through a `LayerPartition`.

Supported partition types:
- `TwoGpuLayerPartition`
- `ExplicitLayerPartition`

Current chat/runtime validation uses explicit multi-GPU partitioning rather than the earlier fixed two-GPU split.

## Generation runtime

Generation is split into:
- `prefill_runtime.py`
- `decode_runtime.py`
- `sample_runtime.py`
- `generation_runtime.py`

The current generation flow is:
- tokenize prompt or chat template
- run prefill and obtain next-token logits
- sample first token
- loop decode one token at a time
- stop on EOS, stop-token sequence, or max token count
- decode output text with special tokens skipped

## Chat completions API

The runtime now exposes an OpenAI-style:
- `POST /v1/chat/completions`

Supported behavior:
- `messages`
- `temperature`
- `top_p`
- `max_tokens` / `max_completion_tokens`
- `stream=false`
- `stream=true`

Current API behavior:
- non-streaming returns standard chat completion JSON
- streaming returns SSE chunks with `chat.completion.chunk`
- final stream chunk carries `finish_reason`
- `[DONE]` is emitted at the end

The current implementation supports text-only chat completions. Tool use and multimodal features are not yet implemented.

## Manual compare utilities

Manual compare now exists on both sides:

### HF side
Used to:
- run layers up to a target layer
- save per-layer outputs
- save expert debug outputs
- save `final_hidden`
- save `final_norm_output`
- save `next_token_logits`
- save top-k decoded next-token logits as JSON

### Runtime side
Used to:
- run runtime layers up to a target layer
- save per-layer outputs
- save sparse/dense aux tensors
- save expert outputs
- save `final_hidden`
- save `final_norm_output`
- save `next_token_logits`
- save top-k decoded next-token logits as JSON

This made it possible to compare HF and runtime first-token distributions directly.

## Current scope

The current implementation is still intentionally narrow:
- correctness-first
- batch size 1 for most validation paths
- small resident subsets for expert-path debugging
- minimal API surface
- limited serving features
- no tool-calling or multimodal API support yet
- no performance tuning as a primary goal yet

## Repository structure

### Control plane
- `server/coordinator.py`: discovery, placement, preload
- `server/control_plane.py`: setup helpers and orchestration pieces

### Runtime/session
- `server/inference_session.py`: runtime/session state
- `server/backbone_store.py`: backbone preload and layer partitioning
- `server/prefill_runtime.py`
- `server/decode_runtime.py`
- `server/sample_runtime.py`
- `server/generation_runtime.py`

### API
- `server/chat_completions_adapter.py`
- `server/openai_api_runtime.py`
- `server/openai_api_app.py`

### Validation and compare
- `server/manual_compare/run_hf_single_layer_manual.py`
- `server/manual_compare/run_runtime_single_layer_manual.py`

### Expert runtime
- `server/moe_layer_runtime.py`
- `server/router_runtime.py`
- `expert_node_v2/`

## Config shape

The active config structure is still centered around:
- nodes
- model root
- run range
- KV cache sizing

Tests continue to use `server/test/config.json`.

## What was cleaned up

This version reflects the main cleanup decisions:
- old `expert_node` removed; `expert_node_v2` is the active node runtime
- `worker_id` replaces `local_gpu_id`
- control and data planes are separated by port
- tensor metadata is explicit and carried over protocol
- node-side tensor interpretation uses normalized metadata only
- full-model runtime is split into prefill/decode/sample/generation pieces
- chat completion serving is now part of the runtime stack
- compare tooling now saves aligned last-layer artifacts on both HF and runtime sides

## Recent fixes that mattered

Recent correctness fixes included:
- replacing fixed two-GPU backbone partitioning with explicit multi-GPU partitioning for runtime validation and serving
- aligning runtime and HF last-layer artifacts:
  - `final_hidden`
  - `final_norm_output`
  - `next_token_logits`
- validating first-token top-k logits directly between HF and runtime
- using tokenizer EOS rather than executor-local ad hoc EOS state
- decoding generated text with `skip_special_tokens=True`

These changes were required to move from structurally working but semantically broken output to correct chat completions output.

## Next likely work

Natural next steps are:
- turn manual acceptance checks into formal API smoke tests
- improve streaming from pseudo-streaming to token-by-token runtime streaming
- add batching / concurrency support
- improve cache-daemon observability and memory accounting
- expand partitioning and runtime configuration flexibility
- performance instrumentation and latency analysis
- future migration of runtime structure toward STELRA / new hardware backends
