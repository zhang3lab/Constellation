# Constellation Expert Runtime v2

A minimal end-to-end runtime for DeepSeek-V3.1 MoE expert inference. The current goal is correctness, clear control/data-plane separation, and easy debugging, not peak performance.

## Current status

Validated end to end on a small resident subset using real DeepSeek-V3.1 weights and real router weights.

Working path:
- node startup and GPU discovery
- control-plane inventory
- placement generation and broadcast
- expert weight preload
- resident expert upload to GPU
- single-expert inference
- top-k dispatch and weighted combine
- repeated-run stability validation

Latest validation status: **ALL VALIDATION PASSED**.

## System model

The system has two roles.

### Server side (Python)
Responsible for:
- reading config
- discovering expert nodes
- collecting inventory
- building placement
- preloading expert weights
- maintaining routing metadata
- opening inference sessions
- dispatching expert inference requests
- combining outputs and running validation

### Expert node side (C++/CUDA)
Responsible for:
- enumerating local GPUs
- starting one worker thread/process per GPU
- exposing one control endpoint per node
- exposing one worker endpoint per GPU
- receiving weight-loading commands
- storing tensor metadata and bytes
- uploading resident experts to GPU memory
- executing inference kernels

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

Each expert is loaded as six tensors:
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

The runtime has two phases.

### 1. Control phase
Performed before inference session creation:
- discover nodes
- fetch inventory
- build placement
- send placement plan
- preload resident experts

### 2. Runtime phase
Performed during inference:
- route token to top-k experts
- send `InferRequest` to the selected workers' `worker_port`
- receive expert outputs
- combine weighted outputs on the server side

Weight loading is not part of the inference session.

## Current scope

The current implementation is intentionally narrow:
- single token
- batch size 1
- hidden size 7168 for the current DeepSeek path
- FP16 input/output on the inference path
- small resident subsets for debugging and validation

## Repository structure

### Control plane
- `server/coordinator.py`: discovery, placement, preload
- `server/main.py`: config loading and orchestration

### Runtime/session
- `server/inference_session.py`: session state and client pool
- `server/moe_layer_runtime.py`: expert dispatch and combine
- `server/router_runtime.py`: router loading and routing

### Validation
- `server/expert_inference_validation.py`
- `server/validation_suite.py`
- `server/test_utils.py`

### Model utilities
- `server/model_locator.py`
- `server/make_model_pkg.py`
- `server/bootstrap_env.py`

### Node runtime
- `expert_node_v2/`

## Environment/bootstrap

A one-shot bootstrap script can:
- install the flash-attn wheel based on local Python / torch / CUDA version
- install `requirements.txt`
- generate a writable package shell for the model directory

The generated package is typically placed under:
- `tmp/DeepSeek_V3_1`

and must include:
- `configuration_deepseek.py`
- `modeling_deepseek.py`

## Config shape

The active config structure is:

```json
{
  "verbose": false,
  "nodes": [
    { "host": "192.168.1.10", "control_port": 40000 },
    { "host": "192.168.1.11", "control_port": 40000 }
  ],
  "model": {
    "family": "deepseek_v3",
    "root": "/model/ModelScope/deepseek-ai/DeepSeek-V3.1",
    "chunk_size": 1048576,
    "expert_mem_bytes": 2147483648,
    "memory_utilization": 0.9
  },
  "run": {
    "mode": "validation",
    "start_layer": 0,
    "end_layer": 60,
    "collect_per_layer": true,
    "experts_per_layer": 256,
    "sparse_layer_start": 3,
    "sparse_layer_end": 60,
    "num_experts": 8
  },
  "kv_cache": {
    "max_batch_size": 1,
    "max_seq_len": 256,
    "page_size": 128,
    "use_triton": true
  }
}
```

Tests use a separate config at `server/test/config.json`.

## What was cleaned up

This version reflects the main cleanup decisions:
- old `expert_node` removed; `expert_node_v2` is the active runtime
- `worker_id` replaces `local_gpu_id`
- control and data planes are separated by port
- tensor metadata is explicit and carried over protocol
- node-side tensor interpretation uses normalized metadata only
- documentation is reduced to one concise runtime note

## Next likely work

Natural next steps are:
- multi-node larger resident subsets
- batching / concurrency work
- more flexible dtype and shape support
- performance instrumentation and latency analysis
- less ad hoc validation/demo code around the runtime core
