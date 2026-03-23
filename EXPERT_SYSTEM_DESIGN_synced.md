# Distributed Expert Serving Design

## Goal

Build a minimal but complete distributed expert-serving system around the current CUDA expert kernels.

The immediate goal is **not** to optimize every subsystem. The immediate goal is to make the full system path work end to end:

1. expert nodes start up
2. server discovers all expert nodes and GPUs
3. server decides expert placement
4. server sends raw weights to the assigned expert nodes
5. expert nodes locally pack the weights to FP8 and upload them to GPU memory
6. all assigned experts become ready
7. server enters serving mode and routes expert requests to the correct GPU workers

This design intentionally prioritizes simplicity, explicit state transitions, and easy debugging.

---

## High-Level Architecture

The system has two roles:

### 1. Server side (Python)
Responsible for:
- reading cluster configuration
- discovering all expert nodes
- collecting GPU inventory
- building expert placement
- sending weights
- coordinating the global ready barrier
- entering serving mode
- maintaining `expert_id -> worker endpoint` routing metadata

### 2. Expert node side (C++ / CUDA)
Responsible for:
- enumerating local GPUs
- starting one worker per GPU
- exposing one control endpoint per node
- receiving placement and weight-loading commands
- locally packing weights to FP8
- uploading packed weights to device memory
- holding resident expert instances
- executing inference with the existing CUDA kernels

---

## Design Boundary

### Python server should own
- orchestration
- cluster discovery
- inventory gathering
- placement planning
- weight dispatch
- global state transitions
- expert-to-worker routing metadata
- control-plane logic

### C++ expert node should own
- GPU lifecycle
- device memory ownership
- CUDA stream management
- weight packing and upload
- expert residency
- kernel launch
- per-GPU inference execution

This boundary keeps CUDA complexity on the node side and keeps cluster logic easy to iterate in Python.

---

## Expert Node Structure

Each machine runs exactly one `expert_node` process.

Inside that process:

### Control plane
A single **control port** per node.

Used for:
- inventory request / reply
- placement delivery
- weight-loading commands
- load progress / ready notifications
- heartbeat
- error reporting

### Data plane
One **worker port** per GPU.

Used for:
- inference requests
- inference responses

### Threads
Recommended minimal layout:
- 1 control thread
- N GPU worker threads, one per GPU

Optional later:
- a dedicated network I/O thread
- background health monitoring

For now, keep it simple.

---

## Why Separate Control Port and Worker Ports

Do **not** use “connect to any worker first and then ask for other ports” as the main protocol.

Instead:

- server always connects to the node's control port first
- control plane returns complete inventory, including all worker ports
- server uses worker ports only after placement is committed

This keeps the protocol much cleaner and makes debugging easier.

---

## Startup Flow

### Phase 1: expert nodes boot
Each expert node:

1. enumerates local GPUs
2. starts one worker thread per GPU
3. assigns one worker port per GPU
4. starts a control listener
5. becomes ready for server discovery

### Phase 2: server discovery
Server reads configured node addresses and connects to each node's control port.

For each node, server requests inventory and receives:

- node id
- number of GPUs
- for each GPU:
  - local GPU id
  - GPU name
  - total memory
  - free memory estimate
  - worker port
  - current status

Server aggregates this into a global GPU inventory.

### Phase 3: placement
After **all** nodes have replied, server computes a global placement plan.

Do not compute placement or start loading weights until global discovery is complete.

### Phase 4: loading
Server sends placement assignments and then sends weights to the appropriate nodes.

In the first version, the server sends each expert's `w_up`, `w_gate`, and `w_down` tensors to the assigned node.

Each target GPU worker:
- receives raw expert weights
- packs them locally to FP8
- uploads packed weights to GPU memory
- constructs the resident expert instance
- reports ready

### Phase 5: ready barrier
Server waits until every assigned expert reports ready.

Only after the full placement is ready should the system transition to serving mode.

### Phase 6: serving
Server maintains routing metadata and sends expert requests directly to the assigned worker ports.

---

## Node Inventory

Each GPU should have a globally unique identifier:

`gpu_uid = <node_id>:<local_gpu_id>`

This identifier should be used everywhere:
- placement
- logging
- heartbeats
- failures
- routing

Each GPU inventory record should include at least:

- `gpu_uid`
- `node_id`
- `local_gpu_id`
- `name`
- `total_mem_bytes`
- `free_mem_bytes`
- `worker_port`
- `status`

---

## Placement

The server should compute placement only after collecting the complete global inventory.

### Current placement strategy
The current implementation uses a simple memory-aware static placement strategy rather than pure round-robin:
- collect the full global GPU inventory first
- consider only GPUs whose `free_mem_bytes` can fit the requested expert footprint
- spread experts across eligible GPUs in a stable order rather than packing one large GPU first
- produce a concrete per-expert assignment with node address, control port, local GPU id, worker port, and globally unique GPU id

This is still intentionally simple and synchronous, but it already reflects actual available memory rather than only topology order.

### Placement output
Placement should be a concrete object, not implicit logic.

For each expert, placement should specify:
- `expert_id`
- `node_id`
- `local_gpu_id`
- `gpu_uid`

Optional metadata:
- `fp8_format`
- `k_chunk`
- `rows_per_cta`
- hidden/intermediate dims

---

## Weight Loading Strategy

### Current implementation stage
The current implementation sends **real tensor bytes** from Python to the expert node using a chunked control-plane protocol.

For DeepSeek-V3.1, the server currently:
- reads `model.safetensors.index.json`
- resolves a requested tensor such as `model.layers.<layer>.mlp.experts.<expert>.{up,gate,down}_proj.weight` to a concrete shard
- loads the tensor from the `.safetensors` shard
- sends the tensor as `LoadWeightsBegin / LoadWeightsChunk / LoadWeightsEnd`

The expert node currently:
- receives chunked tensor bytes
- assembles them into a host-side buffer
- preserves the original `torch.float8_e4m3fn` weight bytes from the model shards
- builds resident packed matrices whose `weights` buffer is the raw model byte stream, whose `scales` are currently all `1.0f`, and whose `fp8_format` is explicitly marked as `TORCH_E4M3FN`
- uploads the resident weight representation to the target GPU
- marks an expert ready only after `w_up`, `w_gate`, and `w_down` are all present and a consolidated `LoadedExpert` has been registered in runtime

### Current format bridge
The current working path no longer repacks DeepSeek expert weights into the older IEEE-like test FP8 format.

Instead, the working production path is:
- server resolves and reads real `torch.float8_e4m3fn` tensor bytes
- node receives and stores the raw bytes
- node constructs resident matrices tagged as `TORCH_E4M3FN`
- node sets per-chunk scales to `1.0f`
- CUDA decode uses a dedicated `TORCH_E4M3FN` LUT path in `matvec_decode_btiled.cu`
- runtime inference consumes that resident format directly

The older `IEEE_E4M3` and `IEEE_E5M2` paths are still kept for synthetic tests and kernel self-checks.

---

## Expert Residency on the Node

Each loaded expert instance should own:

- expert id
- shape metadata
- FP8 format
- `k_chunk`
- packed device weights/scales
- `DeviceMlpView`
- any associated temporary buffers if needed

Inference on that expert should ultimately call the already working CUDA path.

The current CUDA working set is under:

- `expert_node/kernel/expert.h`
- `expert_node/kernel/expert_pack.cu`
- `expert_node/kernel/vector_ops.cu`
- `expert_node/kernel/matvec_decode_btiled.cu`
- `expert_node/kernel/mlp_b4.cu`

---


## Current Implemented Status Snapshot

As of the current implementation, the following path is already working end to end:

1. Python server discovers all configured expert nodes
2. server collects full GPU inventory including worker ports and free memory
3. server computes a concrete placement for experts
4. server sends `PlacementPlan` and nodes materialize expert residency tables
5. server resolves DeepSeek expert tensors from `model.safetensors.index.json`
6. server reads real `torch.float8_e4m3fn` tensors from `.safetensors` shards
7. server streams tensor bytes using `LoadWeightsBegin / Chunk / End`
8. node assembles host buffers and constructs resident matrices tagged as `TORCH_E4M3FN`
9. node uploads resident weights to the assigned GPU and registers the expert in `ExpertRuntime`
10. server sends a real `InferRequest` carrying float32 activation bytes
11. node converts activations to half, looks up the resident expert, and runs the real CUDA MLP path
12. node returns a real `InferResponse` containing half-precision output bytes
13. Python validation code decodes the output and compares it against a PyTorch reference built from the same expert weights
14. repeated inference on the same expert and input is bitwise stable across multiple runs

This means the current system already has a working control plane, real loading path, runtime registry, real CUDA execution path, and correctness/stability validation for single-token expert inference.

---

## Control-Plane State Machine

### Node-level states
- `BOOTING`
- `REGISTERED`
- `ALLOCATED`
- `LOADING`
- `READY`
- `SERVING`
- `FAILED`

### GPU worker states
- `INIT`
- `IDLE`
- `LOADING`
- `READY`
- `BUSY`
- `FAILED`

### Expert instance states
- `EMPTY`
- `ASSIGNED`
- `MATERIALIZING`
- `UPLOADING`
- `READY`
- `FAILED`

Make these states explicit. Do not rely on implicit “half-loaded” conditions.

---

## Ready Barrier

A global ready barrier is important.

Bad behavior:
- some GPUs start serving while others are still loading
- server begins routing to incomplete placements
- requests fail inconsistently during startup

Correct behavior:
- all experts assigned in the placement report ready
- server verifies the full ready set
- server issues one explicit transition into serving mode

This should be a hard rule in the first version.

The server must not publish or use the routing table until the full ready barrier has completed.

---

## Minimal Wire Protocol

The protocol should be simple and language-neutral so Python and C++ can interoperate cleanly.

The first version should use a small fixed binary protocol rather than introducing a full RPC framework.

### Message header
Use a small fixed-size binary header, for example:

- `magic`
- `version`
- `msg_type`
- `request_id`
- `body_len`

Then follow with a raw byte payload.

### Encoding
Do not over-engineer this yet.

First version can use:
- fixed-size fields for simple metadata
- length-prefixed variable-size buffers
- raw byte payloads for tensors / weight chunks

This is enough to get Python server and C++ expert nodes talking.

---

## Minimal Message Types

### Implemented control-port message types
- `InventoryRequest`
- `InventoryReply`
- `PlacementPlan`
- `PlacementAck`
- `LoadWeightsBegin`
- `LoadWeightsChunk`
- `LoadWeightsEnd`
- `LoadWeightsAck`
- `HeartbeatRequest`
- `HeartbeatReply`
- `InferRequest`
- `InferResponse`

The current implementation is still using the node control port for both loading and the validated first execution path. A separate worker-port execution path remains a later cleanup / scale-out step.

---

## Serving Path

After startup and loading, server maintains:

`expert_id -> (node_addr, worker_port, gpu_uid)`

Then serving works like this:

1. the server-side routing logic decides which experts are selected
2. server slices the input for each target expert
3. server sends each slice to the corresponding worker port
4. worker runs expert inference on its resident expert
5. worker returns output
6. server gathers outputs and continues the pipeline

This first version can be synchronous and blocking.

No need for advanced async machinery yet.

---

## What the First Version Should Not Do

Do **not** try to implement all of the following immediately:

- dynamic rebalancing
- hot migration
- automatic failover
- dynamic replication management
- background re-packing
- complicated async runtime
- RDMA
- zero-copy networking
- live rolling updates

All of these can come later.

The first target is simply:

**discover -> place -> load -> ready -> serve**

---

## Recommended Development Order

### Step 1: define the protocol
Stabilize the wire protocol between Python server and C++ expert node.

### Step 2: node inventory
Implement expert node startup and inventory reporting.

Success condition:
- server can connect to each node
- server can print a correct global GPU inventory

### Step 3: placement
Implemented.

Current success condition already achieved:
- server maps every expert to a concrete GPU using full-cluster inventory and memory-fit checks

### Step 4: weight loading
Implemented in a real-data but still simplified form.

Current success condition already achieved:
- server resolves a real DeepSeek expert tensor from `model.safetensors.index.json`
- node receives the tensor via chunked protocol
- node assembles host bytes and uploads them to the assigned GPU

### Step 5: single-expert inference
Implemented with the real CUDA expert path.

Current success condition already achieved:
- server sends `InferRequest` with a real activation payload
- node uploads activation bytes to the correct GPU and converts them to half
- runtime looks up the loaded expert and runs the real CUDA MLP path
- node returns a real `InferResponse` containing half output bytes
- Python-side validation compares the response against a PyTorch reference built from the same DeepSeek expert weights

### Step 6: validation status
Current validation already achieved:
- multiple experts have been loaded from real DeepSeek-V3.1 shards and matched against PyTorch reference outputs
- the server-side single-token path is numerically close to reference (`cos` essentially 1, `max_abs` around `1e-3` after fp16 output rounding)
- repeated inference on the same expert and input is bitwise stable across multiple runs

### Step 7: near-term next step
Keep the validated server-side single-token path stable, and move batching and higher-level scheduling work to the expert-side execution path.

A useful practical baseline now exists for regression testing:
- multi-expert single-token correctness against PyTorch reference
- repeated-call bitwise stability on representative experts
- cross-node placement where different experts can reside on different GPU types

---

## Observability

Even the first version should log clearly.

### Server logs
- node discovery
- GPU inventory
- placement decisions
- weight loading progress
- ready barrier completion
- routing decisions

### Expert node logs
- GPU worker startup
- incoming placement
- weight chunk receive progress
- resident-materialization / upload completion
- expert ready
- inference request start/end
- error details

Readable logs will save a lot of time.

---

## CUDA Notes Integration

The current CUDA path already works for validated single-token expert inference and should be treated as a stable subsystem for now.

Important constraints already known:
- use the new `expert.h`
- keep the current migrated kernel path
- supported `k_chunk` values are `256`, `512`, `1024`
- `2048` is currently unsupported
- the current CUDA path contains a known internal shape adjustment for `w_down`; do not rewrite this path casually

Do not reopen CUDA interface churn unless necessary.

---

## First-Version Principle

The first version should be:

- explicit
- synchronous
- easy to inspect
- easy to log
- easy to restart
- easy to reason about

Not:
- maximally fast
- maximally elegant
- heavily abstracted
- highly dynamic

The system only needs to be “clean enough” and “working enough” to support the next stage.

---

## One-Sentence Summary

The system consists of a Python control-plane server and C++/CUDA expert nodes. The server discovers all node inventories, computes memory-aware global expert placement, resolves real DeepSeek expert tensors from sharded safetensors, streams raw `torch.float8_e4m3fn` expert bytes to the assigned nodes, the nodes materialize resident `TORCH_E4M3FN` weight views on the correct GPUs, register ready experts in a runtime table, and then accept `InferRequest` messages carrying activation payloads that are executed through the validated CUDA expert path and returned as `InferResponse`.

---

## Immediate Next Step

The next practical steps are:

1. keep the validated server-side single-token path as a regression-tested baseline
2. move batching into the expert-side execution path rather than the Python server
3. connect higher-level routing / scheduling logic to multiple already-loaded experts
4. only revisit resident weight format work if a later performance pass shows the direct `TORCH_E4M3FN` path is insufficient
