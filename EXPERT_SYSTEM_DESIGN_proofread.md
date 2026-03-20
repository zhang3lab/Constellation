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

### First version placement strategy
Use a very simple strategy:
- enumerate all GPUs in a stable order
- assign experts round-robin
- skip GPUs that cannot fit the target expert memory footprint

The first version ignores dynamic load and uses only static inventory and memory capacity.

This is enough to get the system working.

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

### Recommended first version
Send **raw weights** from server to expert node, and pack them locally on the node.

This is the most practical first implementation.

Benefits:
- server stays simpler
- node can choose the exact FP8 packing path
- naturally matches the current CUDA code structure
- easy to evolve later

### Why not pre-pack on server first
Possible later, but not necessary now.

Server-side pre-pack adds complexity:
- server must know exact target packing format
- less flexibility for heterogeneous GPU setups
- harder to evolve kernel-side packing assumptions

For now:
- server sends `w_up`, `w_gate`, `w_down`
- node packs
- node uploads
- node marks expert ready

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
- `PACKING`
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

### Server -> Expert Node control port
- `InventoryRequest`
- `PlacementPlan`
- `LoadWeightsBegin`
- `LoadWeightsChunk`
- `LoadWeightsEnd`
- `CommitServing`
- `HeartbeatRequest`

### Expert Node -> Server
- `InventoryReply`
- `LoadProgress`
- `LoadDone`
- `HeartbeatReply`
- `ErrorReport`

### Server -> GPU worker port
- `InferRequest`

### GPU worker -> Server
- `InferResponse`

This is enough for a first full system.

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
Implement a simple round-robin placement planner in Python.

Success condition:
- server can map every expert to a concrete GPU

### Step 4: weight loading
Implement one expert load path:
- server sends one expert's raw weights
- node packs and uploads
- node reports ready

Success condition:
- one expert instance is resident on one GPU

### Step 5: single-expert inference
Implement one inference request and one response path.

Success condition:
- server sends one request to one worker and gets a correct result back

### Step 6: multi-expert serving
Extend to multiple experts and full routing metadata.

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
- pack/upload completion
- expert ready
- inference request start/end
- error details

Readable logs will save a lot of time.

---

## CUDA Notes Integration

The current CUDA path already works and should be treated as a stable subsystem for now.

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

The system consists of a Python control-plane server and C++/CUDA expert nodes. The server first discovers all node inventories, computes global expert placement, sends raw expert weights to the assigned nodes, waits for all experts to become ready after local FP8 packing and GPU upload, and then enters serving mode where requests are routed directly to the responsible GPU workers.

---

## Immediate Next Step

The next practical steps are:

1. define the shared wire protocol
2. implement `InventoryRequest / InventoryReply`
3. make the Python server print the full global inventory of all expert nodes
