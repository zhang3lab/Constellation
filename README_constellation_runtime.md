# Constellation Expert Runtime

A minimal multi-GPU runtime for DeepSeek-V3.1 MoE expert inference, with real expert weights, real router weights, persistent control connections, and deterministic single-token validation.

## Status

Current milestone:

- resident expert preload works
- per-session persistent connections work
- single-expert inference works
- multi-expert top-k dispatch works
- top-8 combine works
- real DeepSeek router works
- deterministic repeated-run stability verified
- multi-GPU LUT/device bug fixed

Validated on a small resident subset (`num_experts=8`) using real DeepSeek-V3.1 expert weights and real gate weights from layer 3.

---

## What this project does

This project implements a minimal end-to-end runtime for MoE expert inference:

1. discover worker nodes and GPUs
2. build an expert placement plan
3. preload resident experts onto worker GPUs
4. open an `InferenceSession` with persistent control connections
5. run single-token expert inference
6. route a token to top-k experts
7. combine expert outputs on the server side
8. validate correctness and deterministic stability

The current implementation focuses on **single-token**, **no batching**, and **small resident subsets** for development and debugging.

---

## Repository structure

### Control plane

- `server/coordinator.py`
  - node discovery
  - placement construction
  - placement broadcast
  - expert weight preload

### Session/runtime

- `server/inference_session.py`
  - per-session state
  - persistent client pool
  - cached router config/tensors

### Expert dispatch / MoE runtime

- `server/moe_layer_runtime.py`
  - `infer_one_expert(...)`
  - `dispatch_topk_experts(...)`
  - `combine_outputs(...)`
  - `run_topk_moe_layer(...)`

### Real router

- `server/router_runtime.py`
  - loads router config from model `config.json`
  - loads `mlp.gate.weight`
  - loads `mlp.gate.e_score_correction_bias`
  - implements DeepSeek-style `sigmoid + noaux_tc` routing
  - supports resident-subset routing for small-scale experiments

### Validation

- `server/expert_inference_validation.py`
  - single-expert correctness
  - multi-expert correctness
  - single-expert stability

- `server/validation_suite.py`
  - full validation entrypoint

- `server/test_utils.py`
  - common test helpers
  - safe input generation
  - stats / compare / stability printing

### Model lookup

- `server/model_locator.py`
  - resolves DeepSeek tensor names and shard locations

---

## Current execution model

The runtime is split into two phases.

### 1. Control phase

Performed before opening an inference session:

- discover nodes
- print node/GPU summary
- build placement
- send placement plan
- preload all resident experts

### 2. Runtime phase

Performed inside an `InferenceSession`:

- open persistent control connections
- run validation or demo inference
- dispatch top-k expert inference requests
- combine outputs locally

This separation is important:
**no weight loading is performed during an inference session**.

---

## Configuration

`server/config.json` currently uses this structure:

```json
{
  "nodes": [
    { "host": "192.168.1.10", "control_port": 5000 },
    { "host": "192.168.1.11", "control_port": 5000 }
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
    "layer_id": 3,
    "num_experts": 8
  }
}
```

### Notes

- `run.mode`
  - `"validation"`: run full validation suite
  - `"demo"`: run a single real-router demo
- `run.layer_id`
  - target MoE layer used for loading and router lookup
- `run.num_experts`
  - number of resident experts to place/preload for this run
  - this can be smaller than the model’s full `n_routed_experts`
- `model.expert_mem_bytes`
  - estimated memory footprint per resident expert used for placement
- `model.chunk_size`
  - weight transfer chunk size during preload

---

## Real router implementation

The real router is derived from the model config and gate tensors:

- `model.layers.{layer}.mlp.gate.weight`
- `model.layers.{layer}.mlp.gate.e_score_correction_bias`

For DeepSeek-V3.1 layer 3:

- `gate.weight` shape: `(256, 7168)`
- `e_score_correction_bias` shape: `(256,)`

Router settings are loaded from the model `config.json`:

- `scoring_func = sigmoid`
- `topk_method = noaux_tc`
- `n_group = 8`
- `topk_group = 4`
- `num_experts_per_tok = 8`
- `norm_topk_prob = true`
- `routed_scaling_factor = 2.5`

### Current routing behavior

Because development runs may preload only a subset of experts (for example `num_experts=8`), the router currently supports **resident-subset routing**:

- compute scores over all routed experts
- mask out non-resident experts
- choose top-k only from resident experts

This allows validating the real router logic without preloading all 256 routed experts.

---

## Validation coverage

The current validation suite checks:

### 1. Multi-expert correctness

Runs several resident experts individually and compares runtime output against local PyTorch reference using real DeepSeek expert weights.

### 2. Single-expert stability

Runs the same resident expert repeatedly and checks that outputs are exactly identical.

### 3. Top-8 reference compare

Uses a fixed top-8 route over resident experts and compares:

- per-expert runtime output vs local reference
- combined runtime output vs combined reference

### 4. Real-router demo + stability

Uses the real DeepSeek router:

- loads actual gate tensors
- computes real top-k routes
- dispatches resident experts
- combines outputs
- repeats multiple runs
- verifies exact deterministic stability

---

## Determinism

Current repeated-run tests show:

- identical `topk_idx`
- identical `topk_weight`
- identical combined output
- exact equality across repeated runs

This has been verified for the current single-token runtime path.

---

## Multi-GPU fix note

A key bug fixed during development:

### Problem

Single global FP8 decode LUT device pointers were reused across multiple GPUs.

This caused:
- GPU 0 experts to run correctly
- GPU 1 experts to fail with illegal memory access

### Fix

LUT state was changed from single-global to **per-device LUT cache**.

Important consequences:

- LUTs are uploaded per active CUDA device
- inference now works across multiple GPUs
- repeated expert dispatch across GPUs is stable

---

## Current limitations

This is still a development runtime. Current limitations include:

- single-token only
- no batching
- no attention-side integration yet
- no full 256-expert preload in normal development runs
- no end-to-end LLM decode loop yet
- resident-subset router masking is a development convenience, not the final full-scale serving path

---

## Typical workflow

### Validation run

```bash
python -m server.main
```

with:

```json
"run": {
  "mode": "validation",
  "layer_id": 3,
  "num_experts": 8
}
```

This performs:

- node discovery
- placement
- preload
- validation suite

### Demo run

```json
"run": {
  "mode": "demo",
  "layer_id": 3,
  "num_experts": 8
}
```

This runs a single real-router inference demo.

---

## Key runtime entrypoints

### Control plane setup

- `setup_control_plane(coord, cfg)`

### Validation

- `run_validation_suite(session)`

### Real router inference

- `run_one_token_moe_real_router(session, hidden, layer_id)`

### Generic top-k MoE execution

- `run_topk_moe_layer(session, hidden, routes)`

---

## Development notes

This project is intentionally kept small and explicit:

- fewer abstractions
- verbose logging
- strong correctness checks
- deterministic behavior prioritized over throughput features

The current codebase is meant to be a solid base for:

- integrating real upper-layer hidden states
- replacing fixed validation inputs with actual model activations
- scaling from resident subset routing to full routed-expert runtime
- later adding batching and higher-level decode orchestration

---

## Next steps

Recommended next steps:

1. connect upper-layer hidden states to the MoE runtime
2. add real-router reference compare against local PyTorch MoE combine
3. scale resident expert count upward
4. move from development-only resident-subset routing toward full routed-expert coverage
5. add batching only after the single-token path remains stable
