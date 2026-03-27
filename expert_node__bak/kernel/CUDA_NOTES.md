# CUDA Notes

## Current working files
- `expert.h`
- `expert_pack.cu`
- `vector_ops.cu`
- `matvec_decode_btiled.cu`
- `mlp_b4.cu`
- `test_correctness_b4.cu`
- `bench_mlp.cu`

## Status
Current CUDA path works with the new `expert.h`.
Do **not** revert `expert.h` to the old version.

Correctness passed for:
- `Fp8Format::E4M3`
- `Fp8Format::E5M2`
- small shape: `hidden=269, inter=131`
- large shape: `hidden=7168, inter=2048`

## Important interface changes

### FP8 enum
Use:
- `Fp8Format::E4M3`
- `Fp8Format::E5M2`

Do not use:
- `FP8_E4M3`
- `FP8_E5M2`

### Shape
Use `MlpShape`.
Do not use old `MatvecShape`.

### Packed matrix fields
`PackedRowMajorMatrix` uses:
- `fp8_format`
- `weights`
- `scales`

Do not use old names like:
- `fmt`
- `data`

## Critical gotcha: down projection
For `w_down`, do **not** reuse the original `shape` directly.

Reason:
- `w_down.cols = inter_dim`
- original `shape.hidden_dim = hidden_dim`

Need a separate `down_shape` with:

```cpp
down_shape.hidden_dim = shape.inter_dim;
```

This was the main reason `launch_matvec_decode_from_float(w_down)` failed before.

## Supported k_chunk
Currently supported:
- `256`
- `512`
- `1024`

Currently unsupported:
- `2048`

If `k_chunk=2048` is passed, matvec will fail.

## Benchmark notes
`bench_mlp.cu` runs and results look reasonable.

Observed:
- for `7168x2048`, low-batch latency prefers `k_chunk=1024`
- at `batch=16`, `k_chunk=512` is better
- `E4M3` and `E5M2` have almost identical performance
- throughput approaches about `30k tok/s` on the tested setup

## If something breaks later
Check in this order:
1. `expert.h` field names vs `.cu` usage
2. `weights / scales / fp8_format` usage in packed matrices
3. `MlpShape` fields are fully filled:
   - `num_tokens`
   - `hidden_dim`
   - `inter_dim`
   - `k_chunk`
   - `rows_per_cta`
   - `fp8_format`
4. `w_down` uses a separate `down_shape`
5. `k_chunk` is not `2048`
6. link errors from mismatched function signatures, especially around `vector_ops.cu`

## Priority
Do not over-polish CUDA now.

Current goal is:
- keep correctness passing
- keep benchmark roughly sane
- stop changing interfaces
- move on to full system implementation
