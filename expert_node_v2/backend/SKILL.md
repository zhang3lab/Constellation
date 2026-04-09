# Adding a new backend

This document describes the expected steps for adding a new backend to `expert_node_v2/backend/`.

The goal is that a new backend should integrate cleanly into the existing system without requiring ad-hoc changes scattered across unrelated files.

Current backends follow the same high-level pattern:
- backend-specific gpu/device info probe
- backend-specific workspace creation
- backend-specific expert upload/free
- backend-specific expert execution
- backend registry entry
- backend tests
- build config entry

---

## 1. Create backend directory

Add a new directory:

- `backend/<new_backend>/`

Typical files are:

- `backend/<new_backend>/gpu_info_<new_backend>_v2.h`
- `backend/<new_backend>/gpu_info_<new_backend>_v2.cc`
- `backend/<new_backend>/backend_workspace_<new_backend>_v2.h`
- `backend/<new_backend>/backend_workspace_<new_backend>_v2.cc`
- `backend/<new_backend>/backend_<new_backend>_v2.h`
- `backend/<new_backend>/backend_<new_backend>_v2.cc`

Optional:
- backend-specific kernels
- backend-specific utility headers
- tests under `backend/<new_backend>/tests/`

---

## 2. Add a new `GpuVendor` enum value if needed

In `common/types.h`:

- add the new vendor to `enum class GpuVendor`
- update `kGpuVendorCount`
- update `gpu_vendor_name(...)`

Example pattern:

```cpp
enum class GpuVendor : std::uint32_t {
    Unknown = 0,
    Cpu = 1,
    Nvidia = 2,
    AMD = 3,
    Intel = 4,
    NewBackend = 5,
};
```

Important:
- keep the enum stable
- `vendor_spans[...]` and backend registry use the enum value as an index
- if you add a new vendor, update any fixed-size arrays accordingly

---

## 3. Implement gpu info probe

Every backend must expose static and dynamic gpu/device info builders.

Match this interface style:

```cpp
bool BuildLocal<Backend>GpuInfosV2(
    std::int32_t worker_id_begin,
    std::uint32_t worker_port_base,
    std::vector<common::StaticGpuInfo>* out);

bool BuildLocal<Backend>DynamicGpuInfosV2(
    std::int32_t worker_id_begin,
    std::vector<common::DynamicGpuInfo>* out);
```

### Static info should fill
- `worker_id`
- `gpu_name`
- `total_mem_bytes`
- `worker_port`
- `gpu_vendor`
- `capability_flags`
- `arch_name`

### Dynamic info should fill
- `worker_id`
- `free_mem_bytes`
- `gpu_status`

Important:
- returned workers must be contiguous from `worker_id_begin`
- local backend device index should correspond to:
  - `local_gpu_id = worker_id - vendor_span.worker_id_begin`

---

## 4. Implement backend workspace

Backends should expose a workspace type plus creation/free logic.

Typical pattern:
- `BackendWorkspace<Backend>V2`
- `CreateBackendWorkspaceV2(...)` dispatches to backend-specific workspace creation
- execution path uses the workspace for runtime scratch buffers or backend handles

Keep vendor-specific resource setup inside the backend implementation.

Do not put backend-specific device setup in generic worker/control code.

---

## 5. Implement expert upload/free

Every backend must provide expert residency operations:

```cpp
bool UploadExpert<Backend>V2(
    int local_gpu_id,
    const ExpertTensorBundleV2& host_bundle,
    ExpertDeviceStorageV2* out_storage);

void FreeExpertWeights<Backend>V2(
    ExpertDeviceStorageV2* storage);
```

These are required for:
- ExpertRegistryV2::Update(...)
- resident replacement
- registry cleanup
- future backend-agnostic expert lifecycle management

Important:
- upload/free must be symmetric
- FreeExpertWeights<Backend>V2(...) must fully release backend-owned storage
- FreeExpertWeights<Backend>V2(...) must leave storage in a fully reset state
- UploadExpert<Backend>V2(...) must leave output storage in a fully reset state on failure
- callers must not rely on a generic clear() helper to clean up backend-managed device storage

---

## 6. Implement expert execution

Backends that can execute experts should expose:

- fused up/gate
- down
- full expert execution

Typical entry points:

```cpp
bool RunExpert<Backend>V2(...);
```

If the backend does not support execution yet, do not partially wire it in as if it were complete.
A backend is considered available only when all required backend registry functions exist.

---

## TODO: kernel implementation conventions

Kernel-writing conventions are not fully documented yet.

We have not finalized a backend-agnostic style guide for:
- kernel file layout
- launch wrapper structure
- dtype/activation conversion helpers
- temporary buffer ownership
- tiling/thread-block conventions
- correctness/benchmark test shape for backend kernels

When AMD or Intel kernels are implemented in earnest, add a dedicated section here covering:
- expected kernel file organization
- naming conventions
- launch API conventions
- shared helper patterns
- test expectations
- performance measurement conventions

---

## 7. Register backend in backend registry

Update:

- `backend/backend_registry_v2.h`
- `backend/backend_registry_v2.cc`

Each backend registry entry must provide all of:

- `build_static`
- `build_dynamic`
- `upload_expert`
- `free_expert_weights`

The registry uses `GpuVendor` index positions. The entry for a backend should be stored at:

- `registry[static_cast<std::size_t>(common::GpuVendor::<Vendor>)]`

Backends that are not enabled should simply not be compiled into the registry entry.

A backend is considered available only if all required function pointers are present.

---

## 8. Hook backend into workspace dispatch

Update generic backend dispatch points such as:

- `backend/backend_workspace_v2.cc`
- any generic backend creation helpers

Pattern:
- derive `local_gpu_id` from vendor span
- switch on `GpuVendor`
- construct backend workspace for that vendor

Do not duplicate vendor-specific logic in unrelated files if it belongs in backend registry or backend-specific implementation.

---

## 9. Hook backend into node info

If backend registry is wired correctly, `node_info.cc` should already pick up the new backend by iterating the registry.

Check:
- static node info includes the new backend span
- dynamic node info includes the new backend dynamic devices
- `worker_id` assignment remains contiguous across all vendors

---

## 10. Hook backend into expert registry

If backend registry is wired correctly, `ExpertRegistryV2::Update(...)` should already pick up the new backend through:

- `FindBackendRegistryEntryV2(...)`

Check:
- resident upload works
- resident replacement frees old storage
- cleanup path frees backend storage correctly

---

## 11. Add build config entries

Update `build_support/config.py`:

### Backend source list
Add backend sources under `BACKENDS`.

Example shape:

```python
"new_backend": {
    "enabled": ENABLE_NEW_BACKEND,
    "src": [
        "backend/new_backend/gpu_info_new_backend_v2.cc",
        "backend/new_backend/backend_workspace_new_backend_v2.cc",
        "backend/new_backend/backend_new_backend_v2.cc",
    ],
},
```

### Feature define
Add:
- `EXPERT_NODE_V2_ENABLE_NEW_BACKEND`

### If special compilation is needed
Prefer routing it through:
- `SOURCE_RULES`
- backend-specific toolchain module

Do not hardcode new backend behavior in unrelated build code if it can be expressed through config.

---

## 12. Add tests

At minimum add:

- `backend/<new_backend>/tests/test_gpu_info_<new_backend>_v2.cc`

If execution is supported, also add:
- correctness test for fused up/gate
- correctness test for down
- correctness/benchmark test for full expert execution

Tests should be backend-local whenever possible.

General codec or shared logic tests belong under:
- `backend/tests/`

Backend-specific tests belong under:
- `backend/<new_backend>/tests/`

---

## 13. Add regression coverage

Update `build_support/config.py` test targets so the backend tests appear in `TEST_TARGETS`.

If regression is driven from `TEST_TARGETS` + backend enablement, no further build system changes should be needed.

The intended workflow is:
- enable backend in config
- add sources and tests in config
- run regression
- new backend tests are picked up automatically

---

## 14. Validate invariants

Before considering the backend integrated, verify:

### Registry
- backend registry entry exists
- all four required function pointers are present

### Worker indexing
- backend workers are contiguous
- `vendor_spans[vendor]` matches actual worker ids
- `local_gpu_id = worker_id - vendor_span.worker_id_begin` is valid

### Memory
- static total memory is nonzero
- dynamic free memory is nonzero when expected
- dynamic free memory does not exceed static total memory

### Upload/free
- upload succeeds
- replacing a resident frees previous storage
- registry clear path does not leak backend storage

### Tests
- backend-local gpu info test passes
- correctness tests pass
- regression includes the new backend when enabled

---

## 15. Design rules

When adding a backend, prefer these rules:

### Put backend-specific logic in backend code
Do not spread backend-specific code across:
- `node_info.cc`
- `expert_registry_v2.cc`
- `worker.cc`
- build scripts

if the logic can live in:
- backend registry
- backend implementation files

### Use registry instead of repeated vendor switches
If a new vendor forces repeated `switch (vendor)` edits in multiple files, that usually means a backend op should move into the registry.

### Keep static and dynamic info separate
Follow the existing split:
- static info = identity/capacity
- dynamic info = free memory/status

### Do not partially define an available backend
A backend should be considered available only if it has the full required backend registry surface.

---

## 16. Minimal checklist

When adding a backend, make sure all of these are done:

- [ ] add new `GpuVendor` if needed
- [ ] add backend directory and source files
- [ ] implement static gpu info builder
- [ ] implement dynamic gpu info builder
- [ ] implement workspace creation
- [ ] implement expert upload
- [ ] implement expert free
- [ ] implement execution path if supported
- [ ] register backend in backend registry
- [ ] add backend sources to build config
- [ ] add backend tests
- [ ] add test targets to build config
- [ ] run regression with backend enabled

---

## 17. Current required backend registry surface

A backend is considered fully integrated only when it provides:

- static gpu info builder
- dynamic gpu info builder
- expert upload
- expert free

Execution support may still evolve, but the registry-backed integration assumes these four exist and are wired.

If any one of these is missing, the backend should not be treated as available.
