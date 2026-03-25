# expert_node_v2 重写计划

## 目标

本次重写只改 `expert_node_v2`，不新开 `server_v2`。现有 Python server / coordinator / validation 体系继续复用，只在协议层新增 3 个 scale kind：

- `w_up_scale`
- `w_gate_scale`
- `w_down_scale`

整体目标有三个：

1. 正确支持 DeepSeek 原生 FP8 block-scale 权重格式
2. 把 `up` 和 `gate` 融合，只保留两个 kernel，中间只落一个临时变量 `h`
3. 预留 oneAPI 支持，后续可按本机情况编译 CUDA / oneAPI / 双后端 binary

---

## 范围划分

### 保留不动
- 现有 `server/` 目录主体
- 现有 Python session / coordinator / validation 逻辑
- 现有 common protocol 框架
- 现有 `TensorKind` 编码方式
- 现有路由 / MoE runtime Python 调用链

### 已做最小协议扩展
`TensorKind` 已从三种扩成六种：

- `WUp = 0`
- `WGate = 1`
- `WDown = 2`
- `WUpScale = 3`
- `WGateScale = 4`
- `WDownScale = 5`

server 侧 `build_tensor_loader(cfg)` 也已支持六种 tensor kind。

### 新写
- `expert_node_v2/` 整个目录
- 新的 host tensor store
- 新的 DeepSeek block-scale loader
- 新的 CUDA backend
- 未来 oneAPI backend

---

## 目录结构

```text
expert_node_v2/
  expert_format_v2.h
  expert_runtime_v2.h
  expert_tensor_store_v2.h

  expert_tensor_store_v2.cc
  expert_loader_v2.cc
  expert_runtime_v2.cc

  cuda/
    backend_cuda_v2.h
    backend_cuda_v2.cc
    fp8_decode_lut_v2.cu
    matvec_blockscale_cuda_v2.cu
    mlp_blockscale_cuda_v2.cu

  oneapi/
    backend_oneapi_v2.h
    backend_oneapi_v2.cc
    matvec_blockscale_oneapi_v2.cpp
    mlp_blockscale_oneapi_v2.cpp
```

设计原则：

- 根目录只放公共结构、loader、runtime
- `cuda/` 只放 CUDA backend 和 `.cu`
- `oneapi/` 只放 oneAPI backend

---

## 核心设计

### 1. 不再沿用旧的 row-chunk scale 语义
旧路径的 scale 语义是：

- `weights`: `[rows, cols]`
- `scales`: `[rows, ceil(cols / k_chunk)]`

解码规则：

```text
decoded_weight = LUT[weights[r, c]] * scales[r, c / k_chunk]
```

这和 DeepSeek 原生权重格式不一致。

DeepSeek 原生 scale 语义是：

- `weights`: `[rows, cols]`
- `weight_scale_inv`: `[ceil(rows / 128), ceil(cols / 128)]`

解码规则：

```text
decoded_weight(r, c) =
    LUT[weights[r, c]] * scales[(r / 128), (c / 128)]
```

`expert_node_v2` 直接支持这种原生 block-scale 语义，不再尝试塞进旧的 `k_chunk=1024` 逻辑里。

---

### 2. weight 和 scale 分开存
不要把 weight 和 scale 混在一个 owning struct 里。

原因：

- 生命周期更清楚
- 协议本来就是分 tensor 发的
- 以后扩展别的 scale layout 更容易
- backend 层只需要在 launch 前组 view

---

### 3. 只保留两个 kernel
目标是把当前三段式算子收成两段：

#### kernel 1：fused up + gate
输入：

- `x[hidden_dim]`
- `W_up[inter_dim, hidden_dim]`
- `W_gate[inter_dim, hidden_dim]`

输出：

- `h[inter_dim]`

定义：

```text
up_i   = dot(W_up[i, :], x)
gate_i = dot(W_gate[i, :], x)
h_i    = silu(gate_i) * up_i
```

#### kernel 2：down
输入：

- `h[inter_dim]`
- `W_down[hidden_dim, inter_dim]`

输出：

- `y[hidden_dim]`

---

### 4. 中间临时变量 `h`
`h` 先用 `float`，理由：

- correctness 阶段更稳
- 真实 hidden 下中间激活范围更安全
- 后续要不要改成 bf16 / fp16 再单独评估

---

### 5. 先做 CUDA，oneAPI 先留接口
当前阶段优先级：

1. host tensor / loader
2. CUDA correctness
3. 单 expert 对拍
4. 再抽象 backend 接口
5. 最后补 oneAPI

也就是说，oneAPI 这次先做“可扩展结构”，不追求马上实现。

---

## 数据结构计划

### HostTensorV2
用于接协议发来的原始 tensor：

- bytes
- shape
- dtype
- ready

### ExpertTensorBundleV2
每个 expert 一共存六份 tensor：

- `w_up`
- `w_up_scale`
- `w_gate`
- `w_gate_scale`
- `w_down`
- `w_down_scale`

`all_ready()` 必须六份齐了才返回 true。

### MatrixHostBlockScaleV2
表示一张已经解析好的矩阵：

- meta
- weight buffer
- block scale buffer

### ExpertWeightsHostV2
一整个 expert 的三张矩阵：

- `w_up`
- `w_gate`
- `w_down`

后续 device 侧也做对称结构。

---

## server 与协议配合方式

### server 侧
server 继续走当前逻辑，不新开 v2：

- `build_tensor_loader(cfg)` 已支持六种 kind
- coordinator 改为发 six-pack，不再是 triplet

### 协议层
协议只扩 tensor kind，不改大框架。

### node_v2 侧
node_v2 负责：

1. 收六份 tensor
2. 判断 all_ready
3. 组装 host 权重
4. upload 到 device
5. 调用 CUDA / oneAPI backend

---

## 首批文件职责

### `expert_format_v2.h`
定义所有公共数据结构：

- `HostTensorV2`
- `ExpertTensorBundleV2`
- `MatrixMetaV2`
- `WeightBufferHostV2`
- `BlockScaleHostV2`
- `MatrixHostBlockScaleV2`
- `ExpertWeightsHostV2`

### `expert_tensor_store_v2.h/.cc`
负责：

- expert -> six tensors 存储
- chunk 拼接
- ready 状态
- debug dump

### `expert_loader_v2.cc`
负责把：

- `weight tensor`
- `scale tensor`

组装成：

- `MatrixHostBlockScaleV2`

重点校验：

- weight dtype 必须是 `torch.float8_e4m3fn`
- scale dtype 必须是 `torch.float32`
- scale shape 必须是 `ceil(rows/128), ceil(cols/128)`

### `cuda/backend_cuda_v2.h/.cc`
负责：

- backend 接口封装
- host -> device upload
- launch fused up+gate
- launch down

### `cuda/matvec_blockscale_cuda_v2.cu`
负责 DeepSeek block-scale decode matvec 基础逻辑。

### `cuda/mlp_blockscale_cuda_v2.cu`
负责两个 kernel：

- fused up+gate
- down

---

## 推荐实现顺序

### Phase 1：先把 loader 层写稳
1. `expert_format_v2.h`
2. `expert_tensor_store_v2.h`
3. `expert_tensor_store_v2.cc`
4. `expert_loader_v2.cc`

目标：

- server 六种 kind 能完整落到 host store
- 单个 expert 六份 tensor ready
- 能正确组装 host-side block-scale matrix

### Phase 2：CUDA host/device 通路
5. `cuda/backend_cuda_v2.h`
6. `cuda/backend_cuda_v2.cc`

目标：

- 把 host weights/scales 上传到 device
- 构造 backend 所需 view

### Phase 3：CUDA kernel
7. `cuda/fp8_decode_lut_v2.cu`
8. `cuda/matvec_blockscale_cuda_v2.cu`
9. `cuda/mlp_blockscale_cuda_v2.cu`

目标：

- 单矩阵 decode 正确
- fused up+gate 正确
- down 正确

### Phase 4：runtime 接起来
10. `expert_runtime_v2.h`
11. `expert_runtime_v2.cc`

目标：

- 从 node handler 收到 infer 请求
- 查到 device resident expert
- 调用两个 kernel
- 返回结果

### Phase 5：oneAPI 占位
12. `oneapi/backend_oneapi_v2.h`
13. `oneapi/backend_oneapi_v2.cc`

先保证：

- 接口存在
- 可以按编译条件启用 / 禁用
- 还不要求完整实现

---

## correctness 计划

### 第一优先级：single expert correctness
固定：

- layer 3
- 单 expert
- 真实 hidden

比较：

- Python reference
- v2 node 输出

只要这一步过了，后面的 top-k / router 就都能接上。

### 第二优先级：真实 hidden 回归
继续复用现有 `server/validation` 路线：

- 用真实 layer-3 hidden 作为输入
- 看 single expert / top-k 是否稳定

### 第三优先级：再做 top-k / full moe
等单 expert 完全对拍后再接回：

- `run_topk_moe_layer`
- `run_moe_layer`

---

## 风险点

### 风险 1：旧 row-chunk scale 心智干扰
v2 要彻底按 DeepSeek 原生 block-scale 写，不要再混旧 `k_chunk` 语义。

### 风险 2：server / node kind 不一致
server 已经能发 6 kinds，但 node_v2 必须严格按同样枚举接。

### 风险 3：中间精度
`h` 当前先用 `float`，等 correctness 过了再讨论压缩。

### 风险 4：oneAPI 过早分心
这次只预留接口，不在 Phase 1~4 阶段分散精力。

---

## 当前建议

立刻开始写：

1. `expert_node_v2/expert_format_v2.h`
2. `expert_node_v2/expert_tensor_store_v2.h`
3. `expert_node_v2/expert_tensor_store_v2.cc`
4. `expert_node_v2/expert_loader_v2.cc`

只要这 4 个文件落地，整个 v2 的 host-side 结构就定下来了，后面 CUDA backend 和 runtime 才容易接。
