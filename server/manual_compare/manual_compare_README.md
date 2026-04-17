# Manual Compare Utilities

这个目录放的是一组手工对拍脚本，用来验证：

- HF reference 路径
- runtime 路径
- HF cache reference 路径

在单层中间张量、最终 hidden/logits，以及 router / MoE 调试信息上的一致性。

---

## 目录说明

### `make_compare_input.py`
生成统一输入，供 HF / runtime / HF cache 三边复用。

---

### `make_partial_deepseek_ref_dir.py`
构造 reference 模型目录，并将 patched modeling 文件拷贝进去。

常见用途：
- 构造 full reference
- 构造 restricted reference
- 构造 HF cache reference

---

### `run_hf_single_layer_manual.py`
HF 单层停靠脚本。

功能：
- 跑到指定层
- 保存前面所有层输出
- 保存目标层调试张量
- 如果目标层是最后一层，也保存：
  - `final_hidden`
  - `logits`

支持：
- dense / sparse 层
- 多 GPU 按层切分
- HF cache borrowed expert 路径

---

### `run_runtime_single_layer_manual.py`
runtime 单层停靠脚本。

功能：
- 跑到指定层
- 保存前面所有层输出
- 保存目标层调试张量
- 如果目标层是最后一层，也保存：
  - `final_hidden`
  - `logits`

特点：
- sparse 层会额外保存 token 级 router / topk / expert 输出调试信息

---

### `compare_universal_single_layer_outputs.py`
通用单层对拍脚本。

功能：
- 比较 HF 和 runtime 导出的中间张量
- 支持比较最后一层的：
  - `final_hidden`
  - `logits`

---

### `compare_attention_backends.py`
统一的 attention backend 对拍入口。

支持：
- `shallowmla`
- `flashmla`
- `all`

---

### `modeling_deepseek_absorbed.py`
HF reference 路径使用的 patched modeling 文件。

用途：
- 吸收 absorbed-latent / restricted experts / full experts / cache borrow 等改动
- 作为 HF 单层停靠和 reference 对拍的模型定义来源

---

### `cache_daemon.py`
expert cache daemon。

功能：
- 维护 CPU resident / GPU resident expert cache
- 提供 batch borrow / batch return
- 通过 CUDA IPC handle 将 borrowed expert 暴露给 HF cache reference

当前语义：
- CPU resident 保留
- GPU resident 按需 promote
- return 后 GPU evict，CPU resident 保留
- client disconnect 时自动回收 lease 并清理 GPU resident

---

### `cache_client.py`
与 cache daemon 对话的客户端。

支持：
- `borrow_expert_batch`
- `return_expert_batch`
- `query`

---

### `borrow_test.py`
单 expert borrow correctness 测试脚本。

---

### `borrow_topk_test.py`
一组 top-k expert borrow / combine correctness 测试脚本。

---

### `query_cache.py`
查看 cache daemon 当前 CPU/GPU resident 状态。

---

## 典型流程

### 1. 生成统一输入
```bash
python -m server.manual_compare.make_compare_input \
  --model-dir tmp/deepseek_absorbed_full_ref \
  --prompt "Hello world" \
  --output tmp/manual_compare/input.json
```

---

### 2. 构造 HF reference 模型目录

#### full reference
```bash
python -m server.manual_compare.make_partial_deepseek_ref_dir \
  --config server/test/config_compare256.json \
  --dst tmp/deepseek_absorbed_full_ref \
  --patched-modeling tmp/patched_modeling/modeling_deepseek_absorbed.py \
  --mode full \
  --force
```

#### cache reference
```bash
python -m server.manual_compare.make_partial_deepseek_ref_dir \
  --config server/test/config_compare256.json \
  --dst tmp/deepseek_absorbed_cache_ref \
  --patched-modeling tmp/patched_modeling/modeling_deepseek_absorbed.py \
  --mode full \
  --force
```

---

### 3. 跑 HF reference
```bash
python -m server.manual_compare.run_hf_single_layer_manual \
  --model-dir tmp/deepseek_absorbed_full_ref \
  --input-json tmp/manual_compare/input.json \
  --output-dir tmp/manual_compare/hf_l60_old \
  --layer-id 60 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

---

### 4. 启动 cache daemon
```bash
python -m server.manual_compare.cache_daemon \
  --host 127.0.0.1 \
  --port 47000 \
  --model-dir /model/ModelScope/deepseek-ai/DeepSeek-V3.1 \
  --resident-dtype bfloat16
```

---

### 5. 跑 HF cache reference
```bash
python -m server.manual_compare.run_hf_single_layer_manual \
  --model-dir tmp/deepseek_absorbed_cache_ref \
  --input-json tmp/manual_compare/input.json \
  --output-dir tmp/manual_compare/hf_l60_cache \
  --layer-id 60 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

---

### 6. 跑 runtime
```bash
python -m server.manual_compare.run_runtime_single_layer_manual \
  --config server/test/config_compare256.json \
  --input-json tmp/manual_compare/input.json \
  --output-dir tmp/manual_compare/runtime_l60 \
  --layer-id 60
```

---

### 7. 做单层对拍

#### HF old vs runtime
```bash
python -m server.manual_compare.compare_universal_single_layer_outputs \
  --hf-dir tmp/manual_compare/hf_l60_old \
  --runtime-dir tmp/manual_compare/runtime_l60 \
  --layer-id 60 \
  --output-json tmp/manual_compare/compare_hf_old_vs_runtime_l60.json
```

#### HF cache vs runtime
```bash
python -m server.manual_compare.compare_universal_single_layer_outputs \
  --hf-dir tmp/manual_compare/hf_l60_cache \
  --runtime-dir tmp/manual_compare/runtime_l60 \
  --layer-id 60 \
  --output-json tmp/manual_compare/compare_hf_cache_vs_runtime_l60.json
```

#### HF old vs HF cache
```bash
python -m server.manual_compare.compare_universal_single_layer_outputs \
  --hf-dir tmp/manual_compare/hf_l9_old \
  --runtime-dir tmp/manual_compare/hf_l9_cache \
  --layer-id 9 \
  --output-json tmp/manual_compare/compare_hf_old_vs_cache_l9.json
```

---

### 8. 查看 cache 状态
```bash
python -m server.manual_compare.query_cache \
  --host 127.0.0.1 \
  --port 47000
```

---

## 当前状态

### HF cache 路径
已验证：

- `hf_old == hf_cache`（已验证层上可做到完全一致）
- cache daemon / CUDA IPC / borrowed expert 路径正确

### runtime 路径
当前观察到：

- 偏差从 `layer 0` 就开始存在
- 在 `layer 3`（第一个 sparse layer）已经能看到 `ffn_hidden` / `router_hidden` 的轻微偏差
- 到 `layer 8` 左右，这些偏差会放大到足以翻转边界 top-k expert
- 到更深层会继续逐层累积

### 当前判断
更像是：

- 系统性小数值偏差累积
- 而不是单点逻辑 bug

当前主怀疑方向包括：

- attention 数值语义
- accumulate dtype
- cast 时机
- kernel rounding / reduction 语义

当前不认为 cache daemon / router group selection / residual / RMSNorm 本身是主要问题。

---

## 推荐使用方式

### correctness 基线
推荐顺序：

1. 先验证 `hf_old == hf_cache`
2. 再用 `hf_cache` 对拍 `runtime`

### 分层定位
推荐优先看：

- `layer 0`
- `layer 3`
- `layer 8`
- `layer 60`

因为它们分别代表：

- 最早的系统性偏差源
- 第一个 sparse layer
- top-k 边界开始翻转的位置
- 深层累积结果
