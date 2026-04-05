# Manual Compare Utilities

这个目录放的是一组手工对拍脚本，用来验证 runtime 路径和 HF 参考路径是否一致。

## 文件

### `compare_attention_backends.py`
统一的 attention backend 对拍入口。

用途：
- 比较 absorbed-latent reference 和 attention backend 输出

支持：
- `shallowmla`
- `flashmla`
- `all`

### `compare_universal_single_layer_outputs.py`
通用单层对拍脚本。

用途：
- 比较 HF 和 runtime 导出的单层中间张量
- 目标层如果是最后一层，也会比较：
  - `final_hidden`
  - `logits`

### `make_compare_input.py`
生成统一输入，供 HF / runtime 两边复用。

### `make_partial_deepseek_ref_dir.py`
构造裁剪后的参考模型目录，常用于 restricted experts 场景。

### `run_hf_single_layer_manual.py`
HF 单层停靠脚本。

用途：
- 跑到指定层
- 保存之前所有层输出
- 保存目标层调试信息
- 如果目标层是最后一层，也保存 `final_hidden` 和 `logits`

### `run_runtime_single_layer_manual.py`
runtime 单层停靠脚本。

用途：
- 跑到指定层
- 保存之前所有层输出
- 保存目标层调试信息
- 如果目标层是最后一层，也保存 `final_hidden` 和 `logits`

### `modeling_deepseek_absorbed.py`
当前 HF 参考实现所用的 patched modeling 文件。

用途：
- 放 absorbed-latent / restricted experts 相关改动
- 作为 HF 单层停靠与参考对拍的实际模型定义来源
- 对齐 `run_hf_single_layer_manual.py` 的运行语义

## 例子

先准备输入：

```bash
python3 server/manual_compare/make_compare_input.py \
  --prompt "Hello world" \
  --output-json tmp/compare_input.json
```

如果要做 restricted reference 目录：

```bash
python3 server/make_partial_deepseek_ref_dir.py \
  --config server/test/config.json \
  --dst tmp/deepseek_restricted_ref \
  --patched-modeling tmp/patched_modeling/modeling_deepseek_absorbed.py \
  --force
```

跑 HF 到第 60 层：

```bash
python3 server/manual_compare/run_hf_single_layer_manual.py \
  --model-dir tmp/deepseek_restricted_ref \
  --input-json tmp/compare_input.json \
  --output-dir tmp/hf_layer60_out \
  --device cuda \
  --layer-id 60
```

跑 runtime 到第 60 层：

```bash
python3 -m server.manual_compare.run_runtime_single_layer_manual \
  --config server/test/config.json \
  --model-dir /model/ModelScope/deepseek-ai/DeepSeek-V3.1 \
  --input-json tmp/compare_input.json \
  --output-dir tmp/runtime_layer60_out \
  --layer-id 60
```

做单层对拍：

```bash
python3 server/manual_compare/compare_universal_single_layer_outputs.py \
  --hf-dir tmp/hf_layer60_out \
  --runtime-dir tmp/runtime_layer60_out \
  --layer-id 60 \
  --output-json tmp/runtime_vs_hf_layer60.json
```

只测 attention backend：

```bash
python3 server/manual_compare/compare_attention_backends.py \
  --attention-backend all
```

## 适用场景

这组脚本主要用于：

- attention backend 验证
- dense / sparse 单层对拍
- router / MoE 调试
- final hidden / logits 一致性验证
