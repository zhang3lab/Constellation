import torch


class LayerPartition:
    def layer_device(self, layer_id: int) -> str:
        raise NotImplementedError

    def embed_device(self) -> str:
        raise NotImplementedError

    def final_norm_device(self) -> str:
        raise NotImplementedError

    def lm_head_device(self) -> str:
        raise NotImplementedError


class TwoGpuLayerPartition(LayerPartition):
    def __init__(self, split_layer: int = 30):
        self.split_layer = int(split_layer)

    def layer_device(self, layer_id: int) -> str:
        return "cuda:0" if int(layer_id) <= self.split_layer else "cuda:1"

    def embed_device(self) -> str:
        return "cuda:0"

    def final_norm_device(self) -> str:
        return "cuda:1"

    def lm_head_device(self) -> str:
        return "cuda:1"


class ExplicitLayerPartition(LayerPartition):
    def __init__(
        self,
        layer_to_device: dict[int, str],
        *,
        embed_device: str,
        final_norm_device: str,
        lm_head_device: str,
    ):
        self.layer_to_device = {int(k): str(v) for k, v in layer_to_device.items()}
        self._embed_device = str(embed_device)
        self._final_norm_device = str(final_norm_device)
        self._lm_head_device = str(lm_head_device)

    def layer_device(self, layer_id: int) -> str:
        return self.layer_to_device[int(layer_id)]

    def embed_device(self) -> str:
        return self._embed_device

    def final_norm_device(self) -> str:
        return self._final_norm_device

    def lm_head_device(self) -> str:
        return self._lm_head_device


class BackboneStore:
    def __init__(self, *, mla_cfg: dict, dtype: torch.dtype, partition: LayerPartition):
        self.mla_cfg = dict(mla_cfg)
        self.dtype = dtype
        self.partition = partition
        self.layers: dict[int, dict] = {}
        self._embed_tokens = None
        self._model_norm = None
        self._lm_head = None

    def layer(self, layer_id: int) -> dict:
        return self.layers[int(layer_id)]

    def set_layer(self, layer_id: int, entry: dict):
        self.layers[int(layer_id)] = entry

    def embed_tokens(self):
        return self._embed_tokens

    def set_embed_tokens(self, t):
        self._embed_tokens = t

    def model_norm(self):
        return self._model_norm

    def set_model_norm(self, t):
        self._model_norm = t

    def lm_head(self):
        return self._lm_head

    def set_lm_head(self, t):
        self._lm_head = t


def _load_cpu_tensor_from_source(
    tensor_name: str,
    *,
    model_loader,
    mapped_store=None,
) -> torch.Tensor:
    if mapped_store is not None and mapped_store.has_tensor(tensor_name):
        return mapped_store.get_torch_cpu(tensor_name)
    return model_loader.load_tensor_fp32_by_name(tensor_name)


def _load_gpu_tensor(
    tensor_name: str,
    *,
    device: str,
    dtype: torch.dtype,
    model_loader,
    mapped_store=None,
) -> torch.Tensor:
    t_cpu = _load_cpu_tensor_from_source(
        tensor_name,
        model_loader=model_loader,
        mapped_store=mapped_store,
    )
    return t_cpu.to(device=device, dtype=dtype)


def preload_non_moe_backbone(
    session,
    *,
    dtype: torch.dtype = torch.bfloat16,
    partition: LayerPartition | None = None,
    mapped_store=None,
) -> BackboneStore:
    model_loader = session.get_deepseek_model_loader()
    if partition is None:
        partition = TwoGpuLayerPartition(split_layer=30)

    store = BackboneStore(
        mla_cfg=model_loader.mla_config(),
        dtype=dtype,
        partition=partition,
    )

    store.set_embed_tokens(
        _load_gpu_tensor(
            "model.embed_tokens.weight",
            device=partition.embed_device(),
            dtype=dtype,
            model_loader=model_loader,
            mapped_store=mapped_store,
        )
    )

    for layer_id in range(61):
        dev = partition.layer_device(layer_id)

        entry = {
            "device": dev,
            "input_layernorm": _load_gpu_tensor(
                f"model.layers.{layer_id}.input_layernorm.weight",
                device=dev,
                dtype=dtype,
                model_loader=model_loader,
                mapped_store=mapped_store,
            ),
            "post_attention_layernorm": _load_gpu_tensor(
                f"model.layers.{layer_id}.post_attention_layernorm.weight",
                device=dev,
                dtype=dtype,
                model_loader=model_loader,
                mapped_store=mapped_store,
            ),
            "attention": {
                "input_layernorm": _load_gpu_tensor(
                    f"model.layers.{layer_id}.input_layernorm.weight",
                    device=dev,
                    dtype=dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "q_a_proj": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.q_a_proj.weight",
                    device=dev,
                    dtype=dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "q_a_layernorm": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight",
                    device=dev,
                    dtype=dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "q_b_proj": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.q_b_proj.weight",
                    device=dev,
                    dtype=dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "kv_a_proj_with_mqa": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight",
                    device=dev,
                    dtype=dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "kv_a_layernorm": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight",
                    device=dev,
                    dtype=dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "kv_b_proj": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.kv_b_proj.weight",
                    device=dev,
                    dtype=dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "o_proj": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.o_proj.weight",
                    device=dev,
                    dtype=dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
            },
        }

        if layer_id < 3:
            entry["dense_ffn"] = {
                "w_up": _load_gpu_tensor(
                    f"model.layers.{layer_id}.mlp.up_proj.weight",
                    device=dev,
                    dtype=dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "w_gate": _load_gpu_tensor(
                    f"model.layers.{layer_id}.mlp.gate_proj.weight",
                    device=dev,
                    dtype=dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "w_down": _load_gpu_tensor(
                    f"model.layers.{layer_id}.mlp.down_proj.weight",
                    device=dev,
                    dtype=dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
            }

        store.set_layer(layer_id, entry)

    store.set_model_norm(
        _load_gpu_tensor(
            "model.norm.weight",
            device=partition.final_norm_device(),
            dtype=dtype,
            model_loader=model_loader,
            mapped_store=mapped_store,
        )
    )
    store.set_lm_head(
        _load_gpu_tensor(
            "lm_head.weight",
            device=partition.lm_head_device(),
            dtype=dtype,
            model_loader=model_loader,
            mapped_store=mapped_store,
        )
    )

    return store
