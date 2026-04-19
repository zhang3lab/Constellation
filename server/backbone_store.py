from dataclasses import dataclass
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


@dataclass(frozen=True)
class LoadSpec:
    enabled: bool
    dtype: torch.dtype


@dataclass(frozen=True)
class BackboneLoadPlan:
    runtime_dtype: torch.dtype
    attention: LoadSpec
    dense_prefix: LoadSpec
    shared_expert: LoadSpec
    router: LoadSpec
    embed: LoadSpec
    norm: LoadSpec
    lm_head: LoadSpec
    layer_ids: frozenset[int] | None = None

    @staticmethod
    def full(
        *,
        default_dtype: torch.dtype = torch.bfloat16,
        router_dtype: torch.dtype = torch.float32,
        layer_ids: set[int] | frozenset[int] | None = None,
    ) -> "BackboneLoadPlan":
        return BackboneLoadPlan(
            runtime_dtype=default_dtype,
            attention=LoadSpec(True, default_dtype),
            dense_prefix=LoadSpec(True, default_dtype),
            shared_expert=LoadSpec(True, default_dtype),
            router=LoadSpec(True, router_dtype),
            embed=LoadSpec(True, default_dtype),
            norm=LoadSpec(True, default_dtype),
            lm_head=LoadSpec(True, default_dtype),
            layer_ids=None if layer_ids is None else frozenset(int(x) for x in layer_ids),
        )

    @staticmethod
    def router_only(
        *,
        router_dtype: torch.dtype = torch.float32,
        layer_ids: set[int] | frozenset[int] | None = None,
    ) -> "BackboneLoadPlan":
        off = LoadSpec(False, torch.bfloat16)
        return BackboneLoadPlan(
            runtime_dtype=torch.float32,
            attention=off,
            dense_prefix=off,
            shared_expert=off,
            router=LoadSpec(True, router_dtype),
            embed=off,
            norm=off,
            lm_head=off,
            layer_ids=None if layer_ids is None else frozenset(int(x) for x in layer_ids),
        )

    @staticmethod
    def attention_only(
        *,
        attention_dtype: torch.dtype = torch.float32,
        embed_dtype: torch.dtype = torch.float32,
        layer_ids: set[int] | frozenset[int] | None = None,
    ) -> "BackboneLoadPlan":
        off = LoadSpec(False, torch.bfloat16)
        return BackboneLoadPlan(
            runtime_dtype=torch.float32,
            attention=LoadSpec(True, attention_dtype),
            dense_prefix=off,
            shared_expert=off,
            router=off,
            embed=LoadSpec(True, embed_dtype),
            norm=off,
            lm_head=off,
            layer_ids=None if layer_ids is None else frozenset(int(x) for x in layer_ids),
        )

    @staticmethod
    def runtime_fp32_no_attention_no_routed_experts(
        *,
        router_dtype: torch.dtype = torch.float32,
    ) -> "BackboneLoadPlan":
        off = LoadSpec(False, torch.bfloat16)
        return BackboneLoadPlan(
            runtime_dtype=torch.float32,
            attention=off,
            dense_prefix=LoadSpec(True, torch.float32),
            shared_expert=LoadSpec(True, torch.float32),
            router=LoadSpec(True, router_dtype),
            embed=LoadSpec(True, torch.float32),
            norm=LoadSpec(True, torch.float32),
            lm_head=LoadSpec(True, torch.float32),
            layer_ids=None,
        )


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


def make_even_explicit_partition(
    *,
    num_layers: int = 61,
    devices: list[str] | tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    embed_device: str | None = None,
    final_norm_device: str | None = None,
    lm_head_device: str | None = None,
) -> ExplicitLayerPartition:
    devices = [str(x) for x in devices]
    if not devices:
        raise RuntimeError("devices must not be empty")
    if num_layers <= 0:
        raise RuntimeError(f"num_layers must be > 0, got {num_layers}")

    layer_to_device: dict[int, str] = {}
    ndev = len(devices)

    for layer_id in range(num_layers):
        dev_idx = min(ndev - 1, (layer_id * ndev) // num_layers)
        layer_to_device[layer_id] = devices[dev_idx]

    if embed_device is None:
        embed_device = devices[0]
    if final_norm_device is None:
        final_norm_device = devices[-1]
    if lm_head_device is None:
        lm_head_device = devices[-1]

    return ExplicitLayerPartition(
        layer_to_device=layer_to_device,
        embed_device=embed_device,
        final_norm_device=final_norm_device,
        lm_head_device=lm_head_device,
    )


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
    partition: LayerPartition | None = None,
    mapped_store=None,
    plan: BackboneLoadPlan | None = None,
) -> BackboneStore:
    model_loader = session.get_deepseek_model_loader()
    if partition is None:
        partition = TwoGpuLayerPartition(split_layer=30)
    if plan is None:
        plan = BackboneLoadPlan.full()

    layer_ids = None if plan.layer_ids is None else {int(x) for x in plan.layer_ids}

    # store.dtype 代表主 hidden/runtime dtype；router 可单独是 fp32
    store = BackboneStore(
        mla_cfg=model_loader.mla_config(),
        dtype=plan.runtime_dtype,
        partition=partition,
    )

    if plan.embed.enabled:
        store.set_embed_tokens(
            _load_gpu_tensor(
                "model.embed_tokens.weight",
                device=partition.embed_device(),
                dtype=plan.embed.dtype,
                model_loader=model_loader,
                mapped_store=mapped_store,
            )
        )

    for layer_id in range(61):
        dev = partition.layer_device(layer_id)
        entry = {
            "device": dev,
        }
     
        load_this_layer = (layer_ids is None or layer_id in layer_ids)
     
        need_attention = load_this_layer and plan.attention.enabled
        need_dense_prefix = load_this_layer and plan.dense_prefix.enabled and layer_id < 3
        need_shared_expert = load_this_layer and plan.shared_expert.enabled and layer_id >= 3
        need_router = load_this_layer and plan.router.enabled and layer_id >= 3
     
        if need_attention or need_dense_prefix or need_shared_expert or need_router:
            entry["input_layernorm"] = _load_gpu_tensor(
                f"model.layers.{layer_id}.input_layernorm.weight",
                device=dev,
                dtype=plan.runtime_dtype,
                model_loader=model_loader,
                mapped_store=mapped_store,
            )
     
        if need_attention or need_dense_prefix or need_shared_expert or need_router:
            entry["post_attention_layernorm"] = _load_gpu_tensor(
                f"model.layers.{layer_id}.post_attention_layernorm.weight",
                device=dev,
                dtype=plan.runtime_dtype,
                model_loader=model_loader,
                mapped_store=mapped_store,
            )
     
        if need_attention:
            entry["attention"] = {
                "input_layernorm": _load_gpu_tensor(
                    f"model.layers.{layer_id}.input_layernorm.weight",
                    device=dev,
                    dtype=plan.attention.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "q_a_proj": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.q_a_proj.weight",
                    device=dev,
                    dtype=plan.attention.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "q_a_layernorm": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight",
                    device=dev,
                    dtype=plan.attention.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "q_b_proj": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.q_b_proj.weight",
                    device=dev,
                    dtype=plan.attention.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "kv_a_proj_with_mqa": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight",
                    device=dev,
                    dtype=plan.attention.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "kv_a_layernorm": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight",
                    device=dev,
                    dtype=plan.attention.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "kv_b_proj": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.kv_b_proj.weight",
                    device=dev,
                    dtype=plan.attention.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "o_proj": _load_gpu_tensor(
                    f"model.layers.{layer_id}.self_attn.o_proj.weight",
                    device=dev,
                    dtype=plan.attention.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
            }
     
        if need_dense_prefix:
            entry["dense_ffn"] = {
                "w_up": _load_gpu_tensor(
                    f"model.layers.{layer_id}.mlp.up_proj.weight",
                    device=dev,
                    dtype=plan.dense_prefix.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "w_gate": _load_gpu_tensor(
                    f"model.layers.{layer_id}.mlp.gate_proj.weight",
                    device=dev,
                    dtype=plan.dense_prefix.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "w_down": _load_gpu_tensor(
                    f"model.layers.{layer_id}.mlp.down_proj.weight",
                    device=dev,
                    dtype=plan.dense_prefix.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
            }
     
        if need_shared_expert:
            entry["shared_expert"] = {
                "w_up": _load_gpu_tensor(
                    f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight",
                    device=dev,
                    dtype=plan.shared_expert.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "w_gate": _load_gpu_tensor(
                    f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight",
                    device=dev,
                    dtype=plan.shared_expert.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "w_down": _load_gpu_tensor(
                    f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight",
                    device=dev,
                    dtype=plan.shared_expert.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
            }
     
        if need_router:
            entry["router"] = {
                "gate_weight": _load_gpu_tensor(
                    f"model.layers.{layer_id}.mlp.gate.weight",
                    device=dev,
                    dtype=plan.router.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
                "e_score_correction_bias": _load_gpu_tensor(
                    f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias",
                    device=dev,
                    dtype=plan.router.dtype,
                    model_loader=model_loader,
                    mapped_store=mapped_store,
                ),
            }
     
        store.set_layer(layer_id, entry)

    if plan.norm.enabled:
        store.set_model_norm(
            _load_gpu_tensor(
                "model.norm.weight",
                device=partition.final_norm_device(),
                dtype=plan.norm.dtype,
                model_loader=model_loader,
                mapped_store=mapped_store,
            )
        )

    if plan.lm_head.enabled:
        store.set_lm_head(
            _load_gpu_tensor(
                "lm_head.weight",
                device=partition.lm_head_device(),
                dtype=plan.lm_head.dtype,
                model_loader=model_loader,
                mapped_store=mapped_store,
            )
        )

    return store
