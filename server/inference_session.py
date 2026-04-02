from __future__ import annotations

import threading
from typing import Dict, Tuple

import torch

from third_party.ShallowMLA.mla import PageAttentionCacheManager, precompute_freqs_cis

from server.backbone_store import (
    BackboneLoadPlan,
    TwoGpuLayerPartition,
    preload_non_moe_backbone,
)
from server.client import NodeClient
from server.deepseek_model_loader import DeepseekModelLoader
from server.mla_runtime import MLARuntime
from server.tensor_cache import FREQ_CIS_TENSOR_NAME, MappedTensorStore


class SessionClientPool:
    def __init__(self):
        self._lock = threading.Lock()
        self._clients: Dict[Tuple[str, int], NodeClient] = {}

    def get(self, host: str, port: int) -> NodeClient:
        key = (host, int(port))
        with self._lock:
            client = self._clients.get(key)
            if client is not None:
                return client

            client = NodeClient(host, int(port))
            client.__enter__()
            self._clients[key] = client
            return client

    def invalidate(self, host: str, port: int) -> None:
        key = (host, int(port))
        with self._lock:
            client = self._clients.pop(key, None)

        if client is not None:
            try:
                client.__exit__(None, None, None)
            except Exception:
                pass

    def close_all(self) -> None:
        with self._lock:
            items = list(self._clients.items())
            self._clients.clear()

        for _, client in items:
            try:
                client.__exit__(None, None, None)
            except Exception:
                pass


class InferenceSession:
    def __init__(self, coord, cfg):
        self.coord = coord
        self.cfg = cfg
        self.client_pool = SessionClientPool()

        self.deepseek_model_loader = None
        self.full_model_executor = None

        self.mapped_tensor_store = None
        self.backbone_store = None
        self.attention_runtime = None
        self.page_attention_cache_managers = None
        self.freq_cis_by_device = None

    def get_deepseek_model_loader(self) -> DeepseekModelLoader:
        if self.deepseek_model_loader is None:
            model_root = str(self.cfg["model"]["root"])
            self.deepseek_model_loader = DeepseekModelLoader(model_root)
        return self.deepseek_model_loader

    def get_router_config(self) -> dict:
        return self.get_deepseek_model_loader().router_config()

    def get_mla_config(self) -> dict:
        return self.get_deepseek_model_loader().mla_config()

    def ensure_freq_cis_by_device(self, *, max_seq_len: int) -> None:
        if self.freq_cis_by_device is not None:
            return
     
        if self.backbone_store is None:
            raise RuntimeError("backbone_store is not initialized")
     
        mla_cfg = self.backbone_store.mla_cfg
        max_seq_len = int(max_seq_len)
     
        devices = set()
        devices.add(str(self.backbone_store.partition.embed_device()))
        devices.add(str(self.backbone_store.partition.final_norm_device()))
        devices.add(str(self.backbone_store.partition.lm_head_device()))
        for layer_id in range(61):
            devices.add(str(self.backbone_store.partition.layer_device(layer_id)))
     
        expected_freq_cis_meta = {
            "qk_rope_head_dim": int(mla_cfg["qk_rope_head_dim"]),
            "seq_len": max_seq_len,
            "seq_len_train": int(mla_cfg["max_seq_len_train"]),
            "beta_fast": float(mla_cfg["beta_fast"]),
            "beta_slow": float(mla_cfg["beta_slow"]),
            "rope_theta": float(mla_cfg["rope_theta"]),
            "rope_factor": float(mla_cfg["rope_factor"]),
        }
     
        freq_cis_master = None
        if (
            self.mapped_tensor_store is not None
            and self.mapped_tensor_store.has_tensor(FREQ_CIS_TENSOR_NAME)
        ):
            got_meta = self.mapped_tensor_store.get_tensor_meta(FREQ_CIS_TENSOR_NAME)
            if got_meta is None:
                raise RuntimeError(f"{FREQ_CIS_TENSOR_NAME} missing metadata in tensor cache")
            if got_meta != expected_freq_cis_meta:
                raise RuntimeError(
                    f"{FREQ_CIS_TENSOR_NAME} metadata mismatch:\n"
                    f"expected={expected_freq_cis_meta}\n"
                    f"got={got_meta}"
                )
            freq_cis_master = self.mapped_tensor_store.get_torch_cpu(FREQ_CIS_TENSOR_NAME)
        else:
            freq_cis_master = precompute_freqs_cis(
                qk_rope_head_dim=int(mla_cfg["qk_rope_head_dim"]),
                seq_len=max_seq_len,
                seq_len_train=int(mla_cfg["max_seq_len_train"]),
                beta_fast=float(mla_cfg["beta_fast"]),
                beta_slow=float(mla_cfg["beta_slow"]),
                rope_theta=float(mla_cfg["rope_theta"]),
                rope_factor=float(mla_cfg["rope_factor"]),
                dtype=torch.float32,
            )
     
        if not isinstance(freq_cis_master, torch.Tensor):
            raise TypeError(
                f"freq_cis_master expected torch.Tensor, got {type(freq_cis_master).__name__}"
            )
     
        freq_cis_master = freq_cis_master.detach().contiguous().cpu()
        if freq_cis_master.dtype != torch.float32:
            freq_cis_master = freq_cis_master.to(torch.float32)
     
        self.freq_cis_by_device = {}
        for dev in devices:
            self.freq_cis_by_device[dev] = freq_cis_master.to(
                device=dev,
                dtype=self.backbone_store.dtype,
            )


    def ensure_full_model_runtime(
        self,
        *,
        tensor_cache_dir: str = "tmp/non_moe_backbone_cache",
        split_layer: int = 30,
        backbone_dtype: torch.dtype = torch.bfloat16,
        kv_cache_cfg: dict,
    ) -> None:
        if self.backbone_store is not None:
            return
     
        if not isinstance(kv_cache_cfg, dict):
            raise RuntimeError("kv_cache_cfg must be a dict")
     
        self.mapped_tensor_store = MappedTensorStore(tensor_cache_dir)
     
        partition = TwoGpuLayerPartition(split_layer=split_layer)
        self.backbone_store = preload_non_moe_backbone(
            self,
            partition=partition,
            mapped_store=self.mapped_tensor_store,
            plan=BackboneLoadPlan.full(
                default_dtype=backbone_dtype,
                router_dtype=torch.float32,
            ),
        )
     
        mla_cfg = self.backbone_store.mla_cfg
        self.attention_runtime = MLARuntime(
            dim=int(mla_cfg["dim"]),
            kv_latent_rank=int(mla_cfg["kv_latent_rank"]),
            q_latent_rank=int(mla_cfg["q_latent_rank"]),
            num_heads=int(mla_cfg["num_heads"]),
            qk_nrope_head_dim=int(mla_cfg["qk_nrope_head_dim"]),
            v_head_dim=int(mla_cfg["v_head_dim"]),
            qk_rope_head_dim=int(mla_cfg["qk_rope_head_dim"]),
            dtype=self.backbone_store.dtype,
            eps=1e-6,
        )
     
        self.page_attention_cache_managers = {}

        max_batch_size = int(kv_cache_cfg["max_batch_size"])
        max_seq_len = int(kv_cache_cfg["max_seq_len"])
        page_size = int(kv_cache_cfg["page_size"])
        use_page_cache_triton = bool(kv_cache_cfg["use_triton"])

        kv_latent_rank = int(mla_cfg["kv_latent_rank"])
        qk_rope_head_dim = int(mla_cfg["qk_rope_head_dim"])

        self.ensure_freq_cis_by_device(
            max_seq_len=int(kv_cache_cfg["max_seq_len"]),
        )

        tokens_capacity = max_batch_size * max_seq_len
        num_pages = int(tokens_capacity * 1.1 / page_size)
        if num_pages * page_size < tokens_capacity:
            num_pages += 1
     
        for layer_id in range(61):
            dev = str(self.backbone_store.layer(layer_id)["device"])
            self.page_attention_cache_managers[layer_id] = PageAttentionCacheManager(
                batch_size=max_batch_size,
                page_size=page_size,
                num_pages=num_pages,
                kv_latent_rank=kv_latent_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                max_seq_len=max_seq_len,
                use_triton=use_page_cache_triton,
                dtype=self.backbone_store.dtype,
                device=dev,
            )

    def reset_full_model_kv_cache(
        self,
        *,
        kv_cache_cfg: dict,
    ) -> None:
        if self.backbone_store is None:
            self.page_attention_cache_managers = None
            return
     
        if not isinstance(kv_cache_cfg, dict):
            raise RuntimeError("kv_cache_cfg must be a dict")
     
        mla_cfg = self.backbone_store.mla_cfg
     
        max_batch_size = int(kv_cache_cfg["max_batch_size"])
        max_seq_len = int(kv_cache_cfg["max_seq_len"])
        page_size = int(kv_cache_cfg["page_size"])
        use_page_cache_triton = bool(kv_cache_cfg["use_triton"])
     
        kv_latent_rank = int(mla_cfg["kv_latent_rank"])
        qk_rope_head_dim = int(mla_cfg["qk_rope_head_dim"])
     
        tokens_capacity = max_batch_size * max_seq_len
        num_pages = int(tokens_capacity * 1.1 / page_size)
        if num_pages * page_size < tokens_capacity:
            num_pages += 1
     
        self.page_attention_cache_managers = {}
        for layer_id in range(61):
            dev = str(self.backbone_store.layer(layer_id)["device"])
            self.page_attention_cache_managers[layer_id] = PageAttentionCacheManager(
                batch_size=max_batch_size,
                page_size=page_size,
                num_pages=num_pages,
                kv_latent_rank=kv_latent_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                max_seq_len=max_seq_len,
                use_triton=use_page_cache_triton,
                dtype=self.backbone_store.dtype,
                device=dev,
            )

    def close(self) -> None:
        self.client_pool.close_all()

        if self.mapped_tensor_store is not None:
            try:
                self.mapped_tensor_store.close()
            except Exception:
                pass

        self.deepseek_model_loader = None
        self.full_model_executor = None

        self.mapped_tensor_store = None
        self.backbone_store = None
        self.attention_runtime = None
        self.page_attention_cache_managers = None
        self.freq_cis_by_device = None

    def __enter__(self) -> "InferenceSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
