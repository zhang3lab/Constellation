import sys
from pathlib import Path
from typing import Optional

import torch

_SHALLOWMLA_ROOT = Path("/root/ShallowMLA")
if str(_SHALLOWMLA_ROOT) not in sys.path:
    sys.path.append(str(_SHALLOWMLA_ROOT))

from mla import MLA, precompute_freqs_cis


class ShallowMLAAttentionWrapper:
    def __init__(
        self,
        *,
        model_loader,
        layer_id: int,
        max_batch_size: int = 4,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        self.layer_id = int(layer_id)
        self.device = torch.device(device)
        self.dtype = dtype

        mla_cfg = model_loader.mla_config()

        self.mla = MLA(
            dim=mla_cfg["dim"],
            kv_latent_rank=mla_cfg["kv_latent_rank"],
            q_latent_rank=mla_cfg["q_latent_rank"],
            num_heads=mla_cfg["num_heads"],
            qk_nrope_head_dim=mla_cfg["qk_nrope_head_dim"],
            v_head_dim=mla_cfg["v_head_dim"],
            qk_rope_head_dim=mla_cfg["qk_rope_head_dim"],
            max_batch_size=max_batch_size,
            max_seq_len=mla_cfg["max_seq_len"],
            dtype=dtype,
            optim_type="torch",
            eps=1e-6,
            use_page_cache=False,
            use_page_cache_triton=False,
        ).to(self.device)

        ws = model_loader.load_attention_block_weights_fp32(self.layer_id)

        with torch.no_grad():
            self.mla.proj_q_down.weight.copy_(ws["q_a_proj"].to(device=self.device, dtype=dtype))
            self.mla.rms_norm_q_weight.copy_(ws["q_a_layernorm"].to(device=self.device, dtype=dtype))
            self.mla.proj_q_up.weight.copy_(ws["q_b_proj"].to(device=self.device, dtype=dtype))

            self.mla.proj_kv_down.weight.copy_(ws["kv_a_proj_with_mqa"].to(device=self.device, dtype=dtype))
            self.mla.rms_norm_kv_weight.copy_(ws["kv_a_layernorm"].to(device=self.device, dtype=dtype))
            self.mla.proj_kv_up.weight.copy_(ws["kv_b_proj"].to(device=self.device, dtype=dtype))

            self.mla.proj_out.weight.copy_(ws["o_proj"].to(device=self.device, dtype=dtype))

        self.input_layernorm_weight = ws["input_layernorm"].to(device=self.device, dtype=dtype)
        self.eps = 1e-6

        self.freq_cis = precompute_freqs_cis(
            qk_rope_head_dim=mla_cfg["qk_rope_head_dim"],
            seq_len=mla_cfg["max_seq_len"],
            seq_len_train=mla_cfg["max_seq_len_train"],
            beta_fast=mla_cfg["beta_fast"],
            beta_slow=mla_cfg["beta_slow"],
            rope_theta=mla_cfg["rope_theta"],
            rope_factor=mla_cfg["rope_factor"],
            dtype=dtype,
        ).to(self.device)

    def rms_norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(
            x,
            (x.shape[-1],),
            self.input_layernorm_weight,
            self.eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.rms_norm_input(x)
        return self.mla(
            x,
            start_pos=start_pos,
            freq_cis=self.freq_cis,
            mask=mask,
        )
