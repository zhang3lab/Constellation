# ShallowMLA patches

0001-fix-absorbed-softmax-scale.patch
- Fix MLA softmax scaling for absorbed-latent attention.
- Use 1 / sqrt(kv_latent_rank + qk_rope_head_dim) instead of 1 / sqrt(qk_nrope_head_dim + qk_rope_head_dim).

0002-use-relative-imports.patch
- Convert ShallowMLA local imports to package-relative imports.
- This allows importing via third_party.ShallowMLA.mla.

0003-fix-yarn-rotary-scaling.patch
- Align YaRN rotary frequency/scaling generation with DeepSeek V3 / Hugging Face behavior.
- Apply the missing RoPE amplitude scaling so runtime `freq_cis` matches HF `rope_cos` / `rope_sin`.
- Fix absorbed-attention rotary parameter generation mismatch that caused `q_rope_post_rotary` to diverge from HF.
