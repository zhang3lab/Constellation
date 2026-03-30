0001-fix-absorbed-softmax-scale.patch
- Fix MLA softmax scaling for absorbed-latent attention.
- Use 1 / sqrt(kv_latent_rank + qk_rope_head_dim).
