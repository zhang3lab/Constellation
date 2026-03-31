# ShallowMLA patches

0001-fix-absorbed-softmax-scale.patch
- Fix MLA softmax scaling for absorbed-latent attention.
- Use 1 / sqrt(kv_latent_rank + qk_rope_head_dim) instead of 1 / sqrt(qk_nrope_head_dim + qk_rope_head_dim).

0002-use-relative-imports.patch
- Convert ShallowMLA local imports to package-relative imports.
- This allows importing via third_party.ShallowMLA.mla.
