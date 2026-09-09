# PR #4951 applicability to H3

Inspected public PR head `30751b2ef15776ea00256e9ba979364a1ecf39a3`; merged as
`6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f` on 2026-09-05.

The SM120, MHA, D128, block64, INT8 Q/K + FP8 V contracts match the proposed
Sage experiment. However, the **exported specialization inventory does not
support H3's actual sparse metadata**. The Python signature alone is insufficient.

H3 uses 1D `block_sizes`, per-query block counts (dense prefix rows versus
prefix + top-k video rows), and arbitrary sorted top-k block indices. It needs:

```text
HAS_BLOCK_NUMS=1, BLOCK_SIZES_MODE=1, FULL_K64_TILES=0,
UNIFORM_NONEMPTY=0, CONTIGUOUS_BLOCK_INDICES=0
```

PR #4951 ships these six tuples in that field order:

```text
(0,0,0,1,1)
(0,0,1,1,1)
(1,0,1,0,0)
(0,1,0,1,1)
(0,2,0,1,1)
(0,3,0,1,1)
```

Its loader requires an exact tuple match and raises if no generated module
exists. Setting uniform counts or contiguous indices would misrepresent H3's
mask. Omitting block sizes would include padded tokens and change attention.
Consequently no end-to-end Cake performance or video-quality claim is made.

To use this backend, generate and validate the missing specialization while
preserving ragged-token masking and arbitrary indices. The implementation also
requires caller-owned contiguous BF16 BHSD output and a 512-byte, 128-byte-aligned
CUDA uint8 TMA descriptor workspace. Keep these buffers alive for their stream's
work and prebuild before CUDA Graph capture.

The PR's directly measured seven-shape geometric mean speedup is 1.113520x over
the older Sage CuTe DSL implementation (`a14241f2`), **not over BF16 or over H3
DiT latency**. Those portfolio shapes differ from H3's B1/H7/S105024 geometry.

Sources:

- https://github.com/flashinfer-ai/flashinfer/pull/4951
- https://github.com/flashinfer-ai/flashinfer/blob/30751b2ef15776ea00256e9ba979364a1ecf39a3/csrc/cake_sage_block_sparse_attention/cake_sage_block_sparse_attention_manifest.json
- https://github.com/flashinfer-ai/flashinfer/blob/30751b2ef15776ea00256e9ba979364a1ecf39a3/flashinfer/cute_dsl/sparse/bsa_attn_sm120.py

Downloaded files and SHA-256 hashes are retained in `pr4951-source.json`.
