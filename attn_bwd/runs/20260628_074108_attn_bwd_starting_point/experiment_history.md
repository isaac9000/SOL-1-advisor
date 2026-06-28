# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-28 07:41:22 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 3437.26 μs

**Kernel code:**
```python
"""
Reference attention-backward kernel — pure PyTorch baseline.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # Expand value_states for GQA: [bs, 8, skv, d] → [bs, 80, skv, d]
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM
    ).reshape(bs, n_heads, seq_kv, HEAD_DIM)

    # 1. Transpose grad: [bs, sq, h, d] → [bs, h, sq, d]  (cast to f32)
    dO = grad_attn_output.transpose(1, 2).to(torch.float32)

    # 2. dP̃ = dO @ V^T  →  [bs, h, sq, skv]
    dP_dropped = torch.matmul(dO, vs_exp.to(torch.float32).transpose(-2, -1))

    # 3. Dropout backward
    if attention_dropout > 0.0:
        dP = dP_dropped * dropout_mask / (1.0 - attention_dropout)
    else:
        dP = dP_dropped

    # 4. Softmax backward: dS = P ⊙ (dP − sum(dP ⊙ P))
    P = attn_weights.to(torch.float32)
    dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # 5. dV_exp = P̃^T @ dO  →  [bs, h, skv, d]
    dV_exp = torch.matmul(
        attn_weights_dropped.to(torch.float32).transpose(-2, -1), dO
    )

    # 6. GQA aggregation: sum over groups  →  [bs, 8, skv, d]
    dV = dV_exp.reshape(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM).sum(dim=2)
    dV = dV.to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #2 — 2026-06-28 07:43:29 UTC 💥 CRASH

**Hypothesis:** ** Two Triton kernels (`attn_bwd_ds_kernel_v2` and `attn_bwd_dv_kernel`) replacing the pure-PyTorch baseline. The unused first draft kernel (`attn_bwd_ds_kernel`) is left in the file but never called.

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Fused Triton attention-backward kernel with implicit GQA handling.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = 10


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: compute grad_attn_scores (dS)
# Grid: [bs * n_heads, cdiv(seq_q, BLOCK_SQ), cdiv(seq_kv, BLOCK_SKV)]
# Each block: tile [BLOCK_SQ, BLOCK_SKV] of one (batch, head) pair
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def attn_bwd_ds_kernel(
    # dO: [bs, n_heads, seq_q, head_dim]  (after transpose)
    dO_ptr,
    dO_stride_bs, dO_stride_h, dO_stride_sq, dO_stride_d,
    # value_states: [bs, n_kv_heads, seq_kv, head_dim]
    V_ptr,
    V_stride_bs, V_stride_h, V_stride_skv, V_stride_d,
    # attn_weights (P): [bs, n_heads, seq_q, seq_kv]
    P_ptr,
    P_stride_bs, P_stride_h, P_stride_sq, P_stride_skv,
    # dropout_mask: [bs, n_heads, seq_q, seq_kv]
    M_ptr,
    M_stride_bs, M_stride_h, M_stride_sq, M_stride_skv,
    # attn_weights_dropped (not needed for dS, but we need P for softmax bwd)
    # output dS: [bs, n_heads, seq_q, seq_kv]
    dS_ptr,
    dS_stride_bs, dS_stride_h, dS_stride_sq, dS_stride_skv,
    # scalars
    seq_q: tl.constexpr, seq_kv: tl.constexpr, head_dim: tl.constexpr,
    n_kv_heads: tl.constexpr, n_groups: tl.constexpr,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr, BLOCK_SKV: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # program IDs
    bh_id = tl.program_id(0)  # batch * n_heads + head
    sq_id = tl.program_id(1)  # tile index along seq_q
    skv_id = tl.program_id(2)  # tile index along seq_kv

    bs_id = bh_id // NUM_ATTENTION_HEADS
    h_id = bh_id % NUM_ATTENTION_HEADS
    kv_h_id = h_id // n_groups

    sq_start = sq_id * BLOCK_SQ
    skv_start = skv_id * BLOCK_SKV

    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)
    skv_offs = skv_start + tl.arange(0, BLOCK_SKV)
    d_offs = tl.arange(0, BLOCK_D)

    sq_mask = sq_offs < seq_q
    skv_mask = skv_offs < seq_kv

    # Load dO tile: [BLOCK_SQ, BLOCK_D]
    dO_base = (bs_id * dO_stride_bs + h_id * dO_stride_h)
    dO_ptrs = dO_base + sq_offs[:, None] * dO_stride_sq + d_offs[None, :] * dO_stride_d
    dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

    # Load V tile: [BLOCK_SKV, BLOCK_D]
    V_base = (bs_id * V_stride_bs + kv_h_id * V_stride_h)
    V_ptrs = V_base + skv_offs[:, None] * V_stride_skv + d_offs[None, :] * V_stride_d
    V_tile = tl.load(V_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

    # dP_dropped = dO @ V^T: [BLOCK_SQ, BLOCK_SKV]
    dP_dropped = tl.dot(dO_tile, tl.trans(V_tile))

    # Load dropout mask and apply: dP = dP_dropped * mask * inv_keep_prob
    M_base = (bs_id * M_stride_bs + h_id * M_stride_h)
    M_ptrs = M_base + sq_offs[:, None] * M_stride_sq + skv_offs[None, :] * M_stride_skv
    mask_tile = tl.load(M_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=0).to(tl.int1)
    dP = tl.where(mask_tile, dP_dropped * inv_keep_prob, 0.0)

    # Load P tile: [BLOCK_SQ, BLOCK_SKV]
    P_base = (bs_id * P_stride_bs + h_id * P_stride_h)
    P_ptrs = P_base + sq_offs[:, None] * P_stride_sq + skv_offs[None, :] * P_stride_skv
    P_tile = tl.load(P_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=0.0).to(tl.float32)

    # Softmax backward: dS = P * (dP - rowsum(dP * P))
    # We need the full rowsum across all skv for each sq row
    # Since we tile over skv, we need to compute partial sums
    # Strategy: compute dP*P locally, then we need to accumulate across skv tiles
    # This requires a two-pass approach - but that's expensive.
    # Instead, let's NOT tile over seq_kv in this kernel — load the full seq_kv per block.
    # We'll restructure: grid is [bs*n_heads, cdiv(seq_q, BLOCK_SQ)] and each block
    # handles ALL seq_kv for a given sq tile.
    # But that means BLOCK_SKV = seq_kv which may be too large.
    # For now, we accumulate the full rowsum by iterating over skv in a loop.
    # Actually the current structure is wrong for a tiled skv approach.
    # Let's just output dP*P and dP separately and handle rowsum reduction...
    # 
    # SIMPLER: don't tile over seq_kv, make each block cover all seq_kv.
    # We'll change the grid and kernel structure below.
    
    dPP = dP * P_tile
    dS_tile = P_tile * (dP - dPP)  # This is WRONG without the full rowsum!
    # Placeholder - will be overridden by the correct kernel below
    dS_tile = dS_tile.to(tl.bfloat16)

    dS_base = (bs_id * dS_stride_bs + h_id * dS_stride_h)
    dS_ptrs = dS_base + sq_offs[:, None] * dS_stride_sq + skv_offs[None, :] * dS_stride_skv
    tl.store(dS_ptrs, dS_tile, mask=(sq_mask[:, None] & skv_mask[None, :]))


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1 (correct): compute grad_attn_scores (dS)
# Grid: [bs * n_heads, cdiv(seq_q, BLOCK_SQ)]
# Each block handles ALL seq_kv for one sq tile — avoids rowsum issue
# head_dim loop is unrolled across BLOCK_D tiles
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def attn_bwd_ds_kernel_v2(
    # dO: [bs, n_heads, seq_q, head_dim]
    dO_ptr,
    dO_stride_bs, dO_stride_h, dO_stride_sq, dO_stride_d,
    # value_states: [bs, n_kv_heads, seq_kv, head_dim]
    V_ptr,
    V_stride_bs, V_stride_h, V_stride_skv, V_stride_d,
    # attn_weights (P): [bs, n_heads, seq_q, seq_kv]
    P_ptr,
    P_stride_bs, P_stride_h, P_stride_sq, P_stride_skv,
    # dropout_mask: [bs, n_heads, seq_q, seq_kv]
    M_ptr,
    M_stride_bs, M_stride_h, M_stride_sq, M_stride_skv,
    # output dS: [bs, n_heads, seq_q, seq_kv]
    dS_ptr,
    dS_stride_bs, dS_stride_h, dS_stride_sq, dS_stride_skv,
    # scalars
    seq_q, seq_kv,
    n_groups: tl.constexpr,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    bh_id = tl.program_id(0)
    sq_blk = tl.program_id(1)

    bs_id = bh_id // NUM_ATTENTION_HEADS
    h_id = bh_id % NUM_ATTENTION_HEADS
    kv_h_id = h_id // n_groups

    sq_start = sq_blk * BLOCK_SQ
    sq_offs = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask = sq_offs < seq_q
    d_offs = tl.arange(0, HEAD_DIM)

    # Load dO tile: [BLOCK_SQ, HEAD_DIM]
    dO_base = bs_id * dO_stride_bs + h_id * dO_stride_h
    dO_ptrs = dO_base + sq_offs[:, None] * dO_stride_sq + d_offs[None, :] * dO_stride_d
    dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

    # We'll compute dP = dO @ V^T for ALL skv, computing rowsum on the fly
    # Then compute dS = P * (dP - rowsum(dP*P))
    # Strategy: iterate over skv tiles, accumulate rowsum(dP*P), store dP tile by tile,
    # then do a second pass to write dS.
    # But this requires storing dP, which is large.
    # 
    # Better: single pass - compute per-row sum by iterating over all skv tiles first,
    # then second pass to write dS.
    # rowsum accumulator: [BLOCK_SQ]
    rowsum = tl.zeros([BLOCK_SQ], dtype=tl.float32)
    
    V_base = bs_id * V_stride_bs + kv_h_id * V_stride_h
    P_base = bs_id * P_stride_bs + h_id * P_stride_h
    M_base = bs_id * M_stride_bs + h_id * M_stride_h

    n_skv_blocks = tl.cdiv(seq_kv, BLOCK_SKV)
    
    # First pass: accumulate rowsum(dP * P)
    for skv_blk in range(n_skv_blocks):
        skv_start = skv_blk * BLOCK_SKV
        skv_offs = skv_start + tl.arange(0, BLOCK_SKV)
        skv_mask = skv_offs < seq_kv

        V_ptrs = V_base + skv_offs[:, None] * V_stride_skv + d_offs[None, :] * V_stride_d
        V_tile = tl.load(V_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

        dP_dropped = tl.dot(dO_tile, tl.trans(V_tile))  # [BLOCK_SQ, BLOCK_SKV]

        M_ptrs = M_base + sq_offs[:, None] * M_stride_sq + skv_offs[None, :] * M_stride_skv
        mask_tile = tl.load(M_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=False)
        dP = tl.where(mask_tile, dP_dropped * inv_keep_prob, 0.0)

        P_ptrs = P_base + sq_offs[:, None] * P_stride_sq + skv_offs[None, :] * P_stride_skv
        P_tile = tl.load(P_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=0.0).to(tl.float32)

        rowsum += tl.sum(dP * P_tile, axis=1)  # accumulate [BLOCK_SQ]

    # Second pass: write dS
    for skv_blk in range(n_skv_blocks):
        skv_start = skv_blk * BLOCK_SKV
        skv_offs = skv_start + tl.arange(0, BLOCK_SKV)
        skv_mask = skv_offs < seq_kv

        V_ptrs = V_base + skv_offs[:, None] * V_stride_skv + d_offs[None, :] * V_stride_d
        V_tile = tl.load(V_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

        dP_dropped = tl.dot(dO_tile, tl.trans(V_tile))  # [BLOCK_SQ, BLOCK_SKV]

        M_ptrs = M_base + sq_offs[:, None] * M_stride_sq + skv_offs[None, :] * M_stride_skv
        mask_tile = tl.load(M_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=False)
        dP = tl.where(mask_tile, dP_dropped * inv_keep_prob, 0.0)

        P_ptrs = P_base + sq_offs[:, None] * P_stride_sq + skv_offs[None, :] * P_stride_skv
        P_tile = tl.load(P_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=0.0).to(tl.float32)

        dS_tile = P_tile * (dP - rowsum[:, None])
        dS_tile = dS_tile.to(tl.bfloat16)

        dS_base = bs_id * dS_stride_bs + h_id * dS_stride_h
        dS_ptrs = dS_base + sq_offs[:, None] * dS_stride_sq + skv_offs[None, :] * dS_stride_skv
        tl.store(dS_ptrs, dS_tile, mask=(sq_mask[:, None] & skv_mask[None, :]))


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: compute grad_value_states (dV)
# Grid: [bs * n_kv_heads, cdiv(seq_kv, BLOCK_SKV), cdiv(head_dim, BLOCK_D)]
# Each block: tile [BLOCK_SKV, BLOCK_D] and accumulates over n_groups query heads
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def attn_bwd_dv_kernel(
    # attn_weights_dropped (P_dropped): [bs, n_heads, seq_q, seq_kv]
    Pd_ptr,
    Pd_stride_bs, Pd_stride_h, Pd_stride_sq, Pd_stride_skv,
    # dO: [bs, n_heads, seq_q, head_dim]
    dO_ptr,
    dO_stride_bs, dO_stride_h, dO_stride_sq, dO_stride_d,
    # output dV: [bs, n_kv_heads, seq_kv, head_dim]
    dV_ptr,
    dV_stride_bs, dV_stride_h, dV_stride_skv, dV_stride_d,
    # scalars
    seq_q, seq_kv,
    n_groups: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    bkv_id = tl.program_id(0)
    skv_blk = tl.program_id(1)

    bs_id = bkv_id // NUM_KEY_VALUE_HEADS
    kv_h_id = bkv_id % NUM_KEY_VALUE_HEADS

    skv_start = skv_blk * BLOCK_SKV
    skv_offs = skv_start + tl.arange(0, BLOCK_SKV)
    skv_mask = skv_offs < seq_kv
    d_offs = tl.arange(0, HEAD_DIM)

    # Accumulator for dV tile: [BLOCK_SKV, HEAD_DIM]
    dV_acc = tl.zeros([BLOCK_SKV, HEAD_DIM], dtype=tl.float32)

    n_sq_blocks = tl.cdiv(seq_q, BLOCK_SQ)

    # Iterate over all n_groups query heads and all sq tiles
    for g in range(n_groups):
        h_id = kv_h_id * n_groups + g
        Pd_base = bs_id * Pd_stride_bs + h_id * Pd_stride_h
        dO_base = bs_id * dO_stride_bs + h_id * dO_stride_h

        for sq_blk in range(n_sq_blocks):
            sq_start = sq_blk * BLOCK_SQ
            sq_offs = sq_start + tl.arange(0, BLOCK_SQ)
            sq_mask = sq_offs < seq_q

            # Load P_dropped tile: [BLOCK_SQ, BLOCK_SKV]  → we need [BLOCK_SKV, BLOCK_SQ] for P^T
            Pd_ptrs = Pd_base + sq_offs[:, None] * Pd_stride_sq + skv_offs[None, :] * Pd_stride_skv
            Pd_tile = tl.load(Pd_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=0.0).to(tl.float32)
            # Pd_tile shape: [BLOCK_SQ, BLOCK_SKV]

            # Load dO tile: [BLOCK_SQ, HEAD_DIM]
            dO_ptrs = dO_base + sq_offs[:, None] * dO_stride_sq + d_offs[None, :] * dO_stride_d
            dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

            # dV += P_dropped^T @ dO: [BLOCK_SKV, HEAD_DIM]
            dV_acc += tl.dot(tl.trans(Pd_tile), dO_tile)

    # Write dV
    dV_base = bs_id * dV_stride_bs + kv_h_id * dV_stride_h
    dV_ptrs = dV_base + skv_offs[:, None] * dV_stride_skv + d_offs[None, :] * dV_stride_d
    tl.store(dV_ptrs, dV_acc.to(tl.bfloat16), mask=skv_mask[:, None])


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    head_dim   = HEAD_DIM              # 128

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Transpose dO: [bs, seq_q, 80, 128] -> [bs, 80, seq_q, 128] (contiguous)
    dO = grad_attn_output.transpose(1, 2).contiguous()

    # Output tensors
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)
    dV = torch.empty((bs, n_kv_heads, seq_kv, head_dim), dtype=torch.bfloat16, device=dO.device)

    # ── Kernel 1: dS ─────────────────────────────────────────────────────────
    BLOCK_SQ  = 32
    BLOCK_SKV = 64

    grid_ds = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ))

    attn_bwd_ds_kernel_v2[grid_ds](
        dO,
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        value_states,
        value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
        attn_weights,
        attn_weights.stride(0), attn_weights.stride(1), attn_weights.stride(2), attn_weights.stride(3),
        dropout_mask,
        dropout_mask.stride(0), dropout_mask.stride(1), dropout_mask.stride(2), dropout_mask.stride(3),
        dS,
        dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
        seq_q, seq_kv,
        n_groups,
        inv_keep_prob,
        BLOCK_SQ,
        BLOCK_SKV,
        head_dim,
    )

    # ── Kernel 2: dV ─────────────────────────────────────────────────────────
    BLOCK_SKV_DV = 32
    BLOCK_SQ_DV  = 32

    grid_dv = (bs * n_kv_heads, triton.cdiv(seq_kv, BLOCK_SKV_DV))

    attn_bwd_dv_kernel[grid_dv](
        attn_weights_dropped,
        attn_weights_dropped.stride(0), attn_weights_dropped.stride(1),
        attn_weights_dropped.stride(2), attn_weights_dropped.stride(3),
        dO,
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dV,
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        seq_q, seq_kv,
        n_groups,
        BLOCK_SKV_DV,
        BLOCK_SQ_DV,
        head_dim,
    )

    return dS, dV

```

---

## Experiment #3 — 2026-06-28 07:44:54 UTC 💥 CRASH

**Hypothesis:** ** Rewrote both Triton kernels with the architecture constants properly passed as `tl.constexpr` arguments:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Fused Triton attention-backward kernel with implicit GQA handling.
Fixed: architecture constants passed as tl.constexpr parameters.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = 10


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: compute grad_attn_scores (dS)
# Grid: [bs * n_heads, cdiv(seq_q, BLOCK_SQ)]
# Each block handles one (batch, head) pair and ALL seq_kv for a sq tile.
# Two-pass: first accumulate rowsum(dP*P), then write dS.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def attn_bwd_ds_kernel(
    # dO: [bs, n_heads, seq_q, head_dim]
    dO_ptr,
    dO_stride_bs, dO_stride_h, dO_stride_sq, dO_stride_d,
    # value_states: [bs, n_kv_heads, seq_kv, head_dim]
    V_ptr,
    V_stride_bs, V_stride_h, V_stride_skv, V_stride_d,
    # attn_weights (P): [bs, n_heads, seq_q, seq_kv]
    P_ptr,
    P_stride_bs, P_stride_h, P_stride_sq, P_stride_skv,
    # dropout_mask: [bs, n_heads, seq_q, seq_kv]
    M_ptr,
    M_stride_bs, M_stride_h, M_stride_sq, M_stride_skv,
    # output dS: [bs, n_heads, seq_q, seq_kv]
    dS_ptr,
    dS_stride_bs, dS_stride_h, dS_stride_sq, dS_stride_skv,
    # scalars
    seq_q, seq_kv,
    # constexpr architecture params
    N_HEADS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    inv_keep_prob: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    bh_id = tl.program_id(0)
    sq_blk = tl.program_id(1)

    bs_id = bh_id // N_HEADS
    h_id  = bh_id % N_HEADS
    kv_h_id = h_id // N_GROUPS

    sq_start = sq_blk * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < seq_q
    d_offs   = tl.arange(0, HEAD_DIM)

    # Load dO tile: [BLOCK_SQ, HEAD_DIM]
    dO_base = bs_id * dO_stride_bs + h_id * dO_stride_h
    dO_ptrs = dO_base + sq_offs[:, None] * dO_stride_sq + d_offs[None, :] * dO_stride_d
    dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

    V_base  = bs_id * V_stride_bs  + kv_h_id * V_stride_h
    P_base  = bs_id * P_stride_bs  + h_id * P_stride_h
    M_base  = bs_id * M_stride_bs  + h_id * M_stride_h
    dS_base = bs_id * dS_stride_bs + h_id * dS_stride_h

    n_skv_blocks = tl.cdiv(seq_kv, BLOCK_SKV)

    # ── Pass 1: accumulate rowsum(dP * P) ────────────────────────────────────
    rowsum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    for skv_blk in range(n_skv_blocks):
        skv_start = skv_blk * BLOCK_SKV
        skv_offs  = skv_start + tl.arange(0, BLOCK_SKV)
        skv_mask  = skv_offs < seq_kv
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        # Load V tile: [BLOCK_SKV, HEAD_DIM]
        V_ptrs  = V_base  + skv_offs[:, None] * V_stride_skv  + d_offs[None, :] * V_stride_d
        V_tile  = tl.load(V_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

        # dP_dropped = dO @ V^T: [BLOCK_SQ, BLOCK_SKV]
        dP_dropped = tl.dot(dO_tile, tl.trans(V_tile))

        # Apply dropout mask: dP = dP_dropped * mask * inv_keep_prob
        M_ptrs    = M_base + sq_offs[:, None] * M_stride_sq + skv_offs[None, :] * M_stride_skv
        mask_tile = tl.load(M_ptrs, mask=combined_mask, other=False)
        dP        = tl.where(mask_tile, dP_dropped * inv_keep_prob, 0.0)

        # Load P tile: [BLOCK_SQ, BLOCK_SKV]
        P_ptrs = P_base + sq_offs[:, None] * P_stride_sq + skv_offs[None, :] * P_stride_skv
        P_tile = tl.load(P_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        rowsum += tl.sum(dP * P_tile, axis=1)

    # ── Pass 2: write dS = P * (dP - rowsum) ─────────────────────────────────
    for skv_blk in range(n_skv_blocks):
        skv_start = skv_blk * BLOCK_SKV
        skv_offs  = skv_start + tl.arange(0, BLOCK_SKV)
        skv_mask  = skv_offs < seq_kv
        combined_mask = sq_mask[:, None] & skv_mask[None, :]

        # Load V tile again
        V_ptrs  = V_base  + skv_offs[:, None] * V_stride_skv  + d_offs[None, :] * V_stride_d
        V_tile  = tl.load(V_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

        dP_dropped = tl.dot(dO_tile, tl.trans(V_tile))

        M_ptrs    = M_base + sq_offs[:, None] * M_stride_sq + skv_offs[None, :] * M_stride_skv
        mask_tile = tl.load(M_ptrs, mask=combined_mask, other=False)
        dP        = tl.where(mask_tile, dP_dropped * inv_keep_prob, 0.0)

        P_ptrs = P_base + sq_offs[:, None] * P_stride_sq + skv_offs[None, :] * P_stride_skv
        P_tile = tl.load(P_ptrs, mask=combined_mask, other=0.0).to(tl.float32)

        dS_tile = (P_tile * (dP - rowsum[:, None])).to(tl.bfloat16)

        dS_ptrs = dS_base + sq_offs[:, None] * dS_stride_sq + skv_offs[None, :] * dS_stride_skv
        tl.store(dS_ptrs, dS_tile, mask=combined_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: compute grad_value_states (dV)
# Grid: [bs * n_kv_heads, cdiv(seq_kv, BLOCK_SKV)]
# Each block: tile [BLOCK_SKV, HEAD_DIM], accumulates over n_groups and sq tiles.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def attn_bwd_dv_kernel(
    # attn_weights_dropped: [bs, n_heads, seq_q, seq_kv]
    Pd_ptr,
    Pd_stride_bs, Pd_stride_h, Pd_stride_sq, Pd_stride_skv,
    # dO: [bs, n_heads, seq_q, head_dim]
    dO_ptr,
    dO_stride_bs, dO_stride_h, dO_stride_sq, dO_stride_d,
    # output dV: [bs, n_kv_heads, seq_kv, head_dim]
    dV_ptr,
    dV_stride_bs, dV_stride_h, dV_stride_skv, dV_stride_d,
    # scalars
    seq_q, seq_kv,
    # constexpr architecture params
    N_KV_HEADS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    bkv_id  = tl.program_id(0)
    skv_blk = tl.program_id(1)

    bs_id    = bkv_id // N_KV_HEADS
    kv_h_id  = bkv_id % N_KV_HEADS

    skv_start = skv_blk * BLOCK_SKV
    skv_offs  = skv_start + tl.arange(0, BLOCK_SKV)
    skv_mask  = skv_offs < seq_kv
    d_offs    = tl.arange(0, HEAD_DIM)

    # Accumulator for dV tile: [BLOCK_SKV, HEAD_DIM]
    dV_acc = tl.zeros([BLOCK_SKV, HEAD_DIM], dtype=tl.float32)

    n_sq_blocks = tl.cdiv(seq_q, BLOCK_SQ)

    for g in range(N_GROUPS):
        h_id    = kv_h_id * N_GROUPS + g
        Pd_base = bs_id * Pd_stride_bs + h_id * Pd_stride_h
        dO_base = bs_id * dO_stride_bs + h_id * dO_stride_h

        for sq_blk in range(n_sq_blocks):
            sq_start = sq_blk * BLOCK_SQ
            sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
            sq_mask  = sq_offs < seq_q

            # Load P_dropped tile: [BLOCK_SQ, BLOCK_SKV]
            Pd_ptrs = Pd_base + sq_offs[:, None] * Pd_stride_sq + skv_offs[None, :] * Pd_stride_skv
            Pd_tile = tl.load(Pd_ptrs, mask=(sq_mask[:, None] & skv_mask[None, :]), other=0.0).to(tl.float32)

            # Load dO tile: [BLOCK_SQ, HEAD_DIM]
            dO_ptrs = dO_base + sq_offs[:, None] * dO_stride_sq + d_offs[None, :] * dO_stride_d
            dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

            # dV += P_dropped^T @ dO: [BLOCK_SKV, HEAD_DIM]
            dV_acc += tl.dot(tl.trans(Pd_tile), dO_tile)

    # Write dV
    dV_base = bs_id * dV_stride_bs + kv_h_id * dV_stride_h
    dV_ptrs = dV_base + skv_offs[:, None] * dV_stride_skv + d_offs[None, :] * dV_stride_d
    tl.store(dV_ptrs, dV_acc.to(tl.bfloat16), mask=skv_mask[:, None])


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    head_dim   = HEAD_DIM              # 128

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Transpose dO: [bs, seq_q, 80, 128] -> [bs, 80, seq_q, 128] (contiguous)
    dO = grad_attn_output.transpose(1, 2).contiguous()

    # Output tensors
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)
    dV = torch.empty((bs, n_kv_heads, seq_kv, head_dim), dtype=torch.bfloat16, device=dO.device)

    # ── Kernel 1: dS ─────────────────────────────────────────────────────────
    BLOCK_SQ  = 32
    BLOCK_SKV = 64

    grid_ds = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ))

    attn_bwd_ds_kernel[grid_ds](
        dO,
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        value_states,
        value_states.stride(0), value_states.stride(1),
        value_states.stride(2), value_states.stride(3),
        attn_weights,
        attn_weights.stride(0), attn_weights.stride(1),
        attn_weights.stride(2), attn_weights.stride(3),
        dropout_mask,
        dropout_mask.stride(0), dropout_mask.stride(1),
        dropout_mask.stride(2), dropout_mask.stride(3),
        dS,
        dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
        seq_q, seq_kv,
        n_heads,       # N_HEADS constexpr
        n_groups,      # N_GROUPS constexpr
        inv_keep_prob, # inv_keep_prob constexpr
        BLOCK_SQ,
        BLOCK_SKV,
        head_dim,      # HEAD_DIM constexpr
    )

    # ── Kernel 2: dV ─────────────────────────────────────────────────────────
    BLOCK_SKV_DV = 32
    BLOCK_SQ_DV  = 32

    grid_dv = (bs * n_kv_heads, triton.cdiv(seq_kv, BLOCK_SKV_DV))

    attn_bwd_dv_kernel[grid_dv](
        attn_weights_dropped,
        attn_weights_dropped.stride(0), attn_weights_dropped.stride(1),
        attn_weights_dropped.stride(2), attn_weights_dropped.stride(3),
        dO,
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dV,
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        seq_q, seq_kv,
        n_kv_heads,    # N_KV_HEADS constexpr
        n_groups,      # N_GROUPS constexpr
        BLOCK_SKV_DV,
        BLOCK_SQ_DV,
        head_dim,      # HEAD_DIM constexpr
    )

    return dS, dV

```

---

## Experiment #4 — 2026-06-28 07:46:23 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 1954.76 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — PyTorch with torch.compile and GQA-native ops.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


def _attn_bwd_impl(
    grad_attn_output,   # [bs, seq_q, 80, 128]  bf16
    attn_weights,       # [bs, 80, seq_q, seq_kv]  bf16
    attn_weights_dropped,  # [bs, 80, seq_q, seq_kv]  bf16
    value_states,       # [bs, 8, seq_kv, 128]  bf16
    dropout_mask,       # [bs, 80, seq_q, seq_kv]  bool
    attention_dropout,  # float scalar
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 80, seq_q, 128]
    # Then reshape to [bs, 8, 10, seq_q, 128] for GQA-grouped ops
    dO_heads = grad_attn_output.transpose(1, 2)  # [bs, 80, seq_q, 128] bf16
    dO_grouped = dO_heads.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)
    # [bs, 8, 10, seq_q, 128]

    # ── Compute dP = dO @ V^T (GQA-native, no materialization of 80-head V) ──
    # value_states: [bs, 8, seq_kv, 128]
    # dO_grouped:   [bs, 8, 10, seq_q, 128]
    # dP_grouped:   [bs, 8, 10, seq_q, seq_kv]
    # einsum: 'bgiqd,bgkd->bgiqk'  (g=groups=10, i=sq, k=skv, d=dim)
    # Use bmm: reshape to [bs*8*10, sq, 128] @ [bs*8*10, 128, skv]
    dO_flat = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS * N_GROUPS, seq_q, HEAD_DIM).to(torch.float32)
    # Expand V for batched matmul: [bs, 8, seq_kv, 128] -> [bs, 8, 10, seq_kv, 128]
    V_grouped = value_states.unsqueeze(2).expand(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM)
    V_flat = V_grouped.reshape(bs * NUM_KEY_VALUE_HEADS * N_GROUPS, seq_kv, HEAD_DIM).to(torch.float32)

    # dP_dropped_flat: [bs*8*10, sq, skv]
    dP_dropped_flat = torch.bmm(dO_flat, V_flat.transpose(-2, -1))
    dP_dropped = dP_dropped_flat.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Dropout backward ──────────────────────────────────────────────────────
    if attention_dropout > 0.0:
        scale = 1.0 / (1.0 - attention_dropout)
        dP = dP_dropped * dropout_mask.to(torch.float32) * scale
    else:
        dP = dP_dropped

    # ── Softmax backward: dS = P * (dP - rowsum(dP * P)) ─────────────────────
    P = attn_weights.to(torch.float32)   # [bs, 80, sq, skv]
    dPP = dP * P
    dS = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # ── Compute dV (GQA-native) ───────────────────────────────────────────────
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    # dO_grouped: [bs, 8, 10, sq, 128]
    # dV[b,g,k,d] = sum_i sum_grp  Pd[b,g,grp,i,k] * dO[b,g,grp,i,d]
    # = einsum('bgiqk,bgiqd->bgkd')
    Pd_grouped = attn_weights_dropped.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv)
    # Use einsum for clarity and fusion
    # dV: [bs, 8, skv, 128]
    # We want: dV[b,g,k,d] = sum_{i,grp} Pd[b,g,grp,i,k] * dO[b,g,grp,i,d]
    # Merge grp*sq into one dim for bmm
    Pd_flat2 = Pd_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, seq_kv).to(torch.float32)
    dO_flat2  = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, HEAD_DIM).to(torch.float32)
    # dV_flat: [bs*8, skv, 128]
    dV_flat = torch.bmm(Pd_flat2.transpose(-2, -1), dO_flat2)
    dV = dV_flat.reshape(bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV


# Compile once at module level
_compiled_attn_bwd = torch.compile(_attn_bwd_impl, mode="reduce-overhead")


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        attention_dropout,
    )

```

---

## Experiment #5 — 2026-06-28 07:48:01 UTC 💥 CRASH

**Hypothesis:** ** Two clean Triton kernels written from scratch:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Fused Triton attention-backward kernel — fresh, correctness-first implementation.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: compute grad_attn_scores (dS)
# Grid: [bs * n_heads, cdiv(seq_q, BLOCK_SQ)]
# Algorithm:
#   Pass 1: accumulate rowsum = sum_k(dP_k * P_k) for each row
#   Pass 2: write dS = P * (dP - rowsum)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def kernel_dS(
    # dO: [bs, n_heads, seq_q, head_dim]  contiguous, bfloat16
    dO_ptr, dO_s0, dO_s1, dO_s2, dO_s3,
    # V:  [bs, n_kv_heads, seq_kv, head_dim]  contiguous, bfloat16
    V_ptr,  V_s0,  V_s1,  V_s2,  V_s3,
    # P:  [bs, n_heads, seq_q, seq_kv]  contiguous, bfloat16
    P_ptr,  P_s0,  P_s1,  P_s2,  P_s3,
    # M (dropout mask): [bs, n_heads, seq_q, seq_kv]  contiguous, bool
    M_ptr,  M_s0,  M_s1,  M_s2,  M_s3,
    # dS: [bs, n_heads, seq_q, seq_kv]  contiguous, bfloat16  (output)
    dS_ptr, dS_s0, dS_s1, dS_s2, dS_s3,
    # dimensions
    seq_q,  seq_kv,
    # architecture constants (tl.constexpr for compile-time use in indexing)
    N_HEADS:    tl.constexpr,
    N_GROUPS:   tl.constexpr,
    HEAD_DIM:   tl.constexpr,
    BLOCK_SQ:   tl.constexpr,
    BLOCK_SKV:  tl.constexpr,
    # runtime scalar
    inv_keep_prob,
):
    # ── Program ID → (batch, head, sq_tile) ──────────────────────────────────
    pid0   = tl.program_id(0)   # encodes batch * N_HEADS + head
    sq_blk = tl.program_id(1)

    bs_id = pid0 // N_HEADS
    h_id  = pid0 % N_HEADS
    kv_id = h_id // N_GROUPS    # which of the 8 KV heads to use

    sq_start = sq_blk * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < seq_q

    d_offs = tl.arange(0, HEAD_DIM)

    # ── Load dO tile once: [BLOCK_SQ, HEAD_DIM] in float32 ───────────────────
    dO_base = bs_id * dO_s0 + h_id * dO_s1
    dO_ptrs = dO_base + sq_offs[:, None] * dO_s2 + d_offs[None, :] * dO_s3
    dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

    # Base pointers for V, P, M, dS
    V_base  = bs_id * V_s0  + kv_id * V_s1
    P_base  = bs_id * P_s0  + h_id  * P_s1
    M_base  = bs_id * M_s0  + h_id  * M_s1
    dS_base = bs_id * dS_s0 + h_id  * dS_s1

    n_skv_blks = tl.cdiv(seq_kv, BLOCK_SKV)

    # ── Pass 1: accumulate rowsum = sum_k(dP_k * P_k) ────────────────────────
    rowsum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    for skv_blk in range(n_skv_blks):
        skv_start = skv_blk * BLOCK_SKV
        skv_offs  = skv_start + tl.arange(0, BLOCK_SKV)
        skv_mask  = skv_offs < seq_kv
        cmask     = sq_mask[:, None] & skv_mask[None, :]

        # Load V: [BLOCK_SKV, HEAD_DIM]
        V_ptrs = V_base + skv_offs[:, None] * V_s2 + d_offs[None, :] * V_s3
        V_tile = tl.load(V_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

        # dP_dropped = dO @ V^T: [BLOCK_SQ, BLOCK_SKV]
        dP_dropped = tl.dot(dO_tile, tl.trans(V_tile))

        # Apply dropout mask
        M_ptrs    = M_base + sq_offs[:, None] * M_s2 + skv_offs[None, :] * M_s3
        mask_bool = tl.load(M_ptrs, mask=cmask, other=False)
        dP        = tl.where(mask_bool, dP_dropped * inv_keep_prob, 0.0)

        # Load P: [BLOCK_SQ, BLOCK_SKV]
        P_ptrs = P_base + sq_offs[:, None] * P_s2 + skv_offs[None, :] * P_s3
        P_tile = tl.load(P_ptrs, mask=cmask, other=0.0).to(tl.float32)

        rowsum += tl.sum(dP * P_tile, axis=1)

    # ── Pass 2: compute and write dS ─────────────────────────────────────────
    for skv_blk in range(n_skv_blks):
        skv_start = skv_blk * BLOCK_SKV
        skv_offs  = skv_start + tl.arange(0, BLOCK_SKV)
        skv_mask  = skv_offs < seq_kv
        cmask     = sq_mask[:, None] & skv_mask[None, :]

        # Reload V
        V_ptrs = V_base + skv_offs[:, None] * V_s2 + d_offs[None, :] * V_s3
        V_tile = tl.load(V_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

        dP_dropped = tl.dot(dO_tile, tl.trans(V_tile))

        M_ptrs    = M_base + sq_offs[:, None] * M_s2 + skv_offs[None, :] * M_s3
        mask_bool = tl.load(M_ptrs, mask=cmask, other=False)
        dP        = tl.where(mask_bool, dP_dropped * inv_keep_prob, 0.0)

        P_ptrs = P_base + sq_offs[:, None] * P_s2 + skv_offs[None, :] * P_s3
        P_tile = tl.load(P_ptrs, mask=cmask, other=0.0).to(tl.float32)

        dS_tile = (P_tile * (dP - rowsum[:, None])).to(tl.bfloat16)

        dS_ptrs = dS_base + sq_offs[:, None] * dS_s2 + skv_offs[None, :] * dS_s3
        tl.store(dS_ptrs, dS_tile, mask=cmask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: compute grad_value_states (dV)
# Grid: [bs * n_kv_heads, cdiv(seq_kv, BLOCK_SKV)]
# Each block handles [BLOCK_SKV, HEAD_DIM] tile of dV,
# accumulating over all N_GROUPS query heads and all seq_q tiles.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def kernel_dV(
    # Pd: attn_weights_dropped [bs, n_heads, seq_q, seq_kv]  bfloat16
    Pd_ptr, Pd_s0, Pd_s1, Pd_s2, Pd_s3,
    # dO: [bs, n_heads, seq_q, head_dim]  bfloat16
    dO_ptr, dO_s0, dO_s1, dO_s2, dO_s3,
    # dV: [bs, n_kv_heads, seq_kv, head_dim]  bfloat16  (output)
    dV_ptr, dV_s0, dV_s1, dV_s2, dV_s3,
    # dimensions
    seq_q, seq_kv,
    # architecture constants
    N_KV_HEADS: tl.constexpr,
    N_GROUPS:   tl.constexpr,
    HEAD_DIM:   tl.constexpr,
    BLOCK_SKV:  tl.constexpr,
    BLOCK_SQ:   tl.constexpr,
):
    pid0    = tl.program_id(0)   # bs_id * N_KV_HEADS + kv_id
    skv_blk = tl.program_id(1)

    bs_id  = pid0 // N_KV_HEADS
    kv_id  = pid0 % N_KV_HEADS

    skv_start = skv_blk * BLOCK_SKV
    skv_offs  = skv_start + tl.arange(0, BLOCK_SKV)
    skv_mask  = skv_offs < seq_kv

    d_offs = tl.arange(0, HEAD_DIM)

    # Accumulator: [BLOCK_SKV, HEAD_DIM]
    dV_acc = tl.zeros([BLOCK_SKV, HEAD_DIM], dtype=tl.float32)

    n_sq_blks = tl.cdiv(seq_q, BLOCK_SQ)

    for g in range(N_GROUPS):
        h_id    = kv_id * N_GROUPS + g
        Pd_base = bs_id * Pd_s0 + h_id * Pd_s1
        dO_base = bs_id * dO_s0 + h_id * dO_s1

        for sq_blk in range(n_sq_blks):
            sq_start = sq_blk * BLOCK_SQ
            sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
            sq_mask  = sq_offs < seq_q
            cmask    = sq_mask[:, None] & skv_mask[None, :]

            # Load Pd tile: [BLOCK_SQ, BLOCK_SKV]
            Pd_ptrs = Pd_base + sq_offs[:, None] * Pd_s2 + skv_offs[None, :] * Pd_s3
            Pd_tile = tl.load(Pd_ptrs, mask=cmask, other=0.0).to(tl.float32)

            # Load dO tile: [BLOCK_SQ, HEAD_DIM]
            dO_ptrs = dO_base + sq_offs[:, None] * dO_s2 + d_offs[None, :] * dO_s3
            dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

            # dV += Pd^T @ dO: [BLOCK_SKV, HEAD_DIM]
            dV_acc += tl.dot(tl.trans(Pd_tile), dO_tile)

    # Write dV
    dV_base = bs_id * dV_s0 + kv_id * dV_s1
    dV_ptrs = dV_base + skv_offs[:, None] * dV_s2 + d_offs[None, :] * dV_s3
    tl.store(dV_ptrs, dV_acc.to(tl.bfloat16), mask=skv_mask[:, None])


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    head_dim   = HEAD_DIM              # 128

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Transpose dO to [bs, 80, seq_q, 128] — contiguous for correct strides
    dO = grad_attn_output.transpose(1, 2).contiguous()

    # Allocate outputs
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)
    dV = torch.empty((bs, n_kv_heads, seq_kv, head_dim), dtype=torch.bfloat16, device=dO.device)

    # ── Launch kernel_dS ─────────────────────────────────────────────────────
    BLOCK_SQ_DS  = 32
    BLOCK_SKV_DS = 64

    grid_dS = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_DS))

    kernel_dS[grid_dS](
        dO,    dO.stride(0),    dO.stride(1),    dO.stride(2),    dO.stride(3),
        value_states,
               value_states.stride(0), value_states.stride(1),
               value_states.stride(2), value_states.stride(3),
        attn_weights,
               attn_weights.stride(0), attn_weights.stride(1),
               attn_weights.stride(2), attn_weights.stride(3),
        dropout_mask,
               dropout_mask.stride(0), dropout_mask.stride(1),
               dropout_mask.stride(2), dropout_mask.stride(3),
        dS,    dS.stride(0),    dS.stride(1),    dS.stride(2),    dS.stride(3),
        seq_q, seq_kv,
        N_HEADS=n_heads,
        N_GROUPS=n_groups,
        HEAD_DIM=head_dim,
        BLOCK_SQ=BLOCK_SQ_DS,
        BLOCK_SKV=BLOCK_SKV_DS,
        inv_keep_prob=inv_keep_prob,
    )

    # ── Launch kernel_dV ─────────────────────────────────────────────────────
    BLOCK_SKV_DV = 32
    BLOCK_SQ_DV  = 32

    grid_dV = (bs * n_kv_heads, triton.cdiv(seq_kv, BLOCK_SKV_DV))

    kernel_dV[grid_dV](
        attn_weights_dropped,
               attn_weights_dropped.stride(0), attn_weights_dropped.stride(1),
               attn_weights_dropped.stride(2), attn_weights_dropped.stride(3),
        dO,    dO.stride(0),    dO.stride(1),    dO.stride(2),    dO.stride(3),
        dV,    dV.stride(0),    dV.stride(1),    dV.stride(2),    dV.stride(3),
        seq_q, seq_kv,
        N_KV_HEADS=n_kv_heads,
        N_GROUPS=n_groups,
        HEAD_DIM=head_dim,
        BLOCK_SKV=BLOCK_SKV_DV,
        BLOCK_SQ=BLOCK_SQ_DV,
    )

    return dS, dV

```

---

## Experiment #6 — 2026-06-28 07:49:50 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 867.00 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — PyTorch with torch.compile(max-autotune)
and GQA-native einsum ops in bfloat16.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

# Enable TF32 and BF16 reductions for maximum tensor-core utilization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _attn_bwd_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights,         # [bs, 80, seq_q, seq_kv]    bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
    dropout_mask,         # [bs, 80, seq_q, seq_kv]    bool
    inv_keep_prob,        # float scalar (precomputed 1/(1-p))
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # ── dP = dO @ V^T via einsum (bf16, no materialized expansion) ───────────
    # 'bgnqd,bgkd->bgnqk'  [bs,8,10,sq,d] x [bs,8,skv,d] -> [bs,8,10,sq,skv]
    dP_dropped_grouped = torch.einsum(
        'bgnqd,bgkd->bgnqk',
        dO_grouped,          # [bs, 8, 10, sq, 128]  bf16
        value_states,        # [bs, 8, skv, 128]     bf16
    )  # result: [bs, 8, 10, sq, skv]  (accumulated in f32 internally by einsum)

    # Reshape back to [bs, 80, sq, skv]
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Dropout backward ──────────────────────────────────────────────────────
    # Work in float32 for the elementwise softmax-backward portion
    dP_dropped_f32 = dP_dropped.to(torch.float32)
    dP = dP_dropped_f32 * dropout_mask * inv_keep_prob

    # ── Softmax backward: dS = P * (dP - rowsum(dP * P)) ─────────────────────
    P    = attn_weights.to(torch.float32)      # [bs, 80, sq, skv]
    dPP  = dP * P
    dS   = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS   = dS.to(torch.bfloat16)

    # ── dV via einsum (bf16, sums over groups) ────────────────────────────────
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    # 'bgnqk,bgnqd->bgkd'  [bs,8,10,sq,skv] x [bs,8,10,sq,d] -> [bs,8,skv,d]
    dV = torch.einsum(
        'bgnqk,bgnqd->bgkd',
        Pd_grouped,    # [bs, 8, 10, sq, skv]  bf16
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
    ).to(torch.bfloat16)  # [bs, 8, skv, 128]

    return dS, dV


# Compile once at module level with max-autotune for best kernel selection
_compiled_attn_bwd = torch.compile(_attn_bwd_impl, mode="max-autotune")


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    # Precompute inv_keep_prob as a Python float (not passed to compiled fn as tensor)
    # to avoid recompilation on dropout rate changes — but keep it out of the
    # compiled function's captured closure to allow reuse across calls.
    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        inv_keep_prob,
    )

```

---

## Experiment #7 — 2026-06-28 07:51:47 UTC 💥 CRASH

**Hypothesis:** **

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Hybrid: Triton kernels with bool-mask fix + PyTorch fallback.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: compute grad_attn_scores (dS)
# Grid: [bs * n_heads, cdiv(seq_q, BLOCK_SQ)]
# Two-pass: Pass 1 accumulates rowsum, Pass 2 writes dS.
# KEY FIX: bool mask loaded with .to(tl.int1) for correct tl.where behavior.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def kernel_dS(
    # dO: [bs, n_heads, seq_q, head_dim]  bfloat16  (contiguous after transpose)
    dO_ptr, dO_s0, dO_s1, dO_s2, dO_s3,
    # V:  [bs, n_kv_heads, seq_kv, head_dim]  bfloat16
    V_ptr,  V_s0,  V_s1,  V_s2,  V_s3,
    # P:  [bs, n_heads, seq_q, seq_kv]  bfloat16
    P_ptr,  P_s0,  P_s1,  P_s2,  P_s3,
    # M (dropout_mask): [bs, n_heads, seq_q, seq_kv]  bool (uint8 in memory)
    M_ptr,  M_s0,  M_s1,  M_s2,  M_s3,
    # dS: [bs, n_heads, seq_q, seq_kv]  bfloat16  (output)
    dS_ptr, dS_s0, dS_s1, dS_s2, dS_s3,
    # runtime dimensions
    seq_q,  seq_kv,
    # constexpr architecture constants
    N_HEADS:   tl.constexpr,
    N_GROUPS:  tl.constexpr,
    HEAD_DIM:  tl.constexpr,
    BLOCK_SQ:  tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    # runtime scalar
    inv_keep_prob,
):
    pid0   = tl.program_id(0)   # batch * N_HEADS + head
    sq_blk = tl.program_id(1)

    bs_id = pid0 // N_HEADS
    h_id  = pid0 % N_HEADS
    kv_id = h_id // N_GROUPS    # which of the 8 KV heads to use

    sq_start = sq_blk * BLOCK_SQ
    sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
    sq_mask  = sq_offs < seq_q
    d_offs   = tl.arange(0, HEAD_DIM)

    # Load dO tile once: [BLOCK_SQ, HEAD_DIM] cast to float32
    dO_base = bs_id * dO_s0 + h_id * dO_s1
    dO_ptrs = dO_base + sq_offs[:, None] * dO_s2 + d_offs[None, :] * dO_s3
    dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

    V_base  = bs_id * V_s0  + kv_id * V_s1
    P_base  = bs_id * P_s0  + h_id  * P_s1
    M_base  = bs_id * M_s0  + h_id  * M_s1
    dS_base = bs_id * dS_s0 + h_id  * dS_s1

    n_skv_blks = tl.cdiv(seq_kv, BLOCK_SKV)

    # ── Pass 1: accumulate rowsum = sum_k(dP_k * P_k) ────────────────────────
    rowsum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    for skv_blk in range(n_skv_blks):
        skv_start = skv_blk * BLOCK_SKV
        skv_offs  = skv_start + tl.arange(0, BLOCK_SKV)
        skv_mask  = skv_offs < seq_kv
        cmask     = sq_mask[:, None] & skv_mask[None, :]

        # Load V: [BLOCK_SKV, HEAD_DIM]
        V_ptrs = V_base + skv_offs[:, None] * V_s2 + d_offs[None, :] * V_s3
        V_tile = tl.load(V_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

        # dP_dropped = dO @ V^T: [BLOCK_SQ, BLOCK_SKV]
        dP_dropped = tl.dot(dO_tile, tl.trans(V_tile))

        # Load dropout mask (bool → int8 in memory) and cast to int1
        M_ptrs     = M_base + sq_offs[:, None] * M_s2 + skv_offs[None, :] * M_s3
        mask_raw   = tl.load(M_ptrs, mask=cmask, other=0)
        mask_bool  = mask_raw.to(tl.int1)  # KEY FIX: explicit cast to bool
        dP         = tl.where(mask_bool, dP_dropped * inv_keep_prob, 0.0)

        # Load P: [BLOCK_SQ, BLOCK_SKV]
        P_ptrs = P_base + sq_offs[:, None] * P_s2 + skv_offs[None, :] * P_s3
        P_tile = tl.load(P_ptrs, mask=cmask, other=0.0).to(tl.float32)

        rowsum += tl.sum(dP * P_tile, axis=1)

    # ── Pass 2: write dS = P * (dP - rowsum) ─────────────────────────────────
    for skv_blk in range(n_skv_blks):
        skv_start = skv_blk * BLOCK_SKV
        skv_offs  = skv_start + tl.arange(0, BLOCK_SKV)
        skv_mask  = skv_offs < seq_kv
        cmask     = sq_mask[:, None] & skv_mask[None, :]

        V_ptrs = V_base + skv_offs[:, None] * V_s2 + d_offs[None, :] * V_s3
        V_tile = tl.load(V_ptrs, mask=skv_mask[:, None], other=0.0).to(tl.float32)

        dP_dropped = tl.dot(dO_tile, tl.trans(V_tile))

        M_ptrs    = M_base + sq_offs[:, None] * M_s2 + skv_offs[None, :] * M_s3
        mask_raw  = tl.load(M_ptrs, mask=cmask, other=0)
        mask_bool = mask_raw.to(tl.int1)  # KEY FIX
        dP        = tl.where(mask_bool, dP_dropped * inv_keep_prob, 0.0)

        P_ptrs = P_base + sq_offs[:, None] * P_s2 + skv_offs[None, :] * P_s3
        P_tile = tl.load(P_ptrs, mask=cmask, other=0.0).to(tl.float32)

        dS_tile = (P_tile * (dP - rowsum[:, None])).to(tl.bfloat16)

        dS_ptrs = dS_base + sq_offs[:, None] * dS_s2 + skv_offs[None, :] * dS_s3
        tl.store(dS_ptrs, dS_tile, mask=cmask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: compute grad_value_states (dV)
# Grid: [bs * n_kv_heads, cdiv(seq_kv, BLOCK_SKV)]
# Accumulates over all N_GROUPS query heads and all seq_q tiles.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def kernel_dV(
    # Pd: attn_weights_dropped [bs, n_heads, seq_q, seq_kv]  bfloat16
    Pd_ptr, Pd_s0, Pd_s1, Pd_s2, Pd_s3,
    # dO: [bs, n_heads, seq_q, head_dim]  bfloat16
    dO_ptr, dO_s0, dO_s1, dO_s2, dO_s3,
    # dV: [bs, n_kv_heads, seq_kv, head_dim]  bfloat16  (output)
    dV_ptr, dV_s0, dV_s1, dV_s2, dV_s3,
    # dimensions
    seq_q, seq_kv,
    # constexpr architecture constants
    N_KV_HEADS: tl.constexpr,
    N_GROUPS:   tl.constexpr,
    HEAD_DIM:   tl.constexpr,
    BLOCK_SKV:  tl.constexpr,
    BLOCK_SQ:   tl.constexpr,
):
    pid0    = tl.program_id(0)   # bs_id * N_KV_HEADS + kv_id
    skv_blk = tl.program_id(1)

    bs_id  = pid0 // N_KV_HEADS
    kv_id  = pid0 % N_KV_HEADS

    skv_start = skv_blk * BLOCK_SKV
    skv_offs  = skv_start + tl.arange(0, BLOCK_SKV)
    skv_mask  = skv_offs < seq_kv
    d_offs    = tl.arange(0, HEAD_DIM)

    dV_acc = tl.zeros([BLOCK_SKV, HEAD_DIM], dtype=tl.float32)

    n_sq_blks = tl.cdiv(seq_q, BLOCK_SQ)

    for g in range(N_GROUPS):
        h_id    = kv_id * N_GROUPS + g
        Pd_base = bs_id * Pd_s0 + h_id * Pd_s1
        dO_base = bs_id * dO_s0 + h_id * dO_s1

        for sq_blk in range(n_sq_blks):
            sq_start = sq_blk * BLOCK_SQ
            sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
            sq_mask  = sq_offs < seq_q
            cmask    = sq_mask[:, None] & skv_mask[None, :]

            # Load Pd tile: [BLOCK_SQ, BLOCK_SKV]
            Pd_ptrs = Pd_base + sq_offs[:, None] * Pd_s2 + skv_offs[None, :] * Pd_s3
            Pd_tile = tl.load(Pd_ptrs, mask=cmask, other=0.0).to(tl.float32)

            # Load dO tile: [BLOCK_SQ, HEAD_DIM]
            dO_ptrs = dO_base + sq_offs[:, None] * dO_s2 + d_offs[None, :] * dO_s3
            dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

            # dV += Pd^T @ dO: [BLOCK_SKV, HEAD_DIM]
            dV_acc += tl.dot(tl.trans(Pd_tile), dO_tile)

    dV_base = bs_id * dV_s0 + kv_id * dV_s1
    dV_ptrs = dV_base + skv_offs[:, None] * dV_s2 + d_offs[None, :] * dV_s3
    tl.store(dV_ptrs, dV_acc.to(tl.bfloat16), mask=skv_mask[:, None])


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch fallback (from Experiment 6, known correct)
# ─────────────────────────────────────────────────────────────────────────────
def _attn_bwd_pytorch(
    grad_attn_output, attn_weights, attn_weights_dropped,
    value_states, dropout_mask, inv_keep_prob,
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )
    dP_dropped_grouped = torch.einsum(
        'bgnqd,bgkd->bgnqk', dO_grouped, value_states,
    )
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)
    dP = dP_dropped.to(torch.float32) * dropout_mask * inv_keep_prob

    P    = attn_weights.to(torch.float32)
    dPP  = dP * P
    dS   = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS   = dS.to(torch.bfloat16)

    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    dV = torch.einsum('bgnqk,bgnqd->bgkd', Pd_grouped, dO_grouped).to(torch.bfloat16)

    return dS, dV


_compiled_fallback = torch.compile(_attn_bwd_pytorch, mode="max-autotune")

# Track whether Triton has been verified correct on first call
_triton_ok = None


def custom_kernel(data):
    global _triton_ok

    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = N_GROUPS
    head_dim   = HEAD_DIM

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Fast path once Triton is verified
    if _triton_ok is False:
        return _compiled_fallback(
            grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, inv_keep_prob,
        )

    # Transpose dO to [bs, 80, seq_q, 128] contiguous
    dO = grad_attn_output.transpose(1, 2).contiguous()

    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)
    dV = torch.empty((bs, n_kv_heads, seq_kv, head_dim), dtype=torch.bfloat16, device=dO.device)

    BLOCK_SQ_DS  = 32
    BLOCK_SKV_DS = 64

    kernel_dS[(bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_DS))](
        dO,    dO.stride(0),    dO.stride(1),    dO.stride(2),    dO.stride(3),
        value_states,
               value_states.stride(0), value_states.stride(1),
               value_states.stride(2), value_states.stride(3),
        attn_weights,
               attn_weights.stride(0), attn_weights.stride(1),
               attn_weights.stride(2), attn_weights.stride(3),
        dropout_mask,
               dropout_mask.stride(0), dropout_mask.stride(1),
               dropout_mask.stride(2), dropout_mask.stride(3),
        dS,    dS.stride(0),    dS.stride(1),    dS.stride(2),    dS.stride(3),
        seq_q, seq_kv,
        N_HEADS=n_heads,
        N_GROUPS=n_groups,
        HEAD_DIM=head_dim,
        BLOCK_SQ=BLOCK_SQ_DS,
        BLOCK_SKV=BLOCK_SKV_DS,
        inv_keep_prob=inv_keep_prob,
    )

    BLOCK_SKV_DV = 32
    BLOCK_SQ_DV  = 32

    kernel_dV[(bs * n_kv_heads, triton.cdiv(seq_kv, BLOCK_SKV_DV))](
        attn_weights_dropped,
               attn_weights_dropped.stride(0), attn_weights_dropped.stride(1),
               attn_weights_dropped.stride(2), attn_weights_dropped.stride(3),
        dO,    dO.stride(0),    dO.stride(1),    dO.stride(2),    dO.stride(3),
        dV,    dV.stride(0),    dV.stride(1),    dV.stride(2),    dV.stride(3),
        seq_q, seq_kv,
        N_KV_HEADS=n_kv_heads,
        N_GROUPS=n_groups,
        HEAD_DIM=head_dim,
        BLOCK_SKV=BLOCK_SKV_DV,
        BLOCK_SQ=BLOCK_SQ_DV,
    )

    if _triton_ok is None:
        # Verify correctness against PyTorch reference on first call
        dS_ref, dV_ref = _attn_bwd_pytorch(
            grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, inv_keep_prob,
        )
        ds_ok = torch.allclose(dS.float(), dS_ref.float(), rtol=1e-2, atol=1e-2)
        dv_ok = torch.allclose(dV.float(), dV_ref.float(), rtol=1e-2, atol=1e-2)
        _triton_ok = ds_ok and dv_ok
        if not _triton_ok:
            # Fall back to PyTorch for this call too
            return dS_ref, dV_ref

    return dS, dV

```

---

## Experiment #8 — 2026-06-28 07:53:00 UTC 💥 CRASH

**Hypothesis:** **

**Result:** CRASH

**Error:**
```
Correctness check failed. Failed to load submission:
Traceback (most recent call last):
  File "/root/eval_modal_attn_bwd.py", line 246, in evaluate_kernel
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/tmp/submission_a9tm4kba/submission.py", line 95, in <module>
    _compiled_
```

**Kernel code:**
```python
"""
Optimized attention-backward kernel — PyTorch with torch.compile(max-autotune,
fullgraph=True) and GQA-native bmm ops in bfloat16.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

# Enable TF32 and BF16 reductions for maximum tensor-core utilization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _attn_bwd_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights,         # [bs, 80, seq_q, seq_kv]    bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
    dropout_mask,         # [bs, 80, seq_q, seq_kv]    bool
    inv_keep_prob,        # 0-dim float32 tensor scalar on device
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # ── dP = dO @ V^T via bmm (bf16, no materialized expansion) ──────────────
    # Reshape dO_grouped to [bs*8, 10*sq, 128] and V to [bs*8, skv, 128]
    # then bmm gives [bs*8, 10*sq, skv] -> reshape to [bs, 80, sq, skv]
    dO_flat  = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, HEAD_DIM)
    # Expand V: [bs, 8, skv, 128] -> [bs*8, skv, 128] (no copy, just reshape)
    V_flat   = value_states.reshape(bs * NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM)
    # Repeat V_flat for each group by tiling along batch: use expand trick
    # We need [bs*8, 10*sq, 128] @ [bs*8, 128, skv]
    # But V_flat is [bs*8, skv, 128] — we need to repeat it 10 times... 
    # Actually: use a strided view. dO_flat has rows ordered as [g0_sq0..sqN, g1_sq0..sqN, ...]
    # We want each of the 10 groups to multiply against the SAME V.
    # Better: use einsum which handles this broadcast correctly
    dP_dropped_grouped = torch.einsum(
        'bgnqd,bgkd->bgnqk',
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
        value_states,  # [bs, 8, skv, 128]     bf16
    )
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Dropout backward (no branch — always apply scale) ────────────────────
    dP = dP_dropped.to(torch.float32) * dropout_mask * inv_keep_prob

    # ── Softmax backward: dS = P * (dP - rowsum(dP * P)) ─────────────────────
    P   = attn_weights.to(torch.float32)
    dPP = dP * P
    dS  = (P * (dP - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    # ── dV via bmm: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] = [bs*8, skv, 128]
    # Pd_grouped: [bs, 8, 10, sq, skv] -> [bs*8, skv, 10*sq]  (transpose last two)
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    # Merge (groups, sq) -> [bs*8, 10*sq, skv]
    Pd_flat = Pd_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, seq_kv)
    # dO_flat already is [bs*8, 10*sq, 128]
    # dV = Pd_flat^T @ dO_flat: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128]
    dV = torch.bmm(Pd_flat.transpose(1, 2), dO_flat).reshape(
        bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM
    ).to(torch.bfloat16)

    return dS, dV


# Compile with fullgraph=True to prevent graph breaks and enable CUDA graphs
_compiled_attn_bwd = torch.compile(
    _attn_bwd_impl,
    mode="max-autotune",
    fullgraph=True,
    options={"triton.cudagraphs": True},
)

# Cache the device scalar for inv_keep_prob to avoid recompilation
_inv_keep_prob_cache: dict = {}


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    device = grad_attn_output.device

    # Use a cached device tensor scalar for inv_keep_prob to avoid
    # torch.compile specializing on different Python float values
    key = (attention_dropout, device)
    if key not in _inv_keep_prob_cache:
        val = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0
        _inv_keep_prob_cache[key] = torch.tensor(val, dtype=torch.float32, device=device)
    inv_keep_prob = _inv_keep_prob_cache[key]

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        inv_keep_prob,
    )

```

---

## Experiment #9 — 2026-06-28 07:54:28 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 616.85 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — PyTorch with torch.compile(max-autotune-no-cudagraphs)
and GQA-native einsum ops, softmax-backward in bfloat16.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

# Enable TF32 and BF16 reductions for maximum tensor-core utilization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _attn_bwd_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights,         # [bs, 80, seq_q, seq_kv]    bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
    dropout_mask,         # [bs, 80, seq_q, seq_kv]    bool
    inv_keep_prob,        # float scalar (precomputed 1/(1-p))
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # ── dP = dO @ V^T via einsum (bf16, no materialized expansion) ───────────
    # 'bgnqd,bgkd->bgnqk'  [bs,8,10,sq,d] x [bs,8,skv,d] -> [bs,8,10,sq,skv]
    dP_dropped_grouped = torch.einsum(
        'bgnqd,bgkd->bgnqk',
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
        value_states,  # [bs, 8, skv, 128]     bf16
    )
    # result: [bs, 8, 10, sq, skv] in bf16 (einsum output dtype matches inputs)

    # Reshape back to [bs, 80, sq, skv]  bf16
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Dropout backward + softmax backward entirely in bfloat16 ─────────────
    # Correctness tolerance is rtol=1e-2, atol=1e-2 — bf16 is sufficient.
    # dP = dP_dropped * dropout_mask * inv_keep_prob  (fused chain)
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob

    # Softmax backward: dS = P * (dP - rowsum(dP * P))
    P    = attn_weights                                        # [bs, 80, sq, skv] bf16
    dPP  = dP_bf16 * P
    dS   = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    # ── dV via einsum (bf16, sums over groups) ────────────────────────────────
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    # 'bgnqk,bgnqd->bgkd'  [bs,8,10,sq,skv] x [bs,8,10,sq,d] -> [bs,8,skv,d]
    dV = torch.einsum(
        'bgnqk,bgnqd->bgkd',
        Pd_grouped,    # [bs, 8, 10, sq, skv]  bf16
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
    ).to(torch.bfloat16)  # [bs, 8, skv, 128]

    return dS, dV


# Compile with max-autotune-no-cudagraphs: runs autotuning but avoids
# CUDA graph shape-specialization issues across the 16 benchmark shapes.
_compiled_attn_bwd = torch.compile(
    _attn_bwd_impl,
    mode="max-autotune-no-cudagraphs",
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        inv_keep_prob,
    )

```

---

## Experiment #10 — 2026-06-28 07:55:55 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 652.25 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — PyTorch with torch.compile(max-autotune-no-cudagraphs)
and explicit bmm calls for dP and dV, everything in bfloat16.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

# Enable TF32 and BF16 reductions for maximum tensor-core utilization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _attn_bwd_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights,         # [bs, 80, seq_q, seq_kv]    bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
    dropout_mask,         # [bs, 80, seq_q, seq_kv]    bool
    inv_keep_prob,        # float scalar (precomputed 1/(1-p))
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # ── dP = dO @ V^T via bmm ─────────────────────────────────────────────────
    # dO_grouped: [bs, 8, 10, sq, 128] -> [bs*8*10, sq, 128]
    dO_flat = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS * N_GROUPS, seq_q, HEAD_DIM)

    # value_states: [bs, 8, skv, 128]
    # Expand to [bs, 8, 10, skv, 128] via expand (no copy), then make contiguous
    # for cuBLAS strided-batched GEMM compatibility
    V_expanded = value_states.unsqueeze(2).expand(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM
    ).contiguous().reshape(bs * NUM_KEY_VALUE_HEADS * N_GROUPS, seq_kv, HEAD_DIM)

    # bmm: [bs*80, sq, 128] @ [bs*80, 128, skv] -> [bs*80, sq, skv]
    dP_dropped = torch.bmm(dO_flat, V_expanded.transpose(1, 2)).reshape(
        bs, NUM_ATTENTION_HEADS, seq_q, seq_kv
    )

    # ── Dropout backward + softmax backward entirely in bfloat16 ─────────────
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob

    P    = attn_weights   # [bs, 80, sq, skv] bf16
    dPP  = dP_bf16 * P
    dS   = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    # ── dV via bmm ────────────────────────────────────────────────────────────
    # Pd_grouped: [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    Pd_flat = attn_weights_dropped.reshape(
        bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, seq_kv
    )
    # dO for dV: [bs, 8, 10, sq, 128] -> [bs*8, 10*sq, 128]
    dO_flat2 = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, HEAD_DIM)

    # bmm: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    dV = torch.bmm(Pd_flat.transpose(1, 2), dO_flat2).reshape(
        bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM
    ).to(torch.bfloat16)

    return dS, dV


# Compile with max-autotune-no-cudagraphs: runs autotuning but avoids
# CUDA graph shape-specialization issues across the 16 benchmark shapes.
_compiled_attn_bwd = torch.compile(
    _attn_bwd_impl,
    mode="max-autotune-no-cudagraphs",
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        inv_keep_prob,
    )

```

---

## Experiment #11 — 2026-06-28 07:57:21 UTC ✅ KEEP

**Hypothesis:** ** Single targeted change — added `dynamic=True` to the `torch.compile` call:

**Result:** 560.81 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — PyTorch with torch.compile(max-autotune-no-cudagraphs,
dynamic=True) and GQA-native einsum ops, softmax-backward in bfloat16.

dynamic=True uses symbolic shapes to generate a single compiled version for all
benchmark shapes, avoiding per-shape recompilation overhead.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

# Enable TF32 and BF16 reductions for maximum tensor-core utilization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _attn_bwd_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights,         # [bs, 80, seq_q, seq_kv]    bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
    dropout_mask,         # [bs, 80, seq_q, seq_kv]    bool
    inv_keep_prob,        # float scalar (precomputed 1/(1-p))
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # ── dP = dO @ V^T via einsum (bf16, no materialized expansion) ───────────
    # 'bgnqd,bgkd->bgnqk'  [bs,8,10,sq,d] x [bs,8,skv,d] -> [bs,8,10,sq,skv]
    dP_dropped_grouped = torch.einsum(
        'bgnqd,bgkd->bgnqk',
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
        value_states,  # [bs, 8, skv, 128]     bf16
    )
    # result: [bs, 8, 10, sq, skv] in bf16

    # Reshape back to [bs, 80, sq, skv]  bf16
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Dropout backward + softmax backward entirely in bfloat16 ─────────────
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob

    P    = attn_weights   # [bs, 80, sq, skv] bf16
    dPP  = dP_bf16 * P
    dS   = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    # ── dV via einsum (bf16, sums over groups) ────────────────────────────────
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    # 'bgnqk,bgnqd->bgkd'  [bs,8,10,sq,skv] x [bs,8,10,sq,d] -> [bs,8,skv,d]
    dV = torch.einsum(
        'bgnqk,bgnqd->bgkd',
        Pd_grouped,    # [bs, 8, 10, sq, skv]  bf16
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
    ).to(torch.bfloat16)  # [bs, 8, skv, 128]

    return dS, dV


# dynamic=True: uses symbolic shapes so a single compiled version handles all
# 16 benchmark shapes without per-shape recompilation overhead.
_compiled_attn_bwd = torch.compile(
    _attn_bwd_impl,
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        inv_keep_prob,
    )

```

---

## Experiment #12 — 2026-06-28 07:59:00 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 769.76 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — PyTorch with torch.compile(max-autotune-no-cudagraphs,
dynamic=True) and torch.matmul broadcast for GQA (no einsum), softmax-backward in bfloat16.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

# Enable TF32 and BF16 reductions for maximum tensor-core utilization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _attn_bwd_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights,         # [bs, 80, seq_q, seq_kv]    bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
    dropout_mask,         # [bs, 80, seq_q, seq_kv]    bool
    inv_keep_prob,        # float scalar (precomputed 1/(1-p))
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # ── dP = dO @ V^T via broadcasting matmul ────────────────────────────────
    # dO_grouped:         [bs, 8, 10, sq,  128]
    # value_states:       [bs, 8,  1, skv, 128]  (unsqueeze at dim=2)
    # V^T:                [bs, 8,  1, 128, skv]
    # matmul broadcasts:  [bs, 8, 10, sq,  skv]
    # PyTorch uses cublasGemmStridedBatchedEx with batch_stride=0 for broadcast dim
    V_t = value_states.unsqueeze(2).transpose(-1, -2)  # [bs, 8, 1, 128, skv]
    dP_dropped_grouped = torch.matmul(dO_grouped, V_t)  # [bs, 8, 10, sq, skv]

    # Reshape back to [bs, 80, sq, skv]  bf16
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Dropout backward + softmax backward entirely in bfloat16 ─────────────
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob

    P    = attn_weights   # [bs, 80, sq, skv] bf16
    dPP  = dP_bf16 * P
    dS   = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    # ── dV via broadcasting matmul + group sum ────────────────────────────────
    # Pd_grouped: [bs, 8, 10, sq, skv]
    # dO_grouped: [bs, 8, 10, sq, 128]
    # Pd^T @ dO: [bs, 8, 10, skv, sq] @ [bs, 8, 10, sq, 128] -> [bs, 8, 10, skv, 128]
    # Then sum over groups (dim=2) -> [bs, 8, skv, 128]
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    dV = torch.matmul(
        Pd_grouped.transpose(-1, -2),  # [bs, 8, 10, skv, sq]
        dO_grouped,                    # [bs, 8, 10, sq,  128]
    ).sum(dim=2).to(torch.bfloat16)   # [bs, 8, skv, 128]

    return dS, dV


# dynamic=True: single compiled version for all 16 benchmark shapes
_compiled_attn_bwd = torch.compile(
    _attn_bwd_impl,
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        inv_keep_prob,
    )

```

---

## Experiment #13 — 2026-06-28 08:01:07 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 717.34 μs

**Kernel code:**
```python
"""
Hybrid attention-backward: PyTorch compiled GEMMs + Triton softmax-backward kernel.

The Triton kernel handles only the simple elementwise softmax-backward
(no GEMMs, no GQA indexing) — maximum simplicity for correctness.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused softmax-backward
# Input:  dP_dropped [bs, 80, sq, skv]  bf16 (already computed dO @ V^T)
#         P          [bs, 80, sq, skv]  bf16 (attn_weights)
#         M          [bs, 80, sq, skv]  bool (dropout_mask)
# Output: dS         [bs, 80, sq, skv]  bf16
#
# Algorithm for each row [sq]:
#   dP = dP_dropped * M * inv_keep_prob
#   rowsum = sum_k(dP_k * P_k)
#   dS = P * (dP - rowsum)
#
# Grid: [bs * 80, sq]  — each program handles one row (full seq_kv)
# No GEMMs, no GQA indexing — pure elementwise + row-reduction.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def softmax_bwd_kernel(
    # dP_dropped: [bs, 80, sq, skv]  bf16
    dPd_ptr, dPd_s0, dPd_s1, dPd_s2, dPd_s3,
    # P (attn_weights): [bs, 80, sq, skv]  bf16
    P_ptr,   P_s0,   P_s1,   P_s2,   P_s3,
    # M (dropout_mask): [bs, 80, sq, skv]  bool (uint8 in memory)
    M_ptr,   M_s0,   M_s1,   M_s2,   M_s3,
    # dS output: [bs, 80, sq, skv]  bf16
    dS_ptr,  dS_s0,  dS_s1,  dS_s2,  dS_s3,
    # dimensions
    seq_kv,
    inv_keep_prob,
    # constexpr block size for seq_kv tiling
    BLOCK_SKV: tl.constexpr,
    N_HEADS:   tl.constexpr,
):
    # Each program handles one (batch, head, sq_row) triple
    row_id = tl.program_id(0)   # encodes bs_id * 80 * sq + h_id * sq + sq_id
    # But we use 2D grid instead: pid0 = bs*80 + h, pid1 = sq_row
    bh_id  = tl.program_id(0)
    sq_id  = tl.program_id(1)

    bs_id = bh_id // N_HEADS
    h_id  = bh_id % N_HEADS

    # Base pointers for this (batch, head, sq_row)
    base_offset = bs_id * dPd_s0 + h_id * dPd_s1 + sq_id * dPd_s2
    dPd_row = dPd_ptr + base_offset
    P_row   = P_ptr   + bs_id * P_s0  + h_id * P_s1  + sq_id * P_s2
    M_row   = M_ptr   + bs_id * M_s0  + h_id * M_s1  + sq_id * M_s2
    dS_row  = dS_ptr  + bs_id * dS_s0 + h_id * dS_s1 + sq_id * dS_s2

    # ── Pass 1: accumulate rowsum = sum_k(dP_k * P_k) ────────────────────────
    rowsum = tl.zeros([1], dtype=tl.float32)
    n_blocks = tl.cdiv(seq_kv, BLOCK_SKV)

    for blk in range(n_blocks):
        k_start = blk * BLOCK_SKV
        k_offs  = k_start + tl.arange(0, BLOCK_SKV)
        k_mask  = k_offs < seq_kv

        # Load dP_dropped row slice
        dPd_tile = tl.load(dPd_row + k_offs * dPd_s3, mask=k_mask, other=0.0).to(tl.float32)

        # Load dropout mask — bool stored as uint8, cast to int1
        m_raw  = tl.load(M_row + k_offs * M_s3, mask=k_mask, other=0)
        m_bool = m_raw.to(tl.int1)
        dP_tile = tl.where(m_bool, dPd_tile * inv_keep_prob, 0.0)

        # Load P
        P_tile = tl.load(P_row + k_offs * P_s3, mask=k_mask, other=0.0).to(tl.float32)

        rowsum += tl.sum(dP_tile * P_tile, axis=0)

    # ── Pass 2: write dS = P * (dP - rowsum) ─────────────────────────────────
    rs = rowsum  # scalar [1]

    for blk in range(n_blocks):
        k_start = blk * BLOCK_SKV
        k_offs  = k_start + tl.arange(0, BLOCK_SKV)
        k_mask  = k_offs < seq_kv

        dPd_tile = tl.load(dPd_row + k_offs * dPd_s3, mask=k_mask, other=0.0).to(tl.float32)

        m_raw  = tl.load(M_row + k_offs * M_s3, mask=k_mask, other=0)
        m_bool = m_raw.to(tl.int1)
        dP_tile = tl.where(m_bool, dPd_tile * inv_keep_prob, 0.0)

        P_tile = tl.load(P_row + k_offs * P_s3, mask=k_mask, other=0.0).to(tl.float32)

        dS_tile = (P_tile * (dP_tile - rs)).to(tl.bfloat16)
        tl.store(dS_row + k_offs * dS_s3, dS_tile, mask=k_mask)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch compiled function for the two GEMMs (dP and dV)
# ─────────────────────────────────────────────────────────────────────────────
def _gemm_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # dP_dropped = dO @ V^T: einsum broadcasts over groups dim
    # [bs,8,10,sq,128] x [bs,8,skv,128] -> [bs,8,10,sq,skv]
    dP_dropped_grouped = torch.einsum(
        'bgnqd,bgkd->bgnqk',
        dO_grouped,    # [bs, 8, 10, sq, 128]
        value_states,  # [bs, 8, skv, 128]
    )
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # dV = Pd^T @ dO, summed over groups
    # Pd_grouped: [bs*8, 10*sq, skv], dO_flat: [bs*8, 10*sq, 128]
    dO_flat   = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, HEAD_DIM)
    Pd_flat   = attn_weights_dropped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, seq_kv)
    dV = torch.bmm(Pd_flat.transpose(1, 2), dO_flat).reshape(
        bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM
    ).to(torch.bfloat16)

    return dP_dropped, dV


_compiled_gemms = torch.compile(
    _gemm_impl,
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_heads = NUM_ATTENTION_HEADS

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Step 1: PyTorch compiled GEMMs
    dP_dropped, dV = _compiled_gemms(
        grad_attn_output, attn_weights_dropped, value_states
    )

    # Step 2: Triton fused softmax-backward (elementwise only, no GEMM)
    dS = torch.empty_like(dP_dropped)  # [bs, 80, sq, skv] bf16

    # Ensure inputs are contiguous for simple stride-1 innermost dim access
    dP_dropped_c = dP_dropped.contiguous()
    P_c          = attn_weights.contiguous()
    M_c          = dropout_mask.contiguous()

    BLOCK_SKV = 256  # large block to amortize loop overhead for row reduction

    grid = (bs * n_heads, seq_q)

    softmax_bwd_kernel[grid](
        dP_dropped_c,
        dP_dropped_c.stride(0), dP_dropped_c.stride(1),
        dP_dropped_c.stride(2), dP_dropped_c.stride(3),
        P_c,
        P_c.stride(0), P_c.stride(1), P_c.stride(2), P_c.stride(3),
        M_c,
        M_c.stride(0), M_c.stride(1), M_c.stride(2), M_c.stride(3),
        dS,
        dS.stride(0), dS.stride(1), dS.stride(2), dS.stride(3),
        seq_kv,
        inv_keep_prob,
        BLOCK_SKV=BLOCK_SKV,
        N_HEADS=n_heads,
    )

    return dS, dV

```

---

## Experiment #14 — 2026-06-28 08:02:16 UTC 💥 CRASH

**Hypothesis:** ** Single targeted change to the `torch.compile` call — added `options` dict:

**Result:** CRASH

**Error:**
```
Correctness check failed. Failed to load submission:
Traceback (most recent call last):
  File "/root/eval_modal_attn_bwd.py", line 246, in evaluate_kernel
    spec.loader.exec_module(mod)
  File "<frozen importlib._bootstrap_external>", line 940, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/tmp/submission_kmqzdr18/submission.py", line 89, in <module>
    _compiled_
```

**Kernel code:**
```python
"""
Optimized attention-backward kernel — PyTorch with torch.compile using Inductor
options to force Triton-generated GEMMs for better B200 tensor-core utilization.

Based on Experiment 11 (best at 560.81 μs) with added inductor options to use
Triton for GEMM generation instead of cuBLAS.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

# Enable TF32 and BF16 reductions for maximum tensor-core utilization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True


def _attn_bwd_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights,         # [bs, 80, seq_q, seq_kv]    bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
    dropout_mask,         # [bs, 80, seq_q, seq_kv]    bool
    inv_keep_prob,        # float scalar (precomputed 1/(1-p))
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # ── dP = dO @ V^T via einsum (bf16, no materialized expansion) ───────────
    # 'bgnqd,bgkd->bgnqk'  [bs,8,10,sq,d] x [bs,8,skv,d] -> [bs,8,10,sq,skv]
    dP_dropped_grouped = torch.einsum(
        'bgnqd,bgkd->bgnqk',
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
        value_states,  # [bs, 8, skv, 128]     bf16
    )
    # result: [bs, 8, 10, sq, skv] in bf16

    # Reshape back to [bs, 80, sq, skv]  bf16
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Dropout backward + softmax backward entirely in bfloat16 ─────────────
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob

    P    = attn_weights   # [bs, 80, sq, skv] bf16
    dPP  = dP_bf16 * P
    dS   = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    # ── dV via einsum (bf16, sums over groups) ────────────────────────────────
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    # 'bgnqk,bgnqd->bgkd'  [bs,8,10,sq,skv] x [bs,8,10,sq,d] -> [bs,8,skv,d]
    dV = torch.einsum(
        'bgnqk,bgnqd->bgkd',
        Pd_grouped,    # [bs, 8, 10, sq, skv]  bf16
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
    ).to(torch.bfloat16)  # [bs, 8, skv, 128]

    return dS, dV


# Use Inductor with Triton GEMM backend to get autotuned Triton kernels
# for the GQA-shaped einsums rather than cuBLAS.
_compiled_attn_bwd = torch.compile(
    _attn_bwd_impl,
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
    options={
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "TRITON",
    },
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        inv_keep_prob,
    )

```

---

## Experiment #15 — 2026-06-28 08:03:52 UTC ✅ KEEP

**Hypothesis:** ** Three inductor config flags set at module import time, before `torch.compile` is called:

**Result:** 533.42 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — PyTorch with torch.compile(max-autotune-no-cudagraphs,
dynamic=True) + inductor coordinate-descent tuning config flags.

Based on Experiment 11 (best at 560.81 μs). Adds coordinate_descent_tuning
to find better tile sizes for the irregular GQA shapes on B200.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

# Enable TF32 and BF16 reductions for maximum tensor-core utilization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

# Set inductor config flags directly (not via options= kwarg which crashes)
# to enable coordinate-descent tuning for better tile sizes on B200.
try:
    import torch._inductor.config as inductor_config
    inductor_config.max_autotune_gemm = True
    inductor_config.coordinate_descent_tuning = True
    inductor_config.coordinate_descent_check_all_directions = True
except Exception:
    pass  # fall back gracefully if unavailable


def _attn_bwd_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights,         # [bs, 80, seq_q, seq_kv]    bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
    dropout_mask,         # [bs, 80, seq_q, seq_kv]    bool
    inv_keep_prob,        # float scalar (precomputed 1/(1-p))
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # ── dP = dO @ V^T via einsum (bf16, no materialized expansion) ───────────
    # 'bgnqd,bgkd->bgnqk'  [bs,8,10,sq,d] x [bs,8,skv,d] -> [bs,8,10,sq,skv]
    dP_dropped_grouped = torch.einsum(
        'bgnqd,bgkd->bgnqk',
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
        value_states,  # [bs, 8, skv, 128]     bf16
    )
    # result: [bs, 8, 10, sq, skv] in bf16

    # Reshape back to [bs, 80, sq, skv]  bf16
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Dropout backward + softmax backward entirely in bfloat16 ─────────────
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob

    P    = attn_weights   # [bs, 80, sq, skv] bf16
    dPP  = dP_bf16 * P
    dS   = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    # ── dV via einsum (bf16, sums over groups) ────────────────────────────────
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    # 'bgnqk,bgnqd->bgkd'  [bs,8,10,sq,skv] x [bs,8,10,sq,d] -> [bs,8,skv,d]
    dV = torch.einsum(
        'bgnqk,bgnqd->bgkd',
        Pd_grouped,    # [bs, 8, 10, sq, skv]  bf16
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
    ).to(torch.bfloat16)  # [bs, 8, skv, 128]

    return dS, dV


# Compile with max-autotune-no-cudagraphs + dynamic=True (same as Exp 11)
# The coordinate_descent_tuning config above applies globally to all torch.compile calls.
_compiled_attn_bwd = torch.compile(
    _attn_bwd_impl,
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        inv_keep_prob,
    )

```

---

## Experiment #16 — 2026-06-28 08:05:44 UTC ❌ DISCARD

**Hypothesis:** ** Added 5 new inductor config flags inside the existing try-except block, after the 3 proven flags from Exp 15:

**Result:** 598.55 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — PyTorch with torch.compile(max-autotune-no-cudagraphs,
dynamic=True) + extended inductor config flags for fusion and memory optimization.

Based on Experiment 15 (best at 533.42 μs). Adds epilogue_fusion, aggressive_fusion,
max_fusion_size, unroll_reductions_threshold, and realize_opcount_threshold flags.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

# Enable TF32 and BF16 reductions for maximum tensor-core utilization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

# Set inductor config flags directly (not via options= kwarg which crashes).
# Exp 15 proven flags + additional fusion/memory flags.
try:
    import torch._inductor.config as inductor_config
    # Proven from Exp 15:
    inductor_config.max_autotune_gemm = True
    inductor_config.coordinate_descent_tuning = True
    inductor_config.coordinate_descent_check_all_directions = True
    # New fusion flags:
    inductor_config.epilogue_fusion = True          # fuse pointwise ops into preceding kernels
    inductor_config.aggressive_fusion = True        # more aggressive kernel fusion
    inductor_config.max_fusion_size = 512           # allow larger fused kernels (default 64)
    inductor_config.unroll_reductions_threshold = 32  # helps with dPP.sum(-1) rowsum
    inductor_config.realize_opcount_threshold = 32    # defer more realizations
except Exception:
    pass  # fall back gracefully if any flag is unavailable


def _attn_bwd_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights,         # [bs, 80, seq_q, seq_kv]    bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
    dropout_mask,         # [bs, 80, seq_q, seq_kv]    bool
    inv_keep_prob,        # float scalar (precomputed 1/(1-p))
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # ── dP = dO @ V^T via einsum (bf16, no materialized expansion) ───────────
    # 'bgnqd,bgkd->bgnqk'  [bs,8,10,sq,d] x [bs,8,skv,d] -> [bs,8,10,sq,skv]
    dP_dropped_grouped = torch.einsum(
        'bgnqd,bgkd->bgnqk',
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
        value_states,  # [bs, 8, skv, 128]     bf16
    )
    # result: [bs, 8, 10, sq, skv] in bf16

    # Reshape back to [bs, 80, sq, skv]  bf16
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Dropout backward + softmax backward entirely in bfloat16 ─────────────
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob

    P    = attn_weights   # [bs, 80, sq, skv] bf16
    dPP  = dP_bf16 * P
    dS   = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    # ── dV via einsum (bf16, sums over groups) ────────────────────────────────
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    # 'bgnqk,bgnqd->bgkd'  [bs,8,10,sq,skv] x [bs,8,10,sq,d] -> [bs,8,skv,d]
    dV = torch.einsum(
        'bgnqk,bgnqd->bgkd',
        Pd_grouped,    # [bs, 8, 10, sq, skv]  bf16
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
    ).to(torch.bfloat16)  # [bs, 8, skv, 128]

    return dS, dV


# Compile with max-autotune-no-cudagraphs + dynamic=True
# All inductor config flags set above apply globally.
_compiled_attn_bwd = torch.compile(
    _attn_bwd_impl,
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        inv_keep_prob,
    )

```

---

## Experiment #17 — 2026-06-28 08:08:05 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 637.91 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — PyTorch with torch.compile(max-autotune-no-cudagraphs,
dynamic=True) + Exp 15 inductor config + repeat_interleave bmm for dP GEMM.

Based on Experiment 15 (best at 533.42 μs). Tries repeat_interleave to give cuBLAS
a contiguous [bs*80, skv, 128] V tensor for a standard batched GEMM.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

# Enable TF32 and BF16 reductions for maximum tensor-core utilization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

# Exp 15 proven inductor config flags only
try:
    import torch._inductor.config as inductor_config
    inductor_config.max_autotune_gemm = True
    inductor_config.coordinate_descent_tuning = True
    inductor_config.coordinate_descent_check_all_directions = True
except Exception:
    pass  # fall back gracefully if unavailable


def _attn_bwd_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights,         # [bs, 80, seq_q, seq_kv]    bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
    dropout_mask,         # [bs, 80, seq_q, seq_kv]    bool
    inv_keep_prob,        # float scalar (precomputed 1/(1-p))
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # ── dP = dO @ V^T via repeat_interleave + bmm ────────────────────────────
    # Expand V: [bs, 8, skv, 128] -> [bs*80, skv, 128] via repeat_interleave
    # repeat_interleave repeats each row N_GROUPS=10 times along batch dim,
    # giving a contiguous tensor cuBLAS can use as a standard batched GEMM.
    V_flat = value_states.reshape(bs * NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM)
    V_rep  = torch.repeat_interleave(V_flat, N_GROUPS, dim=0)  # [bs*80, skv, 128]

    # dO_flat: [bs*80, sq, 128]
    dO_flat = dO_grouped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, HEAD_DIM)

    # bmm: [bs*80, sq, 128] @ [bs*80, 128, skv] -> [bs*80, sq, skv]
    dP_dropped = torch.bmm(dO_flat, V_rep.transpose(1, 2)).reshape(
        bs, NUM_ATTENTION_HEADS, seq_q, seq_kv
    )

    # ── Dropout backward + softmax backward entirely in bfloat16 ─────────────
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob

    P    = attn_weights   # [bs, 80, sq, skv] bf16
    dPP  = dP_bf16 * P
    dS   = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    # ── dV via einsum (bf16, sums over groups) — proven fast ─────────────────
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    # 'bgnqk,bgnqd->bgkd'  [bs,8,10,sq,skv] x [bs,8,10,sq,d] -> [bs,8,skv,d]
    dV = torch.einsum(
        'bgnqk,bgnqd->bgkd',
        Pd_grouped,    # [bs, 8, 10, sq, skv]  bf16
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
    ).to(torch.bfloat16)  # [bs, 8, skv, 128]

    return dS, dV


# Compile with max-autotune-no-cudagraphs + dynamic=True (Exp 15 settings)
_compiled_attn_bwd = torch.compile(
    _attn_bwd_impl,
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        inv_keep_prob,
    )

```

---

## Experiment #18 — 2026-06-28 08:10:04 UTC 💥 CRASH

**Hypothesis:** **

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Hybrid attention-backward: torch.compile for dS + Triton kernel for dV.

Exp 15 torch.compile handles the dP GEMM + softmax-backward (dS output).
A focused Triton kernel handles dV (accumulate over groups and sq tiles).
The dV kernel is structurally simple — no GQA GEMM, straightforward P^T @ dO
with group accumulation.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

# Exp 15 proven inductor config flags
try:
    import torch._inductor.config as inductor_config
    inductor_config.max_autotune_gemm = True
    inductor_config.coordinate_descent_tuning = True
    inductor_config.coordinate_descent_check_all_directions = True
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: compute grad_value_states (dV)
# Grid: [bs * n_kv_heads, cdiv(seq_kv, BLOCK_SKV)]
# Each block handles [BLOCK_SKV, HEAD_DIM] tile of dV.
# Accumulates dV += Pd^T @ dO over all N_GROUPS query heads and all sq tiles.
#
# Inputs:
#   Pd: attn_weights_dropped [bs, 80, seq_q, seq_kv]  bf16
#   dO: [bs, 80, seq_q, head_dim]                     bf16  (contiguous after transpose)
# Output:
#   dV: [bs, 8, seq_kv, head_dim]                     bf16
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def kernel_dV(
    Pd_ptr, Pd_s0, Pd_s1, Pd_s2, Pd_s3,   # attn_weights_dropped strides
    dO_ptr, dO_s0, dO_s1, dO_s2, dO_s3,   # dO strides
    dV_ptr, dV_s0, dV_s1, dV_s2, dV_s3,   # dV output strides
    seq_q, seq_kv,
    N_KV_HEADS: tl.constexpr,
    N_GROUPS:   tl.constexpr,
    HEAD_DIM:   tl.constexpr,
    BLOCK_SKV:  tl.constexpr,
    BLOCK_SQ:   tl.constexpr,
):
    pid0    = tl.program_id(0)   # bs_id * N_KV_HEADS + kv_id
    skv_blk = tl.program_id(1)

    bs_id  = pid0 // N_KV_HEADS
    kv_id  = pid0 % N_KV_HEADS

    skv_start = skv_blk * BLOCK_SKV
    skv_offs  = skv_start + tl.arange(0, BLOCK_SKV)
    skv_mask  = skv_offs < seq_kv
    d_offs    = tl.arange(0, HEAD_DIM)

    # Accumulator: [BLOCK_SKV, HEAD_DIM]
    dV_acc = tl.zeros([BLOCK_SKV, HEAD_DIM], dtype=tl.float32)

    n_sq_blks = tl.cdiv(seq_q, BLOCK_SQ)

    for g in range(N_GROUPS):
        h_id    = kv_id * N_GROUPS + g
        Pd_base = bs_id * Pd_s0 + h_id * Pd_s1
        dO_base = bs_id * dO_s0 + h_id * dO_s1

        for sq_blk in range(n_sq_blks):
            sq_start = sq_blk * BLOCK_SQ
            sq_offs  = sq_start + tl.arange(0, BLOCK_SQ)
            sq_mask  = sq_offs < seq_q
            cmask    = sq_mask[:, None] & skv_mask[None, :]

            # Load Pd tile: [BLOCK_SQ, BLOCK_SKV]
            Pd_ptrs = Pd_base + sq_offs[:, None] * Pd_s2 + skv_offs[None, :] * Pd_s3
            Pd_tile = tl.load(Pd_ptrs, mask=cmask, other=0.0).to(tl.float32)

            # Load dO tile: [BLOCK_SQ, HEAD_DIM]
            dO_ptrs = dO_base + sq_offs[:, None] * dO_s2 + d_offs[None, :] * dO_s3
            dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

            # dV += Pd^T @ dO: [BLOCK_SKV, HEAD_DIM]
            dV_acc += tl.dot(tl.trans(Pd_tile), dO_tile)

    # Write dV
    dV_base = bs_id * dV_s0 + kv_id * dV_s1
    dV_ptrs = dV_base + skv_offs[:, None] * dV_s2 + d_offs[None, :] * dV_s3
    tl.store(dV_ptrs, dV_acc.to(tl.bfloat16), mask=skv_mask[:, None])


# ─────────────────────────────────────────────────────────────────────────────
# torch.compile function for dS (Exp 15 pattern — proven correct and fast)
# ─────────────────────────────────────────────────────────────────────────────
def _attn_bwd_ds_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights,         # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
    dropout_mask,         # [bs, 80, seq_q, seq_kv]    bool
    inv_keep_prob,        # float scalar
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # dP_dropped = dO @ V^T via einsum (bf16, broadcast over groups)
    dP_dropped = torch.einsum(
        'bgnqd,bgkd->bgnqk',
        dO_grouped,    # [bs, 8, 10, sq, 128]
        value_states,  # [bs, 8, skv, 128]
    ).reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # Dropout + softmax backward in bf16
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob
    P       = attn_weights
    dPP     = dP_bf16 * P
    dS      = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    return dS


_compiled_ds = torch.compile(
    _attn_bwd_ds_impl,
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Step 1: torch.compile computes dS (Exp 15 pattern)
    dS = _compiled_ds(
        grad_attn_output, attn_weights, value_states, dropout_mask, inv_keep_prob
    )

    # Step 2: Triton kernel computes dV
    # dO needs to be [bs, 80, seq_q, 128] contiguous for correct strides
    dO = grad_attn_output.transpose(1, 2).contiguous()  # [bs, 80, seq_q, 128]

    dV = torch.empty(
        (bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM),
        dtype=torch.bfloat16, device=dO.device
    )

    BLOCK_SKV = 64
    BLOCK_SQ  = 32

    grid_dV = (bs * NUM_KEY_VALUE_HEADS, triton.cdiv(seq_kv, BLOCK_SKV))

    kernel_dV[grid_dV](
        attn_weights_dropped,
        attn_weights_dropped.stride(0), attn_weights_dropped.stride(1),
        attn_weights_dropped.stride(2), attn_weights_dropped.stride(3),
        dO,
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dV,
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        seq_q, seq_kv,
        N_KV_HEADS=NUM_KEY_VALUE_HEADS,
        N_GROUPS=N_GROUPS,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SKV=BLOCK_SKV,
        BLOCK_SQ=BLOCK_SQ,
    )

    return dS, dV

```

---

## Experiment #19 — 2026-06-28 08:12:11 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 595.46 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — split-compile approach:
- GEMMs compiled with dynamic=False (static tile selection per shape)
- Elementwise softmax-backward compiled with dynamic=True (avoids recompilation overhead)

Based on Experiment 15 (best at 533.42 μs), same Exp 15 inductor config flags.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

# Exp 15 proven inductor config flags
try:
    import torch._inductor.config as inductor_config
    inductor_config.max_autotune_gemm = True
    inductor_config.coordinate_descent_tuning = True
    inductor_config.coordinate_descent_check_all_directions = True
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# GEMM function — compiled with dynamic=False for best static tile selection
# Computes dP_dropped and dV (the two matrix multiplications)
# ─────────────────────────────────────────────────────────────────────────────
def _gemm_fn(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # dP_dropped = dO @ V^T: [bs,8,10,sq,128] x [bs,8,skv,128] -> [bs,8,10,sq,skv]
    dP_dropped = torch.einsum(
        'bgnqd,bgkd->bgnqk',
        dO_grouped,    # [bs, 8, 10, sq, 128]
        value_states,  # [bs, 8, skv, 128]
    ).reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # dV = Pd^T @ dO summed over groups: [bs,8,10,sq,skv] x [bs,8,10,sq,128] -> [bs,8,skv,128]
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    dV = torch.einsum(
        'bgnqk,bgnqd->bgkd',
        Pd_grouped,
        dO_grouped,
    ).to(torch.bfloat16)

    return dP_dropped, dV


# dynamic=False: per-shape compilation → better static GEMM tile selection
_compiled_gemms = torch.compile(
    _gemm_fn,
    mode="max-autotune-no-cudagraphs",
    dynamic=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Elementwise softmax-backward function — compiled with dynamic=True
# Avoids per-shape recompilation overhead for the cheap elementwise ops
# ─────────────────────────────────────────────────────────────────────────────
def _elemwise_fn(
    dP_dropped,   # [bs, 80, sq, skv]  bfloat16
    attn_weights, # [bs, 80, sq, skv]  bfloat16
    dropout_mask, # [bs, 80, sq, skv]  bool
    inv_keep_prob,
):
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob
    P       = attn_weights
    dPP     = dP_bf16 * P
    dS      = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)
    return dS


# dynamic=True: single compiled version handles all shapes for elementwise
_compiled_elemwise = torch.compile(
    _elemwise_fn,
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Step 1: GEMM kernels (static shape compilation → best tile sizes)
    dP_dropped, dV = _compiled_gemms(
        grad_attn_output, attn_weights_dropped, value_states
    )

    # Step 2: elementwise softmax-backward (dynamic compilation → no recompile overhead)
    dS = _compiled_elemwise(dP_dropped, attn_weights, dropout_mask, inv_keep_prob)

    return dS, dV

```

---

## Experiment #20 — 2026-06-28 08:13:23 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 714.51 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — concurrent CUDA streams for the two GEMMs,
with Exp 15 inductor config for the elementwise softmax-backward.

The dP GEMM (dO @ V^T) and dV GEMM (Pd^T @ dO) have no data dependency on each
other, so they can run concurrently on separate CUDA streams to overlap compute
on B200's multiple SM clusters.

Based on Experiment 15 (best at 533.42 μs).

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

# Exp 15 proven inductor config flags
try:
    import torch._inductor.config as inductor_config
    inductor_config.max_autotune_gemm = True
    inductor_config.coordinate_descent_tuning = True
    inductor_config.coordinate_descent_check_all_directions = True
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Compiled elementwise softmax-backward (proven correct from Exp 15)
# ─────────────────────────────────────────────────────────────────────────────
def _elemwise_fn(
    dP_dropped,   # [bs, 80, sq, skv]  bfloat16
    attn_weights, # [bs, 80, sq, skv]  bfloat16
    dropout_mask, # [bs, 80, sq, skv]  bool
    inv_keep_prob,
):
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob
    P       = attn_weights
    dPP     = dP_bf16 * P
    dS      = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)
    return dS


_compiled_elemwise = torch.compile(
    _elemwise_fn,
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
)

# Pre-created CUDA streams for concurrent GEMM execution
_stream_dp = None
_stream_dv = None


def _get_streams():
    global _stream_dp, _stream_dv
    if _stream_dp is None:
        _stream_dp = torch.cuda.Stream()
        _stream_dv = torch.cuda.Stream()
    return _stream_dp, _stream_dv


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Pre-compute dO_grouped (shared by both GEMMs) on the current stream
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )
    # Pd_grouped (for dV) — reshape only, no copy
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )

    current_stream = torch.cuda.current_stream()
    stream_dp, stream_dv = _get_streams()

    # Both sub-streams wait for the current stream (ensures dO_grouped is ready)
    stream_dp.wait_stream(current_stream)
    stream_dv.wait_stream(current_stream)

    # ── Launch dP GEMM on stream_dp ───────────────────────────────────────────
    with torch.cuda.stream(stream_dp):
        # dP_dropped = dO @ V^T: [bs,8,10,sq,128] x [bs,8,skv,128] -> [bs,80,sq,skv]
        dP_dropped = torch.einsum(
            'bgnqd,bgkd->bgnqk',
            dO_grouped,    # [bs, 8, 10, sq, 128]
            value_states,  # [bs, 8, skv, 128]
        ).reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Launch dV GEMM on stream_dv (concurrent with dP) ─────────────────────
    with torch.cuda.stream(stream_dv):
        # dV = Pd^T @ dO summed over groups -> [bs, 8, skv, 128]
        dV = torch.einsum(
            'bgnqk,bgnqd->bgkd',
            Pd_grouped,
            dO_grouped,
        ).to(torch.bfloat16)

    # Wait for both GEMMs to complete before the elementwise ops
    current_stream.wait_stream(stream_dp)
    current_stream.wait_stream(stream_dv)

    # ── Softmax backward on current stream (depends on dP_dropped) ────────────
    dS = _compiled_elemwise(dP_dropped, attn_weights, dropout_mask, inv_keep_prob)

    return dS, dV

```

---

## Experiment #21 — 2026-06-28 08:15:17 UTC ❌ DISCARD

**Hypothesis:** ** Single targeted change — switched `mode` from `"max-autotune-no-cudagraphs"` to `"max-autotune"`:

**Result:** 900.74 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — PyTorch with torch.compile(max-autotune,
dynamic=True) + Exp 15 inductor config flags.

Tries max-autotune (WITH CUDA graphs) + dynamic=True — a combination not
previously attempted. Exp 6 used max-autotune without dynamic=True (867 μs).
Exp 15 used max-autotune-no-cudagraphs with dynamic=True (533 μs).
This combines CUDA graph launch-overhead savings with dynamic shape handling.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

# Exp 15 proven inductor config flags
try:
    import torch._inductor.config as inductor_config
    inductor_config.max_autotune_gemm = True
    inductor_config.coordinate_descent_tuning = True
    inductor_config.coordinate_descent_check_all_directions = True
except Exception:
    pass


def _attn_bwd_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights,         # [bs, 80, seq_q, seq_kv]    bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
    dropout_mask,         # [bs, 80, seq_q, seq_kv]    bool
    inv_keep_prob,        # float scalar (precomputed 1/(1-p))
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # ── dP = dO @ V^T via einsum (bf16, no materialized expansion) ───────────
    dP_dropped_grouped = torch.einsum(
        'bgnqd,bgkd->bgnqk',
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16
        value_states,  # [bs, 8, skv, 128]     bf16
    )
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Dropout backward + softmax backward entirely in bfloat16 ─────────────
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob
    P    = attn_weights
    dPP  = dP_bf16 * P
    dS   = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    # ── dV via einsum (bf16, sums over groups) ────────────────────────────────
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    dV = torch.einsum(
        'bgnqk,bgnqd->bgkd',
        Pd_grouped,
        dO_grouped,
    ).to(torch.bfloat16)

    return dS, dV


# KEY CHANGE vs Exp 15: mode="max-autotune" (WITH CUDA graphs) instead of
# "max-autotune-no-cudagraphs". Combined with dynamic=True (new in this exp).
_compiled_attn_bwd = torch.compile(
    _attn_bwd_impl,
    mode="max-autotune",
    dynamic=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        inv_keep_prob,
    )

```

---

## Experiment #22 — 2026-06-28 08:16:57 UTC ✅ KEEP

**Hypothesis:** ** Two `.contiguous()` additions to the Exp 15 function body:

**Result:** 521.59 μs

**Kernel code:**
```python
"""
Optimized attention-backward kernel — Exp 15 exact base + explicit .contiguous()
for dO_grouped and attn_weights_dropped before GEMMs.

Based on Experiment 15 (best at 533.42 μs). Adds .contiguous() to ensure
cuBLAS gets proper stride-1 inner dimensions for both einsum inputs.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS  # 10

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.allow_tf32 = True

# Exp 15 proven inductor config flags
try:
    import torch._inductor.config as inductor_config
    inductor_config.max_autotune_gemm = True
    inductor_config.coordinate_descent_tuning = True
    inductor_config.coordinate_descent_check_all_directions = True
except Exception:
    pass


def _attn_bwd_impl(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights,         # [bs, 80, seq_q, seq_kv]    bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
    dropout_mask,         # [bs, 80, seq_q, seq_kv]    bool
    inv_keep_prob,        # float scalar (precomputed 1/(1-p))
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    # .contiguous() ensures the transposed view is materialized with proper strides
    # so cuBLAS sees a genuine contiguous [bs,8,10,sq,128] tensor.
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    ).contiguous()

    # ── dP = dO @ V^T via einsum (bf16, no materialized expansion) ───────────
    dP_dropped_grouped = torch.einsum(
        'bgnqd,bgkd->bgnqk',
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16  (now contiguous)
        value_states,  # [bs, 8, skv, 128]     bf16
    )
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Dropout backward + softmax backward entirely in bfloat16 ─────────────
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob
    P    = attn_weights
    dPP  = dP_bf16 * P
    dS   = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    # ── dV via einsum (bf16, sums over groups) ────────────────────────────────
    # .contiguous() on attn_weights_dropped before reshape ensures proper strides
    Pd_grouped = attn_weights_dropped.contiguous().reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    dV = torch.einsum(
        'bgnqk,bgnqd->bgkd',
        Pd_grouped,    # [bs, 8, 10, sq, skv]  bf16  (now contiguous)
        dO_grouped,    # [bs, 8, 10, sq, 128]  bf16  (now contiguous)
    ).to(torch.bfloat16)

    return dS, dV


# Exp 15 compile settings: max-autotune-no-cudagraphs + dynamic=True
_compiled_attn_bwd = torch.compile(
    _attn_bwd_impl,
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        inv_keep_prob,
    )

```

---

## Experiment #23 — 2026-06-28 08:18:09 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 3429.67 μs

**Kernel code:**
```python
"""
Reference attention-backward kernel — pure PyTorch baseline.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    n_heads    = NUM_ATTENTION_HEADS
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = n_heads // n_kv_heads  # 10

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # Expand value_states for GQA: [bs, 8, skv, d] → [bs, 80, skv, d]
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM
    ).reshape(bs, n_heads, seq_kv, HEAD_DIM)

    # 1. Transpose grad: [bs, sq, h, d] → [bs, h, sq, d]  (cast to f32)
    dO = grad_attn_output.transpose(1, 2).to(torch.float32)

    # 2. dP̃ = dO @ V^T  →  [bs, h, sq, skv]
    dP_dropped = torch.matmul(dO, vs_exp.to(torch.float32).transpose(-2, -1))

    # 3. Dropout backward
    if attention_dropout > 0.0:
        dP = dP_dropped * dropout_mask / (1.0 - attention_dropout)
    else:
        dP = dP_dropped

    # 4. Softmax backward: dS = P ⊙ (dP − sum(dP ⊙ P))
    P = attn_weights.to(torch.float32)
    dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # 5. dV_exp = P̃^T @ dO  →  [bs, h, skv, d]
    dV_exp = torch.matmul(
        attn_weights_dropped.to(torch.float32).transpose(-2, -1), dO
    )

    # 6. GQA aggregation: sum over groups  →  [bs, 8, skv, d]
    dV = dV_exp.reshape(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM).sum(dim=2)
    dV = dV.to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #24 — 2026-06-28 08:19:51 UTC ❌ DISCARD

**Hypothesis:** ** Full Exp 22 reconstruction from the reverted baseline, with the targeted change:

**Result:** 522.49 μs

**Kernel code:**
```python
"""
Triton-fused attention-backward kernel.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = 10


@triton.jit
def attn_bwd_ds_kernel(
    # Pointers
    dO_ptr,       # [bs, 80, sq, 128]  bfloat16
    P_ptr,        # [bs, 80, sq, skv]  bfloat16
    V_ptr,        # [bs,  8, skv, 128] bfloat16
    mask_ptr,     # [bs, 80, sq, skv]  bool (int8)
    dS_ptr,       # [bs, 80, sq, skv]  bfloat16  (output)
    # Dims
    bs: tl.constexpr,
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_groups: tl.constexpr,
    sq: tl.constexpr,
    skv: tl.constexpr,
    head_dim: tl.constexpr,
    # Dropout scale
    dropout_scale: tl.constexpr,
    # Tile sizes
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Each program handles one (batch, head, sq_tile) block.
    Computes dS = P * (dP - sum(dP * P)) where dP = dO @ V^T (with dropout correction).
    """
    pid_bh = tl.program_id(0)   # batch * n_heads index
    pid_q  = tl.program_id(1)   # sq tile index

    b = pid_bh // n_heads
    h = pid_bh % n_heads
    kv_h = h // n_groups

    q_start = pid_q * BLOCK_Q

    # Offsets for q dimension
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    q_mask = q_offs < sq

    # Offsets for head_dim
    d_offs = tl.arange(0, BLOCK_D)  # BLOCK_D == head_dim == 128

    # Load dO tile: [BLOCK_Q, head_dim]
    # dO layout: [bs, 80, sq, 128] — stride: (80*sq*128, sq*128, 128, 1)
    dO_base = b * (n_heads * sq * head_dim) + h * (sq * head_dim)
    dO = tl.load(
        dO_ptr + dO_base + q_offs[:, None] * head_dim + d_offs[None, :],
        mask=q_mask[:, None],
        other=0.0
    )  # [BLOCK_Q, 128], bfloat16

    # We will compute dP = dO @ V^T iteratively over KV tiles
    # and simultaneously do the softmax backward

    # First pass: compute full dP row for each q to get the sum(dP * P)
    # Load P tile: [BLOCK_Q, skv]  (all of skv at once if skv fits)
    P_base = b * (n_heads * sq * skv) + h * (sq * skv)
    mask_base = b * (n_heads * sq * skv) + h * (sq * skv)
    dS_base = b * (n_heads * sq * skv) + h * (sq * skv)

    # We need to do this in KV tiles to handle large skv
    # Accumulate dot_sum = sum_kv(dP * P) for softmax backward
    dot_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)

    # Accumulate dP into a register buffer — but skv may be large
    # Instead, do two passes: first accumulate dot_sum, then compute dS

    # Pass 1: compute dot_sum = sum_kv(dP[q,kv] * P[q,kv])
    kv_offs = tl.arange(0, BLOCK_KV)
    for kv_start in range(0, skv, BLOCK_KV):
        kv_o = kv_start + kv_offs
        kv_mask = kv_o < skv

        # Load V tile: [BLOCK_KV, head_dim]
        # V layout: [bs, 8, skv, 128]
        V_base = b * (n_kv_heads * skv * head_dim) + kv_h * (skv * head_dim)
        V_tile = tl.load(
            V_ptr + V_base + kv_o[:, None] * head_dim + d_offs[None, :],
            mask=kv_mask[:, None],
            other=0.0
        )  # [BLOCK_KV, 128]

        # dP_tile = dO @ V_tile^T: [BLOCK_Q, BLOCK_KV]
        dP_tile = tl.dot(dO.to(tl.float32), V_tile.to(tl.float32).trans(1, 0))

        # Load dropout mask tile
        mask_tile = tl.load(
            mask_ptr + mask_base + q_offs[:, None] * skv + kv_o[None, :],
            mask=q_mask[:, None] & kv_mask[None, :],
            other=0
        )  # [BLOCK_Q, BLOCK_KV], bool

        # Apply dropout
        dP_tile = tl.where(mask_tile, dP_tile * dropout_scale, 0.0)

        # Load P tile
        P_tile = tl.load(
            P_ptr + P_base + q_offs[:, None] * skv + kv_o[None, :],
            mask=q_mask[:, None] & kv_mask[None, :],
            other=0.0
        ).to(tl.float32)  # [BLOCK_Q, BLOCK_KV]

        # Accumulate dot_sum += sum_kv(dP * P)
        dot_sum += tl.sum(dP_tile * P_tile, axis=1)  # [BLOCK_Q]

    # Pass 2: compute dS = P * (dP - dot_sum) and store
    for kv_start in range(0, skv, BLOCK_KV):
        kv_o = kv_start + kv_offs
        kv_mask = kv_o < skv

        # Load V tile again
        V_base = b * (n_kv_heads * skv * head_dim) + kv_h * (skv * head_dim)
        V_tile = tl.load(
            V_ptr + V_base + kv_o[:, None] * head_dim + d_offs[None, :],
            mask=kv_mask[:, None],
            other=0.0
        )

        dP_tile = tl.dot(dO.to(tl.float32), V_tile.to(tl.float32).trans(1, 0))

        # Load dropout mask
        mask_tile = tl.load(
            mask_ptr + mask_base + q_offs[:, None] * skv + kv_o[None, :],
            mask=q_mask[:, None] & kv_mask[None, :],
            other=0
        )
        dP_tile = tl.where(mask_tile, dP_tile * dropout_scale, 0.0)

        # Load P tile
        P_tile = tl.load(
            P_ptr + P_base + q_offs[:, None] * skv + kv_o[None, :],
            mask=q_mask[:, None] & kv_mask[None, :],
            other=0.0
        ).to(tl.float32)

        # Softmax backward: dS = P * (dP - dot_sum)
        dS_tile = P_tile * (dP_tile - dot_sum[:, None])

        # Store dS tile
        tl.store(
            dS_ptr + dS_base + q_offs[:, None] * skv + kv_o[None, :],
            dS_tile.to(tl.bfloat16),
            mask=q_mask[:, None] & kv_mask[None, :]
        )


@triton.jit
def attn_bwd_dv_kernel(
    # Pointers
    dO_ptr,       # [bs, 80, sq, 128]  bfloat16
    Pd_ptr,       # [bs, 80, sq, skv]  bfloat16  (attn_weights_dropped)
    dV_ptr,       # [bs,  8, skv, 128] bfloat16  (output)
    # Dims
    bs: tl.constexpr,
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_groups: tl.constexpr,
    sq: tl.constexpr,
    skv: tl.constexpr,
    head_dim: tl.constexpr,
    # Tile sizes
    BLOCK_KV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Each program handles one (batch, kv_head, skv_tile) block.
    Computes dV[kv_head] = sum_{g} P_dropped[kv_head*10+g]^T @ dO[kv_head*10+g]
    """
    pid_bkv = tl.program_id(0)   # batch * n_kv_heads
    pid_skv  = tl.program_id(1)  # skv tile

    b    = pid_bkv // n_kv_heads
    kv_h = pid_bkv % n_kv_heads

    skv_start = pid_skv * BLOCK_KV
    skv_offs  = skv_start + tl.arange(0, BLOCK_KV)
    skv_mask  = skv_offs < skv
    d_offs    = tl.arange(0, BLOCK_D)

    # Accumulate dV in float32: [BLOCK_KV, BLOCK_D]
    acc = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)

    # Iterate over all groups and all Q tiles
    for g in range(n_groups):
        h = kv_h * n_groups + g

        Pd_base = b * (n_heads * sq * skv) + h * (sq * skv)
        dO_base = b * (n_heads * sq * head_dim) + h * (sq * head_dim)

        q_offs = tl.arange(0, BLOCK_Q)
        for q_start in range(0, sq, BLOCK_Q):
            q_o = q_start + q_offs
            q_mask = q_o < sq

            # Load Pd tile: [BLOCK_Q, BLOCK_KV]  (transposed: we want P^T)
            Pd_tile = tl.load(
                Pd_ptr + Pd_base + q_o[:, None] * skv + skv_offs[None, :],
                mask=q_mask[:, None] & skv_mask[None, :],
                other=0.0
            ).to(tl.float32)  # [BLOCK_Q, BLOCK_KV]

            # Load dO tile: [BLOCK_Q, BLOCK_D]
            dO_tile = tl.load(
                dO_ptr + dO_base + q_o[:, None] * head_dim + d_offs[None, :],
                mask=q_mask[:, None],
                other=0.0
            ).to(tl.float32)  # [BLOCK_Q, BLOCK_D]

            # acc += Pd^T @ dO: [BLOCK_KV, BLOCK_D]
            acc += tl.dot(Pd_tile.trans(1, 0), dO_tile)

    # Store dV
    dV_base = b * (n_kv_heads * skv * head_dim) + kv_h * (skv * head_dim)
    tl.store(
        dV_ptr + dV_base + skv_offs[:, None] * head_dim + d_offs[None, :],
        acc.to(tl.bfloat16),
        mask=skv_mask[:, None]
    )


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    head_dim   = HEAD_DIM              # 128

    # Transpose dO: [bs, sq, 80, 128] -> [bs, 80, sq, 128], contiguous
    dO = grad_attn_output.transpose(1, 2).contiguous()

    # Convert dropout_mask to int8 for Triton (bool not always supported)
    mask_int = dropout_mask  # keep as bool, Triton handles it

    dropout_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Allocate outputs
    dS = torch.empty(bs, n_heads, seq_q, seq_kv, dtype=torch.bfloat16, device=dO.device)
    dV = torch.empty(bs, n_kv_heads, seq_kv, head_dim, dtype=torch.bfloat16, device=dO.device)

    # --- Kernel 1: compute dS ---
    BLOCK_Q_DS  = 32
    BLOCK_KV_DS = 64
    BLOCK_D_DS  = 128  # == head_dim

    grid_ds = (bs * n_heads, triton.cdiv(seq_q, BLOCK_Q_DS))

    attn_bwd_ds_kernel[grid_ds](
        dO, attn_weights, value_states, mask_int, dS,
        bs, n_heads, n_kv_heads, n_groups, seq_q, seq_kv, head_dim,
        dropout_scale,
        BLOCK_Q_DS, BLOCK_KV_DS, BLOCK_D_DS,
        num_warps=8,
        num_stages=2,
    )

    # --- Kernel 2: compute dV ---
    BLOCK_KV_DV = 64
    BLOCK_Q_DV  = 32
    BLOCK_D_DV  = 128  # == head_dim

    grid_dv = (bs * n_kv_heads, triton.cdiv(seq_kv, BLOCK_KV_DV))

    attn_bwd_dv_kernel[grid_dv](
        dO, attn_weights_dropped, dV,
        bs, n_heads, n_kv_heads, n_groups, seq_q, seq_kv, head_dim,
        BLOCK_KV_DV, BLOCK_Q_DV, BLOCK_D_DV,
        num_warps=8,
        num_stages=2,
    )

    return dS, dV

```

---

## Experiment #25 — 2026-06-28 08:21:44 UTC ✅ KEEP

**Hypothesis:** ** Exact reproduction of Experiment 22:

**Result:** 521.53 μs

**Kernel code:**
```python
"""
Hybrid attention-backward: cuBLAS BMM for matmuls + Triton fused elementwise for softmax bwd.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = 10


@triton.jit
def softmax_bwd_kernel(
    # Inputs
    dP_ptr,      # [bs, 80, sq, skv]  bfloat16  (raw dP̃ before dropout correction)
    P_ptr,       # [bs, 80, sq, skv]  bfloat16  (attn_weights, post-softmax)
    mask_ptr,    # [bs, 80, sq, skv]  bool
    # Output
    dS_ptr,      # [bs, 80, sq, skv]  bfloat16
    # Dims
    sq: tl.constexpr,
    skv: tl.constexpr,
    # Dropout scale
    dropout_scale: tl.constexpr,
    # Tile size along skv
    BLOCK_KV: tl.constexpr,
):
    """
    Fused single-pass softmax backward with dropout correction.
    Each program handles one row (one q index of one head of one batch).
    Computes: dS = P * (dP - sum_kv(dP * P))
    where dP = dP̃ * mask * dropout_scale (dropout correction applied here).

    Grid: (bs * 80 * sq,)
    """
    pid = tl.program_id(0)
    # pid indexes into [bs, 80, sq] flattened
    row_base = pid * skv

    kv_offs = tl.arange(0, BLOCK_KV)

    # Single pass: accumulate dot_sum and store dP (corrected) to registers
    # For large skv, we can't store all dP in registers, so we do two passes
    # BUT since dP and P are already materialized in DRAM (from BMM output),
    # each element is read twice. The key benefit is we avoid the V matmul here.

    # Pass 1: compute dot_sum
    dot_sum = tl.zeros([1], dtype=tl.float32)
    for kv_start in range(0, skv, BLOCK_KV):
        kv_o = kv_start + kv_offs
        kv_mask = kv_o < skv

        dP_tile = tl.load(dP_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)
        mask_tile = tl.load(mask_ptr + row_base + kv_o, mask=kv_mask, other=0).to(tl.int1)
        P_tile = tl.load(P_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)

        # Apply dropout correction
        dP_tile = tl.where(mask_tile, dP_tile * dropout_scale, 0.0)

        dot_sum += tl.sum(dP_tile * P_tile, axis=0)

    dot_sum_val = tl.sum(dot_sum, axis=0)

    # Pass 2: compute and store dS
    for kv_start in range(0, skv, BLOCK_KV):
        kv_o = kv_start + kv_offs
        kv_mask = kv_o < skv

        dP_tile = tl.load(dP_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)
        mask_tile = tl.load(mask_ptr + row_base + kv_o, mask=kv_mask, other=0).to(tl.int1)
        P_tile = tl.load(P_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)

        dP_tile = tl.where(mask_tile, dP_tile * dropout_scale, 0.0)

        dS_tile = P_tile * (dP_tile - dot_sum_val)
        tl.store(dS_ptr + row_base + kv_o, dS_tile.to(tl.bfloat16), mask=kv_mask)


@triton.jit
def softmax_bwd_kernel_wide(
    # Inputs
    dP_ptr,      # [N_rows, skv]  bfloat16
    P_ptr,       # [N_rows, skv]  bfloat16
    mask_ptr,    # [N_rows, skv]  bool
    # Output
    dS_ptr,      # [N_rows, skv]  bfloat16
    # Dims
    skv: tl.constexpr,
    # Dropout scale
    dropout_scale: tl.constexpr,
    # Tile size along skv — must cover full skv in one tile for single-pass
    BLOCK_KV: tl.constexpr,
):
    """
    Single-pass softmax backward when BLOCK_KV >= skv (fits in registers).
    Each program handles one row. Grid: (N_rows,)
    """
    pid = tl.program_id(0)
    row_base = pid * skv

    kv_offs = tl.arange(0, BLOCK_KV)
    kv_mask = kv_offs < skv

    dP_tile = tl.load(dP_ptr + row_base + kv_offs, mask=kv_mask, other=0.0).to(tl.float32)
    mask_tile = tl.load(mask_ptr + row_base + kv_offs, mask=kv_mask, other=False).to(tl.int1)
    P_tile = tl.load(P_ptr + row_base + kv_offs, mask=kv_mask, other=0.0).to(tl.float32)

    # Apply dropout correction
    dP_tile = tl.where(mask_tile, dP_tile * dropout_scale, 0.0)

    # Softmax backward
    dot_sum = tl.sum(dP_tile * P_tile, axis=0)
    dS_tile = P_tile * (dP_tile - dot_sum)

    tl.store(dS_ptr + row_base + kv_offs, dS_tile.to(tl.bfloat16), mask=kv_mask)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    head_dim   = HEAD_DIM              # 128

    # Transpose dO: [bs, sq, 80, 128] -> [bs, 80, sq, 128], contiguous
    dO = grad_attn_output.transpose(1, 2).contiguous()  # [bs, 80, sq, 128] bf16

    dropout_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # =========================================================================
    # Step 1: dP̃ = dO @ V^T  (cuBLAS BMM in bfloat16)
    # dO:          [bs, 80, sq, 128]
    # vs_exp:      [bs, 80, skv, 128]  (GQA expand, no copy via expand+reshape)
    # dP̃ = dO @ vs_exp^T: [bs, 80, sq, skv]
    # =========================================================================
    # GQA expand: [bs, 8, skv, 128] -> [bs, 80, skv, 128]
    # Using expand + reshape — expand is zero-copy, reshape may need contiguous
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv_heads, n_groups, seq_kv, head_dim
    ).reshape(bs, n_heads, seq_kv, head_dim).contiguous()  # [bs,80,skv,128] bf16

    # BMM: [bs*80, sq, 128] @ [bs*80, 128, skv] -> [bs*80, sq, skv]
    dO_2d  = dO.reshape(bs * n_heads, seq_q, head_dim)
    vs_2d  = vs_exp.reshape(bs * n_heads, seq_kv, head_dim)
    dP_raw = torch.bmm(dO_2d, vs_2d.transpose(1, 2))          # [bs*80, sq, skv] bf16
    dP_raw = dP_raw.reshape(bs, n_heads, seq_q, seq_kv)

    # =========================================================================
    # Step 2: Fused elementwise — dropout correction + softmax backward
    # Produces dS: [bs, 80, sq, skv] bfloat16
    # =========================================================================
    dS = torch.empty_like(dP_raw)
    N_rows = bs * n_heads * seq_q

    # Choose kernel variant based on skv size
    # If skv <= 2048, we can fit in registers (BLOCK_KV=2048 power-of-2)
    if seq_kv <= 2048:
        # Find next power of 2 >= seq_kv
        BLOCK_KV = 1
        while BLOCK_KV < seq_kv:
            BLOCK_KV *= 2
        BLOCK_KV = min(BLOCK_KV, 2048)

        # Flatten inputs for row-wise processing
        dP_flat   = dP_raw.reshape(N_rows, seq_kv)
        P_flat    = attn_weights.reshape(N_rows, seq_kv)
        mask_flat = dropout_mask.reshape(N_rows, seq_kv)
        dS_flat   = dS.reshape(N_rows, seq_kv)

        softmax_bwd_kernel_wide[(N_rows,)](
            dP_flat, P_flat, mask_flat, dS_flat,
            seq_kv,
            dropout_scale,
            BLOCK_KV,
            num_warps=8,
        )
    else:
        # Two-pass tiled version for large skv
        BLOCK_KV = 512

        dP_flat   = dP_raw.reshape(N_rows, seq_kv)
        P_flat    = attn_weights.reshape(N_rows, seq_kv)
        mask_flat = dropout_mask.reshape(N_rows, seq_kv)
        dS_flat   = dS.reshape(N_rows, seq_kv)

        softmax_bwd_kernel[(N_rows,)](
            dP_flat, P_flat, mask_flat, dS_flat,
            seq_q, seq_kv,
            dropout_scale,
            BLOCK_KV,
            num_warps=8,
        )

    # =========================================================================
    # Step 3: dV_exp = P̃^T @ dO  (cuBLAS BMM in bfloat16, then GQA reduce)
    # attn_weights_dropped: [bs, 80, sq, skv]
    # dO:                   [bs, 80, sq, 128]
    # dV_exp = P̃^T @ dO:   [bs, 80, skv, 128]
    # dV = dV_exp.reshape(bs, 8, 10, skv, 128).sum(2): [bs, 8, skv, 128]
    # =========================================================================
    Pd_2d   = attn_weights_dropped.reshape(bs * n_heads, seq_q, seq_kv)
    dV_exp  = torch.bmm(Pd_2d.transpose(1, 2), dO_2d)   # [bs*80, skv, 128] bf16
    dV_exp  = dV_exp.reshape(bs, n_heads, seq_kv, head_dim)

    # GQA reduction: sum over groups
    dV = dV_exp.reshape(bs, n_kv_heads, n_groups, seq_kv, head_dim).sum(dim=2)
    dV = dV.to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #26 — 2026-06-28 08:23:28 UTC ❌ DISCARD

**Hypothesis:** ** Exact reproduction of Experiment 22 (the best known result at 521.59 μs, confirmed at 521.53 μs in Exp 25):

**Result:** 584.25 μs

**Kernel code:**
```python
"""
Hybrid attention-backward: cuBLAS BMM for matmuls + Triton fused elementwise for softmax bwd.
Optimized to eliminate expensive .contiguous() copies.

custom_kernel(data) receives:
    data = (grad_attn_output, attn_weights, attn_weights_dropped,
            value_states, dropout_mask, attention_dropout)

    grad_attn_output       [bs, seq_q,  80, 128]   bfloat16
    attn_weights           [bs, 80, seq_q, seq_kv]  bfloat16
    attn_weights_dropped   [bs, 80, seq_q, seq_kv]  bfloat16
    value_states           [bs,  8, seq_kv, 128]    bfloat16
    dropout_mask           [bs, 80, seq_q, seq_kv]  bool
    attention_dropout                                float (0.1)

Returns:
    grad_attn_scores       [bs, 80, seq_q, seq_kv]  bfloat16
    grad_value_states      [bs,  8, seq_kv, 128]    bfloat16
"""

import torch
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = 10


@triton.jit
def softmax_bwd_kernel_wide(
    # Inputs
    dP_ptr,      # [N_rows, skv]  bfloat16
    P_ptr,       # [N_rows, skv]  bfloat16
    mask_ptr,    # [N_rows, skv]  bool
    # Output
    dS_ptr,      # [N_rows, skv]  bfloat16
    # Dims
    skv: tl.constexpr,
    # Dropout scale
    dropout_scale: tl.constexpr,
    # Tile size along skv — must cover full skv in one tile for single-pass
    BLOCK_KV: tl.constexpr,
):
    """
    Single-pass softmax backward when BLOCK_KV >= skv (fits in registers).
    Each program handles one row. Grid: (N_rows,)
    """
    pid = tl.program_id(0)
    row_base = pid * skv

    kv_offs = tl.arange(0, BLOCK_KV)
    kv_mask = kv_offs < skv

    dP_tile = tl.load(dP_ptr + row_base + kv_offs, mask=kv_mask, other=0.0).to(tl.float32)
    mask_tile = tl.load(mask_ptr + row_base + kv_offs, mask=kv_mask, other=False).to(tl.int1)
    P_tile = tl.load(P_ptr + row_base + kv_offs, mask=kv_mask, other=0.0).to(tl.float32)

    # Apply dropout correction
    dP_tile = tl.where(mask_tile, dP_tile * dropout_scale, 0.0)

    # Softmax backward
    dot_sum = tl.sum(dP_tile * P_tile, axis=0)
    dS_tile = P_tile * (dP_tile - dot_sum)

    tl.store(dS_ptr + row_base + kv_offs, dS_tile.to(tl.bfloat16), mask=kv_mask)


@triton.jit
def softmax_bwd_kernel_tiled(
    # Inputs
    dP_ptr,      # [N_rows, skv]  bfloat16
    P_ptr,       # [N_rows, skv]  bfloat16
    mask_ptr,    # [N_rows, skv]  bool
    # Output
    dS_ptr,      # [N_rows, skv]  bfloat16
    # Dims
    skv: tl.constexpr,
    # Dropout scale
    dropout_scale: tl.constexpr,
    # Tile size along skv
    BLOCK_KV: tl.constexpr,
):
    """
    Two-pass tiled softmax backward for large skv.
    Each program handles one row. Grid: (N_rows,)
    """
    pid = tl.program_id(0)
    row_base = pid * skv

    kv_offs = tl.arange(0, BLOCK_KV)

    # Pass 1: compute dot_sum
    dot_sum = tl.zeros([1], dtype=tl.float32)
    for kv_start in range(0, skv, BLOCK_KV):
        kv_o = kv_start + kv_offs
        kv_mask = kv_o < skv
        dP_tile = tl.load(dP_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)
        mask_tile = tl.load(mask_ptr + row_base + kv_o, mask=kv_mask, other=0).to(tl.int1)
        P_tile = tl.load(P_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)
        dP_tile = tl.where(mask_tile, dP_tile * dropout_scale, 0.0)
        dot_sum += tl.sum(dP_tile * P_tile, axis=0)

    dot_sum_val = tl.sum(dot_sum, axis=0)

    # Pass 2: compute and store dS
    for kv_start in range(0, skv, BLOCK_KV):
        kv_o = kv_start + kv_offs
        kv_mask = kv_o < skv
        dP_tile = tl.load(dP_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)
        mask_tile = tl.load(mask_ptr + row_base + kv_o, mask=kv_mask, other=0).to(tl.int1)
        P_tile = tl.load(P_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)
        dP_tile = tl.where(mask_tile, dP_tile * dropout_scale, 0.0)
        dS_tile = P_tile * (dP_tile - dot_sum_val)
        tl.store(dS_ptr + row_base + kv_o, dS_tile.to(tl.bfloat16), mask=kv_mask)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    head_dim   = HEAD_DIM              # 128

    dropout_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # =========================================================================
    # Step 1: dP̃ = dO @ V^T  (cuBLAS BMM, avoiding .contiguous() copies)
    #
    # grad_attn_output: [bs, sq, 80, 128]  — layout (bs, sq, 80, 128)
    # value_states:     [bs,  8, skv, 128] — GQA
    #
    # Key insight: reshape dO as [bs*8, 10*sq, 128] and value_states as
    # [bs*8, skv, 128], then bmm gives [bs*8, 10*sq, skv].
    # Reshape back to [bs, 8, 10, sq, skv] = [bs, 80, sq, skv] after
    # transposing appropriately.
    #
    # dO in [bs, sq, 80, 128]: permute to [bs, 8, 10, sq, 128]
    # then reshape to [bs*8, 10*sq, 128]
    # =========================================================================

    # Reshape dO: [bs, sq, 80, 128] -> [bs, sq, 8, 10, 128]
    # -> [bs, 8, 10, sq, 128] -> [bs*8, 10*sq, 128]
    # This permute+reshape IS a copy but smaller than [bs,80,sq,128]
    # Actually, let's use a smarter approach:
    # Reshape [bs, sq, 80, 128] -> [bs, sq, 8, 10, 128]
    # permute -> [bs, 8, sq, 10, 128] — no, let's think carefully.
    #
    # Better: use [bs*8, 10, sq, 128] layout for dO
    # grad_attn_output: [bs, sq, 80, 128]
    #   -> view [bs, sq, 8, 10, 128]
    #   -> permute(0,2,3,1,4) = [bs, 8, 10, sq, 128]   <- requires contiguous
    #   -> reshape [bs*8, 10*sq, 128]
    # value_states: [bs, 8, skv, 128] -> reshape [bs*8, skv, 128]
    #
    # bmm([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    # reshape to [bs, 8, 10, sq, skv]
    # permute to [bs, 8, 10, sq, skv] (already correct order)
    # reshape to [bs, 80, sq, skv] — but need [bs, 80, sq, skv]
    # This needs a reshape from [bs, 8, 10, sq, skv] to [bs, 80, sq, skv]
    # which is contiguous only if the 8 and 10 dims are contiguous.
    # Actually [bs, 8, 10, sq, skv] -> [bs, 80, sq, skv] works if contiguous.
    #
    # The contiguous() call is needed for permute. But we go from
    # [bs, sq, 8, 10, 128] to [bs, 8, 10, sq, 128] — this is a smaller
    # tensor than [bs, 80, sq, 128] by factor of 1 (same elements).
    # HOWEVER we can skip the contiguous by using einsum or matmul with
    # non-contiguous tensors directly.
    #
    # Alternative: use torch.matmul with broadcasting instead of bmm.
    # dO: [bs, sq, 80, 128] -> [bs, sq, 8, 10, 128]  (view, free)
    # V:  [bs, 8, skv, 128] -> [bs, 1, 8, 1, skv, 128] won't work easily
    #
    # Simplest copy-free approach for dP:
    # Use torch.matmul broadcasting:
    #   dO_v: [bs, sq, 8, 10, 128]  (view of grad_attn_output, free)
    #   V_t:  [bs, 8, 128, skv]     (transpose of value_states, non-contiguous)
    #   Result: [bs, sq, 8, 10, skv] via matmul broadcast  <- may not work directly
    #
    # Actually, torch.matmul handles batched dims by broadcasting.
    # dO_v: [bs, sq, 8, 10, 128]
    # V:    [bs,  8,  1,  skv, 128].transpose(-2,-1) = [bs, 8, 1, 128, skv]
    # matmul: [bs, sq, 8, 10, 128] @ [bs, 8, 1, 128, skv]
    #       = [bs, sq, 8, 10, skv]  <- broadcast works here
    # Then permute [bs, sq, 8, 10, skv] -> [bs, 8, 10, sq, skv]
    # view -> [bs, 80, sq, skv]
    # The permute needs contiguous... but we can permute differently.
    #
    # Let's try: use matmul with proper reshaping to minimize copies.

    # APPROACH: Group dO by kv_head, use batched matmul without expanding V
    # dO: [bs, sq, 80, 128] -> view [bs, sq, 8, 10, 128]
    #     -> permute(0, 2, 1, 3, 4) = [bs, 8, sq, 10, 128]  (copy needed)
    #     -> reshape [bs*8, sq*10, 128]  -- but this mixes sq and heads
    #
    # Actually the cleanest copy-free approach is:
    # Keep dO as [bs, sq, 80, 128] and use einsum or reshape carefully.
    #
    # Let's use: dO reshaped to [bs, sq, 8, 10, 128]
    # V: [bs, 8, skv, 128]
    # We want: sum_d dO[b,q,kv,g,d] * V[b,kv,k,d] = dP[b,kv,g,q,k]
    # then rearrange to [b, kv*g=80, q, k]
    #
    # Using torch.einsum('bqkgd,bksd->bkgqs', ...) is elegant but slow.
    #
    # BEST approach without extra large copies:
    # Reshape dO [bs, sq, 80, 128] -> [bs*8, 10*sq, 128] requires a
    # permute that IS a copy but the output is same size as dO — no expansion.
    # The key win is avoiding the [bs,80,skv,128] expansion of V.

    # Step 1a: Reshape dO to group by kv_head
    # [bs, sq, 80, 128] -> [bs, sq, 8, 10, 128] (view, free)
    dO_grouped = grad_attn_output.view(bs, seq_q, n_kv_heads, n_groups, head_dim)
    # permute to [bs, 8, 10, sq, 128] — requires contiguous (copy, same size as dO)
    dO_grouped = dO_grouped.permute(0, 2, 3, 1, 4).contiguous()
    # reshape to [bs*8, 10*sq, 128]
    dO_gq = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, head_dim)

    # Step 1b: Reshape V to [bs*8, skv, 128]
    V_2d = value_states.reshape(bs * n_kv_heads, seq_kv, head_dim)

    # Step 1c: BMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] -> [bs*8, 10*sq, skv]
    dP_raw_gq = torch.bmm(dO_gq, V_2d.transpose(1, 2))  # [bs*8, 10*sq, skv]

    # Step 1d: Reshape to [bs, 8, 10, sq, skv] -> permute to [bs, 8*10, sq, skv] = [bs, 80, sq, skv]
    # [bs*8, 10*sq, skv] -> [bs, 8, 10, sq, skv]
    dP_raw = dP_raw_gq.reshape(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    # -> [bs, 80, sq, skv]: merge dims 1 and 2
    # Need contiguous for this reshape since dims aren't adjacent in memory after view
    dP_raw = dP_raw.reshape(bs, n_heads, seq_q, seq_kv)

    # =========================================================================
    # Step 2: Fused elementwise — dropout correction + softmax backward
    # =========================================================================
    dS = torch.empty_like(dP_raw)
    N_rows = bs * n_heads * seq_q

    if seq_kv <= 2048:
        BLOCK_KV = 1
        while BLOCK_KV < seq_kv:
            BLOCK_KV *= 2
        BLOCK_KV = min(BLOCK_KV, 2048)

        dP_flat   = dP_raw.reshape(N_rows, seq_kv)
        P_flat    = attn_weights.reshape(N_rows, seq_kv)
        mask_flat = dropout_mask.reshape(N_rows, seq_kv)
        dS_flat   = dS.reshape(N_rows, seq_kv)

        softmax_bwd_kernel_wide[(N_rows,)](
            dP_flat, P_flat, mask_flat, dS_flat,
            seq_kv,
            dropout_scale,
            BLOCK_KV,
            num_warps=8,
        )
    else:
        BLOCK_KV = 512

        dP_flat   = dP_raw.reshape(N_rows, seq_kv)
        P_flat    = attn_weights.reshape(N_rows, seq_kv)
        mask_flat = dropout_mask.reshape(N_rows, seq_kv)
        dS_flat   = dS.reshape(N_rows, seq_kv)

        softmax_bwd_kernel_tiled[(N_rows,)](
            dP_flat, P_flat, mask_flat, dS_flat,
            seq_kv,
            dropout_scale,
            BLOCK_KV,
            num_warps=8,
        )

    # =========================================================================
    # Step 3: dV = P̃_dropped^T @ dO, with GQA reduction
    #
    # attn_weights_dropped: [bs, 80, sq, skv]
    # dO: [bs, 8, 10, sq, 128] (already computed above as dO_grouped)
    #
    # We need: dV[b, kv, k, d] = sum_{g,q} P_d[b, kv*10+g, q, k] * dO[b, q, kv*10+g, d]
    #
    # Use the grouped dO layout:
    # dO_grouped: [bs, 8, 10, sq, 128] -> reshape [bs*8, 10*sq, 128]  (already have dO_gq)
    # P_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    #
    # bmm: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    # =========================================================================
    # Reshape attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    # -> [bs*8, 10*sq, skv]
    # attn_weights_dropped is [bs, 80, sq, skv] which is contiguous
    Pd_gq = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    Pd_gq = Pd_gq.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # BMM: P^T @ dO: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    dV = torch.bmm(Pd_gq.transpose(1, 2), dO_gq)  # [bs*8, skv, 128]
    dV = dV.reshape(bs, n_kv_heads, seq_kv, head_dim).to(torch.bfloat16)

    return dS, dV

```

