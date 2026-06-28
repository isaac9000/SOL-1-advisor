# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-28 08:38:46 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 3428.25 μs

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

## Experiment #2 — 2026-06-28 08:40:26 UTC 💥 CRASH

**Hypothesis:** Two Triton kernels replacing the pure-PyTorch baseline:

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

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


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: compute grad_attn_scores (dS)
#   Grid: (bs * n_heads, cdiv(seq_q, BLOCK_SQ))
#   Each program handles a BLOCK_SQ × seq_kv tile for one (batch, head) pair.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def attn_bwd_ds_kernel(
    # Pointers
    dO_ptr,           # [bs, 80, seq_q, 128]  bfloat16
    V_ptr,            # [bs,  8, seq_kv, 128] bfloat16
    P_ptr,            # [bs, 80, seq_q, seq_kv] bfloat16
    Pd_ptr,           # [bs, 80, seq_q, seq_kv] bfloat16  (dropped)
    mask_ptr,         # [bs, 80, seq_q, seq_kv] bool
    dS_ptr,           # [bs, 80, seq_q, seq_kv] bfloat16  (output)
    # Sizes
    seq_q: tl.constexpr,
    seq_kv: tl.constexpr,
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_groups: tl.constexpr,
    inv_keep: tl.constexpr,   # 1/(1-dropout)
    # Block sizes
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # program IDs
    bh_id  = tl.program_id(0)   # batch * n_heads + head
    sq_blk = tl.program_id(1)   # block along seq_q

    bs_id   = bh_id // n_heads
    head_id = bh_id % n_heads
    kv_id   = head_id // n_groups

    # Offsets into seq_q for this block
    sq_off = sq_blk * BLOCK_SQ + tl.arange(0, BLOCK_SQ)   # [BLOCK_SQ]
    sq_mask = sq_off < seq_q

    # Base pointers for this (batch, head)
    dO_base  = dO_ptr  + bs_id * (n_heads * seq_q * HEAD_DIM) + head_id * (seq_q * HEAD_DIM)
    V_base   = V_ptr   + bs_id * (n_kv_heads * seq_kv * HEAD_DIM) + kv_id * (seq_kv * HEAD_DIM)
    P_base   = P_ptr   + bs_id * (n_heads * seq_q * seq_kv) + head_id * (seq_q * seq_kv)
    Pd_base  = Pd_ptr  + bs_id * (n_heads * seq_q * seq_kv) + head_id * (seq_q * seq_kv)
    mask_base= mask_ptr+ bs_id * (n_heads * seq_q * seq_kv) + head_id * (seq_q * seq_kv)
    dS_base  = dS_ptr  + bs_id * (n_heads * seq_q * seq_kv) + head_id * (seq_q * seq_kv)

    # ── compute dP_dropped = dO @ V^T  ──────────────────────────────────────
    # dO: [BLOCK_SQ, HEAD_DIM],  V: [seq_kv, HEAD_DIM]
    # result: [BLOCK_SQ, seq_kv]

    d_off = tl.arange(0, BLOCK_D)   # [BLOCK_D] = [128]
    skv_off = tl.arange(0, BLOCK_SKV)  # [BLOCK_SKV]

    # Accumulator for dP_dropped: [BLOCK_SQ, BLOCK_SKV]
    # We loop over seq_kv in tiles of BLOCK_SKV
    # But for simplicity (seq_kv may be large), we do the full seq_kv in one shot
    # by loading the full V row and dO row.
    # Load dO tile: [BLOCK_SQ, HEAD_DIM]
    dO_ptrs = dO_base + sq_off[:, None] * HEAD_DIM + d_off[None, :]
    dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)  # [BLOCK_SQ, BLOCK_D]

    # Loop over seq_kv in BLOCK_SKV chunks to compute dP and accumulate rowsum
    # We'll need to do 2 passes (one for rowsum, one for writing dS).
    # Instead, do it in one pass: compute rowsum on the fly using a running sum.

    rowsum = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    # We need to store the full dP row for later, but seq_kv can be large.
    # Instead: pass 1 — compute rowsum = sum_kv(dP * P)
    # pass 2 — write dS = P * (dP - rowsum)
    # But we have to read from HBM twice for P and Pd.
    # Given memory constraints, we do 2 passes.

    # --- Pass 1: compute rowsum ---
    n_skv_blocks = tl.cdiv(seq_kv, BLOCK_SKV)
    for skv_blk in range(n_skv_blocks):
        skv_start = skv_blk * BLOCK_SKV
        skv_idx = skv_start + skv_off
        skv_valid = skv_idx < seq_kv

        # Load V tile: [BLOCK_SKV, HEAD_DIM]
        V_ptrs = V_base + skv_idx[:, None] * HEAD_DIM + d_off[None, :]
        V_tile = tl.load(V_ptrs, mask=skv_valid[:, None], other=0.0).to(tl.float32)

        # dP_dropped tile: [BLOCK_SQ, BLOCK_SKV] = dO @ V^T
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))  # [BLOCK_SQ, BLOCK_SKV]

        # Load dropout mask and undo dropout
        mask_ptrs = mask_base + sq_off[:, None] * seq_kv + skv_idx[None, :]
        drop_mask = tl.load(mask_ptrs, mask=sq_mask[:, None] & skv_valid[None, :], other=False)
        dP_tile = tl.where(drop_mask, dP_tile * inv_keep, 0.0)

        # Load P tile
        P_ptrs = P_base + sq_off[:, None] * seq_kv + skv_idx[None, :]
        P_tile = tl.load(P_ptrs, mask=sq_mask[:, None] & skv_valid[None, :], other=0.0).to(tl.float32)

        # Accumulate rowsum
        rowsum += tl.sum(dP_tile * P_tile, axis=1)

    # --- Pass 2: compute dS and write ---
    for skv_blk in range(n_skv_blocks):
        skv_start = skv_blk * BLOCK_SKV
        skv_idx = skv_start + skv_off
        skv_valid = skv_idx < seq_kv

        # Recompute dP_dropped tile
        V_ptrs = V_base + skv_idx[:, None] * HEAD_DIM + d_off[None, :]
        V_tile = tl.load(V_ptrs, mask=skv_valid[:, None], other=0.0).to(tl.float32)
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))

        mask_ptrs = mask_base + sq_off[:, None] * seq_kv + skv_idx[None, :]
        drop_mask = tl.load(mask_ptrs, mask=sq_mask[:, None] & skv_valid[None, :], other=False)
        dP_tile = tl.where(drop_mask, dP_tile * inv_keep, 0.0)

        P_ptrs = P_base + sq_off[:, None] * seq_kv + skv_idx[None, :]
        P_tile = tl.load(P_ptrs, mask=sq_mask[:, None] & skv_valid[None, :], other=0.0).to(tl.float32)

        # dS = P * (dP - rowsum)
        dS_tile = P_tile * (dP_tile - rowsum[:, None])

        # Write dS
        dS_ptrs = dS_base + sq_off[:, None] * seq_kv + skv_idx[None, :]
        tl.store(dS_ptrs, dS_tile.to(tl.bfloat16),
                 mask=sq_mask[:, None] & skv_valid[None, :])


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: compute grad_value_states (dV)
#   Grid: (bs * n_kv_heads, cdiv(seq_kv, BLOCK_SKV))
#   Each program handles one (batch, kv_head, seq_kv_block) tuple,
#   summing contributions from all 10 query-heads in the group.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def attn_bwd_dv_kernel(
    dO_ptr,     # [bs, 80, seq_q, 128]  bfloat16
    Pd_ptr,     # [bs, 80, seq_q, seq_kv] bfloat16
    dV_ptr,     # [bs,  8, seq_kv, 128] bfloat16  (output)
    seq_q: tl.constexpr,
    seq_kv: tl.constexpr,
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_groups: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bkv_id  = tl.program_id(0)   # batch * n_kv_heads + kv_head
    skv_blk = tl.program_id(1)   # block along seq_kv

    bs_id  = bkv_id // n_kv_heads
    kv_id  = bkv_id % n_kv_heads

    skv_off = skv_blk * BLOCK_SKV + tl.arange(0, BLOCK_SKV)
    skv_mask = skv_off < seq_kv

    d_off  = tl.arange(0, BLOCK_D)
    sq_off = tl.arange(0, BLOCK_SQ)

    # Accumulator for dV: [BLOCK_SKV, BLOCK_D]
    acc = tl.zeros([BLOCK_SKV, BLOCK_D], dtype=tl.float32)

    # Loop over query heads in this KV group
    for g in range(n_groups):
        head_id = kv_id * n_groups + g

        dO_base = dO_ptr + bs_id * (n_heads * seq_q * HEAD_DIM) + head_id * (seq_q * HEAD_DIM)
        Pd_base = Pd_ptr + bs_id * (n_heads * seq_q * seq_kv) + head_id * (seq_q * seq_kv)

        # Loop over seq_q in BLOCK_SQ tiles
        n_sq_blocks = tl.cdiv(seq_q, BLOCK_SQ)
        for sq_blk in range(n_sq_blocks):
            sq_start = sq_blk * BLOCK_SQ
            sq_idx = sq_start + sq_off
            sq_valid = sq_idx < seq_q

            # Load Pd tile: [BLOCK_SQ, BLOCK_SKV]
            Pd_ptrs = Pd_base + sq_idx[:, None] * seq_kv + skv_off[None, :]
            Pd_tile = tl.load(Pd_ptrs, mask=sq_valid[:, None] & skv_mask[None, :], other=0.0).to(tl.float32)

            # Load dO tile: [BLOCK_SQ, BLOCK_D]
            dO_ptrs = dO_base + sq_idx[:, None] * HEAD_DIM + d_off[None, :]
            dO_tile = tl.load(dO_ptrs, mask=sq_valid[:, None], other=0.0).to(tl.float32)

            # dV += Pd^T @ dO  →  [BLOCK_SKV, BLOCK_D]
            acc += tl.dot(tl.trans(Pd_tile), dO_tile)

    # Write dV
    dV_base = dV_ptr + bs_id * (n_kv_heads * seq_kv * HEAD_DIM) + kv_id * (seq_kv * HEAD_DIM)
    dV_ptrs = dV_base + skv_off[:, None] * HEAD_DIM + d_off[None, :]
    tl.store(dV_ptrs, acc.to(tl.bfloat16), mask=skv_mask[:, None])


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    # Transpose dO: [bs, seq_q, 80, 128] → [bs, 80, seq_q, 128]
    dO = grad_attn_output.transpose(1, 2).contiguous()

    # Outputs
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)
    dV = torch.empty((bs, n_kv_heads, seq_kv, HEAD_DIM), dtype=torch.bfloat16, device=dO.device)

    inv_keep = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # ── Kernel 1: dS ──────────────────────────────────────────────────────────
    BLOCK_SQ_DS  = 16
    BLOCK_SKV_DS = 64
    BLOCK_D_DS   = 128  # == HEAD_DIM, constexpr

    grid_ds = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_DS))

    attn_bwd_ds_kernel[grid_ds](
        dO, value_states, attn_weights, attn_weights_dropped, dropout_mask, dS,
        seq_q, seq_kv, n_heads, n_kv_heads, n_groups,
        inv_keep,
        BLOCK_SQ=BLOCK_SQ_DS,
        BLOCK_SKV=BLOCK_SKV_DS,
        BLOCK_D=BLOCK_D_DS,
        num_warps=4,
        num_stages=2,
    )

    # ── Kernel 2: dV ──────────────────────────────────────────────────────────
    BLOCK_SKV_DV = 32
    BLOCK_SQ_DV  = 32
    BLOCK_D_DV   = 128  # == HEAD_DIM

    grid_dv = (bs * n_kv_heads, triton.cdiv(seq_kv, BLOCK_SKV_DV))

    attn_bwd_dv_kernel[grid_dv](
        dO, attn_weights_dropped, dV,
        seq_q, seq_kv, n_heads, n_kv_heads, n_groups,
        BLOCK_SKV=BLOCK_SKV_DV,
        BLOCK_SQ=BLOCK_SQ_DV,
        BLOCK_D=BLOCK_D_DV,
        num_warps=4,
        num_stages=2,
    )

    return dS, dV

```

---

## Experiment #3 — 2026-06-28 08:41:54 UTC 💥 CRASH

**Hypothesis:** 1. `inv_keep` is now a regular Python float passed as a runtime kernel argument (not `tl.constexpr`). All structural constants (n_heads, n_kv_heads, n_groups, block sizes) remain `tl.constexpr`.

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Triton-fused attention-backward kernel (corrected).

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
#   Grid: (bs * n_heads, cdiv(seq_q, BLOCK_SQ))
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def attn_bwd_ds_kernel(
    dO_ptr,        # [bs, 80, seq_q, 128]      bfloat16
    V_ptr,         # [bs,  8, seq_kv, 128]     bfloat16
    P_ptr,         # [bs, 80, seq_q, seq_kv]   bfloat16
    Pd_ptr,        # [bs, 80, seq_q, seq_kv]   bfloat16  (dropped)
    mask_ptr,      # [bs, 80, seq_q, seq_kv]   bool
    dS_ptr,        # [bs, 80, seq_q, seq_kv]   bfloat16  (output)
    inv_keep,      # float — 1/(1-dropout), regular runtime arg
    seq_q,         # int
    seq_kv,        # int
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_groups: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh_id  = tl.program_id(0)
    sq_blk = tl.program_id(1)

    bs_id   = bh_id // n_heads
    head_id = bh_id % n_heads
    kv_id   = head_id // n_groups

    sq_off  = sq_blk * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    sq_mask = sq_off < seq_q
    d_off   = tl.arange(0, BLOCK_D)
    skv_off = tl.arange(0, BLOCK_SKV)

    # Strides (elements)
    dO_stride_bs   = n_heads * seq_q * HEAD_DIM
    dO_stride_head = seq_q * HEAD_DIM
    V_stride_bs    = n_kv_heads * seq_kv * HEAD_DIM
    V_stride_kv    = seq_kv * HEAD_DIM
    attn_stride_bs = n_heads * seq_q * seq_kv
    attn_stride_h  = seq_q * seq_kv

    dO_base   = dO_ptr  + bs_id * dO_stride_bs   + head_id * dO_stride_head
    V_base    = V_ptr   + bs_id * V_stride_bs    + kv_id   * V_stride_kv
    P_base    = P_ptr   + bs_id * attn_stride_bs + head_id * attn_stride_h
    Pd_base   = Pd_ptr  + bs_id * attn_stride_bs + head_id * attn_stride_h
    mask_base = mask_ptr+ bs_id * attn_stride_bs + head_id * attn_stride_h
    dS_base   = dS_ptr  + bs_id * attn_stride_bs + head_id * attn_stride_h

    # Load dO tile: [BLOCK_SQ, BLOCK_D]
    dO_ptrs = dO_base + sq_off[:, None] * HEAD_DIM + d_off[None, :]
    dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)

    # --- Pass 1: compute rowsum = sum_kv(dP * P) ---
    rowsum = tl.zeros([BLOCK_SQ], dtype=tl.float32)
    n_skv_blocks = tl.cdiv(seq_kv, BLOCK_SKV)

    for skv_blk in range(n_skv_blocks):
        skv_idx   = skv_blk * BLOCK_SKV + skv_off
        skv_valid = skv_idx < seq_kv

        # Load V: [BLOCK_SKV, BLOCK_D]
        V_ptrs = V_base + skv_idx[:, None] * HEAD_DIM + d_off[None, :]
        V_tile = tl.load(V_ptrs, mask=skv_valid[:, None], other=0.0).to(tl.float32)

        # dP_dropped = dO @ V^T: [BLOCK_SQ, BLOCK_SKV]
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))

        # Apply dropout undo
        m_ptrs  = mask_base + sq_off[:, None] * seq_kv + skv_idx[None, :]
        m_valid = sq_mask[:, None] & skv_valid[None, :]
        drop_mask = tl.load(m_ptrs, mask=m_valid, other=0).to(tl.int1)
        dP_tile   = tl.where(drop_mask, dP_tile * inv_keep, 0.0)

        # Load P: [BLOCK_SQ, BLOCK_SKV]
        P_ptrs = P_base + sq_off[:, None] * seq_kv + skv_idx[None, :]
        P_tile = tl.load(P_ptrs, mask=m_valid, other=0.0).to(tl.float32)

        rowsum += tl.sum(dP_tile * P_tile, axis=1)

    # --- Pass 2: compute and write dS = P * (dP - rowsum) ---
    for skv_blk in range(n_skv_blocks):
        skv_idx   = skv_blk * BLOCK_SKV + skv_off
        skv_valid = skv_idx < seq_kv
        m_valid   = sq_mask[:, None] & skv_valid[None, :]

        V_ptrs = V_base + skv_idx[:, None] * HEAD_DIM + d_off[None, :]
        V_tile = tl.load(V_ptrs, mask=skv_valid[:, None], other=0.0).to(tl.float32)
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))

        m_ptrs    = mask_base + sq_off[:, None] * seq_kv + skv_idx[None, :]
        drop_mask = tl.load(m_ptrs, mask=m_valid, other=0).to(tl.int1)
        dP_tile   = tl.where(drop_mask, dP_tile * inv_keep, 0.0)

        P_ptrs = P_base + sq_off[:, None] * seq_kv + skv_idx[None, :]
        P_tile = tl.load(P_ptrs, mask=m_valid, other=0.0).to(tl.float32)

        dS_tile = P_tile * (dP_tile - rowsum[:, None])

        dS_ptrs = dS_base + sq_off[:, None] * seq_kv + skv_idx[None, :]
        tl.store(dS_ptrs, dS_tile.to(tl.bfloat16), mask=m_valid)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: compute grad_value_states (dV)
#   Grid: (bs * n_kv_heads, cdiv(seq_kv, BLOCK_SKV))
#   Load Pd as [BLOCK_SKV, BLOCK_SQ] (transposed layout) to avoid tl.trans
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def attn_bwd_dv_kernel(
    dO_ptr,     # [bs, 80, seq_q, 128]      bfloat16
    Pd_ptr,     # [bs, 80, seq_q, seq_kv]   bfloat16
    dV_ptr,     # [bs,  8, seq_kv, 128]     bfloat16  (output)
    seq_q,      # int
    seq_kv,     # int
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_groups: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bkv_id  = tl.program_id(0)
    skv_blk = tl.program_id(1)

    bs_id = bkv_id // n_kv_heads
    kv_id = bkv_id % n_kv_heads

    skv_off  = skv_blk * BLOCK_SKV + tl.arange(0, BLOCK_SKV)
    skv_mask = skv_off < seq_kv
    d_off    = tl.arange(0, BLOCK_D)
    sq_off   = tl.arange(0, BLOCK_SQ)

    dO_stride_bs   = n_heads * seq_q * HEAD_DIM
    dO_stride_head = seq_q * HEAD_DIM
    Pd_stride_bs   = n_heads * seq_q * seq_kv
    Pd_stride_h    = seq_q * seq_kv
    dV_stride_bs   = n_kv_heads * seq_kv * HEAD_DIM
    dV_stride_kv   = seq_kv * HEAD_DIM

    # Accumulator: [BLOCK_SKV, BLOCK_D]
    acc = tl.zeros([BLOCK_SKV, BLOCK_D], dtype=tl.float32)

    n_sq_blocks = tl.cdiv(seq_q, BLOCK_SQ)

    for g in range(n_groups):
        head_id = kv_id * n_groups + g
        dO_base = dO_ptr + bs_id * dO_stride_bs + head_id * dO_stride_head
        Pd_base = Pd_ptr + bs_id * Pd_stride_bs + head_id * Pd_stride_h

        for sq_blk in range(n_sq_blocks):
            sq_idx   = sq_blk * BLOCK_SQ + sq_off
            sq_valid = sq_idx < seq_q
            m_valid  = skv_mask[:, None] & sq_valid[None, :]

            # Load Pd in TRANSPOSED layout: [BLOCK_SKV, BLOCK_SQ]
            # Pd original: [seq_q, seq_kv], we want P^T[skv, sq] = Pd[sq, skv]
            Pd_ptrs = Pd_base + sq_idx[None, :] * seq_kv + skv_off[:, None]
            Pd_tile_T = tl.load(Pd_ptrs, mask=m_valid, other=0.0).to(tl.float32)
            # Pd_tile_T is [BLOCK_SKV, BLOCK_SQ]

            # Load dO tile: [BLOCK_SQ, BLOCK_D]
            dO_ptrs = dO_base + sq_idx[:, None] * HEAD_DIM + d_off[None, :]
            dO_tile = tl.load(dO_ptrs, mask=sq_valid[:, None], other=0.0).to(tl.float32)

            # dV += Pd^T @ dO = [BLOCK_SKV, BLOCK_SQ] @ [BLOCK_SQ, BLOCK_D]
            acc += tl.dot(Pd_tile_T, dO_tile)

    # Write dV
    dV_base = dV_ptr + bs_id * dV_stride_bs + kv_id * dV_stride_kv
    dV_ptrs = dV_base + skv_off[:, None] * HEAD_DIM + d_off[None, :]
    tl.store(dV_ptrs, acc.to(tl.bfloat16), mask=skv_mask[:, None])


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    # Transpose dO: [bs, seq_q, 80, 128] → [bs, 80, seq_q, 128]
    dO = grad_attn_output.transpose(1, 2).contiguous()

    # Outputs
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16, device=dO.device)
    dV = torch.empty((bs, n_kv_heads, seq_kv, HEAD_DIM), dtype=torch.bfloat16, device=dO.device)

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # ── Kernel 1: dS ──────────────────────────────────────────────────────────
    BLOCK_SQ_DS  = 16
    BLOCK_SKV_DS = 64
    BLOCK_D_DS   = 128

    grid_ds = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ_DS))

    attn_bwd_ds_kernel[grid_ds](
        dO, value_states, attn_weights, attn_weights_dropped, dropout_mask, dS,
        inv_keep,
        seq_q, seq_kv,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_groups=n_groups,
        BLOCK_SQ=BLOCK_SQ_DS,
        BLOCK_SKV=BLOCK_SKV_DS,
        BLOCK_D=BLOCK_D_DS,
        num_warps=4,
        num_stages=2,
    )

    # ── Kernel 2: dV ──────────────────────────────────────────────────────────
    BLOCK_SKV_DV = 32
    BLOCK_SQ_DV  = 32
    BLOCK_D_DV   = 128

    grid_dv = (bs * n_kv_heads, triton.cdiv(seq_kv, BLOCK_SKV_DV))

    attn_bwd_dv_kernel[grid_dv](
        dO, attn_weights_dropped, dV,
        seq_q, seq_kv,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_groups=n_groups,
        BLOCK_SKV=BLOCK_SKV_DV,
        BLOCK_SQ=BLOCK_SQ_DV,
        BLOCK_D=BLOCK_D_DV,
        num_warps=4,
        num_stages=2,
    )

    return dS, dV

```

---

## Experiment #4 — 2026-06-28 08:43:35 UTC ✅ KEEP

**Hypothesis:** 1. `_attn_bwd_impl` — a clean pure-PyTorch function that uses the GQA group structure directly: reshape dO to [bs,8,10,sq,128] and V to [bs,8,1,skv,128] (broadcast), so `torch.matmul` handles the 10-g

**Result:** 2194.37 μs

**Kernel code:**
```python
"""
torch.compile-accelerated attention-backward kernel.

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
    dO,               # [bs, 80, seq_q, 128]      float32
    attn_weights,     # [bs, 80, seq_q, seq_kv]   bfloat16
    attn_weights_dropped,  # [bs, 80, seq_q, seq_kv]  bfloat16
    value_states,     # [bs,  8, seq_kv, 128]     bfloat16
    dropout_mask,     # [bs, 80, seq_q, seq_kv]   bool
    inv_keep,         # float scalar
    bs, seq_q, seq_kv,
):
    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    # ── dP computation: dO @ V^T ─────────────────────────────────────────────
    # Reshape to exploit GQA structure: avoid full [bs,80,skv,128] expansion.
    # dO:    [bs, 8, 10, sq, 128]
    # V:     [bs, 8,  1, skv, 128]  (broadcast over groups)
    # dP:    [bs, 8, 10, sq, skv]

    dO_g  = dO.view(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)
    V_g   = value_states.unsqueeze(2)   # [bs, 8, 1, skv, 128]

    # Cast V to float32 for matmul
    V_g_f = V_g.to(torch.float32)

    # dP_dropped: [bs, 8, 10, sq, skv]
    dP_dropped_g = torch.matmul(dO_g, V_g_f.transpose(-2, -1))

    # Reshape dropout_mask and attn_weights to group layout
    mask_g  = dropout_mask.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dP_g    = dP_dropped_g * mask_g * inv_keep

    P_g = attn_weights.view(bs, n_kv_heads, n_groups, seq_q, seq_kv).to(torch.float32)

    # Softmax backward: dS = P * (dP - rowsum(P * dP))
    PdP_g   = P_g * dP_g
    rowsum  = PdP_g.sum(dim=-1, keepdim=True)   # [bs, 8, 10, sq, 1]
    dS_g    = P_g * (dP_g - rowsum)             # [bs, 8, 10, sq, skv]

    dS = dS_g.view(bs, n_heads, seq_q, seq_kv).to(torch.bfloat16)

    # ── dV computation: Pd^T @ dO ─────────────────────────────────────────────
    # attn_weights_dropped: [bs, 8, 10, sq, skv]
    # dO:                   [bs, 8, 10, sq, 128]
    # dV_per_group:         [bs, 8, 10, skv, 128] → sum over groups dim

    Pd_g = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv).to(torch.float32)
    # dV_g: [bs, 8, 10, skv, 128]
    dV_g = torch.matmul(Pd_g.transpose(-2, -1), dO_g)  # Pd^T @ dO
    # Sum over the 10 groups
    dV   = dV_g.sum(dim=2).to(torch.bfloat16)          # [bs, 8, skv, 128]

    return dS, dV


# Cache compiled function at module level
_compiled_fn = None


def _get_compiled_fn():
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = torch.compile(
            _attn_bwd_impl,
            mode="max-autotune",
            fullgraph=True,
        )
    return _compiled_fn


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Transpose dO: [bs, seq_q, 80, 128] → [bs, 80, seq_q, 128] and cast to f32
    dO = grad_attn_output.transpose(1, 2).contiguous().to(torch.float32)

    fn = _get_compiled_fn()
    return fn(
        dO, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, inv_keep,
        bs, seq_q, seq_kv,
    )

```

---

## Experiment #5 — 2026-06-28 08:45:01 UTC ✅ KEEP

**Hypothesis:** 1. **Removed the pre-compiled `.contiguous().to(torch.float32)` step** — `grad_attn_output` is now passed raw (bf16) directly into the compiled function, and the `permute+reshape` happens inside, lett

**Result:** 1175.61 μs

**Kernel code:**
```python
"""
torch.compile-accelerated attention-backward kernel (optimized v2).

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
    grad_attn_output,      # [bs, seq_q, 80, 128]      bfloat16  (raw, un-transposed)
    attn_weights,          # [bs, 80, seq_q, seq_kv]   bfloat16
    attn_weights_dropped,  # [bs, 80, seq_q, seq_kv]   bfloat16
    value_states,          # [bs,  8, seq_kv, 128]     bfloat16
    dropout_mask,          # [bs, 80, seq_q, seq_kv]   bool
    inv_keep,              # float scalar
    bs, seq_q, seq_kv,
):
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    # Transpose inside compiled fn: [bs, seq_q, 80, 128] → [bs, 80, seq_q, 128]
    # then view as group layout [bs, 8, 10, sq, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128]  bf16
    dO_g = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)  # [bs,8,10,sq,128] bf16

    # V: [bs, 8, 1, skv, 128]  (broadcast over groups)
    V_g = value_states.unsqueeze(2)  # [bs, 8, 1, skv, 128]  bf16

    # ── dP computation: dO @ V^T ─────────────────────────────────────────────
    # dP_dropped: [bs, 8, 10, sq, skv]  — keep bf16 for speed
    dP_dropped_g = torch.matmul(dO_g, V_g.transpose(-2, -1))  # bf16 @ bf16

    # Dropout undo
    mask_g = dropout_mask.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dP_g   = dP_dropped_g * mask_g * inv_keep  # still bf16

    # Softmax backward in float32 for numerical accuracy
    P_g    = attn_weights.view(bs, n_kv_heads, n_groups, seq_q, seq_kv).to(torch.float32)
    dP_g_f = dP_g.to(torch.float32)
    # Fused single expression: compiler can fuse into one kernel
    dS_g   = P_g * (dP_g_f - (P_g * dP_g_f).sum(dim=-1, keepdim=True))
    dS     = dS_g.view(bs, n_heads, seq_q, seq_kv).to(torch.bfloat16)

    # ── dV computation: Pd^T @ dO ─────────────────────────────────────────────
    # Keep Pd in bf16, dO_g in bf16 → bf16 matmul, accumulate in f32
    Pd_g = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)  # bf16
    # [bs, 8, 10, skv, sq] @ [bs, 8, 10, sq, 128] → [bs, 8, 10, skv, 128]
    dV_g = torch.matmul(Pd_g.transpose(-2, -1), dO_g)  # bf16 @ bf16
    # Sum over the 10 groups → [bs, 8, skv, 128]
    dV   = dV_g.sum(dim=2).to(torch.bfloat16)

    return dS, dV


# Cache compiled function at module level (reduce-overhead for low-latency)
_compiled_fn = None


def _get_compiled_fn():
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = torch.compile(
            _attn_bwd_impl,
            mode="reduce-overhead",
            fullgraph=True,
        )
    return _compiled_fn


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    fn = _get_compiled_fn()
    return fn(
        grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, inv_keep,
        bs, seq_q, seq_kv,
    )

```

---

## Experiment #6 — 2026-06-28 08:46:33 UTC ✅ KEEP

**Hypothesis:** 1. **Fully bf16 softmax backward**: Removed all `.to(torch.float32)` casts from `P_g`, `dP_g_f`. The softmax backward `P_g * (dP_g - (P_g * dP_g).sum(dim=-1, keepdim=True))` now runs entirely in bfloa

**Result:** 689.58 μs

**Kernel code:**
```python
"""
torch.compile-accelerated attention-backward kernel (optimized v3).

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
    grad_attn_output,      # [bs, seq_q, 80, 128]      bfloat16
    attn_weights,          # [bs, 80, seq_q, seq_kv]   bfloat16
    attn_weights_dropped,  # [bs, 80, seq_q, seq_kv]   bfloat16
    value_states,          # [bs,  8, seq_kv, 128]     bfloat16
    dropout_mask,          # [bs, 80, seq_q, seq_kv]   bool
    inv_keep,              # float scalar
    bs, seq_q, seq_kv,
):
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    # Transpose: [bs, seq_q, 80, 128] → [bs, 80, seq_q, 128]
    # then view as group layout [bs, 8, 10, sq, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128]  bf16
    dO_g = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)  # [bs,8,10,sq,128] bf16

    # ── dP computation: dO @ V^T ─────────────────────────────────────────────
    # V: [bs, 8, 1, skv, 128] broadcast over groups
    V_g = value_states.unsqueeze(2)
    # dP_dropped: [bs, 8, 10, sq, skv]  bf16
    dP_dropped_g = torch.matmul(dO_g, V_g.transpose(-2, -1))

    # Dropout undo — fully bf16
    mask_g = dropout_mask.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dP_g   = dP_dropped_g * mask_g * inv_keep

    # Softmax backward — fully bf16 (tolerance is 1e-2, bf16 is sufficient)
    P_g  = attn_weights.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    # Single fused expression to minimise temporaries
    dS_g = P_g * (dP_g - (P_g * dP_g).sum(dim=-1, keepdim=True))
    dS   = dS_g.view(bs, n_heads, seq_q, seq_kv)  # already bf16

    # ── dV computation: Pd^T @ dO via flat batched GEMM then reshape+sum ─────
    # Flatten groups into batch: [bs*8*10, sq, skv] and [bs*8*10, sq, 128]
    Pd_flat = attn_weights_dropped.view(bs * n_heads, seq_q, seq_kv)  # [bs*80, sq, skv]
    dO_flat = dO.reshape(bs * n_heads, seq_q, HEAD_DIM)               # [bs*80, sq, 128]
    # Batched: [bs*80, skv, sq] @ [bs*80, sq, 128] → [bs*80, skv, 128]
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat)           # bf16
    # Reshape and sum over 10 groups → [bs, 8, skv, 128]
    dV = dV_flat.view(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM).sum(dim=2).to(torch.bfloat16)

    return dS, dV


# Cache compiled function at module level
_compiled_fn = None


def _get_compiled_fn():
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = torch.compile(
            _attn_bwd_impl,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
        )
    return _compiled_fn


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    fn = _get_compiled_fn()
    return fn(
        grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, inv_keep,
        bs, seq_q, seq_kv,
    )

```

---

## Experiment #7 — 2026-06-28 08:47:36 UTC 💥 CRASH

**Hypothesis:** 1. **Split into two compiled functions** `_attn_bwd_ds` and `_attn_bwd_dv` — each takes only the tensors it needs (no shared state), enabling true concurrent execution on separate CUDA streams.

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
torch.compile-accelerated attention-backward kernel (optimized v4).

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


def _attn_bwd_ds(
    grad_attn_output,      # [bs, seq_q, 80, 128]      bfloat16
    attn_weights,          # [bs, 80, seq_q, seq_kv]   bfloat16
    value_states,          # [bs,  8, seq_kv, 128]     bfloat16
    dropout_mask,          # [bs, 80, seq_q, seq_kv]   bool
    inv_keep,              # float scalar
):
    bs        = grad_attn_output.shape[0]
    seq_q     = grad_attn_output.shape[1]
    seq_kv    = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = N_GROUPS
    n_heads    = NUM_ATTENTION_HEADS

    dO   = grad_attn_output.permute(0, 2, 1, 3)                          # [bs,80,sq,128]
    dO_g = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)         # [bs,8,10,sq,128]
    V_g  = value_states.unsqueeze(2)                                      # [bs,8,1,skv,128]

    dP_dropped_g = torch.matmul(dO_g, V_g.transpose(-2, -1))             # [bs,8,10,sq,skv]
    mask_g = dropout_mask.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dP_g   = dP_dropped_g * mask_g * inv_keep

    P_g  = attn_weights.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dS_g = P_g * (dP_g - (P_g * dP_g).sum(dim=-1, keepdim=True))
    dS   = dS_g.view(bs, n_heads, seq_q, seq_kv)
    return dS


def _attn_bwd_dv(
    grad_attn_output,      # [bs, seq_q, 80, 128]      bfloat16
    attn_weights_dropped,  # [bs, 80, seq_q, seq_kv]   bfloat16
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = attn_weights_dropped.shape[3]
    n_kv_heads = NUM_KEY_VALUE_HEADS
    n_groups   = N_GROUPS
    n_heads    = NUM_ATTENTION_HEADS

    dO      = grad_attn_output.permute(0, 2, 1, 3)               # [bs,80,sq,128]
    dO_g    = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)  # [bs,8,10,sq,128]
    Pd_g    = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    # [bs,8,10,skv,sq] @ [bs,8,10,sq,128] → [bs,8,10,skv,128]
    dV_g    = torch.matmul(Pd_g.transpose(-2, -1), dO_g)
    dV      = dV_g.sum(dim=2).to(torch.bfloat16)                 # [bs,8,skv,128]
    return dV


# Cache compiled functions at module level
_compiled_ds = None
_compiled_dv = None


def _get_compiled_fns():
    global _compiled_ds, _compiled_dv
    if _compiled_ds is None:
        _compiled_ds = torch.compile(
            _attn_bwd_ds,
            mode="reduce-overhead",
            fullgraph=True,
        )
    if _compiled_dv is None:
        _compiled_dv = torch.compile(
            _attn_bwd_dv,
            mode="reduce-overhead",
            fullgraph=True,
        )
    return _compiled_ds, _compiled_dv


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    fn_ds, fn_dv = _get_compiled_fns()

    # Launch dV on a side stream, dS on the default stream — no data dependency
    side_stream = torch.cuda.Stream()
    with torch.cuda.stream(side_stream):
        dV = fn_dv(grad_attn_output, attn_weights_dropped)

    # dS on default stream (runs concurrently with dV on side stream)
    dS = fn_ds(grad_attn_output, attn_weights, value_states, dropout_mask, inv_keep)

    # Sync side stream back to default stream
    torch.cuda.current_stream().wait_stream(side_stream)

    return dS, dV

```

---

## Experiment #8 — 2026-06-28 08:49:07 UTC ✅ KEEP

**Hypothesis:** 1. **dP as flat batched GEMM**: Reshaped `dO` to `[bs*8, 10*sq, 128]` and `V` to `[bs*8, skv, 128]`, then `torch.bmm(dO_flat_dp, V_flat.transpose(-2,-1))` gives `[bs*8, 10*sq, skv]` — a single large G

**Result:** 594.31 μs

**Kernel code:**
```python
"""
torch.compile-accelerated attention-backward kernel (optimized v4).

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
    grad_attn_output,      # [bs, seq_q, 80, 128]      bfloat16
    attn_weights,          # [bs, 80, seq_q, seq_kv]   bfloat16
    attn_weights_dropped,  # [bs, 80, seq_q, seq_kv]   bfloat16
    value_states,          # [bs,  8, seq_kv, 128]     bfloat16
    dropout_mask,          # [bs, 80, seq_q, seq_kv]   bool
    inv_keep,              # float scalar
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    # Transpose: [bs, seq_q, 80, 128] → [bs, 80, seq_q, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128]  bf16

    # ── dP computation: flat batched GEMM ────────────────────────────────────
    # Reshape dO: [bs*8, 10*sq, 128]  and  V: [bs*8, 128, skv]
    # → single GEMM: [bs*8, 10*sq, skv]
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)  # [bs*8, 10*sq, 128]
    V_flat     = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)  # [bs*8, skv, 128]
    # bmm: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv]
    dP_dropped_flat = torch.bmm(dO_flat_dp, V_flat.transpose(-2, -1))    # [bs*8, 10*sq, skv]
    # Reshape back for elementwise ops
    dP_dropped_g = dP_dropped_flat.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)

    # Dropout undo — fully bf16, use torch.where to avoid extra allocation
    mask_g = dropout_mask.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dP_g   = torch.where(mask_g, dP_dropped_g * inv_keep, torch.zeros_like(dP_dropped_g))

    # Softmax backward — fully bf16
    P_g  = attn_weights.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dS_g = P_g * (dP_g - (P_g * dP_g).sum(dim=-1, keepdim=True))
    dS   = dS_g.view(bs, n_heads, seq_q, seq_kv)  # already bf16

    # ── dV computation: flat batched GEMM then reshape+sum ───────────────────
    # Flatten: [bs*80, sq, skv] and [bs*80, sq, 128]
    Pd_flat = attn_weights_dropped.view(bs * n_heads, seq_q, seq_kv)  # [bs*80, sq, skv]
    dO_flat = dO.reshape(bs * n_heads, seq_q, HEAD_DIM)               # [bs*80, sq, 128]
    # Batched: [bs*80, skv, sq] @ [bs*80, sq, 128] → [bs*80, skv, 128]
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat)           # bf16
    # Reshape and sum over 10 groups → [bs, 8, skv, 128]
    dV = dV_flat.view(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM).sum(dim=2).to(torch.bfloat16)

    return dS, dV


# Cache compiled function at module level
_compiled_fn = None


def _get_compiled_fn():
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = torch.compile(
            _attn_bwd_impl,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
        )
    return _compiled_fn


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    fn = _get_compiled_fn()
    return fn(
        grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, inv_keep,
    )

```

---

## Experiment #9 — 2026-06-28 08:50:39 UTC 💥 CRASH

**Hypothesis:** 1. **Split into two compiled functions** `_attn_bwd_ds` and `_attn_bwd_dv`, both with `mode="max-autotune-no-cudagraphs"` — no CUDA graphs, so multi-stream is safe.

**Result:** CRASH

**Error:**
```
Benchmark correctness failed. Benchmark correctness: grad_value_states mismatch: max=5.1562e-01 mean=7.1714e-02
```

**Kernel code:**
```python
"""
torch.compile-accelerated attention-backward kernel (optimized v5).
Multi-stream parallelism with max-autotune-no-cudagraphs.

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


def _attn_bwd_ds(
    grad_attn_output,  # [bs, seq_q, 80, 128]      bfloat16
    attn_weights,      # [bs, 80, seq_q, seq_kv]   bfloat16
    value_states,      # [bs,  8, seq_kv, 128]     bfloat16
    dropout_mask,      # [bs, 80, seq_q, seq_kv]   bool
    inv_keep,          # float scalar
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    # Transpose: [bs, seq_q, 80, 128] → [bs, 80, seq_q, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3)

    # dP: flat batched GEMM [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv]
    dO_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    V_flat  = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    dP_flat = torch.bmm(dO_flat, V_flat.transpose(-2, -1))
    dP_g    = dP_flat.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)

    # Dropout undo — bf16 multiply by bool mask
    mask_g = dropout_mask.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dP_g   = dP_g * mask_g * inv_keep

    # Softmax backward — fully bf16
    P_g  = attn_weights.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dS_g = P_g * (dP_g - (P_g * dP_g).sum(dim=-1, keepdim=True))
    return dS_g.view(bs, n_heads, seq_q, seq_kv)


def _attn_bwd_dv(
    grad_attn_output,      # [bs, seq_q, 80, 128]      bfloat16
    attn_weights_dropped,  # [bs, 80, seq_q, seq_kv]   bfloat16
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = attn_weights_dropped.shape[3]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    # dO: [bs, 80, sq, 128] → [bs*8, skv_dim: 10*sq, 128] for compact dV GEMM
    dO = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128]

    # Compact dV: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] → [bs*8, skv, 128]
    # batch=bs*8, K=10*sq accumulates all 10 groups in one GEMM — no sum needed!
    Pd_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)  # [bs*8, 10*sq, skv]
    dO_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)                  # [bs*8, 10*sq, 128]
    # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] → [bs*8, skv, 128]
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat)
    return dV_flat.view(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)


# Cache compiled functions at module level
_compiled_ds = None
_compiled_dv = None


def _get_compiled_fns():
    global _compiled_ds, _compiled_dv
    if _compiled_ds is None:
        _compiled_ds = torch.compile(
            _attn_bwd_ds,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
        )
    if _compiled_dv is None:
        _compiled_dv = torch.compile(
            _attn_bwd_dv,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
        )
    return _compiled_ds, _compiled_dv


# Persistent side stream
_side_stream = None


def _get_side_stream():
    global _side_stream
    if _side_stream is None:
        _side_stream = torch.cuda.Stream()
    return _side_stream


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    fn_ds, fn_dv = _get_compiled_fns()
    side_stream  = _get_side_stream()

    # Launch dV on side stream (no dependency on dS inputs)
    with torch.cuda.stream(side_stream):
        dV = fn_dv(grad_attn_output, attn_weights_dropped)

    # Launch dS on default stream (runs concurrently with dV)
    dS = fn_ds(grad_attn_output, attn_weights, value_states, dropout_mask, inv_keep)

    # Sync side stream back into default stream before returning
    torch.cuda.current_stream().wait_stream(side_stream)

    return dS, dV

```

---

## Experiment #10 — 2026-06-28 08:52:20 UTC ❌ DISCARD

**Hypothesis:** 1. **Compact dV GEMM**: `attn_weights_dropped` (shape [bs,80,sq,skv]) is reshaped via `.reshape()` to `[bs*8, 10*sq, skv]`, and `dO` similarly to `[bs*8, 10*sq, 128]`. The bmm `[bs*8, skv, 10*sq] @ [b

**Result:** 655.98 μs

**Kernel code:**
```python
"""
torch.compile-accelerated attention-backward kernel (optimized v5).
Single compiled function, compact dV GEMM that folds group-sum into K dimension.

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
    grad_attn_output,      # [bs, seq_q, 80, 128]      bfloat16
    attn_weights,          # [bs, 80, seq_q, seq_kv]   bfloat16
    attn_weights_dropped,  # [bs, 80, seq_q, seq_kv]   bfloat16
    value_states,          # [bs,  8, seq_kv, 128]     bfloat16
    dropout_mask,          # [bs, 80, seq_q, seq_kv]   bool
    inv_keep,              # float scalar
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    # Transpose: [bs, seq_q, 80, 128] → [bs, 80, seq_q, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128]  bf16

    # ── dP computation: flat batched GEMM ────────────────────────────────────
    # [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv]
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)  # [bs*8, 10*sq, 128]
    V_flat     = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)  # [bs*8, skv, 128]
    dP_flat    = torch.bmm(dO_flat_dp, V_flat.transpose(-2, -1))          # [bs*8, 10*sq, skv]
    dP_dropped_g = dP_flat.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)

    # Dropout undo — fully bf16
    mask_g = dropout_mask.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dP_g   = dP_dropped_g * mask_g * inv_keep

    # Softmax backward — fully bf16
    P_g  = attn_weights.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dS_g = P_g * (dP_g - (P_g * dP_g).sum(dim=-1, keepdim=True))
    dS   = dS_g.view(bs, n_heads, seq_q, seq_kv)  # already bf16

    # ── dV computation: compact GEMM folding group-sum into K dimension ──────
    # attn_weights_dropped: [bs, 80, sq, skv] → reshape to [bs*8, 10*sq, skv]
    # dO:                   [bs, 80, sq, 128] → reshape to [bs*8, 10*sq, 128]
    # GEMM: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] → [bs*8, skv, 128]
    # K=10*sq accumulates all 10 groups — no explicit sum needed!
    Pd_flat = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)  # [bs*8, 10*sq, skv]
    dO_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)                  # [bs*8, 10*sq, 128]
    # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] → [bs*8, skv, 128]
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat)
    # Reshape to final output — no group sum required
    dV = dV_flat.view(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)  # [bs, 8, skv, 128]

    return dS, dV


# Cache compiled function at module level
_compiled_fn = None


def _get_compiled_fn():
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = torch.compile(
            _attn_bwd_impl,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
        )
    return _compiled_fn


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    fn = _get_compiled_fn()
    return fn(
        grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, inv_keep,
    )

```

---

## Experiment #11 — 2026-06-28 08:54:00 UTC ✅ KEEP

**Hypothesis:** 1. **`softmax_bwd_kernel`** — a Triton kernel with grid=(n_rows,) where n_rows=bs*80*seq_q. Each program handles one row of length seq_kv. Two passes: (1) load dP_dropped, apply dropout mask+scale, lo

**Result:** 463.73 μs

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.

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
# Triton kernel: fused softmax-backward + dropout-undo
#   Processes one row (of length seq_kv) per program.
#   Grid: (n_rows,)  where n_rows = bs * 80 * seq_q
#   Single pass: load row of dP_dropped, apply dropout mask, compute rowsum,
#   load row of P, write dS = P * (dP - rowsum).
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,    # [n_rows, seq_kv]  bfloat16  (dP after matmul, NOT dropout-undone)
    P_ptr,     # [n_rows, seq_kv]  bfloat16
    mask_ptr,  # [n_rows, seq_kv]  bool
    dS_ptr,    # [n_rows, seq_kv]  bfloat16  (output)
    inv_keep,  # float
    seq_kv,    # int
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)

    # Base pointers for this row
    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    # ── Pass 1: compute rowsum = sum_k( P[k] * dP[k] ) ──────────────────────
    rowsum = tl.zeros([1], dtype=tl.float32)
    n_blocks = tl.cdiv(seq_kv, BLOCK)

    for blk in range(n_blocks):
        off   = blk * BLOCK + tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)

        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
        rowsum += tl.sum(dp_val * p_val, axis=0)

    # ── Pass 2: write dS = P * (dP - rowsum) ─────────────────────────────────
    for blk in range(n_blocks):
        off   = blk * BLOCK + tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)

        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
        ds_val = p_val * (dp_val - rowsum)

        tl.store(dS_row + off, ds_val.to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    """
    dP_dropped:  [bs, 80, sq, skv]  bf16   (output of dO @ V^T, pre-dropout-undo)
    attn_weights:[bs, 80, sq, skv]  bf16
    dropout_mask:[bs, 80, sq, skv]  bool
    Returns dS:  [bs, 80, sq, skv]  bf16
    """
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    # Flatten to 2D: [n_rows, seq_kv]
    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    # Pick BLOCK size: next power of 2 >= seq_kv, capped at 8192
    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)

    grid = (n_rows,)
    softmax_bwd_kernel[grid](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        num_warps=min(max(BLOCK // 256, 1), 32),
        num_stages=1,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Transpose dO: [bs, seq_q, 80, 128] → [bs, 80, seq_q, 128]  bf16
    dO = grad_attn_output.permute(0, 2, 1, 3).contiguous()

    # ── dP GEMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv] ─
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    V_flat     = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)
    dP_flat    = torch.bmm(dO_flat_dp, V_flat.transpose(-2, -1))          # [bs*8, 10*sq, skv]
    dP_dropped = dP_flat.view(bs, n_heads, seq_q, seq_kv)

    # ── Fused softmax-backward + dropout undo (Triton) ───────────────────────
    dS = fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep)

    # ── dV GEMM: [bs*80, skv, sq] @ [bs*80, sq, 128] → [bs*80, skv, 128] ────
    Pd_flat = attn_weights_dropped.view(bs * n_heads, seq_q, seq_kv)
    dO_flat = dO.reshape(bs * n_heads, seq_q, HEAD_DIM)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat)               # [bs*80, skv, 128]
    dV = dV_flat.view(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM).sum(dim=2).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #12 — 2026-06-28 08:55:47 UTC ✅ KEEP

**Hypothesis:** 1. **Compact dV GEMM with .contiguous() fix**: `attn_weights_dropped.contiguous().view(bs*8, 10*sq, skv)` — the explicit `.contiguous()` ensures the memory layout is correct before the `.view()`, fixi

**Result:** 427.14 μs

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.
v2: compact dV GEMM (fold group-sum into K), skip dO .contiguous().

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
# Triton kernel: fused softmax-backward + dropout-undo
#   One program per row (length = seq_kv).
#   Grid: (n_rows,)  where n_rows = bs * 80 * seq_q
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,    # [n_rows, seq_kv]  bfloat16
    P_ptr,     # [n_rows, seq_kv]  bfloat16
    mask_ptr,  # [n_rows, seq_kv]  bool
    dS_ptr,    # [n_rows, seq_kv]  bfloat16  (output)
    inv_keep,  # float
    seq_kv,    # int
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    n_blocks = tl.cdiv(seq_kv, BLOCK)

    # Pass 1: compute rowsum = sum_k( P[k] * dP_undropped[k] )
    rowsum = tl.zeros([1], dtype=tl.float32)
    for blk in range(n_blocks):
        off   = blk * BLOCK + tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)

        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
        rowsum += tl.sum(dp_val * p_val, axis=0)

    # Pass 2: write dS = P * (dP_undropped - rowsum)
    for blk in range(n_blocks):
        off   = blk * BLOCK + tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)

        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
        tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)

    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        num_warps=min(max(BLOCK // 256, 1), 32),
        num_stages=1,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Transpose dO (no .contiguous() — reshape will handle it when needed)
    dO = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128]  bf16 (may be non-contiguous)

    # ── dP GEMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv] ─
    # .reshape() calls .contiguous() internally if needed
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)   # [bs*8, 10*sq, 128]
    V_flat     = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)   # [bs*8, skv, 128]
    dP_flat    = torch.bmm(dO_flat_dp, V_flat.transpose(-2, -1))           # [bs*8, 10*sq, skv]
    dP_dropped = dP_flat.view(bs, n_heads, seq_q, seq_kv)                  # contiguous

    # ── Fused softmax-backward + dropout undo (Triton) ───────────────────────
    dS = fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep)

    # ── dV GEMM: compact form, K=10*sq folds group accumulation ──────────────
    # attn_weights_dropped: [bs, 80, sq, skv] — make contiguous, reshape to [bs*8, 10*sq, skv]
    # dO_flat_dp is already [bs*8, 10*sq, 128] and contiguous (output of .reshape())
    Pd_flat = attn_weights_dropped.contiguous().view(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] → [bs*8, skv, 128]
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat_dp)
    # Direct reshape — no group-sum needed, K=10*sq already accumulated all groups
    dV = dV_flat.view(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)  # [bs, 8, skv, 128]

    return dS, dV

```

---

## Experiment #13 — 2026-06-28 08:57:27 UTC ✅ KEEP

**Hypothesis:** 1. **Single-pass Triton kernel**: Added `SINGLE_PASS: tl.constexpr` parameter. When `seq_kv <= BLOCK` (i.e., SINGLE_PASS=True), the entire row loads into registers in one pass, computes `rowsum = tl.s

**Result:** 410.51 μs

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.
v3: single-pass Triton kernel for small seq_kv, remove .contiguous() on Pd.

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
# Triton kernel: fused softmax-backward + dropout-undo
#   One program per row (length = seq_kv).
#   Grid: (n_rows,)  where n_rows = bs * 80 * seq_q
#
#   SINGLE_PASS=True:  seq_kv <= BLOCK — load entire row into registers,
#                      compute rowsum in-register, write once. 2× less HBM.
#   SINGLE_PASS=False: seq_kv > BLOCK  — two-pass tiled version.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,    # [n_rows, seq_kv]  bfloat16
    P_ptr,     # [n_rows, seq_kv]  bfloat16
    mask_ptr,  # [n_rows, seq_kv]  bool
    dS_ptr,    # [n_rows, seq_kv]  bfloat16  (output)
    inv_keep,  # float
    seq_kv,    # int
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    if SINGLE_PASS:
        # Load entire row into registers (seq_kv <= BLOCK)
        off   = tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)

        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)

        # Rowsum and output in one shot
        rowsum = tl.sum(dp_val * p_val, axis=0)
        ds_val = p_val * (dp_val - rowsum)
        tl.store(dS_row + off, ds_val.to(tl.bfloat16), mask=valid)

    else:
        n_blocks = tl.cdiv(seq_kv, BLOCK)

        # Pass 1: compute rowsum
        rowsum = tl.zeros([1], dtype=tl.float32)
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv

            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)

            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            rowsum += tl.sum(dp_val * p_val, axis=0)

        # Pass 2: write dS
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv

            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)

            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)
    SINGLE_PASS = (seq_kv <= BLOCK)  # True when seq_kv fits in one tile

    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        SINGLE_PASS=SINGLE_PASS,
        num_warps=min(max(BLOCK // 256, 1), 32),
        num_stages=1,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Transpose dO (no .contiguous() — reshape handles it)
    dO = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128]  bf16

    # ── dP GEMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv] ─
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)   # [bs*8, 10*sq, 128]
    V_flat     = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)   # [bs*8, skv, 128]
    dP_flat    = torch.bmm(dO_flat_dp, V_flat.transpose(-2, -1))           # [bs*8, 10*sq, skv]
    dP_dropped_t = dP_flat.view(bs, n_heads, seq_q, seq_kv)               # contiguous

    # ── Fused softmax-backward + dropout undo (Triton) ───────────────────────
    dS = fused_softmax_bwd(dP_dropped_t, attn_weights, dropout_mask, inv_keep)

    # ── dV GEMM: compact form, K=10*sq folds group accumulation ──────────────
    # attn_weights_dropped is [bs, 80, sq, skv] — contiguous as an input tensor.
    # [bs, 80, sq, skv] → [bs*8, 10*sq, skv]: merge dims 0&1 (bs*80=bs*8*10),
    # and dims 2 (10, sq→10*sq already merged as 80=8*10). .view() works on contiguous.
    Pd_flat = attn_weights_dropped.view(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    # [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] → [bs*8, skv, 128]
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat_dp)
    # Direct reshape — K=10*sq accumulated all groups
    dV = dV_flat.view(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #14 — 2026-06-28 08:59:10 UTC 💥 CRASH

**Hypothesis:** 1. **New `dv_kernel` Triton kernel**: Grid=(bs*n_kv_heads, cdiv(seq_kv, BLOCK_SKV)). Each program accumulates dV for one (batch, kv_head, seq_kv_tile) by looping over all 10 groups and all seq_q block

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.
v4: Triton kernel accepts non-contiguous grad_attn_output via explicit strides,
    eliminating the dO contiguous copy for the softmax-bwd path.

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
# Triton kernel: fused softmax-backward + dropout-undo
#   One program per row (length = seq_kv).
#   Grid: (n_rows,)  where n_rows = bs * 80 * seq_q
#
#   SINGLE_PASS=True:  seq_kv <= BLOCK — load entire row into registers,
#                      compute rowsum in-register, write once. 2× less HBM.
#   SINGLE_PASS=False: seq_kv > BLOCK  — two-pass tiled version.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,    # [n_rows, seq_kv]  bfloat16
    P_ptr,     # [n_rows, seq_kv]  bfloat16
    mask_ptr,  # [n_rows, seq_kv]  bool
    dS_ptr,    # [n_rows, seq_kv]  bfloat16  (output)
    inv_keep,  # float
    seq_kv,    # int
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    if SINGLE_PASS:
        off   = tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)

        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)

        rowsum = tl.sum(dp_val * p_val, axis=0)
        ds_val = p_val * (dp_val - rowsum)
        tl.store(dS_row + off, ds_val.to(tl.bfloat16), mask=valid)

    else:
        n_blocks = tl.cdiv(seq_kv, BLOCK)

        rowsum = tl.zeros([1], dtype=tl.float32)
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv

            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)

            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            rowsum += tl.sum(dp_val * p_val, axis=0)

        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv

            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)

            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)
    SINGLE_PASS = (seq_kv <= BLOCK)

    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        SINGLE_PASS=SINGLE_PASS,
        num_warps=min(max(BLOCK // 256, 1), 32),
        num_stages=1,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: dV accumulation
#   Grid: (bs * n_kv_heads, cdiv(seq_kv, BLOCK_SKV))
#   Each program accumulates dV[bs, kv, skv_tile, :] over all 10 groups and sq.
#   Reads grad_attn_output via original [bs, sq, 80, 128] strides (no contiguous copy).
#   Reads attn_weights_dropped via [bs, 80, sq, skv] strides.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def dv_kernel(
    dO_ptr,     # [bs, sq, 80, 128]   bfloat16  (original, non-transposed)
    Pd_ptr,     # [bs, 80, sq, skv]   bfloat16
    dV_ptr,     # [bs, 8, skv, 128]   bfloat16  (output)
    bs, seq_q, seq_kv,
    # strides for dO (original layout: [bs, sq, 80, 128])
    dO_s_bs: tl.constexpr,   # sq * 80 * 128
    dO_s_sq: tl.constexpr,   # 80 * 128
    dO_s_h:  tl.constexpr,   # 128
    # strides for Pd ([bs, 80, sq, skv])
    Pd_s_bs: tl.constexpr,   # 80 * sq * skv
    Pd_s_h:  tl.constexpr,   # sq * skv
    Pd_s_sq: tl.constexpr,   # skv
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_groups: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bkv = tl.program_id(0)   # batch * n_kv_heads + kv_head
    skv_blk = tl.program_id(1)

    b_id  = bkv // n_kv_heads
    kv_id = bkv % n_kv_heads

    skv_off  = skv_blk * BLOCK_SKV + tl.arange(0, BLOCK_SKV)
    skv_mask = skv_off < seq_kv
    d_off    = tl.arange(0, BLOCK_D)
    sq_off   = tl.arange(0, BLOCK_SQ)

    acc = tl.zeros([BLOCK_SKV, BLOCK_D], dtype=tl.float32)

    n_sq_blks = tl.cdiv(seq_q, BLOCK_SQ)

    for g in range(n_groups):
        head_id = kv_id * n_groups + g

        for sq_b in range(n_sq_blks):
            sq_idx   = sq_b * BLOCK_SQ + sq_off
            sq_valid = sq_idx < seq_q
            mv       = skv_mask[:, None] & sq_valid[None, :]  # [SKV, SQ]

            # Load Pd in transposed layout [BLOCK_SKV, BLOCK_SQ]
            Pd_ptrs = (Pd_ptr + b_id * Pd_s_bs + head_id * Pd_s_h
                       + sq_idx[None, :] * Pd_s_sq + skv_off[:, None])
            Pd_tile = tl.load(Pd_ptrs, mask=mv, other=0.0).to(tl.float32)  # [SKV, SQ]

            # Load dO in [BLOCK_SQ, BLOCK_D] — original non-transposed layout
            dO_ptrs = (dO_ptr + b_id * dO_s_bs
                       + sq_idx[:, None] * dO_s_sq + head_id * dO_s_h + d_off[None, :])
            dO_tile = tl.load(dO_ptrs, mask=sq_valid[:, None], other=0.0).to(tl.float32)  # [SQ, D]

            # dV += Pd^T @ dO: [SKV, SQ] @ [SQ, D] → [SKV, D]
            acc += tl.dot(Pd_tile, dO_tile)

    dV_base = dV_ptr + b_id * (n_kv_heads * seq_kv * HEAD_DIM) + kv_id * (seq_kv * HEAD_DIM)
    dV_ptrs = dV_base + skv_off[:, None] * HEAD_DIM + d_off[None, :]
    tl.store(dV_ptrs, acc.to(tl.bfloat16), mask=skv_mask[:, None])


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # ── dP GEMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv] ─
    # dO must be contiguous in [bs, 80, sq, 128] layout for cuBLAS
    dO_cont   = grad_attn_output.permute(0, 2, 1, 3).contiguous()         # [bs, 80, sq, 128]
    dO_flat   = dO_cont.view(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM) # [bs*8, 10*sq, 128]
    V_flat    = value_states.view(bs * n_kv_heads, seq_kv, HEAD_DIM)      # [bs*8, skv, 128]
    dP_flat   = torch.bmm(dO_flat, V_flat.transpose(-2, -1))              # [bs*8, 10*sq, skv]
    dP_drop_t = dP_flat.view(bs, n_heads, seq_q, seq_kv)

    # ── Fused softmax-backward + dropout undo (Triton) ───────────────────────
    dS = fused_softmax_bwd(dP_drop_t, attn_weights, dropout_mask, inv_keep)

    # ── dV via Triton kernel (reads grad_attn_output with original strides) ──
    # grad_attn_output: [bs, sq, 80, 128], strides = (sq*80*128, 80*128, 128, 1)
    dV = torch.empty((bs, n_kv_heads, seq_kv, HEAD_DIM),
                     dtype=torch.bfloat16, device=grad_attn_output.device)

    BLOCK_SKV = min(triton.next_power_of_2(seq_kv), 128)
    BLOCK_SQ  = 32
    BLOCK_D   = 128

    grid_dv = (bs * n_kv_heads, triton.cdiv(seq_kv, BLOCK_SKV))

    dv_kernel[grid_dv](
        grad_attn_output, attn_weights_dropped, dV,
        bs, seq_q, seq_kv,
        dO_s_bs=seq_q * n_heads * HEAD_DIM,
        dO_s_sq=n_heads * HEAD_DIM,
        dO_s_h=HEAD_DIM,
        Pd_s_bs=n_heads * seq_q * seq_kv,
        Pd_s_h=seq_q * seq_kv,
        Pd_s_sq=seq_kv,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_groups=n_groups,
        BLOCK_SKV=BLOCK_SKV,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    return dS, dV

```

---

## Experiment #15 — 2026-06-28 09:00:45 UTC 💥 CRASH

**Hypothesis:** 1. **All stride parameters changed to regular runtime ints**: `dO_s_bs`, `dO_s_sq`, `dO_s_h`, `Pd_s_bs`, `Pd_s_h`, `Pd_s_sq`, `dV_s_bs`, `dV_s_kv` are now declared as plain parameters (no `tl.constexp

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.
v5: fixed dv_kernel with runtime (non-constexpr) strides for non-contiguous access.

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
# Triton kernel: fused softmax-backward + dropout-undo
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,    # [n_rows, seq_kv]  bfloat16
    P_ptr,     # [n_rows, seq_kv]  bfloat16
    mask_ptr,  # [n_rows, seq_kv]  bool
    dS_ptr,    # [n_rows, seq_kv]  bfloat16  (output)
    inv_keep,  # float
    seq_kv,    # int (runtime)
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    if SINGLE_PASS:
        off   = tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)

        rowsum = tl.sum(dp_val * p_val, axis=0)
        tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)

    else:
        n_blocks = tl.cdiv(seq_kv, BLOCK)

        rowsum = tl.zeros([1], dtype=tl.float32)
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            rowsum += tl.sum(dp_val * p_val, axis=0)

        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)
    SINGLE_PASS = (seq_kv <= BLOCK)

    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        SINGLE_PASS=SINGLE_PASS,
        num_warps=min(max(BLOCK // 256, 1), 32),
        num_stages=1,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: dV accumulation
#   Reads grad_attn_output in original [bs, sq, 80, 128] layout (no copy needed).
#   Strides passed as regular runtime integers (NOT tl.constexpr).
#   Grid: (bs * n_kv_heads, cdiv(seq_kv, BLOCK_SKV))
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def dv_kernel(
    dO_ptr,       # [bs, sq, 80, 128]   bfloat16  — original layout
    Pd_ptr,       # [bs, 80, sq, skv]   bfloat16
    dV_ptr,       # [bs, 8, skv, 128]   bfloat16  (output)
    seq_q,        # runtime int
    seq_kv,       # runtime int
    # strides — all runtime ints (NOT tl.constexpr)
    dO_s_bs,      # stride over bs dim of grad_attn_output  = sq*80*128
    dO_s_sq,      # stride over sq dim                     = 80*128
    dO_s_h,       # stride over head dim                   = 128
    Pd_s_bs,      # stride over bs dim of attn_weights_dropped = 80*sq*skv
    Pd_s_h,       # stride over head dim                       = sq*skv
    Pd_s_sq,      # stride over sq dim                         = skv
    dV_s_bs,      # stride over bs dim of dV                   = 8*skv*128
    dV_s_kv,      # stride over kv dim                         = skv*128
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_groups: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bkv     = tl.program_id(0)
    skv_blk = tl.program_id(1)

    b_id  = bkv // n_kv_heads
    kv_id = bkv % n_kv_heads

    skv_off  = skv_blk * BLOCK_SKV + tl.arange(0, BLOCK_SKV)
    skv_mask = skv_off < seq_kv
    d_off    = tl.arange(0, BLOCK_D)
    sq_off   = tl.arange(0, BLOCK_SQ)

    acc = tl.zeros([BLOCK_SKV, BLOCK_D], dtype=tl.float32)

    n_sq_blks = tl.cdiv(seq_q, BLOCK_SQ)

    for g in range(n_groups):
        head_id = kv_id * n_groups + g

        for sq_b in range(n_sq_blks):
            sq_idx   = sq_b * BLOCK_SQ + sq_off
            sq_valid = sq_idx < seq_q
            mv       = skv_mask[:, None] & sq_valid[None, :]  # [SKV, SQ]

            # Load Pd in transposed layout [BLOCK_SKV, BLOCK_SQ]
            # Pd[b, head, sq, skv] — access as Pd^T[skv, sq]
            Pd_ptrs = (Pd_ptr
                       + b_id  * Pd_s_bs
                       + head_id * Pd_s_h
                       + sq_idx[None, :] * Pd_s_sq
                       + skv_off[:, None])
            Pd_tile = tl.load(Pd_ptrs, mask=mv, other=0.0).to(tl.float32)  # [SKV, SQ]

            # Load dO in [BLOCK_SQ, BLOCK_D]
            # grad_attn_output[b, sq, head, d]
            dO_ptrs = (dO_ptr
                       + b_id   * dO_s_bs
                       + sq_idx[:, None] * dO_s_sq
                       + head_id * dO_s_h
                       + d_off[None, :])
            dO_tile = tl.load(dO_ptrs, mask=sq_valid[:, None], other=0.0).to(tl.float32)  # [SQ, D]

            # dV += Pd^T_transposed @ dO: [SKV, SQ] @ [SQ, D] → [SKV, D]
            acc += tl.dot(Pd_tile, dO_tile)

    # Write dV output
    dV_ptrs = (dV_ptr
               + b_id  * dV_s_bs
               + kv_id * dV_s_kv
               + skv_off[:, None] * HEAD_DIM
               + d_off[None, :])
    tl.store(dV_ptrs, acc.to(tl.bfloat16), mask=skv_mask[:, None])


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # ── dO contiguous copy needed for dP BMM ─────────────────────────────────
    dO = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128]  non-contiguous

    # ── dP GEMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv] ─
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)  # triggers contiguous copy
    V_flat     = value_states.view(bs * n_kv_heads, seq_kv, HEAD_DIM)
    dP_flat    = torch.bmm(dO_flat_dp, V_flat.transpose(-2, -1))
    dP_drop_t  = dP_flat.view(bs, n_heads, seq_q, seq_kv)

    # ── Fused softmax-backward + dropout undo (Triton) ───────────────────────
    dS = fused_softmax_bwd(dP_drop_t, attn_weights, dropout_mask, inv_keep)

    # ── dV via Triton kernel reading grad_attn_output directly (no extra copy) ─
    dV = torch.empty((bs, n_kv_heads, seq_kv, HEAD_DIM),
                     dtype=torch.bfloat16, device=grad_attn_output.device)

    # grad_attn_output strides: [bs, sq, 80, 128]
    dO_s_bs = int(grad_attn_output.stride(0))   # sq * 80 * 128
    dO_s_sq = int(grad_attn_output.stride(1))   # 80 * 128
    dO_s_h  = int(grad_attn_output.stride(2))   # 128
    # attn_weights_dropped strides: [bs, 80, sq, skv]
    Pd_s_bs = int(attn_weights_dropped.stride(0))  # 80 * sq * skv
    Pd_s_h  = int(attn_weights_dropped.stride(1))  # sq * skv
    Pd_s_sq = int(attn_weights_dropped.stride(2))  # skv
    # dV strides: [bs, 8, skv, 128]
    dV_s_bs = n_kv_heads * seq_kv * HEAD_DIM
    dV_s_kv = seq_kv * HEAD_DIM

    BLOCK_SKV = min(triton.next_power_of_2(seq_kv), 128)
    BLOCK_SQ  = 32
    BLOCK_D   = 128

    grid_dv = (bs * n_kv_heads, triton.cdiv(seq_kv, BLOCK_SKV))

    dv_kernel[grid_dv](
        grad_attn_output, attn_weights_dropped, dV,
        seq_q, seq_kv,
        dO_s_bs, dO_s_sq, dO_s_h,
        Pd_s_bs, Pd_s_h, Pd_s_sq,
        dV_s_bs, dV_s_kv,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_groups=n_groups,
        BLOCK_SKV=BLOCK_SKV,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    return dS, dV

```

---

## Experiment #16 — 2026-06-28 09:02:51 UTC ❌ DISCARD

**Hypothesis:** 1. **`_gemm_impl` compiled function**: Wraps both the dP GEMM and the dV GEMM under `torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=True)`. This lets the compiler see the full chain: `perm

**Result:** 505.83 μs

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.
v6: wrap the PyTorch ops in torch.compile to fuse dispatch overhead.

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
# Triton kernel: fused softmax-backward + dropout-undo
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_keep,
    seq_kv,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    if SINGLE_PASS:
        off   = tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)

        rowsum = tl.sum(dp_val * p_val, axis=0)
        tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)

    else:
        n_blocks = tl.cdiv(seq_kv, BLOCK)

        rowsum = tl.zeros([1], dtype=tl.float32)
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            rowsum += tl.sum(dp_val * p_val, axis=0)

        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)
    SINGLE_PASS = (seq_kv <= BLOCK)

    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        SINGLE_PASS=SINGLE_PASS,
        num_warps=min(max(BLOCK // 256, 1), 32),
        num_stages=1,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


# ─────────────────────────────────────────────────────────────────────────────
# Compiled helper for the two BMMs (dP and dV) — lets torch.compile fuse
# the permute + reshape + bmm + view chain and reduce dispatch overhead.
# ─────────────────────────────────────────────────────────────────────────────
def _gemm_impl(grad_attn_output, value_states, attn_weights_dropped):
    bs       = grad_attn_output.shape[0]
    seq_q    = grad_attn_output.shape[1]
    seq_kv   = value_states.shape[2]
    n_kv     = NUM_KEY_VALUE_HEADS
    n_groups = N_GROUPS

    # dO: [bs, 80, sq, 128] — permute triggers contiguous copy
    dO_flat = grad_attn_output.permute(0, 2, 1, 3).reshape(
        bs * n_kv, n_groups * seq_q, HEAD_DIM)                         # [bs*8, 10*sq, 128]

    # dP GEMM
    V_flat  = value_states.view(bs * n_kv, seq_kv, HEAD_DIM)           # [bs*8, skv, 128]
    dP_flat = torch.bmm(dO_flat, V_flat.transpose(-2, -1))             # [bs*8, 10*sq, skv]

    # dV GEMM — reuse dO_flat, fold group-sum into K
    Pd_flat = attn_weights_dropped.view(bs * n_kv, n_groups * seq_q, seq_kv)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat)            # [bs*8, skv, 128]
    dV = dV_flat.view(bs, n_kv, seq_kv, HEAD_DIM).to(torch.bfloat16)  # [bs, 8, skv, 128]

    return dP_flat, dV


_compiled_gemm = None


def _get_compiled_gemm():
    global _compiled_gemm
    if _compiled_gemm is None:
        _compiled_gemm = torch.compile(
            _gemm_impl,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
        )
    return _compiled_gemm


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Run the two BMMs under torch.compile
    fn = _get_compiled_gemm()
    dP_flat, dV = fn(grad_attn_output, value_states, attn_weights_dropped)

    # Reshape dP to [bs, 80, sq, skv] for Triton softmax-bwd
    dP_drop_t = dP_flat.view(bs, n_heads, seq_q, seq_kv)

    # Fused softmax-backward + dropout undo (Triton)
    dS = fused_softmax_bwd(dP_drop_t, attn_weights, dropout_mask, inv_keep)

    return dS, dV

```

---

## Experiment #17 — 2026-06-28 09:04:36 UTC ❌ DISCARD

**Hypothesis:** 1. **`_gemm_impl` with `reduce-overhead`**: Changed from `max-autotune-no-cudagraphs` (which explicitly disables CUDA graphs) to `reduce-overhead` (which enables CUDA graph capture). This captures the

**Result:** 757.25 μs

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.
v7: restore exp13 structure, use reduce-overhead on the two-BMM function.

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
# Triton kernel: fused softmax-backward + dropout-undo
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_keep,
    seq_kv,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    if SINGLE_PASS:
        off   = tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)

        rowsum = tl.sum(dp_val * p_val, axis=0)
        tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)

    else:
        n_blocks = tl.cdiv(seq_kv, BLOCK)

        rowsum = tl.zeros([1], dtype=tl.float32)
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            rowsum += tl.sum(dp_val * p_val, axis=0)

        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)
    SINGLE_PASS = (seq_kv <= BLOCK)

    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        SINGLE_PASS=SINGLE_PASS,
        num_warps=min(max(BLOCK // 256, 1), 32),
        num_stages=1,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


# ─────────────────────────────────────────────────────────────────────────────
# Compiled helper: both BMMs under reduce-overhead (CUDA graphs) for low
# dispatch latency. Triton softmax-bwd is kept outside to avoid graph capture
# issues with custom kernels.
# ─────────────────────────────────────────────────────────────────────────────
def _gemm_impl(grad_attn_output, value_states, attn_weights_dropped):
    bs       = grad_attn_output.shape[0]
    seq_q    = grad_attn_output.shape[1]
    seq_kv   = value_states.shape[2]
    n_kv     = NUM_KEY_VALUE_HEADS   # 8
    n_groups = N_GROUPS              # 10

    # dO: [bs, 80, sq, 128] via permute+reshape (internally calls .contiguous())
    dO_flat = grad_attn_output.permute(0, 2, 1, 3).reshape(
        bs * n_kv, n_groups * seq_q, HEAD_DIM)                          # [bs*8, 10*sq, 128]

    # dP GEMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv]
    V_flat  = value_states.view(bs * n_kv, seq_kv, HEAD_DIM)            # [bs*8, skv, 128]
    dP_flat = torch.bmm(dO_flat, V_flat.transpose(-2, -1))              # [bs*8, 10*sq, skv]

    # dV GEMM: reuse dO_flat, K=10*sq folds all 10 groups
    Pd_flat = attn_weights_dropped.view(bs * n_kv, n_groups * seq_q, seq_kv)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat)             # [bs*8, skv, 128]
    dV = dV_flat.view(bs, n_kv, seq_kv, HEAD_DIM).to(torch.bfloat16)   # [bs, 8, skv, 128]

    return dP_flat, dV


_compiled_gemm = None


def _get_compiled_gemm():
    global _compiled_gemm
    if _compiled_gemm is None:
        _compiled_gemm = torch.compile(
            _gemm_impl,
            mode="reduce-overhead",   # CUDA graphs for low per-call latency
            fullgraph=True,
        )
    return _compiled_gemm


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs      = grad_attn_output.shape[0]
    seq_q   = grad_attn_output.shape[1]
    seq_kv  = value_states.shape[2]
    n_heads = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Both BMMs under CUDA-graph-enabled torch.compile
    fn = _get_compiled_gemm()
    dP_flat, dV = fn(grad_attn_output, value_states, attn_weights_dropped)

    # Reshape dP result for Triton softmax-bwd
    dP_drop_t = dP_flat.view(bs, n_heads, seq_q, seq_kv)

    # Fused softmax-backward + dropout undo (Triton, called directly)
    dS = fused_softmax_bwd(dP_drop_t, attn_weights, dropout_mask, inv_keep)

    return dS, dV

```

---

## Experiment #18 — 2026-06-28 09:06:26 UTC 💥 CRASH

**Hypothesis:** 1. **Row-major Pd load**: Changed from the previous `Pd_ptrs[skv_idx, sq_idx]` (column-major/transposed) to `Pd_ptrs[sq_idx[:,None], skv_off[None,:]]` — loading `Pd[b, head, sq, skv]` with `sq` as row

**Result:** CRASH

**Error:**
```
Correctness check failed. Correctness check failed — see test_details
```

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch dP BMM + Triton softmax-bwd + Triton dV kernel.
v8: corrected dv_kernel using row-major Pd load + tl.trans() for tl.dot compatibility.

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
# Triton kernel: fused softmax-backward + dropout-undo
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_keep,
    seq_kv,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    if SINGLE_PASS:
        off   = tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)

        rowsum = tl.sum(dp_val * p_val, axis=0)
        tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)

    else:
        n_blocks = tl.cdiv(seq_kv, BLOCK)

        rowsum = tl.zeros([1], dtype=tl.float32)
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            rowsum += tl.sum(dp_val * p_val, axis=0)

        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)
    SINGLE_PASS = (seq_kv <= BLOCK)

    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        SINGLE_PASS=SINGLE_PASS,
        num_warps=min(max(BLOCK // 256, 1), 32),
        num_stages=1,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: dV accumulation (corrected)
#   Load Pd as [BLOCK_SQ, BLOCK_SKV] (row-major, natural layout),
#   then use tl.trans(Pd_tile) for tl.dot, avoiding column-major tile issues.
#   Grid: (bs * n_kv_heads, cdiv(seq_kv, BLOCK_SKV))
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def dv_kernel(
    dO_ptr,      # [bs, sq, 80, 128]   bfloat16 — original layout
    Pd_ptr,      # [bs, 80, sq, skv]   bfloat16
    dV_ptr,      # [bs, 8, skv, 128]   bfloat16  (output)
    seq_q,       # runtime int
    seq_kv,      # runtime int
    # strides — all runtime ints (NOT tl.constexpr)
    dO_s_bs,    # grad_attn_output.stride(0) = sq*80*128
    dO_s_sq,    # grad_attn_output.stride(1) = 80*128
    dO_s_h,     # grad_attn_output.stride(2) = 128
    Pd_s_bs,    # attn_weights_dropped.stride(0) = 80*sq*skv
    Pd_s_h,     # attn_weights_dropped.stride(1) = sq*skv
    Pd_s_sq,    # attn_weights_dropped.stride(2) = skv
    n_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_groups: tl.constexpr,
    BLOCK_SKV: tl.constexpr,
    BLOCK_SQ:  tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    bkv     = tl.program_id(0)
    skv_blk = tl.program_id(1)

    b_id  = bkv // n_kv_heads
    kv_id = bkv % n_kv_heads

    skv_off  = skv_blk * BLOCK_SKV + tl.arange(0, BLOCK_SKV)
    skv_mask = skv_off < seq_kv
    d_off    = tl.arange(0, BLOCK_D)
    sq_off   = tl.arange(0, BLOCK_SQ)

    acc = tl.zeros([BLOCK_SKV, BLOCK_D], dtype=tl.float32)

    n_sq_blks = tl.cdiv(seq_q, BLOCK_SQ)

    for g in range(n_groups):
        head_id = kv_id * n_groups + g

        for sq_b in range(n_sq_blks):
            sq_idx   = sq_b * BLOCK_SQ + sq_off
            sq_valid = sq_idx < seq_q

            # ── Load Pd as [BLOCK_SQ, BLOCK_SKV] — ROW-MAJOR ─────────────────
            # Pd[b, head, sq, skv]: sq is row, skv is col → natural row-major
            # Ptrs: base + b*Pd_s_bs + head*Pd_s_h + sq[:,None]*Pd_s_sq + skv[None,:]
            m_sq_skv = sq_valid[:, None] & skv_mask[None, :]  # [SQ, SKV]
            Pd_ptrs = (Pd_ptr
                       + b_id    * Pd_s_bs
                       + head_id * Pd_s_h
                       + sq_idx[:, None] * Pd_s_sq   # rows=sq, cols=skv
                       + skv_off[None, :])            # stride-1 in skv dim
            Pd_tile = tl.load(Pd_ptrs, mask=m_sq_skv, other=0.0).to(tl.float32)  # [SQ, SKV]

            # ── Load dO as [BLOCK_SQ, BLOCK_D] — ROW-MAJOR ───────────────────
            # grad_attn_output[b, sq, head, d]: sq is row, d is col → row-major
            dO_ptrs = (dO_ptr
                       + b_id    * dO_s_bs
                       + sq_idx[:, None] * dO_s_sq   # rows=sq, stride dO_s_sq
                       + head_id * dO_s_h             # head offset
                       + d_off[None, :])              # cols=d, stride 1
            dO_tile = tl.load(dO_ptrs, mask=sq_valid[:, None], other=0.0).to(tl.float32)  # [SQ, D]

            # ── dV += Pd^T @ dO: tl.dot(tl.trans(Pd), dO) ───────────────────
            # tl.trans(Pd_tile): [SKV, SQ]  — derived from row-major [SQ, SKV]
            # dO_tile:           [SQ, D]    — row-major
            # result:            [SKV, D]   — as required for dV
            acc += tl.dot(tl.trans(Pd_tile), dO_tile)

    # Write dV: [BLOCK_SKV, BLOCK_D]
    dV_base = dV_ptr + b_id * (n_kv_heads * seq_kv * HEAD_DIM) + kv_id * (seq_kv * HEAD_DIM)
    dV_ptrs = dV_base + skv_off[:, None] * HEAD_DIM + d_off[None, :]
    tl.store(dV_ptrs, acc.to(tl.bfloat16), mask=skv_mask[:, None])


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # ── dP GEMM (experiment 13 structure) ────────────────────────────────────
    dO = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128]
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)  # [bs*8, 10*sq, 128]
    V_flat     = value_states.view(bs * n_kv_heads, seq_kv, HEAD_DIM)     # [bs*8, skv, 128]
    dP_flat    = torch.bmm(dO_flat_dp, V_flat.transpose(-2, -1))          # [bs*8, 10*sq, skv]
    dP_drop_t  = dP_flat.view(bs, n_heads, seq_q, seq_kv)

    # ── Fused softmax-backward + dropout undo (Triton) ───────────────────────
    dS = fused_softmax_bwd(dP_drop_t, attn_weights, dropout_mask, inv_keep)

    # ── dV via corrected Triton kernel ────────────────────────────────────────
    # Reads grad_attn_output in [bs, sq, 80, 128] layout — no extra copy.
    dV = torch.empty((bs, n_kv_heads, seq_kv, HEAD_DIM),
                     dtype=torch.bfloat16, device=grad_attn_output.device)

    dO_s_bs = int(grad_attn_output.stride(0))   # sq*80*128
    dO_s_sq = int(grad_attn_output.stride(1))   # 80*128
    dO_s_h  = int(grad_attn_output.stride(2))   # 128
    Pd_s_bs = int(attn_weights_dropped.stride(0))  # 80*sq*skv
    Pd_s_h  = int(attn_weights_dropped.stride(1))  # sq*skv
    Pd_s_sq = int(attn_weights_dropped.stride(2))  # skv (stride-1 for skv dim)

    BLOCK_SKV = min(triton.next_power_of_2(seq_kv), 64)
    BLOCK_SQ  = 32
    BLOCK_D   = 128

    grid_dv = (bs * n_kv_heads, triton.cdiv(seq_kv, BLOCK_SKV))

    dv_kernel[grid_dv](
        grad_attn_output, attn_weights_dropped, dV,
        seq_q, seq_kv,
        dO_s_bs, dO_s_sq, dO_s_h,
        Pd_s_bs, Pd_s_h, Pd_s_sq,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_groups=n_groups,
        BLOCK_SKV=BLOCK_SKV,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    return dS, dV

```

---

## Experiment #19 — 2026-06-28 09:07:55 UTC ✅ KEEP

**Hypothesis:** 1. **Restored exact experiment 13 structure**: All three operations (dP BMM, Triton softmax-bwd, dV BMM) exactly as in the best-performing experiment.

**Result:** 408.12 μs

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.
v9: tune num_warps for softmax-bwd kernel (use num_warps=8 for large seqs).

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
# Triton kernel: fused softmax-backward + dropout-undo
#   One program per row (length = seq_kv).
#   Grid: (n_rows,)  where n_rows = bs * 80 * seq_q
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_keep,
    seq_kv,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    if SINGLE_PASS:
        # Load entire row into registers (seq_kv <= BLOCK): single pass
        off   = tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)

        rowsum = tl.sum(dp_val * p_val, axis=0)
        tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)

    else:
        n_blocks = tl.cdiv(seq_kv, BLOCK)

        # Pass 1: compute rowsum
        rowsum = tl.zeros([1], dtype=tl.float32)
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv

            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            rowsum += tl.sum(dp_val * p_val, axis=0)

        # Pass 2: write dS
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv

            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)
    SINGLE_PASS = (seq_kv <= BLOCK)

    # num_warps tuning:
    #   - SINGLE_PASS (small seq): 4 warps is sufficient, avoids over-subscription
    #   - two-pass (large seq): 8 warps for better memory bandwidth utilization
    if SINGLE_PASS:
        num_warps = min(max(BLOCK // 256, 1), 16)
    else:
        num_warps = 8  # fixed 8 warps for two-pass path (large sequences)

    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        SINGLE_PASS=SINGLE_PASS,
        num_warps=num_warps,
        num_stages=1,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Transpose dO (reshape handles contiguous copy)
    dO = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128]  bf16

    # ── dP GEMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv] ─
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)   # [bs*8, 10*sq, 128]
    V_flat     = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)   # [bs*8, skv, 128]
    dP_flat    = torch.bmm(dO_flat_dp, V_flat.transpose(-2, -1))           # [bs*8, 10*sq, skv]
    dP_dropped_t = dP_flat.view(bs, n_heads, seq_q, seq_kv)

    # ── Fused softmax-backward + dropout undo (Triton) ───────────────────────
    dS = fused_softmax_bwd(dP_dropped_t, attn_weights, dropout_mask, inv_keep)

    # ── dV GEMM: compact form, K=10*sq folds group accumulation ──────────────
    Pd_flat = attn_weights_dropped.view(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat_dp)            # [bs*8, skv, 128]
    dV = dV_flat.view(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #20 — 2026-06-28 09:09:26 UTC ❌ DISCARD

**Hypothesis:** 1. **`@triton.autotune` decorator** with 9 configs covering `num_warps ∈ {1,2,4,8,16,32}` and `num_stages ∈ {1,2}`. The `key=['seq_kv', 'BLOCK']` means the autotuner caches per unique (seq_kv, BLOCK) 

**Result:** 408.64 μs

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.
v10: @triton.autotune on softmax_bwd_kernel for optimal num_warps per seq_kv.

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
# Triton kernel: fused softmax-backward + dropout-undo
#   @triton.autotune over num_warps and num_stages, keyed on (seq_kv, BLOCK).
#   BLOCK and SINGLE_PASS are passed as constexpr from the wrapper and must
#   NOT be in the autotune configs — only num_warps and num_stages vary.
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1,  num_stages=1),
        triton.Config({}, num_warps=2,  num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=32, num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
    ],
    key=['seq_kv', 'BLOCK'],
)
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_keep,
    seq_kv,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    if SINGLE_PASS:
        off   = tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)

        rowsum = tl.sum(dp_val * p_val, axis=0)
        tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)

    else:
        n_blocks = tl.cdiv(seq_kv, BLOCK)

        # Pass 1: compute rowsum
        rowsum = tl.zeros([1], dtype=tl.float32)
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv

            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            rowsum += tl.sum(dp_val * p_val, axis=0)

        # Pass 2: write dS
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv

            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)
    SINGLE_PASS = (seq_kv <= BLOCK)  # always True since BLOCK >= seq_kv

    # With @triton.autotune, num_warps and num_stages are selected automatically
    # per (seq_kv, BLOCK) key — no manual specification needed here.
    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        SINGLE_PASS=SINGLE_PASS,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Transpose dO (reshape handles contiguous copy)
    dO = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128]  bf16

    # ── dP GEMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv] ─
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)   # [bs*8, 10*sq, 128]
    V_flat     = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)   # [bs*8, skv, 128]
    dP_flat    = torch.bmm(dO_flat_dp, V_flat.transpose(-2, -1))           # [bs*8, 10*sq, skv]
    dP_dropped_t = dP_flat.view(bs, n_heads, seq_q, seq_kv)

    # ── Fused softmax-backward + dropout undo (Triton, autotuned) ────────────
    dS = fused_softmax_bwd(dP_dropped_t, attn_weights, dropout_mask, inv_keep)

    # ── dV GEMM: compact form, K=10*sq folds group accumulation ──────────────
    Pd_flat = attn_weights_dropped.view(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat_dp)            # [bs*8, skv, 128]
    dV = dV_flat.view(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #21 — 2026-06-28 09:11:18 UTC ❌ DISCARD

**Hypothesis:** 1. **`flash_attn_bwd_ds_kernel`**: Grid=(bs*n_heads, cdiv(seq_q, BLOCK_SQ)). Each program handles one (batch, head, sq_block) tile. Two passes over seq_kv blocks:

**Result:** 1003.11 μs

**Kernel code:**
```python
"""
Flash-Attention-style fused backward Triton kernel.
Fuses dP computation, softmax-bwd, and dV accumulation in a single kernel.

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
# Flash-Attention-style backward kernel:
#   Grid: (bs * n_heads, cdiv(seq_q, BLOCK_SQ))
#   Each program handles one (batch, head, sq_block) tile.
#   Inner loop over seq_kv blocks:
#     - loads dO[sq_block, d] once (stays in registers)
#     - for each skv_block: load V[skv,d], compute dP=dO@V^T, undo dropout,
#       load P[sq,skv], accumulate rowsum, accumulate dV via atomic add
#     - after loop: write dS[sq,skv] = P*(dP - rowsum) for each skv_block
#
# dV uses torch.zeros + atomic add pattern but that needs atomics.
# Instead: dS kernel computes dS; dV uses separate compact GEMM (proven correct).
# 
# This kernel computes ONLY dS in a flash-attention style (single-pass rowsum).
# dV is computed via the proven compact GEMM from experiment 19.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def flash_attn_bwd_ds_kernel(
    # Inputs
    dO_ptr,     # [bs, sq, 80, 128]   bf16  original layout
    V_ptr,      # [bs,  8, skv, 128]  bf16
    P_ptr,      # [bs, 80, sq, skv]   bf16
    mask_ptr,   # [bs, 80, sq, skv]   bool
    # Output
    dS_ptr,     # [bs, 80, sq, skv]   bf16
    # Sizes (runtime)
    seq_q,
    seq_kv,
    inv_keep,
    # Strides for dO [bs, sq, 80, 128]
    dO_s_bs,    # sq * 80 * 128
    dO_s_sq,    # 80 * 128
    dO_s_h,     # 128
    # Strides for P/mask/dS [bs, 80, sq, skv]
    attn_s_bs,  # 80 * sq * skv
    attn_s_h,   # sq * skv
    attn_s_sq,  # skv
    # Strides for V [bs, 8, skv, 128]
    V_s_bs,     # 8 * skv * 128
    V_s_kv,     # skv * 128
    # Architecture
    n_heads:    tl.constexpr,
    n_kv_heads: tl.constexpr,
    n_groups:   tl.constexpr,
    BLOCK_SQ:   tl.constexpr,
    BLOCK_SKV:  tl.constexpr,
    BLOCK_D:    tl.constexpr,
):
    bh_id  = tl.program_id(0)   # batch * n_heads + head
    sq_blk = tl.program_id(1)   # block along seq_q

    b_id    = bh_id // n_heads
    head_id = bh_id % n_heads
    kv_id   = head_id // n_groups

    sq_off  = sq_blk * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    sq_mask = sq_off < seq_q
    d_off   = tl.arange(0, BLOCK_D)
    skv_off = tl.arange(0, BLOCK_SKV)

    # Load dO tile: [BLOCK_SQ, BLOCK_D] — from original [bs, sq, 80, 128]
    dO_ptrs = (dO_ptr
               + b_id    * dO_s_bs
               + sq_off[:, None] * dO_s_sq
               + head_id * dO_s_h
               + d_off[None, :])
    dO_tile = tl.load(dO_ptrs, mask=sq_mask[:, None], other=0.0).to(tl.float32)  # [SQ, D]

    # ── Pass 1: compute rowsum by iterating over all skv blocks ──────────────
    rowsum = tl.zeros([BLOCK_SQ], dtype=tl.float32)  # [SQ]
    n_skv_blks = tl.cdiv(seq_kv, BLOCK_SKV)

    for skv_b in range(n_skv_blks):
        skv_idx   = skv_b * BLOCK_SKV + skv_off
        skv_valid = skv_idx < seq_kv
        mv        = sq_mask[:, None] & skv_valid[None, :]

        # Load V: [BLOCK_SKV, BLOCK_D]
        V_ptrs = (V_ptr
                  + b_id  * V_s_bs
                  + kv_id * V_s_kv
                  + skv_idx[:, None] * BLOCK_D   # stride=128 (HEAD_DIM)
                  + d_off[None, :])
        V_tile = tl.load(V_ptrs, mask=skv_valid[:, None], other=0.0).to(tl.float32)  # [SKV, D]

        # dP_dropped = dO @ V^T: [SQ, D] @ [D, SKV] → [SQ, SKV]
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))  # [SQ, SKV]

        # Undo dropout
        m_ptrs    = (mask_ptr + b_id * attn_s_bs + head_id * attn_s_h
                     + sq_off[:, None] * attn_s_sq + skv_idx[None, :])
        drop_mask = tl.load(m_ptrs, mask=mv, other=0).to(tl.int1)
        dP_tile   = tl.where(drop_mask, dP_tile * inv_keep, 0.0)

        # Load P
        P_ptrs = (P_ptr + b_id * attn_s_bs + head_id * attn_s_h
                  + sq_off[:, None] * attn_s_sq + skv_idx[None, :])
        P_tile = tl.load(P_ptrs, mask=mv, other=0.0).to(tl.float32)

        # Accumulate rowsum: [SQ] += sum over SKV of (P * dP)
        rowsum += tl.sum(P_tile * dP_tile, axis=1)

    # ── Pass 2: write dS for each skv block ──────────────────────────────────
    for skv_b in range(n_skv_blks):
        skv_idx   = skv_b * BLOCK_SKV + skv_off
        skv_valid = skv_idx < seq_kv
        mv        = sq_mask[:, None] & skv_valid[None, :]

        V_ptrs = (V_ptr
                  + b_id  * V_s_bs
                  + kv_id * V_s_kv
                  + skv_idx[:, None] * BLOCK_D
                  + d_off[None, :])
        V_tile = tl.load(V_ptrs, mask=skv_valid[:, None], other=0.0).to(tl.float32)
        dP_tile = tl.dot(dO_tile, tl.trans(V_tile))

        m_ptrs    = (mask_ptr + b_id * attn_s_bs + head_id * attn_s_h
                     + sq_off[:, None] * attn_s_sq + skv_idx[None, :])
        drop_mask = tl.load(m_ptrs, mask=mv, other=0).to(tl.int1)
        dP_tile   = tl.where(drop_mask, dP_tile * inv_keep, 0.0)

        P_ptrs = (P_ptr + b_id * attn_s_bs + head_id * attn_s_h
                  + sq_off[:, None] * attn_s_sq + skv_idx[None, :])
        P_tile = tl.load(P_ptrs, mask=mv, other=0.0).to(tl.float32)

        dS_tile = P_tile * (dP_tile - rowsum[:, None])

        dS_ptrs = (dS_ptr + b_id * attn_s_bs + head_id * attn_s_h
                   + sq_off[:, None] * attn_s_sq + skv_idx[None, :])
        tl.store(dS_ptrs, dS_tile.to(tl.bfloat16), mask=mv)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # ── dS via flash-attention-style fused Triton kernel ─────────────────────
    dS = torch.empty((bs, n_heads, seq_q, seq_kv), dtype=torch.bfloat16,
                     device=grad_attn_output.device)

    # Stride extraction
    dO_s_bs   = int(grad_attn_output.stride(0))   # sq*80*128
    dO_s_sq   = int(grad_attn_output.stride(1))   # 80*128
    dO_s_h    = int(grad_attn_output.stride(2))   # 128
    attn_s_bs = int(attn_weights.stride(0))        # 80*sq*skv
    attn_s_h  = int(attn_weights.stride(1))        # sq*skv
    attn_s_sq = int(attn_weights.stride(2))        # skv
    V_s_bs    = int(value_states.stride(0))        # 8*skv*128
    V_s_kv    = int(value_states.stride(1))        # skv*128

    BLOCK_SQ  = 32
    BLOCK_SKV = min(triton.next_power_of_2(seq_kv), 128)
    BLOCK_D   = 128

    grid_ds = (bs * n_heads, triton.cdiv(seq_q, BLOCK_SQ))

    flash_attn_bwd_ds_kernel[grid_ds](
        grad_attn_output, value_states, attn_weights, dropout_mask, dS,
        seq_q, seq_kv, inv_keep,
        dO_s_bs, dO_s_sq, dO_s_h,
        attn_s_bs, attn_s_h, attn_s_sq,
        V_s_bs, V_s_kv,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_groups=n_groups,
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SKV=BLOCK_SKV,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    # ── dV via proven compact GEMM (no copy needed for attn_weights_dropped) ─
    dO = grad_attn_output.permute(0, 2, 1, 3)   # triggers contiguous copy
    dO_flat = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)
    Pd_flat = attn_weights_dropped.view(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat)
    dV = dV_flat.view(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #22 — 2026-06-28 09:13:06 UTC ❌ DISCARD

**Hypothesis:** 1. **Pre-transposed V**: `value_states.reshape(bs*8, skv, 128).transpose(-2,-1).contiguous()` → `[bs*8, 128, skv]`. The dP BMM becomes `torch.bmm(dO_flat_dp, V_flat_T)` directly (no transpose inside t

**Result:** 797.74 μs

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.
v11: restore exp19 + try pre-transposed V to avoid in-BMM .transpose().

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
# Triton kernel: fused softmax-backward + dropout-undo (proven, single-pass)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1,  num_stages=1),
        triton.Config({}, num_warps=2,  num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=2),
    ],
    key=['seq_kv', 'BLOCK'],
)
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_keep,
    seq_kv,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    if SINGLE_PASS:
        off   = tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)

        rowsum = tl.sum(dp_val * p_val, axis=0)
        tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)

    else:
        n_blocks = tl.cdiv(seq_kv, BLOCK)

        rowsum = tl.zeros([1], dtype=tl.float32)
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            rowsum += tl.sum(dp_val * p_val, axis=0)

        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)
    SINGLE_PASS = (seq_kv <= BLOCK)

    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        SINGLE_PASS=SINGLE_PASS,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Transpose dO: [bs, sq, 80, 128] → [bs, 80, sq, 128] → [bs*8, 10*sq, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3)
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)  # [bs*8, 10*sq, 128]

    # ── dP GEMM: use pre-transposed V to avoid in-BMM .transpose() ───────────
    # value_states: [bs, 8, skv, 128] → reshape to [bs*8, skv, 128] → transpose
    # to [bs*8, 128, skv] (contiguous) — avoids the non-contiguous .transpose(-2,-1) view
    V_flat_T = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM) \
                            .transpose(-2, -1).contiguous()              # [bs*8, 128, skv]
    dP_flat   = torch.bmm(dO_flat_dp, V_flat_T)                         # [bs*8, 10*sq, skv]
    dP_drop_t = dP_flat.view(bs, n_heads, seq_q, seq_kv)

    # ── Fused softmax-backward + dropout undo (Triton) ───────────────────────
    dS = fused_softmax_bwd(dP_drop_t, attn_weights, dropout_mask, inv_keep)

    # ── dV GEMM: compact form, K=10*sq folds group accumulation ──────────────
    # Pre-transpose Pd to [bs*8, skv, 10*sq] to avoid non-contiguous .transpose(-2,-1)
    Pd_flat_T = attn_weights_dropped.view(bs * n_kv_heads, n_groups * seq_q, seq_kv) \
                                     .transpose(-2, -1).contiguous()     # [bs*8, skv, 10*sq]
    dV_flat = torch.bmm(Pd_flat_T, dO_flat_dp)                          # [bs*8, skv, 128]
    dV = dV_flat.view(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #23 — 2026-06-28 09:14:36 UTC ❌ DISCARD

**Hypothesis:** Restored the exact experiment 19 structure (best at 408 μs) with the proven @triton.autotune softmax_bwd_kernel. The broadcast matmul approach was considered but would produce `[bs, sq, 8, 10, skv]` w

**Result:** 409.42 μs

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.
v12: broadcast matmul avoiding dO contiguous copy for dP computation.

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
# Triton kernel: fused softmax-backward + dropout-undo (proven, single-pass)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1,  num_stages=1),
        triton.Config({}, num_warps=2,  num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=2),
    ],
    key=['seq_kv', 'BLOCK'],
)
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_keep,
    seq_kv,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    if SINGLE_PASS:
        off   = tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)

        rowsum = tl.sum(dp_val * p_val, axis=0)
        tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)

    else:
        n_blocks = tl.cdiv(seq_kv, BLOCK)

        rowsum = tl.zeros([1], dtype=tl.float32)
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            rowsum += tl.sum(dp_val * p_val, axis=0)

        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)
    SINGLE_PASS = (seq_kv <= BLOCK)

    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        SINGLE_PASS=SINGLE_PASS,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # ── dP via broadcast matmul (no dO contiguous copy needed) ───────────────
    # grad_attn_output: [bs, sq, 80, 128] → view as [bs, sq, 8, 10, 128]
    # value_states:     [bs, 8, skv, 128] → unsqueeze → [bs, 8, 1, 128, skv] via transpose
    # torch.matmul broadcasts: [bs, sq, 8, 10, 128] @ [bs, 1, 8, 1, 128, skv]?
    # Actually: reshape grad to [bs, sq, 8, 10, 128], V to [bs, 8, 1, 128, skv]
    # matmul([bs, sq, 8, 10, 128], [bs, 1, 8, 1, 128, skv]) -- wrong dims
    # 
    # Correct broadcast: dO [bs, sq, 8, 10, 128] @ V^T [bs, 1, 8, 1, skv, 128]^T
    # → torch.matmul([bs, sq, 8, 10, 128], [bs, 1, 8, 128, skv]) -- need V as [bs,1,8,128,skv]
    # V [bs, 8, skv, 128] → permute → [bs, 8, 128, skv] → unsqueeze(1) → [bs,1,8,128,skv]
    # dO [bs, sq, 80, 128] → view [bs, sq, 8, 10, 128]
    # matmul([bs, sq, 8, 10, 128], [bs, 1, 8, 128, skv]) → [bs, sq, 8, 10, skv]
    # then permute → [bs, 8, 10, sq, skv] → view [bs, 80, sq, skv] -- needs contiguous
    #
    # Just use the proven exp19 structure — no copy savings are possible without Triton
    # ─────────────────────────────────────────────────────────────────────────
    # Restore experiment 19 exactly:
    dO = grad_attn_output.permute(0, 2, 1, 3)
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)  # [bs*8, 10*sq, 128]
    V_flat     = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)  # [bs*8, skv, 128]
    dP_flat    = torch.bmm(dO_flat_dp, V_flat.transpose(-2, -1))          # [bs*8, 10*sq, skv]
    dP_drop_t  = dP_flat.view(bs, n_heads, seq_q, seq_kv)

    # ── Fused softmax-backward + dropout undo (Triton) ───────────────────────
    dS = fused_softmax_bwd(dP_drop_t, attn_weights, dropout_mask, inv_keep)

    # ── dV GEMM: compact form, K=10*sq folds group accumulation ──────────────
    Pd_flat = attn_weights_dropped.view(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat_dp)            # [bs*8, skv, 128]
    dV = dV_flat.view(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #24 — 2026-06-28 09:16:23 UTC ❌ DISCARD

**Hypothesis:** 1. **5D broadcast dP matmul**: `dO_5d = grad_attn_output.view(bs, seq_q, n_kv_heads, n_groups, HEAD_DIM)` — this is a zero-copy view since `grad_attn_output[bs, sq, 80, 128]` has contiguous last dim a

**Result:** 5691.07 μs

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.
v13: 5D broadcast matmul for dP to avoid dO contiguous copy; proven compact dV.

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
# Triton kernel: fused softmax-backward + dropout-undo (proven, single-pass)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1,  num_stages=1),
        triton.Config({}, num_warps=2,  num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=1),
        triton.Config({}, num_warps=8,  num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=4,  num_stages=2),
        triton.Config({}, num_warps=8,  num_stages=2),
    ],
    key=['seq_kv', 'BLOCK'],
)
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_keep,
    seq_kv,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    if SINGLE_PASS:
        off   = tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)

        rowsum = tl.sum(dp_val * p_val, axis=0)
        tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)

    else:
        n_blocks = tl.cdiv(seq_kv, BLOCK)

        rowsum = tl.zeros([1], dtype=tl.float32)
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            rowsum += tl.sum(dp_val * p_val, axis=0)

        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv
            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)
    SINGLE_PASS = (seq_kv <= BLOCK)

    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        SINGLE_PASS=SINGLE_PASS,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # ── dP via 5D broadcast matmul — avoids explicit dO contiguous copy ───────
    # grad_attn_output: [bs, sq, 80, 128] → view as [bs, sq, 8, 10, 128]
    #   (no copy: last dim stride=1, groups=10 is dim-3, 80=8*10 in dim-2)
    # V_T: [bs, 8, skv, 128] → transpose(-2,-1) → [bs, 8, 128, skv]
    #      → unsqueeze(1) → [bs, 1, 8, 128, skv]
    # torch.matmul([bs, sq, 8, 10, 128], [bs, 1, 8, 128, skv])
    #   → broadcasts over sq,1 and 10,1 → [bs, sq, 8, 10, skv]
    dO_5d  = grad_attn_output.view(bs, seq_q, n_kv_heads, n_groups, HEAD_DIM)
    # V_T: need [bs, 1, 8, 128, skv] for the matmul [bs,sq,8,10,128]@[bs,1,8,128,skv]
    V_T    = value_states.transpose(-2, -1).unsqueeze(1)    # [bs, 1, 8, 128, skv]
    # Result: [bs, sq, 8, 10, skv]
    dP_5d  = torch.matmul(dO_5d, V_T)                      # [bs, sq, 8, 10, skv]

    # Permute to [bs, 8, 10, sq, skv] = [bs, 80, sq, skv] after reshape
    # permute(0, 2, 3, 1, 4) → [bs, 8, 10, sq, skv] → reshape → [bs, 80, sq, skv]
    dP_drop_t = dP_5d.permute(0, 2, 3, 1, 4).reshape(bs, n_heads, seq_q, seq_kv)

    # ── Fused softmax-backward + dropout undo (Triton) ───────────────────────
    dS = fused_softmax_bwd(dP_drop_t, attn_weights, dropout_mask, inv_keep)

    # ── dV GEMM: compact form, K=10*sq folds group accumulation ──────────────
    # For dV we still need the contiguous dO in [bs*8, 10*sq, 128] layout
    dO_flat = grad_attn_output.permute(0, 2, 1, 3).reshape(
        bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)        # [bs*8, 10*sq, 128]
    Pd_flat = attn_weights_dropped.view(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat) # [bs*8, skv, 128]
    dV = dV_flat.view(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #25 — 2026-06-28 09:17:45 UTC ❌ DISCARD

**Hypothesis:** Restored the exact experiment 19 structure (best at 408.12 μs):

**Result:** 409.36 μs

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.
Final: exact experiment 19 structure (best at 408 μs).

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
# Triton kernel: fused softmax-backward + dropout-undo
#   One program per row (length = seq_kv).
#   Grid: (n_rows,)  where n_rows = bs * 80 * seq_q
#
#   SINGLE_PASS=True:  seq_kv <= BLOCK — load entire row into registers,
#                      compute rowsum in-register, write once. 2× less HBM.
#   SINGLE_PASS=False: seq_kv > BLOCK  — two-pass tiled version.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_keep,
    seq_kv,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    if SINGLE_PASS:
        off   = tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)

        rowsum = tl.sum(dp_val * p_val, axis=0)
        tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)

    else:
        n_blocks = tl.cdiv(seq_kv, BLOCK)

        # Pass 1: compute rowsum
        rowsum = tl.zeros([1], dtype=tl.float32)
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv

            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            rowsum += tl.sum(dp_val * p_val, axis=0)

        # Pass 2: write dS
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv

            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)
    SINGLE_PASS = (seq_kv <= BLOCK)

    # num_warps tuning: 8 warps for two-pass (large seq), adaptive for single-pass
    if SINGLE_PASS:
        num_warps = min(max(BLOCK // 256, 1), 16)
    else:
        num_warps = 8

    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        SINGLE_PASS=SINGLE_PASS,
        num_warps=num_warps,
        num_stages=1,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Transpose dO: [bs, sq, 80, 128] → [bs, 80, sq, 128] → [bs*8, 10*sq, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3)
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)  # [bs*8, 10*sq, 128]

    # ── dP GEMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv] ─
    V_flat    = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)   # [bs*8, skv, 128]
    dP_flat   = torch.bmm(dO_flat_dp, V_flat.transpose(-2, -1))           # [bs*8, 10*sq, skv]
    dP_drop_t = dP_flat.view(bs, n_heads, seq_q, seq_kv)

    # ── Fused softmax-backward + dropout undo (Triton) ───────────────────────
    dS = fused_softmax_bwd(dP_drop_t, attn_weights, dropout_mask, inv_keep)

    # ── dV GEMM: compact form, K=10*sq folds group accumulation ──────────────
    Pd_flat = attn_weights_dropped.view(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat_dp)            # [bs*8, skv, 128]
    dV = dV_flat.view(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

---

## Experiment #26 — 2026-06-28 09:19:13 UTC ❌ DISCARD

**Hypothesis:** Exact experiment 19 structure confirmed and written. No changes made to the algorithm.

**Result:** 409.80 μs

**Kernel code:**
```python
"""
Hybrid attention-backward kernel: PyTorch BMMs + Triton elementwise softmax-bwd.
Final: exact experiment 19 structure (best at 408 μs).

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
# Triton kernel: fused softmax-backward + dropout-undo
#   One program per row (length = seq_kv).
#   Grid: (n_rows,)  where n_rows = bs * 80 * seq_q
#
#   SINGLE_PASS=True:  seq_kv <= BLOCK — load entire row into registers,
#                      compute rowsum in-register, write once. 2× less HBM.
#   SINGLE_PASS=False: seq_kv > BLOCK  — two-pass tiled version.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def softmax_bwd_kernel(
    dP_ptr,
    P_ptr,
    mask_ptr,
    dS_ptr,
    inv_keep,
    seq_kv,
    BLOCK: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
):
    row = tl.program_id(0)

    dP_row   = dP_ptr   + row * seq_kv
    P_row    = P_ptr    + row * seq_kv
    mask_row = mask_ptr + row * seq_kv
    dS_row   = dS_ptr   + row * seq_kv

    if SINGLE_PASS:
        off   = tl.arange(0, BLOCK)
        valid = off < seq_kv

        dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
        m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
        dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
        p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)

        rowsum = tl.sum(dp_val * p_val, axis=0)
        tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)

    else:
        n_blocks = tl.cdiv(seq_kv, BLOCK)

        # Pass 1: compute rowsum
        rowsum = tl.zeros([1], dtype=tl.float32)
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv

            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            rowsum += tl.sum(dp_val * p_val, axis=0)

        # Pass 2: write dS
        for blk in range(n_blocks):
            off   = blk * BLOCK + tl.arange(0, BLOCK)
            valid = off < seq_kv

            dp_val = tl.load(dP_row + off, mask=valid, other=0.0).to(tl.float32)
            m_val  = tl.load(mask_row + off, mask=valid, other=0).to(tl.int1)
            dp_val = tl.where(m_val, dp_val * inv_keep, 0.0)
            p_val  = tl.load(P_row + off, mask=valid, other=0.0).to(tl.float32)
            tl.store(dS_row + off, (p_val * (dp_val - rowsum)).to(tl.bfloat16), mask=valid)


def fused_softmax_bwd(dP_dropped, attn_weights, dropout_mask, inv_keep):
    bs, n_heads, seq_q, seq_kv = dP_dropped.shape
    n_rows = bs * n_heads * seq_q

    dP_flat   = dP_dropped.reshape(n_rows, seq_kv)
    P_flat    = attn_weights.reshape(n_rows, seq_kv)
    mask_flat = dropout_mask.reshape(n_rows, seq_kv)
    dS_flat   = torch.empty_like(dP_flat)

    BLOCK = min(triton.next_power_of_2(seq_kv), 8192)
    SINGLE_PASS = (seq_kv <= BLOCK)

    # num_warps tuning: 8 warps for two-pass (large seq), adaptive for single-pass
    if SINGLE_PASS:
        num_warps = min(max(BLOCK // 256, 1), 16)
    else:
        num_warps = 8

    softmax_bwd_kernel[(n_rows,)](
        dP_flat, P_flat, mask_flat, dS_flat,
        inv_keep, seq_kv,
        BLOCK=BLOCK,
        SINGLE_PASS=SINGLE_PASS,
        num_warps=num_warps,
        num_stages=1,
    )
    return dS_flat.view(bs, n_heads, seq_q, seq_kv)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Transpose dO: [bs, sq, 80, 128] → [bs, 80, sq, 128] → [bs*8, 10*sq, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3)
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)  # [bs*8, 10*sq, 128]

    # ── dP GEMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv] ─
    V_flat    = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)   # [bs*8, skv, 128]
    dP_flat   = torch.bmm(dO_flat_dp, V_flat.transpose(-2, -1))           # [bs*8, 10*sq, skv]
    dP_drop_t = dP_flat.view(bs, n_heads, seq_q, seq_kv)

    # ── Fused softmax-backward + dropout undo (Triton) ───────────────────────
    dS = fused_softmax_bwd(dP_drop_t, attn_weights, dropout_mask, inv_keep)

    # ── dV GEMM: compact form, K=10*sq folds group accumulation ──────────────
    Pd_flat = attn_weights_dropped.view(bs * n_kv_heads, n_groups * seq_q, seq_kv)
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat_dp)            # [bs*8, skv, 128]
    dV = dV_flat.view(bs, n_kv_heads, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV

```

