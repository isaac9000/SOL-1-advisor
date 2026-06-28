# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-28 08:18:05 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 3427.91 μs

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

## Experiment #2 — 2026-06-28 08:19:50 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 2208.86 μs

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

## Experiment #3 — 2026-06-28 08:21:31 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 628.38 μs

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

## Experiment #4 — 2026-06-28 08:23:32 UTC ✅ KEEP

**Hypothesis:** **

**Result:** 518.44 μs

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

---

## Experiment #5 — 2026-06-28 08:25:03 UTC ❌ DISCARD

**Hypothesis:** **

**Result:** 551.64 μs

**Kernel code:**
```python
"""
Hybrid attention-backward: cuBLAS BMM for matmuls + Triton fused elementwise for softmax bwd.
Optimized with CUDA streams to overlap independent dP and dV BMMs.

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

# Pre-create CUDA streams at module load time to avoid per-call overhead
_stream_dp = None
_stream_dv = None

def _get_streams():
    global _stream_dp, _stream_dv
    if _stream_dp is None:
        _stream_dp = torch.cuda.Stream()
        _stream_dv = torch.cuda.Stream()
    return _stream_dp, _stream_dv


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
    # Prepare grouped dO layout (shared by both BMMs)
    # [bs, sq, 80, 128] -> [bs, sq, 8, 10, 128] -> [bs, 8, 10, sq, 128]
    # -> [bs*8, 10*sq, 128]
    # This is on the default stream before forking.
    # =========================================================================
    dO_grouped = grad_attn_output.view(bs, seq_q, n_kv_heads, n_groups, head_dim)
    dO_grouped = dO_grouped.permute(0, 2, 3, 1, 4).contiguous()
    dO_gq = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, head_dim)

    V_2d  = value_states.reshape(bs * n_kv_heads, seq_kv, head_dim)
    Pd_gq = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # Get pre-created streams
    stream_dp, stream_dv = _get_streams()

    current_stream = torch.cuda.current_stream()

    # =========================================================================
    # Stream 1 (stream_dp): compute dP̃ BMM -> softmax bwd -> dS
    # =========================================================================
    with torch.cuda.stream(stream_dp):
        # Make stream_dp wait for the default stream to finish dO_gq prep
        stream_dp.wait_stream(current_stream)

        # BMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] -> [bs*8, 10*sq, skv]
        dP_raw_gq = torch.bmm(dO_gq, V_2d.transpose(1, 2))
        dP_raw = dP_raw_gq.reshape(bs, n_heads, seq_q, seq_kv)

        # Fused elementwise: dropout correction + softmax backward
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
    # Stream 2 (stream_dv): compute dV BMM (independent of stream_dp)
    # =========================================================================
    with torch.cuda.stream(stream_dv):
        # Make stream_dv wait for the default stream to finish dO_gq prep
        stream_dv.wait_stream(current_stream)

        # BMM: P^T @ dO: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
        dV = torch.bmm(Pd_gq.transpose(1, 2), dO_gq)
        dV = dV.reshape(bs, n_kv_heads, seq_kv, head_dim).to(torch.bfloat16)

    # =========================================================================
    # Synchronize both streams back to the default stream
    # =========================================================================
    current_stream.wait_stream(stream_dp)
    current_stream.wait_stream(stream_dv)

    return dS, dV

```

