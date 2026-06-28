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
