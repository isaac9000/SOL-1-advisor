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
