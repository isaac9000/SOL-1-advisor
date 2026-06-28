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
