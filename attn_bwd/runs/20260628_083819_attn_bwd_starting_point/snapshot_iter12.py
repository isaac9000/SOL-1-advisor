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
