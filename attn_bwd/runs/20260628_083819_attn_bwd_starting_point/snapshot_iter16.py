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
