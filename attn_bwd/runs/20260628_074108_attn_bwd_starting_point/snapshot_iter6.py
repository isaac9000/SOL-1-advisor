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
