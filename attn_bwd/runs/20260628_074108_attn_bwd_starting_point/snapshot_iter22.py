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
