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
