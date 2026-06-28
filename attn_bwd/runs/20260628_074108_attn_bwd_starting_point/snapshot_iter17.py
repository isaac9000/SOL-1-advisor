"""
Optimized attention-backward kernel — PyTorch with torch.compile(max-autotune-no-cudagraphs,
dynamic=True) + Exp 15 inductor config + repeat_interleave bmm for dP GEMM.

Based on Experiment 15 (best at 533.42 μs). Tries repeat_interleave to give cuBLAS
a contiguous [bs*80, skv, 128] V tensor for a standard batched GEMM.

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

# Exp 15 proven inductor config flags only
try:
    import torch._inductor.config as inductor_config
    inductor_config.max_autotune_gemm = True
    inductor_config.coordinate_descent_tuning = True
    inductor_config.coordinate_descent_check_all_directions = True
except Exception:
    pass  # fall back gracefully if unavailable


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

    # ── dP = dO @ V^T via repeat_interleave + bmm ────────────────────────────
    # Expand V: [bs, 8, skv, 128] -> [bs*80, skv, 128] via repeat_interleave
    # repeat_interleave repeats each row N_GROUPS=10 times along batch dim,
    # giving a contiguous tensor cuBLAS can use as a standard batched GEMM.
    V_flat = value_states.reshape(bs * NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM)
    V_rep  = torch.repeat_interleave(V_flat, N_GROUPS, dim=0)  # [bs*80, skv, 128]

    # dO_flat: [bs*80, sq, 128]
    dO_flat = dO_grouped.reshape(bs * NUM_ATTENTION_HEADS, seq_q, HEAD_DIM)

    # bmm: [bs*80, sq, 128] @ [bs*80, 128, skv] -> [bs*80, sq, skv]
    dP_dropped = torch.bmm(dO_flat, V_rep.transpose(1, 2)).reshape(
        bs, NUM_ATTENTION_HEADS, seq_q, seq_kv
    )

    # ── Dropout backward + softmax backward entirely in bfloat16 ─────────────
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob

    P    = attn_weights   # [bs, 80, sq, skv] bf16
    dPP  = dP_bf16 * P
    dS   = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    # ── dV via einsum (bf16, sums over groups) — proven fast ─────────────────
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


# Compile with max-autotune-no-cudagraphs + dynamic=True (Exp 15 settings)
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
