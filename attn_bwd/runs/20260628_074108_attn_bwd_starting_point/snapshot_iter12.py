"""
Optimized attention-backward kernel — PyTorch with torch.compile(max-autotune-no-cudagraphs,
dynamic=True) and torch.matmul broadcast for GQA (no einsum), softmax-backward in bfloat16.

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

    # ── dP = dO @ V^T via broadcasting matmul ────────────────────────────────
    # dO_grouped:         [bs, 8, 10, sq,  128]
    # value_states:       [bs, 8,  1, skv, 128]  (unsqueeze at dim=2)
    # V^T:                [bs, 8,  1, 128, skv]
    # matmul broadcasts:  [bs, 8, 10, sq,  skv]
    # PyTorch uses cublasGemmStridedBatchedEx with batch_stride=0 for broadcast dim
    V_t = value_states.unsqueeze(2).transpose(-1, -2)  # [bs, 8, 1, 128, skv]
    dP_dropped_grouped = torch.matmul(dO_grouped, V_t)  # [bs, 8, 10, sq, skv]

    # Reshape back to [bs, 80, sq, skv]  bf16
    dP_dropped = dP_dropped_grouped.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Dropout backward + softmax backward entirely in bfloat16 ─────────────
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob

    P    = attn_weights   # [bs, 80, sq, skv] bf16
    dPP  = dP_bf16 * P
    dS   = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)

    # ── dV via broadcasting matmul + group sum ────────────────────────────────
    # Pd_grouped: [bs, 8, 10, sq, skv]
    # dO_grouped: [bs, 8, 10, sq, 128]
    # Pd^T @ dO: [bs, 8, 10, skv, sq] @ [bs, 8, 10, sq, 128] -> [bs, 8, 10, skv, 128]
    # Then sum over groups (dim=2) -> [bs, 8, skv, 128]
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    dV = torch.matmul(
        Pd_grouped.transpose(-1, -2),  # [bs, 8, 10, skv, sq]
        dO_grouped,                    # [bs, 8, 10, sq,  128]
    ).sum(dim=2).to(torch.bfloat16)   # [bs, 8, skv, 128]

    return dS, dV


# dynamic=True: single compiled version for all 16 benchmark shapes
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
