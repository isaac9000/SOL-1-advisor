"""
Optimized attention-backward kernel — PyTorch with torch.compile and GQA-native ops.

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


def _attn_bwd_impl(
    grad_attn_output,   # [bs, seq_q, 80, 128]  bf16
    attn_weights,       # [bs, 80, seq_q, seq_kv]  bf16
    attn_weights_dropped,  # [bs, 80, seq_q, seq_kv]  bf16
    value_states,       # [bs, 8, seq_kv, 128]  bf16
    dropout_mask,       # [bs, 80, seq_q, seq_kv]  bool
    attention_dropout,  # float scalar
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 80, seq_q, 128]
    # Then reshape to [bs, 8, 10, seq_q, 128] for GQA-grouped ops
    dO_heads = grad_attn_output.transpose(1, 2)  # [bs, 80, seq_q, 128] bf16
    dO_grouped = dO_heads.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM)
    # [bs, 8, 10, seq_q, 128]

    # ── Compute dP = dO @ V^T (GQA-native, no materialization of 80-head V) ──
    # value_states: [bs, 8, seq_kv, 128]
    # dO_grouped:   [bs, 8, 10, seq_q, 128]
    # dP_grouped:   [bs, 8, 10, seq_q, seq_kv]
    # einsum: 'bgiqd,bgkd->bgiqk'  (g=groups=10, i=sq, k=skv, d=dim)
    # Use bmm: reshape to [bs*8*10, sq, 128] @ [bs*8*10, 128, skv]
    dO_flat = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS * N_GROUPS, seq_q, HEAD_DIM).to(torch.float32)
    # Expand V for batched matmul: [bs, 8, seq_kv, 128] -> [bs, 8, 10, seq_kv, 128]
    V_grouped = value_states.unsqueeze(2).expand(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_kv, HEAD_DIM)
    V_flat = V_grouped.reshape(bs * NUM_KEY_VALUE_HEADS * N_GROUPS, seq_kv, HEAD_DIM).to(torch.float32)

    # dP_dropped_flat: [bs*8*10, sq, skv]
    dP_dropped_flat = torch.bmm(dO_flat, V_flat.transpose(-2, -1))
    dP_dropped = dP_dropped_flat.reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Dropout backward ──────────────────────────────────────────────────────
    if attention_dropout > 0.0:
        scale = 1.0 / (1.0 - attention_dropout)
        dP = dP_dropped * dropout_mask.to(torch.float32) * scale
    else:
        dP = dP_dropped

    # ── Softmax backward: dS = P * (dP - rowsum(dP * P)) ─────────────────────
    P = attn_weights.to(torch.float32)   # [bs, 80, sq, skv]
    dPP = dP * P
    dS = P * (dP - dPP.sum(dim=-1, keepdim=True))
    dS = dS.to(torch.bfloat16)

    # ── Compute dV (GQA-native) ───────────────────────────────────────────────
    # attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    # dO_grouped: [bs, 8, 10, sq, 128]
    # dV[b,g,k,d] = sum_i sum_grp  Pd[b,g,grp,i,k] * dO[b,g,grp,i,d]
    # = einsum('bgiqk,bgiqd->bgkd')
    Pd_grouped = attn_weights_dropped.reshape(bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv)
    # Use einsum for clarity and fusion
    # dV: [bs, 8, skv, 128]
    # We want: dV[b,g,k,d] = sum_{i,grp} Pd[b,g,grp,i,k] * dO[b,g,grp,i,d]
    # Merge grp*sq into one dim for bmm
    Pd_flat2 = Pd_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, seq_kv).to(torch.float32)
    dO_flat2  = dO_grouped.reshape(bs * NUM_KEY_VALUE_HEADS, N_GROUPS * seq_q, HEAD_DIM).to(torch.float32)
    # dV_flat: [bs*8, skv, 128]
    dV_flat = torch.bmm(Pd_flat2.transpose(-2, -1), dO_flat2)
    dV = dV_flat.reshape(bs, NUM_KEY_VALUE_HEADS, seq_kv, HEAD_DIM).to(torch.bfloat16)

    return dS, dV


# Compile once at module level
_compiled_attn_bwd = torch.compile(_attn_bwd_impl, mode="reduce-overhead")


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    return _compiled_attn_bwd(
        grad_attn_output,
        attn_weights,
        attn_weights_dropped,
        value_states,
        dropout_mask,
        attention_dropout,
    )
