"""
torch.compile-accelerated attention-backward kernel.

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
    dO,               # [bs, 80, seq_q, 128]      float32
    attn_weights,     # [bs, 80, seq_q, seq_kv]   bfloat16
    attn_weights_dropped,  # [bs, 80, seq_q, seq_kv]  bfloat16
    value_states,     # [bs,  8, seq_kv, 128]     bfloat16
    dropout_mask,     # [bs, 80, seq_q, seq_kv]   bool
    inv_keep,         # float scalar
    bs, seq_q, seq_kv,
):
    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10

    # ── dP computation: dO @ V^T ─────────────────────────────────────────────
    # Reshape to exploit GQA structure: avoid full [bs,80,skv,128] expansion.
    # dO:    [bs, 8, 10, sq, 128]
    # V:     [bs, 8,  1, skv, 128]  (broadcast over groups)
    # dP:    [bs, 8, 10, sq, skv]

    dO_g  = dO.view(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)
    V_g   = value_states.unsqueeze(2)   # [bs, 8, 1, skv, 128]

    # Cast V to float32 for matmul
    V_g_f = V_g.to(torch.float32)

    # dP_dropped: [bs, 8, 10, sq, skv]
    dP_dropped_g = torch.matmul(dO_g, V_g_f.transpose(-2, -1))

    # Reshape dropout_mask and attn_weights to group layout
    mask_g  = dropout_mask.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dP_g    = dP_dropped_g * mask_g * inv_keep

    P_g = attn_weights.view(bs, n_kv_heads, n_groups, seq_q, seq_kv).to(torch.float32)

    # Softmax backward: dS = P * (dP - rowsum(P * dP))
    PdP_g   = P_g * dP_g
    rowsum  = PdP_g.sum(dim=-1, keepdim=True)   # [bs, 8, 10, sq, 1]
    dS_g    = P_g * (dP_g - rowsum)             # [bs, 8, 10, sq, skv]

    dS = dS_g.view(bs, n_heads, seq_q, seq_kv).to(torch.bfloat16)

    # ── dV computation: Pd^T @ dO ─────────────────────────────────────────────
    # attn_weights_dropped: [bs, 8, 10, sq, skv]
    # dO:                   [bs, 8, 10, sq, 128]
    # dV_per_group:         [bs, 8, 10, skv, 128] → sum over groups dim

    Pd_g = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv).to(torch.float32)
    # dV_g: [bs, 8, 10, skv, 128]
    dV_g = torch.matmul(Pd_g.transpose(-2, -1), dO_g)  # Pd^T @ dO
    # Sum over the 10 groups
    dV   = dV_g.sum(dim=2).to(torch.bfloat16)          # [bs, 8, skv, 128]

    return dS, dV


# Cache compiled function at module level
_compiled_fn = None


def _get_compiled_fn():
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = torch.compile(
            _attn_bwd_impl,
            mode="max-autotune",
            fullgraph=True,
        )
    return _compiled_fn


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    # Transpose dO: [bs, seq_q, 80, 128] → [bs, 80, seq_q, 128] and cast to f32
    dO = grad_attn_output.transpose(1, 2).contiguous().to(torch.float32)

    fn = _get_compiled_fn()
    return fn(
        dO, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, inv_keep,
        bs, seq_q, seq_kv,
    )
