"""
torch.compile-accelerated attention-backward kernel (optimized v2).

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
    grad_attn_output,      # [bs, seq_q, 80, 128]      bfloat16  (raw, un-transposed)
    attn_weights,          # [bs, 80, seq_q, seq_kv]   bfloat16
    attn_weights_dropped,  # [bs, 80, seq_q, seq_kv]   bfloat16
    value_states,          # [bs,  8, seq_kv, 128]     bfloat16
    dropout_mask,          # [bs, 80, seq_q, seq_kv]   bool
    inv_keep,              # float scalar
    bs, seq_q, seq_kv,
):
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    # Transpose inside compiled fn: [bs, seq_q, 80, 128] → [bs, 80, seq_q, 128]
    # then view as group layout [bs, 8, 10, sq, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128]  bf16
    dO_g = dO.reshape(bs, n_kv_heads, n_groups, seq_q, HEAD_DIM)  # [bs,8,10,sq,128] bf16

    # V: [bs, 8, 1, skv, 128]  (broadcast over groups)
    V_g = value_states.unsqueeze(2)  # [bs, 8, 1, skv, 128]  bf16

    # ── dP computation: dO @ V^T ─────────────────────────────────────────────
    # dP_dropped: [bs, 8, 10, sq, skv]  — keep bf16 for speed
    dP_dropped_g = torch.matmul(dO_g, V_g.transpose(-2, -1))  # bf16 @ bf16

    # Dropout undo
    mask_g = dropout_mask.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dP_g   = dP_dropped_g * mask_g * inv_keep  # still bf16

    # Softmax backward in float32 for numerical accuracy
    P_g    = attn_weights.view(bs, n_kv_heads, n_groups, seq_q, seq_kv).to(torch.float32)
    dP_g_f = dP_g.to(torch.float32)
    # Fused single expression: compiler can fuse into one kernel
    dS_g   = P_g * (dP_g_f - (P_g * dP_g_f).sum(dim=-1, keepdim=True))
    dS     = dS_g.view(bs, n_heads, seq_q, seq_kv).to(torch.bfloat16)

    # ── dV computation: Pd^T @ dO ─────────────────────────────────────────────
    # Keep Pd in bf16, dO_g in bf16 → bf16 matmul, accumulate in f32
    Pd_g = attn_weights_dropped.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)  # bf16
    # [bs, 8, 10, skv, sq] @ [bs, 8, 10, sq, 128] → [bs, 8, 10, skv, 128]
    dV_g = torch.matmul(Pd_g.transpose(-2, -1), dO_g)  # bf16 @ bf16
    # Sum over the 10 groups → [bs, 8, skv, 128]
    dV   = dV_g.sum(dim=2).to(torch.bfloat16)

    return dS, dV


# Cache compiled function at module level (reduce-overhead for low-latency)
_compiled_fn = None


def _get_compiled_fn():
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = torch.compile(
            _attn_bwd_impl,
            mode="reduce-overhead",
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

    fn = _get_compiled_fn()
    return fn(
        grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, inv_keep,
        bs, seq_q, seq_kv,
    )
