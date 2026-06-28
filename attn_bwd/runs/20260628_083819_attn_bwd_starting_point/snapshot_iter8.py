"""
torch.compile-accelerated attention-backward kernel (optimized v4).

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
    grad_attn_output,      # [bs, seq_q, 80, 128]      bfloat16
    attn_weights,          # [bs, 80, seq_q, seq_kv]   bfloat16
    attn_weights_dropped,  # [bs, 80, seq_q, seq_kv]   bfloat16
    value_states,          # [bs,  8, seq_kv, 128]     bfloat16
    dropout_mask,          # [bs, 80, seq_q, seq_kv]   bool
    inv_keep,              # float scalar
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    n_heads    = NUM_ATTENTION_HEADS   # 80

    # Transpose: [bs, seq_q, 80, 128] → [bs, 80, seq_q, 128]
    dO = grad_attn_output.permute(0, 2, 1, 3)  # [bs, 80, sq, 128]  bf16

    # ── dP computation: flat batched GEMM ────────────────────────────────────
    # Reshape dO: [bs*8, 10*sq, 128]  and  V: [bs*8, 128, skv]
    # → single GEMM: [bs*8, 10*sq, skv]
    dO_flat_dp = dO.reshape(bs * n_kv_heads, n_groups * seq_q, HEAD_DIM)  # [bs*8, 10*sq, 128]
    V_flat     = value_states.reshape(bs * n_kv_heads, seq_kv, HEAD_DIM)  # [bs*8, skv, 128]
    # bmm: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] → [bs*8, 10*sq, skv]
    dP_dropped_flat = torch.bmm(dO_flat_dp, V_flat.transpose(-2, -1))    # [bs*8, 10*sq, skv]
    # Reshape back for elementwise ops
    dP_dropped_g = dP_dropped_flat.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)

    # Dropout undo — fully bf16, use torch.where to avoid extra allocation
    mask_g = dropout_mask.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dP_g   = torch.where(mask_g, dP_dropped_g * inv_keep, torch.zeros_like(dP_dropped_g))

    # Softmax backward — fully bf16
    P_g  = attn_weights.view(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    dS_g = P_g * (dP_g - (P_g * dP_g).sum(dim=-1, keepdim=True))
    dS   = dS_g.view(bs, n_heads, seq_q, seq_kv)  # already bf16

    # ── dV computation: flat batched GEMM then reshape+sum ───────────────────
    # Flatten: [bs*80, sq, skv] and [bs*80, sq, 128]
    Pd_flat = attn_weights_dropped.view(bs * n_heads, seq_q, seq_kv)  # [bs*80, sq, skv]
    dO_flat = dO.reshape(bs * n_heads, seq_q, HEAD_DIM)               # [bs*80, sq, 128]
    # Batched: [bs*80, skv, sq] @ [bs*80, sq, 128] → [bs*80, skv, 128]
    dV_flat = torch.bmm(Pd_flat.transpose(-2, -1), dO_flat)           # bf16
    # Reshape and sum over 10 groups → [bs, 8, skv, 128]
    dV = dV_flat.view(bs, n_kv_heads, n_groups, seq_kv, HEAD_DIM).sum(dim=2).to(torch.bfloat16)

    return dS, dV


# Cache compiled function at module level
_compiled_fn = None


def _get_compiled_fn():
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = torch.compile(
            _attn_bwd_impl,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
        )
    return _compiled_fn


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep = float(1.0 / (1.0 - attention_dropout)) if attention_dropout > 0.0 else 1.0

    fn = _get_compiled_fn()
    return fn(
        grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, inv_keep,
    )
