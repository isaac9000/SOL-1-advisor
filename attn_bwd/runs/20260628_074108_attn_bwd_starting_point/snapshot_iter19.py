"""
Optimized attention-backward kernel — split-compile approach:
- GEMMs compiled with dynamic=False (static tile selection per shape)
- Elementwise softmax-backward compiled with dynamic=True (avoids recompilation overhead)

Based on Experiment 15 (best at 533.42 μs), same Exp 15 inductor config flags.

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


# ─────────────────────────────────────────────────────────────────────────────
# GEMM function — compiled with dynamic=False for best static tile selection
# Computes dP_dropped and dV (the two matrix multiplications)
# ─────────────────────────────────────────────────────────────────────────────
def _gemm_fn(
    grad_attn_output,     # [bs, seq_q, 80, 128]       bfloat16
    attn_weights_dropped, # [bs, 80, seq_q, seq_kv]    bfloat16
    value_states,         # [bs, 8, seq_kv, 128]       bfloat16
):
    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    # dO: [bs, seq_q, 80, 128] -> [bs, 8, 10, seq_q, 128]  bfloat16
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )

    # dP_dropped = dO @ V^T: [bs,8,10,sq,128] x [bs,8,skv,128] -> [bs,8,10,sq,skv]
    dP_dropped = torch.einsum(
        'bgnqd,bgkd->bgnqk',
        dO_grouped,    # [bs, 8, 10, sq, 128]
        value_states,  # [bs, 8, skv, 128]
    ).reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # dV = Pd^T @ dO summed over groups: [bs,8,10,sq,skv] x [bs,8,10,sq,128] -> [bs,8,skv,128]
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )
    dV = torch.einsum(
        'bgnqk,bgnqd->bgkd',
        Pd_grouped,
        dO_grouped,
    ).to(torch.bfloat16)

    return dP_dropped, dV


# dynamic=False: per-shape compilation → better static GEMM tile selection
_compiled_gemms = torch.compile(
    _gemm_fn,
    mode="max-autotune-no-cudagraphs",
    dynamic=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Elementwise softmax-backward function — compiled with dynamic=True
# Avoids per-shape recompilation overhead for the cheap elementwise ops
# ─────────────────────────────────────────────────────────────────────────────
def _elemwise_fn(
    dP_dropped,   # [bs, 80, sq, skv]  bfloat16
    attn_weights, # [bs, 80, sq, skv]  bfloat16
    dropout_mask, # [bs, 80, sq, skv]  bool
    inv_keep_prob,
):
    dP_bf16 = dP_dropped * dropout_mask * inv_keep_prob
    P       = attn_weights
    dPP     = dP_bf16 * P
    dS      = (P * (dP_bf16 - dPP.sum(dim=-1, keepdim=True))).to(torch.bfloat16)
    return dS


# dynamic=True: single compiled version handles all shapes for elementwise
_compiled_elemwise = torch.compile(
    _elemwise_fn,
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Step 1: GEMM kernels (static shape compilation → best tile sizes)
    dP_dropped, dV = _compiled_gemms(
        grad_attn_output, attn_weights_dropped, value_states
    )

    # Step 2: elementwise softmax-backward (dynamic compilation → no recompile overhead)
    dS = _compiled_elemwise(dP_dropped, attn_weights, dropout_mask, inv_keep_prob)

    return dS, dV
