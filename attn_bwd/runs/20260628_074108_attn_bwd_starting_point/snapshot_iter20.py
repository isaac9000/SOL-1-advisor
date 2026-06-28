"""
Optimized attention-backward kernel — concurrent CUDA streams for the two GEMMs,
with Exp 15 inductor config for the elementwise softmax-backward.

The dP GEMM (dO @ V^T) and dV GEMM (Pd^T @ dO) have no data dependency on each
other, so they can run concurrently on separate CUDA streams to overlap compute
on B200's multiple SM clusters.

Based on Experiment 15 (best at 533.42 μs).

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
# Compiled elementwise softmax-backward (proven correct from Exp 15)
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


_compiled_elemwise = torch.compile(
    _elemwise_fn,
    mode="max-autotune-no-cudagraphs",
    dynamic=True,
)

# Pre-created CUDA streams for concurrent GEMM execution
_stream_dp = None
_stream_dv = None


def _get_streams():
    global _stream_dp, _stream_dv
    if _stream_dp is None:
        _stream_dp = torch.cuda.Stream()
        _stream_dv = torch.cuda.Stream()
    return _stream_dp, _stream_dv


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    inv_keep_prob = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # Pre-compute dO_grouped (shared by both GEMMs) on the current stream
    dO_grouped = grad_attn_output.transpose(1, 2).reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, HEAD_DIM
    )
    # Pd_grouped (for dV) — reshape only, no copy
    Pd_grouped = attn_weights_dropped.reshape(
        bs, NUM_KEY_VALUE_HEADS, N_GROUPS, seq_q, seq_kv
    )

    current_stream = torch.cuda.current_stream()
    stream_dp, stream_dv = _get_streams()

    # Both sub-streams wait for the current stream (ensures dO_grouped is ready)
    stream_dp.wait_stream(current_stream)
    stream_dv.wait_stream(current_stream)

    # ── Launch dP GEMM on stream_dp ───────────────────────────────────────────
    with torch.cuda.stream(stream_dp):
        # dP_dropped = dO @ V^T: [bs,8,10,sq,128] x [bs,8,skv,128] -> [bs,80,sq,skv]
        dP_dropped = torch.einsum(
            'bgnqd,bgkd->bgnqk',
            dO_grouped,    # [bs, 8, 10, sq, 128]
            value_states,  # [bs, 8, skv, 128]
        ).reshape(bs, NUM_ATTENTION_HEADS, seq_q, seq_kv)

    # ── Launch dV GEMM on stream_dv (concurrent with dP) ─────────────────────
    with torch.cuda.stream(stream_dv):
        # dV = Pd^T @ dO summed over groups -> [bs, 8, skv, 128]
        dV = torch.einsum(
            'bgnqk,bgnqd->bgkd',
            Pd_grouped,
            dO_grouped,
        ).to(torch.bfloat16)

    # Wait for both GEMMs to complete before the elementwise ops
    current_stream.wait_stream(stream_dp)
    current_stream.wait_stream(stream_dv)

    # ── Softmax backward on current stream (depends on dP_dropped) ────────────
    dS = _compiled_elemwise(dP_dropped, attn_weights, dropout_mask, inv_keep_prob)

    return dS, dV
