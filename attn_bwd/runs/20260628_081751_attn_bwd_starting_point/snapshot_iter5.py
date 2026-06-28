"""
Hybrid attention-backward: cuBLAS BMM for matmuls + Triton fused elementwise for softmax bwd.
Optimized with CUDA streams to overlap independent dP and dV BMMs.

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
import triton
import triton.language as tl

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
N_GROUPS = 10

# Pre-create CUDA streams at module load time to avoid per-call overhead
_stream_dp = None
_stream_dv = None

def _get_streams():
    global _stream_dp, _stream_dv
    if _stream_dp is None:
        _stream_dp = torch.cuda.Stream()
        _stream_dv = torch.cuda.Stream()
    return _stream_dp, _stream_dv


@triton.jit
def softmax_bwd_kernel_wide(
    # Inputs
    dP_ptr,      # [N_rows, skv]  bfloat16
    P_ptr,       # [N_rows, skv]  bfloat16
    mask_ptr,    # [N_rows, skv]  bool
    # Output
    dS_ptr,      # [N_rows, skv]  bfloat16
    # Dims
    skv: tl.constexpr,
    # Dropout scale
    dropout_scale: tl.constexpr,
    # Tile size along skv — must cover full skv in one tile for single-pass
    BLOCK_KV: tl.constexpr,
):
    """
    Single-pass softmax backward when BLOCK_KV >= skv (fits in registers).
    Each program handles one row. Grid: (N_rows,)
    """
    pid = tl.program_id(0)
    row_base = pid * skv

    kv_offs = tl.arange(0, BLOCK_KV)
    kv_mask = kv_offs < skv

    dP_tile = tl.load(dP_ptr + row_base + kv_offs, mask=kv_mask, other=0.0).to(tl.float32)
    mask_tile = tl.load(mask_ptr + row_base + kv_offs, mask=kv_mask, other=False).to(tl.int1)
    P_tile = tl.load(P_ptr + row_base + kv_offs, mask=kv_mask, other=0.0).to(tl.float32)

    # Apply dropout correction
    dP_tile = tl.where(mask_tile, dP_tile * dropout_scale, 0.0)

    # Softmax backward
    dot_sum = tl.sum(dP_tile * P_tile, axis=0)
    dS_tile = P_tile * (dP_tile - dot_sum)

    tl.store(dS_ptr + row_base + kv_offs, dS_tile.to(tl.bfloat16), mask=kv_mask)


@triton.jit
def softmax_bwd_kernel_tiled(
    # Inputs
    dP_ptr,      # [N_rows, skv]  bfloat16
    P_ptr,       # [N_rows, skv]  bfloat16
    mask_ptr,    # [N_rows, skv]  bool
    # Output
    dS_ptr,      # [N_rows, skv]  bfloat16
    # Dims
    skv: tl.constexpr,
    # Dropout scale
    dropout_scale: tl.constexpr,
    # Tile size along skv
    BLOCK_KV: tl.constexpr,
):
    """
    Two-pass tiled softmax backward for large skv.
    Each program handles one row. Grid: (N_rows,)
    """
    pid = tl.program_id(0)
    row_base = pid * skv

    kv_offs = tl.arange(0, BLOCK_KV)

    # Pass 1: compute dot_sum
    dot_sum = tl.zeros([1], dtype=tl.float32)
    for kv_start in range(0, skv, BLOCK_KV):
        kv_o = kv_start + kv_offs
        kv_mask = kv_o < skv
        dP_tile = tl.load(dP_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)
        mask_tile = tl.load(mask_ptr + row_base + kv_o, mask=kv_mask, other=0).to(tl.int1)
        P_tile = tl.load(P_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)
        dP_tile = tl.where(mask_tile, dP_tile * dropout_scale, 0.0)
        dot_sum += tl.sum(dP_tile * P_tile, axis=0)

    dot_sum_val = tl.sum(dot_sum, axis=0)

    # Pass 2: compute and store dS
    for kv_start in range(0, skv, BLOCK_KV):
        kv_o = kv_start + kv_offs
        kv_mask = kv_o < skv
        dP_tile = tl.load(dP_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)
        mask_tile = tl.load(mask_ptr + row_base + kv_o, mask=kv_mask, other=0).to(tl.int1)
        P_tile = tl.load(P_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)
        dP_tile = tl.where(mask_tile, dP_tile * dropout_scale, 0.0)
        dS_tile = P_tile * (dP_tile - dot_sum_val)
        tl.store(dS_ptr + row_base + kv_o, dS_tile.to(tl.bfloat16), mask=kv_mask)


def custom_kernel(data):
    (grad_attn_output, attn_weights, attn_weights_dropped,
     value_states, dropout_mask, attention_dropout) = data

    bs     = grad_attn_output.shape[0]
    seq_q  = grad_attn_output.shape[1]
    seq_kv = value_states.shape[2]

    n_heads    = NUM_ATTENTION_HEADS   # 80
    n_kv_heads = NUM_KEY_VALUE_HEADS   # 8
    n_groups   = N_GROUPS              # 10
    head_dim   = HEAD_DIM              # 128

    dropout_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # =========================================================================
    # Prepare grouped dO layout (shared by both BMMs)
    # [bs, sq, 80, 128] -> [bs, sq, 8, 10, 128] -> [bs, 8, 10, sq, 128]
    # -> [bs*8, 10*sq, 128]
    # This is on the default stream before forking.
    # =========================================================================
    dO_grouped = grad_attn_output.view(bs, seq_q, n_kv_heads, n_groups, head_dim)
    dO_grouped = dO_grouped.permute(0, 2, 3, 1, 4).contiguous()
    dO_gq = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, head_dim)

    V_2d  = value_states.reshape(bs * n_kv_heads, seq_kv, head_dim)
    Pd_gq = attn_weights_dropped.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # Get pre-created streams
    stream_dp, stream_dv = _get_streams()

    current_stream = torch.cuda.current_stream()

    # =========================================================================
    # Stream 1 (stream_dp): compute dP̃ BMM -> softmax bwd -> dS
    # =========================================================================
    with torch.cuda.stream(stream_dp):
        # Make stream_dp wait for the default stream to finish dO_gq prep
        stream_dp.wait_stream(current_stream)

        # BMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] -> [bs*8, 10*sq, skv]
        dP_raw_gq = torch.bmm(dO_gq, V_2d.transpose(1, 2))
        dP_raw = dP_raw_gq.reshape(bs, n_heads, seq_q, seq_kv)

        # Fused elementwise: dropout correction + softmax backward
        dS = torch.empty_like(dP_raw)
        N_rows = bs * n_heads * seq_q

        if seq_kv <= 2048:
            BLOCK_KV = 1
            while BLOCK_KV < seq_kv:
                BLOCK_KV *= 2
            BLOCK_KV = min(BLOCK_KV, 2048)

            dP_flat   = dP_raw.reshape(N_rows, seq_kv)
            P_flat    = attn_weights.reshape(N_rows, seq_kv)
            mask_flat = dropout_mask.reshape(N_rows, seq_kv)
            dS_flat   = dS.reshape(N_rows, seq_kv)

            softmax_bwd_kernel_wide[(N_rows,)](
                dP_flat, P_flat, mask_flat, dS_flat,
                seq_kv,
                dropout_scale,
                BLOCK_KV,
                num_warps=8,
            )
        else:
            BLOCK_KV = 512

            dP_flat   = dP_raw.reshape(N_rows, seq_kv)
            P_flat    = attn_weights.reshape(N_rows, seq_kv)
            mask_flat = dropout_mask.reshape(N_rows, seq_kv)
            dS_flat   = dS.reshape(N_rows, seq_kv)

            softmax_bwd_kernel_tiled[(N_rows,)](
                dP_flat, P_flat, mask_flat, dS_flat,
                seq_kv,
                dropout_scale,
                BLOCK_KV,
                num_warps=8,
            )

    # =========================================================================
    # Stream 2 (stream_dv): compute dV BMM (independent of stream_dp)
    # =========================================================================
    with torch.cuda.stream(stream_dv):
        # Make stream_dv wait for the default stream to finish dO_gq prep
        stream_dv.wait_stream(current_stream)

        # BMM: P^T @ dO: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
        dV = torch.bmm(Pd_gq.transpose(1, 2), dO_gq)
        dV = dV.reshape(bs, n_kv_heads, seq_kv, head_dim).to(torch.bfloat16)

    # =========================================================================
    # Synchronize both streams back to the default stream
    # =========================================================================
    current_stream.wait_stream(stream_dp)
    current_stream.wait_stream(stream_dv)

    return dS, dV
