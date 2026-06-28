"""
Hybrid attention-backward: cuBLAS BMM for matmuls + Triton fused elementwise for softmax bwd.

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


@triton.jit
def softmax_bwd_kernel(
    # Inputs
    dP_ptr,      # [bs, 80, sq, skv]  bfloat16  (raw dP̃ before dropout correction)
    P_ptr,       # [bs, 80, sq, skv]  bfloat16  (attn_weights, post-softmax)
    mask_ptr,    # [bs, 80, sq, skv]  bool
    # Output
    dS_ptr,      # [bs, 80, sq, skv]  bfloat16
    # Dims
    sq: tl.constexpr,
    skv: tl.constexpr,
    # Dropout scale
    dropout_scale: tl.constexpr,
    # Tile size along skv
    BLOCK_KV: tl.constexpr,
):
    """
    Fused single-pass softmax backward with dropout correction.
    Each program handles one row (one q index of one head of one batch).
    Computes: dS = P * (dP - sum_kv(dP * P))
    where dP = dP̃ * mask * dropout_scale (dropout correction applied here).

    Grid: (bs * 80 * sq,)
    """
    pid = tl.program_id(0)
    # pid indexes into [bs, 80, sq] flattened
    row_base = pid * skv

    kv_offs = tl.arange(0, BLOCK_KV)

    # Single pass: accumulate dot_sum and store dP (corrected) to registers
    # For large skv, we can't store all dP in registers, so we do two passes
    # BUT since dP and P are already materialized in DRAM (from BMM output),
    # each element is read twice. The key benefit is we avoid the V matmul here.

    # Pass 1: compute dot_sum
    dot_sum = tl.zeros([1], dtype=tl.float32)
    for kv_start in range(0, skv, BLOCK_KV):
        kv_o = kv_start + kv_offs
        kv_mask = kv_o < skv

        dP_tile = tl.load(dP_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)
        mask_tile = tl.load(mask_ptr + row_base + kv_o, mask=kv_mask, other=0).to(tl.int1)
        P_tile = tl.load(P_ptr + row_base + kv_o, mask=kv_mask, other=0.0).to(tl.float32)

        # Apply dropout correction
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

    # Transpose dO: [bs, sq, 80, 128] -> [bs, 80, sq, 128], contiguous
    dO = grad_attn_output.transpose(1, 2).contiguous()  # [bs, 80, sq, 128] bf16

    dropout_scale = 1.0 / (1.0 - attention_dropout) if attention_dropout > 0.0 else 1.0

    # =========================================================================
    # Step 1: dP̃ = dO @ V^T  (cuBLAS BMM in bfloat16)
    # dO:          [bs, 80, sq, 128]
    # vs_exp:      [bs, 80, skv, 128]  (GQA expand, no copy via expand+reshape)
    # dP̃ = dO @ vs_exp^T: [bs, 80, sq, skv]
    # =========================================================================
    # GQA expand: [bs, 8, skv, 128] -> [bs, 80, skv, 128]
    # Using expand + reshape — expand is zero-copy, reshape may need contiguous
    vs_exp = value_states[:, :, None, :, :].expand(
        bs, n_kv_heads, n_groups, seq_kv, head_dim
    ).reshape(bs, n_heads, seq_kv, head_dim).contiguous()  # [bs,80,skv,128] bf16

    # BMM: [bs*80, sq, 128] @ [bs*80, 128, skv] -> [bs*80, sq, skv]
    dO_2d  = dO.reshape(bs * n_heads, seq_q, head_dim)
    vs_2d  = vs_exp.reshape(bs * n_heads, seq_kv, head_dim)
    dP_raw = torch.bmm(dO_2d, vs_2d.transpose(1, 2))          # [bs*80, sq, skv] bf16
    dP_raw = dP_raw.reshape(bs, n_heads, seq_q, seq_kv)

    # =========================================================================
    # Step 2: Fused elementwise — dropout correction + softmax backward
    # Produces dS: [bs, 80, sq, skv] bfloat16
    # =========================================================================
    dS = torch.empty_like(dP_raw)
    N_rows = bs * n_heads * seq_q

    # Choose kernel variant based on skv size
    # If skv <= 2048, we can fit in registers (BLOCK_KV=2048 power-of-2)
    if seq_kv <= 2048:
        # Find next power of 2 >= seq_kv
        BLOCK_KV = 1
        while BLOCK_KV < seq_kv:
            BLOCK_KV *= 2
        BLOCK_KV = min(BLOCK_KV, 2048)

        # Flatten inputs for row-wise processing
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
        # Two-pass tiled version for large skv
        BLOCK_KV = 512

        dP_flat   = dP_raw.reshape(N_rows, seq_kv)
        P_flat    = attn_weights.reshape(N_rows, seq_kv)
        mask_flat = dropout_mask.reshape(N_rows, seq_kv)
        dS_flat   = dS.reshape(N_rows, seq_kv)

        softmax_bwd_kernel[(N_rows,)](
            dP_flat, P_flat, mask_flat, dS_flat,
            seq_q, seq_kv,
            dropout_scale,
            BLOCK_KV,
            num_warps=8,
        )

    # =========================================================================
    # Step 3: dV_exp = P̃^T @ dO  (cuBLAS BMM in bfloat16, then GQA reduce)
    # attn_weights_dropped: [bs, 80, sq, skv]
    # dO:                   [bs, 80, sq, 128]
    # dV_exp = P̃^T @ dO:   [bs, 80, skv, 128]
    # dV = dV_exp.reshape(bs, 8, 10, skv, 128).sum(2): [bs, 8, skv, 128]
    # =========================================================================
    Pd_2d   = attn_weights_dropped.reshape(bs * n_heads, seq_q, seq_kv)
    dV_exp  = torch.bmm(Pd_2d.transpose(1, 2), dO_2d)   # [bs*80, skv, 128] bf16
    dV_exp  = dV_exp.reshape(bs, n_heads, seq_kv, head_dim)

    # GQA reduction: sum over groups
    dV = dV_exp.reshape(bs, n_kv_heads, n_groups, seq_kv, head_dim).sum(dim=2)
    dV = dV.to(torch.bfloat16)

    return dS, dV
