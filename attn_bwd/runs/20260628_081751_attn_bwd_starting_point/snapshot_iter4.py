"""
Hybrid attention-backward: cuBLAS BMM for matmuls + Triton fused elementwise for softmax bwd.
Optimized to eliminate expensive .contiguous() copies.

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
    # Step 1: dP̃ = dO @ V^T  (cuBLAS BMM, avoiding .contiguous() copies)
    #
    # grad_attn_output: [bs, sq, 80, 128]  — layout (bs, sq, 80, 128)
    # value_states:     [bs,  8, skv, 128] — GQA
    #
    # Key insight: reshape dO as [bs*8, 10*sq, 128] and value_states as
    # [bs*8, skv, 128], then bmm gives [bs*8, 10*sq, skv].
    # Reshape back to [bs, 8, 10, sq, skv] = [bs, 80, sq, skv] after
    # transposing appropriately.
    #
    # dO in [bs, sq, 80, 128]: permute to [bs, 8, 10, sq, 128]
    # then reshape to [bs*8, 10*sq, 128]
    # =========================================================================

    # Reshape dO: [bs, sq, 80, 128] -> [bs, sq, 8, 10, 128]
    # -> [bs, 8, 10, sq, 128] -> [bs*8, 10*sq, 128]
    # This permute+reshape IS a copy but smaller than [bs,80,sq,128]
    # Actually, let's use a smarter approach:
    # Reshape [bs, sq, 80, 128] -> [bs, sq, 8, 10, 128]
    # permute -> [bs, 8, sq, 10, 128] — no, let's think carefully.
    #
    # Better: use [bs*8, 10, sq, 128] layout for dO
    # grad_attn_output: [bs, sq, 80, 128]
    #   -> view [bs, sq, 8, 10, 128]
    #   -> permute(0,2,3,1,4) = [bs, 8, 10, sq, 128]   <- requires contiguous
    #   -> reshape [bs*8, 10*sq, 128]
    # value_states: [bs, 8, skv, 128] -> reshape [bs*8, skv, 128]
    #
    # bmm([bs*8, 10*sq, 128], [bs*8, 128, skv]) -> [bs*8, 10*sq, skv]
    # reshape to [bs, 8, 10, sq, skv]
    # permute to [bs, 8, 10, sq, skv] (already correct order)
    # reshape to [bs, 80, sq, skv] — but need [bs, 80, sq, skv]
    # This needs a reshape from [bs, 8, 10, sq, skv] to [bs, 80, sq, skv]
    # which is contiguous only if the 8 and 10 dims are contiguous.
    # Actually [bs, 8, 10, sq, skv] -> [bs, 80, sq, skv] works if contiguous.
    #
    # The contiguous() call is needed for permute. But we go from
    # [bs, sq, 8, 10, 128] to [bs, 8, 10, sq, 128] — this is a smaller
    # tensor than [bs, 80, sq, 128] by factor of 1 (same elements).
    # HOWEVER we can skip the contiguous by using einsum or matmul with
    # non-contiguous tensors directly.
    #
    # Alternative: use torch.matmul with broadcasting instead of bmm.
    # dO: [bs, sq, 80, 128] -> [bs, sq, 8, 10, 128]  (view, free)
    # V:  [bs, 8, skv, 128] -> [bs, 1, 8, 1, skv, 128] won't work easily
    #
    # Simplest copy-free approach for dP:
    # Use torch.matmul broadcasting:
    #   dO_v: [bs, sq, 8, 10, 128]  (view of grad_attn_output, free)
    #   V_t:  [bs, 8, 128, skv]     (transpose of value_states, non-contiguous)
    #   Result: [bs, sq, 8, 10, skv] via matmul broadcast  <- may not work directly
    #
    # Actually, torch.matmul handles batched dims by broadcasting.
    # dO_v: [bs, sq, 8, 10, 128]
    # V:    [bs,  8,  1,  skv, 128].transpose(-2,-1) = [bs, 8, 1, 128, skv]
    # matmul: [bs, sq, 8, 10, 128] @ [bs, 8, 1, 128, skv]
    #       = [bs, sq, 8, 10, skv]  <- broadcast works here
    # Then permute [bs, sq, 8, 10, skv] -> [bs, 8, 10, sq, skv]
    # view -> [bs, 80, sq, skv]
    # The permute needs contiguous... but we can permute differently.
    #
    # Let's try: use matmul with proper reshaping to minimize copies.

    # APPROACH: Group dO by kv_head, use batched matmul without expanding V
    # dO: [bs, sq, 80, 128] -> view [bs, sq, 8, 10, 128]
    #     -> permute(0, 2, 1, 3, 4) = [bs, 8, sq, 10, 128]  (copy needed)
    #     -> reshape [bs*8, sq*10, 128]  -- but this mixes sq and heads
    #
    # Actually the cleanest copy-free approach is:
    # Keep dO as [bs, sq, 80, 128] and use einsum or reshape carefully.
    #
    # Let's use: dO reshaped to [bs, sq, 8, 10, 128]
    # V: [bs, 8, skv, 128]
    # We want: sum_d dO[b,q,kv,g,d] * V[b,kv,k,d] = dP[b,kv,g,q,k]
    # then rearrange to [b, kv*g=80, q, k]
    #
    # Using torch.einsum('bqkgd,bksd->bkgqs', ...) is elegant but slow.
    #
    # BEST approach without extra large copies:
    # Reshape dO [bs, sq, 80, 128] -> [bs*8, 10*sq, 128] requires a
    # permute that IS a copy but the output is same size as dO — no expansion.
    # The key win is avoiding the [bs,80,skv,128] expansion of V.

    # Step 1a: Reshape dO to group by kv_head
    # [bs, sq, 80, 128] -> [bs, sq, 8, 10, 128] (view, free)
    dO_grouped = grad_attn_output.view(bs, seq_q, n_kv_heads, n_groups, head_dim)
    # permute to [bs, 8, 10, sq, 128] — requires contiguous (copy, same size as dO)
    dO_grouped = dO_grouped.permute(0, 2, 3, 1, 4).contiguous()
    # reshape to [bs*8, 10*sq, 128]
    dO_gq = dO_grouped.reshape(bs * n_kv_heads, n_groups * seq_q, head_dim)

    # Step 1b: Reshape V to [bs*8, skv, 128]
    V_2d = value_states.reshape(bs * n_kv_heads, seq_kv, head_dim)

    # Step 1c: BMM: [bs*8, 10*sq, 128] @ [bs*8, 128, skv] -> [bs*8, 10*sq, skv]
    dP_raw_gq = torch.bmm(dO_gq, V_2d.transpose(1, 2))  # [bs*8, 10*sq, skv]

    # Step 1d: Reshape to [bs, 8, 10, sq, skv] -> permute to [bs, 8*10, sq, skv] = [bs, 80, sq, skv]
    # [bs*8, 10*sq, skv] -> [bs, 8, 10, sq, skv]
    dP_raw = dP_raw_gq.reshape(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    # -> [bs, 80, sq, skv]: merge dims 1 and 2
    # Need contiguous for this reshape since dims aren't adjacent in memory after view
    dP_raw = dP_raw.reshape(bs, n_heads, seq_q, seq_kv)

    # =========================================================================
    # Step 2: Fused elementwise — dropout correction + softmax backward
    # =========================================================================
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
    # Step 3: dV = P̃_dropped^T @ dO, with GQA reduction
    #
    # attn_weights_dropped: [bs, 80, sq, skv]
    # dO: [bs, 8, 10, sq, 128] (already computed above as dO_grouped)
    #
    # We need: dV[b, kv, k, d] = sum_{g,q} P_d[b, kv*10+g, q, k] * dO[b, q, kv*10+g, d]
    #
    # Use the grouped dO layout:
    # dO_grouped: [bs, 8, 10, sq, 128] -> reshape [bs*8, 10*sq, 128]  (already have dO_gq)
    # P_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv] -> [bs*8, 10*sq, skv]
    #
    # bmm: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    # =========================================================================
    # Reshape attn_weights_dropped: [bs, 80, sq, skv] -> [bs, 8, 10, sq, skv]
    # -> [bs*8, 10*sq, skv]
    # attn_weights_dropped is [bs, 80, sq, skv] which is contiguous
    Pd_gq = attn_weights_dropped.reshape(bs, n_kv_heads, n_groups, seq_q, seq_kv)
    Pd_gq = Pd_gq.reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)

    # BMM: P^T @ dO: [bs*8, skv, 10*sq] @ [bs*8, 10*sq, 128] -> [bs*8, skv, 128]
    dV = torch.bmm(Pd_gq.transpose(1, 2), dO_gq)  # [bs*8, skv, 128]
    dV = dV.reshape(bs, n_kv_heads, seq_kv, head_dim).to(torch.bfloat16)

    return dS, dV
