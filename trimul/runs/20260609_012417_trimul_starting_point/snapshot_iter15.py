"""
TriMul submission — fused gate+mask, fused transpose+LN+gate Triton kernels + cuBLAS.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Module-level weight cache
_weight_cache = {}


@triton.jit
def _fused_gate_mask_kernel(
    fused_out_ptr,   # (M, 5*H) input — M = B*N*N
    mask_ptr,        # (M,) mask values
    left_ptr,        # (B*H, N, N) output for left  — layout for contiguous bmm
    right_ptr,       # (B*H, N, N) output for right — layout: (B*H, k, j) for bmm
    out_gate_ptr,    # (M, H) = (B, N, N, H) output for out_gate
    M, N, H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Fuse: read fused_out once, apply sigmoid + mask, write outputs.
    left  written in (B*H, N, N) layout.
    right written in (B*H, N, N) layout: right[b*H+h, k, j] for contiguous bmm.
    out_gate written in (M, H) layout.
    Grid: (cdiv(M, BLOCK_M),)
    """
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    h_offs = tl.arange(0, BLOCK_H)

    # Decode (b, row, col) from m  where m = b*N²+row*N+col
    NN = N * N
    b_idx   = m_offs // NN
    rc      = m_offs % NN
    row_idx = rc // N
    col_idx = rc % N

    mval = tl.load(mask_ptr + m_offs, mask=m_mask, other=0.0)

    row_base = m_offs * (5 * H)
    lp = tl.load(fused_out_ptr + row_base[:, None] + h_offs[None, :], mask=m_mask[:, None], other=0.0)
    rp = tl.load(fused_out_ptr + row_base[:, None] + H + h_offs[None, :], mask=m_mask[:, None], other=0.0)
    lg = tl.load(fused_out_ptr + row_base[:, None] + 2 * H + h_offs[None, :], mask=m_mask[:, None], other=0.0)
    rg = tl.load(fused_out_ptr + row_base[:, None] + 3 * H + h_offs[None, :], mask=m_mask[:, None], other=0.0)
    og = tl.load(fused_out_ptr + row_base[:, None] + 4 * H + h_offs[None, :], mask=m_mask[:, None], other=0.0)

    left_val  = lp * tl.sigmoid(lg) * mval[:, None]
    right_val = rp * tl.sigmoid(rg) * mval[:, None]
    og_sig    = tl.sigmoid(og)

    l_dest = (b_idx[:, None] * H + h_offs[None, :]) * NN + row_idx[:, None] * N + col_idx[:, None]
    tl.store(left_ptr + l_dest, left_val, mask=m_mask[:, None])

    r_dest = (b_idx[:, None] * H + h_offs[None, :]) * NN + col_idx[:, None] * N + row_idx[:, None]
    tl.store(right_ptr + r_dest, right_val, mask=m_mask[:, None])

    g_dest = m_offs[:, None] * H + h_offs[None, :]
    tl.store(out_gate_ptr + g_dest, og_sig, mask=m_mask[:, None])


@triton.jit
def _fused_transpose_ln_gate_kernel(
    bmm_out_ptr,   # (B*H, N, N) — bmm result
    out_gate_ptr,  # (M, H) = (B*N*N, H)
    norm_w_ptr,    # (H,)
    norm_b_ptr,    # (H,)
    out_ptr,       # (M, H) = (B*N*N, H) output
    B, N, H: tl.constexpr,
    BLOCK_IJ: tl.constexpr,
    BLOCK_H: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Single kernel: transpose (B*H,N,N)->(M,H) + LayerNorm + gate multiply.
    Grid: (B * cdiv(N*N, BLOCK_IJ),)
    Each program handles BLOCK_IJ spatial positions (ij) for one batch b.
    Loads tile[h, ij] from bmm_out (coalesced along ij), computes LN per ij column,
    multiplies by out_gate, writes output contiguously.
    """
    pid   = tl.program_id(0)
    NN    = N * N
    NNblk = tl.cdiv(NN, BLOCK_IJ)
    b     = pid // NNblk
    pij   = pid % NNblk
    ij_start = pij * BLOCK_IJ

    ij_offs = ij_start + tl.arange(0, BLOCK_IJ)  # (BLOCK_IJ,)
    ij_mask = ij_offs < NN
    h_offs  = tl.arange(0, BLOCK_H)               # (BLOCK_H,)
    h_mask  = h_offs < H

    # Load tile from bmm_out: shape (BLOCK_H, BLOCK_IJ)
    # bmm_out[b*H+h, ij] is at offset (b*H+h)*NN + ij
    src_base = b * H * NN
    src_ptrs = bmm_out_ptr + src_base + h_offs[:, None] * NN + ij_offs[None, :]
    tile = tl.load(src_ptrs, mask=h_mask[:, None] & ij_mask[None, :], other=0.0)
    # tile: (BLOCK_H, BLOCK_IJ) — each column is the H-vector for one (b,ij) position

    # Load norm params: (BLOCK_H,)
    nw = tl.load(norm_w_ptr + h_offs, mask=h_mask)
    nb = tl.load(norm_b_ptr + h_offs, mask=h_mask)

    # Apply LayerNorm over H dimension (axis=0, i.e. per column)
    mean    = tl.sum(tile, axis=0) / H            # (BLOCK_IJ,)
    diff    = tile - mean[None, :]                 # (BLOCK_H, BLOCK_IJ)
    var     = tl.sum(diff * diff, axis=0) / H      # (BLOCK_IJ,)
    inv_std = tl.rsqrt(var + eps)                  # (BLOCK_IJ,)
    normed  = diff * inv_std[None, :]              # (BLOCK_H, BLOCK_IJ)
    normed  = normed * nw[:, None] + nb[:, None]   # affine

    # Load out_gate: shape (BLOCK_IJ, BLOCK_H) from (M, H) = row-major
    # out_gate[b*NN+ij, h]
    gate_base = b * NN
    g_ptrs = out_gate_ptr + (gate_base + ij_offs[:, None]) * H + h_offs[None, :]
    gate = tl.load(g_ptrs, mask=ij_mask[:, None] & h_mask[None, :], other=0.0)
    # gate: (BLOCK_IJ, BLOCK_H) — need to transpose to match normed (BLOCK_H, BLOCK_IJ)
    gate_t = tl.trans(gate)  # (BLOCK_H, BLOCK_IJ)

    result = normed * gate_t  # (BLOCK_H, BLOCK_IJ)

    # Write output: out[b*NN+ij, h] — shape (BLOCK_IJ, BLOCK_H) contiguous in H
    o_base = b * NN
    o_ptrs = out_ptr + (o_base + ij_offs[:, None]) * H + h_offs[None, :]
    tl.store(o_ptrs, tl.trans(result), mask=ij_mask[:, None] & h_mask[None, :])


def fused_ln_gate(bmm_out, out_gate, norm_w, norm_b, B, N, H):
    """
    Fused: transpose (B*H,N,N)->(M,H) + LayerNorm + gate multiply.
    bmm_out:  (B*H, N, N)
    out_gate: (M, H) where M = B*N*N
    Returns:  (M, H)
    """
    M = B * N * N
    NN = N * N
    out = torch.empty(M, H, device=bmm_out.device, dtype=bmm_out.dtype)

    BLOCK_IJ = 64
    BLOCK_H  = triton.next_power_of_2(H)
    grid = (B * triton.cdiv(NN, BLOCK_IJ),)

    _fused_transpose_ln_gate_kernel[grid](
        bmm_out, out_gate, norm_w, norm_b, out,
        B, N, H,
        BLOCK_IJ=BLOCK_IJ,
        BLOCK_H=BLOCK_H,
        eps=1e-5,
    )
    return out


def fused_gate_mask(fused_out, mask_flat, B, N, hidden_dim):
    """
    fused_out: (M, 5*H) contiguous, M = B*N*N
    mask_flat: (M,) contiguous
    Returns:
      left:     (B*H, N, N) contiguous — ready for bmm as left operand
      right:    (B*H, N, N) contiguous — ready for bmm as right operand (k,j layout)
      out_gate: (M, H) = (B, N, N, H) flat
    """
    M = fused_out.shape[0]
    H = hidden_dim
    left     = torch.empty(B * H, N, N, device=fused_out.device, dtype=fused_out.dtype)
    right    = torch.empty(B * H, N, N, device=fused_out.device, dtype=fused_out.dtype)
    out_gate = torch.empty(M, H,       device=fused_out.device, dtype=fused_out.dtype)

    BLOCK_M = 16
    BLOCK_H = triton.next_power_of_2(H)
    grid = (triton.cdiv(M, BLOCK_M),)

    _fused_gate_mask_kernel[grid](
        fused_out, mask_flat,
        left, right, out_gate,
        M, N, H,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
    )
    return left, right, out_gate


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cache_key = (
        weights['left_proj.weight'].data_ptr(),
        weights['right_proj.weight'].data_ptr(),
        weights['left_gate.weight'].data_ptr(),
        weights['right_gate.weight'].data_ptr(),
        weights['out_gate.weight'].data_ptr(),
        weights['norm.weight'].data_ptr(),
        weights['to_out.weight'].data_ptr(),
    )
    if cache_key not in _weight_cache:
        fused_w = torch.cat([
            weights['left_proj.weight'],
            weights['right_proj.weight'],
            weights['left_gate.weight'],
            weights['right_gate.weight'],
            weights['out_gate.weight'],
        ], dim=0)
        _weight_cache[cache_key] = {
            'fused_w': fused_w,
            'fused_w_fp16': fused_w.to(torch.float16),
            'norm_w': weights['norm.weight'],
            'norm_b': weights['norm.bias'],
            'to_out_norm_w': weights['to_out_norm.weight'],
            'to_out_norm_b': weights['to_out_norm.bias'],
            'to_out_w': weights['to_out.weight'],
            'to_out_w_fp16': weights['to_out.weight'].to(torch.float16),
        }
    cached = _weight_cache[cache_key]

    fused_w       = cached['fused_w_fp16']
    norm_w        = cached['norm_w']
    norm_b        = cached['norm_b']
    to_out_norm_w = cached['to_out_norm_w']
    to_out_norm_b = cached['to_out_norm_b']
    to_out_w      = cached['to_out_w_fp16']
    hidden_dim    = config["hidden_dim"]
    dim           = config["dim"]

    B, N, _, D = input_tensor.shape
    M = B * N * N

    # Step 1: LayerNorm on input
    x = F.layer_norm(input_tensor, (dim,), norm_w, norm_b)

    # Step 2: Fused 5-projection matmul in fp16 -> (M, 5*H), cast back to fp32
    x_flat = x.reshape(M, dim).to(torch.float16)
    fused_out = F.linear(x_flat, fused_w).to(torch.float32)  # (M, 5*H)

    # Step 3: Fused Triton kernel: gate sigmoid + mask multiply in one pass
    # left/right written directly in (B*H, N, N) layout — no permute needed for bmm
    H = hidden_dim
    mask_flat = mask.reshape(M)
    left_bhn, right_bhn, out_gate_flat = fused_gate_mask(fused_out, mask_flat, B, N, H)

    # Step 4: Einsum via bmm — inputs already in (B*H, N, N) layout, no permute!
    # left_bhn[b*H+h, i, k] @ right_bhn[b*H+h, k, j] -> out_bhn[b*H+h, i, j]
    out_bhn = torch.bmm(left_bhn, right_bhn)  # (B*H, N, N)

    # Step 5: Fused LN + gate multiply (replaces permute + F.layer_norm + out*out_gate)
    # Reads out_bhn strided over H dim, applies LN, multiplies out_gate, writes (M,H)
    out_flat = fused_ln_gate(out_bhn, out_gate_flat, to_out_norm_w, to_out_norm_b, B, N, H)

    # Step 6: to_out linear in fp16
    out = F.linear(out_flat.reshape(B, N, N, H).to(torch.float16), to_out_w).to(torch.float32)

    return out
