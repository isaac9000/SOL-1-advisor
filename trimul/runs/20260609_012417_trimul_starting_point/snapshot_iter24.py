"""
TriMul submission — fused gate+mask + fused transpose+LN+gate Triton kernels + cuBLAS GEMMs.
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
    left  written in (B*H, N, N) layout:  left[b*H+h, i, k]  = b*H*N²+h*N²+i*N+k
    right written in (B*H, N, N) layout:  right[b*H+h, k, j] = b*H*N²+h*N²+k*N+j
      (so that bmm(left, right) gives out[b*H+h, i, j] without any permute)
    out_gate written in (M, H) layout (unchanged).

    m = b*N²+i*N+k  for left  (k is the reduction index)
    m = b*N²+j*N+k  for right (k is the reduction index, j is the output index)
    But we process row m and write to both left and right with the same m value;
    m = b*N²+row*N+col.  For left: b, i=row, k=col. For right: b, j=row, k=col.
    """
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    h_offs = tl.arange(0, BLOCK_H)

    # Decode (b, row, col) from m  where m = b*N²+row*N+col
    NN = N * N
    b_idx   = m_offs // NN
    rc      = m_offs % NN
    row_idx = rc // N   # this is 'i' for left, 'j' for right
    col_idx = rc % N    # this is 'k' for both

    # Load mask values for these rows
    mval = tl.load(mask_ptr + m_offs, mask=m_mask, other=0.0)  # (BLOCK_M,)

    # Base pointer for fused_out row
    row_base = m_offs * (5 * H)

    # Load all 5 chunks: each (BLOCK_M, BLOCK_H)
    lp = tl.load(fused_out_ptr + row_base[:, None] + h_offs[None, :], mask=m_mask[:, None], other=0.0)
    rp = tl.load(fused_out_ptr + row_base[:, None] + H + h_offs[None, :], mask=m_mask[:, None], other=0.0)
    lg = tl.load(fused_out_ptr + row_base[:, None] + 2 * H + h_offs[None, :], mask=m_mask[:, None], other=0.0)
    rg = tl.load(fused_out_ptr + row_base[:, None] + 3 * H + h_offs[None, :], mask=m_mask[:, None], other=0.0)
    og = tl.load(fused_out_ptr + row_base[:, None] + 4 * H + h_offs[None, :], mask=m_mask[:, None], other=0.0)

    left_val  = lp * tl.sigmoid(lg) * mval[:, None]   # (BLOCK_M, BLOCK_H)
    right_val = rp * tl.sigmoid(rg) * mval[:, None]
    og_sig    = tl.sigmoid(og)

    # Write left in (B*H, N, N) layout: dest = (b*H+h)*N²+row*N+col = (b*H+h)*NN+row*N+col
    # left[b*H+h, i, k] where i=row_idx, k=col_idx
    l_dest = (b_idx[:, None] * H + h_offs[None, :]) * NN + row_idx[:, None] * N + col_idx[:, None]
    tl.store(left_ptr + l_dest, left_val, mask=m_mask[:, None])

    # Write right in (B*H, N, N) layout: right[b*H+h, k, j] where k=col_idx, j=row_idx
    r_dest = (b_idx[:, None] * H + h_offs[None, :]) * NN + col_idx[:, None] * N + row_idx[:, None]
    tl.store(right_ptr + r_dest, right_val, mask=m_mask[:, None])

    # Write out_gate in (M, H) layout (unchanged)
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
    h_offs  = tl.arange(0, BLOCK_H)               # (BLOCK_H,) — BLOCK_H == H always

    # Load tile from bmm_out: shape (BLOCK_H, BLOCK_IJ)
    # bmm_out[b*H+h, ij] is at offset (b*H+h)*NN + ij
    # No h_mask needed: BLOCK_H == H since H is always a power of 2
    src_base = b * H * NN
    src_ptrs = bmm_out_ptr + src_base + h_offs[:, None] * NN + ij_offs[None, :]
    tile = tl.load(src_ptrs, mask=ij_mask[None, :], other=0.0)

    # Load norm params: (BLOCK_H,) — no mask needed
    nw = tl.load(norm_w_ptr + h_offs)
    nb = tl.load(norm_b_ptr + h_offs)

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
    gate = tl.load(g_ptrs, mask=ij_mask[:, None], other=0.0)
    gate_t = tl.trans(gate)  # (BLOCK_H, BLOCK_IJ)

    result = normed * gate_t  # (BLOCK_H, BLOCK_IJ)

    # Write output: out[b*NN+ij, h] — shape (BLOCK_IJ, BLOCK_H) contiguous in H
    o_base = b * NN
    o_ptrs = out_ptr + (o_base + ij_offs[:, None]) * H + h_offs[None, :]
    tl.store(o_ptrs, tl.trans(result), mask=ij_mask[:, None])


def fused_ln_gate(bmm_out, out_gate, norm_w, norm_b, out_flat, B, N, H):
    """
    Fused: transpose (B*H,N,N)->(M,H) + LayerNorm + gate multiply.
    bmm_out:  (B*H, N, N)
    out_gate: (M, H) where M = B*N*N
    out_flat: pre-allocated (M, H) output buffer
    """
    NN = N * N
    BLOCK_IJ = 64
    BLOCK_H  = triton.next_power_of_2(H)
    grid = (B * triton.cdiv(NN, BLOCK_IJ),)

    _fused_transpose_ln_gate_kernel[grid](
        bmm_out, out_gate, norm_w, norm_b, out_flat,
        B, N, H,
        BLOCK_IJ=BLOCK_IJ,
        BLOCK_H=BLOCK_H,
        eps=1e-5,
    )


def fused_gate_mask(fused_out, mask_flat, left, right, out_gate, B, N, H):
    """
    fused_out: (M, 5*H) contiguous, M = B*N*N
    mask_flat: (M,) contiguous
    left, right, out_gate: pre-allocated output buffers
    """
    M = fused_out.shape[0]
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
            'norm_w': weights['norm.weight'],
            'norm_b': weights['norm.bias'],
            'to_out_norm_w': weights['to_out_norm.weight'],
            'to_out_norm_b': weights['to_out_norm.bias'],
            'to_out_w': weights['to_out.weight'],
        }
    cached = _weight_cache[cache_key]

    fused_w       = cached['fused_w']
    norm_w        = cached['norm_w']
    norm_b        = cached['norm_b']
    to_out_norm_w = cached['to_out_norm_w']
    to_out_norm_b = cached['to_out_norm_b']
    to_out_w      = cached['to_out_w']
    hidden_dim    = config["hidden_dim"]
    dim           = config["dim"]

    B, N, _, D = input_tensor.shape
    M = B * N * N
    H = hidden_dim
    dev = input_tensor.device

    # Step 1: LayerNorm
    x = F.layer_norm(input_tensor, (dim,), norm_w, norm_b)

    # Step 2: Fused 5-projection matmul -> (M, 5*H)
    x_flat = x.reshape(M, dim)
    fused_out = F.linear(x_flat, fused_w)

    # Step 3: Fused Triton gate/mask kernel
    left     = torch.empty(B * H, N, N, device=dev, dtype=torch.float32)
    right    = torch.empty(B * H, N, N, device=dev, dtype=torch.float32)
    out_gate = torch.empty(M, H,       device=dev, dtype=torch.float32)
    mask_flat = mask.reshape(M)
    fused_gate_mask(fused_out, mask_flat, left, right, out_gate, B, N, H)

    # Step 4: bmm in float32 with TF32 enabled
    out_bhn = torch.bmm(left, right)

    # Step 5: Fused transpose+LN+gate
    out_flat = torch.empty(M, H, device=dev, dtype=torch.float32)
    fused_ln_gate(out_bhn, out_gate, to_out_norm_w, to_out_norm_b, out_flat, B, N, H)

    # Step 6: to_out linear
    return F.linear(out_flat.reshape(B, N, N, H), to_out_w)
