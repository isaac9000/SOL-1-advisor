"""
TriMul submission — fused LN+GEMM, fused gate+mask, fused LN+gate Triton kernels.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Module-level weight cache
_weight_cache = {}


@triton.jit
def _fused_ln_linear_kernel(
    # Input: (M, D)
    x_ptr,
    # LN params
    norm_w_ptr, norm_b_ptr,
    # Weight: (K, D)  where K = 5*H
    w_ptr,
    # Output: (M, K)
    out_ptr,
    M, D, K,
    stride_xm,   # = D (row stride of x)
    stride_wk,   # = D (row stride of w)
    stride_om,   # = K (row stride of out)
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Fused LayerNorm + Linear: out[m, k] = dot(LN(x[m,:]), w[k,:])
    Grid: (cdiv(M, BLOCK_M), cdiv(K, BLOCK_K))
    Each program: load BLOCK_M rows of x, compute LN, then dot vs BLOCK_K weight rows.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = m_offs < M
    k_mask = k_offs < K

    # --- Load input x[m, :] and compute LN stats ---
    # Load full D-width rows for LN: (BLOCK_M, BLOCK_D) tiles over D
    # Accumulate mean and variance first
    mean = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        x_tile = tl.load(
            x_ptr + m_offs[:, None] * stride_xm + d_offs[None, :],
            mask=m_mask[:, None] & d_mask[None, :], other=0.0
        )  # (BLOCK_M, BLOCK_D)
        mean += tl.sum(x_tile, axis=1)
    mean = mean / D  # (BLOCK_M,)

    var = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        x_tile = tl.load(
            x_ptr + m_offs[:, None] * stride_xm + d_offs[None, :],
            mask=m_mask[:, None] & d_mask[None, :], other=0.0
        )
        diff = x_tile - mean[:, None]
        var += tl.sum(diff * diff, axis=1)
    var = var / D
    inv_std = tl.rsqrt(var + eps)  # (BLOCK_M,)

    # --- Compute GEMM: acc[m, k] = sum_d LN(x[m,d]) * w[k,d] ---
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        # Load x tile and normalize on-the-fly
        x_tile = tl.load(
            x_ptr + m_offs[:, None] * stride_xm + d_offs[None, :],
            mask=m_mask[:, None] & d_mask[None, :], other=0.0
        )  # (BLOCK_M, BLOCK_D)
        x_norm = (x_tile - mean[:, None]) * inv_std[:, None]  # (BLOCK_M, BLOCK_D)

        # Apply LN affine: norm_w[d], norm_b[d]
        nw = tl.load(norm_w_ptr + d_offs, mask=d_mask, other=0.0)  # (BLOCK_D,)
        nb = tl.load(norm_b_ptr + d_offs, mask=d_mask, other=0.0)
        x_norm = x_norm * nw[None, :] + nb[None, :]  # (BLOCK_M, BLOCK_D)

        # Load weight tile w[k, d]: (BLOCK_K, BLOCK_D)
        w_tile = tl.load(
            w_ptr + k_offs[:, None] * stride_wk + d_offs[None, :],
            mask=k_mask[:, None] & d_mask[None, :], other=0.0
        )

        # Accumulate: (BLOCK_M, BLOCK_D) @ (BLOCK_D, BLOCK_K) -> (BLOCK_M, BLOCK_K)
        acc += tl.dot(x_norm, tl.trans(w_tile))

    # Store output
    tl.store(
        out_ptr + m_offs[:, None] * stride_om + k_offs[None, :],
        acc, mask=m_mask[:, None] & k_mask[None, :]
    )


def fused_ln_linear(x, norm_w, norm_b, weight, M, D, K):
    """
    x: (M, D) — raw input (not yet LN'd)
    norm_w, norm_b: (D,) LN params
    weight: (K, D) — fused projection weight
    Returns: (M, K) = LN(x) @ weight.T
    """
    out = torch.empty(M, K, device=x.device, dtype=x.dtype)

    BLOCK_M = 16
    BLOCK_K = 64
    BLOCK_D = triton.next_power_of_2(min(D, 128))  # tile D; D=128 or 384

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

    _fused_ln_linear_kernel[grid](
        x, norm_w, norm_b, weight, out,
        M, D, K,
        D, D, K,   # strides
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        eps=1e-5,
    )
    return out


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
def _transpose_bhnn_to_mh_kernel(
    src_ptr,   # (B*H, N, N) — input
    dst_ptr,   # (M, H) = (B*N*N, H) — output  where dst[b*N²+i*N+j, h] = src[b*H+h, i, j]
    B, N, H: tl.constexpr,
    BLOCK_IJ: tl.constexpr,   # tile size over N*N spatial positions
    BLOCK_H: tl.constexpr,    # tile size over H (should equal H for H<=512)
):
    """
    Transpose (B*H, N, N) -> (M, H) where M=B*N*N.
    Grid: (B * cdiv(N*N, BLOCK_IJ),)
    Each program handles BLOCK_IJ spatial positions for one batch element,
    loading H values per position from strided src and writing contiguously to dst.
    Uses a (BLOCK_H, BLOCK_IJ) load tile for coalesced reads (H is contiguous in
    src[b*H+h, i, j] after we fix h as the fast index, but src has shape (B*H, N, N)
    so the h-dimension strides by N*N).
    We tile over IJ: for each h, load src[b*H+h, ij_tile] = contiguous in last 2 dims.
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

    # src[b*H+h, ij] = src_ptr + (b*H+h)*NN + ij
    # Load (BLOCK_H, BLOCK_IJ) tile: rows=h, cols=ij
    src_base = b * H * NN
    src_ptrs = src_ptr + src_base + h_offs[:, None] * NN + ij_offs[None, :]
    tile = tl.load(src_ptrs, mask=h_mask[:, None] & ij_mask[None, :], other=0.0)
    # tile shape: (BLOCK_H, BLOCK_IJ)

    # Write dst[b*NN+ij, h] = dst_ptr + (b*NN+ij)*H + h
    dst_base = b * NN
    dst_ptrs = dst_ptr + (dst_base + ij_offs[None, :]) * H + h_offs[:, None]
    # dst_ptrs shape: (BLOCK_H, BLOCK_IJ) — we store tile transposed
    tl.store(dst_ptrs, tile, mask=h_mask[:, None] & ij_mask[None, :])


@triton.jit
def _fused_ln_gate_kernel(
    bmm_mh_ptr,        # (M, H) — contiguous, transposed bmm result
    out_gate_ptr,      # (M, H)
    norm_w_ptr,        # (H,)
    norm_b_ptr,        # (H,)
    out_ptr,           # (M, H) output
    M, H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Fused LayerNorm + out_gate multiply over contiguous (M, H) data.
    Grid: (cdiv(M, BLOCK_M),)
    Each program handles BLOCK_M rows of H elements each.
    """
    pid    = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # Load (BLOCK_M, BLOCK_H) tile
    ptrs = bmm_mh_ptr + m_offs[:, None] * H + h_offs[None, :]
    x = tl.load(ptrs, mask=m_mask[:, None] & h_mask[None, :], other=0.0)

    # LayerNorm over H dim (axis=1)
    mean    = tl.sum(x, axis=1) / H                   # (BLOCK_M,)
    diff    = x - mean[:, None]
    var     = tl.sum(diff * diff, axis=1) / H          # (BLOCK_M,)
    inv_std = tl.rsqrt(var + eps)
    x_norm  = diff * inv_std[:, None]

    # Affine
    nw = tl.load(norm_w_ptr + h_offs, mask=h_mask)    # (BLOCK_H,)
    nb = tl.load(norm_b_ptr + h_offs, mask=h_mask)
    x_norm = x_norm * nw[None, :] + nb[None, :]

    # out_gate
    g_ptrs = out_gate_ptr + m_offs[:, None] * H + h_offs[None, :]
    gate   = tl.load(g_ptrs, mask=m_mask[:, None] & h_mask[None, :], other=0.0)
    x_norm = x_norm * gate

    # Store
    o_ptrs = out_ptr + m_offs[:, None] * H + h_offs[None, :]
    tl.store(o_ptrs, x_norm, mask=m_mask[:, None] & h_mask[None, :])


def fused_ln_gate(bmm_out, out_gate, norm_w, norm_b, B, N, H):
    """
    bmm_out:  (B*H, N, N)
    out_gate: (M, H) where M = B*N*N
    Returns:  (M, H) = LN(bmm_out[b,i,j,:]) * out_gate[b,i,j,:]
    """
    M = B * N * N
    NN = N * N

    # Step A: Transpose (B*H, N, N) -> (M, H) for coalesced LN reads
    bmm_mh = torch.empty(M, H, device=bmm_out.device, dtype=bmm_out.dtype)
    BLOCK_IJ = 64
    BLOCK_H_T = triton.next_power_of_2(H)
    grid_t = (B * triton.cdiv(NN, BLOCK_IJ),)
    _transpose_bhnn_to_mh_kernel[grid_t](
        bmm_out, bmm_mh,
        B, N, H,
        BLOCK_IJ=BLOCK_IJ,
        BLOCK_H=BLOCK_H_T,
    )

    # Step B: Fused LN + gate on contiguous (M, H) data
    out = torch.empty(M, H, device=bmm_out.device, dtype=bmm_out.dtype)
    BLOCK_M = 16
    BLOCK_H_LN = triton.next_power_of_2(H)
    grid_ln = (triton.cdiv(M, BLOCK_M),)
    _fused_ln_gate_kernel[grid_ln](
        bmm_mh, out_gate, norm_w, norm_b, out,
        M, H,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H_LN,
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
            'norm_w': weights['norm.weight'],
            'norm_b': weights['norm.bias'],
            'to_out_norm_w': weights['to_out_norm.weight'],
            'to_out_norm_b': weights['to_out_norm.bias'],
            'to_out_w': weights['to_out.weight'],
        }
    cached = _weight_cache[cache_key]

    fused_w     = cached['fused_w']
    norm_w      = cached['norm_w']
    norm_b      = cached['norm_b']
    to_out_norm_w = cached['to_out_norm_w']
    to_out_norm_b = cached['to_out_norm_b']
    to_out_w    = cached['to_out_w']
    hidden_dim  = config["hidden_dim"]
    dim         = config["dim"]

    B, N, _, D = input_tensor.shape
    M = B * N * N

    # Step 1: LayerNorm on input
    x = F.layer_norm(input_tensor, (dim,), norm_w, norm_b)

    # Step 2: Fused 5-projection matmul -> (M, 5*H)
    x_flat = x.reshape(M, dim)
    fused_out = F.linear(x_flat, fused_w)  # (M, 5*H)

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

    # Step 6: to_out linear
    out = F.linear(out_flat.reshape(B, N, N, H), to_out_w)

    return out
