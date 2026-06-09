"""
TriMul submission — fused gate+mask Triton kernel, PyTorch bmm for einsum.
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
    out = out_bhn.reshape(B, H, N, N).permute(0, 2, 3, 1)  # (B, N, N, H)
    out_gate = out_gate_flat.reshape(B, N, N, H)

    # Step 5: LayerNorm + out_gate + to_out
    out = F.layer_norm(out, (H,), to_out_norm_w, to_out_norm_b)
    out = out * out_gate
    out = F.linear(out, to_out_w)

    return out
