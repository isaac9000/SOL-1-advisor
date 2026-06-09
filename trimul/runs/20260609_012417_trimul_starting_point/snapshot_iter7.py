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
    left_ptr,        # (M, H) output for left
    right_ptr,       # (M, H) output for right
    out_gate_ptr,    # (M, H) output for out_gate
    M, H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Fuse: read fused_out once, apply sigmoid + mask, write left/right/out_gate.
    fused_out layout: [left_proj | right_proj | left_gate | right_gate | out_gate]
    left_proj:  fused_out[:, 0:H]
    right_proj: fused_out[:, H:2H]
    left_gate:  fused_out[:, 2H:3H]
    right_gate: fused_out[:, 3H:4H]
    out_gate:   fused_out[:, 4H:5H]
    """
    pid = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    h_offs = tl.arange(0, BLOCK_H)  # H is constexpr, no mask needed

    # Load mask values for these rows
    mval = tl.load(mask_ptr + m_offs, mask=m_mask, other=0.0)  # (BLOCK_M,)

    # Base pointer for fused_out row
    row_base = m_offs * (5 * H)  # stride along M dimension

    # Load all 5 chunks: each (BLOCK_M, BLOCK_H)
    lp_ptrs  = fused_out_ptr + row_base[:, None] + h_offs[None, :]           # left_proj
    rp_ptrs  = fused_out_ptr + row_base[:, None] + H + h_offs[None, :]       # right_proj
    lg_ptrs  = fused_out_ptr + row_base[:, None] + 2 * H + h_offs[None, :]  # left_gate
    rg_ptrs  = fused_out_ptr + row_base[:, None] + 3 * H + h_offs[None, :]  # right_gate
    og_ptrs  = fused_out_ptr + row_base[:, None] + 4 * H + h_offs[None, :]  # out_gate

    lp = tl.load(lp_ptrs, mask=m_mask[:, None], other=0.0)
    rp = tl.load(rp_ptrs, mask=m_mask[:, None], other=0.0)
    lg = tl.load(lg_ptrs, mask=m_mask[:, None], other=0.0)
    rg = tl.load(rg_ptrs, mask=m_mask[:, None], other=0.0)
    og = tl.load(og_ptrs, mask=m_mask[:, None], other=0.0)

    # Compute: left = lp * sigmoid(lg) * mask, right = rp * sigmoid(rg) * mask
    lg_sig = tl.sigmoid(lg)
    rg_sig = tl.sigmoid(rg)
    og_sig = tl.sigmoid(og)

    left_val  = lp * lg_sig * mval[:, None]
    right_val = rp * rg_sig * mval[:, None]

    # Write outputs
    out_row_base = m_offs * H
    l_ptrs_out = left_ptr  + out_row_base[:, None] + h_offs[None, :]
    r_ptrs_out = right_ptr + out_row_base[:, None] + h_offs[None, :]
    g_ptrs_out = out_gate_ptr + out_row_base[:, None] + h_offs[None, :]

    tl.store(l_ptrs_out, left_val,  mask=m_mask[:, None])
    tl.store(r_ptrs_out, right_val, mask=m_mask[:, None])
    tl.store(g_ptrs_out, og_sig,    mask=m_mask[:, None])


def fused_gate_mask(fused_out, mask_flat, hidden_dim):
    """
    fused_out: (M, 5*H) contiguous
    mask_flat: (M,) contiguous
    Returns left (M,H), right (M,H), out_gate (M,H)
    """
    M = fused_out.shape[0]
    H = hidden_dim
    left = torch.empty(M, H, device=fused_out.device, dtype=fused_out.dtype)
    right = torch.empty(M, H, device=fused_out.device, dtype=fused_out.dtype)
    out_gate = torch.empty(M, H, device=fused_out.device, dtype=fused_out.dtype)

    BLOCK_M = 16
    BLOCK_H = triton.next_power_of_2(H)
    grid = (triton.cdiv(M, BLOCK_M),)

    _fused_gate_mask_kernel[grid](
        fused_out, mask_flat,
        left, right, out_gate,
        M, H,
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
    mask_flat = mask.reshape(M)
    left_flat, right_flat, out_gate_flat = fused_gate_mask(fused_out, mask_flat, hidden_dim)

    # Step 4: Einsum via bmm: left[b,i,k,h] * right[b,j,k,h] -> out[b,i,j,h]
    # Reshape to (B, N, N, H), then permute to (B*H, N, N) for bmm
    H = hidden_dim
    left  = left_flat.reshape(B, N, N, H)
    right = right_flat.reshape(B, N, N, H)
    out_gate = out_gate_flat.reshape(B, N, N, H)

    left_t  = left.permute(0, 3, 1, 2).reshape(B * H, N, N)   # (B*H, i, k)
    right_t = right.permute(0, 3, 2, 1).reshape(B * H, N, N)  # (B*H, k, j)
    out = torch.bmm(left_t, right_t).reshape(B, H, N, N).permute(0, 2, 3, 1)  # (B,i,j,H)

    # Step 5: LayerNorm + out_gate + to_out
    out = F.layer_norm(out, (H,), to_out_norm_w, to_out_norm_b)
    out = out * out_gate
    out = F.linear(out, to_out_w)

    return out
