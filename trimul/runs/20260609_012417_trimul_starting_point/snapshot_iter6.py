"""
TriMul submission — custom Triton kernel for fused einsum + layernorm + gate.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Module-level weight cache
_weight_cache = {}


@triton.jit
def _trimul_einsum_kernel(
    left_ptr, right_ptr,
    out_gate_ptr,
    to_out_norm_w_ptr, to_out_norm_b_ptr,
    out_ptr,
    N, H,
    stride_lb, stride_li, stride_lk,  # left: b,i,k strides (last dim=H is stride 1)
    stride_rb, stride_rj, stride_rk,
    stride_gb, stride_gi, stride_gj,
    stride_ob, stride_oi, stride_oj,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    eps: tl.constexpr,
):
    """
    One program per (b, i, j) output position.
    Computes out[b,i,j,:] = sum_k left[b,i,k,:] * right[b,j,k,:]
    then applies LayerNorm + norm_affine + out_gate.
    Grid: (B * N * N,)
    """
    pid = tl.program_id(0)
    # Decode (b, i, j) from linear pid
    j = pid % N
    ij = pid // N
    i = ij % N
    b = ij // N

    h_offs = tl.arange(0, BLOCK_H)   # (BLOCK_H,)
    h_mask = h_offs < H

    # Base pointers for left[b, i, :, :] and right[b, j, :, :]
    left_base = left_ptr + b * stride_lb + i * stride_li
    right_base = right_ptr + b * stride_rb + j * stride_rj

    # Accumulator over H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Loop over k tiles
    for k_start in range(0, N, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < N

        # Load left[b, i, k_tile, h]: (BLOCK_K, BLOCK_H)
        l_ptrs = left_base + k_offs[:, None] * stride_lk + h_offs[None, :]
        l_tile = tl.load(l_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0)

        # Load right[b, j, k_tile, h]: (BLOCK_K, BLOCK_H)
        r_ptrs = right_base + k_offs[:, None] * stride_rk + h_offs[None, :]
        r_tile = tl.load(r_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0)

        # Accumulate: sum over k, elementwise over h
        acc = acc + tl.sum(l_tile * r_tile, axis=0)  # (BLOCK_H,)

    # LayerNorm over H
    mean = tl.sum(acc, axis=0) / H
    diff = acc - mean
    var = tl.sum(diff * diff, axis=0) / H
    inv_std = tl.rsqrt(var + eps)
    acc_norm = diff * inv_std

    # Affine
    norm_w = tl.load(to_out_norm_w_ptr + h_offs, mask=h_mask)
    norm_b = tl.load(to_out_norm_b_ptr + h_offs, mask=h_mask)
    acc_norm = acc_norm * norm_w + norm_b

    # out_gate[b, i, j, h]
    gate_ptr = out_gate_ptr + b * stride_gb + i * stride_gi + j * stride_gj
    gate = tl.load(gate_ptr + h_offs, mask=h_mask)
    acc_norm = acc_norm * gate

    # Store out[b, i, j, h]
    out_base_ptr = out_ptr + b * stride_ob + i * stride_oi + j * stride_oj
    tl.store(out_base_ptr + h_offs, acc_norm, mask=h_mask)


def trimul_triton(left, right, out_gate, to_out_norm_w, to_out_norm_b):
    """
    Fused: einsum(bikh,bjkh->bijh) + LayerNorm(H) + out_gate multiply
    Returns (B, N, N, H)
    """
    B, N, _, H = left.shape

    out = torch.empty(B, N, N, H, device=left.device, dtype=torch.float32)

    left = left.contiguous()
    right = right.contiguous()
    out_gate = out_gate.contiguous()

    stride_lb, stride_li, stride_lk = left.stride(0), left.stride(1), left.stride(2)
    stride_rb, stride_rj, stride_rk = right.stride(0), right.stride(1), right.stride(2)
    stride_gb, stride_gi, stride_gj = out_gate.stride(0), out_gate.stride(1), out_gate.stride(2)
    stride_ob, stride_oi, stride_oj = out.stride(0), out.stride(1), out.stride(2)

    BLOCK_K = 32
    BLOCK_H = triton.next_power_of_2(H)

    grid = (B * N * N,)

    _trimul_einsum_kernel[grid](
        left, right, out_gate,
        to_out_norm_w, to_out_norm_b,
        out,
        N, H,
        stride_lb, stride_li, stride_lk,
        stride_rb, stride_rj, stride_rk,
        stride_gb, stride_gi, stride_gj,
        stride_ob, stride_oi, stride_oj,
        BLOCK_K=BLOCK_K,
        BLOCK_H=BLOCK_H,
        eps=1e-5,
    )
    return out


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

    fused_w = cached['fused_w']
    norm_w = cached['norm_w']
    norm_b = cached['norm_b']
    to_out_norm_w = cached['to_out_norm_w']
    to_out_norm_b = cached['to_out_norm_b']
    to_out_w = cached['to_out_w']
    hidden_dim = config["hidden_dim"]
    dim = config["dim"]

    # Step 1: LayerNorm on input
    x = F.layer_norm(input_tensor, (dim,), norm_w, norm_b)

    # Step 2: Fused 5-projection matmul
    fused_out = F.linear(x, fused_w)
    left_proj_out = fused_out[..., :hidden_dim]
    right_proj_out = fused_out[..., hidden_dim:2*hidden_dim]
    left_gate_out = fused_out[..., 2*hidden_dim:3*hidden_dim]
    right_gate_out = fused_out[..., 3*hidden_dim:4*hidden_dim]
    out_gate_out = fused_out[..., 4*hidden_dim:]

    # Step 3: Apply gates and mask
    mask_e = mask.unsqueeze(-1)
    left = left_proj_out * left_gate_out.sigmoid() * mask_e
    right = right_proj_out * right_gate_out.sigmoid() * mask_e
    out_gate = out_gate_out.sigmoid()

    # Steps 4+5: Fused Triton: einsum + layernorm + gate
    out = trimul_triton(left, right, out_gate, to_out_norm_w, to_out_norm_b)

    # Step 6: to_out linear projection
    out = F.linear(out, to_out_w)

    return out
