"""
TriMul submission — fused-projection, no-module, TF32+bf16-einsum, weight caching.
"""

import torch
import torch.nn.functional as F

# Module-level weight cache: keyed on data_ptr of left_proj.weight
_weight_cache = {}


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    # Enable TF32 for faster matmuls (tolerance is 2e-2, so this is safe)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Cache the stacked fused weight matrix across calls to avoid repeated torch.cat
    # Key on all weight data_ptrs to uniquely identify each set of weights
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
        left_proj_w = weights['left_proj.weight']
        right_proj_w = weights['right_proj.weight']
        left_gate_w = weights['left_gate.weight']
        right_gate_w = weights['right_gate.weight']
        out_gate_w = weights['out_gate.weight']
        fused_w = torch.cat([left_proj_w, right_proj_w, left_gate_w, right_gate_w, out_gate_w], dim=0)
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

    # Step 1: LayerNorm on input
    x = F.layer_norm(input_tensor, (config["dim"],), norm_w, norm_b)

    # Step 2: Fuse all 5 projections into a single matmul
    fused_out = F.linear(x, fused_w)

    left_proj_out = fused_out[..., :hidden_dim]
    right_proj_out = fused_out[..., hidden_dim:2*hidden_dim]
    left_gate_out = fused_out[..., 2*hidden_dim:3*hidden_dim]
    right_gate_out = fused_out[..., 3*hidden_dim:4*hidden_dim]
    out_gate_out = fused_out[..., 4*hidden_dim:]

    # Step 3: Apply gates and mask
    mask_e = mask.unsqueeze(-1)  # (B, N, N, 1)
    left = left_proj_out * left_gate_out.sigmoid() * mask_e
    right = right_proj_out * right_gate_out.sigmoid() * mask_e
    out_gate = out_gate_out.sigmoid()

    # Step 4: einsum "b i k h, b j k h -> b i j h" via torch.einsum
    # Let PyTorch/cuBLAS pick the best contraction plan with TF32 enabled
    H = hidden_dim
    out = torch.einsum('bikh,bjkh->bijh', left, right)

    # Step 5: LayerNorm + out_gate + to_out projection
    out = F.layer_norm(out, (H,), to_out_norm_w, to_out_norm_b)
    out = out * out_gate
    out = F.linear(out, to_out_w)

    return out
