"""
TriMul submission — fused-projection, no-module, TF32-enabled approach.
"""

import torch
import torch.nn.functional as F


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    # Enable TF32 for faster matmuls (tolerance is 2e-2, so this is safe)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Extract weights directly — no module construction
    norm_w = weights['norm.weight']
    norm_b = weights['norm.bias']
    left_proj_w = weights['left_proj.weight']
    right_proj_w = weights['right_proj.weight']
    left_gate_w = weights['left_gate.weight']
    right_gate_w = weights['right_gate.weight']
    out_gate_w = weights['out_gate.weight']
    to_out_norm_w = weights['to_out_norm.weight']
    to_out_norm_b = weights['to_out_norm.bias']
    to_out_w = weights['to_out.weight']

    # Step 1: LayerNorm on input
    x = F.layer_norm(input_tensor, (config["dim"],), norm_w, norm_b)

    # Step 2: Fuse all 5 projections into a single matmul
    # Stack [left_proj, right_proj, left_gate, right_gate, out_gate] -> (5*H, D)
    hidden_dim = config["hidden_dim"]
    fused_w = torch.cat([left_proj_w, right_proj_w, left_gate_w, right_gate_w, out_gate_w], dim=0)
    # x: (B, N, N, D), fused_w: (5*H, D)
    # F.linear: output (B, N, N, 5*H)
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

    # Step 4: einsum "b i k d, b j k d -> b i j d" via bmm
    # left:  (B, N, N, H) -> treat as (B*N, N, H) — first N is i, second N is k
    # right: (B, N, N, H) -> treat as (B*N, N, H) — first N is j, second N is k
    # out[b,i,j,d] = sum_k left[b,i,k,d] * right[b,j,k,d]
    # Rewrite as: for each (b, d): out[b,:,:,d] = left[b,:,:,d] @ right[b,:,:,d].T
    # Reshape to (B, H, N, N) for efficient bmm
    B, N, _, H = left.shape
    # left: (B, N, N, H) -> (B*H, N, N) ... but that's large
    # Better: (B, N, N, H) -> permute to (B, H, N, N) then bmm per hidden dim isn't great
    # Use: left (B, N, N, H), right (B, N, N, H)
    # out[b,i,j,h] = sum_k left[b,i,k,h] * right[b,j,k,h]
    # = (B, N, N, H) x (B, N, N, H) contracted on dim=2
    # Reshape left -> (B*N, N, H), right -> (B*N, N, H)
    # Then out reshaped as (B, N, N, H) by: left_flat @ right_flat.transpose(-1,-2) -> (B*N, N, N)
    # But that gives out[b*i, j, k] which is wrong dimension ordering.
    # Correct approach: left[b,i,k,h], right[b,j,k,h]
    # -> left.permute(0,3,1,2): (B,H,i,k), right.permute(0,3,2,1): (B,H,k,j)
    # -> bmm -> (B,H,i,j) -> permute(0,2,3,1) -> (B,i,j,H)
    # But B*H can be large; use reshape to (B*H, N, N)
    left_t = left.permute(0, 3, 1, 2).reshape(B * H, N, N)   # (B*H, i, k)
    right_t = right.permute(0, 3, 2, 1).reshape(B * H, N, N)  # (B*H, k, j)
    out = torch.bmm(left_t, right_t).reshape(B, H, N, N).permute(0, 2, 3, 1)  # (B, i, j, H)

    # Step 5: LayerNorm + out_gate + to_out projection
    # out_gate needs to be computed from x at position (b, i, j) — but wait:
    # out_gate in the reference is: out_gate(x).sigmoid() where x is (B,N,N,D)
    # and it's applied to out which is (B,N,N,H) — element-wise after the einsum
    # The out_gate is indexed at (b, i, j) from x — so it's (B, N, N, H)
    out = F.layer_norm(out, (H,), to_out_norm_w, to_out_norm_b)
    out = out * out_gate
    out = F.linear(out, to_out_w)

    return out
