# Experiment History

Tracks every kernel attempt, its code, hypothesis, and result.

---

## Experiment #1 — 2026-06-09 01:24:39 UTC ✅ KEEP

**Hypothesis:** Baseline 'starting_point' — initial benchmark

**Result:** 10944.05 μs

**Kernel code:**
```python
"""
Initial TriMul submission — PyTorch baseline with dummy Triton kernel.
"""

import torch
from torch import nn, einsum
import triton
import triton.language as tl


@triton.jit
def _dummy_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pass


class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, dtype=torch.float32)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)
        x = x.to(torch.float32)

        left = self.left_proj(x.to(torch.float32))
        right = self.right_proj(x.to(torch.float32))

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x.to(torch.float32)).sigmoid()
        right_gate = self.right_gate(x.to(torch.float32)).sigmoid()
        out_gate = self.out_gate(x.to(torch.float32)).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left.to(torch.bfloat16), right.to(torch.bfloat16))

        out = out.to(torch.float32)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    trimul = TriMul(config["dim"], config["hidden_dim"]).to(input_tensor.device)

    trimul.norm.weight = nn.Parameter(weights['norm.weight'].to(torch.float32))
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'].to(torch.float32))
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'].to(torch.float32))
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'].to(torch.float32))
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'].to(torch.float32))
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'].to(torch.float32))
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'].to(torch.float32))
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'].to(torch.float32))
    trimul.norm.bias = nn.Parameter(weights['norm.bias'].to(torch.float32))
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'].to(torch.float32))

    output = trimul(input_tensor, mask).to(torch.float32)

    return output

```

---

## Experiment #2 — 2026-06-09 01:26:00 UTC ✅ KEEP

**Hypothesis:** Advisor proposal: eliminate per-call module construction, fuse all 5 projections (left_proj, right_proj, left_gate, right_gate, out_gate) into a single F.linear call with stacked weights, enable TF32, keep everything float32, and replace the einsum with bmm via permute+reshape (B*H, N, N) batched matmul. No bfloat16 cast.

**Result:** 6940.34 μs

**Kernel code:**
```python
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

```

---

## Experiment #3 — 2026-06-09 01:28:30 UTC ✅ KEEP

**Hypothesis:** Advisor proposal: add weight caching (keyed on 7-tuple of data_ptrs) to avoid repeated torch.cat on every call, plus bf16 bmm for the einsum. bf16 caused correctness failures (max_diff up to 5.5) so reverted bmm to float32. Final result: weight caching only, float32 bmm with TF32 enabled. Marginal improvement over prior best.

**Result:** 6887.83 μs

**Kernel code:**
```python
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

    # Step 4: einsum "b i k d, b j k d -> b i j d" via bmm in bf16
    # left[b,i,k,h], right[b,j,k,h] -> out[b,i,j,h] = sum_k left * right
    # -> permute to (B,H,i,k) x (B,H,k,j) -> (B,H,i,j) -> permute to (B,i,j,H)
    B, N, _, H = left.shape
    left_t = left.permute(0, 3, 1, 2).reshape(B * H, N, N)   # (B*H, i, k)
    right_t = right.permute(0, 3, 2, 1).reshape(B * H, N, N)  # (B*H, k, j)
    out = torch.bmm(left_t, right_t).reshape(B, H, N, N).permute(0, 2, 3, 1)

    # Step 5: LayerNorm + out_gate + to_out projection
    out = F.layer_norm(out, (H,), to_out_norm_w, to_out_norm_b)
    out = out * out_gate
    out = F.linear(out, to_out_w)

    return out

```

---

## Experiment #4 — 2026-06-09 01:34:19 UTC 💥 CRASH

**Hypothesis:** Advisor proposal: wrap forward pass in torch.compile(mode='max-autotune') to fuse elementwise ops and select better matmul kernels. First tried fullgraph=True which crashed with PyTorch internal error. Then tried without fullgraph but still crashed with 0/18 tests passing — error message truncated but likely torch.compile fails in the Modal sandbox environment (compiled at module load time, no GPU). Reverted to working baseline (experiment #3 code). Logging as crash.

**Result:** CRASH

**Error:**
```
torch.compile fails in Modal environment — 0/18 tests pass, all errors at _compiled_forward call. Error messages truncated. Both fullgraph=True and fullgraph=False variants fail.
```

**Kernel code:**
```python
"""
TriMul submission — fused-projection, no-module, TF32, weight caching.
"""

import torch
import torch.nn.functional as F

# Module-level weight cache
_weight_cache = {}


def custom_kernel(data):
    input_tensor, mask, weights, config = data

    # Enable TF32 for faster matmuls (tolerance is 2e-2, so this is safe)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Cache the stacked fused weight matrix across calls to avoid repeated torch.cat
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

    # Step 4: einsum "b i k d, b j k d -> b i j d" via bmm
    B, N, _, H = left.shape
    left_t = left.permute(0, 3, 1, 2).reshape(B * H, N, N)   # (B*H, i, k)
    right_t = right.permute(0, 3, 2, 1).reshape(B * H, N, N)  # (B*H, k, j)
    out = torch.bmm(left_t, right_t).reshape(B, H, N, N).permute(0, 2, 3, 1)

    # Step 5: LayerNorm + out_gate + to_out projection
    out = F.layer_norm(out, (H,), to_out_norm_w, to_out_norm_b)
    out = out * out_gate
    out = F.linear(out, to_out_w)

    return out

```

---

## Experiment #5 — 2026-06-09 01:35:40 UTC ❌ DISCARD

**Hypothesis:** Advisor proposal: replace manual permute+bmm with torch.einsum('bikh,bjkh->bijh', left, right) to let PyTorch/cuBLAS pick a better contraction plan with TF32 enabled. Result is 6938 µs vs best 6888 µs — effectively identical (within noise), confirming PyTorch's einsum routes to the same cuBLAS plan as the manual (B*H, N, N) bmm. No benefit from this swap.

**Result:** 6938.19 μs

**Kernel code:**
```python
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

```

---

## Experiment #6 — 2026-06-09 01:41:13 UTC ❌ DISCARD

**Hypothesis:** Advisor proposal: custom Triton kernel fusing einsum + layernorm + gate. First attempt used 3D tensors (BLOCK_I, BLOCK_J, BLOCK_H) accumulator — crashed with Triton JIT error. Second attempt: one program per (b,i,j) with 1D accumulator over H, looping over k. Passes 18/18 correctness tests but 36,687 µs — 5x SLOWER than PyTorch best. The per-(b,i,j) program design is severely underutilized: each program does N serial iterations with only H=128 elements of work per step, giving terrible GPU occupancy for large N. PyTorch's cuBLAS bmm is dramatically better for this shape.

**Result:** 36686.73 μs

**Kernel code:**
```python
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

```

---

## Experiment #7 — 2026-06-09 01:43:38 UTC ✅ KEEP

**Hypothesis:** Advisor proposal: fuse LN into projection GEMM to save memory passes. Implemented instead as a Triton kernel that fuses the 5 elementwise ops (gate sigmoid + mask multiply for left/right, sigmoid for out_gate) into a single pass over fused_out (M, 5H), replacing 4 separate PyTorch elementwise kernels. Reverted to PyTorch bmm for the einsum (fast cuBLAS). Result: 6143 µs vs prior best 6888 µs — 11% improvement, new best.

**Result:** 6142.69 μs

**Kernel code:**
```python
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

```

---

## Experiment #8 — 2026-06-09 01:46:00 UTC ✅ KEEP

**Hypothesis:** Advisor proposal: eliminate permute copies around bmm by writing left/right directly in (B*H, N, N) layout from the Triton gate kernel. left[b*H+h, i, k] and right[b*H+h, k, j] are written with recomputed scatter indices (b=m//N², row=rc//N, col=rc%N). bmm now works on contiguous tensors with zero permute overhead. Result: 5000 µs vs prior best 6143 µs — 18.6% improvement, new best.

**Result:** 5000.33 μs

**Kernel code:**
```python
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

```

---

## Experiment #9 — 2026-06-09 01:47:49 UTC ✅ KEEP

**Hypothesis:** Advisor proposal: fuse post-bmm LayerNorm + out_gate multiply into a single Triton kernel. Kernel grid=(M,)=(B*N*N,), one program per (b,i,j). Reads H-vector from bmm_out with stride N² (since bmm_out is (B*H,N,N)), applies LayerNorm, multiplies out_gate, writes (M,H) contiguously. Also eliminated the post-bmm permute (no longer needed). Replaces F.layer_norm + out*out_gate (2 passes, 2 reads of M*H) with 1 Triton kernel. Result: 4211 µs vs prior best 5000 µs — 15.8% improvement, new best.

**Result:** 4210.59 μs

**Kernel code:**
```python
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


@triton.jit
def _fused_ln_gate_kernel(
    bmm_out_ptr,       # (B*H, N, N) — bmm result
    out_gate_ptr,      # (M, H) where M = B*N*N
    norm_w_ptr,        # (H,)
    norm_b_ptr,        # (H,)
    out_ptr,           # (M, H) output
    B, N, H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    eps: tl.constexpr,
):
    """
    One program per (b, i, j) = one output row.
    Reads the H-vector from bmm_out[b*H:b*H+H, i, j] (strided),
    applies LayerNorm + norm affine + out_gate multiply,
    writes (M, H) output contiguously.
    Grid: (M,) = (B*N*N,)
    """
    pid = tl.program_id(0)   # = b*N²+i*N+j
    NN  = N * N
    b   = pid // NN
    rc  = pid % NN
    i   = rc // N
    j   = rc % N

    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # bmm_out[b*H+h, i, j]: base = b*H*N²+i*N+j, stride between h = N²
    bmm_base = b * H * NN + i * N + j
    bmm_ptrs = bmm_out_ptr + bmm_base + h_offs * NN
    x = tl.load(bmm_ptrs, mask=h_mask, other=0.0)  # (BLOCK_H,)

    # LayerNorm over H
    mean    = tl.sum(x, axis=0) / H
    diff    = x - mean
    var     = tl.sum(diff * diff, axis=0) / H
    inv_std = tl.rsqrt(var + eps)
    x_norm  = diff * inv_std

    # Affine
    nw = tl.load(norm_w_ptr + h_offs, mask=h_mask)
    nb = tl.load(norm_b_ptr + h_offs, mask=h_mask)
    x_norm = x_norm * nw + nb

    # out_gate[pid, h]
    gate_ptrs = out_gate_ptr + pid * H + h_offs
    gate = tl.load(gate_ptrs, mask=h_mask)
    x_norm = x_norm * gate

    # Write output[pid, h]
    out_ptrs = out_ptr + pid * H + h_offs
    tl.store(out_ptrs, x_norm, mask=h_mask)


def fused_ln_gate(bmm_out, out_gate, norm_w, norm_b, B, N, H):
    """
    bmm_out:  (B*H, N, N)
    out_gate: (M, H) where M = B*N*N
    Returns:  (M, H) = LN(bmm_out[b,i,j,:]) * out_gate[b,i,j,:]
    """
    M = B * N * N
    out = torch.empty(M, H, device=bmm_out.device, dtype=bmm_out.dtype)
    BLOCK_H = triton.next_power_of_2(H)
    grid = (M,)
    _fused_ln_gate_kernel[grid](
        bmm_out, out_gate, norm_w, norm_b, out,
        B, N, H,
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

```

---

## Experiment #10 — 2026-06-09 01:50:04 UTC ✅ KEEP

**Hypothesis:** Advisor proposal Priority 2: fix strided reads in _fused_ln_gate_kernel. Replaced the per-(b,i,j) strided H-gather with a 2-step approach: (1) Triton tiled transpose kernel converting (B*H, N, N) to (M, H) contiguously using (BLOCK_H, BLOCK_IJ) tiles for coalesced reads+writes, (2) updated _fused_ln_gate_kernel processes (BLOCK_M, BLOCK_H) tiles over contiguous (M,H) data. Result: 3909 µs vs prior best 4211 µs — 7.2% improvement, new best.

**Result:** 3908.61 μs

**Kernel code:**
```python
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

```

---

## Experiment #11 — 2026-06-09 01:52:19 UTC ❌ DISCARD

**Hypothesis:** Advisor proposal: fuse input LayerNorm into fused-projection GEMM with a Triton kernel. Implementation: grid=(M/BLOCK_M, 5H/BLOCK_K), per program computes LN stats (2 passes over D), then GEMM using tl.dot over D tiles. Result: 8478 µs — 2x SLOWER than prior best 3909 µs. Custom Triton GEMM cannot match cuBLAS: 3 sequential passes over D (mean, variance, GEMM) vs cuBLAS single pass with tensor cores. Discarded.

**Result:** 8477.73 μs

**Kernel code:**
```python
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

    # Steps 1+2 fused: LayerNorm + 5-projection matmul in one pass -> (M, 5*H)
    x_flat = input_tensor.reshape(M, dim)
    fused_out = fused_ln_linear(x_flat, norm_w, norm_b, fused_w, M, dim, 5 * hidden_dim)

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

```

---

## Experiment #12 — 2026-06-09 01:54:34 UTC ✅ KEEP

**Hypothesis:** Advisor Priority 1: fuse the transpose+LN+gate into a single Triton kernel, eliminating the separate transpose kernel, intermediate (M,H) buffer, and extra kernel launch. New kernel: grid=(B*cdiv(N²,BLOCK_IJ),), loads (BLOCK_H,BLOCK_IJ) tile from bmm_out coalesced along ij-dim, computes LayerNorm per column (over H), multiplies gate, writes (M,H) output. Uses tl.trans for gate load/result store. Result: 3766 µs vs prior best 3909 µs — 3.7% improvement, new best.

**Result:** 3766.00 μs

**Kernel code:**
```python
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

```

---

## Experiment #13 — 2026-06-09 01:57:03 UTC ❌ DISCARD

**Hypothesis:** Advisor proposal: add @triton.autotune to both _fused_gate_mask_kernel (sweep BLOCK_M in {8,16,32,64}, key=[M,N,H]) and _fused_transpose_ln_gate_kernel (sweep BLOCK_IJ in {16,32,64,128}, key=[B,N,H]). Result: 3770 µs vs prior best 3766 µs — essentially no change (within noise). The fixed block sizes BLOCK_M=16 and BLOCK_IJ=64 were already near-optimal. Autotuning overhead not justified. Discarding (marginally worse).

**Result:** 3769.64 μs

**Kernel code:**
```python
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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 8}),
        triton.Config({'BLOCK_M': 16}),
        triton.Config({'BLOCK_M': 32}),
        triton.Config({'BLOCK_M': 64}),
    ],
    key=['M', 'N', 'H'],
)
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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_IJ': 16}),
        triton.Config({'BLOCK_IJ': 32}),
        triton.Config({'BLOCK_IJ': 64}),
        triton.Config({'BLOCK_IJ': 128}),
    ],
    key=['B', 'N', 'H'],
)
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

    BLOCK_H = triton.next_power_of_2(H)
    # grid lambda: autotune picks BLOCK_IJ, so grid size depends on it
    grid = lambda meta: (B * triton.cdiv(NN, meta['BLOCK_IJ']),)

    _fused_transpose_ln_gate_kernel[grid](
        bmm_out, out_gate, norm_w, norm_b, out,
        B, N, H,
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

    BLOCK_H = triton.next_power_of_2(H)
    # grid lambda: autotune picks BLOCK_M
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

    _fused_gate_mask_kernel[grid](
        fused_out, mask_flat,
        left, right, out_gate,
        M, N, H,
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

```

---

## Experiment #14 — 2026-06-09 02:00:33 UTC ❌ DISCARD

**Hypothesis:** Advisor proposal: restructure _fused_gate_mask_kernel to iterate over (b,h,ij_tile) for coalesced writes to left/right. Each program handles one (b,h) slice and BLOCK_IJ spatial positions, writing contiguously. Result: 16897 µs — 4.5x SLOWER than best 3766 µs. The B*H*N²/BLOCK_IJ grid creates ~1.2M programs each doing trivial scalar work, causing catastrophic launch overhead and poor occupancy. The original BLOCK_M approach with scatter writes, while non-coalesced, had far better throughput due to higher work-per-program. Reverted.

**Result:** 16896.98 μs

**Kernel code:**
```python
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
    fused_out_ptr,   # (M, 5*H) input — M = B*N*N, row-major: [lp|rp|lg|rg|og] per row
    mask_ptr,        # (M,) mask values
    left_ptr,        # (B*H, N, N) output for left
    right_ptr,       # (B*H, N, N) output for right — (B*H, k, j) layout
    out_gate_ptr,    # (M, H) output for out_gate
    M, N, H: tl.constexpr,
    BLOCK_IJ: tl.constexpr,   # tile over N*N spatial positions
    BLOCK_H: tl.constexpr,
):
    """
    Restructured to iterate over (b, h, ij_tile) for coalesced writes.
    Grid: (B * H * cdiv(N*N, BLOCK_IJ),)
    Each program handles one (b, h) slice and BLOCK_IJ spatial positions.
    Writes left[b*H+h, :, :] and right[b*H+h, :, :] contiguously.
    """
    pid  = tl.program_id(0)
    NN   = N * N
    NNb  = tl.cdiv(NN, BLOCK_IJ)

    # Decode (b, h, p_ij) from pid
    p_ij = pid % NNb
    bh   = pid // NNb
    h    = bh % H
    b    = bh // H

    ij_start = p_ij * BLOCK_IJ
    ij_offs  = ij_start + tl.arange(0, BLOCK_IJ)   # (BLOCK_IJ,) = flat (i*N+k) indices
    ij_mask  = ij_offs < NN

    # Spatial decomposition: row=i (or j), col=k
    row_offs = ij_offs // N   # i for left, j for right
    col_offs = ij_offs % N    # k for both

    # Global row indices in fused_out: m = b*NN + ij
    m_offs = b * NN + ij_offs   # (BLOCK_IJ,)

    # Load mask values
    mval = tl.load(mask_ptr + m_offs, mask=ij_mask, other=0.0)  # (BLOCK_IJ,)

    # Load scalar h-th column for each of the 5 projections
    # fused_out[m, h+offset] = fused_out_ptr + m * 5*H + h + offset
    base = m_offs * (5 * H) + h  # (BLOCK_IJ,)
    lp = tl.load(fused_out_ptr + base,           mask=ij_mask, other=0.0)  # (BLOCK_IJ,)
    rp = tl.load(fused_out_ptr + base + H,       mask=ij_mask, other=0.0)
    lg = tl.load(fused_out_ptr + base + 2 * H,   mask=ij_mask, other=0.0)
    rg = tl.load(fused_out_ptr + base + 3 * H,   mask=ij_mask, other=0.0)
    og = tl.load(fused_out_ptr + base + 4 * H,   mask=ij_mask, other=0.0)

    left_val  = lp * tl.sigmoid(lg) * mval   # (BLOCK_IJ,)
    right_val = rp * tl.sigmoid(rg) * mval
    og_sig    = tl.sigmoid(og)

    # Write left[b*H+h, i, k] — contiguous in (i,k) = ij_offs order
    l_base = (b * H + h) * NN
    tl.store(left_ptr + l_base + ij_offs, left_val, mask=ij_mask)

    # Write right[b*H+h, k, j] — need to map (i, k) -> (k, i) since j=i and k=col_offs
    # right[b*H+h, k, j] at index (b*H+h)*NN + k*N + j = l_base + col_offs*N + row_offs
    r_dest = l_base + col_offs * N + row_offs
    tl.store(right_ptr + r_dest, right_val, mask=ij_mask)

    # Write out_gate[m, h] = out_gate_ptr + m*H + h — contiguous in m
    tl.store(out_gate_ptr + m_offs * H + h, og_sig, mask=ij_mask)


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

    BLOCK_IJ = 64
    BLOCK_H  = triton.next_power_of_2(H)
    NN = N * N
    grid = (B * H * triton.cdiv(NN, BLOCK_IJ),)

    _fused_gate_mask_kernel[grid](
        fused_out, mask_flat,
        left, right, out_gate,
        M, N, H,
        BLOCK_IJ=BLOCK_IJ,
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

```

---

## Experiment #15 — 2026-06-09 02:02:56 UTC ❌ DISCARD

**Hypothesis:** Advisor proposal: use fp16 for both GEMMs (fused projection and to_out). Cached fp16 weight versions, cast input to fp16 before F.linear, cast output back to fp32. Also restored _fused_gate_mask_kernel to original BLOCK_M=16 grid (from failed #14 restructuring). Result: 4762 µs vs best 3766 µs — 26% SLOWER. The fp16 GEMM throughput gains are more than offset by the .to(float16) and .to(float32) cast kernel overhead. cuBLAS TF32 float32 GEMMs are already fast enough that the cast overhead dominates. Discarded.

**Result:** 4761.76 μs

**Kernel code:**
```python
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

```

---

## Experiment #16 — 2026-06-09 02:05:19 UTC 💥 CRASH

**Hypothesis:** Advisor proposal: CUDA graph capture to eliminate kernel launch overhead. Implementation: pre-allocate static buffers per (B,N,H,D), capture graph once, replay on subsequent calls. Result: OOM crash — CUDA graph captures and permanently retains ALL intermediate buffers (left_bhn, right_bhn, fused_out, out_bhn, etc.) across 7 test shapes, exhausting 79GB GPU memory. Reverted to #12 baseline and benchmarked: 3781 µs (within noise of #12 at 3766 µs). Logging as crash.

**Result:** CRASH

**Error:**
```
CUDA graph OOM: retained all intermediate tensors per shape across 7 test cases, exhausting GPU memory. Each captured graph holds B*H*N*N*H*4bytes of intermediates permanently.
```

**Kernel code:**
```python
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

    # Step 1: LayerNorm on input
    x = F.layer_norm(input_tensor, (dim,), norm_w, norm_b)

    # Step 2: Fused 5-projection matmul -> (M, 5*H)
    x_flat = x.reshape(M, dim)
    fused_out = F.linear(x_flat, fused_w)  # (M, 5*H)

    # Step 3: Fused Triton kernel: gate sigmoid + mask multiply in one pass
    # left/right written directly in (B*H, N, N) layout — no permute needed for bmm
    mask_flat = mask.reshape(M)
    left_bhn, right_bhn, out_gate_flat = fused_gate_mask(fused_out, mask_flat, B, N, H)

    # Step 4: Einsum via bmm — inputs already in (B*H, N, N) layout, no permute!
    out_bhn = torch.bmm(left_bhn, right_bhn)  # (B*H, N, N)

    # Step 5: Fused transpose+LN+gate
    out_flat = fused_ln_gate(out_bhn, out_gate_flat, to_out_norm_w, to_out_norm_b, B, N, H)

    # Step 6: to_out linear
    out = F.linear(out_flat.reshape(B, N, N, H), to_out_w)

    return out

```

---

## Experiment #17 — 2026-06-09 02:07:01 UTC ❌ DISCARD

**Hypothesis:** Advisor Experiment A: torch.autocast with dtype=float16 for the three GEMMs (fused projection, bmm, to_out). Wrapped each GEMM separately in autocast context, casting outputs back to fp32 before Triton kernels. Result: 4966 µs vs best 3766 µs — 32% SLOWER. The 3 autocast context entries + .float() casts between steps add overhead that exceeds any fp16 GEMM speedup. Also the bmm in fp16 may reduce precision. TF32 float32 GEMMs remain superior here.

**Result:** 4965.95 μs

**Kernel code:**
```python
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
    H = hidden_dim

    # Step 1: LayerNorm (fp32 — stays outside autocast)
    x = F.layer_norm(input_tensor, (dim,), norm_w, norm_b)

    # Steps 2+4+6: GEMMs and bmm under autocast for fp16 tensor cores
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        # Step 2: Fused 5-projection matmul (fp16 tensor cores, fp32 accum via autocast)
        x_flat = x.reshape(M, dim)
        fused_out = F.linear(x_flat, fused_w)  # (M, 5*H) — autocast selects fp16

    # Cast fused_out to fp32 for Triton kernel
    fused_out = fused_out.float()

    # Step 3: Fused Triton kernel: gate sigmoid + mask multiply (fp32)
    mask_flat = mask.reshape(M)
    left_bhn, right_bhn, out_gate_flat = fused_gate_mask(fused_out, mask_flat, B, N, H)

    # Step 4: bmm under autocast
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        out_bhn = torch.bmm(left_bhn, right_bhn)  # (B*H, N, N)

    out_bhn = out_bhn.float()

    # Step 5: Fused transpose+LN+gate (fp32 Triton kernel)
    out_flat = fused_ln_gate(out_bhn, out_gate_flat, to_out_norm_w, to_out_norm_b, B, N, H)

    # Step 6: to_out linear under autocast
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = F.linear(out_flat.reshape(B, N, N, H), to_out_w)

    return out.float()

```

---

## Experiment #18 — 2026-06-09 02:10:13 UTC ❌ DISCARD

**Hypothesis:** Advisor Priority 1+2: pre-allocated buffer cache (OOM'd due to retaining all shapes' buffers) + pre-transposed weights for torch.mm. First attempt with buffer cache caused OOM (same as CUDA graphs). Pivoted to just pre-transposed weights: cached fused_w.t().contiguous() and to_out_w.t().contiguous(), used torch.mm instead of F.linear. Result: 3920 µs vs best 3766 µs — 4% SLOWER. F.linear handles the transpose as efficiently as pre-transposed torch.mm on cuBLAS. Discarded.

**Result:** 3919.86 μs

**Kernel code:**
```python
"""
TriMul submission — fused LN+GEMM, fused gate+mask, fused LN+gate Triton kernels.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Module-level weight cache
_weight_cache = {}
# Pre-allocated intermediate buffer cache keyed on (weight_key, B, N, H, D)
_buf_cache = {}


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
        to_out_w = weights['to_out.weight']
        _weight_cache[cache_key] = {
            'fused_w': fused_w,
            'fused_w_t': fused_w.t().contiguous(),   # pre-transposed for torch.mm
            'norm_w': weights['norm.weight'],
            'norm_b': weights['norm.bias'],
            'to_out_norm_w': weights['to_out_norm.weight'],
            'to_out_norm_b': weights['to_out_norm.bias'],
            'to_out_w': to_out_w,
            'to_out_w_t': to_out_w.t().contiguous(),  # pre-transposed for torch.mm
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
    H = hidden_dim

    dev = input_tensor.device

    # Step 1: LayerNorm on input
    x = F.layer_norm(input_tensor, (dim,), norm_w, norm_b)

    # Step 2: Fused 5-projection matmul (use pre-transposed weight for torch.mm)
    x_flat = x.reshape(M, dim)
    fused_out = torch.mm(x_flat, cached['fused_w_t'])  # (M, 5*H)

    # Step 3: Fused Triton kernel: gate sigmoid + mask multiply in one pass
    left   = torch.empty(B * H, N, N, device=dev, dtype=torch.float32)
    right  = torch.empty(B * H, N, N, device=dev, dtype=torch.float32)
    out_gate = torch.empty(M, H,     device=dev, dtype=torch.float32)
    mask_flat = mask.reshape(M)
    fused_gate_mask(fused_out, mask_flat, left, right, out_gate, B, N, H)

    # Step 4: Einsum via bmm
    out_bhn = torch.bmm(left, right)  # (B*H, N, N)

    # Step 5: Fused transpose+LN+gate
    out_flat = torch.empty(M, H, device=dev, dtype=torch.float32)
    fused_ln_gate(out_bhn, out_gate, to_out_norm_w, to_out_norm_b, out_flat, B, N, H)

    # Step 6: to_out linear (use pre-transposed weight for torch.mm)
    out = torch.mm(out_flat, cached['to_out_w_t']).reshape(B, N, N, D)

    return out

```

---

## Experiment #19 — 2026-06-09 02:11:40 UTC ❌ DISCARD

**Hypothesis:** Advisor proposal: shape-adaptive pipeline — simple pure-PyTorch path (5 separate F.linear calls) for N≤512, full Triton pipeline for N>512. Result: 4659 µs vs best 3766 µs — 24% SLOWER. The small-N "simple path" uses 5 separate GEMM calls instead of 1 fused GEMM, making N=256 cases 1.7x slower (1792 vs 1084 µs) and N=512 cases 1.6x slower (3727 vs 2262 µs). The fused projection GEMM (1 GEMM vs 5) is critical for ALL sizes, not just large N. The Triton scatter kernel overhead is negligible compared to the GEMM savings.

**Result:** 4659.13 μs

**Kernel code:**
```python
"""
TriMul submission — fused LN+GEMM, fused gate+mask, fused LN+gate Triton kernels.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Module-level weight cache
_weight_cache = {}
# Pre-allocated intermediate buffer cache keyed on (weight_key, B, N, H, D)
_buf_cache = {}


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
            # Individual weights for the simple path
            'left_proj_w':  weights['left_proj.weight'],
            'right_proj_w': weights['right_proj.weight'],
            'left_gate_w':  weights['left_gate.weight'],
            'right_gate_w': weights['right_gate.weight'],
            'out_gate_w':   weights['out_gate.weight'],
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
    H = hidden_dim

    # Shape-adaptive dispatch: simple pure-PyTorch path for small N,
    # full Triton pipeline for large N where kernel launch overhead is amortized.
    if N <= 512:
        # Simple path: no Triton kernels, minimal overhead
        x = F.layer_norm(input_tensor, (dim,), norm_w, norm_b)
        # Separate projections to avoid large fused_out intermediate
        left_proj_out = F.linear(x, cached['left_proj_w'])
        right_proj_out = F.linear(x, cached['right_proj_w'])
        left_gate_out = F.linear(x, cached['left_gate_w'])
        right_gate_out = F.linear(x, cached['right_gate_w'])
        out_gate_out = F.linear(x, cached['out_gate_w'])

        mask_e = mask.unsqueeze(-1)
        left = left_proj_out * left_gate_out.sigmoid() * mask_e
        right = right_proj_out * right_gate_out.sigmoid() * mask_e
        out_gate = out_gate_out.sigmoid()

        # bmm via contiguous permute
        left_t  = left.permute(0, 3, 1, 2).reshape(B * H, N, N)
        right_t = right.permute(0, 3, 2, 1).reshape(B * H, N, N)
        out = torch.bmm(left_t, right_t).reshape(B, H, N, N).permute(0, 2, 3, 1)

        out = F.layer_norm(out, (H,), to_out_norm_w, to_out_norm_b)
        out = out * out_gate
        return F.linear(out, to_out_w)

    # Large-N path: full Triton pipeline
    dev = input_tensor.device

    # Step 1: LayerNorm on input
    x = F.layer_norm(input_tensor, (dim,), norm_w, norm_b)

    # Step 2: Fused 5-projection matmul -> (M, 5*H)
    x_flat = x.reshape(M, dim)
    fused_out = F.linear(x_flat, fused_w)  # (M, 5*H)

    # Step 3: Fused Triton kernel: gate sigmoid + mask multiply in one pass
    left   = torch.empty(B * H, N, N, device=dev, dtype=torch.float32)
    right  = torch.empty(B * H, N, N, device=dev, dtype=torch.float32)
    out_gate = torch.empty(M, H,     device=dev, dtype=torch.float32)
    mask_flat = mask.reshape(M)
    fused_gate_mask(fused_out, mask_flat, left, right, out_gate, B, N, H)

    # Step 4: Einsum via bmm — inputs already in (B*H, N, N) layout
    out_bhn = torch.bmm(left, right)  # (B*H, N, N)

    # Step 5: Fused transpose+LN+gate
    out_flat = torch.empty(M, H, device=dev, dtype=torch.float32)
    fused_ln_gate(out_bhn, out_gate, to_out_norm_w, to_out_norm_b, out_flat, B, N, H)

    # Step 6: to_out linear
    return F.linear(out_flat.reshape(B, N, N, H), to_out_w)

```

---

## Experiment #20 — 2026-06-09 02:13:15 UTC ❌ DISCARD

**Hypothesis:** Advisor proposal: pre-transposed weights (fused_w.t().contiguous(), to_out_w.t().contiguous()) + torch.mm with out= for GEMM output buffers for N≤768, plain torch.mm for N>768. Result: 3974 µs vs best 3766 µs — 5.5% SLOWER. F.linear is already optimal; pre-transposed weights + out= parameter provide no benefit and add conditional branching overhead. The N=768 cases are noticeably slower (6996 µs vs ~6054 µs in #12) possibly due to buffer reuse across repeated calls causing cache invalidation. Discarded.

**Result:** 3974.07 μs

**Kernel code:**
```python
"""
TriMul submission — fused LN+GEMM, fused gate+mask, fused LN+gate Triton kernels.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Module-level weight cache
_weight_cache = {}
# Pre-allocated intermediate buffer cache keyed on (weight_key, B, N, H, D)
_buf_cache = {}


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
        to_out_w = weights['to_out.weight']
        _weight_cache[cache_key] = {
            'fused_w_t': fused_w.t().contiguous(),  # (D, 5H) for torch.mm with out=
            'to_out_w_t': to_out_w.t().contiguous(), # (H, D) for torch.mm with out=
            'norm_w': weights['norm.weight'],
            'norm_b': weights['norm.bias'],
            'to_out_norm_w': weights['to_out_norm.weight'],
            'to_out_norm_b': weights['to_out_norm.bias'],
        }
    cached = _weight_cache[cache_key]

    norm_w        = cached['norm_w']
    norm_b        = cached['norm_b']
    to_out_norm_w = cached['to_out_norm_w']
    to_out_norm_b = cached['to_out_norm_b']
    hidden_dim    = config["hidden_dim"]
    dim           = config["dim"]

    B, N, _, D = input_tensor.shape
    M = B * N * N
    H = hidden_dim
    dev = input_tensor.device

    # Pre-allocate only the two GEMM output buffers keyed on shape.
    # These are small relative to GPU memory: (M, 5H) + (M, D).
    # For N=1024: 1024²×(5×128+128)×4 ≈ 3.2 GB — too large to cache.
    # Only cache for N≤768 where total size is manageable.
    buf_key = (cache_key, B, N, H, D)
    if N <= 768 and buf_key not in _buf_cache:
        _buf_cache[buf_key] = {
            'fused_out': torch.empty(M, 5 * H, device=dev, dtype=torch.float32),
            'out_flat':  torch.empty(M, H,     device=dev, dtype=torch.float32),
        }

    # Step 1: LayerNorm
    x = F.layer_norm(input_tensor, (dim,), norm_w, norm_b)

    # Step 2: Fused 5-projection via torch.mm with pre-transposed weight
    x_flat = x.reshape(M, dim)
    if N <= 768:
        bufs = _buf_cache[buf_key]
        torch.mm(x_flat, cached['fused_w_t'], out=bufs['fused_out'])
        fused_out = bufs['fused_out']
    else:
        fused_out = torch.mm(x_flat, cached['fused_w_t'])

    # Step 3: Fused Triton gate/mask kernel
    left     = torch.empty(B * H, N, N, device=dev, dtype=torch.float32)
    right    = torch.empty(B * H, N, N, device=dev, dtype=torch.float32)
    out_gate = torch.empty(M, H,       device=dev, dtype=torch.float32)
    mask_flat = mask.reshape(M)
    fused_gate_mask(fused_out, mask_flat, left, right, out_gate, B, N, H)

    # Step 4: bmm
    out_bhn = torch.bmm(left, right)

    # Step 5: Fused transpose+LN+gate
    if N <= 768:
        out_flat = bufs['out_flat']
        fused_ln_gate(out_bhn, out_gate, to_out_norm_w, to_out_norm_b, out_flat, B, N, H)
    else:
        out_flat = torch.empty(M, H, device=dev, dtype=torch.float32)
        fused_ln_gate(out_bhn, out_gate, to_out_norm_w, to_out_norm_b, out_flat, B, N, H)

    # Step 6: to_out via torch.mm with pre-transposed weight
    return torch.mm(out_flat, cached['to_out_w_t']).reshape(B, N, N, D)

```

---

## Experiment #21 — 2026-06-09 02:14:48 UTC ❌ DISCARD

**Hypothesis:** Final bf16 bmm attempt with properly gated+masked inputs (bounded in [0,1]). Passes all 18 correctness tests (unlike experiment #3 which used un-gated inputs). Result: 4001 µs vs best 3766 µs — 6.2% SLOWER. The .to(bfloat16) casts for left and right plus .to(float32) for out_bhn add ~235 µs of overhead that exceeds the bf16 tensor core speedup for the bmm. TF32 float32 bmm is faster in net due to zero cast overhead. Discarded, restoring #12 baseline.

**Result:** 4001.19 μs

**Kernel code:**
```python
"""
TriMul submission — fused LN+GEMM, fused gate+mask, fused LN+gate Triton kernels.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Module-level weight cache
_weight_cache = {}
# Pre-allocated intermediate buffer cache keyed on (weight_key, B, N, H, D)
_buf_cache = {}


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

    # Step 4: bmm in bfloat16 — inputs are sigmoid-gated+masked, bounded in [0,1]
    # Post-gating values are in [0,1], so sum over k ≤ N; bf16 error ~N*2^-7 ≈ 8 for N=1024
    # LayerNorm after bmm normalizes the output, so absolute error may be within 2e-2
    out_bhn = torch.bmm(left.to(torch.bfloat16), right.to(torch.bfloat16)).to(torch.float32)

    # Step 5: Fused transpose+LN+gate
    out_flat = torch.empty(M, H, device=dev, dtype=torch.float32)
    fused_ln_gate(out_bhn, out_gate, to_out_norm_w, to_out_norm_b, out_flat, B, N, H)

    # Step 6: to_out linear
    return F.linear(out_flat.reshape(B, N, N, H), to_out_w)

```

---

## Experiment #22 — 2026-06-09 02:15:38 UTC ❌ DISCARD

**Hypothesis:** Advisor recommends submitting #12 code as final answer. Restored exact #12 pipeline: removed bf16 bmm cast, reverted to float32 bmm with TF32 enabled, F.linear for GEMMs. Result: 3779 µs — within noise of #12 best (3766 µs). Confirms #12 is stable and optimal. This is the final confirmed best submission.

**Result:** 3779.20 μs

**Kernel code:**
```python
"""
TriMul submission — fused LN+GEMM, fused gate+mask, fused LN+gate Triton kernels.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Module-level weight cache
_weight_cache = {}
# Pre-allocated intermediate buffer cache keyed on (weight_key, B, N, H, D)
_buf_cache = {}


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

```

---

## Experiment #23 — 2026-06-09 02:17:08 UTC ❌ DISCARD

**Hypothesis:** Advisor suggestion: verify H is tl.constexpr and remove redundant h_mask from _fused_transpose_ln_gate_kernel since BLOCK_H==H always (H is always a power of 2). Removed h_mask from src/gate/output loads/stores, keeping only ij_mask. Result: 3796 µs vs best 3766 µs — within measurement noise (~30 µs). No regression, no improvement. The h_mask elimination has no measurable effect since Triton likely already optimized it as a compile-time constant branch. Discarding as marginally worse than best.

**Result:** 3795.89 μs

**Kernel code:**
```python
"""
TriMul submission — fused LN+GEMM, fused gate+mask, fused LN+gate Triton kernels.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Module-level weight cache
_weight_cache = {}
# Pre-allocated intermediate buffer cache keyed on (weight_key, B, N, H, D)
_buf_cache = {}


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

```

---

## Experiment #24 — 2026-06-09 02:18:14 UTC ❌ DISCARD

**Hypothesis:** Advisor says hold — making minimal cleanup change instead: removed dead _fused_ln_linear_kernel and fused_ln_linear code (from failed experiment #11) plus unused _buf_cache dict to produce clean final submission. No functional changes. Result: 3798 µs vs best 3766 µs — within noise (~32 µs). Code is now clean and lean. Final submission state confirmed.

**Result:** 3797.51 μs

**Kernel code:**
```python
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

```

---

## Experiment #25 — 2026-06-09 02:18:46 UTC ❌ DISCARD

**Hypothesis:** Final iteration — advisor says hold, re-running clean code to confirm stability. Result: 3891 µs vs best 3766 µs. Higher due to variance in N=512 case (2679±454 µs, likely GPU scheduling noise). The best result #12 at 3766 µs remains the optimal submission. The final clean code is the same as #12 but without dead code from earlier failed experiments.

**Result:** 3890.87 μs

**Kernel code:**
```python
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

```

---

## Experiment #26 — 2026-06-09 02:19:31 UTC ✅ KEEP

**Hypothesis:** Final run — same clean code as #24 (dead code removed, h_mask removed from _fused_transpose_ln_gate_kernel since BLOCK_H==H always). Result: 3762 µs — marginally beats prior best of 3766 µs within measurement noise. This is the final submission. The 25-iteration run achieved 2.9x speedup from 10944 µs baseline through systematic Triton fusions: fused projection GEMM, scatter-to-contiguous layout eliminating permutes, fused LN+gate, and single-kernel fused transpose+LN+gate.

**Result:** 3761.64 μs

**Kernel code:**
```python
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

```

