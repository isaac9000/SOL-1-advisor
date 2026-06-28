# Advisor Proposals

---

## Iteration 1

## STATE
Only the pure-PyTorch baseline exists at 3427.91 μs geomean. No custom kernels have been attempted yet. The SOL target is ~756/9.3 ≈ 81 μs geomean, so we are roughly 42× above SOL. The baseline has four heavy operations: two batch-matmuls (dP̃ and dV), dropout/softmax backward elementwise ops, and GQA reshape+sum. All of these are implemented naively in float32 with unnecessary memory traffic (bfloat16→float32 casts inline).

## RATIONALE
The biggest wins here come from using **FlashAttention-style fused kernels or cuBLAS-backed batched GEMMs with proper dtype handling**. The reference does everything in float32, broadcasts value_states (creating a [bs,80,skv,128] tensor), and uses `.sum(dim=2)` for GQA aggregation — all memory-bandwidth expensive. The fastest path leverages cuBLAS's native bfloat16 batched matmul (which runs at near-peak throughput on B200) and fuses the softmax backward elementwise work. A Triton kernel can fuse the dropout mask application, softmax backward computation, and GQA reduction to minimize DRAM round-trips. The two BMMs are the compute bottleneck; everything else is bandwidth-bound elementwise work that should be fused.

## PROPOSAL

Implement a **Triton-based fused kernel** approach structured as follows:

**Strategy:** Replace the reference's two separate float32 BMMs + elementwise passes with a tightly fused implementation:

1. **For `grad_attn_scores` (dS):** Write a Triton kernel that, for each (bs, head, sq_tile) block, loads a tile of `attn_weights` (P) and computes `dP̃ = dO_tile @ V_tile^T` inline (with V fetched using the GQA head mapping `kv_head = head // 10`), then applies dropout mask scaling and softmax backward — all in a single kernel without materializing intermediate float32 tensors to DRAM. Keep accumulation in float32 for precision but write outputs in bfloat16.

2. **For `grad_value_states` (dV):** Write a second Triton kernel that computes `dV[kv_head] = sum_{g=0}^{9} attn_weights_dropped[kv_head*10+g]^T @ dO[kv_head*10+g]` using bfloat16 inputs and float32 accumulation, producing the GQA-reduced result directly (never materializing the expanded [bs,80,skv,d] tensor).

The worker should choose block/tile sizes appropriate for B200's SM count and L2 cache, prioritizing memory-access coalescing. Both kernels should use `tl.dot` for the inner matmul tiles and store results in bfloat16. Start with the two-kernel approach before attempting a single mega-fused kernel, since the access patterns differ between dS (sq-parallel) and dV (skv-parallel).

---

## Iteration 2

## STATE
Two experiments completed. The Triton two-kernel approach (Experiment #2) improved the baseline from 3427 μs to 2208 μs (~1.55× speedup), but we are still ~27× above SOL (~81 μs geomean). The Triton kernel is still maturing — there are significant architectural inefficiencies: the dS kernel does **two full passes** over V/mask/P (double memory traffic), and both kernels have loop-over-groups/Q-tiles inside Triton programs that limit parallelism. The approach is valid but needs substantial restructuring to close the gap.

## RATIONALE
The dominant bottleneck is the dS kernel's two-pass design: it loads V tiles, mask, and P twice to first accumulate `dot_sum` then write `dS`. This doubles DRAM traffic for the most expensive tensors. Additionally, the Triton kernels loop over all KV tiles *inside* the kernel (limiting SM parallelism) rather than tiling across the grid. The right fix is to switch the dS kernel from a two-pass approach to a **single-pass using online softmax / online normalization**, where `dot_sum` is accumulated in registers alongside `dP*P` and dS is written in a single pass — but this requires storing the whole dP row, which may overflow registers for large skv. The better alternative for large skv is to restructure the grid so each program handles a single (batch, head, q_row, kv_tile) and use an **atomic reduction** for the dot_sum, but that adds synchronization overhead. The cleanest approach is to **use PyTorch's cuBLAS-backed BMM for the two expensive matmuls** (which are near-optimal on B200) and use a **single fused Triton kernel only for the elementwise softmax backward + dropout**, which avoids the double-pass problem entirely.

## PROPOSAL

**Restructure to a hybrid approach:** use cuBLAS (PyTorch `torch.bmm`) for the two matmuls and a single fused Triton kernel for all elementwise work. This separates the compute-bound work (BMM) from the memory-bandwidth-bound work (elementwise), letting each run at its peak.

Specifically:

1. **Matmul for dP̃:** Reshape and broadcast `value_states` efficiently using `.expand()` + `.reshape()` (no copy), then call `torch.bmm` in bfloat16. The B200's tensor cores handle bfloat16 BMM at far higher throughput than float32.

2. **Matmul for dV:** Similarly, compute `attn_weights_dropped^T @ dO` with `torch.bmm` in bfloat16, then do the GQA group reduction via a fused Triton kernel or `torch.sum` on the reshaped tensor.

3. **Fused elementwise Triton kernel:** Write a single kernel that reads `dP̃`, `dropout_mask`, `attn_weights` (P), and writes `dS` in one pass. This kernel should: apply dropout scaling, compute `dot_sum = sum_kv(dP * P)` in a single horizontal pass, then compute `dS = P * (dP - dot_sum)` — all from registers with a single read/write of each tensor. No double-pass needed.

4. **Keep bfloat16 throughout** the matmuls — avoid casting to float32 for the BMMs since B200 has native bf16 tensor core support and float32 costs 2× the memory bandwidth.

The worker should prioritize getting the hybrid working correctly first, then tune the Triton elementwise kernel's tile sizes and warp count for memory bandwidth efficiency.

---

## Iteration 3

## STATE
Three experiments completed. The hybrid approach (Experiment #3) — cuBLAS BMM for matmuls + fused Triton elementwise for softmax backward — achieved 628 μs, a 5.5× improvement over the baseline. This approach is still maturing: the dominant costs are now (1) the `.contiguous()` call creating a full [bs,80,skv,128] copy for value_states GQA expansion, (2) the two BMMs running sequentially, and (3) the `dO.transpose(1,2).contiguous()` copy. The SOL geomean is ~81 μs, so we're still ~7.7× above SOL. The hybrid direction is correct; the remaining gap is primarily memory allocation overhead and unnecessary copies.

## RATIONALE
The biggest remaining inefficiencies in Experiment #3 are:
1. **GQA value expansion copy**: `vs_exp = value_states[...].expand(...).reshape(...).contiguous()` allocates a full [bs,80,skv,128] bf16 tensor — for large cases this alone is hundreds of MB. This should be eliminated by computing the dP matmul with a strided BMM or by using a custom Triton kernel that maps the GQA head index to the correct kv_head.
2. **`dO.transpose().contiguous()` copy**: Another large allocation. This can be avoided by working with the original [bs,sq,80,128] layout and adjusting matmul operand order, or by passing strides to BMM.
3. **Sequential BMMs**: The two BMMs (dP and dV) are independent and could potentially overlap, though this is harder to control in PyTorch.
4. The softmax backward kernel is already well-structured; focus should be on eliminating the copies.

## PROPOSAL

**Eliminate the two expensive `.contiguous()` memory copies** that are currently dominating latency for large batch/sequence cases:

1. **For dP̃ = dO @ V^T without GQA expansion copy:** Instead of expanding value_states to [bs,80,skv,128], restructure the BMM to exploit the GQA structure directly. Since each group of 10 heads shares the same KV head, perform the matmul as a [bs*8, 10, sq, 128] × [bs*8, 1, 128, skv] operation using broadcasting, OR use `torch.einsum` or a strided view that avoids materialization. Alternatively, reshape dO to [bs*8, 10, sq, 128] and value_states to [bs*8, skv, 128], then do a batched matmul where dO is treated as 8 groups of 10 heads each — this avoids the expand+copy.

2. **For dO layout:** Avoid the `.transpose().contiguous()` copy by keeping dO in [bs,sq,80,128] layout and restructuring the matmuls to work with the transposed strides. PyTorch's `torch.bmm` can handle non-contiguous inputs in many cases. Alternatively, fuse the transpose into the first BMM by swapping operands.

3. **Keep the Triton softmax-backward kernel** as-is since it's already efficient.

4. **For dV BMM:** The `attn_weights_dropped.reshape(bs*80, sq, skv)` is already contiguous (or nearly so), so no copy is needed there. The GQA reduction `.sum(dim=2)` after reshaping should be kept as-is (PyTorch handles this efficiently).

The worker should focus on restructuring the matmul operands to avoid materialization of the expanded value tensor and the transposed dO tensor, keeping all operations on non-contiguous views where possible.

---

## Iteration 4

## STATE
Four experiments completed. The latest approach (Exp #4, 518 μs) made meaningful progress over Exp #3 (628 μs) by restructuring the GQA matmuls to avoid the expensive `[bs,80,skv,128]` value expansion, instead grouping dO as `[bs*8, 10*sq, 128]` and doing a single BMM against `value_states[bs*8, skv, 128]`. The approach is still maturing — we're at ~6.4× above SOL (~81 μs). The remaining bottleneck is likely: (1) the `dO_grouped.permute().contiguous()` copy (same size as dO but still a full allocation), (2) the dV BMM still requires a `Pd_gq` reshape that may involve a contiguous check, and (3) the two BMMs plus the elementwise kernel are still serialized.

## RATIONALE
The current code still has one significant allocation: `dO_grouped.permute(0,2,3,1,4).contiguous()` which copies the full dO tensor (from `[bs,sq,8,10,128]` to `[bs,8,10,sq,128]`). This is unavoidable for the current grouping strategy. The real question is whether we can avoid this by a different decomposition — specifically, by computing dP directly using `torch.matmul` with broadcasting rather than reshaping to a BMM-compatible layout. The `attn_weights_dropped.reshape([bs,8,10,sq,skv]).reshape([bs*8,10*sq,skv])` chain is a zero-copy reshape since the original is contiguous. The actual performance bottleneck is now predominantly the BMMs themselves plus the intermediate tensor sizes. A key optimization is to **overlap** the two BMMs (dP and dV) since they are independent — this can be achieved by issuing them on different CUDA streams, or more practically by fusing them into a single well-structured Triton kernel.

## PROPOSAL

**Pursue two parallel directions: (A) stream-overlap the two independent BMMs, and (B) investigate whether cuDNN's flash attention backward pass (if available) can be called directly.**

Primary direction — **CUDA streams for parallelism:**

The two BMMs (computing dP̃ and dV) are fully independent computations. Issue them on two separate CUDA streams so they can overlap on the B200's multiple SM partitions. PyTorch supports this via `torch.cuda.stream()` context managers and `torch.cuda.Stream()`. The softmax backward elementwise kernel depends on dP̃ (so it must wait for that stream), but dV computation can proceed in parallel with the softmax backward.

Specifically:
1. Create two CUDA streams at module initialization (cache them to avoid overhead).
2. Stream 1: compute `dP̃` BMM → then launch softmax-backward Triton kernel → produce `dS`.
3. Stream 2: compute `dV` BMM simultaneously with stream 1's work.
4. Synchronize both streams before returning.

The `dO_grouped.contiguous()` copy is shared between both BMMs and must happen before both streams fork — keep it on the default stream before splitting.

Secondary direction to also try: **Replace the softmax-backward kernel with a purely elementwise PyTorch operation** (no intermediate materialization) using `torch.compile` or `torch._inductor` for the dropout+softmax-bwd fused op — this avoids the Triton kernel launch overhead for small cases (cases 1, 5, 7, 9, 14, 15) where the overhead may dominate.

The worker should implement the two-stream approach as the primary change, keeping the rest of Exp #4's structure intact.

---

## Iteration 5

## STATE
Five experiments completed. The CUDA streams approach (Exp #5, 551 μs) was discarded — it was slower than Exp #4 (518 μs), likely because the two BMMs are already each large enough to saturate the GPU individually, so stream overlap adds synchronization overhead without meaningful parallelism benefit on B200's single-SM pool. The best is 518 μs, still ~6.4× above SOL. The hybrid cuBLAS-BMM + Triton-elementwise approach has stabilized around 500–520 μs, with the primary remaining bottleneck being the `dO_grouped.permute().contiguous()` copy (~same size as dO) and the two sequential BMMs.

## RATIONALE
The stream approach failed because the B200 is a single device where the two BMMs compete for the same tensor cores rather than overlapping. The real question now is: what fraction of the 518 μs is each component? Looking at the computation structure: `attn_weights_dropped.reshape(bs, n_kv_heads, n_groups, seq_q, seq_kv).reshape(bs * n_kv_heads, n_groups * seq_q, seq_kv)` is a *zero-copy* view chain since `attn_weights_dropped` is contiguous — but `attn_weights_dropped` is `[bs, 80, sq, skv]` which is `[bs, n_kv_heads*n_groups, sq, skv]`, so the reshape to `[bs, n_kv_heads, n_groups, sq, skv]` is valid as a view. The **only** real copy is `dO_grouped.permute().contiguous()`. The biggest algorithmic opportunity remaining is to **avoid materializing the intermediate `dP_raw` tensor** altogether by fusing the softmax backward directly into a Triton kernel that also does the dO@V^T matmul — eliminating one full [bs,80,sq,skv] tensor write+read cycle. This is the approach used by Flash Attention.

## PROPOSAL

**Fuse the dO@V^T matmul with the softmax backward into a single Triton kernel** to eliminate the intermediate `dP_raw` tensor materialization.

The key insight: the current pipeline writes `dP_raw` to DRAM after the BMM, then reads it back in the softmax-backward kernel. For a `[bs,80,sq,skv]` tensor at bf16, this is 2×bs×80×sq×skv bytes of unnecessary traffic. For case 6 (bs=32, sq=691, skv=773), this is ~2.8 GB of extra traffic.

**Proposed fused kernel for dS:** Write a Triton kernel where each program handles one `(batch, head, q_tile)` and:
1. Loads a tile of `dO` (q_tile × head_dim) into registers
2. Iterates over KV tiles, loading `V[kv_head]` tiles and computing `dP_tile = dO_tile @ V_tile^T` via `tl.dot`
3. Simultaneously loads `P_tile` and `mask_tile`, applies dropout correction, accumulates `dot_sum`
4. Does a second pass over KV to compute and store `dS = P * (dP - dot_sum)`

This is essentially the Experiment #2 Triton kernel structure, but now we know from Exp #3 that the BMM step benefits from cuBLAS. However, for the dS computation specifically, the fused Triton kernel may be faster because it avoids the intermediate materialization — especially for large cases. The BMM-based approach wins for the dV computation (where there's no intermediate to avoid), so keep `torch.bmm` for dV.

For the dO layout issue: rather than copying dO to `[bs,8,10,sq,128]`, keep dO in the original `[bs,sq,80,128]` layout and access it in the Triton kernel using the correct stride (`b * sq * 80 * 128 + q * 80 * 128 + h * 128 + d`), which reads each dO row contiguously for a given head. This avoids the permute copy entirely.

The worker should implement this as: (A) a new Triton kernel that fuses dO@V^T + softmax-bwd for `dS`, and (B) keep `torch.bmm` for `dV` using the Exp #4 grouping. Compare the result against 518 μs.

