# Anaphora: NF4 Initial Kernel Implementation

**Reviewer:** cc-nf4-mlx-review-0610
**Probolē:** `reviews/nf4-initial-kernel-implementation_probole.md`
**Date:** 2026-06-10
**Scope:** All NF4-related changes in the mlx working tree (uncommitted).

---

## Material Findings

These are bugs or gaps that would cause wrong results, crashes, or silent data corruption at runtime.

### M1. Missing gather kernel templates — runtime crash

**Severity:** Crash on any `gather_qmm(..., mode='nf4')` call.

**Location:** `nf4_quantized.h` (missing), `nf4_quantized.metal` (missing), `quantized.cpp:886-976` (dispatch), `ops.cpp:5362-5452` (no guard).

**Description:** The `gather_qmm` ops path allows NF4 mode through to `GatherQMM::eval_gpu`, which dispatches to `gather_qmv`, `gather_qvm`, or `gather_qmm`. These build kernel names like `nf4_gather_qmm_t_...`, `nf4_gather_qmv_fast_...`, etc. and look for them in the `nf4_quantized` source. But `nf4_quantized.h` has no `nf4_gather_qmm_t`, `nf4_gather_qmm_n`, `nf4_gather_qmv`, `nf4_gather_qmv_fast`, or `nf4_gather_qvm` kernel templates. `nf4_quantized.metal` has no instantiations for them.

The NAX paths (`gather_qmm_nax`, `gather_qmm_rhs_nax`) are correctly excluded with `mode != "nf4"` guards at `quantized.cpp:903,1249`, but the non-NAX gather paths have no such guard and will attempt to compile nonexistent templates.

**Fix options:**
1. Add an explicit `throw` in `gather_qmm` ops if mode is NF4 (simplest, until gather kernels are implemented).
2. Implement the NF4 gather kernel variants (more work, may not be needed yet).

---

### M2. `nf4_gather_qmm_rhs` template referenced but doesn't exist

**Severity:** Crash on `gather_qmm_rhs` with NF4 mode when NAX is unavailable.

**Location:** `jit_kernels.cpp:886-913` (`get_gather_qmm_kernel`).

The function builds a template definition referencing `"nf4_gather_qmm_rhs"` and includes `metal::nf4_quantized()`, but no such template exists in `nf4_quantized.h`. Same for the `get_gather_qmm_nax_kernel` path — it doesn't handle NF4 at all (line 1098-1113 only handles "affine" and "fp").

---

### M3. `nf4_quantize` GPU kernel has broken cross-simdgroup absmax for `group_size=64`

**Severity:** Wrong quantization results if the GPU `nf4_quantize` kernel is ever invoked with `group_size=64`.

**Location:** `nf4_quantized.h:1218-1228`.

The cross-simdgroup reduction uses `tidx.x < 32` and `tidx.x >= 32` to split the two halves. But `tidx.x` is the **global** thread position, not the position within a simdgroup or threadgroup. For groups that don't start at global index 0 (i.e., all groups after the first), the predicate `tidx.x < 32` is wrong. For example, group starting at index 64: all threads have `tidx.x >= 32`, so `absmax_l` = 0 for all of them, and the absmax is only computed from the second half.

Additionally, `simd_max` operates within a single 32-thread simdgroup. For `group_size=64`, one group spans two simdgroups, and `simd_max` cannot cross simdgroup boundaries. The code tries to work around this with the `absmax_l`/`absmax_r` split but uses the wrong predicate.

**Mitigation:** This kernel is currently **bypassed** — `ops.cpp:5006-5008` always uses the graph-based fallback for NF4 quantization. The bypass is effective. However, the kernel is compiled and instantiated in `nf4_quantized.metal`, wasting compile time and potentially confusing future developers.

**Recommendation:** Either fix the kernel (use `tidx.x % group_size < 32` and threadgroup memory for the cross-simdgroup reduction) or remove it from instantiation and add a comment explaining the bypass.

---

### M4. `nf4_quantize` GPU kernel packs nibbles differently than the graph fallback

**Severity:** Data corruption if GPU quantize and graph fallback produce weights consumed by the same dequantize path.

**Location:** `nf4_quantized.h:1240-1245` vs `ops.cpp:4996-5001`.

The GPU kernel packs via `simd_shuffle_down`: pairs adjacent threads, packs low nibble in bits [0:3] and high nibble in bits [4:7], writing one byte per two elements. This produces a byte-level packing.

The graph fallback packs 8 nibbles into a uint32 using `power(2, arange(0, 32, 4))` shifts — the first nibble goes in bits [0:3], second in [4:7], ..., eighth in [28:31].

Both orderings agree on nibble order within a byte (low index = low bits), so the byte-level format is consistent. However, the GPU kernel writes `uint8_t` output while the graph fallback writes `uint32` output via shifted sums. Since the dequantize kernels read from `uint8_t*` / `uint16_t*` cast from `uint32_t*`, and both pack low-index-first within each byte, the formats should be compatible **if endianness is little-endian** (which Metal guarantees). This is likely fine but deserves a verification test.

**Mitigation:** GPU quantize is bypassed, so this is not reachable. Marking as material because if the bypass is ever removed, this compatibility must be verified.

---

## Structural Concerns (High Priority Advisory)

### S1. NF4 `Quantize::eval_gpu` dispatch doesn't adjust thread count for NF4

**Location:** `quantized.cpp:1744-1769`.

The dispatch thread count calculation uses:
```cpp
int packs_per_int = (bits_ == 3 || bits_ == 5) ? 8 : bits_ == 6 ? 4 : 8 / bits_;
int per_thread = dequantize_ ? packs_per_int : std::max(group_size_ / simd_size, 1);
```

For NF4 dequantize: `packs_per_int = 8/4 = 2`, and `nthreads = out.size() / 2`. The `nf4_dequantize` kernel processes 2 values per thread (one byte → two nibbles), so this is correct.

For NF4 quantize (bypassed): `per_thread = max(64/32, 1) = 2`, `nthreads = w.size() / 2`. The `nf4_quantize` kernel processes one element per thread and expects one thread per element. With `nthreads = w.size() / 2`, only half the elements would be processed. This is another reason the GPU quantize path must remain bypassed.

### S2. NF4 `qmm_nax` guard is necessary but asymmetric with `gather_qmm_nax`

The `qmm` function at `quantized.cpp:711` guards NAX with `mode != "nf4"`, and `gather_qmm_rhs` at line 1249 does the same. But the `get_qmm_nax_kernel` function at `jit_kernels.cpp:1055-1073` only knows about "affine" and default (fp) modes — it doesn't handle NF4 and would produce wrong source includes if reached. The runtime guards are sufficient, but the asymmetry is fragile. A single missing guard would cause a hard-to-debug JIT compilation failure.

---

## Advisory Findings

### A1. LUT values in `nf4.h` match the bitsandbytes reference

The 16 NF4 quantization levels in `nf4.h:14-31` are correct and match the QLoRA paper / bitsandbytes implementation. The asymmetry (8 negative, 1 zero, 7 positive) is expected.

### A2. Scale pointer arithmetic in `NF4BlockLoader` is correct

Compared with `QuantizedBlockLoader` in `fp_quantized.h`:

- **FP mode:** Scales are `uint8_t` (1 byte per FP8 scale). Scale init: `scales_ + bi * src_ld / group_size + (bj * pack_factor) / group_size`. No multiplier.
- **NF4 mode:** Scales are `float32` (4 bytes per scale), accessed via `uint8_t*` pointer. Scale init: same index expression but multiplied by `scale_bytes (=4)`. Scale `next()` advances by `scale_bytes` or `scale_step * scale_bytes` or `group_stride * scale_bytes`.

This is correct. The `nf4_dequantize_scale` function (`nf4_quantized.h:44-46`) does `reinterpret_cast<const device float*>(s)` to read 4 bytes from the `uint8_t*` pointer, which is valid as long as the pointer is 4-byte aligned. Since scales come from a float32 array, alignment is guaranteed by the allocator.

### A3. Nibble unpacking in `nf4_qdot` is correct

`nf4_quantized.h:89-96`: Reads `uint16_t` from weight buffer, extracts 4 nibbles via shifts of 0, 4, 8, 12. Each nibble is masked to 4 bits by `nf4_dequantize_value` (which applies `& 0x0f`). This correctly unpacks 4 x 4-bit values from a 16-bit word.

### A4. Template instantiation coverage is adequate

`nf4_quantized.metal` instantiates for:
- Types: float, bfloat16_t, float16_t
- Group sizes: 32, 64, 128
- Kernel variants: qmv_fast, qmv, qvm, qmm_n (batched/unbatched), qmm_t (aligned/unaligned × batched/unbatched), qmv_quad (D=64,128), qvm_split_k (split_k=8,32), qmm_t_splitk, quantize, dequantize

This covers the standard dispatch paths. D=64 and D=128 for `qmv_quad` matches the `dispatch_qmv` guard at `quantized.cpp:1398`.

### A5. NF4 validation in `ops.cpp` is correct

`validate_mode_with_type` at `ops.cpp:4437-4454`:
- Requires floating-point scales (correct — NF4 uses float32 absmax)
- Rejects biases (correct — NF4 has no bias term)
- Returns `scales.dtype()` as output type when no explicit out_type (correct — preserves float32 fidelity)

### A6. NF4 `nf4_dequantize` kernel buffer index 2 is skipped

`nf4_quantized.h:1252`: `device T* out [[buffer(3)]]` — buffer 2 is skipped. This matches the dispatch in `Quantize::eval_gpu` where buffer 2 would be biases (not present for NF4). The Metal runtime allows non-contiguous buffer indices. Not a bug, but unusual.

### A7. Build system integration is correct

Both `CMakeLists.txt` files correctly reference `nf4_quantized` and its dependencies (`nf4.h`, `quantized_utils.h`). The JIT source generation (`make_jit_source`) and metallib compilation (`build_kernel`) entries are consistent. `includes.h` declares `nf4_quantized()` correctly.

### A8. `nf4_qouter` processes only 2 values per byte (correct for 4-bit)

`nf4_quantized.h:120-125`: The outer product loop iterates `values_per_thread / 2` times, extracting low and high nibbles from each byte. This is correct for 4-bit packing (2 values per byte).

### A9. Graph-based quantize fallback uses float precision throughout

`ops.cpp:4977-5004`: The fallback computes `argmin(abs(expand_dims(wn, -1) - lut), -1)` which is O(n × 16) — fine for a one-time operation. The LUT is cast to `w.dtype()` for the distance computation. This could introduce rounding differences if `w` is float16/bfloat16 vs the float32 LUT in the Metal kernel, but since quantize is a one-time operation and the LUT values have limited precision anyway, this is acceptable.

---

## Summary

| Category | Count | Action needed |
|----------|-------|--------------|
| Material | 4 | M1 and M2 need a fix (guard or implement gather kernels). M3 and M4 are mitigated by the graph fallback bypass but should be cleaned up. |
| Structural | 2 | S1 confirms the GPU quantize bypass is necessary. S2 notes fragility in NAX exclusion. |
| Advisory | 9 | No action required; scale arithmetic, LUT values, nibble unpacking, validation, and build system are all correct. |

**Verdict:** The core NF4 matmul kernels (qmv, qvm, qmm) and dequantize are correct. The main gap is the missing gather variants (M1/M2), which will crash at runtime if `gather_qmm` is called with NF4 mode. The GPU quantize kernel (M3/M4) is safely bypassed. Recommend adding an explicit throw for `gather_qmm` + NF4 before exposing this to users.
