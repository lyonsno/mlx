# Anaphora: NF4 Weight Loader

**Reviewer**: cc-nf4-weight-loader-review-0610
**Probole**: `reviews/nf4-weight-loader_probole.md`
**Files reviewed**:
- `python/mlx/nn/layers/nf4_loader.py`
- `/Users/noahlyons/dev/mlx-ideogram4/load_weights.py`
- `mlx/backend/metal/kernels/nf4_quantized.h` (cross-reference only)

---

## Summary

The nibble repacking, uint32 packing, scale reshaping, bfloat16 two-pass loading, and consumed key tracking are all correct for the standard bnb NF4 single-quantization case. The QuantizedLinear swap handles dotted paths with list indices and bias detection correctly.

Two material findings and one minor cleanliness issue.

---

## Findings

### F1 [material] — Double quantization not handled; generic loader will produce garbage scales

**Files**: `nf4_loader.py` (lines 128-133), `load_weights.py` (lines 63-69)

Both loaders read `{base}.absmax` and use it directly as float32 scales. When bnb uses double quantization (`compress_statistics=True`, which is the **default** for `bnb.nn.Linear4bit`), `absmax` in the safetensors is itself quantized (uint8) and must be dequantized using `nested_absmax`, `nested_quant_map`, and `nested_scale_offset`.

The metadata JSON contains `"nested": true` when double quant is active. Neither loader checks this field.

**Impact**: For `load_weights.py` (Ideogram4-specific), this is likely a non-issue if Ideogram4 was saved with `compress_statistics=False`. For `nf4_loader.py` (generic, lives in mlx proper), any model using default bnb settings would get silently wrong scale values.

**Recommendation**: In `nf4_loader.py`, check `meta.get("nested", False)`. If true, either dequantize the nested scales or raise `NotImplementedError("Double quantization (nested absmax) not yet supported")`. This prevents silent wrong results.

### F2 [material] — group_size hardcoded in QuantizedLinear, not propagated from metadata

**File**: `load_weights.py` line 170

```python
ql = nn.QuantizedLinear(
    in_dims, out_dims, bias=has_bias, group_size=64, bits=4, mode="nf4"
)
```

The `group_size` is hardcoded to 64, but the actual blocksize is read from metadata at line 61 (`blocksize = meta.get("blocksize", 64)`). If a model uses a non-64 blocksize, the scales would be reshaped correctly (the dynamic `blocksize` is used), but `QuantizedLinear` would tell the Metal kernel to use `group_size=64`, causing a shape/indexing mismatch.

**Impact**: Latent bug. Standard bnb NF4 uses blocksize=64 so current models work. Would silently produce wrong results for non-standard blocksizes.

**Recommendation**: Pass the blocksize through:
```python
ql = nn.QuantizedLinear(
    in_dims, out_dims, bias=has_bias, group_size=blocksize, bits=4, mode="nf4"
)
```
This requires threading `blocksize` through the `quantized_layers` dict (currently stores `orig_shape` but not `blocksize`).

### F3 [minor] — Missing `nested_scale_offset` in consumed key set

**Files**: Both `nf4_loader.py` (line 143) and `load_weights.py` (line 81)

The consumed suffix list includes `.nested_absmax` and `.nested_quant_map` but not `.nested_scale_offset`. If a double-quantized model is loaded, this key would leak through to the non-quantized tensor pass. In `load_weights.py` it's harmless (`strict=False` ignores it). In `nf4_loader.py` it would appear as a stray entry in the returned dict.

**Recommendation**: Add `".nested_scale_offset"` to the suffix list.

---

## Verified correct

### Nibble repacking order
Python: `np.stack([lo, hi], axis=-1).ravel()` produces element order `[lo0, hi0, lo1, hi1, ...]`. Packed into uint32 with shifts `[0, 4, 8, 12, 16, 20, 24, 28]`, nibble k occupies bits `4k` to `4k+3`.

Kernel (`nf4_qdot`): casts `uint32_t*` → `uint16_t*`, reads `ws[i] >> (4*j)` for j=0..3. On little-endian Metal, this reads nibble k from bits `4k` to `4k+3` of the uint32. **Match confirmed.**

Kernel (`nf4_dequantize_elem`, block loader path): reads `uint8_t`, lo nibble = `w & 0x0f`, hi = `w >> 4`. Same byte-level nibble order. **Match confirmed.**

### np.sum overflow
Max possible value: `0xF * (1 + 0x10 + 0x100 + ... + 0x10000000) = 0xFFFFFFFF`. Fits in uint32. The `.astype(np.uint32)` cast after `np.sum` (which may internally accumulate as uint64) is present and correct.

### Scale reshaping
`absmax.reshape(rows, cols // blocksize)` in C-order. bnb flattens weights row-major before quantizing, so blocks follow row-major order. The kernel expects `in_vec_size / group_size` contiguous scales per row. **Match confirmed.**

### bfloat16 two-pass
Pass 1 (numpy) catches `TypeError`/`ValueError` for bfloat16 tensors. Pass 2 loads only `remaining` keys (computed as `(keys - consumed) - already_loaded`). The `mx.load()` call loads everything but only `remaining` keys are extracted. Quantized weights are not overwritten. **Correct.**

### QuantizedLinear swap
Path navigation handles digit indices (`layers.0.` etc.) via `p.isdigit()`. Bias detection checks `isinstance(old, nn.Linear) and old.bias is not None`. Leaf assignment uses `setattr` or `parent[int(leaf)]` as appropriate. **Correct.**

### Consumed key tracking (standard case)
For single quantization, the 6 consumed keys per base (`base`, `.absmax`, `.quant_map`, `.nested_absmax`, `.nested_quant_map`, `.quant_state.bitsandbytes__nf4`) cover all bnb sibling keys. The `.nested_*` keys won't exist for single quant models, so their inclusion is harmless. **Correct for single quant.**
