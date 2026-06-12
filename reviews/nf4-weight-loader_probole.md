# Probolē: NF4 Weight Loader

## Target

- `python/mlx/nn/layers/nf4_loader.py` (generic bitsandbytes NF4 → MLX loader)
- `/Users/noahlyons/dev/mlx-ideogram4/load_weights.py` (Ideogram4-specific weight loading)

## Review scope

Correctness of the bitsandbytes NF4 safetensors → MLX conversion pipeline.
This code repacks uint8-packed nibbles into uint32, reshapes absmax scales,
handles bfloat16 tensors, and swaps nn.Linear → nn.QuantizedLinear.

## Review context mode

Target files only. Reference: bitsandbytes serialization format as documented
at https://github.com/bitsandbytes-foundation/bitsandbytes and in the
ideogram4 repo's `quantized_loading.py`.

## What to look for

1. **Nibble repacking correctness**: bnb stores 2 nibbles per uint8 (lo first).
   MLX stores 8 nibbles per uint32 (lo at bits 0-3). The interleaving via
   `np.stack([lo, hi], axis=-1).ravel()` must produce the correct element order.
   Verify this matches the dequant kernel's unpacking in `nf4_quantized.h`.

2. **np.sum overflow**: `np.sum(uint32 << shifts)` produces uint64 on numpy.
   The `.astype(np.uint32)` cast must be present and correct.

3. **Scale reshaping**: absmax is stored flat `(N/blocksize,)`. Reshaped to
   `(rows, cols/blocksize)`. Verify row-major ordering matches what the Metal
   kernel expects.

4. **bfloat16 handling**: safetensors with `framework="numpy"` can't load
   bfloat16. The two-pass approach (numpy first, then `mx.load` fallback)
   must not overwrite quantized weights or double-load tensors.

5. **QuantizedLinear swap**: `_swap_to_quantized_linear` navigates the model
   tree by dotted path. Verify it handles list indices (e.g. `layers.0.`),
   bias detection, and group_size/bits propagation correctly.

6. **Consumed key tracking**: The `consumed` set must include ALL bitsandbytes
   sibling keys (`.absmax`, `.quant_map`, `.nested_absmax`, `.nested_quant_map`,
   `.quant_state.bitsandbytes__nf4`). Missing keys would be loaded as regular
   tensors and cause shape mismatches.

## Out of scope

- Metal kernel correctness (separate review)
- Transformer architecture parity (separate review)
- Performance (one-time conversion cost)
