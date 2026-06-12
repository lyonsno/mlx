"""Load bitsandbytes NF4 quantized weights from safetensors into MLX.

Converts bitsandbytes 4-bit NormalFloat (NF4) packed uint8 weights
into MLX's uint32 NF4 format for use with mx.quantized_matmul.

Usage:
    from mlx.nn.layers.nf4_loader import load_bnb_nf4_safetensors

    weights = load_bnb_nf4_safetensors("model.safetensors")
    # Returns dict of:
    #   "layer.weight"        -> mx.array uint32 (repacked for MLX)
    #   "layer.weight_scales" -> mx.array float32 (absmax per block)
    #   "layer.norm.weight"   -> mx.array float32/float16 (non-quantized)
"""

import json
import numpy as np

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None


def _repack_bnb_nf4_to_mlx(packed_uint8: np.ndarray, orig_shape: list[int]) -> np.ndarray:
    """Convert bitsandbytes NF4 packed uint8 to MLX uint32 format.

    bnb stores 2 nibbles per uint8 byte (high nibble = first element,
    low nibble = second element).
    MLX stores 8 nibbles per uint32 (lo nibble at bits 0-3, etc).

    Args:
        packed_uint8: Flat uint8 array from bitsandbytes (N/2 bytes for N elements)
        orig_shape: Original weight shape [out_features, in_features]

    Returns:
        uint32 array of shape (out_features, in_features // 8)
    """
    flat = packed_uint8.ravel()

    # Unpack: each byte contains 2 NF4 indices
    # bnb convention: high nibble = first element, low nibble = second element
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    indices = np.stack([hi, lo], axis=-1).ravel().astype(np.uint32)

    rows, cols = orig_shape
    assert len(indices) == rows * cols, (
        f"Index count {len(indices)} != {rows}*{cols}={rows * cols}"
    )

    # Repack into uint32: 8 nibbles per uint32
    indices_grouped = indices.reshape(-1, 8)
    shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    packed_u32 = np.sum(indices_grouped << shifts, axis=1).astype(np.uint32)

    return packed_u32.reshape(rows, cols // 8)


def _parse_bnb_metadata(meta_tensor: np.ndarray) -> dict:
    """Parse the bitsandbytes quant_state metadata blob.

    This is a JSON string stored as uint8 bytes in the safetensors file.
    Contains: quant_type, blocksize, dtype, shape.
    """
    return json.loads(meta_tensor.tobytes().decode("utf-8"))


def load_bnb_nf4_safetensors(
    path: str,
    *,
    verbose: bool = True,
) -> dict[str, "mx.array"]:
    """Load a bitsandbytes NF4 safetensors file and convert to MLX format.

    Identifies quantized weights by the presence of `.quant_state.bitsandbytes__nf4`
    keys. Non-quantized tensors (norms, embeddings, etc.) are loaded directly.

    Args:
        path: Path to .safetensors file
        verbose: Print progress

    Returns:
        Dict mapping weight names to mx.arrays. Quantized weights produce two
        entries: "name.weight" (uint32) and "name.weight_scales" (float32).
    """
    if safe_open is None:
        raise ImportError("safetensors is required: pip install safetensors")
    if mx is None:
        raise ImportError("mlx is required")

    result = {}
    n_quantized = 0
    n_regular = 0

    # Use safetensors metadata to get dtype info, then load raw for bfloat16
    from safetensors import safe_open as _safe_open

    with _safe_open(path, framework="numpy") as sf:
        keys = set(sf.keys())

        # Find all quantized weight base names
        quantized_bases = set()
        for k in keys:
            if ".quant_state.bitsandbytes__nf4" in k:
                base = k.split(".quant_state.bitsandbytes__nf4")[0]
                quantized_bases.add(base)

        # Process quantized weights
        for base in sorted(quantized_bases):
            meta_key = f"{base}.quant_state.bitsandbytes__nf4"
            absmax_key = f"{base}.absmax"

            if meta_key not in keys or absmax_key not in keys:
                if verbose:
                    print(f"  SKIP {base}: missing metadata or absmax")
                continue

            meta = _parse_bnb_metadata(sf.get_tensor(meta_key))
            orig_shape = meta["shape"]
            blocksize = meta.get("blocksize", 64)

            if meta.get("nested", False):
                raise NotImplementedError(
                    f"Double quantization (nested absmax) not supported for {base}. "
                    "Re-quantize with compress_statistics=False."
                )

            packed = sf.get_tensor(base)
            absmax = sf.get_tensor(absmax_key)

            rows, cols = orig_shape
            mlx_packed = _repack_bnb_nf4_to_mlx(packed, orig_shape)
            n_groups_per_row = cols // blocksize
            absmax_reshaped = absmax.reshape(rows, n_groups_per_row)

            result[base] = mx.array(mlx_packed)
            result[f"{base}_scales"] = mx.array(absmax_reshaped)
            n_quantized += 1

        # Collect keys consumed by quantized weights
        consumed = set()
        for base in quantized_bases:
            consumed.add(base)
            for suffix in [".absmax", ".quant_map",
                           ".nested_absmax", ".nested_quant_map",
                           ".nested_scale_offset",
                           ".quant_state.bitsandbytes__nf4"]:
                consumed.add(base + suffix)

        # Process remaining (non-quantized) tensors
        for k in sorted(keys - consumed):
            try:
                t = sf.get_tensor(k)
                result[k] = mx.array(t)
                n_regular += 1
            except (TypeError, ValueError):
                # bfloat16 not supported by numpy — load raw bytes via MLX
                pass

    # Second pass: load bfloat16 tensors that numpy couldn't handle
    # MLX's safetensors loader handles bfloat16 natively
    remaining = sorted((keys - consumed) - set(result.keys()))
    if remaining:
        try:
            bf16_weights = mx.load(path)
            for k in remaining:
                if k in bf16_weights:
                    result[k] = bf16_weights[k]
                    n_regular += 1
                elif verbose:
                    print(f"  SKIP {k}: not found in mx.load pass")
        except Exception as e:
            if verbose:
                print(f"  bf16 fallback failed: {e}")
                for k in remaining:
                    print(f"  SKIP {k}: bfloat16 not loadable")

    if verbose:
        print(f"  Loaded {n_quantized} quantized + {n_regular} regular tensors "
              f"from {path}")

    return result


def get_quantized_linear_params(
    weights: dict[str, "mx.array"],
    prefix: str,
) -> tuple["mx.array", "mx.array"] | None:
    """Extract quantized weight and scales for a linear layer.

    Args:
        weights: Dict from load_bnb_nf4_safetensors
        prefix: Layer prefix (e.g. "transformer_blocks.0.attn.to_q")

    Returns:
        (weight_uint32, scales_float32) or None if not quantized
    """
    w_key = f"{prefix}.weight"
    s_key = f"{prefix}.weight_scales"
    if w_key in weights and s_key in weights:
        return weights[w_key], weights[s_key]
    return None
