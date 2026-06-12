# Probolē: NF4 Initial Kernel Implementation

## Target

All NF4-related changes in the `~/dev/mlx` checkout, uncommitted working tree.

## Review scope

Independent review of the NF4 (NormalFloat4) quantization mode added to MLX.
This is a first implementation — Metal kernels, C++ dispatch, Python API wiring,
and build system integration. No upstream PR yet; review for correctness,
kernel safety, and API coherence before cleanup.

## Review context mode

Target diff only. No inherited implementation-thread context.

## What was added

1. **Metal kernels** (`mlx/backend/metal/kernels/nf4.h`, `nf4_quantized.h`, `nf4_quantized.metal`):
   NF4 lookup-table dequantization kernels — QMV (quad, fast, general), QVM,
   QMM (transposed, non-transposed), split-K, standalone quantize/dequantize.
   NF4 uses a fixed 16-element LUT derived from normal distribution quantiles.
   Scales are float32 absmax per block (not FP8 like nvfp4/mxfp4).

2. **C++ dispatch** (`quantized.cpp`, `jit_kernels.cpp`, `primitives.h`, `primitives.cpp`):
   `QuantizationMode::NF4` enum, string conversion, kernel name prefix routing
   (`nf4_` prefix), NAX exclusion for NF4, JIT source include routing.

3. **Python API** (`ops.cpp`):
   `mx.quantize(mode='nf4')`, `mx.dequantize(mode='nf4')`,
   `mx.quantized_matmul(mode='nf4')`. NF4 quantize uses graph-based fallback
   (not a dedicated GPU kernel). NF4 accepts float32 scales (not uint8).

4. **Build system** (`CMakeLists.txt` in metal/ and metal/kernels/):
   JIT source generation and metallib compilation for NF4 kernels.

## What to look for

- Kernel correctness: nibble unpacking, LUT indexing, scale application, bounds
- Scale pointer arithmetic: NF4 uses 4-byte float32 scales vs 1-byte FP8 — the
  `scale_bytes` multiplier throughout `nf4_quantized.h` is the main divergence
  from `fp_quantized.h` and the most likely source of bugs
- Block loader `NF4BlockLoader` scale stride correctness
- Template instantiation coverage: group sizes, dtypes, batched/unbatched
- API validation: does `validate_mode_with_type` correctly handle NF4's float32
  scales and null biases?
- Missing kernel variants: gather_qmv, gather_qvm, gather_qmm are not
  implemented for NF4 (only the non-gather variants). Is this a gap?
- The standalone `nf4_quantize` Metal kernel has known issues with
  cross-simdgroup absmax for group_size=64 — the GPU quantize path is bypassed
  in favor of graph-based fallback. Is this documented/safe?

## Out of scope

- Performance benchmarking (kernels are expected to be slow; optimization is future work)
- bitsandbytes weight loading (separate work, not in this diff)
- NAX kernel variants (M5+ only, explicitly skipped)
- Upstream PR readiness (this is pre-cleanup)
