# ARC-ANE

**A Runtime for Compiled ANE Neural Execution**

Direct Python control of Apple Neural Engine (ANE) via reverse-engineered private APIs. Compile MIL programs with baked weights, execute fused transformer blocks on ANE hardware, and cache kernels for repeated inference — no Core ML required.

## What This Is

- A Python runtime for executing custom compute graphs directly on Apple's Neural Engine
- Fused transformer block kernels: RMSNorm + multi-head attention (with RoPE and windowed masking) + SwiGLU MLP + residual connections — compiled into a single ANE evaluation
- Compile-time weight baking via `_ANEInMemoryModel` private APIs
- Dynamic kernel cache keyed by shape, mask structure, and block identity
- C bridge (`libane_bridge.dylib`) wrapping `_ANECompiler` / `_ANEInMemoryModelDescriptor` / `_ANERequest` into a ctypes-friendly interface

## What This Is Not

- A replacement for Core ML, MLX, or any production inference stack
- A general-purpose ANE programming framework
- Tested on anything other than Apple Silicon M-series with macOS 15+

## Results

Measured on Qwen2.5-VL-3B-Instruct vision encoder (32 transformer blocks, dim=1280, 16 heads, head_dim=80), 384x384 input image, Apple Silicon:

### Optimization Progression

| Stage | Warm Latency (ms) | Improvement |
|-------|-------------------:|:-----------:|
| Per-operator ANE kernels, dynamic weights | 5,819 | baseline |
| + Compile-time weight baking (partial) | 4,498 | 1.3x |
| + Remove sequence chunking, full bake | 1,315 | 4.4x |
| + **Fused block kernel** | **260** | **22.4x** |

### Fused Block: Per-Block Latency

| Approach | Per-Block (ms) | ANE Evals/Block |
|----------|---------------:|----------------:|
| Unfused (5 separate ANE calls + CPU ops) | 36.4 | 5 |
| Fused (1 single ANE call) | **7.9** | **1** |
| **Speedup** | **4.6x** | |

### Parity vs Reference (MLX bfloat16)

| Block Type | max_abs_diff | mean_abs_diff |
|------------|-------------:|--------------:|
| Windowed attention (block 0) | 0.066 | 0.003 |
| Full attention (block 7) | 0.023 | 0.003 |

## Architecture

```
Python (numpy)                    ANE Hardware
    |                                 |
    |  pack [1, C, 1, S]             |
    +--- write_input ------>  IOSurface (shared memory)
    |                                 |
    |                         compile MIL -> ANE program
    |                         bake weights as constants
    |                                 |
    +--- eval -------------->  ANE executes fused kernel
    |                                 |
    |  unpack (S, C)                 |
    +<-- read_output --------  IOSurface (shared memory)
```

A fused VisionBlock kernel contains ~94 MIL operations:
- 2x RMSNorm (`reduce_sum` + `pow` + `rsqrt` + `mul`)
- 5x linear projection (`conv` with baked `[oc, ic, 1, 1]` weights)
- 5x bias addition
- 1x RoPE (`slice` + negate + `concat` + `mul` + `add`)
- 1x multi-head attention (`reshape` + `transpose` + `matmul` + masked `softmax` + `matmul`)
- 1x SiLU activation (`sigmoid` + `mul`)
- 2x residual `add`
- I/O casts (fp32 at boundaries, fp16 internally)

## Building

Requires macOS 15+ on Apple Silicon (M1/M2/M3/M4).

```bash
cd arc-ane
make
```

This compiles `libane_bridge.dylib` from `src/ane_bridge.m`. No external dependencies — uses only system frameworks (`Foundation`, `IOSurface`) and private ANE APIs resolved at runtime via `dlopen`.

## Usage

### Basic: Dynamic Matrix Multiply

```python
import numpy as np
from arc_ane import ANEBridgeLibrary
from arc_ane.bridge import run_dyn_matmul

x = np.random.randn(64, 128).astype(np.float32)
w = np.random.randn(128, 256).astype(np.float32)
out = run_dyn_matmul(x, w)  # (64, 256), runs on ANE
```

### Baked-Weight Linear

```python
from arc_ane import compile_baked_linear_kernel
from arc_ane.runtime import run_baked_linear_kernel

kernel = compile_baked_linear_kernel(
    ic=128, oc=256, seq=64,
    logical_kernel_name="my_linear",
    weights=w,  # baked at compile time
)
out = run_baked_linear_kernel(kernel, x)  # only activation transferred per call
```

### Fused Transformer Block

```python
from arc_ane import (
    compile_fused_vision_block,
    run_fused_vision_block,
    build_windowed_attention_mask,
)

# All weights baked at compile time
kernel = compile_fused_vision_block(
    block_weights=weights_dict,
    attention_mask=build_windowed_attention_mask(cu_seqlens, seq_len),
    seq=784, dim=1280, num_heads=16, head_dim=80, intermediate=3420,
    logical_name="block.0",
)

# Per-image: only hidden_states + cos + sin transferred
out = run_fused_vision_block(kernel, hidden_states, cos, sin,
                              seq=784, dim=1280, head_dim=80)
```

## Testing

```bash
make test
```

Or manually:

```bash
ARC_ANE_BRIDGE_PATH=src/libane_bridge.dylib python -m pytest tests/ -v
```

## File Structure

```
arc-ane/
  src/
    ane_bridge.h          # C API header
    ane_bridge.m          # Objective-C bridge implementation
    arc_ane/
      __init__.py         # Public API
      bridge.py           # Python ctypes bindings + MIL generators
      runtime.py          # ANEKernel compile/run/cache management
      fused_block.py      # Fused VisionBlock MIL generator + runtime
  reference/
    ane_runtime.h         # Original ANE runtime (compile/eval/IOSurface)
    ane_mil_gen.h         # MIL generators: conv, matmul, fused QKV, fused FFN
    stories_mil.h         # Fused SDPA + FFN forward kernels (block-level fusion)
    mil_dynamic_gqa.h     # GQA-aware dynamic kernels (Qwen3-0.6B)
    README.md             # Key MIL patterns and weight blob format reference
  tests/
    test_bridge.py        # ANE hardware tests
  Makefile
  README.md
```

The `reference/` directory contains the original Objective-C MIL generators from the [ANE Training project](https://github.com/Maderix/ANE) that established the foundational patterns for ANE kernel programming. These include fused SDPA forward kernels, GQA attention, weight blob construction, and the runtime API that `ane_bridge.m` wraps.

## Limitations

- **Private APIs**: Uses `_ANEClient`, `_ANECompiler`, `_ANEInMemoryModelDescriptor` — undocumented, may break with macOS updates
- **fp16 internal precision**: ANE computes in fp16; input/output are fp32 for compatibility
- **Static shapes**: MIL programs are compiled for fixed tensor shapes; different resolutions need recompilation
- **Single input tensor**: ANE kernels accept one input; multiple tensors are packed via channel concatenation
- **macOS 15+ required**: Tested on M-series chips only

## Disclaimer

This project uses Apple's private, undocumented APIs for research purposes. These APIs may change or break with any macOS update. This is independent research, not affiliated with or endorsed by Apple Inc. See *Sega v. Accolade* (1992) and DMCA Section 1201(f) regarding reverse engineering for interoperability.

## License

MIT
