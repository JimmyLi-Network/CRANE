"""CRANE: Compiled Runtime for Apple Neural Engine.

Direct Python control of Apple Neural Engine via reverse-engineered private APIs.
Supports compile-time weight baking, fused transformer block execution, IOSurface
chaining, and dynamic kernel caching — no Core ML required.
"""

__version__ = "0.2.0"

from crane.bridge import ANEBridgeError, ANEBridgeLibrary, DEFAULT_ANE_BRIDGE_PATH
from crane.runtime import ANEKernel, compile_baked_linear_kernel, compile_dyn_matmul_kernel
from crane.fused_block import (
    build_fused_vision_block_mil,
    build_full_attention_mask,
    build_windowed_attention_mask,
    compile_fused_vision_block,
    run_fused_vision_block,
    run_ping_pong_chained_fused_vision_blocks,
)
