"""Smoke tests for ANE bridge — requires Apple Silicon with ANE."""
import os
from pathlib import Path

import numpy as np
import pytest

from arc_ane.bridge import ANEBridgeLibrary, DEFAULT_ANE_BRIDGE_PATH, ANEBridgeError
from arc_ane.runtime import ANEKernel, compile_dyn_matmul_kernel, run_dyn_matmul_kernel
from arc_ane.bridge import run_dyn_matmul


BRIDGE_PATH = os.environ.get("ARC_ANE_BRIDGE_PATH", DEFAULT_ANE_BRIDGE_PATH)
SKIP_NO_ANE = not Path(BRIDGE_PATH).exists()


@pytest.mark.skipif(SKIP_NO_ANE, reason="ANE bridge dylib not found")
def test_bridge_init() -> None:
    bridge = ANEBridgeLibrary(BRIDGE_PATH)
    assert bridge.lib is not None


@pytest.mark.skipif(SKIP_NO_ANE, reason="ANE bridge dylib not found")
def test_dyn_matmul_identity() -> None:
    x = np.eye(16, dtype=np.float32)
    w = np.eye(16, dtype=np.float32)
    out = run_dyn_matmul(x, w, bridge_path=BRIDGE_PATH)
    np.testing.assert_allclose(out, x, atol=1e-2)


@pytest.mark.skipif(SKIP_NO_ANE, reason="ANE bridge dylib not found")
def test_baked_kernel_compile_and_run() -> None:
    from arc_ane.runtime import compile_baked_linear_kernel, run_baked_linear_kernel

    w = np.eye(16, dtype=np.float32)
    kernel = compile_baked_linear_kernel(
        ic=16, oc=16, seq=16,
        logical_kernel_name="test.identity",
        weights=w,
        bridge_path=BRIDGE_PATH,
    )
    x = np.ones((16, 16), dtype=np.float32)
    out = run_baked_linear_kernel(kernel, x)
    assert out.shape == (16, 16)
    assert np.isfinite(out).all()


@pytest.mark.skipif(SKIP_NO_ANE, reason="ANE bridge dylib not found")
def test_fused_vision_block_small() -> None:
    from arc_ane.fused_block import (
        compile_fused_vision_block,
        run_fused_vision_block,
        build_full_attention_mask,
    )

    seq, dim, heads, hd, inter = 16, 32, 4, 8, 48
    np.random.seed(42)
    weights = {
        "norm1": np.random.randn(dim).astype(np.float32),
        "norm2": np.random.randn(dim).astype(np.float32),
        "qkv_weight": np.random.randn(3 * dim, dim).astype(np.float32) * 0.1,
        "qkv_bias": np.random.randn(3 * dim).astype(np.float32) * 0.01,
        "proj_weight": np.random.randn(dim, dim).astype(np.float32) * 0.1,
        "proj_bias": np.random.randn(dim).astype(np.float32) * 0.01,
        "gate_weight": np.random.randn(inter, dim).astype(np.float32) * 0.1,
        "gate_bias": np.random.randn(inter).astype(np.float32) * 0.01,
        "up_weight": np.random.randn(inter, dim).astype(np.float32) * 0.1,
        "up_bias": np.random.randn(inter).astype(np.float32) * 0.01,
        "down_weight": np.random.randn(dim, inter).astype(np.float32) * 0.1,
        "down_bias": np.random.randn(dim).astype(np.float32) * 0.01,
    }
    mask = build_full_attention_mask(seq)
    kernel = compile_fused_vision_block(
        block_weights=weights,
        attention_mask=mask,
        seq=seq, dim=dim, num_heads=heads, head_dim=hd, intermediate=inter,
        logical_name="test.small_block",
        bridge_path=BRIDGE_PATH,
    )
    x = np.random.randn(seq, dim).astype(np.float32)
    cos = np.ones((seq, hd), dtype=np.float32)
    sin = np.zeros((seq, hd), dtype=np.float32)
    out = run_fused_vision_block(kernel, x, cos, sin, seq=seq, dim=dim, head_dim=hd)
    assert out.shape == (seq, dim)
    assert np.isfinite(out).all()
