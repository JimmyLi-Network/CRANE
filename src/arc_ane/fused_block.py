"""Fused VisionBlock MIL generator and runtime for ANE.

A single ANE kernel that executes the entire VisionBlock:
  RMSNorm1 → QKV → RoPE → masked attention → proj → residual →
  RMSNorm2 → gate → SiLU → up → mul → down → residual

All computation stays on ANE (fp16 internal). Inputs/outputs are fp32
for compatibility with the existing pipeline.

For the current fixed 384×384 path, the fused MIL graph comes in two mask
templates:
  - windowed block (block-diagonal attention mask)
  - full-attention block (no mask)

Weights are still baked per block, so a full 32-layer encoder uses
per-block specializations rather than just two compiled kernels.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from arc_ane.bridge import ANEBridgeLibrary, ANEBridgeError, DEFAULT_ANE_BRIDGE_PATH, _free_bridge_blob
from arc_ane.runtime import ANEKernel

# ---------------------------------------------------------------------------
# MIL header and conv constants (shared across all generators)
# ---------------------------------------------------------------------------

_MIL_HEADER = (
    'program(1.3)\n'
    '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]\n'
    '{\n'
)

_CONV_CONSTS = (
    '        string pt = const()[name=string("pt"), val=string("valid")];\n'
    '        tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];\n'
    '        tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];\n'
    '        tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];\n'
    '        int32 gr = const()[name=string("gr"), val=int32(1)];\n'
)


# ---------------------------------------------------------------------------
# MIL sub-generators
# ---------------------------------------------------------------------------


def _emit_rmsnorm(
    input_var: str,
    output_var: str,
    weight_path: str,
    dim: int,
    seq: int,
    prefix: str,
    eps: float = 1e-6,
) -> str:
    """Emit MIL ops for RMSNorm: x → x * rsqrt(mean(x²) + eps) * weight."""
    inv_d = 1.0 / dim
    return (
        f'        tensor<fp16, [1,{dim},1,{seq}]> {prefix}_sq = mul(x={input_var},y={input_var})[name=string("{prefix}_sq")];\n'
        f'        tensor<int32, [1]> {prefix}_rax = const()[name=string("{prefix}_rax"), val=tensor<int32, [1]>([1])];\n'
        f'        bool {prefix}_kd = const()[name=string("{prefix}_kd"), val=bool(true)];\n'
        f'        tensor<fp16, [1,1,1,{seq}]> {prefix}_ss = reduce_sum(x={prefix}_sq,axes={prefix}_rax,keep_dims={prefix}_kd)[name=string("{prefix}_ss")];\n'
        f'        fp16 {prefix}_invd = const()[name=string("{prefix}_invd"), val=fp16({inv_d})];\n'
        f'        tensor<fp16, [1,1,1,{seq}]> {prefix}_mn = mul(x={prefix}_ss,y={prefix}_invd)[name=string("{prefix}_mn")];\n'
        f'        fp16 {prefix}_eps = const()[name=string("{prefix}_eps"), val=fp16({eps})];\n'
        f'        tensor<fp16, [1,1,1,{seq}]> {prefix}_se = add(x={prefix}_mn,y={prefix}_eps)[name=string("{prefix}_se")];\n'
        f'        fp16 {prefix}_nh = const()[name=string("{prefix}_nh"), val=fp16(-0.5)];\n'
        f'        tensor<fp16, [1,1,1,{seq}]> {prefix}_rs = pow(x={prefix}_se,y={prefix}_nh)[name=string("{prefix}_rs")];\n'
        f'        tensor<fp16, [1,{dim},1,{seq}]> {prefix}_nr = mul(x={input_var},y={prefix}_rs)[name=string("{prefix}_nr")];\n'
        f'        tensor<fp16, [1,{dim},1,1]> {prefix}_rw = const()[name=string("{prefix}_rw"), '
        f'val=tensor<fp16, [1,{dim},1,1]>(BLOBFILE(path=string("{weight_path}"), offset=uint64(64)))];\n'
        f'        tensor<fp16, [1,{dim},1,{seq}]> {output_var} = mul(x={prefix}_nr,y={prefix}_rw)[name=string("{prefix}_out")];\n'
    )


def _emit_conv(
    input_var: str,
    output_var: str,
    weight_path: str,
    ic: int,
    oc: int,
    seq: int,
    name: str,
) -> str:
    """Emit a single conv (linear) op with baked weights."""
    return (
        f'        tensor<fp16, [{oc},{ic},1,1]> {name}_W = const()[name=string("{name}_W"), '
        f'val=tensor<fp16, [{oc},{ic},1,1]>(BLOBFILE(path=string("{weight_path}"), offset=uint64(64)))];\n'
        f'        tensor<fp16, [1,{oc},1,{seq}]> {output_var} = conv(dilations=dl,groups=gr,pad=pd,'
        f'pad_type=pt,strides=st,weight={name}_W,x={input_var})[name=string("{name}_cv")];\n'
    )


def _emit_bias_add(
    input_var: str,
    output_var: str,
    bias_path: str,
    oc: int,
    seq: int,
    name: str,
) -> str:
    """Emit bias addition: y = x + bias (broadcast along seq dim)."""
    bias_const_var = f"{name}_bc"
    return (
        f'        tensor<fp16, [1,{oc},1,1]> {bias_const_var} = const()[name=string("{bias_const_var}"), '
        f'val=tensor<fp16, [1,{oc},1,1]>(BLOBFILE(path=string("{bias_path}"), offset=uint64(64)))];\n'
        f'        tensor<fp16, [1,{oc},1,{seq}]> {output_var} = add(x={input_var},y={bias_const_var})[name=string("{name}_ba")];\n'
    )


# ---------------------------------------------------------------------------
# Full fused VisionBlock MIL generator
# ---------------------------------------------------------------------------


def build_fused_vision_block_mil(
    *,
    seq: int,
    dim: int,
    num_heads: int,
    head_dim: int,
    intermediate: int,
    mask_path: str,
    weight_paths: dict[str, str],
    eps: float = 1e-6,
) -> str:
    """Generate MIL text for a complete fused VisionBlock.

    Single packed input (fp32) [1, dim + 2*head_dim, 1, seq]:
      channels [0:dim]                    — hidden states
      channels [dim:dim+head_dim]         — rotary cos
      channels [dim+head_dim:dim+2*head_dim] — rotary sin

    Single output (fp32) [1, dim, 1, seq]:
      block output hidden states

    Internal computation is fp16.
    """
    input_channels = dim + 2 * head_dim
    lines: list[str] = []
    lines.append(_MIL_HEADER)
    lines.append(
        f'    func main<ios18>('
        f'tensor<fp32, [1, {input_channels}, 1, {seq}]> packed_in'
        f') {{\n'
    )

    # Cast packed input to fp16 and unpack via slicing
    lines.append(f'        string to16 = const()[name=string("to16"), val=string("fp16")];\n')
    lines.append(f'        string to32 = const()[name=string("to32"), val=string("fp32")];\n')
    lines.append(f'        tensor<fp16, [1,{input_channels},1,{seq}]> packed = cast(dtype=to16,x=packed_in)[name=string("cp")];\n')
    # Slice out hidden_states, cos, sin
    lines.append(f'        tensor<int32, [4]> x_begin = const()[name=string("xb"), val=tensor<int32, [4]>([0,0,0,0])];\n')
    lines.append(f'        tensor<int32, [4]> x_size = const()[name=string("xsz"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];\n')
    lines.append(f'        tensor<fp16, [1,{dim},1,{seq}]> x = slice_by_size(x=packed,begin=x_begin,size=x_size)[name=string("sx")];\n')
    lines.append(f'        tensor<int32, [4]> cos_begin = const()[name=string("cosb"), val=tensor<int32, [4]>([0,{dim},0,0])];\n')
    lines.append(f'        tensor<int32, [4]> cos_size = const()[name=string("cossz"), val=tensor<int32, [4]>([1,{head_dim},1,{seq}])];\n')
    lines.append(f'        tensor<fp16, [1,{head_dim},1,{seq}]> cos = slice_by_size(x=packed,begin=cos_begin,size=cos_size)[name=string("scos")];\n')
    lines.append(f'        tensor<int32, [4]> sin_begin = const()[name=string("sinb"), val=tensor<int32, [4]>([0,{dim + head_dim},0,0])];\n')
    lines.append(f'        tensor<fp16, [1,{head_dim},1,{seq}]> sin = slice_by_size(x=packed,begin=sin_begin,size=cos_size)[name=string("ssin")];\n')

    # Conv constants
    lines.append(_CONV_CONSTS)

    # ===== Attention path =====

    # RMSNorm1
    lines.append(_emit_rmsnorm('x', 'xn1', weight_paths['norm1'], dim, seq, 'n1', eps=eps))

    # QKV conv (fused into one conv with oc=3*dim)
    qkv_dim = 3 * dim
    lines.append(_emit_conv('xn1', 'qkv', weight_paths['qkv_weight'], dim, qkv_dim, seq, 'qkv'))
    lines.append(_emit_bias_add('qkv', 'qkvb', weight_paths['qkv_bias'], qkv_dim, seq, 'qkv'))

    # Split QKV: slice along channel dimension
    # qkvb shape: [1, 3*dim, 1, seq] -> Q [1, dim, 1, seq], K, V
    lines.append(f'        tensor<int32, [4]> q_begin = const()[name=string("qb"), val=tensor<int32, [4]>([0,0,0,0])];\n')
    lines.append(f'        tensor<int32, [4]> q_size = const()[name=string("qs"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];\n')
    lines.append(f'        tensor<fp16, [1,{dim},1,{seq}]> Q = slice_by_size(x=qkvb,begin=q_begin,size=q_size)[name=string("sq")];\n')

    lines.append(f'        tensor<int32, [4]> k_begin = const()[name=string("kb"), val=tensor<int32, [4]>([0,{dim},0,0])];\n')
    lines.append(f'        tensor<fp16, [1,{dim},1,{seq}]> K = slice_by_size(x=qkvb,begin=k_begin,size=q_size)[name=string("sk")];\n')

    lines.append(f'        tensor<int32, [4]> v_begin = const()[name=string("vb"), val=tensor<int32, [4]>([0,{2*dim},0,0])];\n')
    lines.append(f'        tensor<fp16, [1,{dim},1,{seq}]> V = slice_by_size(x=qkvb,begin=v_begin,size=q_size)[name=string("sv")];\n')

    # RoPE: Q_rot = Q * cos + rotate_half(Q) * sin
    # In ANE layout [1, dim, 1, seq], channels = dim = num_heads * head_dim
    # cos/sin are [1, head_dim, 1, seq] — need to tile across heads
    # Actually, Qwen2.5-VL visual RoPE applies per-token cos/sin uniformly across heads
    # cos shape [seq, head_dim] → ANE [1, head_dim, 1, seq]
    # Q shape [1, dim, 1, seq] → reshape to [1, num_heads, head_dim, seq]
    # Apply RoPE per head, then reshape back

    # Reshape Q to [1, num_heads, head_dim, seq] for per-head RoPE
    lines.append(f'        tensor<int32, [4]> head_shape = const()[name=string("hs"), val=tensor<int32, [4]>([1,{num_heads},{head_dim},{seq}])];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{head_dim},{seq}]> Qh = reshape(shape=head_shape,x=Q)[name=string("rq")];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{head_dim},{seq}]> Kh = reshape(shape=head_shape,x=K)[name=string("rk")];\n')

    # Broadcast cos/sin from [1, head_dim, 1, seq] → [1, num_heads, head_dim, seq]
    # Use tile/broadcast — but MIL might not have explicit tile
    # Alternative: reshape cos to [1, 1, head_dim, seq] then broadcast works automatically
    lines.append(f'        tensor<int32, [4]> cos_rs = const()[name=string("crs"), val=tensor<int32, [4]>([1,1,{head_dim},{seq}])];\n')
    lines.append(f'        tensor<fp16, [1,1,{head_dim},{seq}]> cos_4d = reshape(shape=cos_rs,x=cos)[name=string("cr4")];\n')
    lines.append(f'        tensor<fp16, [1,1,{head_dim},{seq}]> sin_4d = reshape(shape=cos_rs,x=sin)[name=string("sr4")];\n')

    # rotate_half: split head_dim in half, negate+swap
    half_hd = head_dim // 2
    lines.append(f'        tensor<int32, [4]> rh_b1 = const()[name=string("rb1"), val=tensor<int32, [4]>([0,0,0,0])];\n')
    lines.append(f'        tensor<int32, [4]> rh_s1 = const()[name=string("rs1"), val=tensor<int32, [4]>([1,{num_heads},{half_hd},{seq}])];\n')
    lines.append(f'        tensor<int32, [4]> rh_b2 = const()[name=string("rb2"), val=tensor<int32, [4]>([0,0,{half_hd},0])];\n')

    # Q rotate_half
    lines.append(f'        tensor<fp16, [1,{num_heads},{half_hd},{seq}]> Q_lo = slice_by_size(x=Qh,begin=rh_b1,size=rh_s1)[name=string("ql")];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{half_hd},{seq}]> Q_hi = slice_by_size(x=Qh,begin=rh_b2,size=rh_s1)[name=string("qh")];\n')
    lines.append(f'        fp16 neg1 = const()[name=string("neg"), val=fp16(-1.0)];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{half_hd},{seq}]> Q_hi_neg = mul(x=Q_hi,y=neg1)[name=string("qhn")];\n')
    lines.append(f'        int32 cat_ax = const()[name=string("cax"), val=int32(2)];\n')
    lines.append(f'        bool cat_il = const()[name=string("cil"), val=bool(false)];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{head_dim},{seq}]> Q_rot = concat(axis=cat_ax,interleave=cat_il,values=(Q_hi_neg,Q_lo))[name=string("qr")];\n')

    # K rotate_half
    lines.append(f'        tensor<fp16, [1,{num_heads},{half_hd},{seq}]> K_lo = slice_by_size(x=Kh,begin=rh_b1,size=rh_s1)[name=string("kl")];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{half_hd},{seq}]> K_hi = slice_by_size(x=Kh,begin=rh_b2,size=rh_s1)[name=string("kh")];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{half_hd},{seq}]> K_hi_neg = mul(x=K_hi,y=neg1)[name=string("khn")];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{head_dim},{seq}]> K_rot = concat(axis=cat_ax,interleave=cat_il,values=(K_hi_neg,K_lo))[name=string("kr")];\n')

    # Apply RoPE: Q_roped = Qh * cos + Q_rot * sin
    lines.append(f'        tensor<fp16, [1,{num_heads},{head_dim},{seq}]> Qc = mul(x=Qh,y=cos_4d)[name=string("qc")];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{head_dim},{seq}]> Qs = mul(x=Q_rot,y=sin_4d)[name=string("qs2")];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{head_dim},{seq}]> Qr = add(x=Qc,y=Qs)[name=string("qrp")];\n')

    lines.append(f'        tensor<fp16, [1,{num_heads},{head_dim},{seq}]> Kc = mul(x=Kh,y=cos_4d)[name=string("kc")];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{head_dim},{seq}]> Ks = mul(x=K_rot,y=sin_4d)[name=string("ks2")];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{head_dim},{seq}]> Kr = add(x=Kc,y=Ks)[name=string("krp")];\n')

    # Attention: scores = Qr @ Kr^T * scale
    # Qr: [1, heads, head_dim, seq] → transpose to [1, heads, seq, head_dim]
    lines.append(f'        tensor<int32, [4]> attn_pm = const()[name=string("apm"), val=tensor<int32, [4]>([0,1,3,2])];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{seq},{head_dim}]> Qt = transpose(perm=attn_pm,x=Qr)[name=string("qt")];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{seq},{head_dim}]> Kt = transpose(perm=attn_pm,x=Kr)[name=string("kt")];\n')

    # Reshape V for attention: [1, dim, 1, seq] → [1, heads, head_dim, seq] → [1, heads, seq, head_dim]
    lines.append(f'        tensor<fp16, [1,{num_heads},{head_dim},{seq}]> Vh = reshape(shape=head_shape,x=V)[name=string("rvh")];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{seq},{head_dim}]> Vt = transpose(perm=attn_pm,x=Vh)[name=string("vt")];\n')

    # Q @ K^T
    lines.append(f'        bool mm_tx = const()[name=string("mtx"), val=bool(false)];\n')
    lines.append(f'        bool mm_ty = const()[name=string("mty"), val=bool(true)];\n')
    scale = 1.0 / (head_dim ** 0.5)
    lines.append(f'        tensor<fp16, [1,{num_heads},{seq},{seq}]> scores = matmul(transpose_x=mm_tx,transpose_y=mm_ty,x=Qt,y=Kt)[name=string("mm1")];\n')
    lines.append(f'        fp16 scv = const()[name=string("scv"), val=fp16({scale})];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{seq},{seq}]> scores_s = mul(x=scores,y=scv)[name=string("scl")];\n')

    # Apply attention mask (baked const)
    lines.append(
        f'        tensor<fp16, [1,1,{seq},{seq}]> attn_mask = const()[name=string("mask"), '
        f'val=tensor<fp16, [1,1,{seq},{seq}]>(BLOBFILE(path=string("{mask_path}"), offset=uint64(64)))];\n'
    )
    lines.append(f'        tensor<fp16, [1,{num_heads},{seq},{seq}]> masked = add(x=scores_s,y=attn_mask)[name=string("msk")];\n')

    # Softmax
    lines.append(f'        int32 sm_ax = const()[name=string("sax"), val=int32(-1)];\n')
    lines.append(f'        tensor<fp16, [1,{num_heads},{seq},{seq}]> attn_w = softmax(axis=sm_ax,x=masked)[name=string("sm")];\n')

    # attn_w @ V
    lines.append(f'        tensor<fp16, [1,{num_heads},{seq},{head_dim}]> attn_out = matmul(transpose_x=mm_tx,transpose_y=mm_tx,x=attn_w,y=Vt)[name=string("mm2")];\n')

    # Reshape back: [1, heads, seq, head_dim] → [1, heads, head_dim, seq] → [1, dim, 1, seq]
    lines.append(f'        tensor<fp16, [1,{num_heads},{head_dim},{seq}]> ao_t = transpose(perm=attn_pm,x=attn_out)[name=string("aot")];\n')
    lines.append(f'        tensor<int32, [4]> flat_shape = const()[name=string("fs"), val=tensor<int32, [4]>([1,{dim},1,{seq}])];\n')
    lines.append(f'        tensor<fp16, [1,{dim},1,{seq}]> ao_flat = reshape(shape=flat_shape,x=ao_t)[name=string("aof")];\n')

    # Output projection
    lines.append(_emit_conv('ao_flat', 'proj_out', weight_paths['proj_weight'], dim, dim, seq, 'proj'))
    lines.append(_emit_bias_add('proj_out', 'proj_b', weight_paths['proj_bias'], dim, seq, 'proj'))

    # Residual 1: h = x + proj_b
    lines.append(f'        tensor<fp16, [1,{dim},1,{seq}]> h = add(x=x,y=proj_b)[name=string("res1")];\n')

    # ===== MLP path =====

    # RMSNorm2
    lines.append(_emit_rmsnorm('h', 'xn2', weight_paths['norm2'], dim, seq, 'n2', eps=eps))

    # Gate + SiLU
    lines.append(_emit_conv('xn2', 'gate_out', weight_paths['gate_weight'], dim, intermediate, seq, 'gate'))
    lines.append(_emit_bias_add('gate_out', 'gate_b', weight_paths['gate_bias'], intermediate, seq, 'gate'))
    lines.append(f'        tensor<fp16, [1,{intermediate},1,{seq}]> gate_sig = sigmoid(x=gate_b)[name=string("gsig")];\n')
    lines.append(f'        tensor<fp16, [1,{intermediate},1,{seq}]> gate_silu = mul(x=gate_b,y=gate_sig)[name=string("gsilu")];\n')

    # Up
    lines.append(_emit_conv('xn2', 'up_out', weight_paths['up_weight'], dim, intermediate, seq, 'up'))
    lines.append(_emit_bias_add('up_out', 'up_b', weight_paths['up_bias'], intermediate, seq, 'up'))

    # SwiGLU: gate_silu * up_b
    lines.append(f'        tensor<fp16, [1,{intermediate},1,{seq}]> swiglu = mul(x=gate_silu,y=up_b)[name=string("swiglu")];\n')

    # Down
    lines.append(_emit_conv('swiglu', 'down_out', weight_paths['down_weight'], intermediate, dim, seq, 'down'))
    lines.append(_emit_bias_add('down_out', 'down_b', weight_paths['down_bias'], dim, seq, 'down'))

    # Residual 2: y = h + down_b
    lines.append(f'        tensor<fp16, [1,{dim},1,{seq}]> y16 = add(x=h,y=down_b)[name=string("res2")];\n')

    # Cast output to fp32
    lines.append(f'        tensor<fp32, [1,{dim},1,{seq}]> y = cast(dtype=to32,x=y16)[name=string("out")];\n')

    lines.append('    } -> (y);\n')
    lines.append('}\n')

    return ''.join(lines)


# ---------------------------------------------------------------------------
# Attention mask construction
# ---------------------------------------------------------------------------


def build_windowed_attention_mask(cu_seqlens: np.ndarray, seq: int) -> np.ndarray:
    """Build a block-diagonal attention mask for windowed attention.

    Returns float16 array of shape (seq, seq):
      0.0  where tokens can attend to each other (same window)
      -inf where tokens cannot attend (-65504 for fp16)
    """
    mask = np.full((seq, seq), -65504.0, dtype=np.float16)
    lengths = np.diff(cu_seqlens.astype(np.int32))
    start = 0
    for length in lengths:
        end = start + int(length)
        mask[start:end, start:end] = 0.0
        start = end
    return mask


def build_full_attention_mask(seq: int) -> np.ndarray:
    """All-zeros mask (full attention, no masking)."""
    return np.zeros((seq, seq), dtype=np.float16)


# ---------------------------------------------------------------------------
# Weight blob helpers
# ---------------------------------------------------------------------------


def _build_weight_blob_fp16(bridge: ANEBridgeLibrary, data: np.ndarray) -> bytes:
    """Build an ANE weight blob from a contiguous float32 array."""
    flat = np.ascontiguousarray(data.astype(np.float32).ravel())
    out_len = ctypes.c_size_t()
    ptr = bridge.lib.ane_bridge_build_weight_blob(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        1,  # rows (treat as 1D)
        len(flat),  # cols
        ctypes.byref(out_len),
    )
    if not ptr:
        raise ANEBridgeError("ane_bridge_build_weight_blob failed")
    try:
        return ctypes.string_at(ptr, out_len.value)
    finally:
        _free_bridge_blob(ptr)


def _build_conv_weight_blob(bridge: ANEBridgeLibrary, weight: np.ndarray, oc: int, ic: int) -> bytes:
    """Build blob for a conv weight [oc, ic, 1, 1] from row-major (oc, ic) or (ic, oc).T."""
    w = np.ascontiguousarray(weight.astype(np.float32))
    out_len = ctypes.c_size_t()
    ptr = bridge.lib.ane_bridge_build_weight_blob(
        w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        oc,
        ic,
        ctypes.byref(out_len),
    )
    if not ptr:
        raise ANEBridgeError("ane_bridge_build_weight_blob failed for conv weight")
    try:
        return ctypes.string_at(ptr, out_len.value)
    finally:
        _free_bridge_blob(ptr)


def _build_bias_blob(bridge: ANEBridgeLibrary, bias: np.ndarray) -> bytes:
    """Build blob for a bias vector [oc] stored as [1, oc, 1, 1]."""
    return _build_weight_blob_fp16(bridge, bias.ravel())


def _build_norm_weight_blob(bridge: ANEBridgeLibrary, weight: np.ndarray) -> bytes:
    """Build blob for RMSNorm weight [dim] stored as [1, dim, 1, 1]."""
    return _build_weight_blob_fp16(bridge, weight.ravel())


def _build_mask_blob(bridge: ANEBridgeLibrary, mask: np.ndarray) -> bytes:
    """Build blob for attention mask [seq, seq] stored as [1, 1, seq, seq]."""
    return _build_weight_blob_fp16(bridge, mask.ravel())


# ---------------------------------------------------------------------------
# Fused block compilation
# ---------------------------------------------------------------------------


_FUSED_BLOCK_CACHE: dict[str, ANEKernel] = {}


def _fused_block_cache_key(
    *,
    bridge_path: str,
    logical_name: str,
    seq: int,
    dim: int,
    num_heads: int,
    head_dim: int,
    intermediate: int,
    attention_mask: np.ndarray,
    eps: float,
) -> str:
    """Build a cache key for the current fixed-resolution fused-block regime.

    The current implementation only distinguishes mask classes by the number of
    zero entries, which is sufficient for the fixed 384x384 path where we only
    compile the standard windowed and full-attention masks.
    """
    mask_zeros = int((attention_mask == 0).sum())
    return (
        f"{bridge_path}:{logical_name}:s{seq}:d{dim}:h{num_heads}:hd{head_dim}:"
        f"i{intermediate}:mz{mask_zeros}:eps{eps:.12g}"
    )


def compile_fused_vision_block(
    *,
    block_weights: dict[str, np.ndarray],
    attention_mask: np.ndarray,
    seq: int,
    dim: int,
    num_heads: int,
    head_dim: int,
    intermediate: int,
    logical_name: str,
    eps: float = 1e-6,
    bridge_path: str = DEFAULT_ANE_BRIDGE_PATH,
) -> ANEKernel:
    """Compile a fused VisionBlock kernel with all weights baked in.

    block_weights must contain:
      norm1, norm2,
      qkv_weight (oc, ic), qkv_bias,
      proj_weight (oc, ic), proj_bias,
      gate_weight (oc, ic), gate_bias,
      up_weight (oc, ic), up_bias,
      down_weight (oc, ic), down_bias

    Cache key includes bridge_path, logical_name, seq, dim, num_heads,
    head_dim, intermediate, the current mask class, and eps to prevent stale
    hits under the fixed 384x384 regime.
    """
    cache_key = _fused_block_cache_key(
        bridge_path=bridge_path,
        logical_name=logical_name,
        seq=seq,
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        intermediate=intermediate,
        attention_mask=attention_mask,
        eps=eps,
    )
    if cache_key in _FUSED_BLOCK_CACHE:
        return _FUSED_BLOCK_CACHE[cache_key]

    bridge = ANEBridgeLibrary(bridge_path)

    # Build all weight blobs
    weight_blobs: dict[str, bytes] = {}
    weight_paths: dict[str, str] = {}

    def _add_norm(key: str, path_name: str) -> None:
        path = f"@model_path/weights/{path_name}.bin"
        weight_paths[key] = path
        weight_blobs[path] = _build_norm_weight_blob(bridge, block_weights[key])

    def _add_conv(key: str, path_name: str, oc: int, ic: int) -> None:
        path = f"@model_path/weights/{path_name}.bin"
        weight_paths[f"{key}"] = path
        w = block_weights[key]
        weight_blobs[path] = _build_conv_weight_blob(bridge, w, oc, ic)

    def _add_bias(key: str, path_name: str) -> None:
        path = f"@model_path/weights/{path_name}.bin"
        weight_paths[key] = path
        weight_blobs[path] = _build_bias_blob(bridge, block_weights[key])

    _add_norm('norm1', 'norm1')
    _add_norm('norm2', 'norm2')
    _add_conv('qkv_weight', 'qkv_w', 3 * dim, dim)
    _add_bias('qkv_bias', 'qkv_b')
    _add_conv('proj_weight', 'proj_w', dim, dim)
    _add_bias('proj_bias', 'proj_b')
    _add_conv('gate_weight', 'gate_w', intermediate, dim)
    _add_bias('gate_bias', 'gate_b')
    _add_conv('up_weight', 'up_w', intermediate, dim)
    _add_bias('up_bias', 'up_b')
    _add_conv('down_weight', 'down_w', dim, intermediate)
    _add_bias('down_bias', 'down_b')

    # Attention mask
    mask_path = "@model_path/weights/attn_mask.bin"
    weight_blobs[mask_path] = _build_mask_blob(bridge, attention_mask)

    mil = build_fused_vision_block_mil(
        seq=seq,
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        intermediate=intermediate,
        mask_path=mask_path,
        weight_paths=weight_paths,
        eps=eps,
    )

    input_channels = dim + 2 * head_dim
    kernel = ANEKernel.compile_multi_weights(
        mil_text=mil,
        weight_blobs=weight_blobs,
        input_shapes=((1, input_channels, 1, seq),),
        output_shapes=((1, dim, 1, seq),),
        input_dtypes=(np.float32,),
        output_dtypes=(np.float32,),
        bridge_path=bridge_path,
    )

    _FUSED_BLOCK_CACHE[cache_key] = kernel
    return kernel


# ---------------------------------------------------------------------------
# Runtime: pack inputs, run, unpack output
# ---------------------------------------------------------------------------


def run_fused_vision_block(
    kernel: ANEKernel,
    hidden_states: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    seq: int,
    dim: int,
    head_dim: int,
) -> np.ndarray:
    """Run a compiled fused VisionBlock kernel.

    hidden_states: (seq, dim) row-major
    cos: (seq, head_dim) row-major
    sin: (seq, head_dim) row-major

    Returns: (seq, dim) row-major
    """
    # Pack to ANE layout: concat [hidden_states, cos, sin] along channel dim
    # Each is transposed from (seq, C) → (C, seq) → [1, C, 1, seq]
    h = np.asarray(hidden_states, dtype=np.float32).T  # (dim, seq)
    c = np.asarray(cos, dtype=np.float32).T             # (head_dim, seq)
    s = np.asarray(sin, dtype=np.float32).T             # (head_dim, seq)
    packed = np.ascontiguousarray(
        np.concatenate([h, c, s], axis=0).reshape(1, dim + 2 * head_dim, 1, seq)
    )

    outputs = kernel.run([packed])
    # Unpack: [1, dim, 1, seq] → (seq, dim)
    return np.asarray(outputs[0][0, :, 0, :], dtype=np.float32).T
