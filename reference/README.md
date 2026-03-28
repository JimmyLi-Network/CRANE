# Reference: ANE MIL Kernel Patterns

Objective-C MIL generators from the [ANE Training project](https://github.com/Maderix/ANE) — the original reverse-engineering work that proved ANE is directly programmable via private APIs.

These files are included as reference for anyone implementing new ANE kernels. The Python equivalents in `crane/` were derived from these patterns.

## Files

| File | What it contains |
|------|-----------------|
| `ane_runtime.h` | Original ANE runtime: compile, eval, read/write IOSurface, free. Foundation for `ane_bridge.m`. |
| `ane_mil_gen.h` | MIL generators: baked-weight conv, dynamic matmul, fused QKV (3 parallel convs), fused FFN up (2 parallel convs), weight blob builders. |
| `stories_mil.h` | Full fused forward kernels: SDPA (RMSNorm + QKV + multi-head attention + output proj) and FFN (RMSNorm + SwiGLU MLP). Block-level fusion reference. |
| `mil_dynamic_gqa.h` | GQA-aware dynamic kernels for Qwen3-0.6B (grouped-query attention with Q_DIM != KV_DIM). |

## Key Patterns

### Weight blob format (128-byte header + fp16 data)
```c
buf[0] = 0x01; buf[4] = 0x02;                    // global header
buf[64] = 0xEF; buf[65] = 0xBE; ...              // chunk magic: 0xDEADBEEF
*(uint32_t*)(buf + 72) = wsize;                    // data size in bytes
*(uint32_t*)(buf + 80) = 128;                      // data offset from file start
// fp16 weights start at byte 128
```

### RMSNorm in MIL
```
sq = mul(x, x)
ss = reduce_sum(sq, axis=channel, keep_dims=true)
mn = mul(ss, 1/dim)
se = add(mn, eps)
rs = pow(se, -0.5)
nr = mul(x, rs)
out = mul(nr, weight)
```

### Baked-weight conv (linear projection)
```
W = const()[val = tensor<fp16, [oc, ic, 1, 1]>(BLOBFILE(...))]
y = conv(weight=W, x=input, strides=[1,1], pad=[0,0,0,0])
```

### Multi-head attention
```
Q, K, V = split(conv_qkv(norm(x)))        # 3 parallel convs or 1 fused QKV conv
Q = reshape(Q, [1, heads, head_dim, seq])
scores = matmul(Q, K^T) * scale            # Q @ K^T
masked = add(scores, attention_mask)        # baked mask const
weights = softmax(masked, axis=-1)
out = matmul(weights, V)                    # attn @ V
out = reshape(out, [1, dim, 1, seq])
out = conv_proj(out)                        # output projection
```

## License

MIT — from the original [ANE Training project](https://github.com/Maderix/ANE)
