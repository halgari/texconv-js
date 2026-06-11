# texconv-js

A TypeScript / **WebGPU** port of Microsoft [DirectXTex](https://github.com/microsoft/DirectXTex)'s
`texconv` texture converter, designed to run in the browser. Block compression
runs on the GPU via compute shaders (the same approach as DirectXTex's
`BCDirectCompute`, translated from HLSL to WGSL).

> **Why WebGPU, not WebGL?** The BC6H/BC7 encoders are *compute* shaders. WebGL
> has no compute shaders; only WebGPU does. The HLSL kernels port ~1:1 to WGSL.

## Status

| Milestone | Scope | State |
|-----------|-------|-------|
| **M1** | Scaffold, DXGI format tables, pitch, DDS read/write, PNG→DDS demo | ✅ done |
| M2 | BC7 GPU encoder (WGSL + orchestrator port of `BCDirectCompute.cpp`) | ⏳ |
| M3 | BC6H GPU encoder + HDR (.hdr) input | ⏳ |
| M4 | New WGSL BC1/2/3/4/5 GPU encoders | ⏳ |
| M5 | Mipmaps, resize, sRGB, DDS reader/preview | ⏳ |
| M6 | texconv CLI option surface + demo polish | ⏳ |

## Develop

```bash
npm install
npm run dev        # serve the demo at http://localhost:5173 (needs a WebGPU browser)
npm test           # unit tests (vitest)
npm run typecheck  # tsc --noEmit
npm run build      # build the browser demo artifact into dist-demo/
```

## Architecture

```
src/
  dxgi.ts      DXGI_FORMAT enum + metadata (bits/pixel, block size, sRGB) + ComputePitch
  image.ts     Image / ScratchImage / TexMetadata (mirrors DirectXTex)
  dds.ts       DDS container read/write (DX10 extended header)
  io.ts        browser image decode (createImageBitmap → RGBA8) + download helper
  texconv.ts   high-level convert(image, options) pipeline
  gpu/         WebGPU device + WGSL BC encoders + dispatch orchestrator   (M2+)
demo/          drag-drop browser UI driving the pipeline
test/          vitest unit tests
```

The upstream C++ source of truth lives at `../DirectXTex`. Mapping of the GPU
path (D3D11 → WebGPU): structured buffers → storage buffers, `cbuffer` → uniform
buffer, `Texture2D` SRV → `texture_2d<f32>` + `textureLoad`, ping-pong `err1`/`err2`
buffers and 64-block batches preserved, staging readback → `mapAsync`.

## License

MIT (matching DirectXTex).
