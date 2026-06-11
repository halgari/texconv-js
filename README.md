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
| **M2** | BC7 GPU encoder (WGSL + orchestrator port of `BCDirectCompute.cpp`) | ✅ done |
| **M4** | New WGSL BC1/2/3/4/5 GPU encoders | ✅ done |
| **M5** | GPU mipmaps, resize, sRGB-correct filtering, DDS reader/preview | ✅ done |
| **M6** | texconv CLI option surface + demo polish | ✅ done |
| M3 | BC6H GPU encoder + HDR (.hdr) input | ⏸ deferred (HDR) |

## Develop

```bash
npm install
npm run dev        # serve the demo at http://localhost:5173 (needs a WebGPU browser)
npm test           # CPU unit tests (vitest)
npm run test:gpu   # GPU integration tests, headless via Dawn + vite-node
npm run test:all   # both
npm run typecheck  # tsc --noEmit
npm run build      # build the browser demo artifact into dist-demo/
```

### Testing the GPU path

GPU tests run headlessly through the Dawn-based `webgpu` package under `vite-node`
(so WGSL `?raw` imports and TS work in plain Node). Each suite runs in its own
process and `process.exit(0)`s before Dawn's teardown — a known node-webgpu
segfault-on-exit that would otherwise be reported as a failure. The BC7 encoder is
verified by compressing test images, decoding the result with the reference BC7
decoder (`src/bc7_decode.ts`), and asserting PSNR (≈41 dB on a smooth gradient).

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

### Supported texconv options

`parseTexconvArgs` maps the familiar flags onto `convert()`:

| Flag | Meaning |
|------|---------|
| `-f <fmt>` | output format (DXGI names + `DXT1/3/5`, `RGBA`, `BGRA` aliases) |
| `-w` / `-h` | resize width / height (GPU box filter) |
| `-m <n>` | mip levels (`0` = full chain) |
| `-srgb` / `-srgbi` / `-srgbo` | sRGB-correct filtering |
| `-aw <n>` | BC7 alpha weight |
| `-bc q` / `-bc x` | BC7 quick / 3-subset modes |
| `-px` / `-sx` / `-l` / `-ft` | output name prefix / suffix / lowercase / extension |

File-system flags (`-r`, `-flist`, `-o`, …) are parsed and reported as warnings
in the browser build. The demo accepts these flags directly in a text box.

The upstream C++ source of truth lives at `../DirectXTex`. Mapping of the GPU
path (D3D11 → WebGPU): structured buffers → storage buffers, `cbuffer` → uniform
buffer, `Texture2D` SRV → `texture_2d<f32>` + `textureLoad`, ping-pong `err1`/`err2`
buffers and 64-block batches preserved, staging readback → `mapAsync`.

## License

MIT (matching DirectXTex).
