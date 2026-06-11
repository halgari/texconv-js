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
| **M3** | BC6H GPU encoder (UF16/SF16) + HDR (.hdr) input | ✅ done |
| **M4** | New WGSL BC1/2/3/4/5 GPU encoders | ✅ done |
| **M5** | GPU mipmaps, resize, sRGB-correct filtering, DDS reader/preview | ✅ done |
| **M6** | texconv CLI option surface + demo polish | ✅ done |

> BC6H matches upstream texconv's GPU encoder (`BC6HEncode.hlsl`), including its
> `max(pixel, 0)` input clamp — negative values encode as zero even for SF16.
> `.hdr` (Radiance RGBE) input decodes to linear float; LDR inputs are expanded
> sRGB→linear before BC6H encoding. Mips/resize for HDR run in linear float.

## Develop

This project uses **pnpm** (see `pnpm-lock.yaml`).

```bash
pnpm install
pnpm dev          # serve the demo at http://localhost:5173 (needs a WebGPU browser)
pnpm test         # CPU unit tests (vitest, Node)
pnpm test:gpu     # GPU integration tests in headless Chromium (vitest browser mode)
pnpm test:all     # both
pnpm typecheck    # tsc --noEmit
pnpm build        # build the browser demo artifact into dist-demo/
pnpm build:lib    # bundle the library into dist/ (with pnpm build:types for .d.ts)
```

GPU tests need a one-time Playwright Chromium download:

```bash
pnpm exec playwright install chromium
```

### Testing the GPU path

GPU tests run in real headless Chromium via vitest browser mode (Playwright
provider), i.e. against the same `navigator.gpu` path the library uses in
production. Chromium is launched with the Linux WebGPU flags
(`--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan`); see
`vitest.config.ts`. If headless WebGPU misbehaves on your driver, set
`headless: false` there as an escape hatch. The BC7 encoder is verified by
compressing test images, decoding the result with the reference BC7 decoder
(`src/bc7_decode.ts`), and asserting PSNR (≈41 dB on a smooth gradient).

## Troubleshooting: "no GPU adapter available"

If `navigator.gpu` exists but `requestAdapter()` returns `null` (common on **Linux
Chromium**), WebGPU is present but gated behind flags/drivers:

1. Open `chrome://flags`, enable **Unsafe WebGPU Support**
   (`#enable-unsafe-webgpu`) and, if present, **Vulkan** (`#enable-vulkan`).
2. Restart the browser.
3. Open `chrome://gpu` and confirm **WebGPU** shows *Enabled* / a Vulkan device.
   If WebGPU is "Disabled" or "Software only", your GPU/driver may be on Chromium's
   blocklist — launch with:
   ```
   chromium --enable-unsafe-webgpu --enable-features=Vulkan
   ```
   (some setups also need `--use-angle=vulkan` or `--ignore-gpu-blocklist`).

Firefox Nightly and Chrome/Edge on Windows/macOS generally work without flags.

## Architecture

```
src/
  dxgi.ts      DXGI_FORMAT enum + metadata (bits/pixel, block size, sRGB) + ComputePitch
  image.ts     Image / ScratchImage / TexMetadata (mirrors DirectXTex)
  dds.ts       DDS container read/write (DX10 extended header)
  io.ts        browser image decode (createImageBitmap → RGBA8) + download helper
  hdr.ts       Radiance .hdr (RGBE) reader + LDR→linear-float + tonemap helpers
  texconv.ts   high-level convert(image, options) pipeline
  gpu/         WebGPU device + WGSL BC encoders + dispatch orchestrator   (M2+)
demo/          drag-drop browser UI driving the pipeline
test/          vitest unit tests
```

### Supported texconv options

texconv-js ships no standalone command-line binary — `parseTexconvArgs`
(`src/cli.ts`) parses a texconv-style argument string into `ConvertOptions` so
the same flags drive `convert()` from the browser (the demo exposes them in a
text box). It maps:

| Flag | Meaning |
|------|---------|
| `-f <fmt>` | output format (DXGI names + `DXT1/3/5`, `RGBA`, `BGRA` aliases) |
| `-w` / `-h` | resize width / height (GPU box filter) |
| `-m <n>` | mip levels (`0` = full chain) |
| `-srgb` / `-srgbi` / `-srgbo` | input is sRGB (filtering) / output tagged `_SRGB` / both |
| `-hflip` / `-vflip` | mirror horizontally / vertically |
| `-swizzle <mask>` | channel swizzle (`rgba`/`xyzw`/`0`/`1`, e.g. `abgr`, `rgb1`) |
| `-c <hex>` | colorkey: matching pixels → transparent black |
| `-pmalpha` / `-alpha` | premultiply alpha (after mips) / undo premultiplied alpha |
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
