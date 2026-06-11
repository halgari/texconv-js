//-------------------------------------------------------------------------------------
// GPU integration test for the BC6H encoder: encode HDR float gradients, decode with
// the reference CPU decoder, and assert quality in linear float space.
//-------------------------------------------------------------------------------------

import { beforeAll, expect, test } from "vitest";
import { DXGI_FORMAT } from "../../src/dxgi.js";
import { makeImage, type Image } from "../../src/image.js";
import { compressBC6H } from "../../src/gpu/compressBC6.js";
import { decodeBC6H } from "../../src/bc6h_decode.js";
import { convert } from "../../src/texconv.js";
import { getDevice } from "../../src/gpu/device.js";

/** Smooth HDR gradient with values well above 1.0 (and below 0 when signed). */
function makeHDRImage(width: number, height: number, signed = false): Image {
  const img = makeImage(DXGI_FORMAT.R32G32B32A32_FLOAT, width, height);
  const f = new Float32Array(img.pixels.buffer);
  let o = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const u = x / (width - 1);
      const v = y / (height - 1);
      f[o++] = u * 4.0; // 0..4 — beyond LDR range
      f[o++] = 0.05 + v * 1.5;
      f[o++] = signed ? (u - 0.5) * 2.0 : (u + v) * 0.5;
      f[o++] = 1;
    }
  }
  return img;
}

/** PSNR over RGB in linear float space, peak = max source component. */
function psnrHDR(a: Image, b: Image): number {
  const fa = new Float32Array(a.pixels.buffer);
  const fb = new Float32Array(b.pixels.buffer);
  let sse = 0;
  let n = 0;
  let peak = 0;
  for (let i = 0; i < fa.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const d = fa[i + c] - fb[i + c];
      sse += d * d;
      n++;
      peak = Math.max(peak, Math.abs(fa[i + c]));
    }
  }
  if (sse === 0) return Infinity;
  return 10 * Math.log10((peak * peak) / (sse / n));
}

let device: GPUDevice;
beforeAll(async () => {
  device = await getDevice();
});

test("BC6H_UF16 64x64 HDR gradient PSNR > 40dB", async () => {
  const src = makeHDRImage(64, 64);
  const enc = await compressBC6H(device, src, { format: DXGI_FORMAT.BC6H_UF16 });
  expect(enc.format).toBe(DXGI_FORMAT.BC6H_UF16);
  expect(enc.pixels.byteLength).toBe(256 * 16);
  const q = psnrHDR(src, decodeBC6H(enc));
  expect(q, `${q.toFixed(2)} dB`).toBeGreaterThan(40);
});

test("BC6H_SF16 64x64 gradient PSNR > 40dB", async () => {
  // Positive-range data through the signed format (different quantization scale
  // than UF16, so this still exercises the SF16 code paths).
  const src = makeHDRImage(64, 64);
  const enc = await compressBC6H(device, src, { format: DXGI_FORMAT.BC6H_SF16 });
  expect(enc.format).toBe(DXGI_FORMAT.BC6H_SF16);
  const q = psnrHDR(src, decodeBC6H(enc));
  expect(q, `${q.toFixed(2)} dB`).toBeGreaterThan(40);
});

test("BC6H_SF16 clamps negative inputs to zero (upstream GPU-encoder behavior)", async () => {
  // DirectXTex's DirectCompute encoder does max(pixel, 0) on input for BOTH
  // formats (BC6HEncode.hlsl), so even SF16 never encodes negative values.
  // This test documents that we faithfully match texconv -gpu.
  const src = makeHDRImage(8, 8, true); // blue channel spans -1..1
  const enc = await compressBC6H(device, src, { format: DXGI_FORMAT.BC6H_SF16 });
  const f = new Float32Array(decodeBC6H(enc).pixels.buffer);
  let minBlue = Infinity;
  for (let i = 2; i < f.length; i += 4) minBlue = Math.min(minBlue, f[i]);
  expect(minBlue).toBeGreaterThanOrEqual(0);
});

test("BC6H_UF16 37x21 (non-mult-4) no OOB, PSNR > 35dB", async () => {
  const src = makeHDRImage(37, 21);
  const enc = await compressBC6H(device, src, { format: DXGI_FORMAT.BC6H_UF16 });
  const q = psnrHDR(src, decodeBC6H(enc));
  expect(q, `${q.toFixed(2)} dB`).toBeGreaterThan(35);
});

test("convert() pipeline: LDR source + resize + mips to BC6H_UF16", async () => {
  // 8-bit source goes through sRGB->linear float expansion, float resize, float mips.
  const ldr = makeImage(DXGI_FORMAT.R8G8B8A8_UNORM, 64, 64);
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      const o = y * ldr.rowPitch + x * 4;
      ldr.pixels[o] = x * 4;
      ldr.pixels[o + 1] = y * 4;
      ldr.pixels[o + 2] = 128;
      ldr.pixels[o + 3] = 255;
    }
  }
  const scratch = await convert(ldr, { format: DXGI_FORMAT.BC6H_UF16, width: 32, height: 32, mipLevels: 0 });
  expect(scratch.metadata.format).toBe(DXGI_FORMAT.BC6H_UF16);
  expect(scratch.metadata.mipLevels).toBe(6); // 32x32 -> 6 levels
  expect(scratch.images[0].width).toBe(32);
  expect(scratch.images[scratch.images.length - 1].width).toBe(1);
  // Decoded base must be in plausible linear range (no NaN/garbage).
  const dec = decodeBC6H(scratch.baseImage);
  const f = new Float32Array(dec.pixels.buffer);
  for (let i = 0; i < 16; i++) expect(Number.isFinite(f[i])).toBe(true);
});

test("float source to non-BC6H target is rejected", async () => {
  const src = makeHDRImage(8, 8);
  await expect(convert(src, { format: DXGI_FORMAT.BC7_UNORM })).rejects.toThrow(/BC6H/);
});
