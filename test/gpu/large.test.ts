//-------------------------------------------------------------------------------------
// Regression test for the WebGPU 65535-workgroups-per-dimension limit. Images this
// size made the BC7 multi-subset passes exceed the 1D dispatch cap, silently
// corrupting output (the "red/pink grid"). Now batched / 2D-dispatched.
//-------------------------------------------------------------------------------------

import { beforeAll, expect, test } from "vitest";
import { DXGI_FORMAT } from "../../src/dxgi.js";
import { makeImage, type Image } from "../../src/image.js";
import { compressBC7 } from "../../src/gpu/compressBC.js";
import { compressBCn } from "../../src/gpu/compressBCn.js";
import { decodeBC7 } from "../../src/bc7_decode.js";
import { decodeBCn } from "../../src/bcn_decode.js";
import { getDevice } from "../../src/gpu/device.js";

// 1024x1024 = 256x256 = 65536 blocks, just over WebGPU's 65535 cap — the smallest
// size that still forces the multi-batch / 2D-dispatch path (keeps the test cheap).
const W = 1024, H = 1024;

function makeLargeImage(): Image {
  const img = makeImage(DXGI_FORMAT.R8G8B8A8_UNORM, W, H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = y * img.rowPitch + x * 4;
      img.pixels[o] = (x * 255 / W) | 0;
      img.pixels[o + 1] = (y * 255 / H) | 0;
      img.pixels[o + 2] = 128;
      img.pixels[o + 3] = 255;
    }
  }
  return img;
}

function psnr(a: Image, b: Image, channels: number[]): number {
  let sse = 0, n = 0;
  for (let y = 0; y < H; y += 5) {
    for (let x = 0; x < W; x += 5) {
      const oa = y * a.rowPitch + x * 4, ob = y * b.rowPitch + x * 4;
      for (const c of channels) { const d = a.pixels[oa + c] - b.pixels[ob + c]; sse += d * d; n++; }
    }
  }
  return sse === 0 ? Infinity : 10 * Math.log10(65025 / (sse / n));
}

let device: GPUDevice;
beforeAll(async () => {
  device = await getDevice();
});

test("block count exceeds 1D dispatch cap", () => {
  const blocks = ((W + 3) >> 2) * ((H + 3) >> 2);
  expect(blocks, `${blocks} blocks`).toBeGreaterThan(65535);
});

test("large BC7 PSNR > 40dB", async () => {
  const img = makeLargeImage();
  const bc7 = await compressBC7(device, img, { format: DXGI_FORMAT.BC7_UNORM });
  const q = psnr(img, decodeBC7(bc7), [0, 1, 2]);
  expect(q, `${q.toFixed(1)} dB`).toBeGreaterThan(40);
});

test("large BC1 PSNR > 30dB", async () => {
  const img = makeLargeImage();
  const bc1 = await compressBCn(device, img, DXGI_FORMAT.BC1_UNORM);
  const q = psnr(img, decodeBCn(bc1), [0, 1, 2]);
  expect(q, `${q.toFixed(1)} dB`).toBeGreaterThan(30);
});
