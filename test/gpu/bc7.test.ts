//-------------------------------------------------------------------------------------
// GPU integration test for the BC7 encoder. Runs in headless Chromium via vitest
// browser mode (real navigator.gpu).
//-------------------------------------------------------------------------------------

import { beforeAll, expect, test } from "vitest";
import { DXGI_FORMAT } from "../../src/dxgi.js";
import { makeImage, type Image } from "../../src/image.js";
import { compressBC7 } from "../../src/gpu/compressBC.js";
import { decodeBC7 } from "../../src/bc7_decode.js";
import { getDevice } from "../../src/gpu/device.js";

function makeTestImage(width: number, height: number): Image {
  const img = makeImage(DXGI_FORMAT.R8G8B8A8_UNORM, width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = y * img.rowPitch + x * 4;
      img.pixels[o] = Math.floor((x / (width - 1)) * 255);
      img.pixels[o + 1] = Math.floor((y / (height - 1)) * 255);
      img.pixels[o + 2] = Math.floor(((x + y) / (width + height - 2)) * 255);
      // Smooth alpha gradient (a high-frequency alpha pattern is not something a
      // single BC7 block can represent, and would just add noise to PSNR).
      img.pixels[o + 3] = Math.floor(((width - 1 - x) / (width - 1)) * 255);
    }
  }
  return img;
}

function psnr(a: Image, b: Image): number {
  let sse = 0;
  let count = 0;
  for (let y = 0; y < a.height; y++) {
    for (let x = 0; x < a.width; x++) {
      const oa = y * a.rowPitch + x * 4;
      const ob = y * b.rowPitch + x * 4;
      for (let c = 0; c < 4; c++) {
        const d = a.pixels[oa + c] - b.pixels[ob + c];
        sse += d * d;
        count++;
      }
    }
  }
  if (sse === 0) return Infinity;
  return 10 * Math.log10((255 * 255) / (sse / count));
}

let device: GPUDevice;
beforeAll(async () => {
  device = await getDevice();
});

test("64x64 gradient compresses to BC7 with PSNR > 40dB", async () => {
  const src = makeTestImage(64, 64);
  const bc7 = await compressBC7(device, src, { format: DXGI_FORMAT.BC7_UNORM });
  expect(bc7.format).toBe(DXGI_FORMAT.BC7_UNORM);
  expect(bc7.pixels.byteLength).toBe(256 * 16);
  const q = psnr(src, decodeBC7(bc7));
  expect(q, `${q.toFixed(2)} dB`).toBeGreaterThan(40);
});

test("37x21 (non-mult-4) no OOB, PSNR > 35dB", async () => {
  const src = makeTestImage(37, 21);
  const bc7 = await compressBC7(device, src, { format: DXGI_FORMAT.BC7_UNORM });
  const q = psnr(src, decodeBC7(bc7));
  expect(q, `${q.toFixed(2)} dB`).toBeGreaterThan(35);
});

test("64x64 quick mode produces valid output", async () => {
  const src = makeTestImage(64, 64);
  const bc7q = await compressBC7(device, src, { format: DXGI_FORMAT.BC7_UNORM, quick: true });
  const q = psnr(src, decodeBC7(bc7q));
  expect(q, `${q.toFixed(2)} dB`).toBeGreaterThan(35);
});
