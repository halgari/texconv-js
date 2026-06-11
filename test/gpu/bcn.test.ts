//-------------------------------------------------------------------------------------
// GPU integration test for the BC1/2/3/4/5 encoders. Runs in headless Chromium via
// vitest browser mode.
//-------------------------------------------------------------------------------------

import { beforeAll, expect, test } from "vitest";
import { DXGI_FORMAT } from "../../src/dxgi.js";
import { makeImage, type Image } from "../../src/image.js";
import { compressBCn } from "../../src/gpu/compressBCn.js";
import { decodeBCn } from "../../src/bcn_decode.js";
import { getDevice } from "../../src/gpu/device.js";

function makeTestImage(width: number, height: number): Image {
  const img = makeImage(DXGI_FORMAT.R8G8B8A8_UNORM, width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = y * img.rowPitch + x * 4;
      img.pixels[o] = Math.floor((x / (width - 1)) * 255);
      img.pixels[o + 1] = Math.floor((y / (height - 1)) * 255);
      img.pixels[o + 2] = Math.floor(((x + y) / (width + height - 2)) * 255);
      img.pixels[o + 3] = Math.floor(((width - 1 - x) / (width - 1)) * 255);
    }
  }
  return img;
}

// Per-channel PSNR over a chosen channel mask (which channels matter for a format).
function psnr(a: Image, b: Image, channels: number[]): number {
  let sse = 0;
  let count = 0;
  for (let y = 0; y < a.height; y++) {
    for (let x = 0; x < a.width; x++) {
      const oa = y * a.rowPitch + x * 4;
      const ob = y * b.rowPitch + x * 4;
      for (const c of channels) {
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

const cases: { name: string; fmt: DXGI_FORMAT; channels: number[]; min: number }[] = [
  { name: "BC1 (RGB)", fmt: DXGI_FORMAT.BC1_UNORM, channels: [0, 1, 2], min: 30 },
  { name: "BC2 (RGB+4bit A)", fmt: DXGI_FORMAT.BC2_UNORM, channels: [0, 1, 2, 3], min: 30 },
  { name: "BC3 (RGB+A)", fmt: DXGI_FORMAT.BC3_UNORM, channels: [0, 1, 2, 3], min: 30 },
  { name: "BC4 (R)", fmt: DXGI_FORMAT.BC4_UNORM, channels: [0], min: 40 },
  { name: "BC5 (RG)", fmt: DXGI_FORMAT.BC5_UNORM, channels: [0, 1], min: 40 },
];

for (const c of cases) {
  test(`${c.name} PSNR > ${c.min}dB`, async () => {
    const src = makeTestImage(64, 64);
    const enc = await compressBCn(device, src, c.fmt);
    const dec = decodeBCn(enc);
    const q = psnr(src, dec, c.channels);
    expect(q, `${q.toFixed(2)} dB`).toBeGreaterThan(c.min);
  });
}

test("BC3 30x18 (non-mult-4) must not crash / corrupt", async () => {
  const odd = makeTestImage(30, 18);
  const enc = await compressBCn(device, odd, DXGI_FORMAT.BC3_UNORM);
  const dec = decodeBCn(enc);
  const q = psnr(odd, dec, [0, 1, 2, 3]);
  expect(q, `${q.toFixed(2)} dB`).toBeGreaterThan(28);
});
