//-------------------------------------------------------------------------------------
// GPU pipeline test: resize, mip generation, and DDS round-trip. Runs in headless
// Chromium via vitest browser mode.
//-------------------------------------------------------------------------------------

import { beforeAll, expect, test } from "vitest";
import { DXGI_FORMAT } from "../../src/dxgi.js";
import { makeImage, countMips, type Image } from "../../src/image.js";
import { resizeImage } from "../../src/gpu/resize.js";
import { convert } from "../../src/texconv.js";
import { writeDDS, readDDS } from "../../src/dds.js";
import { getDevice } from "../../src/gpu/device.js";

function gradient(width: number, height: number): Image {
  const img = makeImage(DXGI_FORMAT.R8G8B8A8_UNORM, width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = y * img.rowPitch + x * 4;
      img.pixels[o] = Math.floor((x / (width - 1)) * 255);
      img.pixels[o + 1] = Math.floor((y / (height - 1)) * 255);
      img.pixels[o + 2] = 128;
      img.pixels[o + 3] = 255;
    }
  }
  return img;
}

function flat(width: number, height: number, rgba: [number, number, number, number]): Image {
  const img = makeImage(DXGI_FORMAT.R8G8B8A8_UNORM, width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = y * img.rowPitch + x * 4;
      img.pixels[o] = rgba[0]; img.pixels[o + 1] = rgba[1]; img.pixels[o + 2] = rgba[2]; img.pixels[o + 3] = rgba[3];
    }
  }
  return img;
}

let device: GPUDevice;
beforeAll(async () => {
  device = await getDevice();
});

test("resize 64->16 preserves flat color", async () => {
  const src = flat(64, 64, [10, 200, 50, 255]);
  const out = await resizeImage(device, src, 16, 16, false);
  expect(out.width).toBe(16);
  expect(out.height).toBe(16);
  expect([out.pixels[0], out.pixels[1], out.pixels[2], out.pixels[3]]).toEqual([10, 200, 50, 255]);
});

test("resize 64->32 dims + plausible value", async () => {
  const src = gradient(64, 64);
  const out = await resizeImage(device, src, 32, 32, false);
  // Center pixel should be ~midrange; just sanity-check bounds and dims.
  const mid = out.pixels[(16 * out.rowPitch) + 16 * 4];
  expect(out.width).toBe(32);
  expect(out.height).toBe(32);
  expect(mid, `mid R=${mid}`).toBeGreaterThan(100);
  expect(mid, `mid R=${mid}`).toBeLessThan(160);
});

test("full mip chain to BC3, then DDS round-trip", async () => {
  const src = gradient(64, 48);
  const scratch = await convert(src, { format: DXGI_FORMAT.BC3_UNORM, mipLevels: 0 });
  const expected = countMips(64, 48); // 64x48 -> 7 levels
  expect(scratch.metadata.mipLevels).toBe(expected);
  expect(scratch.images[0].width).toBe(64);
  expect(scratch.images[0].height).toBe(48);
  expect(scratch.images[scratch.images.length - 1].width).toBe(1);

  const dds = writeDDS(scratch);
  const back = readDDS(dds);
  expect(back.metadata.format).toBe(DXGI_FORMAT.BC3_UNORM);
  expect(back.metadata.mipLevels).toBe(expected);
  expect(back.images[0].pixels[0]).toBe(scratch.images[0].pixels[0]);
});

test("resize + compress combined", async () => {
  const src = gradient(100, 100);
  const scratch = await convert(src, { format: DXGI_FORMAT.BC1_UNORM, width: 64, height: 64 });
  expect(scratch.images[0].width).toBe(64);
  expect(scratch.images[0].height).toBe(64);
  expect(scratch.metadata.format).toBe(DXGI_FORMAT.BC1_UNORM);
});
