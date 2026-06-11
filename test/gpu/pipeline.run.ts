//-------------------------------------------------------------------------------------
// GPU pipeline test: resize, mip generation, and DDS round-trip. Run via vite-node.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT } from "../../src/dxgi.js";
import { makeImage, countMips, type Image } from "../../src/image.js";
import { getDevice } from "../../src/gpu/device.js";
import { resizeImage } from "../../src/gpu/resize.js";
import { convert } from "../../src/texconv.js";
import { writeDDS, readDDS } from "../../src/dds.js";
import { check, report } from "./harness.js";

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

const device = await getDevice();

// Resize: flat color must be preserved exactly.
{
  const src = flat(64, 64, [10, 200, 50, 255]);
  const out = await resizeImage(device, src, 16, 16, false);
  const ok = out.width === 16 && out.height === 16 &&
    out.pixels[0] === 10 && out.pixels[1] === 200 && out.pixels[2] === 50 && out.pixels[3] === 255;
  check("resize 64->16 preserves flat color", ok, `${out.pixels[0]},${out.pixels[1]},${out.pixels[2]}`);
}

// Resize: downscale a 2x2-block constant pattern halves correctly.
{
  const src = gradient(64, 64);
  const out = await resizeImage(device, src, 32, 32, false);
  // Center pixel should be ~midrange; just sanity-check bounds and dims.
  const mid = out.pixels[(16 * out.rowPitch) + 16 * 4];
  check("resize 64->32 dims + plausible value", out.width === 32 && out.height === 32 && mid > 100 && mid < 160, `mid R=${mid}`);
}

// Full mip chain to BC3, then DDS round-trip.
{
  const src = gradient(64, 48);
  const scratch = await convert(src, { format: DXGI_FORMAT.BC3_UNORM, mipLevels: 0 });
  const expected = countMips(64, 48); // 64x48 -> 7 levels
  check("mip chain level count", scratch.metadata.mipLevels === expected, `${scratch.metadata.mipLevels} (want ${expected})`);
  check("base level is 64x48", scratch.images[0].width === 64 && scratch.images[0].height === 48);
  check("last level is 1x1", scratch.images[scratch.images.length - 1].width === 1);

  const dds = writeDDS(scratch);
  const back = readDDS(dds);
  check("DDS round-trip format", back.metadata.format === DXGI_FORMAT.BC3_UNORM);
  check("DDS round-trip mip count", back.metadata.mipLevels === expected, `${back.metadata.mipLevels}`);
  check("DDS round-trip base bytes match", back.images[0].pixels[0] === scratch.images[0].pixels[0]);
}

// Resize + compress combined.
{
  const src = gradient(100, 100);
  const scratch = await convert(src, { format: DXGI_FORMAT.BC1_UNORM, width: 64, height: 64 });
  check("resize+compress base dims", scratch.images[0].width === 64 && scratch.images[0].height === 64);
  check("resize+compress format", scratch.metadata.format === DXGI_FORMAT.BC1_UNORM);
}

report();
