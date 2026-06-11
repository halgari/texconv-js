//-------------------------------------------------------------------------------------
// GPU integration test for the BC7 encoder. Run with vite-node (gives us ?raw +
// TS in a plain node process), then process.exit(0) before the Dawn addon's
// teardown segfaults. See test/gpu/run-all.mjs.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT } from "../../src/dxgi.js";
import { makeImage, type Image } from "../../src/image.js";
import { getDevice } from "../../src/gpu/device.js";
import { compressBC7 } from "../../src/gpu/compressBC.js";
import { decodeBC7 } from "../../src/bc7_decode.js";
import { check, report } from "./harness.js";

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

const device = await getDevice();

{
  const src = makeTestImage(64, 64);
  const bc7 = await compressBC7(device, src, { format: DXGI_FORMAT.BC7_UNORM });
  check("64x64 output is BC7", bc7.format === DXGI_FORMAT.BC7_UNORM);
  check("64x64 byte size", bc7.pixels.byteLength === 256 * 16, `${bc7.pixels.byteLength}`);
  const q = psnr(src, decodeBC7(bc7));
  check("64x64 gradient PSNR > 40dB", q > 40, `${q.toFixed(2)} dB`);
}

{
  const src = makeTestImage(37, 21);
  const bc7 = await compressBC7(device, src, { format: DXGI_FORMAT.BC7_UNORM });
  const q = psnr(src, decodeBC7(bc7));
  check("37x21 (non-mult-4) no OOB, PSNR > 35dB", q > 35, `${q.toFixed(2)} dB`);
}

{
  const src = makeTestImage(64, 64);
  const bc7q = await compressBC7(device, src, { format: DXGI_FORMAT.BC7_UNORM, quick: true });
  const q = psnr(src, decodeBC7(bc7q));
  check("64x64 quick mode produces valid output", q > 35, `${q.toFixed(2)} dB`);
}

report();
