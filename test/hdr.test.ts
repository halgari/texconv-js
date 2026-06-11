//-------------------------------------------------------------------------------------
// Unit tests for the Radiance .hdr loader (flat and RLE scanlines) and the
// LDR->float / tonemap helpers.
//-------------------------------------------------------------------------------------

import { expect, test } from "vitest";
import { DXGI_FORMAT } from "../src/dxgi.js";
import { loadHDR, rgba8ToLinearFloat, tonemapToRGBA8 } from "../src/hdr.js";
import { makeImage } from "../src/image.js";

/** Build a minimal flat (non-RLE) .hdr file with the given RGBE pixels. */
function makeHDR(width: number, height: number, rgbe: number[][]): Uint8Array {
  const header = `#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y ${height} +X ${width}\n`;
  const head = new TextEncoder().encode(header);
  const body = new Uint8Array(width * height * 4);
  rgbe.forEach((px, i) => body.set(px, i * 4));
  const out = new Uint8Array(head.length + body.length);
  out.set(head);
  out.set(body, head.length);
  return out;
}

test("flat scanlines decode to linear floats", () => {
  // (128, 64, 255, 129): scale = 2^(129-136) = 1/128 -> (1.0, 0.5, ~1.99)
  const img = loadHDR(makeHDR(2, 1, [[128, 64, 255, 129], [0, 0, 0, 0]]));
  expect(img.format).toBe(DXGI_FORMAT.R32G32B32A32_FLOAT);
  expect(img.width).toBe(2);
  const f = new Float32Array(img.pixels.buffer);
  expect(f[0]).toBeCloseTo(1.0, 5);
  expect(f[1]).toBeCloseTo(0.5, 5);
  expect(f[2]).toBeCloseTo(255 / 128, 5);
  expect(f[3]).toBe(1);
  expect(f[4]).toBe(0); // zero exponent = black
});

test("new-style RLE scanlines decode (run + literal)", () => {
  // 8-wide scanline: per channel a run of 8 identical bytes.
  const width = 8;
  const header = `#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X ${width}\n`;
  const head = new TextEncoder().encode(header);
  const planes: number[] = [2, 2, (width >> 8) & 0xff, width & 0xff];
  for (const v of [200, 100, 50, 130]) planes.push(128 + width, v); // RLE run of 8
  const data = new Uint8Array([...head, ...planes]);

  const img = loadHDR(data);
  const f = new Float32Array(img.pixels.buffer);
  const scale = Math.pow(2, 130 - 136);
  for (let x = 0; x < width; x++) {
    expect(f[x * 4]).toBeCloseTo(200 * scale, 5);
    expect(f[x * 4 + 1]).toBeCloseTo(100 * scale, 5);
    expect(f[x * 4 + 2]).toBeCloseTo(50 * scale, 5);
  }
});

test("new-style RLE literal spans decode (count <= 128)", () => {
  const width = 8;
  const header = `#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X ${width}\n`;
  const head = new TextEncoder().encode(header);
  const planes: number[] = [2, 2, (width >> 8) & 0xff, width & 0xff];
  // Per channel: a literal span of `width` distinct bytes (count <= 128).
  const rVals = [10, 20, 30, 40, 50, 60, 70, 80];
  const eVals = 136; // exponent so scale = 2^(136-136) = 1
  for (const ch of [rVals, rVals.map((v) => v + 1), rVals.map((v) => v + 2)]) {
    planes.push(width, ...ch);
  }
  planes.push(128 + width, eVals); // exponent: run of constant 136
  const img = loadHDR(new Uint8Array([...head, ...planes]));
  const f = new Float32Array(img.pixels.buffer);
  expect(f[0]).toBeCloseTo(10, 5); // r at x=0
  expect(f[4 * 7]).toBeCloseTo(80, 5); // r at x=7
});

test("old-style RLE runs repeat the previous pixel", () => {
  // width 4: pixel0 normal, (1,1,1,2) repeats it twice, then pixel3 normal.
  const header = `#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 4\n`;
  const head = new TextEncoder().encode(header);
  const body = [128, 64, 32, 136, /*run*/ 1, 1, 1, 2, /*px3*/ 200, 100, 50, 136];
  const img = loadHDR(new Uint8Array([...head, ...body]));
  const f = new Float32Array(img.pixels.buffer);
  // x0, x1, x2 share the first pixel's value; x3 differs.
  expect(f[0]).toBeCloseTo(128, 5);
  expect(f[4]).toBeCloseTo(128, 5);
  expect(f[8]).toBeCloseTo(128, 5);
  expect(f[12]).toBeCloseTo(200, 5);
});

test("truncated scanline data throws", () => {
  const header = `#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 4\n`;
  const head = new TextEncoder().encode(header);
  expect(() => loadHDR(new Uint8Array([...head, 10, 20]))).toThrow(/truncated/);
});

test("rgba8ToLinearFloat applies the sRGB curve and tonemapToRGBA8 round-trips LDR", () => {
  const src = makeImage(DXGI_FORMAT.R8G8B8A8_UNORM, 2, 1);
  src.pixels.set([255, 188, 0, 255, 0, 0, 0, 128]);
  const lin = rgba8ToLinearFloat(src);
  const f = new Float32Array(lin.pixels.buffer);
  expect(f[0]).toBeCloseTo(1.0, 5);
  expect(f[1]).toBeCloseTo(Math.pow((188 / 255 + 0.055) / 1.055, 2.4), 4);
  expect(f[3]).toBe(1);
  expect(f[7]).toBeCloseTo(128 / 255, 3);

  // Reinhard maps 1.0 -> 0.5 linear; verify the sRGB-encoded result.
  const tone = tonemapToRGBA8(lin);
  const expected = Math.round((1.055 * Math.pow(0.5, 1 / 2.4) - 0.055) * 255);
  expect(tone.pixels[0]).toBe(expected);
});
