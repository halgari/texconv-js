import { describe, it, expect } from "vitest";
import { DXGI_FORMAT, computePitch } from "../src/dxgi.js";
import { makeImage, ScratchImage } from "../src/image.js";
import { writeDDS, readDDS } from "../src/dds.js";

describe("computePitch", () => {
  it("uncompressed RGBA8", () => {
    expect(computePitch(DXGI_FORMAT.R8G8B8A8_UNORM, 16, 8)).toEqual({ rowPitch: 64, slicePitch: 512 });
  });
  it("BC1 block (8 bytes/block)", () => {
    // 16x16 -> 4x4 blocks -> rowPitch 4*8=32, slice 32*4=128
    expect(computePitch(DXGI_FORMAT.BC1_UNORM, 16, 16)).toEqual({ rowPitch: 32, slicePitch: 128 });
  });
  it("BC7 block (16 bytes/block), non-multiple-of-4", () => {
    // 5x5 -> 2x2 blocks -> rowPitch 2*16=32, slice 32*2=64
    expect(computePitch(DXGI_FORMAT.BC7_UNORM, 5, 5)).toEqual({ rowPitch: 32, slicePitch: 64 });
  });
});

describe("DDS write/read round-trip", () => {
  it("round-trips an uncompressed RGBA8 image via DX10 header", () => {
    const img = makeImage(DXGI_FORMAT.R8G8B8A8_UNORM, 4, 4);
    for (let i = 0; i < img.pixels.length; i++) img.pixels[i] = (i * 7) & 0xff;
    const scratch = ScratchImage.from2D(img);

    const dds = writeDDS(scratch);
    // 4 (magic) + 124 (header) + 20 (dx10) + 4*4*4 (pixels)
    expect(dds.byteLength).toBe(148 + 64);

    const back = readDDS(dds);
    expect(back.metadata.format).toBe(DXGI_FORMAT.R8G8B8A8_UNORM);
    expect(back.metadata.width).toBe(4);
    expect(back.metadata.height).toBe(4);
    expect(back.images.length).toBe(1);
    expect(Array.from(back.images[0].pixels)).toEqual(Array.from(img.pixels));
  });

  it("round-trips a BC7 mip chain header", () => {
    const base = makeImage(DXGI_FORMAT.BC7_UNORM, 8, 8);
    const mip1 = makeImage(DXGI_FORMAT.BC7_UNORM, 4, 4);
    base.pixels.fill(0xab);
    mip1.pixels.fill(0xcd);
    const scratch = ScratchImage.fromMipChain([base, mip1]);

    const back = readDDS(writeDDS(scratch));
    expect(back.metadata.mipLevels).toBe(2);
    expect(back.metadata.format).toBe(DXGI_FORMAT.BC7_UNORM);
    expect(back.images[0].pixels[0]).toBe(0xab);
    expect(back.images[1].pixels[0]).toBe(0xcd);
  });
});
