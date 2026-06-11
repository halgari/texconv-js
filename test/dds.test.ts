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

/** Hand-build a legacy (non-DX10) FourCC DDS for the given format + one block. */
function legacyFourCCDDS(fourCC: string, width: number, height: number, fill: number): Uint8Array {
  const slice = ((width + 3) >> 2) * ((height + 3) >> 2) * 16; // BC2/BC3 = 16 B/block
  const buf = new ArrayBuffer(4 + 124 + slice);
  const view = new DataView(buf);
  view.setUint32(0, 0x20534444, true); // "DDS "
  view.setUint32(4, 124, true); // header size
  view.setUint32(8, 0x1007 | 0x80000, true); // flags (texture + linearsize)
  view.setUint32(12, height, true);
  view.setUint32(16, width, true);
  // DDS_PIXELFORMAT at offset 4 + 72 = 76
  view.setUint32(76, 32, true); // pf size
  view.setUint32(80, 0x4, true); // DDPF_FOURCC
  view.setUint32(84, [...fourCC].reduce((a, ch, i) => a | (ch.charCodeAt(0) << (i * 8)), 0) >>> 0, true);
  new Uint8Array(buf).fill(fill, 4 + 124);
  return new Uint8Array(buf);
}

describe("readDDS legacy FourCC", () => {
  it("maps DXT5 to BC3_UNORM and reads the block", () => {
    const back = readDDS(legacyFourCCDDS("DXT5", 4, 4, 0x5a));
    expect(back.metadata.format).toBe(DXGI_FORMAT.BC3_UNORM);
    expect(back.metadata.width).toBe(4);
    expect(back.images.length).toBe(1);
    expect(back.images[0].pixels[0]).toBe(0x5a);
  });

  it("maps DXT3 to BC2_UNORM", () => {
    const back = readDDS(legacyFourCCDDS("DXT3", 8, 8, 0x11));
    expect(back.metadata.format).toBe(DXGI_FORMAT.BC2_UNORM);
  });

  it("throws on an unsupported FourCC", () => {
    expect(() => readDDS(legacyFourCCDDS("ZZZZ", 4, 4, 0))).toThrow(/unsupported FourCC/);
  });
});
