//-------------------------------------------------------------------------------------
// dds.ts
//
// DDS container read/write. The writer always emits the modern DX10 extended
// header (matching texconv's default), which carries the DXGI_FORMAT directly and
// avoids the legacy FourCC/bitmask ambiguity. A minimal reader covers the DX10 and
// common legacy-FourCC cases for round-tripping.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT, computePitch, isCompressed } from "./dxgi.js";
import { ScratchImage, TEX_DIMENSION, type Image, type TexMetadata } from "./image.js";

const DDS_MAGIC = 0x20534444; // "DDS "

// DDS_HEADER flags
const DDSD_CAPS = 0x1;
const DDSD_HEIGHT = 0x2;
const DDSD_WIDTH = 0x4;
const DDSD_PIXELFORMAT = 0x1000;
const DDSD_MIPMAPCOUNT = 0x20000;
const DDSD_LINEARSIZE = 0x80000;
const DDSD_PITCH = 0x8;
const DDS_HEADER_FLAGS_TEXTURE = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT; // 0x1007

// DDS_PIXELFORMAT flags
const DDPF_FOURCC = 0x4;

// caps
const DDSCAPS_TEXTURE = 0x1000;
const DDSCAPS_COMPLEX = 0x8;
const DDSCAPS_MIPMAP = 0x400000;

const DDS_HEADER_SIZE = 124; // sizeof(DDS_HEADER)
const DDS_PIXELFORMAT_SIZE = 32;
const DDS_DX10_EXT_SIZE = 20; // sizeof(DDS_HEADER_DXT10)

function makeFourCC(s: string): number {
  return (
    (s.charCodeAt(0) & 0xff) |
    ((s.charCodeAt(1) & 0xff) << 8) |
    ((s.charCodeAt(2) & 0xff) << 16) |
    ((s.charCodeAt(3) & 0xff) << 24)
  ) >>> 0;
}

export const enum DDS_ALPHA_MODE {
  UNKNOWN = 0,
  STRAIGHT = 1,
  PREMULTIPLIED = 2,
  OPAQUE = 3,
  CUSTOM = 4,
}

export interface WriteDDSOptions {
  alphaMode?: DDS_ALPHA_MODE;
}

/**
 * Serialize a ScratchImage to a DDS byte buffer using the DX10 extended header.
 * Images are written in DirectXTex subresource order: for a 2D mip chain that is
 * simply mip 0..N within array slice 0.
 */
export function writeDDS(scratch: ScratchImage, opts: WriteDDSOptions = {}): Uint8Array {
  const meta = scratch.metadata;
  const headerBytes = 4 + DDS_HEADER_SIZE + DDS_DX10_EXT_SIZE;

  let dataBytes = 0;
  for (const img of scratch.images) dataBytes += img.slicePitch;

  const buf = new ArrayBuffer(headerBytes + dataBytes);
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);
  let o = 0;
  const u32 = (v: number) => {
    view.setUint32(o, v >>> 0, true);
    o += 4;
  };

  u32(DDS_MAGIC);

  // --- DDS_HEADER (124 bytes) ---
  const compressed = isCompressed(meta.format);
  const base = scratch.images[0];
  let flags = DDS_HEADER_FLAGS_TEXTURE;
  if (meta.mipLevels > 1) flags |= DDSD_MIPMAPCOUNT;
  flags |= compressed ? DDSD_LINEARSIZE : DDSD_PITCH;

  u32(DDS_HEADER_SIZE); // size
  u32(flags); // flags
  u32(meta.height); // height
  u32(meta.width); // width
  u32(compressed ? base.slicePitch : base.rowPitch); // pitchOrLinearSize
  u32(meta.dimension === TEX_DIMENSION.TEXTURE3D ? meta.depth : 0); // depth
  u32(meta.mipLevels); // mipMapCount
  for (let i = 0; i < 11; i++) u32(0); // reserved1[11]

  // DDS_PIXELFORMAT (32 bytes) — always DX10 marker
  u32(DDS_PIXELFORMAT_SIZE); // size
  u32(DDPF_FOURCC); // flags
  u32(makeFourCC("DX10")); // fourCC
  u32(0); // RGBBitCount
  u32(0); // RBitMask
  u32(0); // GBitMask
  u32(0); // BBitMask
  u32(0); // ABitMask

  let caps = DDSCAPS_TEXTURE;
  if (meta.mipLevels > 1) caps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP;
  u32(caps); // caps
  u32(0); // caps2
  u32(0); // caps3
  u32(0); // caps4
  u32(0); // reserved2

  // --- DDS_HEADER_DXT10 (20 bytes) ---
  u32(meta.format); // dxgiFormat
  u32(meta.dimension); // resourceDimension (2/3/4 matches TEX_DIMENSION)
  u32(meta.miscFlags); // miscFlag (TEXTURECUBE)
  u32(meta.arraySize); // arraySize
  u32(opts.alphaMode ?? DDS_ALPHA_MODE.UNKNOWN); // miscFlags2

  // --- pixel data ---
  for (const img of scratch.images) {
    bytes.set(img.pixels.subarray(0, img.slicePitch), o);
    o += img.slicePitch;
  }

  return bytes;
}

// Legacy FourCC -> DXGI_FORMAT for the common reader path.
const FOURCC_TO_DXGI: Record<number, DXGI_FORMAT> = {
  [makeFourCC("DXT1")]: DXGI_FORMAT.BC1_UNORM,
  [makeFourCC("DXT3")]: DXGI_FORMAT.BC2_UNORM,
  [makeFourCC("DXT5")]: DXGI_FORMAT.BC3_UNORM,
  [makeFourCC("BC4U")]: DXGI_FORMAT.BC4_UNORM,
  [makeFourCC("BC4S")]: DXGI_FORMAT.BC4_SNORM,
  [makeFourCC("ATI1")]: DXGI_FORMAT.BC4_UNORM,
  [makeFourCC("BC5U")]: DXGI_FORMAT.BC5_UNORM,
  [makeFourCC("BC5S")]: DXGI_FORMAT.BC5_SNORM,
  [makeFourCC("ATI2")]: DXGI_FORMAT.BC5_UNORM,
};

/**
 * Parse a DDS buffer into a ScratchImage. Supports DX10-extended headers and the
 * common legacy DXTn/BCn FourCC tags. Legacy uncompressed bitmask layouts are not
 * yet decoded (throws) — those will be added with the broader reader in M5.
 */
export function readDDS(data: Uint8Array): ScratchImage {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  if (data.byteLength < 4 + DDS_HEADER_SIZE) throw new Error("DDS: too small");
  if (view.getUint32(0, true) !== DDS_MAGIC) throw new Error("DDS: bad magic");
  if (view.getUint32(4, true) !== DDS_HEADER_SIZE) throw new Error("DDS: bad header size");

  const height = view.getUint32(12, true);
  const width = view.getUint32(16, true);
  let mipLevels = view.getUint32(28, true) || 1;

  // DDS_PIXELFORMAT begins at offset 4 + 72 = 76
  const pfFlags = view.getUint32(80, true);
  const pfFourCC = view.getUint32(84, true);

  let offset = 4 + DDS_HEADER_SIZE;
  let format = DXGI_FORMAT.UNKNOWN;
  let arraySize = 1;
  let miscFlags = 0;
  let miscFlags2 = 0;
  let dimension = TEX_DIMENSION.TEXTURE2D;

  if (pfFlags & DDPF_FOURCC && pfFourCC === makeFourCC("DX10")) {
    if (data.byteLength < offset + DDS_DX10_EXT_SIZE) throw new Error("DDS: truncated DX10 header");
    format = view.getUint32(offset, true);
    dimension = view.getUint32(offset + 4, true) as TEX_DIMENSION;
    miscFlags = view.getUint32(offset + 8, true);
    arraySize = view.getUint32(offset + 12, true) || 1;
    miscFlags2 = view.getUint32(offset + 16, true);
    offset += DDS_DX10_EXT_SIZE;
  } else if (pfFlags & DDPF_FOURCC) {
    format = FOURCC_TO_DXGI[pfFourCC] ?? DXGI_FORMAT.UNKNOWN;
    if (format === DXGI_FORMAT.UNKNOWN) {
      throw new Error(`DDS: unsupported FourCC 0x${pfFourCC.toString(16)}`);
    }
  } else {
    throw new Error("DDS: legacy uncompressed bitmask layouts not yet supported");
  }

  if (format === DXGI_FORMAT.UNKNOWN) throw new Error("DDS: unknown format");
  if (mipLevels < 1) mipLevels = 1;

  // The reader only materializes the slice-0 2D mip chain. Reject array,
  // cubemap, and volume textures rather than returning metadata (arraySize /
  // dimension) that disagrees with the images actually read.
  if (arraySize > 1) {
    throw new Error(`DDS: array textures not yet supported (arraySize=${arraySize})`);
  }
  if (dimension === TEX_DIMENSION.TEXTURE3D) {
    throw new Error("DDS: 3D/volume textures not yet supported");
  }

  const metadata: TexMetadata = {
    width,
    height,
    depth: 1,
    arraySize,
    mipLevels,
    miscFlags,
    miscFlags2,
    format,
    dimension,
  };

  // Read mip chain for array slice 0 only (MVP).
  const images: Image[] = [];
  let w = width;
  let h = height;
  for (let m = 0; m < mipLevels; m++) {
    const { rowPitch, slicePitch } = computePitch(format, w, h);
    if (offset + slicePitch > data.byteLength) throw new Error("DDS: truncated pixel data");
    images.push({
      width: w,
      height: h,
      format,
      rowPitch,
      slicePitch,
      pixels: data.subarray(offset, offset + slicePitch),
    });
    offset += slicePitch;
    w = Math.max(1, w >> 1);
    h = Math.max(1, h >> 1);
  }

  return ScratchImage.fromImages(metadata, images);
}
