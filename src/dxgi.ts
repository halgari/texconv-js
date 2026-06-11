//-------------------------------------------------------------------------------------
// dxgi.ts
//
// DXGI_FORMAT enumeration and format metadata, ported from DirectXTex
// (DirectXTexUtil.cpp). Only the subset of formats relevant to the texconv-js
// pipeline is included; more can be added incrementally.
//-------------------------------------------------------------------------------------

/** Standard DXGI_FORMAT numeric values (a focused subset). */
export enum DXGI_FORMAT {
  UNKNOWN = 0,
  R32G32B32A32_TYPELESS = 1,
  R32G32B32A32_FLOAT = 2,
  R16G16B16A16_FLOAT = 10,
  R16G16B16A16_UNORM = 11,
  R32_FLOAT = 41,
  R8G8B8A8_TYPELESS = 27,
  R8G8B8A8_UNORM = 28,
  R8G8B8A8_UNORM_SRGB = 29,
  R8G8_UNORM = 49,
  R16_UNORM = 56,
  R8_UNORM = 61,
  BC1_TYPELESS = 70,
  BC1_UNORM = 71,
  BC1_UNORM_SRGB = 72,
  BC2_TYPELESS = 73,
  BC2_UNORM = 74,
  BC2_UNORM_SRGB = 75,
  BC3_TYPELESS = 76,
  BC3_UNORM = 77,
  BC3_UNORM_SRGB = 78,
  BC4_TYPELESS = 79,
  BC4_UNORM = 80,
  BC4_SNORM = 81,
  BC5_TYPELESS = 82,
  BC5_UNORM = 83,
  BC5_SNORM = 84,
  B8G8R8A8_UNORM = 87,
  B8G8R8X8_UNORM = 88,
  B8G8R8A8_UNORM_SRGB = 91,
  BC6H_TYPELESS = 94,
  BC6H_UF16 = 95,
  BC6H_SF16 = 96,
  BC7_TYPELESS = 97,
  BC7_UNORM = 98,
  BC7_UNORM_SRGB = 99,
}

/** Flags mirroring DirectXTex CP_FLAGS (only the ones we use). */
export const enum CP_FLAGS {
  NONE = 0,
  /** Legacy DXTn nonsquare tail handling (DDS_FLAGS_BAD_DXTN_TAILS). */
  BAD_DXTN_TAILS = 0x10,
}

/** True for the BCn block-compressed formats. */
export function isCompressed(fmt: DXGI_FORMAT): boolean {
  switch (fmt) {
    case DXGI_FORMAT.BC1_TYPELESS:
    case DXGI_FORMAT.BC1_UNORM:
    case DXGI_FORMAT.BC1_UNORM_SRGB:
    case DXGI_FORMAT.BC2_TYPELESS:
    case DXGI_FORMAT.BC2_UNORM:
    case DXGI_FORMAT.BC2_UNORM_SRGB:
    case DXGI_FORMAT.BC3_TYPELESS:
    case DXGI_FORMAT.BC3_UNORM:
    case DXGI_FORMAT.BC3_UNORM_SRGB:
    case DXGI_FORMAT.BC4_TYPELESS:
    case DXGI_FORMAT.BC4_UNORM:
    case DXGI_FORMAT.BC4_SNORM:
    case DXGI_FORMAT.BC5_TYPELESS:
    case DXGI_FORMAT.BC5_UNORM:
    case DXGI_FORMAT.BC5_SNORM:
    case DXGI_FORMAT.BC6H_TYPELESS:
    case DXGI_FORMAT.BC6H_UF16:
    case DXGI_FORMAT.BC6H_SF16:
    case DXGI_FORMAT.BC7_TYPELESS:
    case DXGI_FORMAT.BC7_UNORM:
    case DXGI_FORMAT.BC7_UNORM_SRGB:
      return true;
    default:
      return false;
  }
}

/** Bytes per 4x4 block for a compressed format (8 for BC1/BC4, else 16). */
export function blockBytes(fmt: DXGI_FORMAT): number {
  switch (fmt) {
    case DXGI_FORMAT.BC1_TYPELESS:
    case DXGI_FORMAT.BC1_UNORM:
    case DXGI_FORMAT.BC1_UNORM_SRGB:
    case DXGI_FORMAT.BC4_TYPELESS:
    case DXGI_FORMAT.BC4_UNORM:
    case DXGI_FORMAT.BC4_SNORM:
      return 8;
    default:
      return 16;
  }
}

/** True for the *_UNORM_SRGB formats. */
export function isSRGB(fmt: DXGI_FORMAT): boolean {
  switch (fmt) {
    case DXGI_FORMAT.R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT.BC1_UNORM_SRGB:
    case DXGI_FORMAT.BC2_UNORM_SRGB:
    case DXGI_FORMAT.BC3_UNORM_SRGB:
    case DXGI_FORMAT.BC7_UNORM_SRGB:
    case DXGI_FORMAT.B8G8R8A8_UNORM_SRGB:
      return true;
    default:
      return false;
  }
}

/** Bits per pixel for non-block formats (used for uncompressed pitch). */
export function bitsPerPixel(fmt: DXGI_FORMAT): number {
  switch (fmt) {
    case DXGI_FORMAT.R32G32B32A32_TYPELESS:
    case DXGI_FORMAT.R32G32B32A32_FLOAT:
      return 128;
    case DXGI_FORMAT.R16G16B16A16_FLOAT:
    case DXGI_FORMAT.R16G16B16A16_UNORM:
      return 64;
    case DXGI_FORMAT.R8G8B8A8_TYPELESS:
    case DXGI_FORMAT.R8G8B8A8_UNORM:
    case DXGI_FORMAT.R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT.B8G8R8A8_UNORM:
    case DXGI_FORMAT.B8G8R8X8_UNORM:
    case DXGI_FORMAT.B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT.R32_FLOAT:
      return 32;
    case DXGI_FORMAT.R8G8_UNORM:
    case DXGI_FORMAT.R16_UNORM:
      return 16;
    case DXGI_FORMAT.R8_UNORM:
      return 8;
    default:
      return 0;
  }
}

export interface Pitch {
  rowPitch: number;
  slicePitch: number;
}

/**
 * Compute the row and slice pitch for a surface, mirroring
 * DirectX::ComputePitch (the CP_FLAGS_PAGED4K/NV12 cases are omitted).
 */
export function computePitch(
  fmt: DXGI_FORMAT,
  width: number,
  height: number,
  flags: CP_FLAGS = CP_FLAGS.NONE,
): Pitch {
  if (fmt === DXGI_FORMAT.UNKNOWN) {
    throw new Error("computePitch: UNKNOWN format");
  }

  if (isCompressed(fmt)) {
    const bpb = blockBytes(fmt);
    let pitch: number;
    let slice: number;
    if (flags & CP_FLAGS.BAD_DXTN_TAILS) {
      const nbw = width >> 2;
      const nbh = height >> 2;
      pitch = Math.max(1, nbw * bpb);
      slice = Math.max(1, pitch * nbh);
    } else {
      const nbw = Math.max(1, (width + 3) >> 2);
      const nbh = Math.max(1, (height + 3) >> 2);
      pitch = nbw * bpb;
      slice = pitch * nbh;
    }
    return { rowPitch: pitch, slicePitch: slice };
  }

  const bpp = bitsPerPixel(fmt);
  if (bpp === 0) {
    throw new Error(`computePitch: unsupported format ${DXGI_FORMAT[fmt] ?? fmt}`);
  }
  // Default scanline packing (CP_FLAGS_NONE): byte-aligned.
  const rowPitch = Math.floor((width * bpp + 7) / 8);
  const slicePitch = rowPitch * height;
  return { rowPitch, slicePitch };
}
