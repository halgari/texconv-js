//-------------------------------------------------------------------------------------
// image.ts
//
// Image / ScratchImage types, mirroring DirectXTex's Image and TexMetadata.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT, computePitch, type CP_FLAGS } from "./dxgi.js";

/** A single 2D subresource (one mip level / array slice). */
export interface Image {
  width: number;
  height: number;
  format: DXGI_FORMAT;
  rowPitch: number;
  slicePitch: number;
  /** Pixel data for this subresource. */
  pixels: Uint8Array;
}

export enum TEX_DIMENSION {
  TEXTURE1D = 2,
  TEXTURE2D = 3,
  TEXTURE3D = 4,
}

/** Describes the shape of a texture: mips, array, dimension. */
export interface TexMetadata {
  width: number;
  height: number;
  depth: number;
  arraySize: number;
  mipLevels: number;
  miscFlags: number;
  miscFlags2: number;
  format: DXGI_FORMAT;
  dimension: TEX_DIMENSION;
}

/** TEX_MISC_FLAG (subset). */
export const TEX_MISC_TEXTURECUBE = 0x4;

/**
 * Allocate an Image of the given size/format with a freshly zeroed pixel buffer.
 */
export function makeImage(
  format: DXGI_FORMAT,
  width: number,
  height: number,
  flags?: CP_FLAGS,
): Image {
  const { rowPitch, slicePitch } = computePitch(format, width, height, flags);
  return {
    width,
    height,
    format,
    rowPitch,
    slicePitch,
    pixels: new Uint8Array(slicePitch),
  };
}

/** Number of mip levels for a full chain down to 1x1. */
export function countMips(width: number, height: number): number {
  let mips = 1;
  let w = width;
  let h = height;
  while (w > 1 || h > 1) {
    w = Math.max(1, w >> 1);
    h = Math.max(1, h >> 1);
    mips++;
  }
  return mips;
}

/**
 * A container holding one or more images plus shape metadata. The 2D, single-array
 * case (the common texconv path) is what the MVP exercises; the structure leaves
 * room for arrays/volumes/cubemaps.
 */
export class ScratchImage {
  metadata: TexMetadata;
  images: Image[];

  private constructor(metadata: TexMetadata, images: Image[]) {
    this.metadata = metadata;
    this.images = images;
  }

  /** Single 2D image, mip 0 only. */
  static from2D(image: Image): ScratchImage {
    return new ScratchImage(
      {
        width: image.width,
        height: image.height,
        depth: 1,
        arraySize: 1,
        mipLevels: 1,
        miscFlags: 0,
        miscFlags2: 0,
        format: image.format,
        dimension: TEX_DIMENSION.TEXTURE2D,
      },
      [image],
    );
  }

  /** A 2D mip chain (array size 1). `images[0]` is the base level. */
  static fromMipChain(images: Image[]): ScratchImage {
    const base = images[0];
    return new ScratchImage(
      {
        width: base.width,
        height: base.height,
        depth: 1,
        arraySize: 1,
        mipLevels: images.length,
        miscFlags: 0,
        miscFlags2: 0,
        format: base.format,
        dimension: TEX_DIMENSION.TEXTURE2D,
      },
      images,
    );
  }

  /** Construct directly from metadata + image list (e.g. when parsing DDS). */
  static fromImages(metadata: TexMetadata, images: Image[]): ScratchImage {
    return new ScratchImage(metadata, images);
  }

  get baseImage(): Image {
    return this.images[0];
  }
}
