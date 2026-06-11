//-------------------------------------------------------------------------------------
// bc7_decode.ts
//
// A reference BC7 decoder (CPU) used for verification and for on-screen preview of
// compressed output. Implements the BPTC UNORM decode per the D3D11 spec.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT } from "./dxgi.js";
import { makeImage, type Image } from "./image.js";

// 2-subset partition table: bit i = subset (0/1) of pixel i.
const PARTITION2 = [
  0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80,
  0xC800, 0xFFEC, 0xFE80, 0xE800, 0xFFE8, 0xFF00, 0xFFF0, 0xF000,
  0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310, 0x3100, 0x8CCE,
  0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C,
  0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a,
  0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
  0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c,
  0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
];

// 3-subset partition table: 2 bits per pixel = subset (0/1/2).
const PARTITION3 = [
  0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050,
  0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090, 0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250,
  0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
  0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200,
  0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424, 0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
  0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
  0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414, 0x96960000,
  0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000, 0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
];

// Anchor (fix-up) index for subset 1 in 2-subset partitions.
const ANCHOR2 = [
  15,15,15,15,15,15,15,15, 15,15,15,15,15,15,15,15,
  15, 2, 8, 2, 2, 8, 8,15, 2, 8, 2, 2, 8, 8, 2, 2,
  15,15, 6, 8, 2, 8,15,15, 2, 8, 2, 2, 2,15,15, 6,
   6, 2, 6, 8,15,15, 2, 2, 15,15,15,15,15, 2, 2,15,
];
// Anchors for subsets 1 and 2 in 3-subset partitions.
const ANCHOR3A = [
   3, 3,15,15, 8, 3,15,15, 8, 8, 6, 6, 6, 5, 3, 3,
   3, 3, 8,15, 3, 3, 6,10, 5, 8, 8, 6, 8, 5,15,15,
   8,15, 3, 5, 6,10, 8,15, 15, 3,15, 5,15,15,15,15,
   3,15, 5, 5, 5, 8, 5,10, 5,10, 8,13,15,12, 3, 3,
];
const ANCHOR3B = [
  15, 8, 8, 3,15,15, 3, 8,15,15,15,15,15,15,15, 8,
  15, 8,15, 3,15, 8,15, 8, 3,15, 6,10,15,15,10, 8,
  15, 3,15,10,10, 8, 9,10, 6,15, 8,15, 3, 6, 6, 8,
  15, 3,15,15,15,15,15,15,15,15,15,15, 3,15,15, 8,
];

const WEIGHTS2 = [0, 21, 43, 64];
const WEIGHTS3 = [0, 9, 18, 27, 37, 46, 55, 64];
const WEIGHTS4 = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

interface ModeInfo {
  ns: number; // subsets
  pb: number; // partition bits
  rb: number; // rotation bits
  isb: number; // index selector bits
  cb: number; // color bits
  ab: number; // alpha bits
  epb: number; // per-endpoint p-bits (0/1)
  spb: number; // shared p-bits (0/1)
  ib: number; // index bits
  ib2: number; // secondary index bits
}

const MODES: ModeInfo[] = [
  { ns: 3, pb: 4, rb: 0, isb: 0, cb: 4, ab: 0, epb: 1, spb: 0, ib: 3, ib2: 0 },
  { ns: 2, pb: 6, rb: 0, isb: 0, cb: 6, ab: 0, epb: 0, spb: 1, ib: 3, ib2: 0 },
  { ns: 3, pb: 6, rb: 0, isb: 0, cb: 5, ab: 0, epb: 0, spb: 0, ib: 2, ib2: 0 },
  { ns: 2, pb: 6, rb: 0, isb: 0, cb: 7, ab: 0, epb: 1, spb: 0, ib: 2, ib2: 0 },
  { ns: 1, pb: 0, rb: 2, isb: 1, cb: 5, ab: 6, epb: 0, spb: 0, ib: 2, ib2: 3 },
  { ns: 1, pb: 0, rb: 2, isb: 0, cb: 7, ab: 8, epb: 0, spb: 0, ib: 2, ib2: 2 },
  { ns: 1, pb: 0, rb: 0, isb: 0, cb: 7, ab: 7, epb: 1, spb: 0, ib: 4, ib2: 0 },
  { ns: 2, pb: 6, rb: 0, isb: 0, cb: 5, ab: 5, epb: 1, spb: 0, ib: 2, ib2: 0 },
];

class BitReader {
  private bit = 0;
  constructor(private readonly bytes: Uint8Array, private readonly base: number) {}
  read(n: number): number {
    let v = 0;
    for (let i = 0; i < n; i++) {
      const byte = this.bytes[this.base + (this.bit >> 3)];
      const b = (byte >> (this.bit & 7)) & 1;
      v |= b << i;
      this.bit++;
    }
    return v >>> 0;
  }
}

function subsetOf(mode: ModeInfo, partition: number, pixel: number): number {
  if (mode.ns === 1) return 0;
  if (mode.ns === 2) return (PARTITION2[partition] >> pixel) & 1;
  return (PARTITION3[partition] >>> (pixel * 2)) & 3;
}

function isAnchor(mode: ModeInfo, partition: number, pixel: number): boolean {
  if (pixel === 0) return true;
  if (mode.ns === 2) return pixel === ANCHOR2[partition];
  if (mode.ns === 3) return pixel === ANCHOR3A[partition] || pixel === ANCHOR3B[partition];
  return false;
}

function expand(value: number, bits: number): number {
  const v = (value << (8 - bits)) & 0xff;
  return (v | (v >> bits)) & 0xff;
}

/** Decode one 16-byte BC7 block at `offset` into 16 RGBA pixels (row-major). */
export function decodeBC7Block(bytes: Uint8Array, offset: number, out: Uint8Array): void {
  // Determine mode = index of first set bit (LSB first).
  let mode = 0;
  {
    let b = 0;
    while (mode < 8) {
      const byte = bytes[offset + (b >> 3)];
      if ((byte >> (b & 7)) & 1) break;
      mode++;
      b++;
    }
  }
  if (mode >= 8) {
    // Reserved/invalid mode -> output zero.
    for (let i = 0; i < 64; i++) out[i] = 0;
    return;
  }
  const m = MODES[mode];
  const r = new BitReader(bytes, offset);
  r.read(mode + 1); // consume mode bits

  const partition = m.pb ? r.read(m.pb) : 0;
  const rotation = m.rb ? r.read(m.rb) : 0;
  const indexSelector = m.isb ? r.read(m.isb) : 0;

  const numEndpoints = m.ns * 2;
  const rC = new Array<number>(numEndpoints);
  const gC = new Array<number>(numEndpoints);
  const bC = new Array<number>(numEndpoints);
  const aC = new Array<number>(numEndpoints);
  for (let i = 0; i < numEndpoints; i++) rC[i] = r.read(m.cb);
  for (let i = 0; i < numEndpoints; i++) gC[i] = r.read(m.cb);
  for (let i = 0; i < numEndpoints; i++) bC[i] = r.read(m.cb);
  if (m.ab) for (let i = 0; i < numEndpoints; i++) aC[i] = r.read(m.ab);
  else for (let i = 0; i < numEndpoints; i++) aC[i] = 255;

  // p-bits
  const pbits = new Array<number>(numEndpoints).fill(0);
  if (m.epb) {
    for (let i = 0; i < numEndpoints; i++) pbits[i] = r.read(1);
  } else if (m.spb) {
    const sp: number[] = [];
    for (let s = 0; s < m.ns; s++) sp.push(r.read(1));
    for (let i = 0; i < numEndpoints; i++) pbits[i] = sp[i >> 1];
  }

  // Assemble full 8-bit endpoints.
  const ep: { r: number; g: number; b: number; a: number }[] = [];
  for (let i = 0; i < numEndpoints; i++) {
    let cb = m.cb;
    let ab = m.ab || 0;
    let rr = rC[i], gg = gC[i], bb = bC[i], aa = aC[i];
    if (m.epb || m.spb) {
      rr = (rr << 1) | pbits[i];
      gg = (gg << 1) | pbits[i];
      bb = (bb << 1) | pbits[i];
      if (m.ab) aa = (aa << 1) | pbits[i];
      cb += 1;
      if (m.ab) ab += 1;
    }
    ep.push({
      r: expand(rr, cb),
      g: expand(gg, cb),
      b: expand(bb, cb),
      a: m.ab ? expand(aa, ab) : 255,
    });
  }

  // Read indices. Primary first (per pixel, anchors carry one fewer bit), then
  // secondary (modes 4/5 only).
  const primary = new Array<number>(16);
  for (let p = 0; p < 16; p++) {
    const bits = m.ib - (isAnchor(m, partition, p) ? 1 : 0);
    primary[p] = bits > 0 ? r.read(bits) : 0;
  }
  const secondary = new Array<number>(16).fill(0);
  if (m.ib2) {
    for (let p = 0; p < 16; p++) {
      const bits = m.ib2 - (p === 0 ? 1 : 0);
      secondary[p] = bits > 0 ? r.read(bits) : 0;
    }
  }

  const wPrimary = m.ib === 2 ? WEIGHTS2 : m.ib === 3 ? WEIGHTS3 : WEIGHTS4;
  const wSecondary = m.ib2 === 2 ? WEIGHTS2 : m.ib2 === 3 ? WEIGHTS3 : WEIGHTS4;

  for (let p = 0; p < 16; p++) {
    const s = subsetOf(m, partition, p);
    const e0 = ep[s * 2];
    const e1 = ep[s * 2 + 1];

    let colorIdx: number;
    let alphaIdx: number;
    let cw: number[];
    let aw: number[];
    if (m.ib2) {
      // Modes 4/5: separate color and alpha index sets, optionally swapped.
      if (indexSelector) {
        colorIdx = secondary[p];
        alphaIdx = primary[p];
        cw = wSecondary;
        aw = wPrimary;
      } else {
        colorIdx = primary[p];
        alphaIdx = secondary[p];
        cw = wPrimary;
        aw = wSecondary;
      }
    } else {
      colorIdx = primary[p];
      alphaIdx = primary[p];
      cw = wPrimary;
      aw = wPrimary;
    }

    const wc = cw[colorIdx];
    const wa = aw[alphaIdx];
    let R = (e0.r * (64 - wc) + e1.r * wc + 32) >> 6;
    let G = (e0.g * (64 - wc) + e1.g * wc + 32) >> 6;
    let B = (e0.b * (64 - wc) + e1.b * wc + 32) >> 6;
    let A = (e0.a * (64 - wa) + e1.a * wa + 32) >> 6;

    // Rotation (modes 4/5): swap alpha with the selected channel.
    if (rotation === 1) { const t = A; A = R; R = t; }
    else if (rotation === 2) { const t = A; A = G; G = t; }
    else if (rotation === 3) { const t = A; A = B; B = t; }

    const o = p * 4;
    out[o] = R; out[o + 1] = G; out[o + 2] = B; out[o + 3] = A;
  }
}

/** Decode a full BC7 image to an R8G8B8A8_UNORM Image. */
export function decodeBC7(src: Image): Image {
  const { width, height } = src;
  const dst = makeImage(DXGI_FORMAT.R8G8B8A8_UNORM, width, height);
  const xblocks = Math.max(1, (width + 3) >> 2);
  const block = new Uint8Array(64);
  for (let by = 0; by * 4 < height; by++) {
    for (let bx = 0; bx < xblocks; bx++) {
      const off = (by * xblocks + bx) * 16;
      decodeBC7Block(src.pixels, off, block);
      for (let py = 0; py < 4; py++) {
        const y = by * 4 + py;
        if (y >= height) break;
        for (let px = 0; px < 4; px++) {
          const x = bx * 4 + px;
          if (x >= width) break;
          const d = y * dst.rowPitch + x * 4;
          const s = (py * 4 + px) * 4;
          dst.pixels[d] = block[s];
          dst.pixels[d + 1] = block[s + 1];
          dst.pixels[d + 2] = block[s + 2];
          dst.pixels[d + 3] = block[s + 3];
        }
      }
    }
  }
  return dst;
}
