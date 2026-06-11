//-------------------------------------------------------------------------------------
// bc6h_decode.ts
//
// A reference BC6H decoder (CPU) used for verification and for on-screen preview of
// compressed output. A direct port of DirectXTex's D3DX_BC6H::Decode (BC6HBC7.cpp),
// handling both BC6H_UF16 (unsigned) and BC6H_SF16 (signed) half-float blocks.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT } from "./dxgi.js";
import { makeImage, type Image } from "./image.js";

// 2-subset partition table (g_aPartitionTable[1]): bit i = region (0/1) of pixel i.
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

// Fix-up (anchor) index for region 1 in 2-region partitions (g_aFixUp[1][shape][1]).
const FIXUP2 = [
  15,15,15,15,15,15,15,15, 15,15,15,15,15,15,15,15,
  15, 2, 8, 2, 2, 8, 8,15, 2, 8, 2, 2, 8, 8, 2, 2,
  15,15, 6, 8, 2, 8,15,15, 2, 8, 2, 2, 2,15,15, 6,
   6, 2, 6, 8,15,15, 2, 2, 15,15,15,15,15, 2, 2,15,
];

// g_aWeights3 / g_aWeights4
const WEIGHTS3 = [0, 9, 18, 27, 37, 46, 55, 64];
const WEIGHTS4 = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

const BC67_WEIGHT_MAX = 64;
const BC67_WEIGHT_SHIFT = 6;
const BC67_WEIGHT_ROUND = 32;

// D3DX_BC6H::EField values used by the per-bit mode descriptors.
// W = endpoint 0.A, X = endpoint 0.B, Y = endpoint 1.A, Z = endpoint 1.B.
const NA = 0; // N/A
const M = 1; // Mode
const D = 2; // Shape
const RW = 3, RX = 4, RY = 5, RZ = 6;
const GW = 7, GX = 8, GY = 9, GZ = 10;
const BW = 11, BX = 12, BY = 13, BZ = 14;

// Endpoint slot (0 = e0.A, 1 = e0.B, 2 = e1.A, 3 = e1.B) and channel (0=r,1=g,2=b)
// for a color field f in [RW..BZ]: slot = (f - RW) & 3, channel = (f - RW) >> 2.

type ModeDescriptor = readonly [number, number]; // [EField, bit index]

interface ModeInfo {
  uMode: number;
  uPartitions: number;
  bTransformed: boolean;
  uIndexPrec: number;
  /** RGBPrec[region][A=0|B=1] = [rBits, gBits, bBits] (ms_aInfo RGBAPrec, alpha dropped). */
  prec: readonly (readonly (readonly [number, number, number])[])[];
}

// D3DX_BC6H::ms_aDesc - per-mode bit layout, 82 descriptors per mode.
const MODE_DESC: ReadonlyArray<ReadonlyArray<ModeDescriptor>> = [
  [ // Mode 1 (0x00) - 10 5 5 5
    [M, 0], [M, 1], [GY, 4], [BY, 4], [BZ, 4], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [RW, 6], [RW, 7], [RW, 8], [RW, 9], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GW, 6], [GW, 7], [GW, 8], [GW, 9], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [BW, 6], [BW, 7], [BW, 8], [BW, 9], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RX, 4],
    [GZ, 4], [GY, 0], [GY, 1], [GY, 2], [GY, 3], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GX, 4],
    [BZ, 0], [GZ, 0], [GZ, 1], [GZ, 2], [GZ, 3], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BX, 4],
    [BZ, 1], [BY, 0], [BY, 1], [BY, 2], [BY, 3], [RY, 0], [RY, 1], [RY, 2], [RY, 3], [RY, 4],
    [BZ, 2], [RZ, 0], [RZ, 1], [RZ, 2], [RZ, 3], [RZ, 4], [BZ, 3], [D, 0], [D, 1], [D, 2],
    [D, 3], [D, 4],
  ],
  [ // Mode 2 (0x01) - 7 6 6 6
    [M, 0], [M, 1], [GY, 5], [GZ, 4], [GZ, 5], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [RW, 6], [BZ, 0], [BZ, 1], [BY, 4], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GW, 6], [BY, 5], [BZ, 2], [GY, 4], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [BW, 6], [BZ, 3], [BZ, 5], [BZ, 4], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RX, 4],
    [RX, 5], [GY, 0], [GY, 1], [GY, 2], [GY, 3], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GX, 4],
    [GX, 5], [GZ, 0], [GZ, 1], [GZ, 2], [GZ, 3], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BX, 4],
    [BX, 5], [BY, 0], [BY, 1], [BY, 2], [BY, 3], [RY, 0], [RY, 1], [RY, 2], [RY, 3], [RY, 4],
    [RY, 5], [RZ, 0], [RZ, 1], [RZ, 2], [RZ, 3], [RZ, 4], [RZ, 5], [D, 0], [D, 1], [D, 2],
    [D, 3], [D, 4],
  ],
  [ // Mode 3 (0x02) - 11 5 4 4
    [M, 0], [M, 1], [M, 2], [M, 3], [M, 4], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [RW, 6], [RW, 7], [RW, 8], [RW, 9], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GW, 6], [GW, 7], [GW, 8], [GW, 9], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [BW, 6], [BW, 7], [BW, 8], [BW, 9], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RX, 4],
    [RW,10], [GY, 0], [GY, 1], [GY, 2], [GY, 3], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GW,10],
    [BZ, 0], [GZ, 0], [GZ, 1], [GZ, 2], [GZ, 3], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BW,10],
    [BZ, 1], [BY, 0], [BY, 1], [BY, 2], [BY, 3], [RY, 0], [RY, 1], [RY, 2], [RY, 3], [RY, 4],
    [BZ, 2], [RZ, 0], [RZ, 1], [RZ, 2], [RZ, 3], [RZ, 4], [BZ, 3], [D, 0], [D, 1], [D, 2],
    [D, 3], [D, 4],
  ],
  [ // Mode 4 (0x06) - 11 4 5 4
    [M, 0], [M, 1], [M, 2], [M, 3], [M, 4], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [RW, 6], [RW, 7], [RW, 8], [RW, 9], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GW, 6], [GW, 7], [GW, 8], [GW, 9], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [BW, 6], [BW, 7], [BW, 8], [BW, 9], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RW,10],
    [GZ, 4], [GY, 0], [GY, 1], [GY, 2], [GY, 3], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GX, 4],
    [GW,10], [GZ, 0], [GZ, 1], [GZ, 2], [GZ, 3], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BW,10],
    [BZ, 1], [BY, 0], [BY, 1], [BY, 2], [BY, 3], [RY, 0], [RY, 1], [RY, 2], [RY, 3], [BZ, 0],
    [BZ, 2], [RZ, 0], [RZ, 1], [RZ, 2], [RZ, 3], [GY, 4], [BZ, 3], [D, 0], [D, 1], [D, 2],
    [D, 3], [D, 4],
  ],
  [ // Mode 5 (0x0a) - 11 4 4 5
    [M, 0], [M, 1], [M, 2], [M, 3], [M, 4], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [RW, 6], [RW, 7], [RW, 8], [RW, 9], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GW, 6], [GW, 7], [GW, 8], [GW, 9], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [BW, 6], [BW, 7], [BW, 8], [BW, 9], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RW,10],
    [BY, 4], [GY, 0], [GY, 1], [GY, 2], [GY, 3], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GW,10],
    [BZ, 0], [GZ, 0], [GZ, 1], [GZ, 2], [GZ, 3], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BX, 4],
    [BW,10], [BY, 0], [BY, 1], [BY, 2], [BY, 3], [RY, 0], [RY, 1], [RY, 2], [RY, 3], [BZ, 1],
    [BZ, 2], [RZ, 0], [RZ, 1], [RZ, 2], [RZ, 3], [BZ, 4], [BZ, 3], [D, 0], [D, 1], [D, 2],
    [D, 3], [D, 4],
  ],
  [ // Mode 6 (0x0e) - 9 5 5 5
    [M, 0], [M, 1], [M, 2], [M, 3], [M, 4], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [RW, 6], [RW, 7], [RW, 8], [BY, 4], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GW, 6], [GW, 7], [GW, 8], [GY, 4], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [BW, 6], [BW, 7], [BW, 8], [BZ, 4], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RX, 4],
    [GZ, 4], [GY, 0], [GY, 1], [GY, 2], [GY, 3], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GX, 4],
    [BZ, 0], [GZ, 0], [GZ, 1], [GZ, 2], [GZ, 3], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BX, 4],
    [BZ, 1], [BY, 0], [BY, 1], [BY, 2], [BY, 3], [RY, 0], [RY, 1], [RY, 2], [RY, 3], [RY, 4],
    [BZ, 2], [RZ, 0], [RZ, 1], [RZ, 2], [RZ, 3], [RZ, 4], [BZ, 3], [D, 0], [D, 1], [D, 2],
    [D, 3], [D, 4],
  ],
  [ // Mode 7 (0x12) - 8 6 5 5
    [M, 0], [M, 1], [M, 2], [M, 3], [M, 4], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [RW, 6], [RW, 7], [GZ, 4], [BY, 4], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GW, 6], [GW, 7], [BZ, 2], [GY, 4], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [BW, 6], [BW, 7], [BZ, 3], [BZ, 4], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RX, 4],
    [RX, 5], [GY, 0], [GY, 1], [GY, 2], [GY, 3], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GX, 4],
    [BZ, 0], [GZ, 0], [GZ, 1], [GZ, 2], [GZ, 3], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BX, 4],
    [BZ, 1], [BY, 0], [BY, 1], [BY, 2], [BY, 3], [RY, 0], [RY, 1], [RY, 2], [RY, 3], [RY, 4],
    [RY, 5], [RZ, 0], [RZ, 1], [RZ, 2], [RZ, 3], [RZ, 4], [RZ, 5], [D, 0], [D, 1], [D, 2],
    [D, 3], [D, 4],
  ],
  [ // Mode 8 (0x16) - 8 5 6 5
    [M, 0], [M, 1], [M, 2], [M, 3], [M, 4], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [RW, 6], [RW, 7], [BZ, 0], [BY, 4], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GW, 6], [GW, 7], [GY, 5], [GY, 4], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [BW, 6], [BW, 7], [GZ, 5], [BZ, 4], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RX, 4],
    [GZ, 4], [GY, 0], [GY, 1], [GY, 2], [GY, 3], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GX, 4],
    [GX, 5], [GZ, 0], [GZ, 1], [GZ, 2], [GZ, 3], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BX, 4],
    [BZ, 1], [BY, 0], [BY, 1], [BY, 2], [BY, 3], [RY, 0], [RY, 1], [RY, 2], [RY, 3], [RY, 4],
    [BZ, 2], [RZ, 0], [RZ, 1], [RZ, 2], [RZ, 3], [RZ, 4], [BZ, 3], [D, 0], [D, 1], [D, 2],
    [D, 3], [D, 4],
  ],
  [ // Mode 9 (0x1a) - 8 5 5 6
    [M, 0], [M, 1], [M, 2], [M, 3], [M, 4], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [RW, 6], [RW, 7], [BZ, 1], [BY, 4], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GW, 6], [GW, 7], [BY, 5], [GY, 4], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [BW, 6], [BW, 7], [BZ, 5], [BZ, 4], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RX, 4],
    [GZ, 4], [GY, 0], [GY, 1], [GY, 2], [GY, 3], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GX, 4],
    [BZ, 0], [GZ, 0], [GZ, 1], [GZ, 2], [GZ, 3], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BX, 4],
    [BX, 5], [BY, 0], [BY, 1], [BY, 2], [BY, 3], [RY, 0], [RY, 1], [RY, 2], [RY, 3], [RY, 4],
    [BZ, 2], [RZ, 0], [RZ, 1], [RZ, 2], [RZ, 3], [RZ, 4], [BZ, 3], [D, 0], [D, 1], [D, 2],
    [D, 3], [D, 4],
  ],
  [ // Mode 10 (0x1e) - 6 6 6 6
    [M, 0], [M, 1], [M, 2], [M, 3], [M, 4], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [GZ, 4], [BZ, 0], [BZ, 1], [BY, 4], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GY, 5], [BY, 5], [BZ, 2], [GY, 4], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [GZ, 5], [BZ, 3], [BZ, 5], [BZ, 4], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RX, 4],
    [RX, 5], [GY, 0], [GY, 1], [GY, 2], [GY, 3], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GX, 4],
    [GX, 5], [GZ, 0], [GZ, 1], [GZ, 2], [GZ, 3], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BX, 4],
    [BX, 5], [BY, 0], [BY, 1], [BY, 2], [BY, 3], [RY, 0], [RY, 1], [RY, 2], [RY, 3], [RY, 4],
    [RY, 5], [RZ, 0], [RZ, 1], [RZ, 2], [RZ, 3], [RZ, 4], [RZ, 5], [D, 0], [D, 1], [D, 2],
    [D, 3], [D, 4],
  ],
  [ // Mode 11 (0x03) - 10 10
    [M, 0], [M, 1], [M, 2], [M, 3], [M, 4], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [RW, 6], [RW, 7], [RW, 8], [RW, 9], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GW, 6], [GW, 7], [GW, 8], [GW, 9], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [BW, 6], [BW, 7], [BW, 8], [BW, 9], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RX, 4],
    [RX, 5], [RX, 6], [RX, 7], [RX, 8], [RX, 9], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GX, 4],
    [GX, 5], [GX, 6], [GX, 7], [GX, 8], [GX, 9], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BX, 4],
    [BX, 5], [BX, 6], [BX, 7], [BX, 8], [BX, 9], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0],
    [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0],
    [NA, 0], [NA, 0],
  ],
  [ // Mode 12 (0x07) - 11 9
    [M, 0], [M, 1], [M, 2], [M, 3], [M, 4], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [RW, 6], [RW, 7], [RW, 8], [RW, 9], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GW, 6], [GW, 7], [GW, 8], [GW, 9], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [BW, 6], [BW, 7], [BW, 8], [BW, 9], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RX, 4],
    [RX, 5], [RX, 6], [RX, 7], [RX, 8], [RW,10], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GX, 4],
    [GX, 5], [GX, 6], [GX, 7], [GX, 8], [GW,10], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BX, 4],
    [BX, 5], [BX, 6], [BX, 7], [BX, 8], [BW,10], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0],
    [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0],
    [NA, 0], [NA, 0],
  ],
  [ // Mode 13 (0x0b) - 12 8
    [M, 0], [M, 1], [M, 2], [M, 3], [M, 4], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [RW, 6], [RW, 7], [RW, 8], [RW, 9], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GW, 6], [GW, 7], [GW, 8], [GW, 9], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [BW, 6], [BW, 7], [BW, 8], [BW, 9], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RX, 4],
    [RX, 5], [RX, 6], [RX, 7], [RW,11], [RW,10], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GX, 4],
    [GX, 5], [GX, 6], [GX, 7], [GW,11], [GW,10], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BX, 4],
    [BX, 5], [BX, 6], [BX, 7], [BW,11], [BW,10], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0],
    [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0],
    [NA, 0], [NA, 0],
  ],
  [ // Mode 14 (0x0f) - 16 4
    [M, 0], [M, 1], [M, 2], [M, 3], [M, 4], [RW, 0], [RW, 1], [RW, 2], [RW, 3], [RW, 4],
    [RW, 5], [RW, 6], [RW, 7], [RW, 8], [RW, 9], [GW, 0], [GW, 1], [GW, 2], [GW, 3], [GW, 4],
    [GW, 5], [GW, 6], [GW, 7], [GW, 8], [GW, 9], [BW, 0], [BW, 1], [BW, 2], [BW, 3], [BW, 4],
    [BW, 5], [BW, 6], [BW, 7], [BW, 8], [BW, 9], [RX, 0], [RX, 1], [RX, 2], [RX, 3], [RW,15],
    [RW,14], [RW,13], [RW,12], [RW,11], [RW,10], [GX, 0], [GX, 1], [GX, 2], [GX, 3], [GW,15],
    [GW,14], [GW,13], [GW,12], [GW,11], [GW,10], [BX, 0], [BX, 1], [BX, 2], [BX, 3], [BW,15],
    [BW,14], [BW,13], [BW,12], [BW,11], [BW,10], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0],
    [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0], [NA, 0],
    [NA, 0], [NA, 0],
  ],
];

// D3DX_BC6H::ms_aInfo - Mode, Partitions, Transformed, IndexPrec, RGBPrec
const MODE_INFO: ReadonlyArray<ModeInfo> = [
  { uMode: 0x00, uPartitions: 1, bTransformed: true,  uIndexPrec: 3, prec: [[[10,10,10],[5,5,5]], [[5,5,5],[5,5,5]]] }, // Mode 1
  { uMode: 0x01, uPartitions: 1, bTransformed: true,  uIndexPrec: 3, prec: [[[7,7,7],[6,6,6]], [[6,6,6],[6,6,6]]] }, // Mode 2
  { uMode: 0x02, uPartitions: 1, bTransformed: true,  uIndexPrec: 3, prec: [[[11,11,11],[5,4,4]], [[5,4,4],[5,4,4]]] }, // Mode 3
  { uMode: 0x06, uPartitions: 1, bTransformed: true,  uIndexPrec: 3, prec: [[[11,11,11],[4,5,4]], [[4,5,4],[4,5,4]]] }, // Mode 4
  { uMode: 0x0a, uPartitions: 1, bTransformed: true,  uIndexPrec: 3, prec: [[[11,11,11],[4,4,5]], [[4,4,5],[4,4,5]]] }, // Mode 5
  { uMode: 0x0e, uPartitions: 1, bTransformed: true,  uIndexPrec: 3, prec: [[[9,9,9],[5,5,5]], [[5,5,5],[5,5,5]]] }, // Mode 6
  { uMode: 0x12, uPartitions: 1, bTransformed: true,  uIndexPrec: 3, prec: [[[8,8,8],[6,5,5]], [[6,5,5],[6,5,5]]] }, // Mode 7
  { uMode: 0x16, uPartitions: 1, bTransformed: true,  uIndexPrec: 3, prec: [[[8,8,8],[5,6,5]], [[5,6,5],[5,6,5]]] }, // Mode 8
  { uMode: 0x1a, uPartitions: 1, bTransformed: true,  uIndexPrec: 3, prec: [[[8,8,8],[5,5,6]], [[5,5,6],[5,5,6]]] }, // Mode 9
  { uMode: 0x1e, uPartitions: 1, bTransformed: false, uIndexPrec: 3, prec: [[[6,6,6],[6,6,6]], [[6,6,6],[6,6,6]]] }, // Mode 10
  { uMode: 0x03, uPartitions: 0, bTransformed: false, uIndexPrec: 4, prec: [[[10,10,10],[10,10,10]], [[0,0,0],[0,0,0]]] }, // Mode 11
  { uMode: 0x07, uPartitions: 0, bTransformed: true,  uIndexPrec: 4, prec: [[[11,11,11],[9,9,9]], [[0,0,0],[0,0,0]]] }, // Mode 12
  { uMode: 0x0b, uPartitions: 0, bTransformed: true,  uIndexPrec: 4, prec: [[[12,12,12],[8,8,8]], [[0,0,0],[0,0,0]]] }, // Mode 13
  { uMode: 0x0f, uPartitions: 0, bTransformed: true,  uIndexPrec: 4, prec: [[[16,16,16],[4,4,4]], [[0,0,0],[0,0,0]]] }, // Mode 14
];

// D3DX_BC6H::ms_aModeToInfo - maps the 5-bit mode field to a MODE_INFO index (-1 = invalid).
const MODE_TO_INFO = [
  0,   // Mode 1   - 0x00
  1,   // Mode 2   - 0x01
  2,   // Mode 3   - 0x02
  10,  // Mode 11  - 0x03
  -1,  // Invalid  - 0x04
  -1,  // Invalid  - 0x05
  3,   // Mode 4   - 0x06
  11,  // Mode 12  - 0x07
  -1,  // Invalid  - 0x08
  -1,  // Invalid  - 0x09
  4,   // Mode 5   - 0x0a
  12,  // Mode 13  - 0x0b
  -1,  // Invalid  - 0x0c
  -1,  // Invalid  - 0x0d
  5,   // Mode 6   - 0x0e
  13,  // Mode 14  - 0x0f
  -1,  // Invalid  - 0x10
  -1,  // Invalid  - 0x11
  6,   // Mode 7   - 0x12
  -1,  // Reserved - 0x13
  -1,  // Invalid  - 0x14
  -1,  // Invalid  - 0x15
  7,   // Mode 8   - 0x16
  -1,  // Reserved - 0x17
  -1,  // Invalid  - 0x18
  -1,  // Invalid  - 0x19
  8,   // Mode 9   - 0x1a
  -1,  // Reserved - 0x1b
  -1,  // Invalid  - 0x1c
  -1,  // Invalid  - 0x1d
  9,   // Mode 10  - 0x1e
  -1,  // Reserved - 0x1f
];

class BitReader {
  bit = 0;
  constructor(private readonly bytes: Uint8Array, private readonly base: number) {}
  readBit(): number {
    const byte = this.bytes[this.base + (this.bit >> 3)];
    const b = (byte >> (this.bit & 7)) & 1;
    this.bit++;
    return b;
  }
  read(n: number): number {
    let v = 0;
    for (let i = 0; i < n; i++) v |= this.readBit() << i;
    return v >>> 0;
  }
}

// SIGN_EXTEND(x, nb) from BC6HBC7.cpp.
function signExtend(x: number, nb: number): number {
  return ((x & (1 << (nb - 1))) !== 0 ? ~0 ^ ((1 << nb) - 1) : 0) | x;
}

// D3DX_BC6H::Unquantize
function unquantize(comp: number, bitsPerComp: number, bSigned: boolean): number {
  let unq = 0;
  if (bSigned) {
    if (bitsPerComp >= 16) {
      unq = comp;
    } else {
      let s = 0;
      if (comp < 0) {
        s = 1;
        comp = -comp;
      }
      if (comp === 0) unq = 0;
      else if (comp >= (1 << (bitsPerComp - 1)) - 1) unq = 0x7FFF;
      else unq = ((comp << 15) + 0x4000) >> (bitsPerComp - 1);
      if (s) unq = -unq;
    }
  } else {
    if (bitsPerComp >= 15) unq = comp;
    else if (comp === 0) unq = 0;
    else if (comp === (1 << bitsPerComp) - 1) unq = 0xFFFF;
    else unq = ((comp << 16) + 0x8000) >> bitsPerComp;
  }
  return unq;
}

// D3DX_BC6H::FinishUnquantize
function finishUnquantize(comp: number, bSigned: boolean): number {
  if (bSigned) {
    return comp < 0 ? -((-comp * 31) >> 5) : (comp * 31) >> 5; // scale the magnitude by 31/32
  }
  return (comp * 31) >> 6; // scale the magnitude by 31/64
}

// INTColor::INT2F16 - convert an unquantized component to a half-float bit pattern.
function int2F16(input: number, bSigned: boolean): number {
  if (bSigned) {
    let s = 0;
    if (input < 0) {
      s = 0x8000;
      input = -input;
    }
    return (s | input) & 0xFFFF;
  }
  return input & 0xFFFF;
}

/** Convert a half-float bit pattern to a 32-bit float (XMConvertHalfToFloat). */
export function halfToFloat(bits: number): number {
  const s = (bits & 0x8000) !== 0 ? -1 : 1;
  const e = (bits >> 10) & 0x1f;
  const m = bits & 0x3ff;
  if (e === 0) return s * m * 2 ** -24; // zero / denormal
  if (e === 31) return m !== 0 ? NaN : s * Infinity;
  return s * (1024 + m) * 2 ** (e - 25); // s * (1 + m/1024) * 2^(e-15)
}

// IsFixUpOffset for BC6H (uPartitions is 0 or 1; region-0 fixup is always pixel 0).
function isFixUpOffset(uPartitions: number, uShape: number, uOffset: number): boolean {
  if (uOffset === 0) return true;
  return uPartitions === 1 && uOffset === FIXUP2[uShape];
}

function fillBlack(out: Float32Array): void {
  for (let i = 0; i < 16; i++) {
    out[i * 4] = 0;
    out[i * 4 + 1] = 0;
    out[i * 4 + 2] = 0;
    out[i * 4 + 3] = 1;
  }
}

/**
 * Decode one 16-byte BC6H block at `offset` into 16 RGBA float pixels (row-major).
 * Port of D3DX_BC6H::Decode.
 */
export function decodeBC6HBlock(
  bytes: Uint8Array,
  offset: number,
  bSigned: boolean,
  out: Float32Array,
): void {
  const r = new BitReader(bytes, offset);
  let uMode = r.read(2);
  if (uMode !== 0x00 && uMode !== 0x01) {
    uMode = (r.read(3) << 2) | uMode;
  }

  const infoIdx = MODE_TO_INFO[uMode];
  if (infoIdx < 0) {
    // Per the BC6H format spec, reserved/invalid modes decode to opaque black.
    fillBlack(out);
    return;
  }

  const desc = MODE_DESC[infoIdx];
  const info = MODE_INFO[infoIdx];

  // Endpoints: [e0.A, e0.B, e1.A, e1.B], each [r, g, b].
  const ep = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  let uShape = 0;

  // Read header
  const uHeaderBits = info.uPartitions > 0 ? 82 : 65;
  while (r.bit < uHeaderBits) {
    const uCurBit = r.bit;
    if (r.readBit()) {
      const field = desc[uCurBit][0];
      const bit = desc[uCurBit][1];
      if (field === D) {
        uShape |= 1 << bit;
      } else if (field >= RW && field <= BZ) {
        ep[(field - RW) & 3][(field - RW) >> 2] |= 1 << bit;
      } else {
        // Invalid header bits encountered during decoding.
        fillBlack(out);
        return;
      }
    }
  }

  // Sign extend necessary endpoints.
  if (bSigned) {
    const p00 = info.prec[0][0];
    for (let c = 0; c < 3; c++) ep[0][c] = signExtend(ep[0][c], p00[c]);
  }
  if (bSigned || info.bTransformed) {
    for (let p = 0; p <= info.uPartitions; p++) {
      if (p !== 0) {
        const pA = info.prec[p][0];
        for (let c = 0; c < 3; c++) ep[p * 2][c] = signExtend(ep[p * 2][c], pA[c]);
      }
      const pB = info.prec[p][1];
      for (let c = 0; c < 3; c++) ep[p * 2 + 1][c] = signExtend(ep[p * 2 + 1][c], pB[c]);
    }
  }

  // Inverse transform the endpoints (deltas anchored at e0.A).
  if (info.bTransformed) {
    const prec = info.prec[0][0];
    for (let c = 0; c < 3; c++) {
      const mask = (1 << prec[c]) - 1;
      ep[1][c] = (ep[1][c] + ep[0][c]) & mask;
      ep[2][c] = (ep[2][c] + ep[0][c]) & mask;
      ep[3][c] = (ep[3][c] + ep[0][c]) & mask;
      if (bSigned) {
        ep[1][c] = signExtend(ep[1][c], prec[c]);
        ep[2][c] = signExtend(ep[2][c], prec[c]);
        ep[3][c] = signExtend(ep[3][c], prec[c]);
      }
    }
  }

  // Unquantize endpoints (always with region-0 endpoint-A precision, per the C++).
  const prec0 = info.prec[0][0];
  const unq: number[][] = [];
  for (let e = 0; e < 4; e++) {
    unq.push([
      unquantize(ep[e][0], prec0[0], bSigned),
      unquantize(ep[e][1], prec0[1], bSigned),
      unquantize(ep[e][2], prec0[2], bSigned),
    ]);
  }

  // Read indices and interpolate.
  const aWeights = info.uPartitions > 0 ? WEIGHTS3 : WEIGHTS4;
  for (let i = 0; i < 16; i++) {
    const uNumBits = isFixUpOffset(info.uPartitions, uShape, i)
      ? info.uIndexPrec - 1
      : info.uIndexPrec;
    if (r.bit + uNumBits > 128) {
      // Invalid block encountered during decoding.
      fillBlack(out);
      return;
    }
    const uIndex = r.read(uNumBits);
    if (uIndex >= (info.uPartitions > 0 ? 8 : 16)) {
      // Invalid index encountered during decoding.
      fillBlack(out);
      return;
    }

    const uRegion = info.uPartitions > 0 ? (PARTITION2[uShape] >> i) & 1 : 0;
    const a = unq[uRegion * 2];
    const b = unq[uRegion * 2 + 1];

    const w = aWeights[uIndex];
    const iw = BC67_WEIGHT_MAX - w;
    const fr = finishUnquantize((a[0] * iw + b[0] * w + BC67_WEIGHT_ROUND) >> BC67_WEIGHT_SHIFT, bSigned);
    const fg = finishUnquantize((a[1] * iw + b[1] * w + BC67_WEIGHT_ROUND) >> BC67_WEIGHT_SHIFT, bSigned);
    const fb = finishUnquantize((a[2] * iw + b[2] * w + BC67_WEIGHT_ROUND) >> BC67_WEIGHT_SHIFT, bSigned);

    const o = i * 4;
    out[o] = halfToFloat(int2F16(fr, bSigned));
    out[o + 1] = halfToFloat(int2F16(fg, bSigned));
    out[o + 2] = halfToFloat(int2F16(fb, bSigned));
    out[o + 3] = 1.0;
  }
}

/** Decode a full BC6H (UF16 or SF16) image to an R32G32B32A32_FLOAT Image. */
export function decodeBC6H(img: Image): Image {
  let bSigned: boolean;
  if (img.format === DXGI_FORMAT.BC6H_UF16) bSigned = false;
  else if (img.format === DXGI_FORMAT.BC6H_SF16) bSigned = true;
  else throw new Error(`decodeBC6H: unsupported format ${img.format} (expected BC6H_UF16 or BC6H_SF16)`);

  const { width, height } = img;
  const dst = makeImage(DXGI_FORMAT.R32G32B32A32_FLOAT, width, height);
  const dstF = new Float32Array(dst.pixels.buffer, dst.pixels.byteOffset, dst.pixels.byteLength >> 2);
  const rowFloats = dst.rowPitch >> 2;
  const xblocks = Math.max(1, (width + 3) >> 2);
  const block = new Float32Array(64);
  for (let by = 0; by * 4 < height; by++) {
    for (let bx = 0; bx < xblocks; bx++) {
      const off = (by * xblocks + bx) * 16;
      decodeBC6HBlock(img.pixels, off, bSigned, block);
      for (let py = 0; py < 4; py++) {
        const y = by * 4 + py;
        if (y >= height) break;
        for (let px = 0; px < 4; px++) {
          const x = bx * 4 + px;
          if (x >= width) break;
          const d = y * rowFloats + x * 4;
          const s = (py * 4 + px) * 4;
          dstF[d] = block[s];
          dstF[d + 1] = block[s + 1];
          dstF[d + 2] = block[s + 2];
          dstF[d + 3] = block[s + 3];
        }
      }
    }
  }
  return dst;
}
