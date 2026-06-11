// BC6H GPU encoder — faithful WGSL port of DirectXTex/Shaders/BC6HEncode.hlsl.
//
// Three compute entry points run as a pipeline driven by compressBC6.ts:
//   TryModeG10CS (mode 11..14 probe, 4 blocks/group)
//   TryModeLE10CS (modes 1..10 via g_mode_id 0..9, 2 blocks/group, ping-pong)
//   EncodeBlockCS (2 blocks/group)
// The HLSL `REF_DEVICE` path is the one ported (barriers on, no early-out),
// matching DirectXTex's shipped compiled shaders.
//
// Translation notes:
//   * f32tof16 -> pack2x16float low half; f16tof32 -> unpack2x16float .x.
//     (HLSL f32tof16 truncates while pack2x16float rounding is implementation
//     defined; difference only affects encoder quality, never validity.)
//   * HLSL int2x3 (2 rows of int3) -> array<vec3<i32>, 2> (mat2x3i).
//   * asint/asuint/asfloat -> bitcast; componentwise ternaries -> select().
//   * inout matrix params -> value-in/value-out functions (avoids pointer-
//     into-composite arguments).
//   * EncodeBlockCS smuggled the partial index bits through pixel_hr (float2)
//     with asuint round-trips; we add an explicit vec2<u32> `scratch` field to
//     the shared struct instead so NaN bit patterns can't be altered.

alias mat2x3i = array<vec3<i32>, 2>;

const SIGNED_F16: u32 = 96u;    // DXGI_FORMAT_BC6H_SF16
const UNSIGNED_F16: u32 = 95u;  // DXGI_FORMAT_BC6H_UF16
const MAX_FLOAT: f32 = 3.402823466e38;
const MIN_FLOAT: f32 = -3.402823466e38;
const MAX_INT: i32 = 2147483647;
const MIN_INT: i32 = -2147483648;

const THREAD_GROUP_SIZE: u32 = 64u;
const BLOCK_SIZE_X: u32 = 4u;
const BLOCK_SIZE_Y: u32 = 4u;

struct Constants {
  g_tex_width: u32,
  g_num_block_x: u32,
  g_format: u32,
  g_mode_id: u32,
  g_start_block_id: u32,
  g_num_total_blocks: u32,
  g_alpha_weight: f32,  // unused by BC6H; kept for cbuffer layout parity
  g_reserved: u32,
};

@group(0) @binding(0) var<uniform> params: Constants;
@group(0) @binding(1) var g_Input: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> g_InBuff: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> g_OutBuff: array<vec4<u32>>;

struct SharedData {
  pixel: vec3<f32>,
  pixel_ph: vec3<i32>,
  pixel_hr: vec3<f32>,
  pixel_lum: f32,
  error: f32,
  best_mode: u32,
  best_partition: u32,
  endPoint_low: vec3<i32>,
  endPoint_high: vec3<i32>,
  endPoint_lum_low: f32,
  endPoint_lum_high: f32,
  scratch: vec2<u32>,  // index-bit accumulator in EncodeBlockCS (was pixel_hr.xy)
};
var<workgroup> shared_temp: array<SharedData, THREAD_GROUP_SIZE>;

//------------------------------------------------------------------------------
// Constant tables
//------------------------------------------------------------------------------
const candidateModeMemory = array<u32, 14>(
  0x00u, 0x01u, 0x02u, 0x06u, 0x0Au, 0x0Eu, 0x12u, 0x16u, 0x1Au, 0x1Eu, 0x03u, 0x07u, 0x0Bu, 0x0Fu,
);
const candidateModeFlag = array<u32, 14>(1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u);
const candidateModeTransformed = array<bool, 14>(
  true, true, true, true, true, true, true, true, true, false, false, true, true, true,
);
const candidateModePrec = array<vec4<u32>, 14>(
  vec4(10u, 5u, 5u, 5u), vec4(7u, 6u, 6u, 6u),
  vec4(11u, 5u, 4u, 4u), vec4(11u, 4u, 5u, 4u), vec4(11u, 4u, 4u, 5u), vec4(9u, 5u, 5u, 5u),
  vec4(8u, 6u, 5u, 5u), vec4(8u, 5u, 6u, 5u), vec4(8u, 5u, 5u, 6u), vec4(6u, 6u, 6u, 6u),
  vec4(10u, 10u, 10u, 10u), vec4(11u, 9u, 9u, 9u), vec4(12u, 8u, 8u, 8u), vec4(16u, 4u, 4u, 4u),
);

const candidateSectionBit = array<u32, 32>(
  0xCCCCu, 0x8888u, 0xEEEEu, 0xECC8u,
  0xC880u, 0xFEECu, 0xFEC8u, 0xEC80u,
  0xC800u, 0xFFECu, 0xFE80u, 0xE800u,
  0xFFE8u, 0xFF00u, 0xFFF0u, 0xF000u,
  0xF710u, 0x008Eu, 0x7100u, 0x08CEu,
  0x008Cu, 0x7310u, 0x3100u, 0x8CCEu,
  0x088Cu, 0x3110u, 0x6666u, 0x366Cu,
  0x17E8u, 0x0FF0u, 0x718Eu, 0x399Cu,
);

const candidateFixUpIndex1D = array<u32, 32>(
  15u, 15u, 15u, 15u,
  15u, 15u, 15u, 15u,
  15u, 15u, 15u, 15u,
  15u, 15u, 15u, 15u,
  15u, 2u, 8u, 2u,
  2u, 8u, 8u, 15u,
  2u, 8u, 2u, 2u,
  8u, 8u, 2u, 2u,
);

// 0, 9, 18, 27, 37, 46, 55, 64
const aStep1 = array<u32, 64>(
  0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u,
  1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u,
  2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u,
  3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u,
  3u, 4u, 4u, 4u, 4u, 4u, 4u, 4u,
  4u, 4u, 5u, 5u, 5u, 5u, 5u, 5u,
  5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u,
  6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u,
);

// 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64
const aStep2 = array<u32, 64>(
  0u, 0u, 0u, 1u, 1u, 1u, 1u, 2u,
  2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u,
  4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u,
  6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u,
  7u, 8u, 8u, 8u, 8u, 9u, 9u, 9u,
  9u, 10u, 10u, 10u, 10u, 10u, 11u, 11u,
  11u, 11u, 12u, 12u, 12u, 12u, 13u, 13u,
  13u, 13u, 14u, 14u, 14u, 14u, 15u, 15u,
);

const aWeight3 = array<i32, 8>(0, 9, 18, 27, 37, 46, 55, 64);
const aWeight4 = array<i32, 16>(0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64);

const RGB2LUM = vec3<f32>(0.2126, 0.7152, 0.0722);

//------------------------------------------------------------------------------
// Half-float helpers
//------------------------------------------------------------------------------
fn f32tof16_1(x: f32) -> u32 { return pack2x16float(vec2<f32>(x, 0.0)) & 0xffffu; }
fn f16tof32_1(x: u32) -> f32 { return unpack2x16float(x).x; }

fn float2half(pixel_f: vec3<f32>) -> vec3<u32> {
  return vec3<u32>(f32tof16_1(pixel_f.x), f32tof16_1(pixel_f.y), f32tof16_1(pixel_f.z));
}
fn half2float(color_h: vec3<u32>) -> vec3<f32> {
  return vec3<f32>(f16tof32_1(color_h.x), f16tof32_1(color_h.y), f16tof32_1(color_h.z));
}

// 1 << s, componentwise, in i32.
fn shl1(s: vec3<u32>) -> vec3<i32> { return vec3<i32>(1) << s; }

//------------------------------------------------------------------------------
// Quantization
//------------------------------------------------------------------------------
fn start_quantize(pixel_h: vec3<u32>) -> vec3<i32> {
  if (params.g_format == UNSIGNED_F16) {
    return bitcast<vec3<i32>>((pixel_h << vec3<u32>(6u)) / vec3<u32>(31u));
  } else {
    let pos = select(
      bitcast<vec3<i32>>((pixel_h << vec3<u32>(5u)) / vec3<u32>(31u)),
      vec3<i32>(0x7fff),
      pixel_h == vec3<u32>(0x7bffu));
    let neg = select(
      -bitcast<vec3<i32>>(((pixel_h & vec3<u32>(0x7fffu)) << vec3<u32>(5u)) / vec3<u32>(31u)),
      vec3<i32>(bitcast<i32>(0xffff8001u)),
      pixel_h == vec3<u32>(0x7bffu));
    return select(neg, pos, pixel_h < vec3<u32>(0x8000u));
  }
}

fn quantize_vec(ep: vec3<i32>, prec: u32) -> vec3<i32> {
  if (params.g_format == UNSIGNED_F16) {
    if (prec >= 15u) { return ep; }
    let q = select(
      (ep << vec3<u32>(prec)) >> vec3<u32>(16u),
      vec3<i32>((1i << prec) - 1),
      ep == vec3<i32>(0xFFFF));
    return select(q, ep, ep == vec3<i32>(0));
  } else {
    if (prec >= 16u) { return ep; }
    let h = (1i << (prec - 1u)) - 1;
    let pos = select((ep << vec3<u32>(prec - 1u)) >> vec3<u32>(15u), vec3<i32>(h), ep == vec3<i32>(0x7FFF));
    let neg = select(-(((-ep) << vec3<u32>(prec - 1u)) >> vec3<u32>(15u)), vec3<i32>(-h), -ep == vec3<i32>(0x7FFF));
    let q = select(neg, pos, ep >= vec3<i32>(0));
    return select(q, ep, ep == vec3<i32>(0));
  }
}

fn quantize_ep(ep: mat2x3i, prec: u32) -> mat2x3i {
  return mat2x3i(quantize_vec(ep[0], prec), quantize_vec(ep[1], prec));
}

struct QuantResult {
  ep: mat2x3i,
  bad: bool,
};

// HLSL finish_quantize_0 / finish_quantize (identical bodies; the difference
// was only inout-vs-out bBadQuantize, handled at the call sites here).
fn finish_quantize_0(ep_in: mat2x3i, prec: vec4<u32>, transformed: bool) -> QuantResult {
  var r: QuantResult;
  var ep = ep_in;
  if (transformed) {
    let h = shl1(prec.yzw - vec3<u32>(1u));  // 1 << (prec.yzw - 1)
    let e1 = ep[1];
    let bBadComponent = select(-e1 > h, e1 >= h, e1 >= vec3<i32>(0));
    r.bad = any(bBadComponent);
    ep[0] = ep[0] & vec3<i32>((1i << prec.x) - 1);
    let clampPos = select(e1, h - vec3<i32>(1), e1 >= h);
    let clampNeg = select(e1 & (shl1(prec.yzw) - vec3<i32>(1)), h, -e1 > h);
    ep[1] = select(clampNeg, clampPos, e1 >= vec3<i32>(0));
  } else {
    let m = vec3<i32>((1i << prec.x) - 1);
    ep[0] = ep[0] & m;
    ep[1] = ep[1] & m;
    r.bad = false;
  }
  r.ep = ep;
  return r;
}

fn finish_quantize_1(ep_in: mat2x3i, prec: vec4<u32>, transformed: bool) -> QuantResult {
  var r: QuantResult;
  var ep = ep_in;
  if (transformed) {
    let h = shl1(prec.yzw - vec3<u32>(1u));
    let bad0 = select(-ep[0] > h, ep[0] >= h, ep[0] >= vec3<i32>(0));
    let bad1 = select(-ep[1] > h, ep[1] >= h, ep[1] >= vec3<i32>(0));
    r.bad = any(bad0) || any(bad1);
    let mask = shl1(prec.yzw) - vec3<i32>(1);
    ep[0] = select(
      select(ep[0] & mask, h, -ep[0] > h),
      select(ep[0], h - vec3<i32>(1), ep[0] >= h),
      ep[0] >= vec3<i32>(0));
    ep[1] = select(
      select(ep[1] & mask, h, -ep[1] > h),
      select(ep[1], h - vec3<i32>(1), ep[1] >= h),
      ep[1] >= vec3<i32>(0));
  } else {
    let m = vec3<i32>((1i << prec.x) - 1);
    ep[0] = ep[0] & m;
    ep[1] = ep[1] & m;
    r.bad = false;
  }
  r.ep = ep;
  return r;
}

//------------------------------------------------------------------------------
// Unquantization
//------------------------------------------------------------------------------
fn sign_extend_vec(prec: vec3<u32>, color: vec3<i32>) -> vec3<i32> {
  let p = shl1(prec - vec3<u32>(1u));
  return select(color, (color & (p - vec3<i32>(1))) - p, (color & p) != vec3<i32>(0));
}

fn start_unquantize_one(ep_in: mat2x3i, prec: vec4<u32>, transformed: bool) -> mat2x3i {
  var ep = ep_in;
  if (params.g_format == SIGNED_F16) {
    ep[0] = sign_extend_vec(vec3<u32>(prec.x), ep[0]);
  }
  if (params.g_format == SIGNED_F16 || transformed) {
    ep[1] = sign_extend_vec(prec.yzw, ep[1]);
  }
  if (transformed) {
    ep[1] += ep[0];
  }
  return ep;
}

fn start_unquantize_two(ep_in: array<mat2x3i, 2>, prec: vec4<u32>, transformed: bool) -> array<mat2x3i, 2> {
  var ep = ep_in;
  if (params.g_format == SIGNED_F16) {
    ep[0][0] = sign_extend_vec(vec3<u32>(prec.x), ep[0][0]);
  }
  if (params.g_format == SIGNED_F16 || transformed) {
    ep[0][1] = sign_extend_vec(prec.yzw, ep[0][1]);
    ep[1][0] = sign_extend_vec(prec.yzw, ep[1][0]);
    ep[1][1] = sign_extend_vec(prec.yzw, ep[1][1]);
  }
  if (transformed) {
    ep[0][1] += ep[0][0];
    ep[1][0] += ep[0][0];
    ep[1][1] += ep[0][0];
  }
  return ep;
}

fn unquantize_vec(c_in: vec3<i32>, prec: u32) -> vec3<i32> {
  var c = c_in;
  if (params.g_format == UNSIGNED_F16) {
    if (prec < 15u) {
      let scaled = ((c << vec3<u32>(16u)) + vec3<i32>(0x8000)) >> vec3<u32>(prec);
      let q = select(scaled, vec3<i32>(0xFFFF), c == vec3<i32>((1i << prec) - 1));
      c = select(c, q, c != vec3<i32>(0));
    }
  } else {
    if (prec < 16u) {
      let s = c < vec3<i32>(0);
      var a = abs(c);
      let scaled = ((a << vec3<u32>(15u)) + vec3<i32>(0x4000)) >> vec3<u32>(prec - 1u);
      let q = select(scaled, vec3<i32>(0x7FFF), a >= vec3<i32>((1i << (prec - 1u)) - 1));
      a = select(a, q, a != vec3<i32>(0));
      c = select(a, -a, s);
    }
  }
  return c;
}

fn unquantize_ep(ep: mat2x3i, prec: u32) -> mat2x3i {
  return mat2x3i(unquantize_vec(ep[0], prec), unquantize_vec(ep[1], prec));
}

fn finish_unquantize(color_in: vec3<i32>) -> vec3<u32> {
  var color = color_in;
  if (params.g_format == UNSIGNED_F16) {
    color = (color * 31) >> vec3<u32>(6u);
  } else {
    color = select((color * 31) >> vec3<u32>(5u), -((-color * 31) >> vec3<u32>(5u)), color < vec3<i32>(0));
    color = select(color, (-color) | vec3<i32>(0x8000), color < vec3<i32>(0));
  }
  return bitcast<vec3<u32>>(color);
}

fn generate_palette_unquantized8(low: vec3<i32>, high: vec3<i32>, i: u32) -> vec3<u32> {
  let w = aWeight3[i];
  let tmp = (low * (64 - w) + high * w + vec3<i32>(32)) >> vec3<u32>(6u);
  return finish_unquantize(tmp);
}

fn generate_palette_unquantized16(low: vec3<i32>, high: vec3<i32>, i: u32) -> vec3<u32> {
  let w = aWeight4[i];
  let tmp = (low * (64 - w) + high * w + vec3<i32>(32)) >> vec3<u32>(6u);
  return finish_unquantize(tmp);
}

//------------------------------------------------------------------------------
// Pass 1: TryModeG10CS — probe modes 11..14 (single-region)
//------------------------------------------------------------------------------
@compute @workgroup_size(THREAD_GROUP_SIZE)
fn TryModeG10CS(@builtin(local_invocation_index) GI: u32, @builtin(workgroup_id) groupID: vec3<u32>) {
  let MAX_USED_THREAD: u32 = 16u;
  let BLOCK_IN_GROUP: u32 = THREAD_GROUP_SIZE / MAX_USED_THREAD;  // 4
  let blockInGroup: u32 = GI / MAX_USED_THREAD;
  let blockID: u32 = params.g_start_block_id + groupID.x * BLOCK_IN_GROUP + blockInGroup;
  let threadBase: u32 = blockInGroup * MAX_USED_THREAD;
  let threadInBlock: u32 = GI - threadBase;

  let block_y = blockID / params.g_num_block_x;
  let block_x = blockID - block_y * params.g_num_block_x;
  let base_x = block_x * BLOCK_SIZE_X;
  let base_y = block_y * BLOCK_SIZE_Y;

  if (threadInBlock < 16u) {
    var px = textureLoad(g_Input, vec2<u32>(base_x + threadInBlock % 4u, base_y + threadInBlock / 4u), 0).rgb;
    px = max(px, vec3<f32>(0.0));
    shared_temp[GI].pixel = px;
    let pixel_h = float2half(px);
    let hr = half2float(pixel_h);
    shared_temp[GI].pixel_hr = hr;
    let lum = dot(hr, RGB2LUM);
    shared_temp[GI].pixel_lum = lum;
    let ph = start_quantize(pixel_h);
    shared_temp[GI].pixel_ph = ph;

    shared_temp[GI].endPoint_low = ph;
    shared_temp[GI].endPoint_high = ph;
    shared_temp[GI].endPoint_lum_low = lum;
    shared_temp[GI].endPoint_lum_high = lum;
  }
  workgroupBarrier();

  if (threadInBlock < 8u) {
    if (shared_temp[GI].endPoint_lum_low > shared_temp[GI + 8u].endPoint_lum_low) {
      shared_temp[GI].endPoint_low = shared_temp[GI + 8u].endPoint_low;
      shared_temp[GI].endPoint_lum_low = shared_temp[GI + 8u].endPoint_lum_low;
    }
    if (shared_temp[GI].endPoint_lum_high < shared_temp[GI + 8u].endPoint_lum_high) {
      shared_temp[GI].endPoint_high = shared_temp[GI + 8u].endPoint_high;
      shared_temp[GI].endPoint_lum_high = shared_temp[GI + 8u].endPoint_lum_high;
    }
  }
  workgroupBarrier();
  if (threadInBlock < 4u) {
    if (shared_temp[GI].endPoint_lum_low > shared_temp[GI + 4u].endPoint_lum_low) {
      shared_temp[GI].endPoint_low = shared_temp[GI + 4u].endPoint_low;
      shared_temp[GI].endPoint_lum_low = shared_temp[GI + 4u].endPoint_lum_low;
    }
    if (shared_temp[GI].endPoint_lum_high < shared_temp[GI + 4u].endPoint_lum_high) {
      shared_temp[GI].endPoint_high = shared_temp[GI + 4u].endPoint_high;
      shared_temp[GI].endPoint_lum_high = shared_temp[GI + 4u].endPoint_lum_high;
    }
  }
  workgroupBarrier();
  if (threadInBlock < 2u) {
    if (shared_temp[GI].endPoint_lum_low > shared_temp[GI + 2u].endPoint_lum_low) {
      shared_temp[GI].endPoint_low = shared_temp[GI + 2u].endPoint_low;
      shared_temp[GI].endPoint_lum_low = shared_temp[GI + 2u].endPoint_lum_low;
    }
    if (shared_temp[GI].endPoint_lum_high < shared_temp[GI + 2u].endPoint_lum_high) {
      shared_temp[GI].endPoint_high = shared_temp[GI + 2u].endPoint_high;
      shared_temp[GI].endPoint_lum_high = shared_temp[GI + 2u].endPoint_lum_high;
    }
  }
  workgroupBarrier();
  if (threadInBlock < 1u) {
    if (shared_temp[GI].endPoint_lum_low > shared_temp[GI + 1u].endPoint_lum_low) {
      shared_temp[GI].endPoint_low = shared_temp[GI + 1u].endPoint_low;
      shared_temp[GI].endPoint_lum_low = shared_temp[GI + 1u].endPoint_lum_low;
    }
    if (shared_temp[GI].endPoint_lum_high < shared_temp[GI + 1u].endPoint_lum_high) {
      shared_temp[GI].endPoint_high = shared_temp[GI + 1u].endPoint_high;
      shared_temp[GI].endPoint_lum_high = shared_temp[GI + 1u].endPoint_lum_high;
    }
  }
  workgroupBarrier();

  // ergod mode_type 11:14
  if (threadInBlock == 0u) {
    var ep0 = shared_temp[threadBase + 0u].endPoint_low;
    var ep1 = shared_temp[threadBase + 0u].endPoint_high;

    // compute_index
    let span = vec3<f32>(ep1 - ep0);
    let span_norm_sqr = dot(span, span);
    let dotProduct = dot(span, vec3<f32>(shared_temp[threadBase + 0u].pixel_ph - ep0));
    if (span_norm_sqr > 0.0 && dotProduct >= 0.0 && u32(dotProduct * 63.49999 / span_norm_sqr) > 32u) {
      let tmp = ep0;
      ep0 = ep1;
      ep1 = tmp;

      shared_temp[GI].endPoint_low = ep0;
      shared_temp[GI].endPoint_high = ep1;
    }
  }
  workgroupBarrier();

  if (threadInBlock < 4u) {
    var endPoint: mat2x3i;
    endPoint[0] = shared_temp[threadBase + 0u].endPoint_low;
    endPoint[1] = shared_temp[threadBase + 0u].endPoint_high;

    let span = vec3<f32>(endPoint[1] - endPoint[0]);
    let span_norm_sqr = dot(span, span);

    let prec = candidateModePrec[threadInBlock + 10u];
    var endPoint_q = quantize_ep(endPoint, prec.x);

    let transformed = candidateModeTransformed[threadInBlock + 10u];
    if (transformed) {
      endPoint_q[1] -= endPoint_q[0];
    }

    let fq = finish_quantize_0(endPoint_q, prec, transformed);
    endPoint_q = fq.ep;
    let bBadQuantize = fq.bad;

    endPoint_q = start_unquantize_one(endPoint_q, prec, transformed);
    endPoint_q = unquantize_ep(endPoint_q, prec.x);

    var error: f32 = 0.0;
    for (var j = 0u; j < 16u; j++) {
      let dotProduct = dot(span, vec3<f32>(shared_temp[threadBase + j].pixel_ph - endPoint[0]));
      var index: u32 = 0u;
      if (span_norm_sqr > 0.0 && dotProduct > 0.0) {
        if (dotProduct < span_norm_sqr) {
          index = aStep2[u32(dotProduct * 63.49999 / span_norm_sqr)];
        } else {
          index = aStep2[63];
        }
      }

      let pixel_rh = generate_palette_unquantized16(endPoint_q[0], endPoint_q[1], index);
      var pixel_r = half2float(pixel_rh);
      pixel_r -= shared_temp[threadBase + j].pixel_hr;
      error += dot(pixel_r, pixel_r);
    }
    if (bBadQuantize) {
      error = 1e20;
    }

    shared_temp[GI].error = error;
    shared_temp[GI].best_mode = candidateModeFlag[threadInBlock + 10u];
  }
  workgroupBarrier();

  if (threadInBlock < 2u) {
    if (shared_temp[GI].error > shared_temp[GI + 2u].error) {
      shared_temp[GI].error = shared_temp[GI + 2u].error;
      shared_temp[GI].best_mode = shared_temp[GI + 2u].best_mode;
    }
  }
  workgroupBarrier();
  if (threadInBlock < 1u) {
    if (shared_temp[GI].error > shared_temp[GI + 1u].error) {
      shared_temp[GI].error = shared_temp[GI + 1u].error;
      shared_temp[GI].best_mode = shared_temp[GI + 1u].best_mode;
    }

    g_OutBuff[blockID] = vec4<u32>(bitcast<u32>(shared_temp[GI].error), shared_temp[GI].best_mode, 0u, 0u);
  }
}

//------------------------------------------------------------------------------
// Pass 2: TryModeLE10CS — probe two-region modes 1..10 (g_mode_id selects)
//------------------------------------------------------------------------------
@compute @workgroup_size(THREAD_GROUP_SIZE)
fn TryModeLE10CS(@builtin(local_invocation_index) GI: u32, @builtin(workgroup_id) groupID: vec3<u32>) {
  let MAX_USED_THREAD: u32 = 32u;
  let BLOCK_IN_GROUP: u32 = THREAD_GROUP_SIZE / MAX_USED_THREAD;  // 2
  let blockInGroup: u32 = GI / MAX_USED_THREAD;
  let blockID: u32 = params.g_start_block_id + groupID.x * BLOCK_IN_GROUP + blockInGroup;
  let threadBase: u32 = blockInGroup * MAX_USED_THREAD;
  let threadInBlock: u32 = GI - threadBase;

  let block_y = blockID / params.g_num_block_x;
  let block_x = blockID - block_y * params.g_num_block_x;
  let base_x = block_x * BLOCK_SIZE_X;
  let base_y = block_y * BLOCK_SIZE_Y;

  if (threadInBlock < 16u) {
    var px = textureLoad(g_Input, vec2<u32>(base_x + threadInBlock % 4u, base_y + threadInBlock / 4u), 0).rgb;
    px = max(px, vec3<f32>(0.0));
    shared_temp[GI].pixel = px;
    let pixel_h = float2half(px);
    let hr = half2float(pixel_h);
    shared_temp[GI].pixel_hr = hr;
    shared_temp[GI].pixel_lum = dot(hr, RGB2LUM);
    shared_temp[GI].pixel_ph = start_quantize(pixel_h);
  }
  workgroupBarrier();

  // ergod mode_type 1:10
  if (threadInBlock < 32u) {
    // find_axis
    var endPoint: array<mat2x3i, 2>;
    endPoint[0][0] = vec3<i32>(MAX_INT);
    endPoint[0][1] = vec3<i32>(MIN_INT);
    endPoint[1][0] = vec3<i32>(MAX_INT);
    endPoint[1][1] = vec3<i32>(MIN_INT);

    var endPoint_lum: array<vec2<f32>, 2>;
    endPoint_lum[0] = vec2<f32>(MAX_FLOAT, MIN_FLOAT);
    endPoint_lum[1] = vec2<f32>(MAX_FLOAT, MIN_FLOAT);

    let bit = candidateSectionBit[threadInBlock];
    for (var i = 0u; i < 16u; i++) {
      let pixel_ph = shared_temp[threadBase + i].pixel_ph;
      let pixel_lum = shared_temp[threadBase + i].pixel_lum;
      if (((bit >> i) & 1u) != 0u) {
        if (endPoint_lum[1].x > pixel_lum) {
          endPoint[1][0] = pixel_ph;
          endPoint_lum[1].x = pixel_lum;
        }
        if (endPoint_lum[1].y < pixel_lum) {
          endPoint[1][1] = pixel_ph;
          endPoint_lum[1].y = pixel_lum;
        }
      } else {
        if (endPoint_lum[0].x > pixel_lum) {
          endPoint[0][0] = pixel_ph;
          endPoint_lum[0].x = pixel_lum;
        }
        if (endPoint_lum[0].y < pixel_lum) {
          endPoint[0][1] = pixel_ph;
          endPoint_lum[0].y = pixel_lum;
        }
      }
    }

    // compute_index
    var span: array<vec3<f32>, 2>;
    var span_norm_sqr: array<f32, 2>;
    for (var p = 0u; p < 2u; p++) {
      span[p] = vec3<f32>(endPoint[p][1] - endPoint[p][0]);
      span_norm_sqr[p] = dot(span[p], span[p]);

      var fixupIdx: u32 = 0u;
      if (p != 0u) {
        fixupIdx = candidateFixUpIndex1D[threadInBlock];
      }
      let dotProduct = dot(span[p], vec3<f32>(shared_temp[threadBase + fixupIdx].pixel_ph - endPoint[p][0]));
      if (span_norm_sqr[p] > 0.0 && dotProduct >= 0.0 && u32(dotProduct * 63.49999 / span_norm_sqr[p]) > 32u) {
        span[p] = -span[p];
        let tmp = endPoint[p][0];
        endPoint[p][0] = endPoint[p][1];
        endPoint[p][1] = tmp;
      }
    }

    let prec = candidateModePrec[params.g_mode_id];
    var endPoint_q = endPoint;
    endPoint_q[0] = quantize_ep(endPoint_q[0], prec.x);
    endPoint_q[1] = quantize_ep(endPoint_q[1], prec.x);

    let transformed = candidateModeTransformed[params.g_mode_id];
    if (transformed) {
      endPoint_q[0][1] -= endPoint_q[0][0];
      endPoint_q[1][0] -= endPoint_q[0][0];
      endPoint_q[1][1] -= endPoint_q[0][0];
    }

    var bBadQuantize = false;
    let fq0 = finish_quantize_0(endPoint_q[0], prec, transformed);
    endPoint_q[0] = fq0.ep;
    bBadQuantize = bBadQuantize || fq0.bad;
    let fq1 = finish_quantize_1(endPoint_q[1], prec, transformed);
    endPoint_q[1] = fq1.ep;
    bBadQuantize = bBadQuantize || fq1.bad;

    endPoint_q = start_unquantize_two(endPoint_q, prec, transformed);

    endPoint_q[0] = unquantize_ep(endPoint_q[0], prec.x);
    endPoint_q[1] = unquantize_ep(endPoint_q[1], prec.x);

    var error: f32 = 0.0;
    for (var j = 0u; j < 16u; j++) {
      var pixel_rh: vec3<u32>;
      if (((bit >> j) & 1u) != 0u) {
        let dotProduct = dot(span[1], vec3<f32>(shared_temp[threadBase + j].pixel_ph - endPoint[1][0]));
        var index: u32 = 0u;
        if (span_norm_sqr[1] > 0.0 && dotProduct > 0.0) {
          if (dotProduct < span_norm_sqr[1]) {
            index = aStep1[u32(dotProduct * 63.49999 / span_norm_sqr[1])];
          } else {
            index = aStep1[63];
          }
        }
        pixel_rh = generate_palette_unquantized8(endPoint_q[1][0], endPoint_q[1][1], index);
      } else {
        let dotProduct = dot(span[0], vec3<f32>(shared_temp[threadBase + j].pixel_ph - endPoint[0][0]));
        var index: u32 = 0u;
        if (span_norm_sqr[0] > 0.0 && dotProduct > 0.0) {
          if (dotProduct < span_norm_sqr[0]) {
            index = aStep1[u32(dotProduct * 63.49999 / span_norm_sqr[0])];
          } else {
            index = aStep1[63];
          }
        }
        pixel_rh = generate_palette_unquantized8(endPoint_q[0][0], endPoint_q[0][1], index);
      }

      var pixel_r = half2float(pixel_rh);
      pixel_r -= shared_temp[threadBase + j].pixel_hr;
      error += dot(pixel_r, pixel_r);
    }
    if (bBadQuantize) {
      error = 1e20;
    }

    shared_temp[GI].error = error;
    shared_temp[GI].best_mode = candidateModeFlag[params.g_mode_id];
    shared_temp[GI].best_partition = threadInBlock;
  }
  workgroupBarrier();

  if (threadInBlock < 16u) {
    if (shared_temp[GI].error > shared_temp[GI + 16u].error) {
      shared_temp[GI].error = shared_temp[GI + 16u].error;
      shared_temp[GI].best_mode = shared_temp[GI + 16u].best_mode;
      shared_temp[GI].best_partition = shared_temp[GI + 16u].best_partition;
    }
  }
  workgroupBarrier();
  if (threadInBlock < 8u) {
    if (shared_temp[GI].error > shared_temp[GI + 8u].error) {
      shared_temp[GI].error = shared_temp[GI + 8u].error;
      shared_temp[GI].best_mode = shared_temp[GI + 8u].best_mode;
      shared_temp[GI].best_partition = shared_temp[GI + 8u].best_partition;
    }
  }
  workgroupBarrier();
  if (threadInBlock < 4u) {
    if (shared_temp[GI].error > shared_temp[GI + 4u].error) {
      shared_temp[GI].error = shared_temp[GI + 4u].error;
      shared_temp[GI].best_mode = shared_temp[GI + 4u].best_mode;
      shared_temp[GI].best_partition = shared_temp[GI + 4u].best_partition;
    }
  }
  workgroupBarrier();
  if (threadInBlock < 2u) {
    if (shared_temp[GI].error > shared_temp[GI + 2u].error) {
      shared_temp[GI].error = shared_temp[GI + 2u].error;
      shared_temp[GI].best_mode = shared_temp[GI + 2u].best_mode;
      shared_temp[GI].best_partition = shared_temp[GI + 2u].best_partition;
    }
  }
  workgroupBarrier();
  if (threadInBlock < 1u) {
    if (shared_temp[GI].error > shared_temp[GI + 1u].error) {
      shared_temp[GI].error = shared_temp[GI + 1u].error;
      shared_temp[GI].best_mode = shared_temp[GI + 1u].best_mode;
      shared_temp[GI].best_partition = shared_temp[GI + 1u].best_partition;
    }

    if (bitcast<f32>(g_InBuff[blockID].x) > shared_temp[GI].error) {
      g_OutBuff[blockID] = vec4<u32>(
        bitcast<u32>(shared_temp[GI].error),
        shared_temp[GI].best_mode,
        shared_temp[GI].best_partition,
        0u);
    } else {
      g_OutBuff[blockID] = g_InBuff[blockID];
    }
  }
}

//------------------------------------------------------------------------------
// Pass 3: EncodeBlockCS — re-derive endpoints for the winning mode and pack bits
//------------------------------------------------------------------------------
@compute @workgroup_size(THREAD_GROUP_SIZE)
fn EncodeBlockCS(@builtin(local_invocation_index) GI: u32, @builtin(workgroup_id) groupID: vec3<u32>) {
  let MAX_USED_THREAD: u32 = 32u;
  let BLOCK_IN_GROUP: u32 = THREAD_GROUP_SIZE / MAX_USED_THREAD;  // 2
  let blockInGroup: u32 = GI / MAX_USED_THREAD;
  let blockID: u32 = params.g_start_block_id + groupID.x * BLOCK_IN_GROUP + blockInGroup;
  let threadBase: u32 = blockInGroup * MAX_USED_THREAD;
  let threadInBlock: u32 = GI - threadBase;

  let block_y = blockID / params.g_num_block_x;
  let block_x = blockID - block_y * params.g_num_block_x;
  let base_x = block_x * BLOCK_SIZE_X;
  let base_y = block_y * BLOCK_SIZE_Y;

  if (threadInBlock < 16u) {
    var px = textureLoad(g_Input, vec2<u32>(base_x + threadInBlock % 4u, base_y + threadInBlock / 4u), 0).rgb;
    px = max(px, vec3<f32>(0.0));
    shared_temp[GI].pixel = px;
    shared_temp[GI].pixel_lum = dot(px, RGB2LUM);
    let pixel_h = float2half(px);
    shared_temp[GI].pixel_ph = start_quantize(pixel_h);
  }
  workgroupBarrier();

  let best_mode = g_InBuff[blockID].y;
  let best_partition = g_InBuff[blockID].z;

  var blk = vec4<u32>(0u);

  if (threadInBlock < 32u) {
    var endPoint: mat2x3i;
    endPoint[0] = vec3<i32>(MAX_INT);
    endPoint[1] = vec3<i32>(MIN_INT);

    var endPoint_lum = vec2<f32>(MAX_FLOAT, MIN_FLOAT);

    let pixel_ph = shared_temp[threadBase + (threadInBlock & 0xFu)].pixel_ph;
    let pixel_lum = shared_temp[threadBase + (threadInBlock & 0xFu)].pixel_lum;
    if (threadInBlock < 16u) {
      if (best_mode > 10u) {
        endPoint[0] = pixel_ph;
        endPoint[1] = pixel_ph;
        endPoint_lum = vec2<f32>(pixel_lum);
      } else {
        let bits = candidateSectionBit[best_partition];
        if (((bits >> threadInBlock) & 1u) == 0u) {
          endPoint[0] = pixel_ph;
          endPoint[1] = pixel_ph;
          endPoint_lum = vec2<f32>(pixel_lum);
        }
      }
    } else {
      if (best_mode <= 10u) {
        let bits = candidateSectionBit[best_partition];
        if (((bits >> (threadInBlock & 0xFu)) & 1u) == 1u) {
          endPoint[0] = pixel_ph;
          endPoint[1] = pixel_ph;
          endPoint_lum = vec2<f32>(pixel_lum);
        }
      }
    }

    shared_temp[GI].endPoint_low = endPoint[0];
    shared_temp[GI].endPoint_high = endPoint[1];

    shared_temp[GI].endPoint_lum_low = endPoint_lum.x;
    shared_temp[GI].endPoint_lum_high = endPoint_lum.y;
  }
  workgroupBarrier();
  if ((threadInBlock & 0xFu) < 8u) {
    if (shared_temp[GI].endPoint_lum_low > shared_temp[GI + 8u].endPoint_lum_low) {
      shared_temp[GI].endPoint_low = shared_temp[GI + 8u].endPoint_low;
      shared_temp[GI].endPoint_lum_low = shared_temp[GI + 8u].endPoint_lum_low;
    }
    if (shared_temp[GI].endPoint_lum_high < shared_temp[GI + 8u].endPoint_lum_high) {
      shared_temp[GI].endPoint_high = shared_temp[GI + 8u].endPoint_high;
      shared_temp[GI].endPoint_lum_high = shared_temp[GI + 8u].endPoint_lum_high;
    }
  }
  workgroupBarrier();
  if ((threadInBlock & 0xFu) < 4u) {
    if (shared_temp[GI].endPoint_lum_low > shared_temp[GI + 4u].endPoint_lum_low) {
      shared_temp[GI].endPoint_low = shared_temp[GI + 4u].endPoint_low;
      shared_temp[GI].endPoint_lum_low = shared_temp[GI + 4u].endPoint_lum_low;
    }
    if (shared_temp[GI].endPoint_lum_high < shared_temp[GI + 4u].endPoint_lum_high) {
      shared_temp[GI].endPoint_high = shared_temp[GI + 4u].endPoint_high;
      shared_temp[GI].endPoint_lum_high = shared_temp[GI + 4u].endPoint_lum_high;
    }
  }
  workgroupBarrier();
  if ((threadInBlock & 0xFu) < 2u) {
    if (shared_temp[GI].endPoint_lum_low > shared_temp[GI + 2u].endPoint_lum_low) {
      shared_temp[GI].endPoint_low = shared_temp[GI + 2u].endPoint_low;
      shared_temp[GI].endPoint_lum_low = shared_temp[GI + 2u].endPoint_lum_low;
    }
    if (shared_temp[GI].endPoint_lum_high < shared_temp[GI + 2u].endPoint_lum_high) {
      shared_temp[GI].endPoint_high = shared_temp[GI + 2u].endPoint_high;
      shared_temp[GI].endPoint_lum_high = shared_temp[GI + 2u].endPoint_lum_high;
    }
  }
  workgroupBarrier();
  if ((threadInBlock & 0xFu) < 1u) {
    if (shared_temp[GI].endPoint_lum_low > shared_temp[GI + 1u].endPoint_lum_low) {
      shared_temp[GI].endPoint_low = shared_temp[GI + 1u].endPoint_low;
      shared_temp[GI].endPoint_lum_low = shared_temp[GI + 1u].endPoint_lum_low;
    }
    if (shared_temp[GI].endPoint_lum_high < shared_temp[GI + 1u].endPoint_lum_high) {
      shared_temp[GI].endPoint_high = shared_temp[GI + 1u].endPoint_high;
      shared_temp[GI].endPoint_lum_high = shared_temp[GI + 1u].endPoint_lum_high;
    }
  }
  workgroupBarrier();

  if (threadInBlock < 2u) {
    // find_axis
    var endPoint: mat2x3i;
    endPoint[0] = shared_temp[threadBase + threadInBlock * 16u].endPoint_low;
    endPoint[1] = shared_temp[threadBase + threadInBlock * 16u].endPoint_high;

    var fixup: u32 = 0u;
    if ((1u == threadInBlock) && (best_mode <= 10u)) {
      fixup = candidateFixUpIndex1D[best_partition];
    }

    let span = vec3<f32>(endPoint[1] - endPoint[0]);
    let span_norm_sqr = dot(span, span);
    let dotProduct = dot(span, vec3<f32>(shared_temp[threadBase + fixup].pixel_ph - endPoint[0]));
    if (span_norm_sqr > 0.0 && dotProduct >= 0.0 && u32(dotProduct * 63.49999 / span_norm_sqr) > 32u) {
      let tmp = endPoint[0];
      endPoint[0] = endPoint[1];
      endPoint[1] = tmp;
    }

    shared_temp[GI].endPoint_low = endPoint[0];
    shared_temp[GI].endPoint_high = endPoint[1];
  }
  workgroupBarrier();

  if (threadInBlock < 16u) {
    var bits: u32 = 0u;
    if (best_mode <= 10u) {
      bits = candidateSectionBit[best_partition];
    }

    var span: vec3<f32>;
    var dotProduct: f32;
    if (((bits >> threadInBlock) & 1u) != 0u) {
      span = vec3<f32>(shared_temp[threadBase + 1u].endPoint_high - shared_temp[threadBase + 1u].endPoint_low);
      dotProduct = dot(span, vec3<f32>(shared_temp[threadBase + threadInBlock].pixel_ph - shared_temp[threadBase + 1u].endPoint_low));
    } else {
      span = vec3<f32>(shared_temp[threadBase + 0u].endPoint_high - shared_temp[threadBase + 0u].endPoint_low);
      dotProduct = dot(span, vec3<f32>(shared_temp[threadBase + threadInBlock].pixel_ph - shared_temp[threadBase + 0u].endPoint_low));
    }
    let span_norm_sqr = dot(span, span);

    if (best_mode > 10u) {
      var index: u32 = 0u;
      if (span_norm_sqr > 0.0 && dotProduct > 0.0) {
        if (dotProduct < span_norm_sqr) {
          index = aStep2[u32(dotProduct * 63.49999 / span_norm_sqr)];
        } else {
          index = aStep2[63];
        }
      }
      if (threadInBlock == 0u) {
        blk.z |= index << 1u;
      } else if (threadInBlock < 8u) {
        blk.z |= index << (threadInBlock * 4u);
      } else {
        blk.w |= index << ((threadInBlock - 8u) * 4u);
      }
    } else {
      var index: u32 = 0u;
      if (span_norm_sqr > 0.0 && dotProduct > 0.0) {
        if (dotProduct < span_norm_sqr) {
          index = aStep1[u32(dotProduct * 63.49999 / span_norm_sqr)];
        } else {
          index = aStep1[63];
        }
      }

      let fixup = candidateFixUpIndex1D[best_partition];
      let offset = vec2<u32>(u32(fixup != 2u), u32(fixup == 15u));

      if (threadInBlock == 0u) {
        blk.z |= index << 18u;
      } else if (threadInBlock < 3u) {
        blk.z |= index << (20u + (threadInBlock - 1u) * 3u);
      } else if (threadInBlock < 5u) {
        blk.z |= index << (25u + (threadInBlock - 3u) * 3u + offset.x);
      } else if (threadInBlock == 5u) {
        blk.w |= index >> (1u - offset.x);  // index >> !offset.x
        if (offset.x == 0u) {
          blk.z |= index << 31u;
        }
      } else if (threadInBlock < 9u) {
        blk.w |= index << (2u + (threadInBlock - 6u) * 3u + offset.x);
      } else {
        blk.w |= index << (11u + (threadInBlock - 9u) * 3u + offset.y);
      }
    }

    shared_temp[GI].scratch = blk.zw;
  }
  workgroupBarrier();
  if (threadInBlock < 8u) {
    shared_temp[GI].scratch |= shared_temp[GI + 8u].scratch;
  }
  workgroupBarrier();
  if (threadInBlock < 4u) {
    shared_temp[GI].scratch |= shared_temp[GI + 4u].scratch;
  }
  workgroupBarrier();
  if (threadInBlock < 2u) {
    shared_temp[GI].scratch |= shared_temp[GI + 2u].scratch;
  }
  workgroupBarrier();
  if (threadInBlock < 1u) {
    shared_temp[GI].scratch |= shared_temp[GI + 1u].scratch;

    let s = shared_temp[GI].scratch;
    blk.z = s.x;
    blk.w = s.y;
  }
  workgroupBarrier();

  let transformed = candidateModeTransformed[best_mode - 1u];
  let prec = candidateModePrec[best_mode - 1u];
  if (threadInBlock == 2u) {
    var endPoint_q: mat2x3i;
    endPoint_q[0] = shared_temp[threadBase + 0u].endPoint_low;
    endPoint_q[1] = shared_temp[threadBase + 0u].endPoint_high;

    endPoint_q = quantize_ep(endPoint_q, prec.x);
    if (transformed) {
      endPoint_q[1] -= endPoint_q[0];
    }

    shared_temp[GI].endPoint_low = endPoint_q[0];
    shared_temp[GI].endPoint_high = endPoint_q[1];
  }
  workgroupBarrier();
  if (threadInBlock == 3u) {
    let ep0 = shared_temp[threadBase + 2u].endPoint_low;
    var endPoint_q: mat2x3i;
    endPoint_q[0] = shared_temp[threadBase + 1u].endPoint_low;
    endPoint_q[1] = shared_temp[threadBase + 1u].endPoint_high;

    if (best_mode <= 10u) {
      endPoint_q = quantize_ep(endPoint_q, prec.x);
      if (transformed) {
        endPoint_q[0] -= ep0;
        endPoint_q[1] -= ep0;
      }

      shared_temp[GI].endPoint_low = endPoint_q[0];
      shared_temp[GI].endPoint_high = endPoint_q[1];
    }
  }
  workgroupBarrier();

  if (threadInBlock < 2u) {
    var endPoint_q: mat2x3i;
    endPoint_q[0] = shared_temp[threadBase + threadInBlock + 2u].endPoint_low;
    endPoint_q[1] = shared_temp[threadBase + threadInBlock + 2u].endPoint_high;

    if (threadInBlock == 0u) {
      // best_mode > 10 used HLSL finish_quantize, <= 10 finish_quantize_0;
      // their bodies are identical (only the out/inout flag differed).
      let fq = finish_quantize_0(endPoint_q, prec, transformed);
      endPoint_q = fq.ep;
    } else {  // threadInBlock == 1
      if (best_mode <= 10u) {
        let fq = finish_quantize_1(endPoint_q, prec, transformed);
        endPoint_q = fq.ep;
      }
    }

    shared_temp[GI].endPoint_low = endPoint_q[0];
    shared_temp[GI].endPoint_high = endPoint_q[1];
  }
  workgroupBarrier();

  if (threadInBlock == 0u) {
    var endPoint_q: array<mat2x3i, 2>;
    endPoint_q[0][0] = shared_temp[threadBase + 0u].endPoint_low;
    endPoint_q[0][1] = shared_temp[threadBase + 0u].endPoint_high;
    endPoint_q[1][0] = shared_temp[threadBase + 1u].endPoint_low;
    endPoint_q[1][1] = shared_temp[threadBase + 1u].endPoint_high;

    if (best_mode > 10u) {
      blk = block_package_one(blk, endPoint_q[0], best_mode);
    } else {
      blk = block_package_two(blk, endPoint_q, best_mode, best_partition);
    }

    g_OutBuff[blockID] = blk;
  }
}

//------------------------------------------------------------------------------
// block_package — bit packing (mechanical translation of the HLSL bit tables)
//------------------------------------------------------------------------------
// for mode 1 - 10
fn block_package_two(blk_in: vec4<u32>, endPoint: array<mat2x3i, 2>, mode_type: u32, partition_index: u32) -> vec4<u32> {
  var blk = blk_in;
  blk.x = 0u;
  blk.y = 0u;
  blk.z &= 0xFFFC0000u;

  if (mode_type == candidateModeFlag[0]) {

    blk.x |= ((candidateModeMemory[0] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[0] >> 1u) & 1u) << 1u;
    blk.x |= u32((endPoint[1][0].g >> 4u) & 1) << 2u;
    blk.x |= u32((endPoint[1][0].b >> 4u) & 1) << 3u;
    blk.x |= u32((endPoint[1][1].b >> 4u) & 1) << 4u;
    blk.x |= u32((endPoint[0][0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0][0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0][0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0][0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0][0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0][0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[0][0].r >> 6u) & 1) << 11u;
    blk.x |= u32((endPoint[0][0].r >> 7u) & 1) << 12u;
    blk.x |= u32((endPoint[0][0].r >> 8u) & 1) << 13u;
    blk.x |= u32((endPoint[0][0].r >> 9u) & 1) << 14u;
    blk.x |= u32((endPoint[0][0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0][0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0][0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0][0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0][0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0][0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[0][0].g >> 6u) & 1) << 21u;
    blk.x |= u32((endPoint[0][0].g >> 7u) & 1) << 22u;
    blk.x |= u32((endPoint[0][0].g >> 8u) & 1) << 23u;
    blk.x |= u32((endPoint[0][0].g >> 9u) & 1) << 24u;
    blk.x |= u32((endPoint[0][0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0][0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0][0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0][0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0][0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0][0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[0][0].b >> 6u) & 1) << 31u;
    blk.y |= u32((endPoint[0][0].b >> 7u) & 1) << 0u;
    blk.y |= u32((endPoint[0][0].b >> 8u) & 1) << 1u;
    blk.y |= u32((endPoint[0][0].b >> 9u) & 1) << 2u;
    blk.y |= u32((endPoint[0][1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[0][1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[0][1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[0][1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[0][1].r >> 4u) & 1) << 7u;
    blk.y |= u32((endPoint[1][1].g >> 4u) & 1) << 8u;
    blk.y |= u32((endPoint[1][0].g >> 0u) & 1) << 9u;
    blk.y |= u32((endPoint[1][0].g >> 1u) & 1) << 10u;
    blk.y |= u32((endPoint[1][0].g >> 2u) & 1) << 11u;
    blk.y |= u32((endPoint[1][0].g >> 3u) & 1) << 12u;
    blk.y |= u32((endPoint[0][1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[0][1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[0][1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[0][1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[0][1].g >> 4u) & 1) << 17u;
    blk.y |= u32((endPoint[1][1].b >> 0u) & 1) << 18u;
    blk.y |= u32((endPoint[1][1].g >> 0u) & 1) << 19u;
    blk.y |= u32((endPoint[1][1].g >> 1u) & 1) << 20u;
    blk.y |= u32((endPoint[1][1].g >> 2u) & 1) << 21u;
    blk.y |= u32((endPoint[1][1].g >> 3u) & 1) << 22u;
    blk.y |= u32((endPoint[0][1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[0][1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[0][1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[0][1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[0][1].b >> 4u) & 1) << 27u;
    blk.y |= u32((endPoint[1][1].b >> 1u) & 1) << 28u;
    blk.y |= u32((endPoint[1][0].b >> 0u) & 1) << 29u;
    blk.y |= u32((endPoint[1][0].b >> 1u) & 1) << 30u;
    blk.y |= u32((endPoint[1][0].b >> 2u) & 1) << 31u;
    blk.z |= u32((endPoint[1][0].b >> 3u) & 1) << 0u;
    blk.z |= u32((endPoint[1][0].r >> 0u) & 1) << 1u;
    blk.z |= u32((endPoint[1][0].r >> 1u) & 1) << 2u;
    blk.z |= u32((endPoint[1][0].r >> 2u) & 1) << 3u;
    blk.z |= u32((endPoint[1][0].r >> 3u) & 1) << 4u;
    blk.z |= u32((endPoint[1][0].r >> 4u) & 1) << 5u;
    blk.z |= u32((endPoint[1][1].b >> 2u) & 1) << 6u;
    blk.z |= u32((endPoint[1][1].r >> 0u) & 1) << 7u;
    blk.z |= u32((endPoint[1][1].r >> 1u) & 1) << 8u;
    blk.z |= u32((endPoint[1][1].r >> 2u) & 1) << 9u;
    blk.z |= u32((endPoint[1][1].r >> 3u) & 1) << 10u;
    blk.z |= u32((endPoint[1][1].r >> 4u) & 1) << 11u;
    blk.z |= u32((endPoint[1][1].b >> 3u) & 1) << 12u;
    blk.z |= ((partition_index >> 0u) & 1u) << 13u;
    blk.z |= ((partition_index >> 1u) & 1u) << 14u;
    blk.z |= ((partition_index >> 2u) & 1u) << 15u;
    blk.z |= ((partition_index >> 3u) & 1u) << 16u;
    blk.z |= ((partition_index >> 4u) & 1u) << 17u;
  }
  else if (mode_type == candidateModeFlag[1]) {

    blk.x |= ((candidateModeMemory[1] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[1] >> 1u) & 1u) << 1u;
    blk.x |= u32((endPoint[1][0].g >> 5u) & 1) << 2u;
    blk.x |= u32((endPoint[1][1].g >> 4u) & 1) << 3u;
    blk.x |= u32((endPoint[1][1].g >> 5u) & 1) << 4u;
    blk.x |= u32((endPoint[0][0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0][0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0][0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0][0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0][0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0][0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[0][0].r >> 6u) & 1) << 11u;
    blk.x |= u32((endPoint[1][1].b >> 0u) & 1) << 12u;
    blk.x |= u32((endPoint[1][1].b >> 1u) & 1) << 13u;
    blk.x |= u32((endPoint[1][0].b >> 4u) & 1) << 14u;
    blk.x |= u32((endPoint[0][0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0][0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0][0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0][0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0][0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0][0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[0][0].g >> 6u) & 1) << 21u;
    blk.x |= u32((endPoint[1][0].b >> 5u) & 1) << 22u;
    blk.x |= u32((endPoint[1][1].b >> 2u) & 1) << 23u;
    blk.x |= u32((endPoint[1][0].g >> 4u) & 1) << 24u;
    blk.x |= u32((endPoint[0][0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0][0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0][0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0][0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0][0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0][0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[0][0].b >> 6u) & 1) << 31u;
    blk.y |= u32((endPoint[1][1].b >> 3u) & 1) << 0u;
    blk.y |= u32((endPoint[1][1].b >> 5u) & 1) << 1u;
    blk.y |= u32((endPoint[1][1].b >> 4u) & 1) << 2u;
    blk.y |= u32((endPoint[0][1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[0][1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[0][1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[0][1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[0][1].r >> 4u) & 1) << 7u;
    blk.y |= u32((endPoint[0][1].r >> 5u) & 1) << 8u;
    blk.y |= u32((endPoint[1][0].g >> 0u) & 1) << 9u;
    blk.y |= u32((endPoint[1][0].g >> 1u) & 1) << 10u;
    blk.y |= u32((endPoint[1][0].g >> 2u) & 1) << 11u;
    blk.y |= u32((endPoint[1][0].g >> 3u) & 1) << 12u;
    blk.y |= u32((endPoint[0][1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[0][1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[0][1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[0][1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[0][1].g >> 4u) & 1) << 17u;
    blk.y |= u32((endPoint[0][1].g >> 5u) & 1) << 18u;
    blk.y |= u32((endPoint[1][1].g >> 0u) & 1) << 19u;
    blk.y |= u32((endPoint[1][1].g >> 1u) & 1) << 20u;
    blk.y |= u32((endPoint[1][1].g >> 2u) & 1) << 21u;
    blk.y |= u32((endPoint[1][1].g >> 3u) & 1) << 22u;
    blk.y |= u32((endPoint[0][1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[0][1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[0][1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[0][1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[0][1].b >> 4u) & 1) << 27u;
    blk.y |= u32((endPoint[0][1].b >> 5u) & 1) << 28u;
    blk.y |= u32((endPoint[1][0].b >> 0u) & 1) << 29u;
    blk.y |= u32((endPoint[1][0].b >> 1u) & 1) << 30u;
    blk.y |= u32((endPoint[1][0].b >> 2u) & 1) << 31u;
    blk.z |= u32((endPoint[1][0].b >> 3u) & 1) << 0u;
    blk.z |= u32((endPoint[1][0].r >> 0u) & 1) << 1u;
    blk.z |= u32((endPoint[1][0].r >> 1u) & 1) << 2u;
    blk.z |= u32((endPoint[1][0].r >> 2u) & 1) << 3u;
    blk.z |= u32((endPoint[1][0].r >> 3u) & 1) << 4u;
    blk.z |= u32((endPoint[1][0].r >> 4u) & 1) << 5u;
    blk.z |= u32((endPoint[1][0].r >> 5u) & 1) << 6u;
    blk.z |= u32((endPoint[1][1].r >> 0u) & 1) << 7u;
    blk.z |= u32((endPoint[1][1].r >> 1u) & 1) << 8u;
    blk.z |= u32((endPoint[1][1].r >> 2u) & 1) << 9u;
    blk.z |= u32((endPoint[1][1].r >> 3u) & 1) << 10u;
    blk.z |= u32((endPoint[1][1].r >> 4u) & 1) << 11u;
    blk.z |= u32((endPoint[1][1].r >> 5u) & 1) << 12u;
    blk.z |= ((partition_index >> 0u) & 1u) << 13u;
    blk.z |= ((partition_index >> 1u) & 1u) << 14u;
    blk.z |= ((partition_index >> 2u) & 1u) << 15u;
    blk.z |= ((partition_index >> 3u) & 1u) << 16u;
    blk.z |= ((partition_index >> 4u) & 1u) << 17u;
  }
  else if (mode_type == candidateModeFlag[2]) {

    blk.x |= ((candidateModeMemory[2] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[2] >> 1u) & 1u) << 1u;
    blk.x |= ((candidateModeMemory[2] >> 2u) & 1u) << 2u;
    blk.x |= ((candidateModeMemory[2] >> 3u) & 1u) << 3u;
    blk.x |= ((candidateModeMemory[2] >> 4u) & 1u) << 4u;
    blk.x |= u32((endPoint[0][0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0][0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0][0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0][0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0][0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0][0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[0][0].r >> 6u) & 1) << 11u;
    blk.x |= u32((endPoint[0][0].r >> 7u) & 1) << 12u;
    blk.x |= u32((endPoint[0][0].r >> 8u) & 1) << 13u;
    blk.x |= u32((endPoint[0][0].r >> 9u) & 1) << 14u;
    blk.x |= u32((endPoint[0][0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0][0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0][0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0][0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0][0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0][0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[0][0].g >> 6u) & 1) << 21u;
    blk.x |= u32((endPoint[0][0].g >> 7u) & 1) << 22u;
    blk.x |= u32((endPoint[0][0].g >> 8u) & 1) << 23u;
    blk.x |= u32((endPoint[0][0].g >> 9u) & 1) << 24u;
    blk.x |= u32((endPoint[0][0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0][0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0][0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0][0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0][0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0][0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[0][0].b >> 6u) & 1) << 31u;
    blk.y |= u32((endPoint[0][0].b >> 7u) & 1) << 0u;
    blk.y |= u32((endPoint[0][0].b >> 8u) & 1) << 1u;
    blk.y |= u32((endPoint[0][0].b >> 9u) & 1) << 2u;
    blk.y |= u32((endPoint[0][1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[0][1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[0][1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[0][1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[0][1].r >> 4u) & 1) << 7u;
    blk.y |= u32((endPoint[0][0].r >> 10u) & 1) << 8u;
    blk.y |= u32((endPoint[1][0].g >> 0u) & 1) << 9u;
    blk.y |= u32((endPoint[1][0].g >> 1u) & 1) << 10u;
    blk.y |= u32((endPoint[1][0].g >> 2u) & 1) << 11u;
    blk.y |= u32((endPoint[1][0].g >> 3u) & 1) << 12u;
    blk.y |= u32((endPoint[0][1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[0][1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[0][1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[0][1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[0][0].g >> 10u) & 1) << 17u;
    blk.y |= u32((endPoint[1][1].b >> 0u) & 1) << 18u;
    blk.y |= u32((endPoint[1][1].g >> 0u) & 1) << 19u;
    blk.y |= u32((endPoint[1][1].g >> 1u) & 1) << 20u;
    blk.y |= u32((endPoint[1][1].g >> 2u) & 1) << 21u;
    blk.y |= u32((endPoint[1][1].g >> 3u) & 1) << 22u;
    blk.y |= u32((endPoint[0][1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[0][1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[0][1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[0][1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[0][0].b >> 10u) & 1) << 27u;
    blk.y |= u32((endPoint[1][1].b >> 1u) & 1) << 28u;
    blk.y |= u32((endPoint[1][0].b >> 0u) & 1) << 29u;
    blk.y |= u32((endPoint[1][0].b >> 1u) & 1) << 30u;
    blk.y |= u32((endPoint[1][0].b >> 2u) & 1) << 31u;
    blk.z |= u32((endPoint[1][0].b >> 3u) & 1) << 0u;
    blk.z |= u32((endPoint[1][0].r >> 0u) & 1) << 1u;
    blk.z |= u32((endPoint[1][0].r >> 1u) & 1) << 2u;
    blk.z |= u32((endPoint[1][0].r >> 2u) & 1) << 3u;
    blk.z |= u32((endPoint[1][0].r >> 3u) & 1) << 4u;
    blk.z |= u32((endPoint[1][0].r >> 4u) & 1) << 5u;
    blk.z |= u32((endPoint[1][1].b >> 2u) & 1) << 6u;
    blk.z |= u32((endPoint[1][1].r >> 0u) & 1) << 7u;
    blk.z |= u32((endPoint[1][1].r >> 1u) & 1) << 8u;
    blk.z |= u32((endPoint[1][1].r >> 2u) & 1) << 9u;
    blk.z |= u32((endPoint[1][1].r >> 3u) & 1) << 10u;
    blk.z |= u32((endPoint[1][1].r >> 4u) & 1) << 11u;
    blk.z |= u32((endPoint[1][1].b >> 3u) & 1) << 12u;
    blk.z |= ((partition_index >> 0u) & 1u) << 13u;
    blk.z |= ((partition_index >> 1u) & 1u) << 14u;
    blk.z |= ((partition_index >> 2u) & 1u) << 15u;
    blk.z |= ((partition_index >> 3u) & 1u) << 16u;
    blk.z |= ((partition_index >> 4u) & 1u) << 17u;
  }
  else if (mode_type == candidateModeFlag[3]) {

    blk.x |= ((candidateModeMemory[3] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[3] >> 1u) & 1u) << 1u;
    blk.x |= ((candidateModeMemory[3] >> 2u) & 1u) << 2u;
    blk.x |= ((candidateModeMemory[3] >> 3u) & 1u) << 3u;
    blk.x |= ((candidateModeMemory[3] >> 4u) & 1u) << 4u;
    blk.x |= u32((endPoint[0][0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0][0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0][0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0][0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0][0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0][0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[0][0].r >> 6u) & 1) << 11u;
    blk.x |= u32((endPoint[0][0].r >> 7u) & 1) << 12u;
    blk.x |= u32((endPoint[0][0].r >> 8u) & 1) << 13u;
    blk.x |= u32((endPoint[0][0].r >> 9u) & 1) << 14u;
    blk.x |= u32((endPoint[0][0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0][0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0][0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0][0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0][0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0][0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[0][0].g >> 6u) & 1) << 21u;
    blk.x |= u32((endPoint[0][0].g >> 7u) & 1) << 22u;
    blk.x |= u32((endPoint[0][0].g >> 8u) & 1) << 23u;
    blk.x |= u32((endPoint[0][0].g >> 9u) & 1) << 24u;
    blk.x |= u32((endPoint[0][0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0][0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0][0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0][0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0][0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0][0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[0][0].b >> 6u) & 1) << 31u;
    blk.y |= u32((endPoint[0][0].b >> 7u) & 1) << 0u;
    blk.y |= u32((endPoint[0][0].b >> 8u) & 1) << 1u;
    blk.y |= u32((endPoint[0][0].b >> 9u) & 1) << 2u;
    blk.y |= u32((endPoint[0][1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[0][1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[0][1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[0][1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[0][0].r >> 10u) & 1) << 7u;
    blk.y |= u32((endPoint[1][1].g >> 4u) & 1) << 8u;
    blk.y |= u32((endPoint[1][0].g >> 0u) & 1) << 9u;
    blk.y |= u32((endPoint[1][0].g >> 1u) & 1) << 10u;
    blk.y |= u32((endPoint[1][0].g >> 2u) & 1) << 11u;
    blk.y |= u32((endPoint[1][0].g >> 3u) & 1) << 12u;
    blk.y |= u32((endPoint[0][1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[0][1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[0][1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[0][1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[0][1].g >> 4u) & 1) << 17u;
    blk.y |= u32((endPoint[0][0].g >> 10u) & 1) << 18u;
    blk.y |= u32((endPoint[1][1].g >> 0u) & 1) << 19u;
    blk.y |= u32((endPoint[1][1].g >> 1u) & 1) << 20u;
    blk.y |= u32((endPoint[1][1].g >> 2u) & 1) << 21u;
    blk.y |= u32((endPoint[1][1].g >> 3u) & 1) << 22u;
    blk.y |= u32((endPoint[0][1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[0][1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[0][1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[0][1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[0][0].b >> 10u) & 1) << 27u;
    blk.y |= u32((endPoint[1][1].b >> 1u) & 1) << 28u;
    blk.y |= u32((endPoint[1][0].b >> 0u) & 1) << 29u;
    blk.y |= u32((endPoint[1][0].b >> 1u) & 1) << 30u;
    blk.y |= u32((endPoint[1][0].b >> 2u) & 1) << 31u;
    blk.z |= u32((endPoint[1][0].b >> 3u) & 1) << 0u;
    blk.z |= u32((endPoint[1][0].r >> 0u) & 1) << 1u;
    blk.z |= u32((endPoint[1][0].r >> 1u) & 1) << 2u;
    blk.z |= u32((endPoint[1][0].r >> 2u) & 1) << 3u;
    blk.z |= u32((endPoint[1][0].r >> 3u) & 1) << 4u;
    blk.z |= u32((endPoint[1][1].b >> 0u) & 1) << 5u;
    blk.z |= u32((endPoint[1][1].b >> 2u) & 1) << 6u;
    blk.z |= u32((endPoint[1][1].r >> 0u) & 1) << 7u;
    blk.z |= u32((endPoint[1][1].r >> 1u) & 1) << 8u;
    blk.z |= u32((endPoint[1][1].r >> 2u) & 1) << 9u;
    blk.z |= u32((endPoint[1][1].r >> 3u) & 1) << 10u;
    blk.z |= u32((endPoint[1][0].g >> 4u) & 1) << 11u;
    blk.z |= u32((endPoint[1][1].b >> 3u) & 1) << 12u;
    blk.z |= ((partition_index >> 0u) & 1u) << 13u;
    blk.z |= ((partition_index >> 1u) & 1u) << 14u;
    blk.z |= ((partition_index >> 2u) & 1u) << 15u;
    blk.z |= ((partition_index >> 3u) & 1u) << 16u;
    blk.z |= ((partition_index >> 4u) & 1u) << 17u;
  }
  else if (mode_type == candidateModeFlag[4]) {

    blk.x |= ((candidateModeMemory[4] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[4] >> 1u) & 1u) << 1u;
    blk.x |= ((candidateModeMemory[4] >> 2u) & 1u) << 2u;
    blk.x |= ((candidateModeMemory[4] >> 3u) & 1u) << 3u;
    blk.x |= ((candidateModeMemory[4] >> 4u) & 1u) << 4u;
    blk.x |= u32((endPoint[0][0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0][0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0][0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0][0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0][0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0][0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[0][0].r >> 6u) & 1) << 11u;
    blk.x |= u32((endPoint[0][0].r >> 7u) & 1) << 12u;
    blk.x |= u32((endPoint[0][0].r >> 8u) & 1) << 13u;
    blk.x |= u32((endPoint[0][0].r >> 9u) & 1) << 14u;
    blk.x |= u32((endPoint[0][0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0][0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0][0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0][0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0][0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0][0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[0][0].g >> 6u) & 1) << 21u;
    blk.x |= u32((endPoint[0][0].g >> 7u) & 1) << 22u;
    blk.x |= u32((endPoint[0][0].g >> 8u) & 1) << 23u;
    blk.x |= u32((endPoint[0][0].g >> 9u) & 1) << 24u;
    blk.x |= u32((endPoint[0][0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0][0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0][0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0][0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0][0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0][0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[0][0].b >> 6u) & 1) << 31u;
    blk.y |= u32((endPoint[0][0].b >> 7u) & 1) << 0u;
    blk.y |= u32((endPoint[0][0].b >> 8u) & 1) << 1u;
    blk.y |= u32((endPoint[0][0].b >> 9u) & 1) << 2u;
    blk.y |= u32((endPoint[0][1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[0][1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[0][1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[0][1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[0][0].r >> 10u) & 1) << 7u;
    blk.y |= u32((endPoint[1][0].b >> 4u) & 1) << 8u;
    blk.y |= u32((endPoint[1][0].g >> 0u) & 1) << 9u;
    blk.y |= u32((endPoint[1][0].g >> 1u) & 1) << 10u;
    blk.y |= u32((endPoint[1][0].g >> 2u) & 1) << 11u;
    blk.y |= u32((endPoint[1][0].g >> 3u) & 1) << 12u;
    blk.y |= u32((endPoint[0][1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[0][1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[0][1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[0][1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[0][0].g >> 10u) & 1) << 17u;
    blk.y |= u32((endPoint[1][1].b >> 0u) & 1) << 18u;
    blk.y |= u32((endPoint[1][1].g >> 0u) & 1) << 19u;
    blk.y |= u32((endPoint[1][1].g >> 1u) & 1) << 20u;
    blk.y |= u32((endPoint[1][1].g >> 2u) & 1) << 21u;
    blk.y |= u32((endPoint[1][1].g >> 3u) & 1) << 22u;
    blk.y |= u32((endPoint[0][1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[0][1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[0][1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[0][1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[0][1].b >> 4u) & 1) << 27u;
    blk.y |= u32((endPoint[0][0].b >> 10u) & 1) << 28u;
    blk.y |= u32((endPoint[1][0].b >> 0u) & 1) << 29u;
    blk.y |= u32((endPoint[1][0].b >> 1u) & 1) << 30u;
    blk.y |= u32((endPoint[1][0].b >> 2u) & 1) << 31u;
    blk.z |= u32((endPoint[1][0].b >> 3u) & 1) << 0u;
    blk.z |= u32((endPoint[1][0].r >> 0u) & 1) << 1u;
    blk.z |= u32((endPoint[1][0].r >> 1u) & 1) << 2u;
    blk.z |= u32((endPoint[1][0].r >> 2u) & 1) << 3u;
    blk.z |= u32((endPoint[1][0].r >> 3u) & 1) << 4u;
    blk.z |= u32((endPoint[1][1].b >> 1u) & 1) << 5u;
    blk.z |= u32((endPoint[1][1].b >> 2u) & 1) << 6u;
    blk.z |= u32((endPoint[1][1].r >> 0u) & 1) << 7u;
    blk.z |= u32((endPoint[1][1].r >> 1u) & 1) << 8u;
    blk.z |= u32((endPoint[1][1].r >> 2u) & 1) << 9u;
    blk.z |= u32((endPoint[1][1].r >> 3u) & 1) << 10u;
    blk.z |= u32((endPoint[1][1].b >> 4u) & 1) << 11u;
    blk.z |= u32((endPoint[1][1].b >> 3u) & 1) << 12u;
    blk.z |= ((partition_index >> 0u) & 1u) << 13u;
    blk.z |= ((partition_index >> 1u) & 1u) << 14u;
    blk.z |= ((partition_index >> 2u) & 1u) << 15u;
    blk.z |= ((partition_index >> 3u) & 1u) << 16u;
    blk.z |= ((partition_index >> 4u) & 1u) << 17u;
  }
  else if (mode_type == candidateModeFlag[5]) {

    blk.x |= ((candidateModeMemory[5] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[5] >> 1u) & 1u) << 1u;
    blk.x |= ((candidateModeMemory[5] >> 2u) & 1u) << 2u;
    blk.x |= ((candidateModeMemory[5] >> 3u) & 1u) << 3u;
    blk.x |= ((candidateModeMemory[5] >> 4u) & 1u) << 4u;
    blk.x |= u32((endPoint[0][0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0][0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0][0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0][0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0][0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0][0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[0][0].r >> 6u) & 1) << 11u;
    blk.x |= u32((endPoint[0][0].r >> 7u) & 1) << 12u;
    blk.x |= u32((endPoint[0][0].r >> 8u) & 1) << 13u;
    blk.x |= u32((endPoint[1][0].b >> 4u) & 1) << 14u;
    blk.x |= u32((endPoint[0][0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0][0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0][0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0][0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0][0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0][0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[0][0].g >> 6u) & 1) << 21u;
    blk.x |= u32((endPoint[0][0].g >> 7u) & 1) << 22u;
    blk.x |= u32((endPoint[0][0].g >> 8u) & 1) << 23u;
    blk.x |= u32((endPoint[1][0].g >> 4u) & 1) << 24u;
    blk.x |= u32((endPoint[0][0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0][0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0][0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0][0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0][0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0][0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[0][0].b >> 6u) & 1) << 31u;
    blk.y |= u32((endPoint[0][0].b >> 7u) & 1) << 0u;
    blk.y |= u32((endPoint[0][0].b >> 8u) & 1) << 1u;
    blk.y |= u32((endPoint[1][1].b >> 4u) & 1) << 2u;
    blk.y |= u32((endPoint[0][1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[0][1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[0][1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[0][1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[0][1].r >> 4u) & 1) << 7u;
    blk.y |= u32((endPoint[1][1].g >> 4u) & 1) << 8u;
    blk.y |= u32((endPoint[1][0].g >> 0u) & 1) << 9u;
    blk.y |= u32((endPoint[1][0].g >> 1u) & 1) << 10u;
    blk.y |= u32((endPoint[1][0].g >> 2u) & 1) << 11u;
    blk.y |= u32((endPoint[1][0].g >> 3u) & 1) << 12u;
    blk.y |= u32((endPoint[0][1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[0][1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[0][1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[0][1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[0][1].g >> 4u) & 1) << 17u;
    blk.y |= u32((endPoint[1][1].b >> 0u) & 1) << 18u;
    blk.y |= u32((endPoint[1][1].g >> 0u) & 1) << 19u;
    blk.y |= u32((endPoint[1][1].g >> 1u) & 1) << 20u;
    blk.y |= u32((endPoint[1][1].g >> 2u) & 1) << 21u;
    blk.y |= u32((endPoint[1][1].g >> 3u) & 1) << 22u;
    blk.y |= u32((endPoint[0][1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[0][1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[0][1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[0][1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[0][1].b >> 4u) & 1) << 27u;
    blk.y |= u32((endPoint[1][1].b >> 1u) & 1) << 28u;
    blk.y |= u32((endPoint[1][0].b >> 0u) & 1) << 29u;
    blk.y |= u32((endPoint[1][0].b >> 1u) & 1) << 30u;
    blk.y |= u32((endPoint[1][0].b >> 2u) & 1) << 31u;
    blk.z |= u32((endPoint[1][0].b >> 3u) & 1) << 0u;
    blk.z |= u32((endPoint[1][0].r >> 0u) & 1) << 1u;
    blk.z |= u32((endPoint[1][0].r >> 1u) & 1) << 2u;
    blk.z |= u32((endPoint[1][0].r >> 2u) & 1) << 3u;
    blk.z |= u32((endPoint[1][0].r >> 3u) & 1) << 4u;
    blk.z |= u32((endPoint[1][0].r >> 4u) & 1) << 5u;
    blk.z |= u32((endPoint[1][1].b >> 2u) & 1) << 6u;
    blk.z |= u32((endPoint[1][1].r >> 0u) & 1) << 7u;
    blk.z |= u32((endPoint[1][1].r >> 1u) & 1) << 8u;
    blk.z |= u32((endPoint[1][1].r >> 2u) & 1) << 9u;
    blk.z |= u32((endPoint[1][1].r >> 3u) & 1) << 10u;
    blk.z |= u32((endPoint[1][1].r >> 4u) & 1) << 11u;
    blk.z |= u32((endPoint[1][1].b >> 3u) & 1) << 12u;
    blk.z |= ((partition_index >> 0u) & 1u) << 13u;
    blk.z |= ((partition_index >> 1u) & 1u) << 14u;
    blk.z |= ((partition_index >> 2u) & 1u) << 15u;
    blk.z |= ((partition_index >> 3u) & 1u) << 16u;
    blk.z |= ((partition_index >> 4u) & 1u) << 17u;
  }
  else if (mode_type == candidateModeFlag[6]) {

    blk.x |= ((candidateModeMemory[6] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[6] >> 1u) & 1u) << 1u;
    blk.x |= ((candidateModeMemory[6] >> 2u) & 1u) << 2u;
    blk.x |= ((candidateModeMemory[6] >> 3u) & 1u) << 3u;
    blk.x |= ((candidateModeMemory[6] >> 4u) & 1u) << 4u;
    blk.x |= u32((endPoint[0][0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0][0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0][0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0][0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0][0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0][0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[0][0].r >> 6u) & 1) << 11u;
    blk.x |= u32((endPoint[0][0].r >> 7u) & 1) << 12u;
    blk.x |= u32((endPoint[1][1].g >> 4u) & 1) << 13u;
    blk.x |= u32((endPoint[1][0].b >> 4u) & 1) << 14u;
    blk.x |= u32((endPoint[0][0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0][0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0][0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0][0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0][0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0][0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[0][0].g >> 6u) & 1) << 21u;
    blk.x |= u32((endPoint[0][0].g >> 7u) & 1) << 22u;
    blk.x |= u32((endPoint[1][1].b >> 2u) & 1) << 23u;
    blk.x |= u32((endPoint[1][0].g >> 4u) & 1) << 24u;
    blk.x |= u32((endPoint[0][0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0][0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0][0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0][0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0][0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0][0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[0][0].b >> 6u) & 1) << 31u;
    blk.y |= u32((endPoint[0][0].b >> 7u) & 1) << 0u;
    blk.y |= u32((endPoint[1][1].b >> 3u) & 1) << 1u;
    blk.y |= u32((endPoint[1][1].b >> 4u) & 1) << 2u;
    blk.y |= u32((endPoint[0][1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[0][1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[0][1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[0][1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[0][1].r >> 4u) & 1) << 7u;
    blk.y |= u32((endPoint[0][1].r >> 5u) & 1) << 8u;
    blk.y |= u32((endPoint[1][0].g >> 0u) & 1) << 9u;
    blk.y |= u32((endPoint[1][0].g >> 1u) & 1) << 10u;
    blk.y |= u32((endPoint[1][0].g >> 2u) & 1) << 11u;
    blk.y |= u32((endPoint[1][0].g >> 3u) & 1) << 12u;
    blk.y |= u32((endPoint[0][1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[0][1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[0][1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[0][1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[0][1].g >> 4u) & 1) << 17u;
    blk.y |= u32((endPoint[1][1].b >> 0u) & 1) << 18u;
    blk.y |= u32((endPoint[1][1].g >> 0u) & 1) << 19u;
    blk.y |= u32((endPoint[1][1].g >> 1u) & 1) << 20u;
    blk.y |= u32((endPoint[1][1].g >> 2u) & 1) << 21u;
    blk.y |= u32((endPoint[1][1].g >> 3u) & 1) << 22u;
    blk.y |= u32((endPoint[0][1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[0][1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[0][1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[0][1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[0][1].b >> 4u) & 1) << 27u;
    blk.y |= u32((endPoint[1][1].b >> 1u) & 1) << 28u;
    blk.y |= u32((endPoint[1][0].b >> 0u) & 1) << 29u;
    blk.y |= u32((endPoint[1][0].b >> 1u) & 1) << 30u;
    blk.y |= u32((endPoint[1][0].b >> 2u) & 1) << 31u;
    blk.z |= u32((endPoint[1][0].b >> 3u) & 1) << 0u;
    blk.z |= u32((endPoint[1][0].r >> 0u) & 1) << 1u;
    blk.z |= u32((endPoint[1][0].r >> 1u) & 1) << 2u;
    blk.z |= u32((endPoint[1][0].r >> 2u) & 1) << 3u;
    blk.z |= u32((endPoint[1][0].r >> 3u) & 1) << 4u;
    blk.z |= u32((endPoint[1][0].r >> 4u) & 1) << 5u;
    blk.z |= u32((endPoint[1][0].r >> 5u) & 1) << 6u;
    blk.z |= u32((endPoint[1][1].r >> 0u) & 1) << 7u;
    blk.z |= u32((endPoint[1][1].r >> 1u) & 1) << 8u;
    blk.z |= u32((endPoint[1][1].r >> 2u) & 1) << 9u;
    blk.z |= u32((endPoint[1][1].r >> 3u) & 1) << 10u;
    blk.z |= u32((endPoint[1][1].r >> 4u) & 1) << 11u;
    blk.z |= u32((endPoint[1][1].r >> 5u) & 1) << 12u;
    blk.z |= ((partition_index >> 0u) & 1u) << 13u;
    blk.z |= ((partition_index >> 1u) & 1u) << 14u;
    blk.z |= ((partition_index >> 2u) & 1u) << 15u;
    blk.z |= ((partition_index >> 3u) & 1u) << 16u;
    blk.z |= ((partition_index >> 4u) & 1u) << 17u;
  }
  else if (mode_type == candidateModeFlag[7]) {

    blk.x |= ((candidateModeMemory[7] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[7] >> 1u) & 1u) << 1u;
    blk.x |= ((candidateModeMemory[7] >> 2u) & 1u) << 2u;
    blk.x |= ((candidateModeMemory[7] >> 3u) & 1u) << 3u;
    blk.x |= ((candidateModeMemory[7] >> 4u) & 1u) << 4u;
    blk.x |= u32((endPoint[0][0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0][0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0][0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0][0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0][0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0][0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[0][0].r >> 6u) & 1) << 11u;
    blk.x |= u32((endPoint[0][0].r >> 7u) & 1) << 12u;
    blk.x |= u32((endPoint[1][1].b >> 0u) & 1) << 13u;
    blk.x |= u32((endPoint[1][0].b >> 4u) & 1) << 14u;
    blk.x |= u32((endPoint[0][0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0][0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0][0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0][0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0][0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0][0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[0][0].g >> 6u) & 1) << 21u;
    blk.x |= u32((endPoint[0][0].g >> 7u) & 1) << 22u;
    blk.x |= u32((endPoint[1][0].g >> 5u) & 1) << 23u;
    blk.x |= u32((endPoint[1][0].g >> 4u) & 1) << 24u;
    blk.x |= u32((endPoint[0][0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0][0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0][0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0][0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0][0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0][0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[0][0].b >> 6u) & 1) << 31u;
    blk.y |= u32((endPoint[0][0].b >> 7u) & 1) << 0u;
    blk.y |= u32((endPoint[1][1].g >> 5u) & 1) << 1u;
    blk.y |= u32((endPoint[1][1].b >> 4u) & 1) << 2u;
    blk.y |= u32((endPoint[0][1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[0][1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[0][1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[0][1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[0][1].r >> 4u) & 1) << 7u;
    blk.y |= u32((endPoint[1][1].g >> 4u) & 1) << 8u;
    blk.y |= u32((endPoint[1][0].g >> 0u) & 1) << 9u;
    blk.y |= u32((endPoint[1][0].g >> 1u) & 1) << 10u;
    blk.y |= u32((endPoint[1][0].g >> 2u) & 1) << 11u;
    blk.y |= u32((endPoint[1][0].g >> 3u) & 1) << 12u;
    blk.y |= u32((endPoint[0][1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[0][1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[0][1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[0][1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[0][1].g >> 4u) & 1) << 17u;
    blk.y |= u32((endPoint[0][1].g >> 5u) & 1) << 18u;
    blk.y |= u32((endPoint[1][1].g >> 0u) & 1) << 19u;
    blk.y |= u32((endPoint[1][1].g >> 1u) & 1) << 20u;
    blk.y |= u32((endPoint[1][1].g >> 2u) & 1) << 21u;
    blk.y |= u32((endPoint[1][1].g >> 3u) & 1) << 22u;
    blk.y |= u32((endPoint[0][1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[0][1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[0][1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[0][1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[0][1].b >> 4u) & 1) << 27u;
    blk.y |= u32((endPoint[1][1].b >> 1u) & 1) << 28u;
    blk.y |= u32((endPoint[1][0].b >> 0u) & 1) << 29u;
    blk.y |= u32((endPoint[1][0].b >> 1u) & 1) << 30u;
    blk.y |= u32((endPoint[1][0].b >> 2u) & 1) << 31u;
    blk.z |= u32((endPoint[1][0].b >> 3u) & 1) << 0u;
    blk.z |= u32((endPoint[1][0].r >> 0u) & 1) << 1u;
    blk.z |= u32((endPoint[1][0].r >> 1u) & 1) << 2u;
    blk.z |= u32((endPoint[1][0].r >> 2u) & 1) << 3u;
    blk.z |= u32((endPoint[1][0].r >> 3u) & 1) << 4u;
    blk.z |= u32((endPoint[1][0].r >> 4u) & 1) << 5u;
    blk.z |= u32((endPoint[1][1].b >> 2u) & 1) << 6u;
    blk.z |= u32((endPoint[1][1].r >> 0u) & 1) << 7u;
    blk.z |= u32((endPoint[1][1].r >> 1u) & 1) << 8u;
    blk.z |= u32((endPoint[1][1].r >> 2u) & 1) << 9u;
    blk.z |= u32((endPoint[1][1].r >> 3u) & 1) << 10u;
    blk.z |= u32((endPoint[1][1].r >> 4u) & 1) << 11u;
    blk.z |= u32((endPoint[1][1].b >> 3u) & 1) << 12u;
    blk.z |= ((partition_index >> 0u) & 1u) << 13u;
    blk.z |= ((partition_index >> 1u) & 1u) << 14u;
    blk.z |= ((partition_index >> 2u) & 1u) << 15u;
    blk.z |= ((partition_index >> 3u) & 1u) << 16u;
    blk.z |= ((partition_index >> 4u) & 1u) << 17u;
  }
  else if (mode_type == candidateModeFlag[8]) {

    blk.x |= ((candidateModeMemory[8] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[8] >> 1u) & 1u) << 1u;
    blk.x |= ((candidateModeMemory[8] >> 2u) & 1u) << 2u;
    blk.x |= ((candidateModeMemory[8] >> 3u) & 1u) << 3u;
    blk.x |= ((candidateModeMemory[8] >> 4u) & 1u) << 4u;
    blk.x |= u32((endPoint[0][0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0][0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0][0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0][0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0][0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0][0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[0][0].r >> 6u) & 1) << 11u;
    blk.x |= u32((endPoint[0][0].r >> 7u) & 1) << 12u;
    blk.x |= u32((endPoint[1][1].b >> 1u) & 1) << 13u;
    blk.x |= u32((endPoint[1][0].b >> 4u) & 1) << 14u;
    blk.x |= u32((endPoint[0][0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0][0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0][0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0][0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0][0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0][0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[0][0].g >> 6u) & 1) << 21u;
    blk.x |= u32((endPoint[0][0].g >> 7u) & 1) << 22u;
    blk.x |= u32((endPoint[1][0].b >> 5u) & 1) << 23u;
    blk.x |= u32((endPoint[1][0].g >> 4u) & 1) << 24u;
    blk.x |= u32((endPoint[0][0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0][0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0][0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0][0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0][0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0][0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[0][0].b >> 6u) & 1) << 31u;
    blk.y |= u32((endPoint[0][0].b >> 7u) & 1) << 0u;
    blk.y |= u32((endPoint[1][1].b >> 5u) & 1) << 1u;
    blk.y |= u32((endPoint[1][1].b >> 4u) & 1) << 2u;
    blk.y |= u32((endPoint[0][1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[0][1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[0][1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[0][1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[0][1].r >> 4u) & 1) << 7u;
    blk.y |= u32((endPoint[1][1].g >> 4u) & 1) << 8u;
    blk.y |= u32((endPoint[1][0].g >> 0u) & 1) << 9u;
    blk.y |= u32((endPoint[1][0].g >> 1u) & 1) << 10u;
    blk.y |= u32((endPoint[1][0].g >> 2u) & 1) << 11u;
    blk.y |= u32((endPoint[1][0].g >> 3u) & 1) << 12u;
    blk.y |= u32((endPoint[0][1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[0][1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[0][1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[0][1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[0][1].g >> 4u) & 1) << 17u;
    blk.y |= u32((endPoint[1][1].b >> 0u) & 1) << 18u;
    blk.y |= u32((endPoint[1][1].g >> 0u) & 1) << 19u;
    blk.y |= u32((endPoint[1][1].g >> 1u) & 1) << 20u;
    blk.y |= u32((endPoint[1][1].g >> 2u) & 1) << 21u;
    blk.y |= u32((endPoint[1][1].g >> 3u) & 1) << 22u;
    blk.y |= u32((endPoint[0][1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[0][1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[0][1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[0][1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[0][1].b >> 4u) & 1) << 27u;
    blk.y |= u32((endPoint[0][1].b >> 5u) & 1) << 28u;
    blk.y |= u32((endPoint[1][0].b >> 0u) & 1) << 29u;
    blk.y |= u32((endPoint[1][0].b >> 1u) & 1) << 30u;
    blk.y |= u32((endPoint[1][0].b >> 2u) & 1) << 31u;
    blk.z |= u32((endPoint[1][0].b >> 3u) & 1) << 0u;
    blk.z |= u32((endPoint[1][0].r >> 0u) & 1) << 1u;
    blk.z |= u32((endPoint[1][0].r >> 1u) & 1) << 2u;
    blk.z |= u32((endPoint[1][0].r >> 2u) & 1) << 3u;
    blk.z |= u32((endPoint[1][0].r >> 3u) & 1) << 4u;
    blk.z |= u32((endPoint[1][0].r >> 4u) & 1) << 5u;
    blk.z |= u32((endPoint[1][1].b >> 2u) & 1) << 6u;
    blk.z |= u32((endPoint[1][1].r >> 0u) & 1) << 7u;
    blk.z |= u32((endPoint[1][1].r >> 1u) & 1) << 8u;
    blk.z |= u32((endPoint[1][1].r >> 2u) & 1) << 9u;
    blk.z |= u32((endPoint[1][1].r >> 3u) & 1) << 10u;
    blk.z |= u32((endPoint[1][1].r >> 4u) & 1) << 11u;
    blk.z |= u32((endPoint[1][1].b >> 3u) & 1) << 12u;
    blk.z |= ((partition_index >> 0u) & 1u) << 13u;
    blk.z |= ((partition_index >> 1u) & 1u) << 14u;
    blk.z |= ((partition_index >> 2u) & 1u) << 15u;
    blk.z |= ((partition_index >> 3u) & 1u) << 16u;
    blk.z |= ((partition_index >> 4u) & 1u) << 17u;
  }
  else if (mode_type == candidateModeFlag[9]) {

    blk.x |= ((candidateModeMemory[9] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[9] >> 1u) & 1u) << 1u;
    blk.x |= ((candidateModeMemory[9] >> 2u) & 1u) << 2u;
    blk.x |= ((candidateModeMemory[9] >> 3u) & 1u) << 3u;
    blk.x |= ((candidateModeMemory[9] >> 4u) & 1u) << 4u;
    blk.x |= u32((endPoint[0][0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0][0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0][0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0][0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0][0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0][0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[1][1].g >> 4u) & 1) << 11u;
    blk.x |= u32((endPoint[1][1].b >> 0u) & 1) << 12u;
    blk.x |= u32((endPoint[1][1].b >> 1u) & 1) << 13u;
    blk.x |= u32((endPoint[1][0].b >> 4u) & 1) << 14u;
    blk.x |= u32((endPoint[0][0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0][0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0][0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0][0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0][0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0][0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[1][0].g >> 5u) & 1) << 21u;
    blk.x |= u32((endPoint[1][0].b >> 5u) & 1) << 22u;
    blk.x |= u32((endPoint[1][1].b >> 2u) & 1) << 23u;
    blk.x |= u32((endPoint[1][0].g >> 4u) & 1) << 24u;
    blk.x |= u32((endPoint[0][0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0][0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0][0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0][0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0][0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0][0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[1][1].g >> 5u) & 1) << 31u;
    blk.y |= u32((endPoint[1][1].b >> 3u) & 1) << 0u;
    blk.y |= u32((endPoint[1][1].b >> 5u) & 1) << 1u;
    blk.y |= u32((endPoint[1][1].b >> 4u) & 1) << 2u;
    blk.y |= u32((endPoint[0][1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[0][1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[0][1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[0][1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[0][1].r >> 4u) & 1) << 7u;
    blk.y |= u32((endPoint[0][1].r >> 5u) & 1) << 8u;
    blk.y |= u32((endPoint[1][0].g >> 0u) & 1) << 9u;
    blk.y |= u32((endPoint[1][0].g >> 1u) & 1) << 10u;
    blk.y |= u32((endPoint[1][0].g >> 2u) & 1) << 11u;
    blk.y |= u32((endPoint[1][0].g >> 3u) & 1) << 12u;
    blk.y |= u32((endPoint[0][1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[0][1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[0][1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[0][1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[0][1].g >> 4u) & 1) << 17u;
    blk.y |= u32((endPoint[0][1].g >> 5u) & 1) << 18u;
    blk.y |= u32((endPoint[1][1].g >> 0u) & 1) << 19u;
    blk.y |= u32((endPoint[1][1].g >> 1u) & 1) << 20u;
    blk.y |= u32((endPoint[1][1].g >> 2u) & 1) << 21u;
    blk.y |= u32((endPoint[1][1].g >> 3u) & 1) << 22u;
    blk.y |= u32((endPoint[0][1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[0][1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[0][1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[0][1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[0][1].b >> 4u) & 1) << 27u;
    blk.y |= u32((endPoint[0][1].b >> 5u) & 1) << 28u;
    blk.y |= u32((endPoint[1][0].b >> 0u) & 1) << 29u;
    blk.y |= u32((endPoint[1][0].b >> 1u) & 1) << 30u;
    blk.y |= u32((endPoint[1][0].b >> 2u) & 1) << 31u;
    blk.z |= u32((endPoint[1][0].b >> 3u) & 1) << 0u;
    blk.z |= u32((endPoint[1][0].r >> 0u) & 1) << 1u;
    blk.z |= u32((endPoint[1][0].r >> 1u) & 1) << 2u;
    blk.z |= u32((endPoint[1][0].r >> 2u) & 1) << 3u;
    blk.z |= u32((endPoint[1][0].r >> 3u) & 1) << 4u;
    blk.z |= u32((endPoint[1][0].r >> 4u) & 1) << 5u;
    blk.z |= u32((endPoint[1][0].r >> 5u) & 1) << 6u;
    blk.z |= u32((endPoint[1][1].r >> 0u) & 1) << 7u;
    blk.z |= u32((endPoint[1][1].r >> 1u) & 1) << 8u;
    blk.z |= u32((endPoint[1][1].r >> 2u) & 1) << 9u;
    blk.z |= u32((endPoint[1][1].r >> 3u) & 1) << 10u;
    blk.z |= u32((endPoint[1][1].r >> 4u) & 1) << 11u;
    blk.z |= u32((endPoint[1][1].r >> 5u) & 1) << 12u;
    blk.z |= ((partition_index >> 0u) & 1u) << 13u;
    blk.z |= ((partition_index >> 1u) & 1u) << 14u;
    blk.z |= ((partition_index >> 2u) & 1u) << 15u;
    blk.z |= ((partition_index >> 3u) & 1u) << 16u;
    blk.z |= ((partition_index >> 4u) & 1u) << 17u;
  }
  return blk;
}

// for mode 11 - 14
fn block_package_one(blk_in: vec4<u32>, endPoint: mat2x3i, mode_type: u32) -> vec4<u32> {
  var blk = blk_in;
  blk.x = 0u;
  blk.y = 0u;
  blk.z &= 0xFFFFFFFEu;

  if (mode_type == candidateModeFlag[10]) {

    blk.x |= ((candidateModeMemory[10] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[10] >> 1u) & 1u) << 1u;
    blk.x |= ((candidateModeMemory[10] >> 2u) & 1u) << 2u;
    blk.x |= ((candidateModeMemory[10] >> 3u) & 1u) << 3u;
    blk.x |= ((candidateModeMemory[10] >> 4u) & 1u) << 4u;
    blk.x |= u32((endPoint[0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[0].r >> 6u) & 1) << 11u;
    blk.x |= u32((endPoint[0].r >> 7u) & 1) << 12u;
    blk.x |= u32((endPoint[0].r >> 8u) & 1) << 13u;
    blk.x |= u32((endPoint[0].r >> 9u) & 1) << 14u;
    blk.x |= u32((endPoint[0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[0].g >> 6u) & 1) << 21u;
    blk.x |= u32((endPoint[0].g >> 7u) & 1) << 22u;
    blk.x |= u32((endPoint[0].g >> 8u) & 1) << 23u;
    blk.x |= u32((endPoint[0].g >> 9u) & 1) << 24u;
    blk.x |= u32((endPoint[0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[0].b >> 6u) & 1) << 31u;
    blk.y |= u32((endPoint[0].b >> 7u) & 1) << 0u;
    blk.y |= u32((endPoint[0].b >> 8u) & 1) << 1u;
    blk.y |= u32((endPoint[0].b >> 9u) & 1) << 2u;
    blk.y |= u32((endPoint[1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[1].r >> 4u) & 1) << 7u;
    blk.y |= u32((endPoint[1].r >> 5u) & 1) << 8u;
    blk.y |= u32((endPoint[1].r >> 6u) & 1) << 9u;
    blk.y |= u32((endPoint[1].r >> 7u) & 1) << 10u;
    blk.y |= u32((endPoint[1].r >> 8u) & 1) << 11u;
    blk.y |= u32((endPoint[1].r >> 9u) & 1) << 12u;
    blk.y |= u32((endPoint[1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[1].g >> 4u) & 1) << 17u;
    blk.y |= u32((endPoint[1].g >> 5u) & 1) << 18u;
    blk.y |= u32((endPoint[1].g >> 6u) & 1) << 19u;
    blk.y |= u32((endPoint[1].g >> 7u) & 1) << 20u;
    blk.y |= u32((endPoint[1].g >> 8u) & 1) << 21u;
    blk.y |= u32((endPoint[1].g >> 9u) & 1) << 22u;
    blk.y |= u32((endPoint[1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[1].b >> 4u) & 1) << 27u;
    blk.y |= u32((endPoint[1].b >> 5u) & 1) << 28u;
    blk.y |= u32((endPoint[1].b >> 6u) & 1) << 29u;
    blk.y |= u32((endPoint[1].b >> 7u) & 1) << 30u;
    blk.y |= u32((endPoint[1].b >> 8u) & 1) << 31u;
    blk.z |= u32((endPoint[1].b >> 9u) & 1) << 0u;
  }
  else if (mode_type == candidateModeFlag[11]) {

    blk.x |= ((candidateModeMemory[11] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[11] >> 1u) & 1u) << 1u;
    blk.x |= ((candidateModeMemory[11] >> 2u) & 1u) << 2u;
    blk.x |= ((candidateModeMemory[11] >> 3u) & 1u) << 3u;
    blk.x |= ((candidateModeMemory[11] >> 4u) & 1u) << 4u;
    blk.x |= u32((endPoint[0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[0].r >> 6u) & 1) << 11u;
    blk.x |= u32((endPoint[0].r >> 7u) & 1) << 12u;
    blk.x |= u32((endPoint[0].r >> 8u) & 1) << 13u;
    blk.x |= u32((endPoint[0].r >> 9u) & 1) << 14u;
    blk.x |= u32((endPoint[0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[0].g >> 6u) & 1) << 21u;
    blk.x |= u32((endPoint[0].g >> 7u) & 1) << 22u;
    blk.x |= u32((endPoint[0].g >> 8u) & 1) << 23u;
    blk.x |= u32((endPoint[0].g >> 9u) & 1) << 24u;
    blk.x |= u32((endPoint[0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[0].b >> 6u) & 1) << 31u;
    blk.y |= u32((endPoint[0].b >> 7u) & 1) << 0u;
    blk.y |= u32((endPoint[0].b >> 8u) & 1) << 1u;
    blk.y |= u32((endPoint[0].b >> 9u) & 1) << 2u;
    blk.y |= u32((endPoint[1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[1].r >> 4u) & 1) << 7u;
    blk.y |= u32((endPoint[1].r >> 5u) & 1) << 8u;
    blk.y |= u32((endPoint[1].r >> 6u) & 1) << 9u;
    blk.y |= u32((endPoint[1].r >> 7u) & 1) << 10u;
    blk.y |= u32((endPoint[1].r >> 8u) & 1) << 11u;
    blk.y |= u32((endPoint[0].r >> 10u) & 1) << 12u;
    blk.y |= u32((endPoint[1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[1].g >> 4u) & 1) << 17u;
    blk.y |= u32((endPoint[1].g >> 5u) & 1) << 18u;
    blk.y |= u32((endPoint[1].g >> 6u) & 1) << 19u;
    blk.y |= u32((endPoint[1].g >> 7u) & 1) << 20u;
    blk.y |= u32((endPoint[1].g >> 8u) & 1) << 21u;
    blk.y |= u32((endPoint[0].g >> 10u) & 1) << 22u;
    blk.y |= u32((endPoint[1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[1].b >> 4u) & 1) << 27u;
    blk.y |= u32((endPoint[1].b >> 5u) & 1) << 28u;
    blk.y |= u32((endPoint[1].b >> 6u) & 1) << 29u;
    blk.y |= u32((endPoint[1].b >> 7u) & 1) << 30u;
    blk.y |= u32((endPoint[1].b >> 8u) & 1) << 31u;
    blk.z |= u32((endPoint[0].b >> 10u) & 1) << 0u;
  }
  else if (mode_type == candidateModeFlag[12]) {  // violate the spec in  [0].low

    blk.x |= ((candidateModeMemory[12] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[12] >> 1u) & 1u) << 1u;
    blk.x |= ((candidateModeMemory[12] >> 2u) & 1u) << 2u;
    blk.x |= ((candidateModeMemory[12] >> 3u) & 1u) << 3u;
    blk.x |= ((candidateModeMemory[12] >> 4u) & 1u) << 4u;
    blk.x |= u32((endPoint[0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[0].r >> 6u) & 1) << 11u;
    blk.x |= u32((endPoint[0].r >> 7u) & 1) << 12u;
    blk.x |= u32((endPoint[0].r >> 8u) & 1) << 13u;
    blk.x |= u32((endPoint[0].r >> 9u) & 1) << 14u;
    blk.x |= u32((endPoint[0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[0].g >> 6u) & 1) << 21u;
    blk.x |= u32((endPoint[0].g >> 7u) & 1) << 22u;
    blk.x |= u32((endPoint[0].g >> 8u) & 1) << 23u;
    blk.x |= u32((endPoint[0].g >> 9u) & 1) << 24u;
    blk.x |= u32((endPoint[0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[0].b >> 6u) & 1) << 31u;
    blk.y |= u32((endPoint[0].b >> 7u) & 1) << 0u;
    blk.y |= u32((endPoint[0].b >> 8u) & 1) << 1u;
    blk.y |= u32((endPoint[0].b >> 9u) & 1) << 2u;
    blk.y |= u32((endPoint[1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[1].r >> 4u) & 1) << 7u;
    blk.y |= u32((endPoint[1].r >> 5u) & 1) << 8u;
    blk.y |= u32((endPoint[1].r >> 6u) & 1) << 9u;
    blk.y |= u32((endPoint[1].r >> 7u) & 1) << 10u;
    blk.y |= u32((endPoint[0].r >> 11u) & 1) << 11u;
    blk.y |= u32((endPoint[0].r >> 10u) & 1) << 12u;
    blk.y |= u32((endPoint[1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[1].g >> 4u) & 1) << 17u;
    blk.y |= u32((endPoint[1].g >> 5u) & 1) << 18u;
    blk.y |= u32((endPoint[1].g >> 6u) & 1) << 19u;
    blk.y |= u32((endPoint[1].g >> 7u) & 1) << 20u;
    blk.y |= u32((endPoint[0].g >> 11u) & 1) << 21u;
    blk.y |= u32((endPoint[0].g >> 10u) & 1) << 22u;
    blk.y |= u32((endPoint[1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[1].b >> 4u) & 1) << 27u;
    blk.y |= u32((endPoint[1].b >> 5u) & 1) << 28u;
    blk.y |= u32((endPoint[1].b >> 6u) & 1) << 29u;
    blk.y |= u32((endPoint[1].b >> 7u) & 1) << 30u;
    blk.y |= u32((endPoint[0].b >> 11u) & 1) << 31u;
    blk.z |= u32((endPoint[0].b >> 10u) & 1) << 0u;
  }
  else if (mode_type == candidateModeFlag[13]) {

    blk.x |= ((candidateModeMemory[13] >> 0u) & 1u) << 0u;
    blk.x |= ((candidateModeMemory[13] >> 1u) & 1u) << 1u;
    blk.x |= ((candidateModeMemory[13] >> 2u) & 1u) << 2u;
    blk.x |= ((candidateModeMemory[13] >> 3u) & 1u) << 3u;
    blk.x |= ((candidateModeMemory[13] >> 4u) & 1u) << 4u;
    blk.x |= u32((endPoint[0].r >> 0u) & 1) << 5u;
    blk.x |= u32((endPoint[0].r >> 1u) & 1) << 6u;
    blk.x |= u32((endPoint[0].r >> 2u) & 1) << 7u;
    blk.x |= u32((endPoint[0].r >> 3u) & 1) << 8u;
    blk.x |= u32((endPoint[0].r >> 4u) & 1) << 9u;
    blk.x |= u32((endPoint[0].r >> 5u) & 1) << 10u;
    blk.x |= u32((endPoint[0].r >> 6u) & 1) << 11u;
    blk.x |= u32((endPoint[0].r >> 7u) & 1) << 12u;
    blk.x |= u32((endPoint[0].r >> 8u) & 1) << 13u;
    blk.x |= u32((endPoint[0].r >> 9u) & 1) << 14u;
    blk.x |= u32((endPoint[0].g >> 0u) & 1) << 15u;
    blk.x |= u32((endPoint[0].g >> 1u) & 1) << 16u;
    blk.x |= u32((endPoint[0].g >> 2u) & 1) << 17u;
    blk.x |= u32((endPoint[0].g >> 3u) & 1) << 18u;
    blk.x |= u32((endPoint[0].g >> 4u) & 1) << 19u;
    blk.x |= u32((endPoint[0].g >> 5u) & 1) << 20u;
    blk.x |= u32((endPoint[0].g >> 6u) & 1) << 21u;
    blk.x |= u32((endPoint[0].g >> 7u) & 1) << 22u;
    blk.x |= u32((endPoint[0].g >> 8u) & 1) << 23u;
    blk.x |= u32((endPoint[0].g >> 9u) & 1) << 24u;
    blk.x |= u32((endPoint[0].b >> 0u) & 1) << 25u;
    blk.x |= u32((endPoint[0].b >> 1u) & 1) << 26u;
    blk.x |= u32((endPoint[0].b >> 2u) & 1) << 27u;
    blk.x |= u32((endPoint[0].b >> 3u) & 1) << 28u;
    blk.x |= u32((endPoint[0].b >> 4u) & 1) << 29u;
    blk.x |= u32((endPoint[0].b >> 5u) & 1) << 30u;
    blk.x |= u32((endPoint[0].b >> 6u) & 1) << 31u;
    blk.y |= u32((endPoint[0].b >> 7u) & 1) << 0u;
    blk.y |= u32((endPoint[0].b >> 8u) & 1) << 1u;
    blk.y |= u32((endPoint[0].b >> 9u) & 1) << 2u;
    blk.y |= u32((endPoint[1].r >> 0u) & 1) << 3u;
    blk.y |= u32((endPoint[1].r >> 1u) & 1) << 4u;
    blk.y |= u32((endPoint[1].r >> 2u) & 1) << 5u;
    blk.y |= u32((endPoint[1].r >> 3u) & 1) << 6u;
    blk.y |= u32((endPoint[0].r >> 15u) & 1) << 7u;
    blk.y |= u32((endPoint[0].r >> 14u) & 1) << 8u;
    blk.y |= u32((endPoint[0].r >> 13u) & 1) << 9u;
    blk.y |= u32((endPoint[0].r >> 12u) & 1) << 10u;
    blk.y |= u32((endPoint[0].r >> 11u) & 1) << 11u;
    blk.y |= u32((endPoint[0].r >> 10u) & 1) << 12u;
    blk.y |= u32((endPoint[1].g >> 0u) & 1) << 13u;
    blk.y |= u32((endPoint[1].g >> 1u) & 1) << 14u;
    blk.y |= u32((endPoint[1].g >> 2u) & 1) << 15u;
    blk.y |= u32((endPoint[1].g >> 3u) & 1) << 16u;
    blk.y |= u32((endPoint[0].g >> 15u) & 1) << 17u;
    blk.y |= u32((endPoint[0].g >> 14u) & 1) << 18u;
    blk.y |= u32((endPoint[0].g >> 13u) & 1) << 19u;
    blk.y |= u32((endPoint[0].g >> 12u) & 1) << 20u;
    blk.y |= u32((endPoint[0].g >> 11u) & 1) << 21u;
    blk.y |= u32((endPoint[0].g >> 10u) & 1) << 22u;
    blk.y |= u32((endPoint[1].b >> 0u) & 1) << 23u;
    blk.y |= u32((endPoint[1].b >> 1u) & 1) << 24u;
    blk.y |= u32((endPoint[1].b >> 2u) & 1) << 25u;
    blk.y |= u32((endPoint[1].b >> 3u) & 1) << 26u;
    blk.y |= u32((endPoint[0].b >> 15u) & 1) << 27u;
    blk.y |= u32((endPoint[0].b >> 14u) & 1) << 28u;
    blk.y |= u32((endPoint[0].b >> 13u) & 1) << 29u;
    blk.y |= u32((endPoint[0].b >> 12u) & 1) << 30u;
    blk.y |= u32((endPoint[0].b >> 11u) & 1) << 31u;
    blk.z |= u32((endPoint[0].b >> 10u) & 1) << 0u;
  }
  return blk;
}
