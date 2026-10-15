// BC7 GPU encoder — faithful WGSL port of DirectXTex/Shaders/BC7Encode.hlsl.
//
// Four compute entry points run as a pipeline driven by compressBC.ts:
//   TryMode456CS -> TryMode137CS -> TryMode02CS -> EncodeBlockCS
// using ping-pong error buffers (g_InBuff/g_OutBuff). The HLSL `REF_DEVICE` path
// is the one ported (barriers on, no early-out), matching DirectXTex's shipped
// compiled shaders.
//
// Translation notes:
//   * HLSL uint2x4 (2x4 integer matrix) -> array<vec4<u32>, 2>  (mat2x4u).
//   * swizzle writes (pixel.ra = pixel.ar) are component swaps, expanded inline.
//   * vector shifts require a vector shift amount: v >> vec4<u32>(6u).
//   * integer dot products use explicit helpers (dotNi/dotNu).

alias mat2x4u = array<vec4<u32>, 2>;

const MAX_UINT: u32 = 0xFFFFFFFFu;
const MIN_UINT: u32 = 0u;

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
  g_alpha_weight: f32,
  g_reserved: u32,
};

@group(0) @binding(0) var<uniform> params: Constants;
@group(0) @binding(1) var g_Input: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> g_InBuff: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> g_OutBuff: array<vec4<u32>>;

struct BufferShared {
  pixel: vec4<u32>,
  error: u32,
  mode: u32,
  part: u32,
  index_selector: u32,
  rotation: u32,
  endPoint_low: vec4<u32>,
  endPoint_high: vec4<u32>,
  endPoint_low_quantized: vec4<u32>,
  endPoint_high_quantized: vec4<u32>,
};
var<workgroup> shared_temp: array<BufferShared, THREAD_GROUP_SIZE>;

//------------------------------------------------------------------------------
// Constant tables
//------------------------------------------------------------------------------
const candidateSectionBit = array<u32, 64>(
  0xCCCCu, 0x8888u, 0xEEEEu, 0xECC8u, 0xC880u, 0xFEECu, 0xFEC8u, 0xEC80u,
  0xC800u, 0xFFECu, 0xFE80u, 0xE800u, 0xFFE8u, 0xFF00u, 0xFFF0u, 0xF000u,
  0xF710u, 0x008Eu, 0x7100u, 0x08CEu, 0x008Cu, 0x7310u, 0x3100u, 0x8CCEu,
  0x088Cu, 0x3110u, 0x6666u, 0x366Cu, 0x17E8u, 0x0FF0u, 0x718Eu, 0x399Cu,
  0xaaaau, 0xf0f0u, 0x5a5au, 0x33ccu, 0x3c3cu, 0x55aau, 0x9696u, 0xa55au,
  0x73ceu, 0x13c8u, 0x324cu, 0x3bdcu, 0x6996u, 0xc33cu, 0x9966u, 0x660u,
  0x272u, 0x4e4u, 0x4e40u, 0x2720u, 0xc936u, 0x936cu, 0x39c6u, 0x639cu,
  0x9336u, 0x9cc6u, 0x817eu, 0xe718u, 0xccf0u, 0xfccu, 0x7744u, 0xee22u,
);

const candidateSectionBit2 = array<u32, 64>(
  0xaa685050u, 0x6a5a5040u, 0x5a5a4200u, 0x5450a0a8u, 0xa5a50000u, 0xa0a05050u, 0x5555a0a0u, 0x5a5a5050u,
  0xaa550000u, 0xaa555500u, 0xaaaa5500u, 0x90909090u, 0x94949494u, 0xa4a4a4a4u, 0xa9a59450u, 0x2a0a4250u,
  0xa5945040u, 0x0a425054u, 0xa5a5a500u, 0x55a0a0a0u, 0xa8a85454u, 0x6a6a4040u, 0xa4a45000u, 0x1a1a0500u,
  0x0050a4a4u, 0xaaa59090u, 0x14696914u, 0x69691400u, 0xa08585a0u, 0xaa821414u, 0x50a4a450u, 0x6a5a0200u,
  0xa9a58000u, 0x5090a0a8u, 0xa8a09050u, 0x24242424u, 0x00aa5500u, 0x24924924u, 0x24499224u, 0x50a50a50u,
  0x500aa550u, 0xaaaa4444u, 0x66660000u, 0xa5a0a5a0u, 0x50a050a0u, 0x69286928u, 0x44aaaa44u, 0x66666600u,
  0xaa444444u, 0x54a854a8u, 0x95809580u, 0x96969600u, 0xa85454a8u, 0x80959580u, 0xaa141414u, 0x96960000u,
  0xaaaa1414u, 0xa05050a0u, 0xa0a5a5a0u, 0x96000000u, 0x40804080u, 0xa9a8a9a8u, 0xaaaaaa44u, 0x2a4a5254u,
);

const candidateFixUpIndex1D = array<vec2<u32>, 128>(
  vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),
  vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),
  vec2(15u,0u),vec2(2u,0u),vec2(8u,0u),vec2(2u,0u),vec2(2u,0u),vec2(8u,0u),vec2(8u,0u),vec2(15u,0u),
  vec2(2u,0u),vec2(8u,0u),vec2(2u,0u),vec2(2u,0u),vec2(8u,0u),vec2(8u,0u),vec2(2u,0u),vec2(2u,0u),
  vec2(15u,0u),vec2(15u,0u),vec2(6u,0u),vec2(8u,0u),vec2(2u,0u),vec2(8u,0u),vec2(15u,0u),vec2(15u,0u),
  vec2(2u,0u),vec2(8u,0u),vec2(2u,0u),vec2(2u,0u),vec2(2u,0u),vec2(15u,0u),vec2(15u,0u),vec2(6u,0u),
  vec2(6u,0u),vec2(2u,0u),vec2(6u,0u),vec2(8u,0u),vec2(15u,0u),vec2(15u,0u),vec2(2u,0u),vec2(2u,0u),
  vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(2u,0u),vec2(2u,0u),vec2(15u,0u),
  vec2(3u,15u),vec2(3u,8u),vec2(15u,8u),vec2(15u,3u),vec2(8u,15u),vec2(3u,15u),vec2(15u,3u),vec2(15u,8u),
  vec2(8u,15u),vec2(8u,15u),vec2(6u,15u),vec2(6u,15u),vec2(6u,15u),vec2(5u,15u),vec2(3u,15u),vec2(3u,8u),
  vec2(3u,15u),vec2(3u,8u),vec2(8u,15u),vec2(15u,3u),vec2(3u,15u),vec2(3u,8u),vec2(6u,15u),vec2(10u,8u),
  vec2(5u,3u),vec2(8u,15u),vec2(8u,6u),vec2(6u,10u),vec2(8u,15u),vec2(5u,15u),vec2(15u,10u),vec2(15u,8u),
  vec2(8u,15u),vec2(15u,3u),vec2(3u,15u),vec2(5u,10u),vec2(6u,10u),vec2(10u,8u),vec2(8u,9u),vec2(15u,10u),
  vec2(15u,6u),vec2(3u,15u),vec2(15u,8u),vec2(5u,15u),vec2(15u,3u),vec2(15u,6u),vec2(15u,6u),vec2(15u,8u),
  vec2(3u,15u),vec2(15u,3u),vec2(5u,15u),vec2(5u,15u),vec2(5u,15u),vec2(8u,15u),vec2(5u,15u),vec2(10u,15u),
  vec2(5u,15u),vec2(10u,15u),vec2(8u,15u),vec2(13u,15u),vec2(15u,3u),vec2(12u,15u),vec2(3u,15u),vec2(3u,8u),
);

const candidateFixUpIndex1DOrdered = array<vec2<u32>, 128>(
  vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),
  vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),
  vec2(15u,0u),vec2(2u,0u),vec2(8u,0u),vec2(2u,0u),vec2(2u,0u),vec2(8u,0u),vec2(8u,0u),vec2(15u,0u),
  vec2(2u,0u),vec2(8u,0u),vec2(2u,0u),vec2(2u,0u),vec2(8u,0u),vec2(8u,0u),vec2(2u,0u),vec2(2u,0u),
  vec2(15u,0u),vec2(15u,0u),vec2(6u,0u),vec2(8u,0u),vec2(2u,0u),vec2(8u,0u),vec2(15u,0u),vec2(15u,0u),
  vec2(2u,0u),vec2(8u,0u),vec2(2u,0u),vec2(2u,0u),vec2(2u,0u),vec2(15u,0u),vec2(15u,0u),vec2(6u,0u),
  vec2(6u,0u),vec2(2u,0u),vec2(6u,0u),vec2(8u,0u),vec2(15u,0u),vec2(15u,0u),vec2(2u,0u),vec2(2u,0u),
  vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(15u,0u),vec2(2u,0u),vec2(2u,0u),vec2(15u,0u),
  vec2(3u,15u),vec2(3u,8u),vec2(8u,15u),vec2(3u,15u),vec2(8u,15u),vec2(3u,15u),vec2(3u,15u),vec2(8u,15u),
  vec2(8u,15u),vec2(8u,15u),vec2(6u,15u),vec2(6u,15u),vec2(6u,15u),vec2(5u,15u),vec2(3u,15u),vec2(3u,8u),
  vec2(3u,15u),vec2(3u,8u),vec2(8u,15u),vec2(3u,15u),vec2(3u,15u),vec2(3u,8u),vec2(6u,15u),vec2(8u,10u),
  vec2(3u,5u),vec2(8u,15u),vec2(6u,8u),vec2(6u,10u),vec2(8u,15u),vec2(5u,15u),vec2(10u,15u),vec2(8u,15u),
  vec2(8u,15u),vec2(3u,15u),vec2(3u,15u),vec2(5u,10u),vec2(6u,10u),vec2(8u,10u),vec2(8u,9u),vec2(10u,15u),
  vec2(6u,15u),vec2(3u,15u),vec2(8u,15u),vec2(5u,15u),vec2(3u,15u),vec2(6u,15u),vec2(6u,15u),vec2(8u,15u),
  vec2(3u,15u),vec2(3u,15u),vec2(5u,15u),vec2(5u,15u),vec2(5u,15u),vec2(8u,15u),vec2(5u,15u),vec2(10u,15u),
  vec2(5u,15u),vec2(10u,15u),vec2(8u,15u),vec2(13u,15u),vec2(3u,15u),vec2(12u,15u),vec2(3u,15u),vec2(3u,8u),
);

const aWeight = array<array<u32, 16>, 3>(
  array<u32,16>(0u,4u,9u,13u,17u,21u,26u,30u,34u,38u,43u,47u,51u,55u,60u,64u),
  array<u32,16>(0u,9u,18u,27u,37u,46u,55u,64u,0u,0u,0u,0u,0u,0u,0u,0u),
  array<u32,16>(0u,21u,43u,64u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u),
);

const aStep = array<array<u32, 64>, 3>(
  array<u32,64>(
    0u,0u,0u,1u,1u,1u,1u,2u, 2u,2u,2u,2u,3u,3u,3u,3u,
    4u,4u,4u,4u,5u,5u,5u,5u, 6u,6u,6u,6u,6u,7u,7u,7u,
    7u,8u,8u,8u,8u,9u,9u,9u, 9u,10u,10u,10u,10u,10u,11u,11u,
    11u,11u,12u,12u,12u,12u,13u,13u, 13u,13u,14u,14u,14u,14u,15u,15u),
  array<u32,64>(
    0u,0u,0u,0u,0u,1u,1u,1u, 1u,1u,1u,1u,1u,1u,2u,2u,
    2u,2u,2u,2u,2u,2u,2u,3u, 3u,3u,3u,3u,3u,3u,3u,3u,
    3u,4u,4u,4u,4u,4u,4u,4u, 4u,4u,5u,5u,5u,5u,5u,5u,
    5u,5u,5u,6u,6u,6u,6u,6u, 6u,6u,6u,6u,7u,7u,7u,7u),
  array<u32,64>(
    0u,0u,0u,0u,0u,0u,0u,0u, 0u,0u,0u,1u,1u,1u,1u,1u,
    1u,1u,1u,1u,1u,1u,1u,1u, 1u,1u,1u,1u,1u,1u,1u,1u,
    1u,2u,2u,2u,2u,2u,2u,2u, 2u,2u,2u,2u,2u,2u,2u,2u,
    2u,2u,2u,2u,2u,2u,3u,3u, 3u,3u,3u,3u,3u,3u,3u,3u),
);

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------
fn dot3i(a: vec3<i32>, b: vec3<i32>) -> i32 { return a.x * b.x + a.y * b.y + a.z * b.z; }
fn dot4i(a: vec4<i32>, b: vec4<i32>) -> i32 { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
fn dot3u(a: vec3<u32>, b: vec3<u32>) -> u32 { return a.x * b.x + a.y * b.y + a.z * b.z; }

fn ComputeError(a: vec4<u32>, b: vec4<u32>) -> u32 {
  return u32(f32(dot3u(a.xyz, b.xyz)) + params.g_alpha_weight * f32(a.w * b.w));
}

// Componentwise: make `a` hold the larger of (a,b) per channel, `b` the smaller.
fn Ensure_A_Is_Larger(a: vec4<u32>, b: vec4<u32>) -> array<vec4<u32>, 2> {
  let hi = max(a, b);
  let lo = min(a, b);
  return array<vec4<u32>, 2>(hi, lo);
}

fn quantize(color: vec4<u32>, uPrec: u32) -> vec4<u32> {
  let scaled = (color << vec4<u32>(8u)) + color;
  return (scaled * ((1u << uPrec) - 1u) + vec4<u32>(32768u)) >> vec4<u32>(16u);
}

fn unquantize(color: vec4<u32>, uPrec: u32) -> vec4<u32> {
  let c = color << vec4<u32>(8u - uPrec);
  return c | (c >> vec4<u32>(uPrec));
}

fn loadPixel(base_x: u32, base_y: u32, t: u32) -> vec4<u32> {
  let texel = textureLoad(g_Input, vec2<u32>(base_x + t % 4u, base_y + t / 4u), 0);
  return clamp(vec4<u32>(texel * 255.0), vec4<u32>(0u), vec4<u32>(255u));
}

// Reconstruct a pixel from interpolated endpoints (rgb path), uint domain.
fn interp(epLow: vec4<u32>, epHigh: vec4<u32>, w: u32) -> vec4<u32> {
  return ((64u - w) * epLow + w * epHigh + vec4<u32>(32u)) >> vec4<u32>(6u);
}

//------------------------------------------------------------------------------
// compress_endpoints* — quantize endpoints per mode, return quantized, modify ep
//------------------------------------------------------------------------------
fn compress_endpoints0(endPoint: ptr<function, mat2x4u>, P: vec2<u32>) -> mat2x4u {
  var quantized: mat2x4u;
  for (var j = 0u; j < 2u; j++) {
    let ep = (*endPoint)[j];
    var q = quantize(ep.xyzz, 5u) & vec4<u32>(0xFFFFFFFEu);
    q.x |= P[j]; q.y |= P[j]; q.z |= P[j];
    q.w = 0xFFu;
    var e = unquantize(vec4<u32>(q.xyz, q.z), 5u);
    e.w = 0xFFu;
    (*endPoint)[j] = vec4<u32>(e.xyz, 0xFFu);
    quantized[j] = q << vec4<u32>(3u);
  }
  return quantized;
}

fn compress_endpoints1(endPoint: ptr<function, mat2x4u>, P: vec2<u32>) -> mat2x4u {
  var quantized: mat2x4u;
  for (var j = 0u; j < 2u; j++) {
    let ep = (*endPoint)[j];
    var q = quantize(ep.xyzz, 7u) & vec4<u32>(0xFFFFFFFEu);
    q.x |= P[j]; q.y |= P[j]; q.z |= P[j];
    q.w = 0xFFu;
    var e = unquantize(vec4<u32>(q.xyz, q.z), 7u);
    (*endPoint)[j] = vec4<u32>(e.xyz, 0xFFu);
    quantized[j] = q << vec4<u32>(1u);
  }
  return quantized;
}

fn compress_endpoints2(endPoint: ptr<function, mat2x4u>) -> mat2x4u {
  var quantized: mat2x4u;
  for (var j = 0u; j < 2u; j++) {
    let ep = (*endPoint)[j];
    var q = quantize(ep.xyzz, 5u);
    q.w = 0xFFu;
    var e = unquantize(vec4<u32>(q.xyz, q.z), 5u);
    (*endPoint)[j] = vec4<u32>(e.xyz, 0xFFu);
    quantized[j] = q << vec4<u32>(3u);
  }
  return quantized;
}

fn compress_endpoints3(endPoint: ptr<function, mat2x4u>, P: vec2<u32>) -> mat2x4u {
  var quantized: mat2x4u;
  for (var j = 0u; j < 2u; j++) {
    let ep = (*endPoint)[j];
    var q = ep & vec4<u32>(0xFFFFFFFEu);
    q.x |= P[j]; q.y |= P[j]; q.z |= P[j];
    q.w = 0xFFu;
    (*endPoint)[j] = vec4<u32>(q.xyz, 0xFFu);
    quantized[j] = q;
  }
  return quantized;
}

fn compress_endpoints4(endPoint: ptr<function, mat2x4u>) -> mat2x4u {
  var quantized: mat2x4u;
  for (var j = 0u; j < 2u; j++) {
    let ep = (*endPoint)[j];
    var qrgb = quantize(ep.xyzz, 5u).xyz;
    let qa = quantize(vec4<u32>(ep.w), 6u).x;
    let ergb = unquantize(vec4<u32>(qrgb, qrgb.z), 5u).xyz;
    let ea = unquantize(vec4<u32>(qa), 6u).x;
    (*endPoint)[j] = vec4<u32>(ergb, ea);
    quantized[j] = vec4<u32>(qrgb << vec3<u32>(3u), qa << 2u);
  }
  return quantized;
}

fn compress_endpoints5(endPoint: ptr<function, mat2x4u>) -> mat2x4u {
  var quantized: mat2x4u;
  for (var j = 0u; j < 2u; j++) {
    let ep = (*endPoint)[j];
    let qrgb = quantize(ep.xyzz, 7u).xyz;
    let qa = ep.w;
    let ergb = unquantize(vec4<u32>(qrgb, qrgb.z), 7u).xyz;
    (*endPoint)[j] = vec4<u32>(ergb, ep.w);
    quantized[j] = vec4<u32>(qrgb << vec3<u32>(1u), qa);
  }
  return quantized;
}

fn compress_endpoints6(endPoint: ptr<function, mat2x4u>, P: vec2<u32>) -> mat2x4u {
  var quantized: mat2x4u;
  for (var j = 0u; j < 2u; j++) {
    var q = (*endPoint)[j] & vec4<u32>(0xFFFFFFFEu);
    q |= vec4<u32>(P[j]);
    (*endPoint)[j] = q;
    quantized[j] = q;
  }
  return quantized;
}

fn compress_endpoints7(endPoint: ptr<function, mat2x4u>, P: vec2<u32>) -> mat2x4u {
  var quantized: mat2x4u;
  for (var j = 0u; j < 2u; j++) {
    var q = quantize((*endPoint)[j], 6u) & vec4<u32>(0xFFFFFFFEu);
    q |= vec4<u32>(P[j]);
    (*endPoint)[j] = unquantize(q, 6u);
    quantized[j] = q << vec4<u32>(2u);
  }
  return quantized;
}

//------------------------------------------------------------------------------
// block_package* accessors (HLSL #define macros)
//------------------------------------------------------------------------------
fn ep_l(tb: u32, s: u32) -> vec4<u32> { return shared_temp[tb + s].endPoint_low_quantized; }
fn ep_h(tb: u32, s: u32) -> vec4<u32> { return shared_temp[tb + s].endPoint_high_quantized; }
fn cidx(tb: u32, i: u32) -> u32 { return shared_temp[tb + i].error; }
fn aidx(tb: u32, i: u32) -> u32 { return shared_temp[tb + i].mode; }

fn block_package0(part: u32, tb: u32) -> vec4<u32> {
  var bx: u32 = 0x01u | ((part - 64u) << 1u)
    | ((ep_l(tb,0u).x & 0xF0u) << 1u) | ((ep_h(tb,0u).x & 0xF0u) << 5u)
    | ((ep_l(tb,1u).x & 0xF0u) << 9u) | ((ep_h(tb,1u).x & 0xF0u) << 13u)
    | ((ep_l(tb,2u).x & 0xF0u) << 17u) | ((ep_h(tb,2u).x & 0xF0u) << 21u)
    | ((ep_l(tb,0u).y & 0xF0u) << 25u);
  var by: u32 = ((ep_l(tb,0u).y & 0xF0u) >> 7u) | ((ep_h(tb,0u).y & 0xF0u) >> 3u)
    | ((ep_l(tb,1u).y & 0xF0u) << 1u) | ((ep_h(tb,1u).y & 0xF0u) << 5u)
    | ((ep_l(tb,2u).y & 0xF0u) << 9u) | ((ep_h(tb,2u).y & 0xF0u) << 13u)
    | ((ep_l(tb,0u).z & 0xF0u) << 17u) | ((ep_h(tb,0u).z & 0xF0u) << 21u)
    | ((ep_l(tb,1u).z & 0xF0u) << 25u);
  var bz: u32 = ((ep_l(tb,1u).z & 0xF0u) >> 7u) | ((ep_h(tb,1u).z & 0xF0u) >> 3u)
    | ((ep_l(tb,2u).z & 0xF0u) << 1u) | ((ep_h(tb,2u).z & 0xF0u) << 5u)
    | ((ep_l(tb,0u).x & 0x08u) << 10u) | ((ep_h(tb,0u).x & 0x08u) << 11u)
    | ((ep_l(tb,1u).x & 0x08u) << 12u) | ((ep_h(tb,1u).x & 0x08u) << 13u)
    | ((ep_l(tb,2u).x & 0x08u) << 14u) | ((ep_h(tb,2u).x & 0x08u) << 15u)
    | (cidx(tb,0u) << 19u);
  var bw: u32 = 0u;
  let f0 = candidateFixUpIndex1DOrdered[part].x;
  let f1 = candidateFixUpIndex1DOrdered[part].y;
  var i: u32 = 1u;
  for (; i <= min(f0, 4u); i++) {
    bz |= cidx(tb, i) << (i * 3u + 18u);
  }
  if (f0 < 4u) {
    bz |= cidx(tb, 4u) << 29u;
    i += 1u;
  } else {
    bw |= (cidx(tb, 4u) & 0x04u) >> 2u;
    for (; i <= f0; i++) {
      bw |= cidx(tb, i) << (i * 3u - 14u);
    }
  }
  for (; i <= f1; i++) {
    bw |= cidx(tb, i) << (i * 3u - 15u);
  }
  for (; i < 16u; i++) {
    bw |= cidx(tb, i) << (i * 3u - 16u);
  }
  return vec4<u32>(bx, by, bz, bw);
}

fn block_package1(part: u32, tb: u32) -> vec4<u32> {
  var bx: u32 = 0x02u | (part << 2u)
    | ((ep_l(tb,0u).x & 0xFCu) << 6u) | ((ep_h(tb,0u).x & 0xFCu) << 12u)
    | ((ep_l(tb,1u).x & 0xFCu) << 18u) | ((ep_h(tb,1u).x & 0xFCu) << 24u);
  var by: u32 = ((ep_l(tb,0u).y & 0xFCu) >> 2u) | ((ep_h(tb,0u).y & 0xFCu) << 4u)
    | ((ep_l(tb,1u).y & 0xFCu) << 10u) | ((ep_h(tb,1u).y & 0xFCu) << 16u)
    | ((ep_l(tb,0u).z & 0xFCu) << 22u) | ((ep_h(tb,0u).z & 0xFCu) << 28u);
  var bz: u32 = ((ep_h(tb,0u).z & 0xFCu) >> 4u) | ((ep_l(tb,1u).z & 0xFCu) << 2u)
    | ((ep_h(tb,1u).z & 0xFCu) << 8u)
    | ((ep_l(tb,0u).x & 0x02u) << 15u) | ((ep_l(tb,1u).x & 0x02u) << 16u)
    | (cidx(tb,0u) << 18u);
  var bw: u32 = 0u;
  let f0 = candidateFixUpIndex1DOrdered[part].x;
  if (f0 == 15u) {
    bw = (cidx(tb,15u) << 30u) | (cidx(tb,14u) << 27u) | (cidx(tb,13u) << 24u) | (cidx(tb,12u) << 21u) | (cidx(tb,11u) << 18u) | (cidx(tb,10u) << 15u)
      | (cidx(tb,9u) << 12u) | (cidx(tb,8u) << 9u) | (cidx(tb,7u) << 6u) | (cidx(tb,6u) << 3u) | cidx(tb,5u);
    bz |= (cidx(tb,4u) << 29u) | (cidx(tb,3u) << 26u) | (cidx(tb,2u) << 23u) | (cidx(tb,1u) << 20u) | (cidx(tb,0u) << 18u);
  } else if (f0 == 2u) {
    bw = (cidx(tb,15u) << 29u) | (cidx(tb,14u) << 26u) | (cidx(tb,13u) << 23u) | (cidx(tb,12u) << 20u) | (cidx(tb,11u) << 17u) | (cidx(tb,10u) << 14u)
      | (cidx(tb,9u) << 11u) | (cidx(tb,8u) << 8u) | (cidx(tb,7u) << 5u) | (cidx(tb,6u) << 2u) | (cidx(tb,5u) >> 1u);
    bz |= (cidx(tb,5u) << 31u) | (cidx(tb,4u) << 28u) | (cidx(tb,3u) << 25u) | (cidx(tb,2u) << 23u) | (cidx(tb,1u) << 20u) | (cidx(tb,0u) << 18u);
  } else if (f0 == 8u) {
    bw = (cidx(tb,15u) << 29u) | (cidx(tb,14u) << 26u) | (cidx(tb,13u) << 23u) | (cidx(tb,12u) << 20u) | (cidx(tb,11u) << 17u) | (cidx(tb,10u) << 14u)
      | (cidx(tb,9u) << 11u) | (cidx(tb,8u) << 9u) | (cidx(tb,7u) << 6u) | (cidx(tb,6u) << 3u) | cidx(tb,5u);
    bz |= (cidx(tb,4u) << 29u) | (cidx(tb,3u) << 26u) | (cidx(tb,2u) << 23u) | (cidx(tb,1u) << 20u) | (cidx(tb,0u) << 18u);
  } else {
    bw = (cidx(tb,15u) << 29u) | (cidx(tb,14u) << 26u) | (cidx(tb,13u) << 23u) | (cidx(tb,12u) << 20u) | (cidx(tb,11u) << 17u) | (cidx(tb,10u) << 14u)
      | (cidx(tb,9u) << 11u) | (cidx(tb,8u) << 8u) | (cidx(tb,7u) << 5u) | (cidx(tb,6u) << 3u) | cidx(tb,5u);
    bz |= (cidx(tb,4u) << 29u) | (cidx(tb,3u) << 26u) | (cidx(tb,2u) << 23u) | (cidx(tb,1u) << 20u) | (cidx(tb,0u) << 18u);
  }
  return vec4<u32>(bx, by, bz, bw);
}

fn block_package2(part: u32, tb: u32) -> vec4<u32> {
  var bx: u32 = 0x04u | ((part - 64u) << 3u)
    | ((ep_l(tb,0u).x & 0xF8u) << 6u) | ((ep_h(tb,0u).x & 0xF8u) << 11u)
    | ((ep_l(tb,1u).x & 0xF8u) << 16u) | ((ep_h(tb,1u).x & 0xF8u) << 21u)
    | ((ep_l(tb,2u).x & 0xF8u) << 26u);
  var by: u32 = ((ep_l(tb,2u).x & 0xF8u) >> 6u) | ((ep_h(tb,2u).x & 0xF8u) >> 1u)
    | ((ep_l(tb,0u).y & 0xF8u) << 4u) | ((ep_h(tb,0u).y & 0xF8u) << 9u)
    | ((ep_l(tb,1u).y & 0xF8u) << 14u) | ((ep_h(tb,1u).y & 0xF8u) << 19u)
    | ((ep_l(tb,2u).y & 0xF8u) << 24u);
  var bz: u32 = ((ep_h(tb,2u).y & 0xF8u) >> 3u) | ((ep_l(tb,0u).z & 0xF8u) << 2u)
    | ((ep_h(tb,0u).z & 0xF8u) << 7u) | ((ep_l(tb,1u).z & 0xF8u) << 12u)
    | ((ep_h(tb,1u).z & 0xF8u) << 17u) | ((ep_l(tb,2u).z & 0xF8u) << 22u)
    | ((ep_h(tb,2u).z & 0xF8u) << 27u);
  var bw: u32 = ((ep_h(tb,2u).z & 0xF8u) >> 5u) | (cidx(tb,0u) << 3u);
  let f0 = candidateFixUpIndex1DOrdered[part].x;
  let f1 = candidateFixUpIndex1DOrdered[part].y;
  var i: u32 = 1u;
  for (; i <= f0; i++) { bw |= cidx(tb, i) << (i * 2u + 2u); }
  for (; i <= f1; i++) { bw |= cidx(tb, i) << (i * 2u + 1u); }
  for (; i < 16u; i++) { bw |= cidx(tb, i) << (i * 2u); }
  return vec4<u32>(bx, by, bz, bw);
}

fn block_package3(part: u32, tb: u32) -> vec4<u32> {
  var bx: u32 = 0x08u | (part << 4u)
    | ((ep_l(tb,0u).x & 0xFEu) << 9u) | ((ep_h(tb,0u).x & 0xFEu) << 16u)
    | ((ep_l(tb,1u).x & 0xFEu) << 23u) | ((ep_h(tb,1u).x & 0xFEu) << 30u);
  var by: u32 = ((ep_h(tb,1u).x & 0xFEu) >> 2u) | ((ep_l(tb,0u).y & 0xFEu) << 5u)
    | ((ep_h(tb,0u).y & 0xFEu) << 12u) | ((ep_l(tb,1u).y & 0xFEu) << 19u)
    | ((ep_h(tb,1u).y & 0xFEu) << 26u);
  var bz: u32 = ((ep_h(tb,1u).y & 0xFEu) >> 6u) | ((ep_l(tb,0u).z & 0xFEu) << 1u)
    | ((ep_h(tb,0u).z & 0xFEu) << 8u) | ((ep_l(tb,1u).z & 0xFEu) << 15u)
    | ((ep_h(tb,1u).z & 0xFEu) << 22u)
    | ((ep_l(tb,0u).x & 0x01u) << 30u) | ((ep_h(tb,0u).x & 0x01u) << 31u);
  var bw: u32 = ((ep_l(tb,1u).x & 0x01u) << 0u) | ((ep_h(tb,1u).x & 0x01u) << 1u)
    | (cidx(tb,0u) << 2u);
  let f0 = candidateFixUpIndex1DOrdered[part].x;
  var i: u32 = 1u;
  for (; i <= f0; i++) { bw |= cidx(tb, i) << (i * 2u + 1u); }
  for (; i < 16u; i++) { bw |= cidx(tb, i) << (i * 2u); }
  return vec4<u32>(bx, by, bz, bw);
}

fn block_package4(rotation: u32, index_selector: u32, tb: u32) -> vec4<u32> {
  let bx: u32 = 0x10u | ((rotation & 3u) << 5u) | ((index_selector & 1u) << 7u)
    | ((ep_l(tb,0u).x & 0xF8u) << 5u) | ((ep_h(tb,0u).x & 0xF8u) << 10u)
    | ((ep_l(tb,0u).y & 0xF8u) << 15u) | ((ep_h(tb,0u).y & 0xF8u) << 20u)
    | ((ep_l(tb,0u).z & 0xF8u) << 25u);
  let by: u32 = ((ep_l(tb,0u).z & 0xF8u) >> 7u) | ((ep_h(tb,0u).z & 0xF8u) >> 2u)
    | ((ep_l(tb,0u).w & 0xFCu) << 4u) | ((ep_h(tb,0u).w & 0xFCu) << 10u)
    | ((cidx(tb,0u) & 1u) << 18u) | (cidx(tb,1u) << 19u) | (cidx(tb,2u) << 21u) | (cidx(tb,3u) << 23u)
    | (cidx(tb,4u) << 25u) | (cidx(tb,5u) << 27u) | (cidx(tb,6u) << 29u) | (cidx(tb,7u) << 31u);
  let bz: u32 = (cidx(tb,7u) >> 1u) | (cidx(tb,8u) << 1u) | (cidx(tb,9u) << 3u) | (cidx(tb,10u) << 5u)
    | (cidx(tb,11u) << 7u) | (cidx(tb,12u) << 9u) | (cidx(tb,13u) << 11u) | (cidx(tb,14u) << 13u)
    | (cidx(tb,15u) << 15u) | ((aidx(tb,0u) & 3u) << 17u) | (aidx(tb,1u) << 19u) | (aidx(tb,2u) << 22u)
    | (aidx(tb,3u) << 25u) | (aidx(tb,4u) << 28u) | (aidx(tb,5u) << 31u);
  let bw: u32 = (aidx(tb,5u) >> 1u) | (aidx(tb,6u) << 2u) | (aidx(tb,7u) << 5u) | (aidx(tb,8u) << 8u)
    | (aidx(tb,9u) << 11u) | (aidx(tb,10u) << 14u) | (aidx(tb,11u) << 17u) | (aidx(tb,12u) << 20u)
    | (aidx(tb,13u) << 23u) | (aidx(tb,14u) << 26u) | (aidx(tb,15u) << 29u);
  return vec4<u32>(bx, by, bz, bw);
}

fn block_package5(rotation: u32, tb: u32) -> vec4<u32> {
  let bx: u32 = 0x20u | (rotation << 6u)
    | ((ep_l(tb,0u).x & 0xFEu) << 7u) | ((ep_h(tb,0u).x & 0xFEu) << 14u)
    | ((ep_l(tb,0u).y & 0xFEu) << 21u) | ((ep_h(tb,0u).y & 0xFEu) << 28u);
  let by: u32 = ((ep_h(tb,0u).y & 0xFEu) >> 4u) | ((ep_l(tb,0u).z & 0xFEu) << 3u)
    | ((ep_h(tb,0u).z & 0xFEu) << 10u) | (ep_l(tb,0u).w << 18u) | (ep_h(tb,0u).w << 26u);
  let bz: u32 = (ep_h(tb,0u).w >> 6u)
    | (cidx(tb,0u) << 2u) | (cidx(tb,1u) << 3u) | (cidx(tb,2u) << 5u) | (cidx(tb,3u) << 7u)
    | (cidx(tb,4u) << 9u) | (cidx(tb,5u) << 11u) | (cidx(tb,6u) << 13u) | (cidx(tb,7u) << 15u)
    | (cidx(tb,8u) << 17u) | (cidx(tb,9u) << 19u) | (cidx(tb,10u) << 21u) | (cidx(tb,11u) << 23u)
    | (cidx(tb,12u) << 25u) | (cidx(tb,13u) << 27u) | (cidx(tb,14u) << 29u) | (cidx(tb,15u) << 31u);
  let bw: u32 = (cidx(tb,15u) >> 1u) | (aidx(tb,0u) << 1u) | (aidx(tb,1u) << 2u) | (aidx(tb,2u) << 4u)
    | (aidx(tb,3u) << 6u) | (aidx(tb,4u) << 8u) | (aidx(tb,5u) << 10u) | (aidx(tb,6u) << 12u)
    | (aidx(tb,7u) << 14u) | (aidx(tb,8u) << 16u) | (aidx(tb,9u) << 18u) | (aidx(tb,10u) << 20u)
    | (aidx(tb,11u) << 22u) | (aidx(tb,12u) << 24u) | (aidx(tb,13u) << 26u) | (aidx(tb,14u) << 28u)
    | (aidx(tb,15u) << 30u);
  return vec4<u32>(bx, by, bz, bw);
}

fn block_package6(tb: u32) -> vec4<u32> {
  let bx: u32 = 0x40u
    | ((ep_l(tb,0u).x & 0xFEu) << 6u) | ((ep_h(tb,0u).x & 0xFEu) << 13u)
    | ((ep_l(tb,0u).y & 0xFEu) << 20u) | ((ep_h(tb,0u).y & 0xFEu) << 27u);
  let by: u32 = ((ep_h(tb,0u).y & 0xFEu) >> 5u) | ((ep_l(tb,0u).z & 0xFEu) << 2u)
    | ((ep_h(tb,0u).z & 0xFEu) << 9u) | ((ep_l(tb,0u).w & 0xFEu) << 16u)
    | ((ep_h(tb,0u).w & 0xFEu) << 23u)
    | ((ep_l(tb,0u).x & 0x01u) << 31u);
  let bz: u32 = (ep_h(tb,0u).x & 0x01u)
    | (cidx(tb,0u) << 1u) | (cidx(tb,1u) << 4u) | (cidx(tb,2u) << 8u) | (cidx(tb,3u) << 12u)
    | (cidx(tb,4u) << 16u) | (cidx(tb,5u) << 20u) | (cidx(tb,6u) << 24u) | (cidx(tb,7u) << 28u);
  let bw: u32 = (cidx(tb,8u) << 0u) | (cidx(tb,9u) << 4u) | (cidx(tb,10u) << 8u) | (cidx(tb,11u) << 12u)
    | (cidx(tb,12u) << 16u) | (cidx(tb,13u) << 20u) | (cidx(tb,14u) << 24u) | (cidx(tb,15u) << 28u);
  return vec4<u32>(bx, by, bz, bw);
}

fn block_package7(part: u32, tb: u32) -> vec4<u32> {
  var bx: u32 = 0x80u | (part << 8u)
    | ((ep_l(tb,0u).x & 0xF8u) << 11u) | ((ep_h(tb,0u).x & 0xF8u) << 16u)
    | ((ep_l(tb,1u).x & 0xF8u) << 21u) | ((ep_h(tb,1u).x & 0xF8u) << 26u);
  var by: u32 = ((ep_h(tb,1u).x & 0xF8u) >> 6u) | ((ep_l(tb,0u).y & 0xF8u) >> 1u)
    | ((ep_h(tb,0u).y & 0xF8u) << 4u) | ((ep_l(tb,1u).y & 0xF8u) << 9u)
    | ((ep_h(tb,1u).y & 0xF8u) << 14u) | ((ep_l(tb,0u).z & 0xF8u) << 19u)
    | ((ep_h(tb,0u).z & 0xF8u) << 24u);
  var bz: u32 = ((ep_l(tb,1u).z & 0xF8u) >> 3u) | ((ep_h(tb,1u).z & 0xF8u) << 2u)
    | ((ep_l(tb,0u).w & 0xF8u) << 7u) | ((ep_h(tb,0u).w & 0xF8u) << 12u)
    | ((ep_l(tb,1u).w & 0xF8u) << 17u) | ((ep_h(tb,1u).w & 0xF8u) << 22u)
    | ((ep_l(tb,0u).x & 0x04u) << 28u) | ((ep_h(tb,0u).x & 0x04u) << 29u);
  var bw: u32 = ((ep_l(tb,1u).x & 0x04u) >> 2u) | ((ep_h(tb,1u).x & 0x04u) >> 1u)
    | (cidx(tb,0u) << 2u);
  let f0 = candidateFixUpIndex1DOrdered[part].x;
  var i: u32 = 1u;
  for (; i <= f0; i++) { bw |= cidx(tb, i) << (i * 2u + 1u); }
  for (; i < 16u; i++) { bw |= cidx(tb, i) << (i * 2u); }
  return vec4<u32>(bx, by, bz, bw);
}

//------------------------------------------------------------------------------
// Pass 1: modes 4, 5, 6 (1 subset)
//------------------------------------------------------------------------------
@compute @workgroup_size(THREAD_GROUP_SIZE)
fn TryMode456CS(@builtin(local_invocation_index) GI: u32, @builtin(workgroup_id) groupID: vec3<u32>) {
  let MAX_USED_THREAD: u32 = 16u;
  let BLOCK_IN_GROUP: u32 = THREAD_GROUP_SIZE / MAX_USED_THREAD;
  let blockInGroup: u32 = GI / MAX_USED_THREAD;
  let blockID: u32 = params.g_start_block_id + groupID.x * BLOCK_IN_GROUP + blockInGroup;
  let threadBase: u32 = blockInGroup * MAX_USED_THREAD;
  let threadInBlock: u32 = GI - threadBase;

  let block_y = blockID / params.g_num_block_x;
  let block_x = blockID - block_y * params.g_num_block_x;
  let base_x = block_x * BLOCK_SIZE_X;
  let base_y = block_y * BLOCK_SIZE_Y;

  if (threadInBlock < 16u) {
    let px = loadPixel(base_x, base_y, threadInBlock);
    shared_temp[GI].pixel = px;
    shared_temp[GI].endPoint_low = px;
    shared_temp[GI].endPoint_high = px;
  }
  workgroupBarrier();

  if (threadInBlock < 8u) {
    shared_temp[GI].endPoint_low = min(shared_temp[GI].endPoint_low, shared_temp[GI + 8u].endPoint_low);
    shared_temp[GI].endPoint_high = max(shared_temp[GI].endPoint_high, shared_temp[GI + 8u].endPoint_high);
  }
  workgroupBarrier();
  if (threadInBlock < 4u) {
    shared_temp[GI].endPoint_low = min(shared_temp[GI].endPoint_low, shared_temp[GI + 4u].endPoint_low);
    shared_temp[GI].endPoint_high = max(shared_temp[GI].endPoint_high, shared_temp[GI + 4u].endPoint_high);
  }
  workgroupBarrier();
  if (threadInBlock < 2u) {
    shared_temp[GI].endPoint_low = min(shared_temp[GI].endPoint_low, shared_temp[GI + 2u].endPoint_low);
    shared_temp[GI].endPoint_high = max(shared_temp[GI].endPoint_high, shared_temp[GI + 2u].endPoint_high);
  }
  workgroupBarrier();
  if (threadInBlock < 1u) {
    shared_temp[GI].endPoint_low = min(shared_temp[GI].endPoint_low, shared_temp[GI + 1u].endPoint_low);
    shared_temp[GI].endPoint_high = max(shared_temp[GI].endPoint_high, shared_temp[GI + 1u].endPoint_high);
  }
  workgroupBarrier();

  var endPoint: mat2x4u;
  endPoint[0] = shared_temp[threadBase].endPoint_low;
  endPoint[1] = shared_temp[threadBase].endPoint_high;

  var error: u32 = 0xFFFFFFFFu;
  var mode: u32 = 0u;
  var index_selector: u32 = 0u;
  var rotation: u32 = 0u;

  var indexPrec: vec2<u32>;
  if (threadInBlock < 8u) {
    if (0u == (threadInBlock & 1u)) {
      index_selector = 0u;
      indexPrec = vec2<u32>(2u, 1u);
    } else {
      index_selector = 1u;
      indexPrec = vec2<u32>(1u, 2u);
    }
  } else {
    indexPrec = vec2<u32>(2u, 2u);
  }

  if (threadInBlock < 12u) {
    if ((threadInBlock < 2u) || (8u == threadInBlock)) {
      rotation = 0u;
    } else if ((threadInBlock < 4u) || (9u == threadInBlock)) {
      endPoint[0] = vec4<u32>(endPoint[0].w, endPoint[0].y, endPoint[0].z, endPoint[0].x);
      endPoint[1] = vec4<u32>(endPoint[1].w, endPoint[1].y, endPoint[1].z, endPoint[1].x);
      rotation = 1u;
    } else if ((threadInBlock < 6u) || (10u == threadInBlock)) {
      endPoint[0] = vec4<u32>(endPoint[0].x, endPoint[0].w, endPoint[0].z, endPoint[0].y);
      endPoint[1] = vec4<u32>(endPoint[1].x, endPoint[1].w, endPoint[1].z, endPoint[1].y);
      rotation = 2u;
    } else if ((threadInBlock < 8u) || (11u == threadInBlock)) {
      endPoint[0] = vec4<u32>(endPoint[0].x, endPoint[0].y, endPoint[0].w, endPoint[0].z);
      endPoint[1] = vec4<u32>(endPoint[1].x, endPoint[1].y, endPoint[1].w, endPoint[1].z);
      rotation = 3u;
    }

    if (threadInBlock < 8u) {
      mode = 4u;
      compress_endpoints4(&endPoint);
    } else {
      mode = 5u;
      compress_endpoints5(&endPoint);
    }

    var pixel = shared_temp[threadBase + 0u].pixel;
    if (1u == rotation) { pixel = vec4<u32>(pixel.w, pixel.y, pixel.z, pixel.x); }
    else if (2u == rotation) { pixel = vec4<u32>(pixel.x, pixel.w, pixel.z, pixel.y); }
    else if (3u == rotation) { pixel = vec4<u32>(pixel.x, pixel.y, pixel.w, pixel.z); }

    var span = vec4<i32>(endPoint[1]) - vec4<i32>(endPoint[0]);
    var span_norm_sqr = vec2<i32>(dot3i(span.xyz, span.xyz), span.w * span.w);

    var dp = vec2<i32>(
      dot3i(vec3<i32>(pixel.xyz) - vec3<i32>(endPoint[0].xyz), vec3<i32>(pixel.xyz) - vec3<i32>(endPoint[0].xyz)),
      dot3i(vec3<i32>(pixel.xyz) - vec3<i32>(endPoint[1].xyz), vec3<i32>(pixel.xyz) - vec3<i32>(endPoint[1].xyz)));
    if (dp.x > dp.y) {
      span = vec4<i32>(-span.xyz, span.w);
      let t = endPoint[0]; let u = endPoint[1];
      endPoint[0] = vec4<u32>(u.xyz, t.w);
      endPoint[1] = vec4<u32>(t.xyz, u.w);
    }
    dp = vec2<i32>(
      (i32(pixel.w) - i32(endPoint[0].w)) * (i32(pixel.w) - i32(endPoint[0].w)),
      (i32(pixel.w) - i32(endPoint[1].w)) * (i32(pixel.w) - i32(endPoint[1].w)));
    if (dp.x > dp.y) {
      span.w = -span.w;
      let tw = endPoint[0].w; endPoint[0].w = endPoint[1].w; endPoint[1].w = tw;
    }

    error = 0u;
    for (var i = 0u; i < 16u; i++) {
      pixel = shared_temp[threadBase + i].pixel;
      if (1u == rotation) { pixel = vec4<u32>(pixel.w, pixel.y, pixel.z, pixel.x); }
      else if (2u == rotation) { pixel = vec4<u32>(pixel.x, pixel.w, pixel.z, pixel.y); }
      else if (3u == rotation) { pixel = vec4<u32>(pixel.x, pixel.y, pixel.w, pixel.z); }

      let dpx = dot3i(span.xyz, vec3<i32>(pixel.xyz) - vec3<i32>(endPoint[0].xyz));
      var color_index: u32;
      if (span_norm_sqr.x <= 0 || dpx <= 0) { color_index = 0u; }
      else if (dpx < span_norm_sqr.x) { color_index = aStep[indexPrec.x][u32(f32(dpx) * 63.49999 / f32(span_norm_sqr.x))]; }
      else { color_index = aStep[indexPrec.x][63]; }

      let dpy = span.w * (i32(pixel.w) - i32(endPoint[0].w));
      var alpha_index: u32;
      if (span_norm_sqr.y <= 0 || dpy <= 0) { alpha_index = 0u; }
      else if (dpy < span_norm_sqr.y) { alpha_index = aStep[indexPrec.y][u32(f32(dpy) * 63.49999 / f32(span_norm_sqr.y))]; }
      else { alpha_index = aStep[indexPrec.y][63]; }

      var pixel_r: vec4<u32>;
      pixel_r = vec4<u32>(interp(endPoint[0], endPoint[1], aWeight[indexPrec.x][color_index]).xyz,
        ((64u - aWeight[indexPrec.y][alpha_index]) * endPoint[0].w + aWeight[indexPrec.y][alpha_index] * endPoint[1].w + 32u) >> 6u);

      let ord = Ensure_A_Is_Larger(pixel_r, pixel);
      var diff = ord[0] - ord[1];
      if (1u == rotation) { diff = vec4<u32>(diff.w, diff.y, diff.z, diff.x); }
      else if (2u == rotation) { diff = vec4<u32>(diff.x, diff.w, diff.z, diff.y); }
      else if (3u == rotation) { diff = vec4<u32>(diff.x, diff.y, diff.w, diff.z); }
      error += ComputeError(diff, diff);
    }
  } else if (threadInBlock < 16u) {
    let p = threadInBlock - 12u;
    compress_endpoints6(&endPoint, vec2<u32>(p >> 0u, p >> 1u) & vec2<u32>(1u));

    var pixel = shared_temp[threadBase + 0u].pixel;
    var span = vec4<i32>(endPoint[1]) - vec4<i32>(endPoint[0]);
    var span_norm_sqr = dot4i(span, span);
    let dp0 = dot4i(span, vec4<i32>(pixel) - vec4<i32>(endPoint[0]));
    if (span_norm_sqr > 0 && dp0 >= 0 && u32(f32(dp0) * 63.49999) > u32(32 * span_norm_sqr)) {
      span = -span;
      let t = endPoint[0]; endPoint[0] = endPoint[1]; endPoint[1] = t;
    }

    error = 0u;
    for (var i = 0u; i < 16u; i++) {
      pixel = shared_temp[threadBase + i].pixel;
      let dpx = dot4i(span, vec4<i32>(pixel) - vec4<i32>(endPoint[0]));
      var color_index: u32;
      if (span_norm_sqr <= 0 || dpx <= 0) { color_index = 0u; }
      else if (dpx < span_norm_sqr) { color_index = aStep[0][u32(f32(dpx) * 63.49999 / f32(span_norm_sqr))]; }
      else { color_index = aStep[0][63]; }

      let pixel_r = interp(endPoint[0], endPoint[1], aWeight[0][color_index]);
      let ord = Ensure_A_Is_Larger(pixel_r, pixel);
      let diff = ord[0] - ord[1];
      error += ComputeError(diff, diff);
    }

    mode = 6u;
    rotation = p;
  }

  shared_temp[GI].error = error;
  shared_temp[GI].mode = mode;
  shared_temp[GI].index_selector = index_selector;
  shared_temp[GI].rotation = rotation;
  workgroupBarrier();

  for (var step = 8u; step >= 1u; step = step >> 1u) {
    if (threadInBlock < step) {
      if (shared_temp[GI].error > shared_temp[GI + step].error) {
        shared_temp[GI].error = shared_temp[GI + step].error;
        shared_temp[GI].mode = shared_temp[GI + step].mode;
        shared_temp[GI].index_selector = shared_temp[GI + step].index_selector;
        shared_temp[GI].rotation = shared_temp[GI + step].rotation;
      }
    }
    workgroupBarrier();
  }

  if (threadInBlock < 1u) {
    g_OutBuff[blockID] = vec4<u32>(shared_temp[GI].error,
      (shared_temp[GI].index_selector << 31u) | shared_temp[GI].mode, 0u, shared_temp[GI].rotation);
  }
}

//------------------------------------------------------------------------------
// Pass 2: modes 1, 3, 7 (2 subsets)
//------------------------------------------------------------------------------
@compute @workgroup_size(THREAD_GROUP_SIZE)
fn TryMode137CS(@builtin(local_invocation_index) GI: u32, @builtin(workgroup_id) groupID: vec3<u32>) {
  let MAX_USED_THREAD: u32 = 64u;
  let BLOCK_IN_GROUP: u32 = THREAD_GROUP_SIZE / MAX_USED_THREAD;
  let blockInGroup: u32 = GI / MAX_USED_THREAD;
  let blockID: u32 = params.g_start_block_id + groupID.x * BLOCK_IN_GROUP + blockInGroup;
  let threadBase: u32 = blockInGroup * MAX_USED_THREAD;
  let threadInBlock: u32 = GI - threadBase;

  let block_y = blockID / params.g_num_block_x;
  let block_x = blockID - block_y * params.g_num_block_x;
  let base_x = block_x * BLOCK_SIZE_X;
  let base_y = block_y * BLOCK_SIZE_Y;

  if (threadInBlock < 16u) {
    shared_temp[GI].pixel = loadPixel(base_x, base_y, threadInBlock);
  }
  workgroupBarrier();

  shared_temp[GI].error = 0xFFFFFFFFu;

  let part = threadInBlock;
  var endPoint: array<mat2x4u, 2>;
  var endPointBackup: array<mat2x4u, 2>;
  if (threadInBlock < 64u) {
    endPoint[0][0] = vec4<u32>(MAX_UINT); endPoint[0][1] = vec4<u32>(MIN_UINT);
    endPoint[1][0] = vec4<u32>(MAX_UINT); endPoint[1][1] = vec4<u32>(MIN_UINT);
    let bits = candidateSectionBit[part];
    for (var i = 0u; i < 16u; i++) {
      let pixel = shared_temp[threadBase + i].pixel;
      if (((bits >> i) & 0x01u) == 1u) {
        endPoint[1][0] = min(endPoint[1][0], pixel);
        endPoint[1][1] = max(endPoint[1][1], pixel);
      } else {
        endPoint[0][0] = min(endPoint[0][0], pixel);
        endPoint[0][1] = max(endPoint[0][1], pixel);
      }
    }
    endPointBackup[0] = endPoint[0];
    endPointBackup[1] = endPoint[1];

    var max_p: u32;
    if (1u == params.g_mode_id) { max_p = 2u; } else { max_p = 4u; }

    var final_p = array<u32, 2>(0u, 0u);
    var errs = array<u32, 2>(MAX_UINT, MAX_UINT);
    for (var p = 0u; p < max_p; p++) {
      endPoint[0] = endPointBackup[0];
      endPoint[1] = endPointBackup[1];

      for (var s = 0u; s < 2u; s++) {
        if (params.g_mode_id == 1u) { compress_endpoints1(&endPoint[s], vec2<u32>(p)); }
        else if (params.g_mode_id == 3u) { compress_endpoints3(&endPoint[s], vec2<u32>(p, p >> 1u) & vec2<u32>(1u)); }
        else if (params.g_mode_id == 7u) { compress_endpoints7(&endPoint[s], vec2<u32>(p, p >> 1u) & vec2<u32>(1u)); }
      }

      var span0 = vec4<i32>(endPoint[0][1]) - vec4<i32>(endPoint[0][0]);
      var span1 = vec4<i32>(endPoint[1][1]) - vec4<i32>(endPoint[1][0]);
      if (params.g_mode_id != 7u) { span0.w = 0; span1.w = 0; }

      var sns0 = dot4i(span0, span0);
      var sns1 = dot4i(span1, span1);

      var dp = dot4i(span0, vec4<i32>(shared_temp[threadBase + 0u].pixel) - vec4<i32>(endPoint[0][0]));
      if (sns0 > 0 && dp > 0 && u32(f32(dp) * 63.49999) > u32(32 * sns0)) {
        span0 = -span0;
        let t = endPoint[0][0]; endPoint[0][0] = endPoint[0][1]; endPoint[0][1] = t;
      }
      dp = dot4i(span1, vec4<i32>(shared_temp[threadBase + candidateFixUpIndex1D[part].x].pixel) - vec4<i32>(endPoint[1][0]));
      if (sns1 > 0 && dp > 0 && u32(f32(dp) * 63.49999) > u32(32 * sns1)) {
        span1 = -span1;
        let t = endPoint[1][0]; endPoint[1][0] = endPoint[1][1]; endPoint[1][1] = t;
      }

      var step_selector: u32;
      if (params.g_mode_id != 1u) { step_selector = 2u; } else { step_selector = 1u; }

      var p_error = array<u32, 2>(0u, 0u);
      for (var i = 0u; i < 16u; i++) {
        let subset_index = (bits >> i) & 0x01u;
        var color_index: u32;
        if (subset_index == 1u) {
          let d = dot4i(span1, vec4<i32>(shared_temp[threadBase + i].pixel) - vec4<i32>(endPoint[1][0]));
          if (sns1 <= 0 || d <= 0) { color_index = 0u; }
          else if (d < sns1) { color_index = aStep[step_selector][u32(f32(d) * 63.49999 / f32(sns1))]; }
          else { color_index = aStep[step_selector][63]; }
        } else {
          let d = dot4i(span0, vec4<i32>(shared_temp[threadBase + i].pixel) - vec4<i32>(endPoint[0][0]));
          if (sns0 <= 0 || d <= 0) { color_index = 0u; }
          else if (d < sns0) { color_index = aStep[step_selector][u32(f32(d) * 63.49999 / f32(sns0))]; }
          else { color_index = aStep[step_selector][63]; }
        }

        // Same values as `endPoint[subset_index][0]` and `[1]`. Indexing the
        // function-local `endPoint` array with a runtime value here sends Mesa's
        // Intel Vulkan driver (ANV, seen on Mesa 26.2.2 with an Arc Pro B60) into
        // an unbounded NIR optimisation that never finishes compiling this entry
        // point and grows until the OOM killer ends the process. With `select`
        // it compiles in ~70 ms.
        let ep_lo = select(endPoint[0][0], endPoint[1][0], subset_index == 1u);
        let ep_hi = select(endPoint[0][1], endPoint[1][1], subset_index == 1u);
        var pixel_r = interp(ep_lo, ep_hi, aWeight[step_selector][color_index]);
        if (params.g_mode_id != 7u) { pixel_r.w = 255u; }

        let pixel = shared_temp[threadBase + i].pixel;
        let ord = Ensure_A_Is_Larger(pixel_r, pixel);
        let diff = ord[0] - ord[1];
        let pe = ComputeError(diff, diff);
        if (subset_index == 1u) { p_error[1] += pe; } else { p_error[0] += pe; }
      }

      for (var s = 0u; s < 2u; s++) {
        if (p_error[s] < errs[s]) { errs[s] = p_error[s]; final_p[s] = p; }
      }
    }

    shared_temp[GI].error = errs[0] + errs[1];
    shared_temp[GI].mode = params.g_mode_id;
    shared_temp[GI].part = part;
    if (params.g_mode_id == 1u) { shared_temp[GI].rotation = (final_p[1] << 1u) | final_p[0]; }
    else { shared_temp[GI].rotation = (final_p[1] << 2u) | final_p[0]; }
  }
  workgroupBarrier();

  for (var step = 32u; step >= 1u; step = step >> 1u) {
    if (threadInBlock < step) {
      if (shared_temp[GI].error > shared_temp[GI + step].error) {
        shared_temp[GI].error = shared_temp[GI + step].error;
        shared_temp[GI].mode = shared_temp[GI + step].mode;
        shared_temp[GI].part = shared_temp[GI + step].part;
        shared_temp[GI].rotation = shared_temp[GI + step].rotation;
      }
    }
    workgroupBarrier();
  }

  if (threadInBlock < 1u) {
    if (g_InBuff[blockID].x > shared_temp[GI].error) {
      g_OutBuff[blockID] = vec4<u32>(shared_temp[GI].error, shared_temp[GI].mode, shared_temp[GI].part, shared_temp[GI].rotation);
    } else {
      g_OutBuff[blockID] = g_InBuff[blockID];
    }
  }
}

//------------------------------------------------------------------------------
// Pass 3: modes 0, 2 (3 subsets)
//------------------------------------------------------------------------------
@compute @workgroup_size(THREAD_GROUP_SIZE)
fn TryMode02CS(@builtin(local_invocation_index) GI: u32, @builtin(workgroup_id) groupID: vec3<u32>) {
  let MAX_USED_THREAD: u32 = 64u;
  let BLOCK_IN_GROUP: u32 = THREAD_GROUP_SIZE / MAX_USED_THREAD;
  let blockInGroup: u32 = GI / MAX_USED_THREAD;
  let blockID: u32 = params.g_start_block_id + groupID.x * BLOCK_IN_GROUP + blockInGroup;
  let threadBase: u32 = blockInGroup * MAX_USED_THREAD;
  let threadInBlock: u32 = GI - threadBase;

  let block_y = blockID / params.g_num_block_x;
  let block_x = blockID - block_y * params.g_num_block_x;
  let base_x = block_x * BLOCK_SIZE_X;
  let base_y = block_y * BLOCK_SIZE_Y;

  if (threadInBlock < 16u) {
    shared_temp[GI].pixel = loadPixel(base_x, base_y, threadInBlock);
  }
  workgroupBarrier();

  shared_temp[GI].error = 0xFFFFFFFFu;

  var num_partitions: u32;
  if (0u == params.g_mode_id) { num_partitions = 16u; } else { num_partitions = 64u; }

  var endPoint: array<mat2x4u, 3>;
  var endPointBackup: array<mat2x4u, 3>;
  var color_index: array<u32, 16>;
  if (threadInBlock < num_partitions) {
    let part = threadInBlock + 64u;
    endPoint[0][0] = vec4<u32>(MAX_UINT); endPoint[0][1] = vec4<u32>(MIN_UINT);
    endPoint[1][0] = vec4<u32>(MAX_UINT); endPoint[1][1] = vec4<u32>(MIN_UINT);
    endPoint[2][0] = vec4<u32>(MAX_UINT); endPoint[2][1] = vec4<u32>(MIN_UINT);
    let bits2 = candidateSectionBit2[part - 64u];
    for (var i = 0u; i < 16u; i++) {
      let pixel = shared_temp[threadBase + i].pixel;
      let subset_index = (bits2 >> (i * 2u)) & 0x03u;
      if (subset_index == 2u) {
        endPoint[2][0] = min(endPoint[2][0], pixel);
        endPoint[2][1] = max(endPoint[2][1], pixel);
      } else if (subset_index == 1u) {
        endPoint[1][0] = min(endPoint[1][0], pixel);
        endPoint[1][1] = max(endPoint[1][1], pixel);
      } else {
        endPoint[0][0] = min(endPoint[0][0], pixel);
        endPoint[0][1] = max(endPoint[0][1], pixel);
      }
    }
    endPointBackup[0] = endPoint[0];
    endPointBackup[1] = endPoint[1];
    endPointBackup[2] = endPoint[2];

    var max_p: u32;
    if (0u == params.g_mode_id) { max_p = 4u; } else { max_p = 1u; }

    var final_p = array<u32, 3>(0u, 0u, 0u);
    var errs = array<u32, 3>(MAX_UINT, MAX_UINT, MAX_UINT);
    for (var p = 0u; p < max_p; p++) {
      endPoint[0] = endPointBackup[0];
      endPoint[1] = endPointBackup[1];
      endPoint[2] = endPointBackup[2];

      for (var s = 0u; s < 3u; s++) {
        if (0u == params.g_mode_id) { compress_endpoints0(&endPoint[s], vec2<u32>(p, p >> 1u) & vec2<u32>(1u)); }
        else { compress_endpoints2(&endPoint[s]); }
      }

      var step_selector: u32 = 1u;
      if (2u == params.g_mode_id) { step_selector = 2u; }

      var span: array<vec4<i32>, 3>;
      span[0] = vec4<i32>(endPoint[0][1]) - vec4<i32>(endPoint[0][0]);
      span[1] = vec4<i32>(endPoint[1][1]) - vec4<i32>(endPoint[1][0]);
      span[2] = vec4<i32>(endPoint[2][1]) - vec4<i32>(endPoint[2][0]);
      span[0].w = 0; span[1].w = 0; span[2].w = 0;
      var sns = array<i32, 3>(dot4i(span[0], span[0]), dot4i(span[1], span[1]), dot4i(span[2], span[2]));

      let ci = array<u32, 3>(0u, candidateFixUpIndex1D[part].x, candidateFixUpIndex1D[part].y);
      for (var s = 0u; s < 3u; s++) {
        let dp = dot4i(span[s], vec4<i32>(shared_temp[threadBase + ci[s]].pixel) - vec4<i32>(endPoint[s][0]));
        if (sns[s] > 0 && dp > 0 && u32(f32(dp) * 63.49999) > u32(32 * sns[s])) {
          span[s] = -span[s];
          let t = endPoint[s][0]; endPoint[s][0] = endPoint[s][1]; endPoint[s][1] = t;
        }
      }

      var p_error = array<u32, 3>(0u, 0u, 0u);
      for (var i = 0u; i < 16u; i++) {
        let subset_index = (bits2 >> (i * 2u)) & 0x03u;
        let d = dot4i(span[subset_index], vec4<i32>(shared_temp[threadBase + i].pixel) - vec4<i32>(endPoint[subset_index][0]));
        let snsv = sns[subset_index];
        var cix: u32;
        if (snsv <= 0 || d <= 0) { cix = 0u; }
        else if (d < snsv) { cix = aStep[step_selector][u32(f32(d) * 63.49999 / f32(snsv))]; }
        else { cix = aStep[step_selector][63]; }
        color_index[i] = cix;

        var pixel_r = interp(endPoint[subset_index][0], endPoint[subset_index][1], aWeight[step_selector][cix]);
        pixel_r.w = 255u;

        let pixel = shared_temp[threadBase + i].pixel;
        let ord = Ensure_A_Is_Larger(pixel_r, pixel);
        let diff = ord[0] - ord[1];
        let pe = ComputeError(diff, diff);
        if (subset_index == 2u) { p_error[2] += pe; }
        else if (subset_index == 1u) { p_error[1] += pe; }
        else { p_error[0] += pe; }
      }

      for (var s = 0u; s < 3u; s++) {
        if (p_error[s] < errs[s]) { errs[s] = p_error[s]; final_p[s] = p; }
      }
    }

    shared_temp[GI].error = errs[0] + errs[1] + errs[2];
    shared_temp[GI].part = part;
    shared_temp[GI].rotation = (final_p[2] << 4u) | (final_p[1] << 2u) | final_p[0];
  }
  workgroupBarrier();

  for (var step = 32u; step >= 1u; step = step >> 1u) {
    if (threadInBlock < step) {
      if (shared_temp[GI].error > shared_temp[GI + step].error) {
        shared_temp[GI].error = shared_temp[GI + step].error;
        shared_temp[GI].part = shared_temp[GI + step].part;
        shared_temp[GI].rotation = shared_temp[GI + step].rotation;
      }
    }
    workgroupBarrier();
  }

  if (threadInBlock < 1u) {
    if (g_InBuff[blockID].x > shared_temp[GI].error) {
      g_OutBuff[blockID] = vec4<u32>(shared_temp[GI].error, params.g_mode_id, shared_temp[GI].part, shared_temp[GI].rotation);
    } else {
      g_OutBuff[blockID] = g_InBuff[blockID];
    }
  }
}

//------------------------------------------------------------------------------
// Pass 4: emit the final block for the winning mode
//------------------------------------------------------------------------------
@compute @workgroup_size(THREAD_GROUP_SIZE)
fn EncodeBlockCS(@builtin(local_invocation_index) GI: u32, @builtin(workgroup_id) groupID: vec3<u32>) {
  let MAX_USED_THREAD: u32 = 16u;
  let BLOCK_IN_GROUP: u32 = THREAD_GROUP_SIZE / MAX_USED_THREAD;
  let blockInGroup: u32 = GI / MAX_USED_THREAD;
  let blockID: u32 = params.g_start_block_id + groupID.x * BLOCK_IN_GROUP + blockInGroup;
  let threadBase: u32 = blockInGroup * MAX_USED_THREAD;
  let threadInBlock: u32 = GI - threadBase;

  let block_y = blockID / params.g_num_block_x;
  let block_x = blockID - block_y * params.g_num_block_x;
  let base_x = block_x * BLOCK_SIZE_X;
  let base_y = block_y * BLOCK_SIZE_Y;

  let mode = g_InBuff[blockID].y & 0x7FFFFFFFu;
  let part = g_InBuff[blockID].z;
  let index_selector = (g_InBuff[blockID].y >> 31u) & 1u;
  let rotation = g_InBuff[blockID].w;

  if (threadInBlock < 16u) {
    var pixel = loadPixel(base_x, base_y, threadInBlock);
    if ((4u == mode) || (5u == mode)) {
      if (1u == rotation) { pixel = vec4<u32>(pixel.w, pixel.y, pixel.z, pixel.x); }
      else if (2u == rotation) { pixel = vec4<u32>(pixel.x, pixel.w, pixel.z, pixel.y); }
      else if (3u == rotation) { pixel = vec4<u32>(pixel.x, pixel.y, pixel.w, pixel.z); }
    }
    shared_temp[GI].pixel = pixel;
  }
  workgroupBarrier();

  let bits = candidateSectionBit[part];
  let bits2 = candidateSectionBit2[part - 64u];

  var ep: mat2x4u;
  ep[0] = vec4<u32>(MAX_UINT);
  ep[1] = vec4<u32>(MIN_UINT);
  var ep_quantized: mat2x4u;

  for (var ii: i32 = 2; ii >= 0; ii = ii - 1) {
    if (threadInBlock < 16u) {
      var lep0 = vec4<u32>(MAX_UINT);
      var lep1 = vec4<u32>(MIN_UINT);
      let pixel = shared_temp[GI].pixel;
      let subset_index = (bits >> threadInBlock) & 0x01u;
      let subset_index2 = (bits2 >> (threadInBlock * 2u)) & 0x03u;
      if (0 == ii) {
        if ((0u == mode) || (2u == mode)) {
          if (0u == subset_index2) { lep0 = pixel; lep1 = pixel; }
        } else if ((1u == mode) || (3u == mode) || (7u == mode)) {
          if (0u == subset_index) { lep0 = pixel; lep1 = pixel; }
        } else if ((4u == mode) || (5u == mode) || (6u == mode)) {
          lep0 = pixel; lep1 = pixel;
        }
      } else if (1 == ii) {
        if ((0u == mode) || (2u == mode)) {
          if (1u == subset_index2) { lep0 = pixel; lep1 = pixel; }
        } else if ((1u == mode) || (3u == mode) || (7u == mode)) {
          if (1u == subset_index) { lep0 = pixel; lep1 = pixel; }
        }
      } else {
        if ((0u == mode) || (2u == mode)) {
          if (2u == subset_index2) { lep0 = pixel; lep1 = pixel; }
        }
      }
      shared_temp[GI].endPoint_low = lep0;
      shared_temp[GI].endPoint_high = lep1;
    }
    workgroupBarrier();

    if (threadInBlock < 8u) {
      shared_temp[GI].endPoint_low = min(shared_temp[GI].endPoint_low, shared_temp[GI + 8u].endPoint_low);
      shared_temp[GI].endPoint_high = max(shared_temp[GI].endPoint_high, shared_temp[GI + 8u].endPoint_high);
    }
    workgroupBarrier();
    if (threadInBlock < 4u) {
      shared_temp[GI].endPoint_low = min(shared_temp[GI].endPoint_low, shared_temp[GI + 4u].endPoint_low);
      shared_temp[GI].endPoint_high = max(shared_temp[GI].endPoint_high, shared_temp[GI + 4u].endPoint_high);
    }
    workgroupBarrier();
    if (threadInBlock < 2u) {
      shared_temp[GI].endPoint_low = min(shared_temp[GI].endPoint_low, shared_temp[GI + 2u].endPoint_low);
      shared_temp[GI].endPoint_high = max(shared_temp[GI].endPoint_high, shared_temp[GI + 2u].endPoint_high);
    }
    workgroupBarrier();
    if (threadInBlock < 1u) {
      shared_temp[GI].endPoint_low = min(shared_temp[GI].endPoint_low, shared_temp[GI + 1u].endPoint_low);
      shared_temp[GI].endPoint_high = max(shared_temp[GI].endPoint_high, shared_temp[GI + 1u].endPoint_high);
    }
    workgroupBarrier();

    if (ii == i32(threadInBlock)) {
      ep[0] = shared_temp[threadBase].endPoint_low;
      ep[1] = shared_temp[threadBase].endPoint_high;
    }
  }

  if (threadInBlock < 3u) {
    var P: vec2<u32>;
    if (1u == mode) { P = vec2<u32>((rotation >> threadInBlock) & 1u); }
    else { P = vec2<u32>(rotation >> (threadInBlock * 2u + 0u), rotation >> (threadInBlock * 2u + 1u)) & vec2<u32>(1u); }

    if (0u == mode) { ep_quantized = compress_endpoints0(&ep, P); }
    else if (1u == mode) { ep_quantized = compress_endpoints1(&ep, P); }
    else if (2u == mode) { ep_quantized = compress_endpoints2(&ep); }
    else if (3u == mode) { ep_quantized = compress_endpoints3(&ep, P); }
    else if (4u == mode) { ep_quantized = compress_endpoints4(&ep); }
    else if (5u == mode) { ep_quantized = compress_endpoints5(&ep); }
    else if (6u == mode) { ep_quantized = compress_endpoints6(&ep, P); }
    else { ep_quantized = compress_endpoints7(&ep, P); }

    var span = vec4<i32>(ep[1]) - vec4<i32>(ep[0]);
    if (mode < 4u) { span.w = 0; }

    if ((4u == mode) || (5u == mode)) {
      if (0u == threadInBlock) {
        let span_norm_sqr = vec2<i32>(dot3i(span.xyz, span.xyz), span.w * span.w);
        let p0 = shared_temp[threadBase + 0u].pixel;
        let dp = vec2<i32>(dot3i(span.xyz, vec3<i32>(p0.xyz) - vec3<i32>(ep[0].xyz)), span.w * (i32(p0.w) - i32(ep[0].w)));
        if (span_norm_sqr.x > 0 && dp.x > 0 && u32(f32(dp.x) * 63.49999) > u32(32 * span_norm_sqr.x)) {
          let t = ep[0]; let u = ep[1];
          ep[0] = vec4<u32>(u.xyz, t.w); ep[1] = vec4<u32>(t.xyz, u.w);
          let tq = ep_quantized[0]; let uq = ep_quantized[1];
          ep_quantized[0] = vec4<u32>(uq.xyz, tq.w); ep_quantized[1] = vec4<u32>(tq.xyz, uq.w);
        }
        if (span_norm_sqr.y > 0 && dp.y > 0 && u32(f32(dp.y) * 63.49999) > u32(32 * span_norm_sqr.y)) {
          let tw = ep[0].w; ep[0].w = ep[1].w; ep[1].w = tw;
          let tqw = ep_quantized[0].w; ep_quantized[0].w = ep_quantized[1].w; ep_quantized[1].w = tqw;
        }
      }
    } else {
      var pp: u32;
      if (0u == threadInBlock) { pp = 0u; }
      else if (1u == threadInBlock) { pp = candidateFixUpIndex1D[part].x; }
      else { pp = candidateFixUpIndex1D[part].y; }

      let span_norm_sqr = dot4i(span, span);
      let dp = dot4i(span, vec4<i32>(shared_temp[threadBase + pp].pixel) - vec4<i32>(ep[0]));
      if (span_norm_sqr > 0 && dp > 0 && u32(f32(dp) * 63.49999) > u32(32 * span_norm_sqr)) {
        let t = ep[0]; ep[0] = ep[1]; ep[1] = t;
        let tq = ep_quantized[0]; ep_quantized[0] = ep_quantized[1]; ep_quantized[1] = tq;
      }
    }

    shared_temp[GI].endPoint_low = ep[0];
    shared_temp[GI].endPoint_high = ep[1];
    shared_temp[GI].endPoint_low_quantized = ep_quantized[0];
    shared_temp[GI].endPoint_high_quantized = ep_quantized[1];
  }
  workgroupBarrier();

  if (threadInBlock < 16u) {
    var color_index = 0u;
    var alpha_index = 0u;

    var indexPrec: vec2<u32>;
    if ((0u == mode) || (1u == mode)) { indexPrec = vec2<u32>(1u); }
    else if (6u == mode) { indexPrec = vec2<u32>(0u); }
    else if (4u == mode) {
      if (0u == index_selector) { indexPrec = vec2<u32>(2u, 1u); } else { indexPrec = vec2<u32>(1u, 2u); }
    } else { indexPrec = vec2<u32>(2u); }

    var subset_index: u32;
    if ((0u == mode) || (2u == mode)) { subset_index = (bits2 >> (threadInBlock * 2u)) & 0x03u; }
    else if ((1u == mode) || (3u == mode) || (7u == mode)) { subset_index = (bits >> threadInBlock) & 0x01u; }
    else { subset_index = 0u; }

    var lep: mat2x4u;
    lep[0] = shared_temp[threadBase + subset_index].endPoint_low;
    lep[1] = shared_temp[threadBase + subset_index].endPoint_high;

    var span = vec4<i32>(lep[1]) - vec4<i32>(lep[0]);
    if (mode < 4u) { span.w = 0; }

    if ((4u == mode) || (5u == mode)) {
      let snsx = dot3i(span.xyz, span.xyz);
      let snsy = span.w * span.w;
      let px = shared_temp[threadBase + threadInBlock].pixel;
      let dpx = dot3i(span.xyz, vec3<i32>(px.xyz) - vec3<i32>(lep[0].xyz));
      if (snsx <= 0 || dpx <= 0) { color_index = 0u; }
      else if (dpx < snsx) { color_index = aStep[indexPrec.x][u32(f32(dpx) * 63.49999 / f32(snsx))]; }
      else { color_index = aStep[indexPrec.x][63]; }
      let dpy = span.w * (i32(px.w) - i32(lep[0].w));
      if (snsy <= 0 || dpy <= 0) { alpha_index = 0u; }
      else if (dpy < snsy) { alpha_index = aStep[indexPrec.y][u32(f32(dpy) * 63.49999 / f32(snsy))]; }
      else { alpha_index = aStep[indexPrec.y][63]; }
      if (index_selector != 0u) { let t = color_index; color_index = alpha_index; alpha_index = t; }
    } else {
      let sns = dot4i(span, span);
      let px = shared_temp[threadBase + threadInBlock].pixel;
      let dp = dot4i(span, vec4<i32>(px) - vec4<i32>(lep[0]));
      if (sns <= 0 || dp <= 0) { color_index = 0u; }
      else if (dp < sns) { color_index = aStep[indexPrec.x][u32(f32(dp) * 63.49999 / f32(sns))]; }
      else { color_index = aStep[indexPrec.x][63]; }
    }

    shared_temp[GI].error = color_index;
    shared_temp[GI].mode = alpha_index;
  }
  workgroupBarrier();

  if (0u == threadInBlock) {
    var block: vec4<u32>;
    if (0u == mode) { block = block_package0(part, threadBase); }
    else if (1u == mode) { block = block_package1(part, threadBase); }
    else if (2u == mode) { block = block_package2(part, threadBase); }
    else if (3u == mode) { block = block_package3(part, threadBase); }
    else if (4u == mode) { block = block_package4(rotation, index_selector, threadBase); }
    else if (5u == mode) { block = block_package5(rotation, threadBase); }
    else if (6u == mode) { block = block_package6(threadBase); }
    else { block = block_package7(part, threadBase); }

    g_OutBuff[blockID] = block;
  }
}
