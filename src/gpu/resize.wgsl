// GPU box-filter resize / mip downsample. One thread per destination texel; each
// averages its source footprint. Optionally linearizes sRGB before averaging (and
// re-encodes after) so mip colors are correct.

struct Constants {
  srcW: u32,
  srcH: u32,
  dstW: u32,
  dstH: u32,
  srgb: u32, // 1 = treat data as sRGB during averaging
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Constants;
@group(0) @binding(1) var g_Input: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> g_Out: array<u32>; // packed RGBA8
@group(0) @binding(3) var<storage, read_write> g_OutF: array<vec4<f32>>; // float path

fn srgbToLinear(c: vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
  return select(hi, lo, c <= vec3<f32>(0.04045));
}
fn linearToSrgb(c: vec3<f32>) -> vec3<f32> {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn packRGBA8(c: vec4<f32>) -> u32 {
  let q = vec4<u32>(round(clamp(c, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0));
  return q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
}

// Box-filtered source average for one destination texel (shared by both paths).
fn boxAverage(x: u32, y: u32) -> vec4<f32> {
  var fx0 = (x * params.srcW) / params.dstW;
  var fy0 = (y * params.srcH) / params.dstH;
  var fx1 = ((x + 1u) * params.srcW) / params.dstW;
  var fy1 = ((y + 1u) * params.srcH) / params.dstH;
  fx1 = max(fx1, fx0 + 1u);
  fy1 = max(fy1, fy0 + 1u);
  fx1 = min(fx1, params.srcW);
  fy1 = min(fy1, params.srcH);

  var accum = vec4<f32>(0.0);
  var n = 0.0;
  for (var sy = fy0; sy < fy1; sy++) {
    for (var sx = fx0; sx < fx1; sx++) {
      let t = textureLoad(g_Input, vec2<u32>(sx, sy), 0);
      var rgb = t.rgb;
      if (params.srgb != 0u) { rgb = srgbToLinear(rgb); }
      accum += vec4<f32>(rgb, t.a);
      n += 1.0;
    }
  }
  return accum / max(n, 1.0);
}

@compute @workgroup_size(8, 8)
fn resize(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.dstW || y >= params.dstH) { return; }

  let accum = boxAverage(x, y);
  var rgb = accum.rgb;
  if (params.srgb != 0u) { rgb = linearToSrgb(rgb); }

  g_Out[y * params.dstW + x] = packRGBA8(vec4<f32>(rgb, accum.a));
}

// Float (HDR) path: rgba32float input, raw vec4f output, always linear (srgb=0).
@compute @workgroup_size(8, 8)
fn resize_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.dstW || y >= params.dstH) { return; }
  g_OutF[y * params.dstW + x] = boxAverage(x, y);
}
