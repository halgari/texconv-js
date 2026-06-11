// BC1/BC2/BC3/BC4/BC5 GPU encoders (WGSL). These have no GPU reference in
// DirectXTex (which compresses them on the CPU), so they are authored here.
//
// Strategy: one thread per 4x4 block (no shared memory / barriers). Endpoints come
// from the per-block bounding box with a small inset (the van Waveren real-time DXT
// approach); indices are nearest-palette. Palettes match the D3D/DirectXTex decode
// so the output is valid for any DDS consumer:
//   BC1 4-color: p0=c0, p1=c1, p2=(2c0+c1)/3, p3=(c0+2c1)/3
//   BC4 8-value: p0=r0, p1=r1, p2=(6r0+r1)/7 ... p7=(r0+6r1)/7  (r0>r1)

struct Constants {
  width: u32,
  height: u32,
  num_block_x: u32,
  num_total_blocks: u32,
};

@group(0) @binding(0) var<uniform> params: Constants;
@group(0) @binding(1) var g_Input: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> g_OutBuff: array<u32>;

fn loadTexel(bx: u32, by: u32, px: u32, py: u32) -> vec4<f32> {
  // Edge-clamp out-of-range samples (replicate) for partial edge blocks.
  let x = min(bx * 4u + px, params.width - 1u);
  let y = min(by * 4u + py, params.height - 1u);
  return clamp(textureLoad(g_Input, vec2<u32>(x, y), 0) * 255.0, vec4<f32>(0.0), vec4<f32>(255.0));
}

// --- BC1 color block -> (word0 = c0|c1<<16, word1 = 2-bit indices) ---
fn to565(c: vec3<f32>) -> u32 {
  let r = u32(round(clamp(c.r, 0.0, 255.0) / 255.0 * 31.0));
  let g = u32(round(clamp(c.g, 0.0, 255.0) / 255.0 * 63.0));
  let b = u32(round(clamp(c.b, 0.0, 255.0) / 255.0 * 31.0));
  return (r << 11u) | (g << 5u) | b;
}
fn expand565(w: u32) -> vec3<f32> {
  let r = (w >> 11u) & 31u;
  let g = (w >> 5u) & 63u;
  let b = w & 31u;
  return vec3<f32>(
    f32((r << 3u) | (r >> 2u)),
    f32((g << 2u) | (g >> 4u)),
    f32((b << 3u) | (b >> 2u)),
  );
}

fn encodeBC1Color(c: array<vec3<f32>, 16>) -> vec2<u32> {
  var mn = vec3<f32>(255.0);
  var mx = vec3<f32>(0.0);
  for (var i = 0u; i < 16u; i++) {
    mn = min(mn, c[i]);
    mx = max(mx, c[i]);
  }
  // Inset the bounding box slightly (reduces error for interior samples).
  let inset = (mx - mn) / 16.0;
  mn = min(mn + inset, vec3<f32>(255.0));
  mx = max(mx - inset, vec3<f32>(0.0));

  var c0 = to565(mx);
  var c1 = to565(mn);
  var e0 = expand565(c0);
  var e1 = expand565(c1);
  // 4-color mode requires c0 > c1. (c0 >= c1 always holds since mx >= mn.)
  if (c0 < c1) {
    let tc = c0; c0 = c1; c1 = tc;
    let te = e0; e0 = e1; e1 = te;
  }

  var pal: array<vec3<f32>, 4>;
  pal[0] = e0;
  pal[1] = e1;
  pal[2] = (2.0 * e0 + e1) / 3.0;
  pal[3] = (e0 + 2.0 * e1) / 3.0;

  var indices = 0u;
  for (var i = 0u; i < 16u; i++) {
    var best = 0u;
    var bestd = 1e30;
    for (var k = 0u; k < 4u; k++) {
      let d = c[i] - pal[k];
      let dist = dot(d, d);
      if (dist < bestd) { bestd = dist; best = k; }
    }
    indices |= best << (i * 2u);
  }
  return vec2<u32>(c0 | (c1 << 16u), indices);
}

// --- BC4 single-channel block -> (word0, word1) of the 8-byte block ---
fn encodeBC4(v: array<f32, 16>) -> vec2<u32> {
  var mn = 255.0;
  var mx = 0.0;
  for (var i = 0u; i < 16u; i++) {
    mn = min(mn, v[i]);
    mx = max(mx, v[i]);
  }
  let r0 = u32(round(mx)); // r0 >= r1
  let r1 = u32(round(mn));

  var pal: array<f32, 8>;
  pal[0] = f32(r0);
  pal[1] = f32(r1);
  for (var k = 1u; k < 7u; k++) {
    pal[k + 1u] = (f32(r0) * f32(7u - k) + f32(r1) * f32(k)) / 7.0;
  }

  var lo = r0 | (r1 << 8u);
  var hi = 0u;
  for (var p = 0u; p < 16u; p++) {
    var best = 0u;
    var bestd = 1e30;
    for (var k = 0u; k < 8u; k++) {
      let d = abs(pal[k] - v[p]);
      if (d < bestd) { bestd = d; best = k; }
    }
    let bitpos = 16u + p * 3u; // indices begin at bit 16
    if (bitpos < 32u) {
      lo |= best << bitpos;
      if (bitpos + 3u > 32u) { hi |= best >> (32u - bitpos); }
    } else {
      hi |= best << (bitpos - 32u);
    }
  }
  return vec2<u32>(lo, hi);
}

// --- BC2 explicit 4-bit alpha block -> (word0, word1) ---
fn encodeBC2Alpha(a: array<f32, 16>) -> vec2<u32> {
  var lo = 0u;
  var hi = 0u;
  for (var p = 0u; p < 16u; p++) {
    let a4 = u32(round(clamp(a[p], 0.0, 255.0) / 255.0 * 15.0));
    if (p < 8u) { lo |= a4 << (p * 4u); } else { hi |= a4 << ((p - 8u) * 4u); }
  }
  return vec2<u32>(lo, hi);
}

fn loadColorBlock(bx: u32, by: u32) -> array<vec3<f32>, 16> {
  var c: array<vec3<f32>, 16>;
  for (var p = 0u; p < 16u; p++) {
    c[p] = loadTexel(bx, by, p % 4u, p / 4u).xyz;
  }
  return c;
}

@compute @workgroup_size(64)
fn encodeBC1(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockID = gid.x;
  if (blockID >= params.num_total_blocks) { return; }
  let bx = blockID % params.num_block_x;
  let by = blockID / params.num_block_x;
  let r = encodeBC1Color(loadColorBlock(bx, by));
  g_OutBuff[blockID * 2u] = r.x;
  g_OutBuff[blockID * 2u + 1u] = r.y;
}

@compute @workgroup_size(64)
fn encodeBC2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockID = gid.x;
  if (blockID >= params.num_total_blocks) { return; }
  let bx = blockID % params.num_block_x;
  let by = blockID / params.num_block_x;
  var a: array<f32, 16>;
  for (var p = 0u; p < 16u; p++) { a[p] = loadTexel(bx, by, p % 4u, p / 4u).w; }
  let alpha = encodeBC2Alpha(a);
  let color = encodeBC1Color(loadColorBlock(bx, by));
  let o = blockID * 4u;
  g_OutBuff[o] = alpha.x;
  g_OutBuff[o + 1u] = alpha.y;
  g_OutBuff[o + 2u] = color.x;
  g_OutBuff[o + 3u] = color.y;
}

@compute @workgroup_size(64)
fn encodeBC3(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockID = gid.x;
  if (blockID >= params.num_total_blocks) { return; }
  let bx = blockID % params.num_block_x;
  let by = blockID / params.num_block_x;
  var a: array<f32, 16>;
  for (var p = 0u; p < 16u; p++) { a[p] = loadTexel(bx, by, p % 4u, p / 4u).w; }
  let alpha = encodeBC4(a);
  let color = encodeBC1Color(loadColorBlock(bx, by));
  let o = blockID * 4u;
  g_OutBuff[o] = alpha.x;
  g_OutBuff[o + 1u] = alpha.y;
  g_OutBuff[o + 2u] = color.x;
  g_OutBuff[o + 3u] = color.y;
}

@compute @workgroup_size(64)
fn encodeBC4Red(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockID = gid.x;
  if (blockID >= params.num_total_blocks) { return; }
  let bx = blockID % params.num_block_x;
  let by = blockID / params.num_block_x;
  var r: array<f32, 16>;
  for (var p = 0u; p < 16u; p++) { r[p] = loadTexel(bx, by, p % 4u, p / 4u).x; }
  let red = encodeBC4(r);
  g_OutBuff[blockID * 2u] = red.x;
  g_OutBuff[blockID * 2u + 1u] = red.y;
}

@compute @workgroup_size(64)
fn encodeBC5(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blockID = gid.x;
  if (blockID >= params.num_total_blocks) { return; }
  let bx = blockID % params.num_block_x;
  let by = blockID / params.num_block_x;
  var r: array<f32, 16>;
  var g: array<f32, 16>;
  for (var p = 0u; p < 16u; p++) {
    let t = loadTexel(bx, by, p % 4u, p / 4u);
    r[p] = t.x;
    g[p] = t.y;
  }
  let red = encodeBC4(r);
  let green = encodeBC4(g);
  let o = blockID * 4u;
  g_OutBuff[o] = red.x;
  g_OutBuff[o + 1u] = red.y;
  g_OutBuff[o + 2u] = green.x;
  g_OutBuff[o + 3u] = green.y;
}
