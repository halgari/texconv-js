//-------------------------------------------------------------------------------------
// Unit tests for the CPU transforms (-hflip/-vflip/-swizzle/-c/-pmalpha/-alpha),
// the BGRA output path, and the -srgb flag semantics.
//-------------------------------------------------------------------------------------

import { describe, expect, it } from "vitest";
import { DXGI_FORMAT } from "../src/dxgi.js";
import { makeImage } from "../src/image.js";
import { colorKeyImage, flipImage, parseSwizzle, premultiplyAlpha, swizzleImage } from "../src/transform.js";
import { convert } from "../src/texconv.js";
import { parseTexconvArgs } from "../src/cli.js";

function img2x2(pixels: number[]): ReturnType<typeof makeImage> {
  const img = makeImage(DXGI_FORMAT.R8G8B8A8_UNORM, 2, 2);
  img.pixels.set(pixels);
  return img;
}

// Distinct pixel per corner: [TL, TR, BL, BR] with r encoding position.
const corners = img2x2([
  10, 0, 0, 255,  20, 0, 0, 255,
  30, 0, 0, 255,  40, 0, 0, 255,
]);

describe("flipImage", () => {
  it("hflip mirrors columns", () => {
    const f = flipImage(corners, true, false);
    expect([f.pixels[0], f.pixels[4], f.pixels[8], f.pixels[12]]).toEqual([20, 10, 40, 30]);
  });
  it("vflip mirrors rows", () => {
    const f = flipImage(corners, false, true);
    expect([f.pixels[0], f.pixels[4], f.pixels[8], f.pixels[12]]).toEqual([30, 40, 10, 20]);
  });
  it("works on float (HDR) images too", () => {
    const hdr = makeImage(DXGI_FORMAT.R32G32B32A32_FLOAT, 2, 1);
    new Float32Array(hdr.pixels.buffer).set([1, 0, 0, 1, 2, 0, 0, 1]);
    const f = new Float32Array(flipImage(hdr, true, false).pixels.buffer);
    expect(f[0]).toBe(2);
    expect(f[4]).toBe(1);
  });
});

describe("swizzle", () => {
  it("parses masks with upstream fill-forward semantics", () => {
    expect(parseSwizzle("abgr")).toEqual([3, 2, 1, 0]);
    expect(parseSwizzle("rgb1")).toEqual([0, 1, 2, 5]);
    expect(parseSwizzle("r")).toEqual([0, 0, 0, 0]); // short mask replicates
    expect(parseSwizzle("xyzw")).toEqual([0, 1, 2, 3]);
    expect(() => parseSwizzle("rq")).toThrow(/invalid/);
  });
  it("rearranges channels with constants", () => {
    const img = img2x2([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    const s = swizzleImage(img, "bg01");
    expect([...s.pixels.subarray(0, 4)]).toEqual([3, 2, 0, 255]);
  });
});

describe("colorKeyImage", () => {
  it("keys matching pixels to transparent black and forces others opaque", () => {
    const img = img2x2([
      0, 255, 0, 200,   // pure green — keyed
      10, 250, 5, 200,  // within 0.2 tolerance — keyed
      200, 0, 0, 200,   // far — kept, alpha forced to 255
      0, 0, 255, 200,
    ]);
    const k = colorKeyImage(img, 0x00ff00);
    expect([...k.pixels.subarray(0, 4)]).toEqual([0, 0, 0, 0]);
    expect([...k.pixels.subarray(4, 8)]).toEqual([0, 0, 0, 0]);
    expect([...k.pixels.subarray(8, 12)]).toEqual([200, 0, 0, 255]);
  });
});

describe("premultiplyAlpha", () => {
  it("premultiplies and undoes", () => {
    const img = img2x2([200, 100, 50, 128, 0, 0, 0, 0, 255, 255, 255, 255, 10, 20, 30, 64]);
    const pm = premultiplyAlpha(img);
    expect(pm.pixels[0]).toBe(Math.round((200 * 128) / 255));
    expect(pm.pixels[3]).toBe(128); // alpha unchanged
    const undone = premultiplyAlpha(pm, true);
    expect(Math.abs(undone.pixels[0] - 200)).toBeLessThanOrEqual(1);
  });
});

describe("convert integration (CPU-only paths)", () => {
  it("-f BGRA produces B8G8R8A8 with swapped channels", async () => {
    const img = img2x2([10, 20, 30, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const scratch = await convert(img, { format: DXGI_FORMAT.B8G8R8A8_UNORM });
    expect(scratch.metadata.format).toBe(DXGI_FORMAT.B8G8R8A8_UNORM);
    expect([...scratch.baseImage.pixels.subarray(0, 4)]).toEqual([30, 20, 10, 40]);
  });

  it("applies hflip + swizzle through convert()", async () => {
    const scratch = await convert(corners, {
      format: DXGI_FORMAT.R8G8B8A8_UNORM,
      hflip: true,
      swizzle: "gr",
    });
    // hflip puts 20 at x=0; swizzle "gr" -> [g, r, r, r] => pixel (20,0,..) -> (0,20,20,20)
    expect([...scratch.baseImage.pixels.subarray(0, 4)]).toEqual([0, 20, 20, 20]);
  });
});

describe("srgb flag semantics", () => {
  it("-srgbi sets filter only; -srgbo retags output; -srgb does both", () => {
    const i = parseTexconvArgs("-f BC3_UNORM -srgbi".split(" "));
    expect(i.options.srgbFilter).toBe(true);
    expect(i.options.format).toBe(DXGI_FORMAT.BC3_UNORM);

    const o = parseTexconvArgs("-f BC3_UNORM -srgbo".split(" "));
    expect(o.options.srgbFilter).toBeUndefined();
    expect(o.options.format).toBe(DXGI_FORMAT.BC3_UNORM_SRGB);

    const both = parseTexconvArgs("-srgb -f BC7_UNORM".split(" ")); // -f after, still promoted
    expect(both.options.srgbFilter).toBe(true);
    expect(both.options.format).toBe(DXGI_FORMAT.BC7_UNORM_SRGB);
  });

  it("parses the new transform flags", () => {
    const { options, warnings } = parseTexconvArgs(
      "-f BC7_UNORM -hflip -vflip -swizzle abgr -c 00FF00 -pmalpha".split(" "),
    );
    expect(options.hflip).toBe(true);
    expect(options.vflip).toBe(true);
    expect(options.swizzle).toBe("abgr");
    expect(options.colorKey).toBe(0x00ff00);
    expect(options.premultiplyAlpha).toBe(true);
    expect(warnings).toEqual([]);
  });

  it("warns on bad swizzle/colorkey and pmalpha+alpha conflicts", () => {
    const { options, warnings } = parseTexconvArgs("-swizzle rq -c zzz -pmalpha -alpha".split(" "));
    expect(warnings.some((w) => w.includes("invalid mask"))).toBe(true);
    expect(warnings.some((w) => w.includes("colorkey"))).toBe(true);
    expect(warnings.some((w) => w.includes("mutually exclusive"))).toBe(true);
    expect(options.straightAlpha).toBeUndefined();
  });
});
