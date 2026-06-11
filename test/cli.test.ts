import { describe, it, expect } from "vitest";
import { DXGI_FORMAT } from "../src/dxgi.js";
import { parseTexconvArgs, formatFromName, outputFilename } from "../src/cli.js";

describe("formatFromName", () => {
  it("resolves canonical names and aliases", () => {
    expect(formatFromName("BC7_UNORM")).toBe(DXGI_FORMAT.BC7_UNORM);
    expect(formatFromName("bc3_unorm")).toBe(DXGI_FORMAT.BC3_UNORM);
    expect(formatFromName("DXT5")).toBe(DXGI_FORMAT.BC3_UNORM);
    expect(formatFromName("DXT1")).toBe(DXGI_FORMAT.BC1_UNORM);
    expect(formatFromName("DXGI_FORMAT_BC7_UNORM")).toBe(DXGI_FORMAT.BC7_UNORM);
    expect(formatFromName("nope")).toBeUndefined();
  });
});

describe("parseTexconvArgs", () => {
  it("parses a typical command", () => {
    const { options, files } = parseTexconvArgs(
      "-f BC7_UNORM -m 0 -w 256 -h 256 -aw 2 texture.png".split(" "),
    );
    expect(options.format).toBe(DXGI_FORMAT.BC7_UNORM);
    expect(options.mipLevels).toBe(0);
    expect(options.width).toBe(256);
    expect(options.height).toBe(256);
    expect(options.alphaWeight).toBe(2);
    expect(files).toEqual(["texture.png"]);
  });

  it("maps -bc flags and -srgb", () => {
    const { options } = parseTexconvArgs("-f BC7_UNORM -bc qx -srgb".split(" "));
    expect(options.bc7Quick).toBe(true);
    expect(options.bc7Use3Subsets).toBe(true);
    expect(options.srgbFilter).toBe(true);
  });

  it("collects warnings for unsupported flags", () => {
    const { warnings } = parseTexconvArgs("-f DXT5 -r -bc d -zzz".split(" "));
    expect(warnings.some((w) => w.includes("-r"))).toBe(true);
    expect(warnings.some((w) => w.includes("d"))).toBe(true);
    expect(warnings.some((w) => w.toLowerCase().includes("unrecognized"))).toBe(true);
  });

  it("builds output filenames with prefix/suffix/lowercase", () => {
    const { output } = parseTexconvArgs("-px pre_ -sx _suf -l".split(" "));
    expect(outputFilename("MyTex.PNG", output)).toBe("pre_mytex_suf.dds");
  });

  it("honors -ft for the output extension", () => {
    const { output } = parseTexconvArgs("-ft tga".split(" "));
    expect(outputFilename("tex.png", output)).toBe("tex.tga");
    const dotted = parseTexconvArgs("-ft .png".split(" ")).output;
    expect(outputFilename("tex.dds", dotted)).toBe("tex.png");
  });

  it("warns on missing/non-numeric -w/-m values instead of producing NaN", () => {
    const { options, warnings } = parseTexconvArgs("-w abc -h 64 -m".split(" "));
    expect(options.width).toBeUndefined();
    expect(options.height).toBe(64);
    expect(options.mipLevels).toBeUndefined();
    expect(warnings.some((w) => w.includes("-w width"))).toBe(true);
    expect(warnings.some((w) => w.includes("-m mip levels"))).toBe(true);
  });

  it("reports value-consuming passthrough flags as ignored, not unrecognized", () => {
    const { warnings, files } = parseTexconvArgs("-nits 100 -fl 12.1 in.png".split(" "));
    expect(files).toEqual(["in.png"]); // values consumed, not treated as inputs
    expect(warnings.every((w) => !w.toLowerCase().includes("unrecognized"))).toBe(true);
    expect(warnings.some((w) => w.includes("no effect"))).toBe(true);
  });
});
