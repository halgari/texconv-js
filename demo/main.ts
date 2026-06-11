//-------------------------------------------------------------------------------------
// demo/main.ts — drag-drop UI driving the texconv-js pipeline.
//-------------------------------------------------------------------------------------

import {
  DXGI_FORMAT,
  convert,
  loadImageRGBA8,
  writeDDS,
  downloadBytes,
  parseTexconvArgs,
  outputFilename,
  type ConvertOptions,
  type Image,
} from "../src/index.js";
import { decodeBC7 } from "../src/bc7_decode.js";
import { decodeBCn } from "../src/bcn_decode.js";
import { isCompressed } from "../src/dxgi.js";
import { CompareViewer } from "./compare.js";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const drop = $("drop");
const fileInput = $<HTMLInputElement>("file");
const formatSel = $<HTMLSelectElement>("format");
const convertBtn = $<HTMLButtonElement>("convert");
const downloadBtn = $<HTMLButtonElement>("download");
const logEl = $("log");
const gpuEl = $("gpu");
const hintEl = $("viewerHint");
const statsEl = $("stats");

const viewer = new CompareViewer();

let current: Image | null = null;
let currentName = "texture";
let currentFileBytes = 0;
let lastDds: Uint8Array | null = null;
let lastFilename = "texture.dds";

function log(msg: string) {
  logEl.textContent += msg + "\n";
}

function fmtBytes(n: number): string {
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(2)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}

// Probe for an actual adapter, not just navigator.gpu — on Linux Chromium the API
// can be present while no adapter exists (WebGPU/Vulkan gated behind flags).
async function probeGPU() {
  if (!("gpu" in navigator)) {
    gpuEl.textContent = "WebGPU NOT available — only uncompressed output works";
    return;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      gpuEl.textContent = "● WebGPU ready";
      gpuEl.classList.add("ok");
    } else {
      gpuEl.innerHTML =
        "WebGPU present but <b>no GPU adapter</b> — on Linux Chromium enable " +
        "<code>chrome://flags/#enable-unsafe-webgpu</code> (and <code>#enable-vulkan</code>), " +
        "restart, and check <code>chrome://gpu</code>.";
    }
  } catch (e) {
    gpuEl.textContent = `WebGPU adapter error: ${(e as Error).message}`;
  }
}
void probeGPU();

async function setImage(file: File) {
  logEl.textContent = "";
  currentName = file.name.replace(/\.[^.]+$/, "");
  currentFileBytes = file.size;
  log(`Loading ${file.name} …`);
  current = await loadImageRGBA8(file);
  log(`Decoded ${current.width}×${current.height} RGBA8 (${fmtBytes(current.pixels.byteLength)})`);
  viewer.showSource(current);
  statsEl.classList.remove("show");
  hintEl.textContent = "convert to compare";
  convertBtn.disabled = false;
  lastDds = null;
  downloadBtn.disabled = true;
}

const mipsCb = $<HTMLInputElement>("mips");
const rwInput = $<HTMLInputElement>("rw");
const rhInput = $<HTMLInputElement>("rh");
const argsInput = $<HTMLInputElement>("args");

// Build ConvertOptions from either the texconv-args box (if non-empty) or the
// individual UI controls.
function buildOptions(): { options: ConvertOptions; filename: string } {
  const args = argsInput.value.trim();
  if (args) {
    const parsed = parseTexconvArgs(args.split(/\s+/));
    for (const w of parsed.warnings) log(`warning: ${w}`);
    return { options: parsed.options, filename: outputFilename(currentName, parsed.output) };
  }
  const options: ConvertOptions = {
    format: DXGI_FORMAT[formatSel.value as keyof typeof DXGI_FORMAT],
  };
  if (mipsCb.checked) options.mipLevels = 0;
  if (rwInput.value) options.width = Number(rwInput.value);
  if (rhInput.value) options.height = Number(rhInput.value);
  return { options, filename: `${currentName}.dds` };
}

convertBtn.addEventListener("click", async () => {
  if (!current) return;
  try {
    const { options, filename } = buildOptions();
    const format = options.format;
    log(`Converting to ${DXGI_FORMAT[format]}${options.mipLevels === 0 ? " +mips" : ""} …`);
    const t0 = performance.now();
    const scratch = await convert(current, options);
    const dds = writeDDS(scratch);
    const ms = performance.now() - t0;
    log(`Done in ${ms.toFixed(1)} ms → ${scratch.metadata.mipLevels} mip(s), ${fmtBytes(dds.byteLength)} .dds`);

    // Round-trip the base level so the viewer compares what the GPU actually encoded.
    let decoded: Image;
    if (format === DXGI_FORMAT.BC7_UNORM || format === DXGI_FORMAT.BC7_UNORM_SRGB) {
      decoded = decodeBC7(scratch.baseImage);
    } else if (isCompressed(format)) {
      decoded = decodeBCn(scratch.baseImage);
    } else {
      decoded = scratch.baseImage;
    }
    const quality = viewer.setImages(current, decoded, DXGI_FORMAT[format]);
    hintEl.textContent = "drag the divider to compare";

    // Compression differential: the .dds payload vs the same images uncompressed.
    const rawBytes = scratch.images.reduce((sum, img) => sum + img.width * img.height * 4, 0);
    const saved = 1 - dds.byteLength / rawBytes;
    $("statSaved").textContent =
      saved > 0.005
        ? `▼ ${(saved * 100).toFixed(1)}% smaller`
        : "no compression (uncompressed format)";
    $("statInput").textContent = fmtBytes(currentFileBytes);
    $("statRaw").textContent = fmtBytes(rawBytes);
    $("statDds").textContent = fmtBytes(dds.byteLength);
    $("statRatio").textContent = `${(rawBytes / dds.byteLength).toFixed(2)} : 1`;
    $("statPsnr").textContent = isCompressed(format)
      ? (quality === Infinity ? "lossless" : `${quality.toFixed(1)} dB`)
      : "lossless";
    $("statTime").textContent = `${ms.toFixed(1)} ms`;
    statsEl.classList.add("show");

    lastDds = dds;
    lastFilename = filename;
    downloadBtn.disabled = false;
  } catch (err) {
    log(`Error: ${(err as Error).message}`);
  }
});

downloadBtn.addEventListener("click", () => {
  if (lastDds) downloadBytes(lastDds, lastFilename, "image/vnd-ms.dds");
});

fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (f) void setImage(f);
});

drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("hover");
});
drop.addEventListener("dragleave", () => drop.classList.remove("hover"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("hover");
  const f = e.dataTransfer?.files?.[0];
  if (f) void setImage(f);
});
