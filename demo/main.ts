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

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const drop = $("drop");
const fileInput = $<HTMLInputElement>("file");
const formatSel = $<HTMLSelectElement>("format");
const convertBtn = $<HTMLButtonElement>("convert");
const preview = $<HTMLCanvasElement>("preview");
const outPreview = $<HTMLCanvasElement>("outpreview");
const logEl = $("log");
const gpuEl = $("gpu");

let current: Image | null = null;
let currentName = "texture";

function log(msg: string) {
  logEl.textContent += msg + "\n";
}

// Probe for an actual adapter, not just navigator.gpu — on Linux Chromium the API
// can be present while no adapter exists (WebGPU/Vulkan gated behind flags).
async function probeGPU() {
  if (!("gpu" in navigator)) {
    gpuEl.textContent = "WebGPU NOT available in this browser — only uncompressed output works";
    return;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      gpuEl.textContent = "WebGPU adapter ready ✓";
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
  log(`Loading ${file.name} …`);
  current = await loadImageRGBA8(file);
  log(`Decoded ${current.width}×${current.height} RGBA8 (${current.pixels.byteLength} bytes)`);
  drawTo(preview, current);
  convertBtn.disabled = false;
}

// Draw an R8G8B8A8 Image (which may carry row padding) into a canvas.
function drawTo(canvas: HTMLCanvasElement, img: Image) {
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext("2d")!;
  const tight = new Uint8ClampedArray(img.width * img.height * 4);
  for (let y = 0; y < img.height; y++) {
    tight.set(img.pixels.subarray(y * img.rowPitch, y * img.rowPitch + img.width * 4), y * img.width * 4);
  }
  ctx.putImageData(new ImageData(tight, img.width, img.height), 0, 0);
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
    const ms = (performance.now() - t0).toFixed(1);
    log(`Done in ${ms} ms → ${scratch.metadata.mipLevels} mip(s), ${dds.byteLength} byte .dds`);

    // Round-trip preview so the user can eyeball compression quality.
    if (format === DXGI_FORMAT.BC7_UNORM || format === DXGI_FORMAT.BC7_UNORM_SRGB) {
      drawTo(outPreview, decodeBC7(scratch.baseImage));
    } else if (isCompressed(format)) {
      drawTo(outPreview, decodeBCn(scratch.baseImage));
    } else {
      drawTo(outPreview, scratch.baseImage);
    }
    downloadBytes(dds, filename, "image/vnd-ms.dds");
  } catch (err) {
    log(`Error: ${(err as Error).message}`);
  }
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
