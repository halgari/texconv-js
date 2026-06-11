//-------------------------------------------------------------------------------------
// demo/main.ts — drag-drop UI driving the texconv-js pipeline.
//-------------------------------------------------------------------------------------

import {
  DXGI_FORMAT,
  convert,
  loadImageRGBA8,
  writeDDS,
  downloadBytes,
  type Image,
} from "../src/index.js";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const drop = $("drop");
const fileInput = $<HTMLInputElement>("file");
const formatSel = $<HTMLSelectElement>("format");
const convertBtn = $<HTMLButtonElement>("convert");
const preview = $<HTMLCanvasElement>("preview");
const logEl = $("log");
const gpuEl = $("gpu");

let current: Image | null = null;
let currentName = "texture";

function log(msg: string) {
  logEl.textContent += msg + "\n";
}

gpuEl.textContent = "gpu" in navigator ? "WebGPU available ✓" : "WebGPU NOT available — only uncompressed output works";

async function setImage(file: File) {
  logEl.textContent = "";
  currentName = file.name.replace(/\.[^.]+$/, "");
  log(`Loading ${file.name} …`);
  current = await loadImageRGBA8(file);
  log(`Decoded ${current.width}×${current.height} RGBA8 (${current.pixels.byteLength} bytes)`);
  drawPreview(current);
  convertBtn.disabled = false;
}

function drawPreview(img: Image) {
  preview.width = img.width;
  preview.height = img.height;
  const ctx = preview.getContext("2d")!;
  const clamped = new Uint8ClampedArray(img.pixels.subarray(0, img.width * img.height * 4));
  ctx.putImageData(new ImageData(clamped, img.width, img.height), 0, 0);
}

convertBtn.addEventListener("click", async () => {
  if (!current) return;
  try {
    const format = DXGI_FORMAT[formatSel.value as keyof typeof DXGI_FORMAT];
    log(`Converting to ${formatSel.value} …`);
    const t0 = performance.now();
    const scratch = await convert(current, { format });
    const dds = writeDDS(scratch);
    const ms = (performance.now() - t0).toFixed(1);
    log(`Done in ${ms} ms → ${dds.byteLength} byte .dds`);
    downloadBytes(dds, `${currentName}.dds`, "image/vnd-ms.dds");
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
