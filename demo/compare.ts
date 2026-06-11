//-------------------------------------------------------------------------------------
// demo/compare.ts — source vs decoded-output comparison viewer.
//
// Three modes: slider (output revealed right of a draggable divider), side-by-side,
// and an amplified per-pixel diff. The source is rescaled to the output's dimensions
// when they differ (e.g. after -w/-h), so pixels always align.
//-------------------------------------------------------------------------------------

import type { Image } from "../src/index.js";

export type CompareMode = "slider" | "side" | "diff";

const DIFF_AMPLIFY = 8;

/** Copy an Image's pixels (which may carry row padding) into tight ImageData. */
function toImageData(img: Image): ImageData {
  const tight = new Uint8ClampedArray(img.width * img.height * 4);
  for (let y = 0; y < img.height; y++) {
    tight.set(img.pixels.subarray(y * img.rowPitch, y * img.rowPitch + img.width * 4), y * img.width * 4);
  }
  return new ImageData(tight, img.width, img.height);
}

/** Rescale `src` to w×h using canvas drawImage (only called when dims differ). */
function rescale(src: ImageData, w: number, h: number): ImageData {
  const from = document.createElement("canvas");
  from.width = src.width;
  from.height = src.height;
  from.getContext("2d")!.putImageData(src, 0, 0);
  const to = document.createElement("canvas");
  to.width = w;
  to.height = h;
  const ctx = to.getContext("2d")!;
  ctx.drawImage(from, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

function diffImageData(a: ImageData, b: ImageData): ImageData {
  const out = new ImageData(a.width, a.height);
  for (let i = 0; i < a.data.length; i += 4) {
    out.data[i] = Math.min(255, Math.abs(a.data[i] - b.data[i]) * DIFF_AMPLIFY);
    out.data[i + 1] = Math.min(255, Math.abs(a.data[i + 1] - b.data[i + 1]) * DIFF_AMPLIFY);
    out.data[i + 2] = Math.min(255, Math.abs(a.data[i + 2] - b.data[i + 2]) * DIFF_AMPLIFY);
    out.data[i + 3] = 255;
  }
  return out;
}

/** PSNR over all four channels (matches the GPU test methodology). */
function psnr(a: ImageData, b: ImageData): number {
  let sse = 0;
  for (let i = 0; i < a.data.length; i++) {
    const d = a.data[i] - b.data[i];
    sse += d * d;
  }
  if (sse === 0) return Infinity;
  return 10 * Math.log10((255 * 255) / (sse / a.data.length));
}

function draw(canvas: HTMLCanvasElement, data: ImageData): void {
  canvas.width = data.width;
  canvas.height = data.height;
  canvas.getContext("2d")!.putImageData(data, 0, 0);
}

export class CompareViewer {
  private stage = document.getElementById("stage") as HTMLDivElement;
  private srcCanvas = document.getElementById("srcCanvas") as HTMLCanvasElement;
  private outWrap = document.getElementById("outWrap") as HTMLDivElement;
  private outCanvas = document.getElementById("outCanvas") as HTMLCanvasElement;
  private diffCanvas = document.getElementById("diffCanvas") as HTMLCanvasElement;
  private handle = document.getElementById("handle") as HTMLDivElement;
  private labelL = document.getElementById("labelL") as HTMLSpanElement;
  private labelR = document.getElementById("labelR") as HTMLSpanElement;
  private placeholder = document.getElementById("viewerPlaceholder") as HTMLDivElement;
  private buttons: Record<CompareMode, HTMLButtonElement> = {
    slider: document.getElementById("modeSlider") as HTMLButtonElement,
    side: document.getElementById("modeSide") as HTMLButtonElement,
    diff: document.getElementById("modeDiff") as HTMLButtonElement,
  };

  private mode: CompareMode = "slider";
  private sliderPos = 0.5;
  private hasOutput = false;

  constructor() {
    for (const mode of Object.keys(this.buttons) as CompareMode[]) {
      this.buttons[mode].addEventListener("click", () => this.setMode(mode));
    }
    // Drag anywhere on the stage to move the divider (slider mode only).
    this.stage.addEventListener("pointerdown", (e) => {
      if (this.mode !== "slider" || !this.hasOutput) return;
      this.stage.setPointerCapture(e.pointerId);
      this.moveSlider(e);
      const onMove = (ev: PointerEvent) => this.moveSlider(ev);
      const onUp = () => {
        this.stage.removeEventListener("pointermove", onMove);
        this.stage.removeEventListener("pointerup", onUp);
      };
      this.stage.addEventListener("pointermove", onMove);
      this.stage.addEventListener("pointerup", onUp);
    });
  }

  /** Show just the source image (after load, before any conversion). */
  showSource(src: Image): void {
    draw(this.srcCanvas, toImageData(src));
    this.stage.classList.toggle("small", src.width < 384);
    this.placeholder.hidden = true;
    this.stage.hidden = false;
    this.hasOutput = false;
    this.outWrap.hidden = true;
    this.handle.hidden = true;
    this.labelL.textContent = "SOURCE";
    this.labelR.textContent = "";
    for (const b of Object.values(this.buttons)) b.disabled = true;
  }

  /**
   * Show a source/output pair. Returns the PSNR of output vs (rescaled) source.
   */
  setImages(src: Image, out: Image, outLabel: string): number {
    let srcData = toImageData(src);
    if (src.width !== out.width || src.height !== out.height) {
      srcData = rescale(srcData, out.width, out.height);
      this.labelL.textContent = "SOURCE (scaled)";
    } else {
      this.labelL.textContent = "SOURCE";
    }
    const outData = toImageData(out);
    draw(this.srcCanvas, srcData);
    draw(this.outCanvas, outData);
    draw(this.diffCanvas, diffImageData(srcData, outData));
    this.stage.classList.toggle("small", out.width < 384);

    this.placeholder.hidden = true;
    this.stage.hidden = false;
    this.hasOutput = true;
    this.labelR.textContent = outLabel;
    for (const b of Object.values(this.buttons)) b.disabled = false;
    this.setMode(this.mode);
    return psnr(srcData, outData);
  }

  setMode(mode: CompareMode): void {
    this.mode = mode;
    for (const m of Object.keys(this.buttons) as CompareMode[]) {
      this.buttons[m].classList.toggle("active", m === mode);
    }
    this.stage.classList.toggle("side", mode === "side");
    this.stage.classList.toggle("diff", mode === "diff");

    this.srcCanvas.hidden = mode === "diff";
    this.outWrap.hidden = mode === "diff" || !this.hasOutput;
    this.diffCanvas.hidden = mode !== "diff";
    this.handle.hidden = mode !== "slider" || !this.hasOutput;
    this.labelL.hidden = mode === "diff";
    this.labelR.hidden = mode === "diff";

    if (mode === "slider") this.applySlider();
    else this.outCanvas.style.clipPath = "";
  }

  private moveSlider(e: PointerEvent): void {
    const rect = this.stage.getBoundingClientRect();
    this.sliderPos = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    this.applySlider();
  }

  private applySlider(): void {
    const pct = this.sliderPos * 100;
    this.outCanvas.style.clipPath = `inset(0 0 0 ${pct}%)`;
    this.handle.style.left = `calc(${pct}% - 1px)`;
  }
}
