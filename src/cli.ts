//-------------------------------------------------------------------------------------
// cli.ts
//
// Parses a subset of the texconv command-line option surface into ConvertOptions,
// so the same flags users know from texconv drive texconv-js. File-system oriented
// options (recursion, file lists, output directories) are parsed where they affect
// the output name and otherwise reported as warnings.
//-------------------------------------------------------------------------------------

import { DXGI_FORMAT } from "./dxgi.js";
import type { ConvertOptions } from "./texconv.js";

/** Resolve a texconv -f format name (case-insensitive, with aliases) to a DXGI_FORMAT. */
export function formatFromName(name: string): DXGI_FORMAT | undefined {
  const key = name.toUpperCase().replace(/^DXGI_FORMAT_/, "");
  const aliases: Record<string, DXGI_FORMAT> = {
    DXT1: DXGI_FORMAT.BC1_UNORM,
    DXT2: DXGI_FORMAT.BC2_UNORM,
    DXT3: DXGI_FORMAT.BC2_UNORM,
    DXT4: DXGI_FORMAT.BC3_UNORM,
    DXT5: DXGI_FORMAT.BC3_UNORM,
    RGBA: DXGI_FORMAT.R8G8B8A8_UNORM,
    BGRA: DXGI_FORMAT.B8G8R8A8_UNORM,
  };
  if (key in aliases) return aliases[key];
  const v = (DXGI_FORMAT as unknown as Record<string, number>)[key];
  return typeof v === "number" ? (v as DXGI_FORMAT) : undefined;
}

export interface ParsedCommand {
  options: ConvertOptions;
  /** Output filename pieces (texconv -px/-sx/-o/-l/-ft). */
  output: { prefix: string; suffix: string; outputDir?: string; toLower: boolean; fileType?: string };
  /** Positional input file arguments. */
  files: string[];
  /** Non-fatal notes about ignored/unsupported flags. */
  warnings: string[];
}

// Flags that consume the following token as their value.
const VALUE_FLAGS = new Set([
  "f", "w", "h", "m", "aw", "bc", "if", "px", "sx", "o", "ft", "fl", "gpu",
  "wicq", "at", "nmapamp", "c", "nits", "swizzle", "fl",
]);

/** Parse texconv-style argv (without the program name) into a ParsedCommand. */
export function parseTexconvArgs(argv: string[]): ParsedCommand {
  const options: ConvertOptions = { format: DXGI_FORMAT.R8G8B8A8_UNORM };
  const output: ParsedCommand["output"] = { prefix: "", suffix: "", toLower: false };
  const files: string[] = [];
  const warnings: string[] = [];

  for (let i = 0; i < argv.length; i++) {
    const tok = argv[i];
    if (tok[0] !== "-" && tok[0] !== "/") {
      files.push(tok);
      continue;
    }
    const name = tok.slice(1).toLowerCase();
    const takesValue = VALUE_FLAGS.has(name);
    const value = takesValue ? argv[++i] : undefined;

    switch (name) {
      case "f": {
        const fmt = value ? formatFromName(value) : undefined;
        if (fmt === undefined) warnings.push(`Unknown format '${value}'`);
        else options.format = fmt;
        break;
      }
      case "w": options.width = Number(value); break;
      case "h": options.height = Number(value); break;
      case "m": options.mipLevels = Number(value); break;
      case "aw": options.alphaWeight = Number(value); break;
      case "srgb": case "srgbi": case "srgbo": options.srgbFilter = true; break;
      case "bc": {
        for (const ch of value ?? "") {
          if (ch === "q") options.bc7Quick = true;
          else if (ch === "x") options.bc7Use3Subsets = true;
          else warnings.push(`-bc '${ch}' not supported (ignored)`);
        }
        break;
      }
      case "if":
        if (value && !/^(point|linear|box|fant)$/i.test(value)) warnings.push(`Filter '${value}' approximated by box`);
        break;
      case "px": output.prefix = value ?? ""; break;
      case "sx": output.suffix = value ?? ""; break;
      case "o": output.outputDir = value; break;
      case "l": output.toLower = true; break;
      case "ft": output.fileType = value; break;
      case "y": break; // overwrite — implicit in browser
      case "nologo": case "timing": case "gpu": case "nogpu": break; // GPU is implicit
      case "r": case "flist": case "singleproc": case "nowic":
        warnings.push(`-${name} has no effect in the browser build`);
        break;
      default:
        warnings.push(`Unrecognized option -${name}${takesValue ? ` ${value}` : ""}`);
    }
  }

  return { options, output, files, warnings };
}

/** Build an output filename from the parsed naming options (DDS only for now). */
export function outputFilename(inputName: string, output: ParsedCommand["output"]): string {
  let base = inputName.replace(/\.[^.]+$/, "");
  if (output.toLower) base = base.toLowerCase();
  const ext = output.fileType ? output.fileType.replace(/^\./, "") : "dds";
  return `${output.prefix}${base}${output.suffix}.${ext}`;
}
