// Chụp ảnh tĩnh ở các mốc trong từng cảnh để kiểm tra bố cục.
// Chạy: node scripts/stills.mjs <thư_mục_ra> [s01,s05,...]
import { bundle } from "@remotion/bundler";
import { renderStill, selectComposition } from "@remotion/renderer";
import { readFileSync, mkdirSync } from "node:fs";
import path from "node:path";

const out = process.argv[2];
const only = process.argv[3]?.split(",");
mkdirSync(out, { recursive: true });
const tl = JSON.parse(readFileSync("src/bai_0_01/timeline.json", "utf8"));
const serveUrl = await bundle({ entryPoint: path.resolve("src/index.ts") });
const composition = await selectComposition({
  serveUrl,
  id: "Bai0-01-AI-ML-DL",
});
for (const s of tl.scenes) {
  if (only && !only.includes(s.id)) continue;
  for (const p of [0.35, 0.97]) {
    const frame = s.from + Math.round(s.dur * p);
    await renderStill({
      serveUrl,
      composition,
      frame,
      output: path.join(out, `${s.id}_${Math.round(p * 100)}.jpeg`),
      imageFormat: "jpeg",
      scale: 0.5,
    });
  }
  console.log("ok", s.id);
}
