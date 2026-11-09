// Đọc script.json + file mp3 → timeline.json (cho Remotion) và phụ đề .srt (bàn giao).
// Chạy: node scripts/build_timeline.mjs
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import assert from "node:assert";

const FPS = 30;
const LEAD = 0.4; // giây im lặng trước lời đọc mỗi cảnh
const TAIL = 0.7; // giây giữ hình sau lời đọc
const THINK = 5; // giây đếm ngược ở cảnh 20
const ROOT = new URL("..", import.meta.url).pathname.replace(
  /^\/([A-Za-z]:)/,
  "$1",
);
const SRT_OUT = `${ROOT}/../giai_doan_0/bai_0.01_ai_ml_dl/phu_de.srt`;

const script = JSON.parse(
  readFileSync(`${ROOT}/src/bai_0_01/script.json`, "utf8"),
);

const duration = (file) =>
  parseFloat(
    execFileSync(
      "npx",
      [
        "remotion",
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        file,
      ],
      {
        cwd: ROOT,
        shell: true,
      },
    )
      .toString()
      .trim(),
  );

export const clean = (t) =>
  t
    .replace(/\[[^\]]*\]\s*/g, "")
    .replace(/\s+/g, " ")
    .trim();

// Tách lời thành dòng phụ đề ≤ ~80 ký tự, ưu tiên cắt ở dấu câu.
export const chunk = (text, max = 80) => {
  const out = [];
  for (const sentence of clean(text).match(/[^.?!:]+[.?!:]*/g) ?? []) {
    let s = sentence.trim();
    while (s.length > max) {
      const cut = s.lastIndexOf(", ", max);
      const at = cut > 20 ? cut + 1 : s.lastIndexOf(" ", max);
      out.push(s.slice(0, at).trim());
      s = s.slice(at).trim();
    }
    if (s) out.push(s);
  }
  return out;
};

assert.deepStrictEqual(chunk("[pause] Một, hai. Ba?"), ["Một, hai.", "Ba?"]);
assert.ok(chunk("a ".repeat(100)).every((c) => c.length <= 80));

// cảnh 20 = s20a + đếm ngược + s20b; còn lại 1 file/cảnh
const groups = [];
for (const s of script) {
  if (s.id === "s20b") groups.at(-1).parts.push(s);
  else groups.push({ id: s.id.replace(/[ab]$/, ""), parts: [s] });
}

const scenes = [];
const captions = [];
let t = 0;
for (const g of groups) {
  const sceneStart = t;
  let local = LEAD;
  const audio = [];
  g.parts.forEach((p, i) => {
    const file = `${ROOT}/public/bai_0_01/audio/${p.id}.mp3`;
    if (!existsSync(file)) throw new Error(`Thiếu file âm thanh ${file}`);
    if (i > 0) local += THINK;
    const d = duration(file);
    audio.push({
      src: `bai_0_01/audio/${p.id}.mp3`,
      from: Math.round(local * FPS),
    });
    // thời gian mỗi dòng phụ đề tỉ lệ theo số ký tự (ước lượng, không phải căn chỉnh từng từ)
    const lines = chunk(p.tts);
    const total = lines.reduce((a, l) => a + l.length, 0);
    let c = sceneStart + local;
    for (const l of lines) {
      const len = (d * l.length) / total;
      captions.push({ start: c, end: c + len, text: l });
      c += len;
    }
    if (i === 0 && g.parts.length > 1) audio[0].endSec = local + d; // mốc bắt đầu đếm ngược
    local += d;
  });
  local += TAIL;
  const dur = Math.round(local * FPS);
  scenes.push({
    id: g.id,
    from: Math.round(sceneStart * FPS),
    dur,
    audio,
    countdownAt: audio[0].endSec
      ? Math.round(audio[0].endSec * FPS)
      : undefined,
  });
  t = sceneStart + dur / FPS;
}

const totalFrames = scenes.reduce((a, s) => a + s.dur, 0);
writeFileSync(
  `${ROOT}/src/bai_0_01/timeline.json`,
  JSON.stringify({ fps: FPS, totalFrames, scenes, captions }, null, 1),
);

const ts = (x) => {
  const ms = Math.round(x * 1000);
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${p(Math.floor(ms / 3600000))}:${p(Math.floor(ms / 60000) % 60)}:${p(Math.floor(ms / 1000) % 60)},${p(ms % 1000, 3)}`;
};
writeFileSync(
  SRT_OUT,
  captions
    .map((c, i) => `${i + 1}\n${ts(c.start)} --> ${ts(c.end)}\n${c.text}\n`)
    .join("\n"),
  "utf8",
);
console.log(
  `OK: ${scenes.length} cảnh, ${(totalFrames / FPS / 60).toFixed(2)} phút, ${captions.length} dòng phụ đề`,
);
