import React from "react";
import { C, MONO } from "./ui";

const KW = /^(def|if|in|return|for|import|from)$/;

// Tô màu tối giản: chuỗi (SPAM cam, Bình thường xanh lá), từ khóa, chú thích.
const tokens = (line: string) => {
  const out: { t: string; c: string }[] = [];
  const re = /("[^"]*"?)|(#.*$)|([A-Za-z_]+)|(\s+)|(.)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(line))) {
    const [t] = m;
    let c = C.text;
    if (m[1]) c = t.includes("SPAM") ? C.spam : t.includes("Bình thường") ? C.ok : "#A5F3FC";
    else if (m[2]) c = C.muted;
    else if (m[3] && KW.test(t)) c = "#F472B6";
    else if (m[3] && /^(CountVectorizer|MultinomialNB|fit|fit_transform)$/.test(t)) c = C.ai;
    out.push({ t, c });
  }
  return out;
};

/**
 * Khối code có hiệu ứng gõ chữ.
 * `progress` 0→1 = tỉ lệ ký tự đã hiện; `hl` = chỉ số dòng được tô nền.
 */
export const CodeBlock: React.FC<{ code: string; progress: number; hl?: Record<number, string>; size?: number }> = ({
  code,
  progress,
  hl = {},
  size = 40,
}) => {
  const lines = code.split("\n");
  let budget = Math.floor(code.length * progress);
  return (
    <div
      style={{
        background: "#020617",
        border: `3px solid ${C.line}`,
        borderRadius: 24,
        padding: "32px 40px",
        fontFamily: MONO,
        fontSize: size,
        lineHeight: 1.5,
        whiteSpace: "pre",
      }}
    >
      {lines.map((line, i) => {
        const show = Math.max(0, Math.min(line.length, budget));
        budget -= line.length + 1;
        let left = show;
        return (
          <div key={i} style={{ background: hl[i] ?? "transparent", borderRadius: 8, minHeight: size * 1.5, padding: "0 8px" }}>
            {tokens(line).map((tk, j) => {
              const part = tk.t.slice(0, Math.max(0, left));
              left -= tk.t.length;
              return (
                <span key={j} style={{ color: tk.c }}>
                  {part}
                </span>
              );
            })}
          </div>
        );
      })}
    </div>
  );
};
