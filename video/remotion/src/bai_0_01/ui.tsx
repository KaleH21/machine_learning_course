import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { loadFont } from "@remotion/google-fonts/BeVietnamPro";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";

export const FONT = loadFont("normal", {
  weights: ["400", "600", "800"],
  subsets: ["latin", "vietnamese"],
}).fontFamily;
export const MONO = loadMono("normal", {
  weights: ["400", "700"],
  subsets: ["latin", "vietnamese"],
}).fontFamily;

// Màu ngữ nghĩa cố định theo storyboard
export const C = {
  bg: "#0F172A",
  panel: "#1E293B",
  line: "#334155",
  text: "#F1F5F9",
  muted: "#94A3B8",
  ai: "#93C5FD",
  ml: "#3B82F6",
  dl: "#1D4ED8",
  spam: "#F97316",
  ok: "#22C55E",
  rule: "#FACC15",
  bad: "#EF4444",
};

/** Chiều cao vùng nội dung (phía trên dải phụ đề 140px). */
export const SAFE_H = 940;

export const useSpring = (at: number, frames = 18) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  return spring({
    frame: f - at,
    fps,
    config: { damping: 200 },
    durationInFrames: frames,
  });
};

/** Hiện dần + trượt lên tại frame `at`. */
export const Pop: React.FC<{
  at: number;
  dy?: number;
  scale?: boolean;
  style?: React.CSSProperties;
  children: React.ReactNode;
}> = ({ at, dy = 30, scale, style, children }) => {
  const s = useSpring(at);
  return (
    <div
      style={{
        opacity: s,
        transform: scale
          ? `scale(${0.6 + 0.4 * s})`
          : `translateY(${(1 - s) * dy}px)`,
        ...style,
      }}
    >
      {children}
    </div>
  );
};

/** Giá trị 0→1 tuyến tính giữa hai frame, có kẹp biên. */
export const lin = (f: number, a: number, b: number) =>
  interpolate(f, [a, b], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

export const Abs: React.FC<{
  x: number;
  y: number;
  w?: number;
  center?: boolean;
  style?: React.CSSProperties;
  children: React.ReactNode;
}> = ({ x, y, w, center = true, style, children }) => (
  <div
    style={{
      position: "absolute",
      left: x,
      top: y,
      width: w ?? "max-content",
      transform: center ? "translate(-50%, -50%)" : undefined,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      ...style,
    }}
  >
    {children}
  </div>
);

export const Title: React.FC<{
  children: React.ReactNode;
  color?: string;
  size?: number;
  style?: React.CSSProperties;
}> = ({ children, color = C.text, size = 72, style }) => (
  <div
    style={{
      fontFamily: FONT,
      fontWeight: 800,
      fontSize: size,
      color,
      lineHeight: 1.15,
      textAlign: "center",
      ...style,
    }}
  >
    {children}
  </div>
);

export const Text: React.FC<{
  children: React.ReactNode;
  color?: string;
  size?: number;
  weight?: number;
  style?: React.CSSProperties;
}> = ({ children, color = C.text, size = 48, weight = 600, style }) => (
  <div
    style={{
      fontFamily: FONT,
      fontWeight: weight,
      fontSize: size,
      color,
      lineHeight: 1.25,
      textAlign: "center",
      ...style,
    }}
  >
    {children}
  </div>
);

export const Box: React.FC<{
  children: React.ReactNode;
  color?: string;
  fill?: string;
  w?: number;
  h?: number;
  style?: React.CSSProperties;
}> = ({ children, color = C.line, fill = C.panel, w, h, style }) => (
  <div
    style={{
      width: w,
      height: h,
      border: `4px solid ${color}`,
      background: fill,
      borderRadius: 24,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: "18px 28px",
      boxSizing: "border-box",
      ...style,
    }}
  >
    {children}
  </div>
);

/** Thẻ email nhỏ. */
export const EmailCard: React.FC<{
  text: string;
  tone?: "neutral" | "spam" | "ok" | "bad";
  w?: number;
  size?: number;
  tag?: string;
}> = ({ text, tone = "neutral", w = 520, size = 34, tag }) => {
  const border = { neutral: C.muted, spam: C.spam, ok: C.ok, bad: C.bad }[tone];
  return (
    <div
      style={{
        width: w,
        background: "#F8FAFC",
        borderRadius: 16,
        border: `5px solid ${border}`,
        padding: "14px 20px",
        boxSizing: "border-box",
        display: "flex",
        alignItems: "center",
        gap: 16,
        fontFamily: FONT,
        fontWeight: 600,
        fontSize: size,
        color: "#0F172A",
        position: "relative",
      }}
    >
      <MailIcon size={size * 1.4} color={border} />
      <span style={{ flex: 1 }}>{text}</span>
      {tag ? (
        <span
          style={{
            position: "absolute",
            right: -14,
            top: -26,
            background: border,
            color: "#fff",
            borderRadius: 10,
            padding: "2px 12px",
            fontSize: size * 0.8,
          }}
        >
          {tag}
        </span>
      ) : null}
    </div>
  );
};

// ---------- Biểu tượng SVG tự vẽ ----------
type IP = { size?: number; color?: string };

export const Check: React.FC<IP> = ({ size = 56, color = C.ok }) => (
  <svg width={size} height={size} viewBox="0 0 24 24">
    <circle cx="12" cy="12" r="11" fill={color} />
    <path
      d="M6.5 12.5l3.5 3.5 7.5-8"
      stroke="#fff"
      strokeWidth="2.6"
      fill="none"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);
export const Cross: React.FC<IP> = ({ size = 56, color = C.bad }) => (
  <svg width={size} height={size} viewBox="0 0 24 24">
    <circle cx="12" cy="12" r="11" fill={color} />
    <path
      d="M8 8l8 8M16 8l-8 8"
      stroke="#fff"
      strokeWidth="2.6"
      strokeLinecap="round"
    />
  </svg>
);
export const MailIcon: React.FC<IP> = ({ size = 80, color = C.ai }) => (
  <svg width={size} height={size * 0.75} viewBox="0 0 32 24">
    <rect
      x="1"
      y="1"
      width="30"
      height="22"
      rx="3"
      fill="none"
      stroke={color}
      strokeWidth="2.5"
    />
    <path d="M2 3l14 11L30 3" fill="none" stroke={color} strokeWidth="2.5" />
  </svg>
);
export const FaceLock: React.FC<IP> = ({ size = 200, color = C.ai }) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <rect
      x="22"
      y="4"
      width="56"
      height="92"
      rx="10"
      fill={C.panel}
      stroke={color}
      strokeWidth="4"
    />
    <circle cx="50" cy="40" r="15" fill="none" stroke={color} strokeWidth="4" />
    <circle cx="44" cy="37" r="2" fill={color} />
    <circle cx="56" cy="37" r="2" fill={color} />
    <path d="M44 45q6 5 12 0" stroke={color} strokeWidth="3" fill="none" />
    <rect x="40" y="66" width="20" height="16" rx="3" fill={color} />
    <path
      d="M44 66v-5a6 6 0 0112 0v5"
      stroke={color}
      strokeWidth="3.5"
      fill="none"
    />
  </svg>
);
export const Inbox: React.FC<IP> = ({ size = 200, color = C.ai }) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <rect
      x="8"
      y="14"
      width="84"
      height="72"
      rx="8"
      fill={C.panel}
      stroke={color}
      strokeWidth="4"
    />
    <rect x="16" y="24" width="68" height="14" rx="3" fill="#334155" />
    <rect x="16" y="44" width="68" height="14" rx="3" fill="#334155" />
    <rect x="16" y="64" width="40" height="14" rx="3" fill={C.spam} />
    <text
      x="36"
      y="75"
      textAnchor="middle"
      fontSize="10"
      fontWeight="800"
      fill="#fff"
      fontFamily="sans-serif"
    >
      SPAM
    </text>
  </svg>
);
export const MapRoute: React.FC<IP> = ({ size = 200, color = C.ai }) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <rect
      x="8"
      y="10"
      width="84"
      height="80"
      rx="8"
      fill={C.panel}
      stroke={color}
      strokeWidth="4"
    />
    <path d="M8 40h84M40 10v80M70 10v80" stroke="#334155" strokeWidth="5" />
    <path
      d="M22 76 Q40 70 40 52 T70 28"
      stroke={C.ok}
      strokeWidth="5"
      fill="none"
      strokeDasharray="8 5"
    />
    <circle cx="22" cy="76" r="6" fill={C.ok} />
    <path
      d="M70 14a8 8 0 018 8c0 6-8 14-8 14s-8-8-8-14a8 8 0 018-8z"
      fill={C.spam}
    />
  </svg>
);
export const Chess: React.FC<IP> = ({ size = 200, color = C.ai }) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    {Array.from({ length: 16 }).map((_, i) => (
      <rect
        key={i}
        x={10 + (i % 4) * 20}
        y={10 + Math.floor(i / 4) * 20}
        width="20"
        height="20"
        fill={(i + Math.floor(i / 4)) % 2 ? "#334155" : "#CBD5E1"}
      />
    ))}
    <rect
      x="10"
      y="10"
      width="80"
      height="80"
      fill="none"
      stroke={color}
      strokeWidth="4"
    />
    <path
      d="M42 74h16l-3-10h-10zM50 30a7 7 0 100 14 7 7 0 000-14zM45 44h10l2 20H43z"
      fill="#0F172A"
      stroke="#fff"
      strokeWidth="2"
    />
  </svg>
);
export const Chat: React.FC<IP> = ({ size = 200, color = C.ai }) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <path
      d="M12 18h76a6 6 0 016 6v40a6 6 0 01-6 6H40l-18 16V70H12a6 6 0 01-6-6V24a6 6 0 016-6z"
      fill={C.panel}
      stroke={color}
      strokeWidth="4"
    />
    <circle cx="32" cy="44" r="5" fill={color} />
    <circle cx="50" cy="44" r="5" fill={color} />
    <circle cx="68" cy="44" r="5" fill={color} />
  </svg>
);
export const House: React.FC<IP> = ({ size = 200, color = C.ai }) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <path
      d="M10 48L50 14l40 34"
      stroke={color}
      strokeWidth="5"
      fill="none"
      strokeLinejoin="round"
    />
    <rect
      x="20"
      y="44"
      width="60"
      height="44"
      fill={C.panel}
      stroke={color}
      strokeWidth="4"
    />
    <rect x="42" y="62" width="16" height="26" fill={color} />
    <rect x="26" y="52" width="12" height="12" fill={C.rule} />
  </svg>
);
export const Plate: React.FC<IP> = ({ size = 200, color = C.ai }) => (
  <svg width={size} height={size * 0.5} viewBox="0 0 100 50">
    <rect
      x="3"
      y="3"
      width="94"
      height="44"
      rx="6"
      fill="#F8FAFC"
      stroke={color}
      strokeWidth="4"
    />
    <text
      x="50"
      y="33"
      textAnchor="middle"
      fontSize="15"
      fontWeight="800"
      fill="#0F172A"
      fontFamily="monospace"
    >
      30A-123.45
    </text>
  </svg>
);
export const Laptop: React.FC<IP> = ({ size = 200, color = C.ai }) => (
  <svg width={size} height={size * 0.7} viewBox="0 0 100 70">
    <rect
      x="14"
      y="4"
      width="72"
      height="48"
      rx="4"
      fill={C.panel}
      stroke={color}
      strokeWidth="4"
    />
    <path d="M4 60h92l-6 8H10z" fill={color} />
  </svg>
);
export const Gpu: React.FC<IP> = ({ size = 200, color = C.ai }) => (
  <svg width={size} height={size * 0.6} viewBox="0 0 100 60">
    <rect
      x="4"
      y="6"
      width="86"
      height="40"
      rx="5"
      fill={C.panel}
      stroke={color}
      strokeWidth="4"
    />
    <circle cx="30" cy="26" r="13" fill="none" stroke={color} strokeWidth="3" />
    <circle cx="64" cy="26" r="13" fill="none" stroke={color} strokeWidth="3" />
    <path
      d="M14 46v10M24 46v10M34 46v10M44 46v10M54 46v10"
      stroke={C.rule}
      strokeWidth="4"
    />
  </svg>
);
export const TableIcon: React.FC<IP> = ({ size = 200, color = C.ai }) => (
  <svg width={size} height={size * 0.75} viewBox="0 0 100 75">
    <rect
      x="4"
      y="4"
      width="92"
      height="67"
      rx="4"
      fill={C.panel}
      stroke={color}
      strokeWidth="4"
    />
    <path
      d="M4 20h92M4 37h92M4 54h92M36 4v67M66 4v67"
      stroke={color}
      strokeWidth="3"
    />
    <rect x="6" y="6" width="88" height="13" fill={C.ok} opacity="0.5" />
  </svg>
);
export const ImageIcon: React.FC<IP> = ({ size = 140, color = C.ai }) => (
  <svg width={size} height={size * 0.8} viewBox="0 0 100 80">
    <rect
      x="4"
      y="4"
      width="92"
      height="72"
      rx="6"
      fill={C.panel}
      stroke={color}
      strokeWidth="4"
    />
    <circle cx="30" cy="28" r="9" fill={C.rule} />
    <path d="M8 70l28-28 18 18 14-12 26 22z" fill={color} />
  </svg>
);
export const WaveIcon: React.FC<IP> = ({ size = 140, color = C.ai }) => (
  <svg width={size} height={size * 0.8} viewBox="0 0 100 80">
    {[8, 22, 36, 50, 64, 78, 92].map((x, i) => {
      const h = [16, 40, 60, 30, 54, 24, 12][i];
      return (
        <rect
          key={x}
          x={x - 4}
          y={40 - h / 2}
          width="8"
          height={h}
          rx="4"
          fill={color}
        />
      );
    })}
  </svg>
);
export const DocIcon: React.FC<IP> = ({ size = 140, color = C.ai }) => (
  <svg width={size * 0.8} height={size} viewBox="0 0 80 100">
    <path
      d="M6 4h50l18 18v74H6z"
      fill={C.panel}
      stroke={color}
      strokeWidth="4"
    />
    <path
      d="M16 36h48M16 50h48M16 64h48M16 78h30"
      stroke={color}
      strokeWidth="5"
      strokeLinecap="round"
    />
  </svg>
);
export const Lens: React.FC<IP> = ({ size = 100, color = C.ok }) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <circle cx="42" cy="42" r="28" fill="none" stroke={color} strokeWidth="8" />
    <path
      d="M62 62l26 26"
      stroke={color}
      strokeWidth="10"
      strokeLinecap="round"
    />
  </svg>
);
export const BlackBox: React.FC<IP> = ({ size = 100 }) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <path
      d="M50 8l40 20v44L50 92 10 72V28z"
      fill="#020617"
      stroke={C.muted}
      strokeWidth="4"
    />
    <text
      x="50"
      y="62"
      textAnchor="middle"
      fontSize="36"
      fontWeight="800"
      fill={C.muted}
      fontFamily="sans-serif"
    >
      ?
    </text>
  </svg>
);
export const Person: React.FC<IP> = ({ size = 200, color = C.ai }) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <circle cx="50" cy="30" r="16" fill={color} />
    <path d="M18 92c0-22 14-36 32-36s32 14 32 36z" fill={color} />
  </svg>
);
export const Child: React.FC<IP> = ({ size = 160, color = C.ai }) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <circle cx="50" cy="40" r="18" fill={color} />
    <path d="M26 94c0-18 10-30 24-30s24 12 24 30z" fill={color} />
  </svg>
);
export const Hand: React.FC<IP> = ({ size = 90, color = C.rule }) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <path d="M20 80l50-50 10 10-50 50H20z" fill={color} />
    <path d="M70 30l8-8 10 10-8 8z" fill={C.spam} />
  </svg>
);
export const Tea: React.FC<IP> = ({ size = 220, color = C.ai }) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <path
      d="M58 4l-6 18"
      stroke={C.spam}
      strokeWidth="5"
      strokeLinecap="round"
    />
    <path
      d="M24 22h52l-7 72H31z"
      fill="#E7C9A0"
      stroke={color}
      strokeWidth="4"
    />
    <rect x="22" y="18" width="56" height="8" rx="3" fill={color} />
    {[38, 50, 62, 44, 56].map((x, i) => (
      <circle key={i} cx={x} cy={i < 3 ? 82 : 72} r="5" fill="#3F2A1D" />
    ))}
  </svg>
);
export const Calendar: React.FC<IP & { label: string }> = ({
  size = 220,
  color = C.ai,
  label,
}) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <rect
      x="8"
      y="14"
      width="84"
      height="78"
      rx="8"
      fill="#F8FAFC"
      stroke={color}
      strokeWidth="4"
    />
    <rect x="8" y="14" width="84" height="22" rx="8" fill={C.spam} />
    <text
      x="50"
      y="72"
      textAnchor="middle"
      fontSize="15"
      fontWeight="800"
      fill="#0F172A"
      fontFamily={FONT}
    >
      {label}
    </text>
  </svg>
);

/** Mèo/chó phẳng dùng cho cảnh 4. */
export const Cat: React.FC<IP & { earless?: boolean }> = ({
  size = 160,
  color = "#FDBA74",
  earless,
}) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    {earless ? null : (
      <path d="M22 36L28 8l18 18zM78 36L72 8 54 26z" fill={color} />
    )}
    <circle cx="50" cy="52" r="32" fill={color} />
    <circle cx="38" cy="48" r="4" fill="#0F172A" />
    <circle cx="62" cy="48" r="4" fill="#0F172A" />
    <path d="M46 60h8l-4 5z" fill="#BE185D" />
    <path
      d="M20 58h18M20 66h18M62 58h18M62 66h18"
      stroke="#0F172A"
      strokeWidth="2"
    />
  </svg>
);
export const Dog: React.FC<IP> = ({ size = 160, color = "#A8A29E" }) => (
  <svg width={size} height={size} viewBox="0 0 100 100">
    <ellipse cx="22" cy="46" rx="12" ry="24" fill="#57534E" />
    <ellipse cx="78" cy="46" rx="12" ry="24" fill="#57534E" />
    <circle cx="50" cy="52" r="30" fill={color} />
    <circle cx="40" cy="46" r="4" fill="#0F172A" />
    <circle cx="60" cy="46" r="4" fill="#0F172A" />
    <ellipse cx="50" cy="62" rx="9" ry="6" fill="#0F172A" />
  </svg>
);
export const Computer: React.FC<{
  glow?: number;
  label?: string;
  w?: number;
  color?: string;
}> = ({ glow = 0, label = "Máy tính", w = 320, color = C.ml }) => (
  <div
    style={{
      width: w,
      height: w * 0.62,
      borderRadius: 28,
      border: `5px solid ${color}`,
      background: C.panel,
      boxShadow: `0 0 ${20 + 80 * glow}px ${color}${Math.round(40 + 150 * glow)
        .toString(16)
        .padStart(2, "0")}`,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: FONT,
      fontWeight: 800,
      fontSize: 46,
      color: C.text,
    }}
  >
    {label}
  </div>
);
