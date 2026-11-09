import React from "react";
import { AbsoluteFill, interpolate, Sequence, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { Audio } from "@remotion/media";
import timelineJson from "./timeline.json";
import { SCENES } from "./scenes";
import { C, FONT, SAFE_H } from "./ui";

type Timeline = {
  fps: number;
  totalFrames: number;
  scenes: { id: string; from: number; dur: number; countdownAt?: number; audio: { src: string; from: number }[] }[];
  captions: { start: number; end: number; text: string }[];
};
const timeline = timelineJson as Timeline;
export const TIMELINE = timeline;

const Fade: React.FC<{ d: number; children: React.ReactNode }> = ({ d, children }) => {
  const f = useCurrentFrame();
  const o = interpolate(f, [0, 10, d - 8, d], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return <AbsoluteFill style={{ opacity: o }}>{children}</AbsoluteFill>;
};

const Captions: React.FC = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = f / fps;
  const cap = timeline.captions.find((c) => t >= c.start && t < c.end);
  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        bottom: 0,
        height: 1080 - SAFE_H,
        background: "#020617",
        borderTop: `2px solid ${C.line}`,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "0 120px",
        boxSizing: "border-box",
      }}
    >
      {cap ? (
        <div style={{ fontFamily: FONT, fontWeight: 600, fontSize: 42, color: "#fff", textAlign: "center", lineHeight: 1.25 }}>{cap.text}</div>
      ) : null}
    </div>
  );
};

export const Lesson: React.FC = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    {timeline.scenes.map((s) => {
      const Scene = SCENES[s.id];
      return (
        <Sequence key={s.id} from={s.from} durationInFrames={s.dur} name={s.id}>
          <div style={{ position: "absolute", left: 0, top: 0, width: 1920, height: SAFE_H, overflow: "hidden" }}>
            <Fade d={s.dur}>
              <Scene d={s.dur} countdownAt={s.countdownAt} />
            </Fade>
          </div>
          {s.audio.map((a) => (
            <Sequence key={a.src} from={a.from} layout="none">
              <Audio src={staticFile(a.src)} />
            </Sequence>
          ))}
        </Sequence>
      );
    })}
    <Captions />
  </AbsoluteFill>
);
