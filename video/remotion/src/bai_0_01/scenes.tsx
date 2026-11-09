import React from "react";
import { AbsoluteFill, interpolate, random, useCurrentFrame } from "remotion";
import { CodeBlock } from "./code";
import {
  Abs, BlackBox, Box, C, Calendar, Cat, Chat, Check, Chess, Child, Computer, Cross, DocIcon, Dog, EmailCard, FaceLock, FONT,
  Gpu, Hand, House, ImageIcon, Inbox, Laptop, Lens, lin, MapRoute, Person, Plate, Pop, TableIcon, Tea, Text, Title, useSpring, WaveIcon,
} from "./ui";

export type SceneProps = { d: number; countdownAt?: number };
const useAt = (d: number) => (p: number) => Math.round(p * d);

// ---------- PHẦN A ----------
export const S01: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const f = useCurrentFrame();
  const shrink = lin(f, at(0.55), at(0.65));
  const items = [
    { el: <FaceLock size={220} />, label: "Mở khóa bằng mặt" },
    { el: <Inbox size={220} />, label: "Tự gom thư rác" },
    { el: <MapRoute size={220} />, label: "Tránh kẹt xe" },
  ];
  return (
    <AbsoluteFill>
      <div style={{ position: "absolute", top: interpolate(shrink, [0, 1], [300, 90]), width: "100%", display: "flex", justifyContent: "center", gap: 140, transform: `scale(${1 - 0.35 * shrink})` }}>
        {items.map((it, i) => (
          <Pop key={i} at={at(0.1 + i * 0.1)} scale>
            {it.el}
            <Text size={44} color={C.muted}>{it.label}</Text>
          </Pop>
        ))}
      </div>
      <Abs x={960} y={640}>
        <Pop at={at(0.6)} scale>
          <Title size={150}>
            <span style={{ color: C.ai }}>AI</span> · <span style={{ color: C.ml }}>ML</span> · <span style={{ color: "#60A5FA", textShadow: `0 0 0 ${C.dl}` }}>DL</span>
            <span style={{ color: C.rule }}> ?</span>
          </Title>
          <Title size={72}>khác nhau thế nào?</Title>
        </Pop>
      </Abs>
    </AbsoluteFill>
  );
};

const GATE_X = 960;
export const S02: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const f = useCurrentFrame();
  const day3 = f >= at(0.5);
  // email ngày 1: chạy tới cổng rồi bị chặn
  const e1 = lin(f, at(0.25), at(0.36));
  const variants = ["trúng thuởng", "tr.úng th.ưởng", "nhận quà miễn phí"];
  return (
    <AbsoluteFill>
      <Abs x={190} y={150}>
        <Pop at={0}>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 0 }}>
            <Person size={150} color={C.ai} />
            <Laptop size={130} />
          </div>
          <Text size={40} color={C.muted}>Anh Nam</Text>
        </Pop>
      </Abs>
      <Abs x={960} y={80}>
        <Title size={64} color={day3 ? C.bad : C.ok}>{day3 ? "Ngày 3: kẻ lừa đảo đổi cách viết" : "Ngày 1: viết một luật"}</Title>
      </Abs>
      {/* băng chuyền */}
      <div style={{ position: "absolute", left: 0, right: 0, top: 560, height: 12, background: C.line }} />
      {/* cổng luật */}
      <Abs x={GATE_X} y={470}>
        <Pop at={at(0.15)}>
          <div style={{ width: 300, height: 300, border: `8px solid ${C.rule}`, borderBottom: "none", borderRadius: "24px 24px 0 0", display: "flex", alignItems: "flex-start", justifyContent: "center" }}>
            <div style={{ fontFamily: FONT, fontWeight: 800, fontSize: 34, color: C.rule, background: C.bg, padding: "6px 12px", marginTop: -36, whiteSpace: "nowrap" }}>"trúng thưởng" → SPAM</div>
          </div>
        </Pop>
      </Abs>
      {!day3 ? (
        <Abs x={interpolate(e1, [0, 1], [-300, GATE_X - 330])} y={500}>
          <EmailCard text="Bạn đã trúng thưởng!" tone={f > at(0.36) ? "spam" : "neutral"} w={460} />
          {f > at(0.37) ? (
            <Pop at={at(0.37)} style={{ display: "flex", gap: 16, alignItems: "center", marginTop: 24 }}>
              <Check /> <Text color={C.spam}>Bị chặn: SPAM</Text>
            </Pop>
          ) : null}
        </Abs>
      ) : null}
      {day3
        ? variants.map((v, i) => {
            const p = lin(f, at(0.6 + i * 0.09), at(0.72 + i * 0.09));
            const passed = p > 0.75;
            return (
              <Abs key={v} x={interpolate(p, [0, 1], [-320, 1640 - i * 0])} y={500 - (i === 1 ? 0 : 0)} style={{ opacity: p > 0 ? 1 : 0 }}>
                <EmailCard text={v} tone={passed ? "bad" : "neutral"} w={470} />
              </Abs>
            );
          })
        : null}
      {day3 ? (
        <Abs x={960} y={780}>
          <div style={{ display: "flex", gap: 40 }}>
            {variants.map((v, i) => (
              <Pop key={v} at={at(0.7 + i * 0.09)} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <Cross size={48} />
                <Text size={40}>{v}</Text>
              </Pop>
            ))}
          </div>
          <Pop at={at(0.92)}>
            <Text color={C.bad} style={{ marginTop: 24 }}>Lọt qua luật!</Text>
          </Pop>
        </Abs>
      ) : null}
    </AbsoluteFill>
  );
};

export const S03: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const f = useCurrentFrame();
  const phase2 = f >= at(0.5);
  const count = Math.round(interpolate(f, [at(0.05), at(0.22)], [1, 500], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }));
  const cards = Math.round(count / 10);
  if (!phase2) {
    const blink = f > at(0.27) && Math.floor(f / 8) % 2 === 0;
    return (
      <AbsoluteFill>
        <Abs x={960} y={80}>
          <Title size={64}>Ngày 30: <span style={{ color: C.rule }}>{count}</span> luật</Title>
        </Abs>
        <div style={{ position: "absolute", left: 160, top: 170, width: 900, display: "flex", flexWrap: "wrap", gap: 10 }}>
          {Array.from({ length: cards }).map((_, i) => (
            <div key={i} style={{ width: 80, height: 46, background: C.rule, opacity: 0.55 + 0.45 * random(i), borderRadius: 6, fontFamily: FONT, fontSize: 18, fontWeight: 800, color: "#0F172A", display: "flex", alignItems: "center", justifyContent: "center" }}>
              if…
            </div>
          ))}
        </div>
        <Abs x={1450} y={430}>
          <Pop at={at(0.27)}>
            <EmailCard text="Thưởng Tết – từ Sếp" tone={blink ? "bad" : "spam"} w={520} tag="bị chặn" />
            <Text color={C.bad} style={{ marginTop: 24 }}>Chặn nhầm!</Text>
          </Pop>
        </Abs>
      </AbsoluteFill>
    );
  }
  const glow = lin(f, at(0.62), at(0.78));
  return (
    <AbsoluteFill>
      <Abs x={960} y={80}>
        <Title size={60}>Ngày 31: 10.000 email có nhãn → máy tự tìm luật</Title>
      </Abs>
      {Array.from({ length: 8 }).map((_, i) => {
        const p = lin(f, at(0.52 + i * 0.015), at(0.62 + i * 0.015));
        const spam = i % 2 === 0;
        return (
          <Abs key={i} x={interpolate(p, [0, 1], [380, 960])} y={interpolate(p, [0, 1], [220 + i * 70, 470])} style={{ opacity: 1 - lin(p, 0.85, 1) }}>
            <EmailCard text={spam ? "Nhận quà ngay…" : "Lịch họp…"} tone={spam ? "spam" : "ok"} w={360} size={28} tag={spam ? "SPAM" : "OK"} />
          </Abs>
        );
      })}
      <Abs x={1100} y={470}>
        <Computer glow={glow} />
      </Abs>
      <Abs x={960} y={800}>
        <Pop at={at(0.84)} scale>
          <Title size={84} color={C.ml}>Machine Learning <span style={{ color: C.text }}>(Học máy)</span></Title>
        </Pop>
      </Abs>
    </AbsoluteFill>
  );
};

// ---------- PHẦN B ----------
export const S04: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const rules = ["4 chân", "Có ria", "Tai nhọn", "Kêu meo meo"];
  const tiles = ["cat", "dog", "cat", "dog", "cat", "cat"];
  return (
    <AbsoluteFill>
      <div style={{ position: "absolute", left: 958, top: 30, bottom: 30, width: 4, background: C.line }} />
      {/* Trái */}
      <Abs x={480} y={80}><Title size={56} color={C.rule}>Cách 1: Đưa luật</Title></Abs>
      <div style={{ position: "absolute", left: 90, top: 170 }}>
        {rules.map((r, i) => (
          <Pop key={r} at={at(0.07 + i * 0.04)} style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 14 }}>
            <Check size={46} color={C.rule} /> <Text size={44}>{r}</Text>
          </Pop>
        ))}
      </div>
      <Abs x={700} y={560}>
        <Pop at={at(0.3)} style={{ flexDirection: "row", display: "flex", alignItems: "flex-end", gap: 20 }}>
          <Child size={150} />
          <div style={{ position: "relative" }}>
            <Cat earless size={170} />
          </div>
          <Text size={110} color={C.rule} style={{ position: "absolute", left: 40, top: -100 }}>?</Text>
        </Pop>
        <Pop at={at(0.32)}><Text size={36} color={C.muted}>Mèo cụt tai → bối rối</Text></Pop>
      </Abs>
      {/* Phải */}
      <Abs x={1440} y={80}><Title size={56} color={C.ml}>Cách 2: Đưa ví dụ</Title></Abs>
      <div style={{ position: "absolute", left: 1040, top: 150, display: "grid", gridTemplateColumns: "repeat(3, 250px)", gap: 14 }}>
        {tiles.map((t, i) => (
          <Pop key={i} at={at(0.45 + i * 0.025)} scale>
            <div style={{ background: C.panel, borderRadius: 18, display: "flex", alignItems: "center", gap: 6, padding: "6px 10px" }}>
              {t === "cat" ? <Cat size={110} /> : <Dog size={110} />}
              <Text size={36} color={t === "cat" ? "#FDBA74" : "#D6D3D1"}>{t === "cat" ? "Mèo" : "Chó"}</Text>
            </div>
          </Pop>
        ))}
      </div>
      <Abs x={1440} y={620}>
        <Pop at={at(0.66)} style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: 20 }}>
          <Cat earless size={150} />
          <Text size={60} color={C.ok}>Mèo!</Text>
          <Check size={64} />
        </Pop>
      </Abs>
      <Abs x={480} y={830}><Pop at={at(0.84)}><Box color={C.rule}><Text size={46}>= Lập trình truyền thống</Text></Box></Pop></Abs>
      <Abs x={1440} y={830}><Pop at={at(0.92)}><Box color={C.ml}><Text size={46}>= Machine Learning</Text></Box></Pop></Abs>
    </AbsoluteFill>
  );
};

export const S05: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const f = useCurrentFrame();
  const swap = lin(f, at(0.42), at(0.52));
  const renamed = f >= at(0.66);
  const ml = swap > 0.5;
  const IN2 = { x: 330, y: 520 };
  const OUT = { x: 1600, y: 410 };
  const arc = Math.sin(swap * Math.PI) * 220;
  const rulePos = { x: interpolate(swap, [0, 1], [IN2.x, OUT.x]), y: interpolate(swap, [0, 1], [IN2.y, OUT.y]) - arc };
  const resPos = { x: interpolate(swap, [0, 1], [OUT.x, IN2.x]), y: interpolate(swap, [0, 1], [OUT.y, IN2.y]) + arc };
  const newEmail = lin(f, at(0.8), at(0.9));
  return (
    <AbsoluteFill>
      <Abs x={960} y={70}>
        <Title size={60} color={ml ? C.ml : C.rule}>{ml ? "Machine Learning" : "Lập trình truyền thống"}</Title>
      </Abs>
      <Abs x={960} y={150}>
        <Text size={50} color={C.muted}>{ml ? "Dữ liệu + Kết quả → Luật (mô hình)" : "Dữ liệu + Luật → Kết quả"}</Text>
      </Abs>
      {/* mũi tên */}
      <svg width={1920} height={940} style={{ position: "absolute" }}>
        <defs>
          <marker id="ah" markerWidth="10" markerHeight="10" refX="6" refY="5" orient="auto">
            <path d="M0 0L10 5L0 10z" fill={C.muted} />
          </marker>
        </defs>
        <path d="M480 300 L780 390" stroke={C.muted} strokeWidth="6" markerEnd="url(#ah)" />
        <path d="M480 520 L780 440" stroke={C.muted} strokeWidth="6" markerEnd="url(#ah)" />
        <path d="M1140 410 L1400 410" stroke={C.muted} strokeWidth="6" markerEnd="url(#ah)" />
      </svg>
      <Abs x={330} y={300}><Pop at={at(0.1)}><Box w={300} h={130}><Text>Dữ liệu</Text></Box></Pop></Abs>
      <Abs x={960} y={410}><Pop at={at(0.14)}><Computer w={340} /></Pop></Abs>
      <Abs x={rulePos.x} y={rulePos.y}>
        <Pop at={at(0.18)}>
          <Box w={300} h={130} color={renamed ? C.ml : C.rule} style={{ boxShadow: renamed ? `0 0 40px ${C.ml}` : undefined }}>
            <Text color={renamed ? C.ml : C.rule}>{renamed ? "Mô hình" : "Luật"}</Text>
          </Box>
        </Pop>
      </Abs>
      <Abs x={resPos.x} y={resPos.y}>
        <Pop at={at(0.25)}><Box w={300} h={130} color={C.ok}><Text color={C.ok}>Kết quả</Text></Box></Pop>
      </Abs>
      {/* dùng mô hình cho email mới */}
      {f >= at(0.78) ? (
        <div style={{ position: "absolute", top: 700, left: 0, right: 0, display: "flex", justifyContent: "center", alignItems: "center", gap: 36 }}>
          <Pop at={at(0.78)}><EmailCard text="Email mới" w={330} /></Pop>
          <Text size={60} color={C.muted}>→</Text>
          <Pop at={at(0.8)}><Box color={C.ml}><Text color={C.ml}>Mô hình</Text></Box></Pop>
          <Text size={60} color={C.muted} style={{ opacity: newEmail }}>→</Text>
          <Pop at={at(0.9)} scale><Box color={C.spam} fill={C.spam}><Text>SPAM</Text></Box></Pop>
        </div>
      ) : null}
    </AbsoluteFill>
  );
};

// ---------- PHẦN C ----------
export const S06: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const open = useSpring(at(0.05), 30);
  return (
    <AbsoluteFill>
      <Abs x={960} y={230}>
        <svg width={760} height={320} viewBox="0 0 760 320" style={{ transform: `scaleX(${0.2 + 0.8 * open})`, opacity: open }}>
          <path d="M20 240 Q380 -120 740 240 Q650 200 560 240 Q470 200 380 240 Q290 200 200 240 Q110 200 20 240z" fill={C.ai} />
          <path d="M380 240 V300 q0 18 -18 18" stroke={C.ai} strokeWidth="12" fill="none" strokeLinecap="round" />
        </svg>
        <Title size={64} color={C.ai} style={{ marginTop: -10 }}>AI – Trí tuệ nhân tạo</Title>
      </Abs>
      <div style={{ position: "absolute", top: 450, width: "100%", display: "flex", justifyContent: "center", gap: 120 }}>
        {[
          { el: <MapRoute size={180} />, t: "Tìm đường" },
          { el: <Chess size={180} />, t: "Chơi cờ", tag: true },
          { el: <Chat size={180} />, t: "Trả lời câu hỏi" },
        ].map((it, i) => (
          <Pop key={i} at={at(0.32 + i * 0.07)} style={{ position: "relative" }}>
            {it.el}
            <Text size={42} color={C.muted}>{it.t}</Text>
            {it.tag ? (
              <Pop at={at(0.68)} style={{ position: "absolute", right: -40, top: -10 }}>
                <div style={{ background: C.rule, color: "#0F172A", fontFamily: FONT, fontWeight: 800, fontSize: 34, padding: "4px 14px", borderRadius: 10 }}>if-else</div>
              </Pop>
            ) : null}
          </Pop>
        ))}
      </div>
      <Abs x={960} y={850}>
        <Pop at={at(0.72)}><Text size={52} color={C.rule}>Không nhất thiết phải "học"</Text></Pop>
      </Abs>
    </AbsoluteFill>
  );
};

export const S07: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const f = useCurrentFrame();
  const rows = [
    ["25 tr", "1", true],
    ["8 tr", "4", false],
    ["40 tr", "2", true],
    ["12 tr", "3", false],
  ] as const;
  const feed = lin(f, at(0.4), at(0.55));
  const cell: React.CSSProperties = { fontFamily: FONT, fontWeight: 600, fontSize: 38, color: C.text, padding: "10px 18px", borderBottom: `2px solid ${C.line}`, textAlign: "center" };
  const newRow = lin(f, at(0.62), at(0.74));
  return (
    <AbsoluteFill>
      <Abs x={960} y={70}><Title size={66} color={C.ml}>Machine Learning – Học máy</Title></Abs>
      <Abs x={960} y={150}><Text color={C.muted}>Tự rút quy luật từ dữ liệu</Text></Abs>
      <Abs x={interpolate(feed, [0, 1], [460, 1050])} y={470} style={{ opacity: 1 - feed * 0.9, transform: `translate(-50%,-50%) scale(${1 - 0.5 * feed})` }}>
        <Pop at={at(0.12)}>
          <table style={{ borderCollapse: "collapse", background: C.panel, borderRadius: 16 }}>
            <thead>
              <tr>{["Thu nhập", "Số khoản vay", "Trả đúng hạn"].map((h) => <th key={h} style={{ ...cell, color: C.ai }}>{h}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  <td style={cell}>{r[0]}</td>
                  <td style={cell}>{r[1]}</td>
                  <td style={cell}>{r[2] ? <Check size={40} /> : <Cross size={40} />}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <Text size={30} color={C.muted} style={{ marginTop: 10 }}>Số liệu minh họa</Text>
        </Pop>
      </Abs>
      <Abs x={1150} y={470}><Pop at={at(0.3)}><Computer label="Mô hình" glow={feed} w={340} /></Pop></Abs>
      <Abs x={interpolate(newRow, [0, 1], [380, 820])} y={760} style={{ opacity: f > at(0.6) ? 1 : 0 }}>
        <Box><Text size={40}>Khách mới: 30 tr · 1 khoản · ?</Text></Box>
      </Abs>
      <Abs x={1530} y={760}>
        <Pop at={at(0.78)} style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: 16 }}>
          <Text size={44} color={C.ok}>Dự đoán: trả đúng hạn</Text> <Check size={52} />
        </Pop>
      </Abs>
    </AbsoluteFill>
  );
};

export const S08: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const f = useCurrentFrame();
  const layers = [4, 6, 6, 3];
  const LX = [700, 960, 1220, 1460];
  const nodeY = (n: number, j: number) => 470 + (j - (n - 1) / 2) * 92;
  const sig = interpolate(f, [at(0.25), at(0.55)], [0, layers.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill>
      <Abs x={960} y={70}><Title size={66} color="#60A5FA">Deep Learning – Học sâu</Title></Abs>
      <Abs x={960} y={150}><Text color={C.muted}>Mạng nơ-ron nhiều lớp · ảnh, âm thanh, văn bản</Text></Abs>
      {/* ảnh mèo dạng điểm ảnh */}
      <div style={{ position: "absolute", left: 200, top: 300, display: "grid", gridTemplateColumns: "repeat(8, 40px)", gap: 3 }}>
        {Array.from({ length: 64 }).map((_, i) => {
          const x = i % 8, y = Math.floor(i / 8);
          const inCat = (x - 3.5) ** 2 + (y - 4.2) ** 2 < 10 || ((y === 0 || y === 1) && (x === 1 || x === 6));
          const eye = y === 4 && (x === 2 || x === 5);
          return <div key={i} style={{ width: 40, height: 40, background: eye ? "#0F172A" : inCat ? "#FDBA74" : "#334155", opacity: 0.6 + 0.4 * random(`p${i}`) }} />;
        })}
      </div>
      <svg width={1920} height={940} style={{ position: "absolute" }}>
        {layers.slice(0, -1).map((n, l) =>
          Array.from({ length: n }).flatMap((_, a) =>
            Array.from({ length: layers[l + 1] }).map((__, b) => (
              <line key={`${l}-${a}-${b}`} x1={LX[l]} y1={nodeY(n, a)} x2={LX[l + 1]} y2={nodeY(layers[l + 1], b)} stroke={sig > l + 0.5 ? C.ai : C.line} strokeWidth={2} opacity={sig > l + 0.5 ? 0.8 : 0.5} />
            )),
          ),
        )}
        {layers.map((n, l) =>
          Array.from({ length: n }).map((_, j) => <circle key={`${l}-${j}`} cx={LX[l]} cy={nodeY(n, j)} r={26} fill={sig > l ? C.ml : C.panel} stroke={C.ai} strokeWidth={4} />),
        )}
      </svg>
      <Abs x={1640} y={470}><Pop at={at(0.56)}><Text size={60} color={C.ok}>Mèo</Text></Pop></Abs>
      <div style={{ position: "absolute", top: 760, width: "100%", display: "flex", justifyContent: "center", gap: 100, alignItems: "center" }}>
        {[<FaceLock key="a" size={120} />, <Plate key="b" size={200} />, <Chat key="c" size={120} />].map((el, i) => (
          <Pop key={i} at={at(0.62 + i * 0.06)}>{el}</Pop>
        ))}
      </div>
    </AbsoluteFill>
  );
};

export const S09: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const ring = (r: number, color: string, start: number, label: string, ex: string, labelY: number, dark?: boolean) => {
    const s = useSpring(at(start), 24);
    return (
      <div style={{ position: "absolute", left: 720 - r, top: 470 - r, width: r * 2, height: r * 2, borderRadius: "50%", background: color, transform: `scale(${s})`, opacity: s, display: "flex", justifyContent: "center" }}>
        <div style={{ marginTop: labelY, textAlign: "center" }}>
          <Text size={52} weight={800} color={dark ? "#0F172A" : "#fff"}>{label}</Text>
          <Text size={34} color={dark ? "#0F172A" : "#E0E7FF"}>{ex}</Text>
        </div>
      </div>
    );
  };
  return (
    <AbsoluteFill>
      {ring(430, C.ai, 0.12, "AI", "Cờ vua if-else", 30, true)}
      {ring(300, C.ml, 0.2, "ML", "Lọc spam, giá nhà", 40)}
      {ring(165, C.dl, 0.28, "DL", "Face ID, ChatGPT", 60)}
      <div style={{ position: "absolute", left: 1240, top: 250, width: 620 }}>
        <Pop at={at(0.55)}><Title size={84}>AI ⊃ ML ⊃ DL</Title></Pop>
        <Pop at={at(0.62)}><Text style={{ marginTop: 30 }}>Mọi DL là ML</Text></Pop>
        <Pop at={at(0.7)}><Text>Mọi ML là AI</Text></Pop>
        <Pop at={at(0.84)} style={{ display: "flex", justifyContent: "center", gap: 16, marginTop: 30, alignItems: "center" }}>
          <Cross size={52} /> <Text color={C.rule}>Chiều ngược lại: không</Text>
        </Pop>
      </div>
    </AbsoluteFill>
  );
};

export const S10: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const rows: [string, React.ReactNode, React.ReactNode][] = [
    ["Dữ liệu", "Dạng bảng", "Ảnh · âm thanh · văn bản"],
    ["Lượng dữ liệu", "Vài trăm – vài chục nghìn dòng", "Rất nhiều"],
    ["Phần cứng", <span key="a" style={{ display: "flex", alignItems: "center", gap: 14 }}><Laptop size={80} /> Máy tính thường</span>, <span key="b" style={{ display: "flex", alignItems: "center", gap: 14 }}><Gpu size={80} /> Card đồ họa (GPU)</span>],
    ["Giải thích", <span key="c" style={{ display: "flex", alignItems: "center", gap: 14 }}><Lens size={56} /> Dễ</span>, <span key="d" style={{ display: "flex", alignItems: "center", gap: 14 }}><BlackBox size={60} /> Khó – hộp đen</span>],
  ];
  const starts = [0.17, 0.3, 0.55, 0.7];
  return (
    <AbsoluteFill>
      <div style={{ position: "absolute", left: 80, right: 80, top: 40, display: "grid", gridTemplateColumns: "340px 1fr 1fr", rowGap: 18, columnGap: 24, alignItems: "center" }}>
        <div />
        <Pop at={at(0.03)} style={{ display: "flex", alignItems: "center", gap: 20, justifyContent: "center" }}>
          <TableIcon size={110} /> <Title size={54} color={C.ml}>ML cổ điển</Title>
        </Pop>
        <Pop at={at(0.08)} style={{ display: "flex", alignItems: "center", gap: 10, justifyContent: "center" }}>
          <ImageIcon size={80} /><WaveIcon size={80} /><DocIcon size={70} /> <Title size={54} color="#60A5FA">Deep Learning</Title>
        </Pop>
        {rows.map((r, i) => (
          <React.Fragment key={i}>
            <Pop at={at(starts[i])}><Text size={42} color={C.muted} style={{ textAlign: "left" }}>{r[0]}</Text></Pop>
            <Pop at={at(starts[i])}><Box style={{ minHeight: 110 }}><Text size={40}>{r[1]}</Text></Box></Pop>
            <Pop at={at(starts[i] + 0.05)}><Box style={{ minHeight: 110 }}><Text size={40}>{r[2]}</Text></Box></Pop>
          </React.Fragment>
        ))}
      </div>
      <Abs x={960} y={850}>
        <Pop at={at(0.86)} scale>
          <Box color={C.rule} fill="#422006"><Text size={52} color={C.rule}>Dữ liệu bảng → thử ML cổ điển trước</Text></Box>
        </Pop>
      </Abs>
    </AbsoluteFill>
  );
};

// ---------- PHẦN D ----------
export const S11: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const f = useCurrentFrame();
  if (f < at(0.5)) {
    const mails = [
      ["Quà tặng miễn phí…", true],
      ["Lịch họp dự án…", false],
      ["Khuyến mãi 100%…", true],
      ["Báo cáo doanh thu…", false],
    ] as const;
    return (
      <AbsoluteFill>
        <Abs x={960} y={80}><Title size={66}>Học có giám sát <span style={{ color: C.muted }}>(Supervised)</span></Title></Abs>
        <div style={{ position: "absolute", top: 230, width: "100%", display: "grid", gridTemplateColumns: "repeat(2, 600px)", justifyContent: "center", gap: "60px 80px" }}>
          {mails.map(([t, s], i) => {
            const tagIn = f > at(0.22 + i * 0.03);
            return (
              <Pop key={i} at={at(0.05 + i * 0.03)}>
                <EmailCard text={t} tone={tagIn ? (s ? "spam" : "ok") : "neutral"} w={600} tag={tagIn ? (s ? "SPAM" : "Bình thường") : undefined} />
              </Pop>
            );
          })}
        </div>
        <Abs x={960} y={700}>
          <Pop at={at(0.25)}><Text size={56} color={C.rule}>Có đáp án = nhãn (label)</Text></Pop>
          <Pop at={at(0.38)}><Text size={42} color={C.muted} style={{ marginTop: 20 }}>Như bài tập có đáp án cuối sách</Text></Pop>
        </Abs>
      </AbsoluteFill>
    );
  }
  const needle = interpolate(f, [at(0.62), at(0.74)], [0, 3.2], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const fall = lin(f, at(0.86), at(0.94));
  const AX0 = 140, AX1 = 820; // trục 0–6 tỷ
  return (
    <AbsoluteFill>
      <div style={{ position: "absolute", left: 958, top: 30, bottom: 30, width: 4, background: C.line }} />
      <Abs x={480} y={80}><Title size={56} color={C.ml}>Hồi quy → con số</Title></Abs>
      <Abs x={480} y={150}><Text size={40} color={C.muted}>Regression</Text></Abs>
      <Abs x={480} y={330}><Pop at={at(0.55)}><House size={190} /></Pop></Abs>
      <svg width={960} height={940} style={{ position: "absolute", left: 0 }}>
        <line x1={AX0} y1={620} x2={AX1} y2={620} stroke={C.muted} strokeWidth="5" />
        {[0, 1, 2, 3, 4, 5, 6].map((v) => (
          <g key={v}>
            <line x1={AX0 + ((AX1 - AX0) * v) / 6} y1={605} x2={AX0 + ((AX1 - AX0) * v) / 6} y2={635} stroke={C.muted} strokeWidth="4" />
            <text x={AX0 + ((AX1 - AX0) * v) / 6} y={680} fill={C.muted} fontSize="34" textAnchor="middle" fontFamily={FONT}>{v}</text>
          </g>
        ))}
        <path d={`M${AX0 + ((AX1 - AX0) * needle) / 6} 600 l-16 -30 h32z`} fill={C.rule} />
      </svg>
      <Abs x={480} y={770}>
        <Pop at={at(0.74)}><Text size={60} color={C.rule}>≈ 3,2 tỷ</Text></Pop>
        <Text size={30} color={C.muted}>(tỷ đồng · số liệu minh họa)</Text>
      </Abs>
      <Abs x={1440} y={80}><Title size={56} color={C.ml}>Phân loại → nhóm</Title></Abs>
      <Abs x={1440} y={150}><Text size={40} color={C.muted}>Classification</Text></Abs>
      <Abs x={interpolate(fall, [0, 1], [1440, 1240])} y={interpolate(fall, [0, 1], [330, 640])} style={{ opacity: 1 - lin(fall, 0.8, 1) }}>
        <Pop at={at(0.8)}><EmailCard text="Nhận quà ngay!" w={400} /></Pop>
      </Abs>
      {[
        { x: 1240, label: "SPAM", c: C.spam },
        { x: 1640, label: "Bình thường", c: C.ok },
      ].map((b) => (
        <Abs key={b.label} x={b.x} y={720}>
          <Pop at={at(0.78)}>
            <div style={{ width: 300, height: 170, border: `6px solid ${b.c}`, borderTop: "none", borderRadius: "0 0 40px 40px", display: "flex", alignItems: "flex-end", justifyContent: "center", paddingBottom: 20, background: b.label === "SPAM" && fall >= 1 ? `${C.spam}33` : undefined }}>
              <Text size={42} color={b.c}>{b.label}</Text>
            </div>
          </Pop>
        </Abs>
      ))}
    </AbsoluteFill>
  );
};

export const S12: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const f = useCurrentFrame();
  const move = lin(f, at(0.4), at(0.62));
  const centers = [
    { x: 560, y: 420, c: C.spam },
    { x: 1000, y: 640, c: C.ok },
    { x: 1400, y: 360, c: C.rule },
  ];
  return (
    <AbsoluteFill>
      <Abs x={960} y={70}><Title size={64}>Học không giám sát <span style={{ color: C.muted }}>(Unsupervised)</span></Title></Abs>
      <Abs x={960} y={150}><Text color={C.muted}>Không có nhãn → máy tự gom nhóm</Text></Abs>
      {Array.from({ length: 42 }).map((_, i) => {
        const g = centers[i % 3];
        const x0 = 260 + random(`x${i}`) * 1400, y0 = 240 + random(`y${i}`) * 580;
        const x1 = g.x + (random(`a${i}`) - 0.5) * 220, y1 = g.y + (random(`b${i}`) - 0.5) * 180;
        return (
          <Pop key={i} at={at(0.05) + i} scale style={{ position: "absolute", left: interpolate(move, [0, 1], [x0, x1]) - 20, top: interpolate(move, [0, 1], [y0, y1]) - 20 }}>
            <div style={{ width: 40, height: 40, borderRadius: "50%", background: move > 0.6 ? g.c : C.muted }} />
          </Pop>
        );
      })}
      {centers.map((g, i) => (
        <Abs key={i} x={g.x} y={g.y - 170}>
          <Pop at={at(0.66)}><Text size={40} color={g.c}>Nhóm {i + 1}</Text></Pop>
        </Abs>
      ))}
      <Abs x={560} y={590}>
        <Pop at={at(0.84)} style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: 10 }}>
          <Hand size={70} />
          <Box color={C.spam}><Text size={40}>"Hay mua đồ ăn vặt"</Text></Box>
        </Pop>
      </Abs>
      <Abs x={960} y={870}>
        <Pop at={at(0.86)}><Text size={44} color={C.rule}>Máy chỉ gom nhóm · con người đặt tên</Text></Pop>
      </Abs>
    </AbsoluteFill>
  );
};

export const S13: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const f = useCurrentFrame();
  const N = 5, S = 130, X0 = 300, Y0 = 170;
  const red = ["1,1", "3,1", "1,3", "3,3", "2,3"];
  const goal = [4, 4];
  // 3 lượt thử: lượt 1 giẫm ô đỏ, lượt 2 tới đích đường dài, lượt 3 đường ngắn
  const eps: [number, number][][] = [
    [[0, 0], [1, 0], [1, 1]],
    [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4], [4, 4]],
    [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [3, 2], [4, 2], [4, 3], [4, 4]],
  ];
  const ep = Math.min(2, Math.floor(lin(f, at(0.1), at(0.85)) * 3));
  const epStart = at(0.1) + ((at(0.85) - at(0.1)) / 3) * ep;
  const path = eps[ep];
  const k = Math.min(path.length - 1, Math.max(0, (f - epStart) / 7));
  const i0 = Math.floor(k), i1 = Math.min(path.length - 1, i0 + 1), t = k - i0;
  const pos = [path[i0][0] + (path[i1][0] - path[i0][0]) * t, path[i0][1] + (path[i1][1] - path[i0][1]) * t];
  const done = k >= path.length - 1;
  const reward = ep === 0 ? "−1" : "+10";
  return (
    <AbsoluteFill>
      <Abs x={1360} y={110}><Title size={60}>Học tăng cường</Title></Abs>
      <Abs x={1360} y={180}><Text size={40} color={C.muted}>Reinforcement Learning</Text></Abs>
      {Array.from({ length: N * N }).map((_, i) => {
        const x = i % N, y = Math.floor(i / N);
        const isRed = red.includes(`${x},${y}`);
        const isGoal = x === goal[0] && y === goal[1];
        return <div key={i} style={{ position: "absolute", left: X0 - 180 + x * S, top: Y0 + y * S, width: S - 8, height: S - 8, borderRadius: 14, background: isRed ? "#7F1D1D" : isGoal ? "#14532D" : C.panel, border: `3px solid ${isRed ? C.bad : isGoal ? C.ok : C.line}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 34, fontFamily: FONT, fontWeight: 800, color: C.ok }}>{isGoal ? "ĐÍCH" : ""}</div>;
      })}
      <div style={{ position: "absolute", left: X0 - 180 + pos[0] * S + 18, top: Y0 + pos[1] * S + 18, width: S - 44, height: S - 44, borderRadius: "50%", background: C.ai, border: "6px solid #fff" }} />
      <Abs x={1360} y={380}>
        <Text size={44} color={C.muted}>Lượt thử {ep + 1}</Text>
        {done ? (
          <Pop at={Math.round(epStart + (path.length - 1) * 7)} scale>
            <Title size={110} color={ep === 0 ? C.bad : C.ok}>{reward}</Title>
          </Pop>
        ) : null}
        {ep === 2 && done ? <Text size={38} color={C.muted}>Đường đi ngắn hơn</Text> : null}
      </Abs>
      <Abs x={1360} y={760}>
        <Pop at={at(0.25)}><Box color={C.rule}><Text size={44}>Thử – sai – thưởng/phạt</Text></Box></Pop>
        <Pop at={at(0.55)}><Text size={38} color={C.muted} style={{ marginTop: 20 }}>AI chơi game · robot tập đi</Text></Pop>
      </Abs>
    </AbsoluteFill>
  );
};

export const S14: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const rows = [
    { k: "Có giám sát", data: "Có đáp án (nhãn)", q: "Cái này là gì? Bao nhiêu?", icon: <MailIconTag key="a" /> },
    { k: "Không giám sát", data: "Không có đáp án", q: "Có những nhóm nào?", icon: <Dots key="b" /> },
    { k: "Tăng cường", data: "Chỉ có thưởng/phạt", q: "Nên làm hành động nào?", icon: <Title key="c" size={64} color={C.ok}>+10</Title> },
  ];
  const starts = [0.05, 0.4, 0.7];
  return (
    <AbsoluteFill>
      <div style={{ position: "absolute", left: 80, right: 80, top: 60, display: "grid", gridTemplateColumns: "180px 420px 1fr 1fr", gap: 20, alignItems: "center" }}>
        <div /><Text color={C.muted}>Kiểu học</Text><Text color={C.muted}>Dữ liệu</Text><Text color={C.muted}>Trả lời</Text>
        {rows.map((r, i) => (
          <React.Fragment key={r.k}>
            <Pop at={at(starts[i])} style={{ display: "flex", justifyContent: "center" }}>{r.icon}</Pop>
            <Pop at={at(starts[i])}><Box color={C.ml}><Text size={46} color={C.ai}>{r.k}</Text></Box></Pop>
            <Pop at={at(starts[i] + 0.04)}><Box style={{ minHeight: 120 }}><Text size={42}>{r.data}</Text></Box></Pop>
            <Pop at={at(starts[i] + 0.08)}><Box style={{ minHeight: 120 }}><Text size={42} color={C.rule}>{r.q}</Text></Box></Pop>
          </React.Fragment>
        ))}
      </div>
    </AbsoluteFill>
  );
};
const MailIconTag = () => <div style={{ position: "relative" }}><EmailCard text="" tone="spam" w={120} size={30} /></div>;
const Dots = () => (
  <svg width={140} height={100} viewBox="0 0 140 100">
    {[[20, 30, C.spam], [36, 18, C.spam], [30, 44, C.spam], [100, 30, C.ok], [116, 44, C.ok], [66, 80, C.rule], [82, 72, C.rule]].map(([x, y, c], i) => (
      <circle key={i} cx={x as number} cy={y as number} r={11} fill={c as string} />
    ))}
  </svg>
);

// ---------- PHẦN E ----------
export const S15: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const f = useCurrentFrame();
  const pulse = 0.6 + 0.4 * Math.sin(f / 6);
  const node = (x: number, y: number, start: number, label: string, color: string, w = 300, small?: string) => (
    <Abs x={x} y={y}>
      <Pop at={at(start)} scale>
        <Box color={color} w={w} style={{ minHeight: 120 }}>
          <Text size={44} color={color}>{label}</Text>
          {small ? <Text size={30} color={C.muted}>{small}</Text> : null}
        </Box>
      </Pop>
    </Abs>
  );
  const line = (x1: number, y1: number, x2: number, y2: number, start: number) => {
    const p = lin(f, at(start), at(start) + 15);
    return <line x1={x1} y1={y1} x2={x1 + (x2 - x1) * p} y2={y1 + (y2 - y1) * p} stroke={C.muted} strokeWidth="5" />;
  };
  return (
    <AbsoluteFill>
      <Abs x={960} y={50}><Title size={56}>Bản đồ khóa học</Title></Abs>
      <svg width={1920} height={940} style={{ position: "absolute" }}>
        {line(560, 190, 760, 190, 0.1)}
        {line(1160, 190, 1360, 190, 0.16)}
        {line(1500, 250, 460, 420, 0.27)}
        {line(1500, 250, 960, 420, 0.27)}
        {line(1500, 250, 1500, 420, 0.27)}
        {line(460, 540, 960, 760, 0.83)}
        {line(960, 540, 960, 760, 0.83)}
      </svg>
      {node(400, 190, 0.06, "Python", C.ai, 320, "Nền tảng")}
      {node(960, 190, 0.12, "Machine Learning", C.ml, 400)}
      {node(1500, 190, 0.18, "Deep Learning", "#60A5FA", 380)}
      {node(460, 480, 0.3, "Computer Vision", C.ok, 420, "Thị giác máy tính · đọc biển số")}
      {node(960, 480, 0.47, "NLP", C.spam, 420, "Xử lý ngôn ngữ · khen hay chê")}
      {node(1500, 480, 0.6, "Lĩnh vực khác", C.muted, 420, "(khóa học không đi sâu)")}
      <Abs x={1240} y={640}>
        <Pop at={at(0.68)}>
          <div style={{ background: "#431407", border: `3px dashed ${C.spam}`, borderRadius: 18, padding: "10px 24px" }}>
            <Text size={34}>ChatGPT = AI tạo sinh cho văn bản (DL gặp NLP)</Text>
          </div>
        </Pop>
      </Abs>
      {node(960, 820, 0.85, "Triển khai sản phẩm", C.rule, 640, "API · Docker · Cloud")}
      <div style={{ position: "absolute", left: 220, top: 260, display: "flex", alignItems: "center", gap: 10, opacity: f > at(0.08) ? 1 : 0 }}>
        <div style={{ width: 30, height: 30, borderRadius: "50%", background: C.rule, transform: `scale(${pulse})`, boxShadow: `0 0 24px ${C.rule}` }} />
        <Text size={32} color={C.rule}>Bạn ở đây (giai đoạn 0)</Text>
      </div>
    </AbsoluteFill>
  );
};

// ---------- PHẦN F ----------
const CODE1 = `def loc_spam_bang_luat(email):
    if "trúng thưởng" in email.lower():
        return "SPAM"
    return "Bình thường"`;
const CODE2 = `emails = [
    "Bạn đã trúng thưởng iPhone, nhận ngay hôm nay",
    "Quà tặng miễn phí, bấm link nhận ngay",
    ...  # thêm 2 email spam
    "Lịch họp dự án chuyển sang 3 giờ chiều",
    ...  # thêm 3 email bình thường
]
nhan = ["SPAM", "SPAM", "SPAM", "SPAM",
        "Bình thường", "Bình thường", "Bình thường", "Bình thường"]`;
const CODE3 = `bo_dem_tu = CountVectorizer()
X = bo_dem_tu.fit_transform(emails)
mo_hinh = MultinomialNB()
mo_hinh.fit(X, nhan)   # cho máy học`;

export const S16: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const f = useCurrentFrame();
  const second = f >= at(0.46);
  return (
    <AbsoluteFill style={{ alignItems: "center" }}>
      <div style={{ marginTop: 40 }}>
        <Title size={56} color={second ? C.ml : C.rule}>{second ? "Cách 2: Machine Learning – dữ liệu + nhãn" : "Cách 1: Luật thủ công"}</Title>
      </div>
      <div style={{ marginTop: 40 }}>
        {second ? (
          <CodeBlock code={CODE2} size={38} progress={lin(f, at(0.48), at(0.8))} />
        ) : (
          <CodeBlock code={CODE1} size={48} progress={lin(f, at(0.08), at(0.3))} hl={f > at(0.3) ? { 1: "#FACC1533" } : {}} />
        )}
      </div>
      {second ? (
        <Pop at={at(0.82)} style={{ marginTop: 30 }}>
          <Text size={44}>
            <span style={{ color: C.spam }}>4 email SPAM</span> + <span style={{ color: C.ok }}>4 email bình thường</span> · nhãn do người dán
          </Text>
        </Pop>
      ) : null}
    </AbsoluteFill>
  );
};

export const S17: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const f = useCurrentFrame();
  if (f < at(0.36)) {
    return (
      <AbsoluteFill style={{ alignItems: "center" }}>
        <div style={{ marginTop: 40 }}><Title size={56} color={C.ml}>Cho máy học</Title></div>
        <div style={{ marginTop: 50 }}>
          <CodeBlock code={CODE3} size={48} progress={lin(f, at(0.02), at(0.14))} hl={{ 0: f > at(0.12) && f < at(0.2) ? "#3B82F633" : "transparent", 1: f > at(0.12) && f < at(0.2) ? "#3B82F633" : "transparent", 3: f > at(0.2) ? "#22C55E44" : "transparent" }} />
        </div>
        <Pop at={at(0.13)} style={{ marginTop: 30 }}><Text size={44} color={C.muted}>Bộ đếm từ: chữ → dãy số</Text></Pop>
        <Pop at={at(0.21)} style={{ marginTop: 10 }}><Text size={48} color={C.ok}>fit = cho máy học, tự tìm quy luật</Text></Pop>
      </AbsoluteFill>
    );
  }
  const words = ["Chúc mừng, bạn nhận ", "quà", " ", "miễn phí", " ", "ngay"];
  const under = f > at(0.86);
  return (
    <AbsoluteFill>
      <Abs x={960} y={70}><Title size={56}>Thử trên email mới</Title></Abs>
      <Abs x={960} y={230}>
        <Pop at={at(0.38)}>
          <div style={{ background: "#F8FAFC", borderRadius: 20, padding: "24px 40px", fontFamily: FONT, fontWeight: 700, fontSize: 52, color: "#0F172A" }}>
            {words.map((w, i) => (
              <span key={i} style={{ textDecoration: under && i % 2 === 1 ? `underline 6px ${C.spam}` : "none", textUnderlineOffset: 10 }}>{w}</span>
            ))}
          </div>
          <Text size={34} color={C.muted} style={{ marginTop: 14 }}>Không có chữ "trúng thưởng"</Text>
        </Pop>
      </Abs>
      <div style={{ position: "absolute", top: 440, width: "100%", display: "flex", justifyContent: "center", gap: 80 }}>
        <Pop at={at(0.5)}>
          <Box color={C.rule} w={680} h={300}>
            <Text size={44} color={C.rule}>Luật thủ công</Text>
            <div style={{ display: "flex", alignItems: "center", gap: 20, marginTop: 20 }}>
              <Text size={60}>Bình thường</Text> <Cross size={70} />
            </div>
            <Text size={34} color={C.bad}>bị lọt</Text>
          </Box>
        </Pop>
        <Pop at={at(0.6)}>
          <Box color={C.ml} w={680} h={300} style={{ boxShadow: f > at(0.6) ? `0 0 40px ${C.ml}` : undefined }}>
            <Text size={44} color={C.ai}>Machine Learning</Text>
            <div style={{ display: "flex", alignItems: "center", gap: 20, marginTop: 20 }}>
              <Text size={60} color={C.spam}>SPAM</Text> <Check size={70} />
            </div>
            <Text size={34} color={C.ok}>bắt được</Text>
          </Box>
        </Pop>
      </div>
      <Abs x={960} y={860}>
        <Pop at={at(0.72)}><Text size={44} color={C.muted}>Không ai viết luật "miễn phí là spam"</Text></Pop>
      </Abs>
    </AbsoluteFill>
  );
};

// ---------- PHẦN G ----------
const FlipCard: React.FC<{ at: number; flipAt: number; front: string; back: string; icon: React.ReactNode }> = ({ at, flipAt, front, back, icon }) => {
  const f = useCurrentFrame();
  const a = interpolate(f, [flipAt, flipAt + 14], [0, 180], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const showBack = a > 90;
  return (
    <Pop at={at}>
      <div style={{ width: 540, height: 560, perspective: 1600 }}>
        <div style={{ width: "100%", height: "100%", transform: `rotateY(${showBack ? a - 180 : a}deg)`, borderRadius: 28, border: `5px solid ${showBack ? C.ok : C.bad}`, background: showBack ? "#052E16" : "#450A0A", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 30, padding: 30, boxSizing: "border-box" }}>
          {showBack ? <Check size={80} /> : <Cross size={80} />}
          {showBack ? icon : null}
          <Text size={showBack ? 40 : 50}>{showBack ? back : front}</Text>
        </div>
      </div>
    </Pop>
  );
};

export const S18: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  return (
    <AbsoluteFill>
      <Abs x={960} y={60}><Title size={56}>3 hiểu nhầm hay gặp</Title></Abs>
      <div style={{ position: "absolute", top: 150, width: "100%", display: "flex", justifyContent: "center", gap: 50 }}>
        <FlipCard at={at(0.06)} flipAt={at(0.14)} front="AI phải biết học" back="Cờ vua if-else vẫn là AI" icon={<Chess size={140} />} />
        <FlipCard at={at(0.28)} flipAt={at(0.38)} front="DL luôn tốt hơn" back="Dữ liệu bảng: thử ML cổ điển trước" icon={<TableIcon size={150} />} />
        <FlipCard at={at(0.52)} flipAt={at(0.68)} front="8 email là đủ" back="Cần hàng nghìn email · đo mô hình ở bài 2.5" icon={<MailIcon_ />} />
      </div>
    </AbsoluteFill>
  );
};
const MailIcon_ = () => (
  <div style={{ display: "flex", gap: 6 }}>
    {Array.from({ length: 8 }).map((_, i) => (
      <svg key={i} width={34} height={26} viewBox="0 0 32 24"><rect x="1" y="1" width="30" height="22" rx="3" fill="none" stroke={C.ai} strokeWidth="3" /></svg>
    ))}
  </div>
);

export const S19: React.FC<SceneProps> = ({ d }) => {
  const at = useAt(d);
  const f = useCurrentFrame();
  const tiles = [
    { start: 0.12, t: "Dữ liệu + Kết quả → Luật", icon: <Computer w={200} label="Mô hình" /> },
    { start: 0.42, t: "AI ⊃ ML ⊃ DL", icon: <Rings /> },
    { start: 0.58, t: "3 kiểu học", icon: <Dots /> },
    { start: 0.8, t: "Dữ liệu bảng → ML cổ điển", icon: <TableIcon size={140} /> },
  ];
  return (
    <AbsoluteFill>
      <div style={{ position: "absolute", left: 160, right: 160, top: 50, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 40 }}>
        {tiles.map((x, i) => {
          const lit = f >= at(x.start);
          return (
            <Box key={i} color={lit ? C.ml : C.line} h={400} style={{ opacity: lit ? 1 : 0.35, boxShadow: lit ? `0 0 30px ${C.ml}55` : undefined, gap: 24 }}>
              {x.icon}
              <Text size={50}>{x.t}</Text>
            </Box>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
const Rings = () => (
  <svg width={180} height={180} viewBox="0 0 100 100">
    <circle cx="50" cy="50" r="48" fill={C.ai} />
    <circle cx="50" cy="56" r="33" fill={C.ml} />
    <circle cx="50" cy="64" r="18" fill={C.dl} />
  </svg>
);

export const S20: React.FC<SceneProps> = ({ d, countdownAt = Math.round(d / 2) }) => {
  const f = useCurrentFrame();
  const left = 5 - Math.floor((f - countdownAt) / 30);
  const answered = f >= countdownAt + 150;
  return (
    <AbsoluteFill>
      <div style={{ position: "absolute", top: 80, width: "100%", display: "flex", justifyContent: "center", gap: 60, alignItems: "center" }}>
        <Pop at={5}><Tea size={240} /></Pop>
        <Pop at={12}><Calendar size={220} label="Ngày mai" /></Pop>
        <Pop at={20}><Title size={80} color={C.rule}>Bán bao nhiêu ly?</Title></Pop>
      </div>
      <Abs x={960} y={520}>
        <Pop at={30}><Title size={84}>Hồi quy hay Phân loại?</Title></Pop>
      </Abs>
      <Abs x={960} y={770}>
        {f >= countdownAt && !answered ? (
          <div style={{ width: 170, height: 170, borderRadius: "50%", border: `8px solid ${C.rule}`, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Title size={100} color={C.rule}>{left}</Title>
          </div>
        ) : null}
        {answered ? (
          <Pop at={countdownAt + 150} scale style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: 24 }}>
            <Check size={90} />
            <Title size={80} color={C.ok}>Hồi quy – đoán một con số</Title>
          </Pop>
        ) : null}
      </Abs>
    </AbsoluteFill>
  );
};

export const S21: React.FC<SceneProps> = () => (
  <AbsoluteFill style={{ alignItems: "center", justifyContent: "center" }}>
    <Pop at={5}><Text size={52} color={C.muted}>Tiếp theo</Text></Pop>
    <Pop at={12}><Title size={110} color={C.ai}>Bài 0.2</Title></Pop>
    <Pop at={20}><Title size={60}>Cài đặt môi trường và notebook đầu tiên</Title></Pop>
  </AbsoluteFill>
);

export const SCENES: Record<string, React.FC<SceneProps>> = {
  s01: S01, s02: S02, s03: S03, s04: S04, s05: S05, s06: S06, s07: S07, s08: S08, s09: S09, s10: S10, s11: S11,
  s12: S12, s13: S13, s14: S14, s15: S15, s16: S16, s17: S17, s18: S18, s19: S19, s20: S20, s21: S21,
};
