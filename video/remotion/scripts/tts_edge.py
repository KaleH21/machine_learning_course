# Tạo giọng đọc miễn phí bằng edge-tts từ script.json.
# Chạy: python scripts/tts_edge.py [s01 s02 ...]   (không truyền id = tạo tất cả)
import asyncio, json, re, sys
from pathlib import Path
import edge_tts

VOICE = "vi-VN-HoaiMyNeural"
RATE = "-5%"  # chậm nhẹ cho bài giảng
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "bai_0_01" / "audio"


def clean(t: str) -> str:
    # bỏ thẻ điều giọng của ElevenLabs ([pause], [warmly]...)
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]\s*", "", t)).strip()


assert clean("[pause] Một  hai. [warmly]Ba") == "Một hai. Ba"


async def main(only):
    OUT.mkdir(parents=True, exist_ok=True)
    for s in json.loads((ROOT / "src" / "bai_0_01" / "script.json").read_text(encoding="utf-8")):
        out = OUT / f"{s['id']}.mp3"
        if only and s["id"] not in only:
            continue
        if not only and out.exists() and out.stat().st_size > 10_000:
            continue  # đã có, bỏ qua (truyền id cụ thể để tạo lại)
        for attempt in range(6):  # dịch vụ hay trả NoAudioReceived khi bị giới hạn tạm thời
            try:
                await edge_tts.Communicate(clean(s["tts"]), VOICE, rate=RATE).save(str(out))
                break
            except edge_tts.exceptions.NoAudioReceived:
                out.unlink(missing_ok=True)
                if attempt == 5:
                    raise
                await asyncio.sleep(5 * (attempt + 1))
        print("ok", s["id"])
        await asyncio.sleep(2)


asyncio.run(main(sys.argv[1:]))
