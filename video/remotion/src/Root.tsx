import { Composition } from "remotion";
import { Lesson, TIMELINE } from "./bai_0_01/Lesson";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Bai0-01-AI-ML-DL"
      component={Lesson}
      durationInFrames={TIMELINE.totalFrames}
      fps={TIMELINE.fps}
      width={1920}
      height={1080}
    />
  );
};
