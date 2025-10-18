import React from "react";

interface StarProps {
  top: number;       // %
  left: number;      // %
  size: number;      // px
  opacity: number;   // 0–1
  driftX: number;    // max horizontal drift in px
  driftY: number;    // max vertical drift in px
  driftDuration: number; // seconds
  twinkleDuration: number; // seconds
  twinkleDelay?: number; // seconds
}

const Star: React.FC<StarProps> = ({
  top,
  left,
  size,
  opacity,
  driftX,
  driftY,
  driftDuration,
  twinkleDuration,
  twinkleDelay = 0,
}) => {
  const style: React.CSSProperties = {
    position: "absolute",
    top: `${top}%`,
    left: `${left}%`,
    width: `${size}px`,
    height: `${size}px`,
    borderRadius: "50%",
    backgroundColor: "white",
    opacity,
    pointerEvents: "none",
    animation: `drift ${driftDuration}s ease-in-out infinite alternate,
                twinkle ${twinkleDuration}s ease-in-out infinite`,
    animationDelay: `0s, ${twinkleDelay}s`,
    "--dx": `${driftX}px`,
    "--dy": `${driftY}px`,
  } as React.CSSProperties;

  return <div style={style} />;
};

export default Star;