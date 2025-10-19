import React from "react";
import Star from "./Star";

interface StarData {
  top: number;
  left: number;
  size: number;
  opacity: number;
  driftX: number;
  driftY: number;
  driftDuration: number;
  twinkleDuration: number;
  twinkleDelay: number;
}

interface StarfieldProps {
  numStars?: number;
  clusters?: number;
  connectLines?: boolean;
  state?: string;
}

const StarfieldWithConstellations: React.FC<StarfieldProps> = ({
  numStars = 70,
  clusters = 4,
  connectLines = true,
  state = 'ready',
}) => {
  const stars: StarData[] = [];

  const clusterCenters = Array.from({ length: clusters }, () => ({
    top: Math.random() * 80 + 10,
    left: Math.random() * 80 + 10,
  }));

  for (let i = 0; i < numStars; i++) {
    let top: number, left: number;

    if (clusterCenters.length && Math.random() < 0.3) {
      const cluster = clusterCenters[Math.floor(Math.random() * clusterCenters.length)];
      top = cluster.top + (Math.random() * 6 - 3);
      left = cluster.left + (Math.random() * 6 - 3);
    } else {
      top = Math.random() * 100;
      left = Math.random() * 100;
    }

    stars.push({
      top,
      left,
      size: Math.random() * 2 + 1.5,
      opacity: Math.random() * 0.5 + 0.5,
      driftX: 0,
      driftY: 0,
      driftDuration: 0,
      twinkleDuration: Math.random() * 5 + 2,
      twinkleDelay: Math.random() * 5,
    });
  }

  const lines: { x1: number; y1: number; x2: number; y2: number }[] = [];

  if (connectLines) {
    const starCoords = stars.map((s) => ({ x: s.left, y: s.top }));
    starCoords.forEach((star, i) => {
      const distances = starCoords
        .map((s2, j) => ({ index: j, dist: Math.hypot(s2.x - star.x, s2.y - star.y) }))
        .filter((d) => d.index !== i)
        .sort((a, b) => a.dist - b.dist)
        .slice(0, 2); // connect to 2 closest
      distances.forEach((d) => {
        lines.push({
          x1: star.x,
          y1: star.y,
          x2: starCoords[d.index].x,
          y2: starCoords[d.index].y,
        });
      });
    });
  }

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "#040406ff", // dark space
        zIndex: -2,
        pointerEvents: "none",
      }}
    >
      {state === 'ready' && (
        <>
          {stars.map((s, i) => (
            <Star key={i} {...s} />
          ))}

          {connectLines && (
            <svg
              style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
              width="100%"
              height="100%"
            >
              {lines.map((l, i) => (
                <line
                  key={i}
                  x1={`${l.x1}%`}
                  y1={`${l.y1}%`}
                  x2={`${l.x2}%`}
                  y2={`${l.y2}%`}
                  stroke="rgba(255,255,255,0.15)"
                  strokeWidth={0.75}
                />
              ))}
      </svg>
    )}
  </>
)}
    </div>
  );
};

export default StarfieldWithConstellations;