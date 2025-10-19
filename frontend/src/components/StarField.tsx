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
  state?: string;
}

const Starfield: React.FC<StarfieldProps> = ({ numStars = 70, clusters = 4, state='ready'}) => {
  const stars: StarData[] = [];
  
  // cluster centers
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
      driftX: Math.random() * 200,
      driftY: Math.random() * 200,
      driftDuration: Math.random() * 60 + 40,   // 40–100s
      twinkleDuration: Math.random() * 5 + 2,   // 2–7s
      twinkleDelay: Math.random() * 5,          // random phase
    });
  }

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "transparent",
        zIndex: -1,
        pointerEvents: "none",
      }}
    >
      {state === 'ready' &&
        stars.map((s, i) => (
          <Star key={i} {...s} />
        ))
      }
    </div>
  );
};

export default Starfield;
