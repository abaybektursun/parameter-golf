"use client";

import * as Plot from "@observablehq/plot";
import { usePlot } from "./usePlot";

interface ProbePoint {
  label: string;
  layer_idx: number;
  loss_nats: number;
  bits_per_token: number;
  top1_accuracy: number;
  delta_bits_per_token: number | null;
  is_encoder: boolean;
  is_decoder: boolean;
}

interface Props {
  probePoints: ProbePoint[];
  numEncoderLayers: number;
}

export default function LogitLensChart({ probePoints, numEncoderLayers }: Props) {
  // Assign sequential x index for clean spacing
  const data = probePoints.map((p, i) => ({ ...p, index: i }));
  const encoderEnd = numEncoderLayers; // index where decoder starts (after embed)

  const yMax = Math.ceil(Math.max(...data.map((d) => d.loss_nats)) * 1.05);

  const containerRef = usePlot(
    (width) =>
      Plot.plot({
        width,
        height: 420,
        style: { background: "transparent", color: "#71717a", fontSize: "12px" },
        marginLeft: 56,
        marginRight: 56,
        marginBottom: 50,
        x: {
          label: null,
          tickFormat: (_: unknown, i: number) => data[i]?.label ?? "",
          tickRotate: -35,
        },
        y: {
          label: "Loss (nats)",
          grid: true,
          domain: [0, yMax],
        },
        marks: [
          // Encoder background
          Plot.rectY(
            [{ x1: -0.5, x2: encoderEnd + 0.5, y1: 0, y2: yMax }],
            {
              x1: "x1",
              x2: "x2",
              y1: "y1",
              y2: "y2",
              fill: "#1e40af",
              fillOpacity: 0.06,
            }
          ),
          // Decoder background
          Plot.rectY(
            [{ x1: encoderEnd + 0.5, x2: data.length - 0.5, y1: 0, y2: yMax }],
            {
              x1: "x1",
              x2: "x2",
              y1: "y1",
              y2: "y2",
              fill: "#991b1b",
              fillOpacity: 0.06,
            }
          ),
          // Loss line
          Plot.lineY(data, {
            x: "index",
            y: "loss_nats",
            stroke: "#1e40af",
            strokeWidth: 2.5,
            curve: "catmull-rom",
          }),
          // Loss dots
          Plot.dot(data, {
            x: "index",
            y: "loss_nats",
            fill: (d: typeof data[number]) => (d.delta_bits_per_token !== null && d.delta_bits_per_token > 0 ? "#991b1b" : "#1e40af"),
            r: 6,
            stroke: "#e4e4e7",
            strokeWidth: 2,
            tip: true,
            title: (d: typeof data[number]) =>
              `${d.label}\nLoss: ${d.loss_nats.toFixed(2)} nats\nAccuracy: ${(d.top1_accuracy * 100).toFixed(1)}%${
                d.delta_bits_per_token !== null
                  ? `\nΔ bits/tok: ${d.delta_bits_per_token > 0 ? "+" : ""}${d.delta_bits_per_token.toFixed(2)}`
                  : ""
              }`,
          }),
          // Accuracy line (secondary, scaled to fit y domain)
          Plot.lineY(data, {
            x: "index",
            y: (d: typeof data[number]) => d.top1_accuracy * yMax,
            stroke: "#166534",
            strokeWidth: 1.5,
            strokeDasharray: "6,3",
            curve: "catmull-rom",
          }),
          // Labels for encoder/decoder — use frameAnchor to avoid hardcoded y
          Plot.text(["Encoder"], {
            frameAnchor: "top-left",
            dx: 8,
            dy: 8,
            fill: "#1e40af",
            fillOpacity: 0.4,
            fontSize: 13,
            fontWeight: "600",
          }),
          Plot.text(["Decoder"], {
            frameAnchor: "top-right",
            dx: -8,
            dy: 8,
            fill: "#991b1b",
            fillOpacity: 0.4,
            fontSize: 13,
            fontWeight: "600",
          }),
        ],
      }),
    [data, encoderEnd, yMax]
  );

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-6">
      <div className="flex items-center gap-6 mb-4 text-sm text-[var(--muted)]">
        <span className="flex items-center gap-2">
          <span className="inline-block w-3 h-0.5 bg-[var(--accent-blue)]" />
          Loss (nats)
        </span>
        <span className="flex items-center gap-2">
          <span
            className="inline-block w-3 h-0.5 bg-[var(--accent-green)]"
            style={{ borderTop: "1px dashed var(--accent-green)" }}
          />
          Top-1 accuracy (scaled)
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-[var(--accent-red)]" />
          Loss increased (readability dropped)
        </span>
      </div>
      <div ref={containerRef} />
    </div>
  );
}
