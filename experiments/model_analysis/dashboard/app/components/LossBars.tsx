"use client";

import * as Plot from "@observablehq/plot";
import { usePlot } from "./usePlot";

interface Bucket {
  dimension: string;
  name: string;
  bpb: number;
  pct_bytes: number;
  pct_loss: number;
}

interface Props {
  buckets: Bucket[];
  dimension: "frequency" | "position" | "type";
  title: string;
}

export default function LossBars({ buckets, dimension, title }: Props) {
  const data = buckets.filter((b) => b.dimension === dimension);

  const containerRef = usePlot(
    (width) =>
      Plot.plot({
        width,
        height: 280,
        style: { background: "transparent", color: "#71717a", fontSize: "12px" },
        marginLeft: 100,
        marginRight: 20,
        marginBottom: 40,
        x: { label: "BPB", grid: true, domain: [0, Math.max(...data.map((d) => d.bpb)) * 1.15] },
        y: { label: null },
        color: {
          type: "linear",
          scheme: "RdYlBu",
          domain: [3.5, 0.8],
          reverse: false,
        },
        marks: [
          Plot.barX(data, {
            y: "name",
            x: "bpb",
            fill: "bpb",
            sort: { y: "-x" },
            rx: 3,
          }),
          Plot.text(data, {
            y: "name",
            x: "bpb",
            text: (d: Bucket) => `${d.bpb.toFixed(2)}  (${d.pct_bytes.toFixed(1)}% bytes)`,
            dx: 6,
            textAnchor: "start",
            fill: "#71717a",
            fontSize: 11,
          }),
          Plot.ruleX([1.134], {
            stroke: "#991b1b",
            strokeWidth: 1,
            strokeDasharray: "4,3",
          }),
        ],
      }),
    [data]
  );

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-6">
      <h3 className="text-lg font-medium mb-1">{title}</h3>
      <p className="text-sm text-[var(--muted)] mb-4">
        Dashed red line = overall BPB (1.134)
      </p>
      <div ref={containerRef} />
    </div>
  );
}
