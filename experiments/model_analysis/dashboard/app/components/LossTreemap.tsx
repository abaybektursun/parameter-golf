"use client";

import ReactECharts from "echarts-for-react";

interface Bucket {
  dimension: string;
  name: string;
  bpb: number;
  pct_bytes: number;
  pct_loss: number;
}

interface Props {
  buckets: Bucket[];
}

export default function LossTreemap({ buckets }: Props) {
  const typeBuckets = buckets.filter((b) => b.dimension === "type");

  const option = {
    backgroundColor: "transparent",
    tooltip: {
      backgroundColor: "#ffffff",
      borderColor: "#e4e4e7",
      textStyle: { color: "#27272a", fontSize: 12 },
      formatter: (p: { data: { name: string; bpb: number; pct_bytes: number; pct_loss: number } }) => {
        const d = p.data;
        return `<b>${d.name}</b><br/>BPB: ${d.bpb.toFixed(2)}<br/>Bytes: ${d.pct_bytes.toFixed(1)}%<br/>Loss: ${d.pct_loss.toFixed(1)}%`;
      },
    },
    series: [
      {
        type: "treemap",
        width: "100%",
        height: "100%",
        roam: false,
        nodeClick: false,
        breadcrumb: { show: false },
        label: {
          show: true,
          formatter: (p: { data: { name: string; bpb: number; pct_bytes: number } }) => {
            const d = p.data;
            return `{title|${d.name}}\n{sub|${d.bpb.toFixed(2)} BPB · ${d.pct_bytes.toFixed(1)}%}`;
          },
          rich: {
            title: { fontSize: 14, fontWeight: 600, lineHeight: 22 },
            sub: { fontSize: 12, lineHeight: 18, color: "rgba(255,255,255,0.7)" },
          },
          overflow: "truncate" as const,
          ellipsis: "...",
        },
        itemStyle: {
          borderColor: "#e4e4e7",
          borderWidth: 3,
          borderRadius: 4,
          gapWidth: 2,
        },
        visualMin: 0.8,
        visualMax: 3.5,
        visualDimension: "bpb",
        levels: [
          {
            colorMappingBy: "value" as const,
            itemStyle: { borderWidth: 0, gapWidth: 3 },
          },
        ],
        data: typeBuckets.map((b) => ({
          name: b.name,
          value: b.pct_bytes,
          bpb: b.bpb,
          pct_bytes: b.pct_bytes,
          pct_loss: b.pct_loss,
          itemStyle: {
            color: bpbColor(b.bpb),
          },
        })),
      },
    ],
  };

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-6">
      <h3 className="text-lg font-medium mb-1">Token Type Treemap</h3>
      <p className="text-sm text-[var(--muted)] mb-4">
        Area = share of bytes. Color = BPB difficulty (blue = easy, red = hard).
      </p>
      <ReactECharts option={option} style={{ height: 360 }} />
    </div>
  );
}

function bpbColor(bpb: number): string {
  // Map BPB to RdYlBu-like scale: low BPB (0.8) = blue, high BPB (3.5) = red
  const t = Math.max(0, Math.min(1, (bpb - 0.8) / (3.5 - 0.8)));
  // Blue (#4575b4) -> Yellow (#ffffbf) -> Red (#d73027)
  if (t < 0.5) {
    const s = t * 2;
    return lerpColor([69, 117, 180], [255, 255, 191], s);
  }
  const s = (t - 0.5) * 2;
  return lerpColor([255, 255, 191], [215, 48, 39], s);
}

function lerpColor(a: number[], b: number[], t: number): string {
  const r = Math.round(a[0] + (b[0] - a[0]) * t);
  const g = Math.round(a[1] + (b[1] - a[1]) * t);
  const bl = Math.round(a[2] + (b[2] - a[2]) * t);
  return `rgb(${r},${g},${bl})`;
}
