"use client";

import * as Plot from "@observablehq/plot";
import { usePlot } from "./usePlot";

interface Matrix {
  label: string;
  layer: number;
  type: string;
  condition_number: number;
}

interface Props {
  matrices: Matrix[];
}

const TYPE_ORDER = ["Q", "K", "V", "Out", "MLP_up", "MLP_down"];

export default function ConditionHeatmap({ matrices }: Props) {
  const data = matrices.filter((m) => m.layer >= 0 && TYPE_ORDER.includes(m.type));

  const containerRef = usePlot(
    (width) =>
      Plot.plot({
        width: Math.min(width, 640),
        height: 440,
        style: { background: "transparent", color: "#71717a", fontSize: "12px" },
        marginLeft: 64,
        marginBottom: 50,
        padding: 0.05,
        x: {
          label: null,
          domain: TYPE_ORDER,
          tickRotate: -25,
        },
        y: {
          label: "Layer",
          domain: Array.from({ length: 11 }, (_, i) => i),
          tickFormat: (d: number) => `L${d}`,
        },
        color: {
          type: "log",
          scheme: "YlOrRd",
          domain: [2, 55000],
          label: "Condition #",
          legend: true,
        },
        marks: [
          Plot.cell(data, {
            x: "type",
            y: "layer",
            fill: "condition_number",
            rx: 3,
            tip: true,
            title: (d: Matrix) =>
              `${d.label}\nCondition #: ${d.condition_number.toFixed(0)}`,
          }),
          Plot.text(
            data.filter((d) => d.condition_number >= 1000),
            {
              x: "type",
              y: "layer",
              text: (d: Matrix) =>
                d.condition_number >= 10000
                  ? `${(d.condition_number / 1000).toFixed(0)}k`
                  : `${(d.condition_number / 1000).toFixed(1)}k`,
              fill: "white",
              fontSize: 10,
              fontWeight: "600",
            }
          ),
        ],
      }),
    [data]
  );

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-6">
      <h3 className="text-lg font-medium mb-1">Condition Number Heatmap</h3>
      <p className="text-sm text-[var(--muted)] mb-4">
        Higher condition number = expected to be more fragile. Q matrices top
        the chart — yet are least sensitive to quantization.
      </p>
      <div ref={containerRef} className="flex justify-center" />
    </div>
  );
}
