"use client";

import ReactECharts from "echarts-for-react";

interface Step {
  step: number;
  train_loss: number | null;
  val_loss: number | null;
  val_bpb: number | null;
  train_time_ms: number;
}

interface Event {
  step: number;
  type: string;
  label: string;
}

interface Props {
  steps: Step[];
  events: Event[];
}

function fmtStep(s: number) {
  return s >= 1000 ? `${s / 1000}k` : String(s);
}

export default function TrainingCurve({ steps, events }: Props) {
  const trainData = steps
    .filter((s) => s.train_loss !== null && s.step > 0)
    .map((s) => [s.step, s.train_loss]);

  const valData = steps
    .filter((s) => s.val_loss !== null)
    .map((s) => [s.step, s.val_loss]);

  const option = {
    backgroundColor: "transparent",
    grid: { left: 56, right: 56, top: 24, bottom: 40, containLabel: false },
    tooltip: {
      trigger: "axis" as const,
      backgroundColor: "#ffffff",
      borderColor: "#e4e4e7",
      textStyle: { color: "#27272a", fontSize: 12 },
    },
    xAxis: {
      type: "value" as const,
      name: "Step",
      nameLocation: "middle" as const,
      nameGap: 28,
      axisLabel: {
        color: "#71717a",
        formatter: (v: number) => fmtStep(v),
      },
      axisLine: { lineStyle: { color: "#e4e4e7" } },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value" as const,
      name: "Loss (nats)",
      nameTextStyle: { color: "#71717a" },
      min: 1.5,
      max: 9,
      axisLabel: { color: "#71717a" },
      axisLine: { lineStyle: { color: "#e4e4e7" } },
      splitLine: { lineStyle: { color: "#f4f4f5" } },
    },
    series: [
      {
        name: "Train Loss",
        type: "line",
        data: trainData,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: "#1e40af", width: 1.5 },
        itemStyle: { color: "#1e40af" },
      },
      {
        name: "Val Loss",
        type: "scatter",
        data: valData,
        symbolSize: 8,
        itemStyle: { color: "#166534" },
      },
      {
        name: "Events",
        type: "line",
        markLine: {
          silent: true,
          symbol: "none",
          label: {
            show: true,
            position: "end" as const,
            formatter: (p: { dataIndex: number }) => String.fromCharCode(65 + p.dataIndex),
            color: "#92400e",
            fontSize: 10,
            fontWeight: 600,
          },
          lineStyle: {
            color: "#92400e",
            type: "dashed" as const,
            width: 1,
          },
          data: events.map((e) => ({ xAxis: e.step })),
        },
        data: [],
      },
    ],
  };

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-6">
      <div className="flex flex-wrap gap-x-4 gap-y-1 mb-3 text-xs text-[var(--muted)]">
        {events.map((e, i) => (
          <span key={i}>
            <span className="text-[var(--accent-amber)] font-semibold mr-1">{String.fromCharCode(65 + i)}</span>
            {fmtStep(e.step)} — {e.label}
          </span>
        ))}
      </div>
      <ReactECharts option={option} style={{ height: 400 }} />
      <div className="flex items-center gap-6 mt-2 text-sm text-[var(--muted)]">
        <span className="flex items-center gap-2">
          <span className="inline-block w-3 h-0.5 bg-[var(--accent-blue)]" />
          Train loss
        </span>
        <span className="flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-[var(--accent-green)]" />
          Val loss
        </span>
      </div>
    </div>
  );
}
