"use client";

import * as Plot from "@observablehq/plot";
import { usePlot } from "./usePlot";

interface HeadData {
  layer: number;
  head: number;
  prev_token_score: number;
  induction_score: number;
  bos_score: number;
  entropy: number;
  self_attn_score: number;
  offset_scores: number[];
  classification: string;
}

interface Summary {
  induction_count: number;
  prev_token_count: number;
  positional_count: number;
  other_count: number;
}

interface Props {
  heads: HeadData[];
  summary: Summary;
}

const CLASS_COLORS: Record<string, string> = {
  induction: "#1e40af",
  previous_token: "#166534",
  positional: "#92400e",
  other: "#d4d4d8",
};

const CLASS_LABELS: Record<string, string> = {
  induction: "Induction",
  previous_token: "Prev-token",
  positional: "Positional",
  other: "Other",
};

export default function AttentionHeadMap({ heads, summary }: Props) {
  // Classification heatmap
  const classRef = usePlot(
    (width) =>
      Plot.plot({
        width: Math.min(width, 500),
        height: 480,
        style: { background: "transparent", color: "#71717a", fontSize: "12px" },
        marginLeft: 50,
        marginBottom: 40,
        padding: 0.08,
        x: {
          label: "Head",
          domain: Array.from({ length: 8 }, (_, i) => i),
          tickFormat: (d: number) => `H${d}`,
        },
        y: {
          label: "Layer",
          domain: Array.from({ length: 11 }, (_, i) => i),
          tickFormat: (d: number) => `L${d}`,
        },
        color: {
          domain: ["induction", "previous_token", "positional", "other"],
          range: [CLASS_COLORS.induction, CLASS_COLORS.previous_token, CLASS_COLORS.positional, CLASS_COLORS.other],
          legend: true,
          tickFormat: (d: string) => CLASS_LABELS[d] ?? d,
        },
        marks: [
          Plot.cell(heads, {
            x: "head",
            y: "layer",
            fill: "classification",
            rx: 3,
            tip: true,
            title: (d: HeadData) =>
              `L${d.layer}H${d.head} — ${CLASS_LABELS[d.classification] ?? d.classification}\n` +
              `Prev-token: ${d.prev_token_score.toFixed(4)}\n` +
              `Induction: ${d.induction_score.toFixed(4)}\n` +
              `BOS: ${d.bos_score.toFixed(4)}\n` +
              `Entropy: ${d.entropy.toFixed(2)} nats`,
          }),
          Plot.text(heads, {
            x: "head",
            y: "layer",
            text: (d: HeadData) => CLASS_LABELS[d.classification]?.[0] ?? "?",
            fill: "white",
            fontSize: 11,
            fontWeight: "600",
          }),
        ],
      }),
    [heads]
  );

  // Induction score intensity heatmap
  const indRef = usePlot(
    (width) =>
      Plot.plot({
        width: Math.min(width, 500),
        height: 480,
        style: { background: "transparent", color: "#71717a", fontSize: "12px" },
        marginLeft: 50,
        marginBottom: 40,
        padding: 0.08,
        x: {
          label: "Head",
          domain: Array.from({ length: 8 }, (_, i) => i),
          tickFormat: (d: number) => `H${d}`,
        },
        y: {
          label: "Layer",
          domain: Array.from({ length: 11 }, (_, i) => i),
          tickFormat: (d: number) => `L${d}`,
        },
        color: {
          type: "linear",
          scheme: "Blues",
          label: "Induction score",
          legend: true,
        },
        marks: [
          Plot.cell(heads, {
            x: "head",
            y: "layer",
            fill: "induction_score",
            rx: 3,
            tip: true,
            title: (d: HeadData) =>
              `L${d.layer}H${d.head}\nInduction: ${d.induction_score.toFixed(4)}\n` +
              `Prev-token: ${d.prev_token_score.toFixed(4)}`,
          }),
          Plot.text(
            heads.filter((d) => d.induction_score >= 0.01),
            {
              x: "head",
              y: "layer",
              text: (d: HeadData) => d.induction_score.toFixed(3),
              fill: (d: HeadData) => (d.induction_score > 0.05 ? "white" : "#f4f4f5"),
              fontSize: 9,
              fontWeight: "600",
            }
          ),
        ],
      }),
    [heads]
  );

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-6">
          <h3 className="text-lg font-medium mb-1">Head Classification</h3>
          <p className="text-sm text-[var(--muted)] mb-4">
            {summary.induction_count} induction, {summary.prev_token_count} previous-token,{" "}
            {summary.positional_count} positional, {summary.other_count} other
          </p>
          <div ref={classRef} className="flex justify-center" />
        </div>

        <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-6">
          <h3 className="text-lg font-medium mb-1">Induction Score Intensity</h3>
          <p className="text-sm text-[var(--muted)] mb-4">
            Higher values indicate stronger A B … A → B copying behavior
          </p>
          <div ref={indRef} className="flex justify-center" />
        </div>
      </div>
    </div>
  );
}
