"use client";

import * as Plot from "@observablehq/plot";
import { usePlot } from "./usePlot";

interface CalibrationBin {
  bin_lo: number;
  bin_hi: number;
  count: number;
  pct_tokens: number;
  avg_confidence: number;
  accuracy: number;
  gap: number;
  avg_loss: number;
  pct_loss: number;
}

interface PCorrectBin {
  bin_lo: number;
  bin_hi: number;
  count: number;
  pct_tokens: number;
  avg_p_correct: number;
  avg_loss: number;
  pct_loss: number;
}

interface Props {
  calibrationBins: CalibrationBin[];
  pcorrectBins: PCorrectBin[];
  ece: number;
  overconfidentPctTokens: number;
  overconfidentPctLoss: number;
}

export default function CalibrationChart({
  calibrationBins,
  pcorrectBins,
  ece,
  overconfidentPctTokens,
  overconfidentPctLoss,
}: Props) {
  // Reliability diagram: plot calibration gap (confidence − accuracy)
  // Raw confidence-vs-accuracy is useless when ECE < 0.3% — the curve sits on the diagonal within 3px.
  // Instead, plot the gap directly so the overconfidence pattern is visible.
  const maxAbsGap = Math.max(...calibrationBins.map((d) => Math.abs(d.gap)));
  const gapDomain = Math.max(maxAbsGap * 1.5, 0.01);

  const reliabilityRef = usePlot(
    (width) =>
      Plot.plot({
        width: Math.min(width, 560),
        height: 420,
        style: { background: "transparent", color: "#71717a", fontSize: "12px" },
        marginLeft: 56,
        marginBottom: 48,
        x: { label: "Mean Confidence (max predicted prob)", domain: [0, 1], grid: true },
        y: { label: "Calibration Gap (accuracy − confidence)", domain: [-gapDomain, gapDomain], grid: true },
        marks: [
          // Perfect calibration = zero line
          Plot.ruleY([0], { stroke: "#d4d4d8", strokeWidth: 1, strokeDasharray: "6,4" }),
          // Gap bars
          Plot.rectY(calibrationBins, {
            x1: "bin_lo",
            x2: "bin_hi",
            y: "gap",
            fill: (d: CalibrationBin) => (d.gap < 0 ? "#991b1b" : "#1e40af"),
            fillOpacity: 0.6,
            tip: true,
            title: (d: CalibrationBin) =>
              `Confidence: ${(d.avg_confidence * 100).toFixed(1)}%\nAccuracy: ${(d.accuracy * 100).toFixed(1)}%\nGap: ${(d.gap * 100).toFixed(2)}%\nTokens: ${d.pct_tokens.toFixed(1)}%`,
          }),
          // Gap curve
          Plot.line(calibrationBins, {
            x: "avg_confidence",
            y: "gap",
            stroke: "#27272a",
            strokeWidth: 1.5,
            curve: "catmull-rom",
          }),
          Plot.dot(calibrationBins, {
            x: "avg_confidence",
            y: "gap",
            fill: (d: CalibrationBin) => (d.gap < 0 ? "#991b1b" : "#1e40af"),
            r: 4,
            stroke: "#e4e4e7",
            strokeWidth: 1.5,
          }),
          // ECE annotation
          Plot.text([`ECE = ${(ece * 100).toFixed(2)}%`], {
            frameAnchor: "top-left",
            dx: 8,
            dy: 8,
            fill: "#27272a",
            fontSize: 13,
            fontWeight: "600",
          }),
          // Region labels
          Plot.text(["Underconfident"], {
            frameAnchor: "top-right",
            dx: -8,
            dy: 8,
            fill: "#1e40af",
            fillOpacity: 0.6,
            fontSize: 11,
          }),
          Plot.text(["Overconfident"], {
            frameAnchor: "bottom-right",
            dx: -8,
            dy: -8,
            fill: "#991b1b",
            fillOpacity: 0.6,
            fontSize: 11,
          }),
        ],
      }),
    [calibrationBins, ece, gapDomain]
  );

  // Loss attribution by P(correct)
  const lossRef = usePlot(
    (width) =>
      Plot.plot({
        width: Math.min(width, 560),
        height: 420,
        style: { background: "transparent", color: "#71717a", fontSize: "12px" },
        marginLeft: 56,
        marginBottom: 48,
        x: { label: "P(correct token)", domain: [0, 1] },
        y: { label: "% of Total Loss", grid: true },
        marks: [
          Plot.rectY(pcorrectBins, {
            x1: "bin_lo",
            x2: "bin_hi",
            y: "pct_loss",
            fill: (d: PCorrectBin) => d.avg_p_correct,
            tip: true,
            title: (d: PCorrectBin) =>
              `P(correct): [${d.bin_lo.toFixed(2)}, ${d.bin_hi.toFixed(2)})\nTokens: ${d.pct_tokens.toFixed(1)}%\nLoss share: ${d.pct_loss.toFixed(1)}%\nAvg loss: ${d.avg_loss.toFixed(3)} nats`,
          }),
        ],
        color: {
          type: "linear",
          scheme: "RdYlBu",
          domain: [0, 1],
          reverse: false,
          label: "P(correct)",
          legend: true,
        },
      }),
    [pcorrectBins]
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-6">
        <h3 className="text-lg font-medium mb-1">Calibration Gap</h3>
        <p className="text-sm text-[var(--muted)] mb-4">
          Accuracy − confidence per bin. Zero = perfectly calibrated.
          {overconfidentPctTokens > 50
            ? ` Model is overconfident on ${overconfidentPctTokens.toFixed(0)}% of tokens.`
            : ` Model is underconfident on ${(100 - overconfidentPctTokens).toFixed(0)}% of tokens.`}
        </p>
        <div ref={reliabilityRef} className="flex justify-center" />
      </div>

      <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-6">
        <h3 className="text-lg font-medium mb-1">Loss by P(correct)</h3>
        <p className="text-sm text-[var(--muted)] mb-4">
          Where loss comes from. Low P(correct) = model assigned little probability
          to the right answer — these tokens dominate loss.
        </p>
        <div ref={lossRef} className="flex justify-center" />
      </div>
    </div>
  );
}
