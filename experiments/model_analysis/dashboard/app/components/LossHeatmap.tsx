"use client";

import { useState } from "react";

interface Token {
  t: string;
  id: number;
  l?: number;
  top5?: { t: string; id: number; p: number }[];
}

interface Sequence {
  idx: number;
  tokens: Token[];
}

interface Props {
  sequences: Sequence[];
  lossThreshold: number;
}

function tokenColor(loss: number, threshold: number): string {
  const ratio = Math.min(loss / (threshold * 1.3), 1);
  const hue = 120 * (1 - ratio);
  const alpha = 0.12 + 0.50 * ratio;
  return `hsla(${hue}, 60%, 35%, ${alpha})`;
}

export default function LossHeatmap({ sequences, lossThreshold }: Props) {
  const [seqIdx, setSeqIdx] = useState(0);
  if (sequences.length === 0) return null;
  const seq = sequences[seqIdx];

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-medium mb-1">Loss Heatmap</h3>
          <p className="text-sm text-[var(--muted)]">
            Green = predicted well, red = surprised. Hover tokens for loss values.
            Dotted underline = high-loss token with top-k data.
          </p>
        </div>
        <select
          value={seqIdx}
          onChange={(e) => setSeqIdx(Number(e.target.value))}
          className="bg-[var(--background)] border border-[var(--border)] rounded px-3 py-1.5 text-sm"
        >
          {sequences.map((_, i) => (
            <option key={i} value={i}>
              Sequence {i}
            </option>
          ))}
        </select>
      </div>

      {/* Color legend */}
      <div className="flex items-center gap-3 mb-4 text-xs text-[var(--muted)]">
        <span>Low loss</span>
        <div
          className="h-3 w-32 rounded"
          style={{
            background: "linear-gradient(to right, hsla(120,60%,35%,0.5), hsla(60,60%,35%,0.5), hsla(0,60%,35%,0.7))",
          }}
        />
        <span>High loss</span>
      </div>

      {/* Rendered text */}
      <div
        className="font-mono text-[13px] leading-relaxed overflow-auto max-h-[600px] rounded-lg bg-[var(--background)] p-4"
        style={{ whiteSpace: "pre-wrap", wordBreak: "break-word" }}
      >
        {seq.tokens.map((tok, i) => {
          const loss = tok.l;
          const bg = loss !== undefined ? tokenColor(loss, lossThreshold) : "transparent";
          const title =
            loss !== undefined
              ? `"${tok.t.trim()}" (id ${tok.id})\n${loss.toFixed(3)} nats / ${(loss / Math.LN2).toFixed(3)} bits\nP(correct) = ${(Math.exp(-loss) * 100).toFixed(2)}%`
              : `"${tok.t.trim()}" (id ${tok.id}) — first token, no loss`;
          return (
            <span
              key={i}
              title={title}
              style={{
                backgroundColor: bg,
                borderRadius: "2px",
                textDecoration: tok.top5 ? "underline" : "none",
                textDecorationStyle: tok.top5 ? ("dotted" as const) : undefined,
                textDecorationColor: tok.top5 ? "var(--accent-red)" : undefined,
                textUnderlineOffset: "3px",
              }}
            >
              {tok.t}
            </span>
          );
        })}
      </div>
    </div>
  );
}
