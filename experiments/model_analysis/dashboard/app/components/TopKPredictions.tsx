"use client";

import { useState, useMemo } from "react";

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
}

export default function TopKPredictions({ sequences }: Props) {
  const [seqIdx, setSeqIdx] = useState(0);
  if (sequences.length === 0) return null;
  const seq = sequences[seqIdx];

  // Collect high-loss tokens with their position, sorted by loss descending
  const highLoss = useMemo(() => {
    const items: { pos: number; tok: Token }[] = [];
    for (let i = 0; i < seq.tokens.length; i++) {
      if (seq.tokens[i].top5) {
        items.push({ pos: i, tok: seq.tokens[i] });
      }
    }
    items.sort((a, b) => (b.tok.l ?? 0) - (a.tok.l ?? 0));
    return items;
  }, [seq]);

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-medium mb-1">Top-k Predictions at High-Loss Tokens</h3>
          <p className="text-sm text-[var(--muted)]">
            {highLoss.length} tokens above the 90th-percentile loss threshold.
            Sorted by loss (highest first).
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

      <div className="space-y-3 overflow-auto max-h-[700px]">
        {highLoss.map(({ pos, tok }) => {
          const loss = tok.l!;
          const actualProb = Math.exp(-loss);
          const contextStart = Math.max(0, pos - 5);
          const contextEnd = Math.min(seq.tokens.length, pos + 6);
          const contextTokens = seq.tokens.slice(contextStart, contextEnd);
          const highlightIdx = pos - contextStart;

          // Check if actual token appears in top-5
          const actualInTop5 = tok.top5!.findIndex((p) => p.id === tok.id);

          return (
            <div
              key={pos}
              className="rounded-lg border border-[var(--border)] bg-[var(--background)] p-4"
            >
              {/* Context line */}
              <div className="font-mono text-sm mb-3" style={{ whiteSpace: "pre-wrap" }}>
                <span className="text-[var(--muted)]">...</span>
                {contextTokens.map((ct, ci) => (
                  <span
                    key={ci}
                    className={ci === highlightIdx ? "bg-red-500/30 rounded px-0.5" : ""}
                  >
                    {ct.t}
                  </span>
                ))}
                <span className="text-[var(--muted)]">...</span>
              </div>

              {/* Loss info */}
              <div className="flex items-center gap-4 mb-2 text-xs text-[var(--muted)]">
                <span>
                  Position {pos} &middot; Loss: {loss.toFixed(3)} nats (
                  {(loss / Math.LN2).toFixed(3)} bits) &middot; P(correct):{" "}
                  {(actualProb * 100).toFixed(2)}%
                </span>
              </div>

              {/* Predictions table */}
              <div className="grid grid-cols-[auto_1fr_auto_auto] gap-x-4 gap-y-1 text-sm font-mono">
                <span className="text-[var(--muted)] text-xs">Rank</span>
                <span className="text-[var(--muted)] text-xs">Predicted</span>
                <span className="text-[var(--muted)] text-xs">Prob</span>
                <span className="text-[var(--muted)] text-xs">Bar</span>
                {tok.top5!.map((pred, rank) => {
                  const isActual = pred.id === tok.id;
                  return (
                    <div key={rank} className="contents">
                      <span className="text-[var(--muted)]">{rank + 1}.</span>
                      <span className={isActual ? "text-[var(--accent-green)]" : ""}>
                        &quot;{pred.t}&quot;
                        {isActual && " \u2190 actual"}
                      </span>
                      <span className="text-right">{(pred.p * 100).toFixed(1)}%</span>
                      <div className="flex items-center">
                        <div
                          className={`h-2 rounded ${isActual ? "bg-[var(--accent-green)]" : "bg-[var(--accent-blue)]"}`}
                          style={{ width: `${Math.max(2, pred.p * 200)}px` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Actual token if not in top-5 */}
              {actualInTop5 === -1 && (
                <div className="mt-2 text-xs text-[var(--accent-red)]">
                  Actual: &quot;{tok.t}&quot; (rank {">"}5, prob {(actualProb * 100).toFixed(2)}%)
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
