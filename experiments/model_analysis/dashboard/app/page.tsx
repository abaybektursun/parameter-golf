import Section from "./components/Section";
import StatsCard from "./components/StatsCard";
import LogitLensChart from "./components/LogitLensChart";
import ConditionHeatmap from "./components/ConditionHeatmap";
import SVCurveViewer from "./components/SVCurveViewer";
import CrossScatter from "./components/CrossScatter";
import QuantSensitivityHeatmap from "./components/QuantSensitivityHeatmap";
import CalibrationChart from "./components/CalibrationChart";
import AttentionHeadMap from "./components/AttentionHeadMap";
import LossHeatmap from "./components/LossHeatmap";
import TopKPredictions from "./components/TopKPredictions";
import SideBySideGeneration from "./components/SideBySideGeneration";
import InsightCallout from "./components/InsightCallout";

import trainingLog from "../public/data/training_log.json";
import logitLens from "../public/data/logit_lens.json";
import svdSummary from "../public/data/svd_summary.json";
import svCurves from "../public/data/sv_curves.json";
import quantSensitivity from "../public/data/quant_sensitivity.json";
import calibration from "../public/data/confidence_calibration.json";
import attentionHeads from "../public/data/attention_heads.json";
import tokenInterp from "../public/data/token_interpretability.json";

export default function DashboardPage() {
  const wallMinutes = (
    (trainingLog.metadata.total_steps * trainingLog.metadata.step_avg_ms) /
    1000 /
    60
  ).toFixed(0);

  return (
    <main className="min-h-screen">
      {/* Hero — thesis first */}
      <div className="max-w-4xl mx-auto px-4 pt-16 pb-6">
        <p className="text-sm text-[var(--muted)] mb-2">
          PR #1019 — AR Self-Gen GPTQ + XSA-all + BigramHash 3072
        </p>
        <h1 className="text-3xl font-semibold mb-3">
          Model Analysis
        </h1>
        <p className="text-base text-[var(--muted)] max-w-2xl">
          We trained a 27M-parameter GPT to 1.134 BPB on FineWeb, then dissected
          it. Three findings: MLP matrices need more quantization bits than
          attention — despite Q and Out having 3,000× higher condition numbers. A
          single layer (L7) contributes −4.35 bits/token of readability — more
          than any other single layer after L0. And calibration is nearly perfect at 0.24% ECE, so
          temperature scaling is not worth pursuing.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-8">
          <StatsCard label="Final BPB" value="1.134" />
          <StatsCard label="Parameters" value="27.1" unit="M" />
          <StatsCard label="Steps" value="7,000" />
          <StatsCard label="Wall Time" value={wallMinutes} unit="min" />
          <StatsCard label="Hardware" value="2×H100" unit="SXM" />
        </div>
      </div>

      {/* MLP Needs More Bits — the punchline */}
      <Section
        id="quant-sensitivity"
        title="MLP Needs More Bits Than Attention"
        subtitle="Each matrix quantized individually to int6, all others held at full precision"
      >
        <p className="text-[var(--muted)] text-[15px] leading-relaxed mb-8">
          Quantization replaces precise weights with rounded approximations to
          make models smaller and faster. To find which weights matter most, we
          quantize each matrix individually to int6 while keeping everything else
          at full precision — like loosening one bolt at a time to find which
          ones hold the structure together.
        </p>
        <QuantSensitivityHeatmap
          perMatrix={quantSensitivity.per_matrix}
          perLayer={quantSensitivity.per_layer}
          baselineBpb={quantSensitivity.baseline_bpb}
          fullModelDelta={quantSensitivity.full_model_int6_delta}
        />
        <InsightCallout>
          MLP accounts for 6,247 × 10⁻⁶ total sensitivity. All four attention
          matrices (Q, K, V, Out) together: 1,600 × 10⁻⁶. For mixed-precision
          GPTQ: allocate more bits to MLP. Layer 10 is most sensitive per-layer
          (+1,057 × 10⁻⁶).
        </InsightCallout>
      </Section>

      {/* Why Condition Number Is Misleading — the explanation */}
      <Section
        id="svd"
        title="Why Condition Number Is Misleading"
        subtitle="Q matrices have condition numbers up to 54,000 — yet MLP is 4× more sensitive than all attention combined"
      >
        <p className="text-[var(--muted)] text-[15px] leading-relaxed mb-8">
          Every weight matrix can be decomposed (via SVD) into independent
          channels ranked by importance. Condition number is the ratio between
          the strongest and weakest channels — Q&apos;s ratio of 54,000× means
          it amplifies some directions 54,000× more than others, which should
          make rounding errors catastrophic. It doesn&apos;t. The singular value
          curves below show why: Q concentrates its work in just 10% of its
          channels (its &ldquo;stable rank&rdquo;). The other 90% carry near-zero
          signal, so quantization errors there do nothing. MLP uses 33% of its
          channels — 3× more capacity in active use, so rounding errors have
          3× more ways to cause damage.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <ConditionHeatmap matrices={svdSummary.matrices} />
          <SVCurveViewer curves={svCurves.curves} />
        </div>
        <div className="mt-8">
          <CrossScatter
            probePoints={logitLens.probe_points}
            matrices={svdSummary.matrices}
          />
        </div>
        <InsightCallout>
          Condition number alone is a poor predictor of quantization damage.
          Effective utilization (stable rank / full rank) combined with
          dimensionality is what matters. This is directly measured, not
          theoretical.
        </InsightCallout>
      </Section>

      {/* Layer 7 Does Most of the Work */}
      <Section
        id="logit-lens"
        title="Layer 7 Does Most of the Work"
        subtitle="Projecting each layer's residual stream through the unembedding matrix"
      >
        <p className="text-[var(--muted)] text-[15px] leading-relaxed mb-8">
          The logit lens probes what the model &ldquo;believes&rdquo; at each
          layer by projecting the hidden state through the final prediction
          head. If loss drops after a layer, that layer moved the model closer
          to the right answer. If loss rises, the layer sacrificed intermediate
          readability — reorganizing representations for downstream layers to
          use via skip connections.
        </p>
        <LogitLensChart
          probePoints={logitLens.probe_points}
          numEncoderLayers={logitLens.architecture.num_encoder_layers}
        />
        <InsightCallout>
          Layer 7 contributes −4.35 bits/token — more than any other single
          layer after L0&apos;s initial embedding projection (−15.80).
          Layers 3–5 show increased loss (residual stream becomes less readable).
          Hypothesis: encoder layers 3–4 are reorganizing representations for
          the skip connections (enc 3→dec 6, enc 4→dec 5), and decoder layer 5
          pays the cost of integrating the enc 4 skip, sacrificing
          intermediate readability. This is consistent with the architecture but
          not proven causally. Layer 10 adds only −0.47 bits/token — a candidate
          for narrowing, pending empirical validation.
        </InsightCallout>
      </Section>

      {/* Calibration Is Not the Bottleneck */}
      <Section
        id="calibration"
        title="Calibration Is Not the Bottleneck"
        subtitle={`ECE = ${(calibration.ece * 100).toFixed(2)}% across 62M tokens`}
      >
        <p className="text-[var(--muted)] text-[15px] leading-relaxed mb-8">
          Calibration measures whether a model&apos;s confidence matches reality —
          when it says 80% confident, is it right 80% of the time? Expected
          Calibration Error (ECE) quantifies this gap across all confidence
          levels. A model with low ECE but high loss means the model knows what
          it doesn&apos;t know — the bottleneck is prediction accuracy, not
          misplaced confidence.
        </p>
        <CalibrationChart
          calibrationBins={calibration.calibration_bins}
          pcorrectBins={calibration.pcorrect_bins}
          ece={calibration.ece}
          overconfidentPctTokens={calibration.overconfident_pct_tokens}
          overconfidentPctLoss={calibration.overconfident_pct_loss}
        />
        <InsightCallout>
          Do not pursue calibration tuning. 70% of total loss comes from tokens
          where P(correct) {"<"} 5% — the model is well-calibrated but often
          wrong. The bottleneck is accuracy, not confidence. Temperature scaling
          and label smoothing would yield negligible gains.
        </InsightCallout>
      </Section>

      {/* Further Exploration divider */}
      <div className="max-w-4xl mx-auto px-4 pt-16 pb-2">
        <p className="text-sm text-[var(--muted)] uppercase tracking-wide">
          Further Exploration
        </p>
      </div>

      {/* Attention Heads */}
      <Section
        id="attention-heads"
        title="What Each Head Learns"
        subtitle={`Classifying ${attentionHeads.heads.length} attention heads by function (Olsson et al. 2022)`}
      >
        <p className="text-[var(--muted)] text-[15px] leading-relaxed mb-8">
          Each attention head learns a different pattern for routing information
          between tokens. By measuring what each head actually attends to — the
          previous token, repeated sequences (induction), or absolute
          positions — we classify heads by function. This reveals whether the
          model is doing sophisticated in-context learning or relying on simpler
          n-gram statistics.
        </p>
        <AttentionHeadMap
          heads={attentionHeads.heads}
          summary={attentionHeads.summary}
        />
        <InsightCallout>
          22 of 88 heads are previous-token heads, concentrated in the encoder
          (14 vs 8 decoder). Only 2 show induction behavior (L0H5, L3H2) with
          marginal scores (~0.02). At this scale, the model relies on n-gram
          statistics, not in-context copying — induction-head-based
          interventions are not worth pursuing.
        </InsightCallout>
      </Section>

      {/* Token-Level Interpretability */}
      {tokenInterp.sequences.length > 0 && (
        <Section
          id="token-interpretability"
          title="Reading the Model&apos;s Mind"
          subtitle="Token-level loss, top-k predictions at failure points, and generation vs. reality"
        >
          <p className="text-[var(--muted)] text-[15px] leading-relaxed mb-8">
            The most direct form of interpretability: look at individual tokens,
            see where the model fails, examine what it predicted instead, and
            compare its free-form generation against reality. This turns
            aggregate statistics into concrete examples you can read and reason
            about.
          </p>
          <div className="space-y-6">
            <LossHeatmap
              sequences={tokenInterp.sequences}
              lossThreshold={tokenInterp.metadata.loss_threshold_nats}
            />
            <TopKPredictions sequences={tokenInterp.sequences} />
            <SideBySideGeneration
              sequences={tokenInterp.sequences}
              promptLen={tokenInterp.metadata.prompt_len}
              genLen={tokenInterp.metadata.gen_len}
              temperature={tokenInterp.metadata.temperature}
            />
          </div>
        </Section>
      )}

      {/* Next Steps */}
      <Section
        id="next-steps"
        title="Next Steps"
        subtitle="How to turn these findings into experiments"
      >
        <div className="space-y-8">
          <div>
            <h3 className="text-base font-semibold mb-2">
              1. Mixed-precision GPTQ guided by sensitivity data
            </h3>
            <p className="text-[var(--muted)] text-[15px] leading-relaxed">
              The per-matrix sensitivity map gives a direct bit-budget recipe.
              MLP matrices (6,247 × 10⁻⁶ total) need 7–8 bits; attention
              matrices (1,600 × 10⁻⁶ total) can survive at 4–5 bits. This
              should recover most of the 0.0083 BPB full-model int6 penalty
              while keeping the model smaller than uniform high-precision.
            </p>
          </div>
          <div>
            <h3 className="text-base font-semibold mb-2">
              2. Stable-rank-guided bit allocation
            </h3>
            <p className="text-[var(--muted)] text-[15px] leading-relaxed">
              Generalize beyond hand-tuned bit budgets: for each matrix, set
              bits proportional to effective utilization (stable rank / full
              rank) × dimensionality. Matrices with more active channels need
              finer precision. This can be computed from a single SVD pass
              before GPTQ calibration begins.
            </p>
          </div>
          <div>
            <h3 className="text-base font-semibold mb-2">
              3. Layer 10 narrowing experiment
            </h3>
            <p className="text-[var(--muted)] text-[15px] leading-relaxed">
              Layer 10 is the most quantization-sensitive layer (+1,057 × 10⁻⁶)
              but contributes only −0.47 bits/token via logit lens — the least
              of any layer. Experiment: reduce L10&apos;s hidden dimension and
              measure whether the freed parameter budget can be reallocated to
              more productive layers. This requires retraining, not just
              post-hoc pruning.
            </p>
          </div>
          <div>
            <h3 className="text-base font-semibold mb-2">
              4. Skip connection value audit
            </h3>
            <p className="text-[var(--muted)] text-[15px] leading-relaxed">
              The logit lens shows encoder layers 3–4 sacrifice readability to
              prepare representations for their skip connections (enc 3→dec 6,
              enc 4→dec 5). This cost is visible but the benefit is only
              inferred. Ablation: zero out individual skip connections during
              eval to directly measure each one&apos;s contribution to final BPB.
              If any skip adds less than the readability it costs, it&apos;s a
              candidate for removal.
            </p>
          </div>
          <div>
            <h3 className="text-base font-semibold mb-2">
              5. Scale-conditional architecture decisions
            </h3>
            <p className="text-[var(--muted)] text-[15px] leading-relaxed">
              The attention head analysis shows this 27M-parameter model has
              essentially no in-context learning — only 2 marginal induction
              heads. This means interventions that rely on in-context copying
              (retrieval augmentation, few-shot prompting strategies) are
              premature at this scale. When scaling up, monitor induction head
              emergence to know when those techniques become viable.
            </p>
          </div>
        </div>
      </Section>

      {/* Conclusion — the recipe */}
      <div className="max-w-4xl mx-auto px-4 py-16">
        <h2 className="text-2xl font-semibold mb-4">What to Do Differently</h2>
        <ol className="list-decimal list-inside space-y-3 text-[15px] leading-relaxed">
          <li>
            <span className="font-medium">Give MLP more GPTQ bits.</span>{" "}
            <span className="text-[var(--muted)]">
              MLP is 4× more sensitive than all attention matrices combined
              (6,247 vs 1,600 × 10⁻⁶), despite Q and Out having up to 3,000×
              higher condition numbers.
            </span>
          </li>
          <li>
            <span className="font-medium">Focus on word prediction.</span>{" "}
            <span className="text-[var(--muted)]">
              94% of bytes, 90% of loss. Rare token types are harder per byte
              but only 10% of total loss.
            </span>
          </li>
          <li>
            <span className="font-medium">
              Don&apos;t bother with calibration tuning.
            </span>{" "}
            <span className="text-[var(--muted)]">
              ECE = 0.24% across 62M tokens. The bottleneck is accuracy, not
              confidence.
            </span>
          </li>
          <li>
            <span className="font-medium">
              Investigate narrowing layer 10.
            </span>{" "}
            <span className="text-[var(--muted)]">
              It contributes only −0.47 bits/token via logit lens but is the
              most quantization-sensitive layer. Needs empirical validation.
            </span>
          </li>
        </ol>
      </div>

      {/* Footer */}
      <footer className="max-w-4xl mx-auto px-4 py-16 text-center text-sm text-[var(--muted)] border-t border-[var(--border)]">
        <p>
          Model: 11L GPT, 512d, XSA-all, BigramHash 3072×112, Parallel Muon
        </p>
        <p className="mt-1">
          Trained seed 314, 2×H100 SXM, 7000 steps · Post-EMA val BPB: 1.1338
        </p>
      </footer>
    </main>
  );
}
