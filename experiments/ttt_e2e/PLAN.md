# TTT-E2E for Parameter Golf

Adapt End-to-End Test-Time Training (arxiv 2512.23675) to tiny language models
under Parameter Golf constraints (16 MB artifact, 10 min train, 10 min eval, 8xH100).

## Motivation

SLOT showed -0.0037 BPB of test-time adaptation headroom (1.1125 -> 1.1088) but
violates causality (Condition 3). TTT-E2E is the legal mechanism to capture this
headroom via score-first adaptation.

25 prior naive-TTT attempts failed because:
- Full MLP weight updates after GPTQ disrupt quantized rounding decisions
- Model was not meta-learned for test-time adaptation
- Loss landscape is not shaped for gradient-based adaptation at test time

The paper's key finding: "the weights of TTT-E2E only need to be good at the
present mini-batch of tokens, since TTT will produce future weights for the future
tokens." A tiny model can't be a great generalist — but it CAN specialize per
mini-batch, with meta-learning teaching it how to specialize.

## Architecture: Sequential Prime MLP (from paper's actual code)

The paper's code (github.com/test-time-training/e2e) reveals the dual MLP is
SEQUENTIAL, not parallel. The prime (dynamic) MLP runs BEFORE the main (static)
MLP, each with its own RMSNorm and residual connection:

```
# Prefix blocks (layers 0-7): standard, frozen entirely
h = h + attn(norm(h))
h = h + MLP(ffn_norm(h))

# Suffix blocks (layers 8-10): with prime MLP
h = h + attn(norm(h))
h = h + prime_MLP(prime_norm(h))    # runs FIRST — adapted at test time
h = h + MLP(ffn_norm(h))            # runs SECOND — frozen at test time
```

Why sequential matters: the prime MLP steers the representation BEFORE the frozen
main MLP processes it. The main MLP's learned features then act on the adapted
representation, amplifying the adapter's effect. This is strictly more powerful
than parallel addition (h = MLP(x) + adapter(x)).

### Prime MLP Specification

- Architecture: 512 -> r -> 512 with LeakyReLU(0.5)^2 (match main MLP activation)
- Own RMSNorm (prime_norm) before it
- Own residual connection
- r=256 (tunable: 128-512)
- Initialization: down_proj = zeros (output is zero at Phase 2 start, so the
  model at Phase 2 start == model at Phase 1 end). up_proj = standard init.

The paper uses SwiGLU (3 matrices) for prime MLPs matching their main MLP. We use
LeakyReLU(0.5)^2 (2 matrices) matching ours. Fewer params per layer.

### Parameter Budget

Prime MLP params per layer: (512 * r + r * 512) = 1024r
With r=256: 262K per layer
With RMSNorm: +512 params (negligible)
3 layers total: 786K params * 2 bytes = 1.57 MB at bf16

Main model (GPTQ'd): ~14 MB
Prime MLPs (bf16): ~1.57 MB
Code + overhead: ~0.3 MB
Total: ~15.87 MB -- fits 16 MB

For tighter fit, r=192: 3 * 197K * 2 = 1.18 MB, total ~15.48 MB.
For more capacity, r=320: 3 * 328K * 2 = 1.97 MB, total ~16.27 MB -- over.

### What the Code Revealed vs the Paper

| Detail | Paper says | Code shows |
|---|---|---|
| MLP combination | "add a static, second MLP" (vague) | Sequential: prime MLP first, then main MLP, each with own residual |
| Prime MLP activation | Not specified | SwiGLU (same as their main MLP) |
| Prime initialization | Not specified | Random (standard init), NOT zero |
| Inner optimizer | "eta" learning rate | SGD with lr=0.01, no gradient clipping |
| mini_batch_size | "b=1K" | Configurable, default 1024 |
| suffix_len | "last 1/4 of blocks" | Configurable (default 0, set per experiment) |
| Prefix computation | Not explicit | Run ONCE, output cached; suffix blocks scan over chunks |
| Inner param selection | "only MLP layers" | Configurable via spec_inner pattern matching |
| Train mode | Implied | Explicit "pretrain" vs "meta" switch |

## Meta-Learning: FOMAML (not full MAML)

The paper uses full MAML (second-order). This requires grad-of-grad, which:
- FA3 does not support (confirmed: no double backward in flash/cudnn/efficient attention)
- Math backend fallback is O(N^2) memory, kills performance
- Costs 3.4x training slowdown
- Paper implemented in JAX; no PyTorch solution exists

FOMAML drops the second-order term. Both loops use standard first-order gradients.

| Property          | Full MAML (paper) | FOMAML        |
|-------------------|-------------------|---------------|
| FA3 compatible    | No                | Yes           |
| Training overhead | 3.4x              | ~2x           |
| Quality vs full   | 100%              | 95-100%       |
| Grad-of-grad      | Required          | Not needed    |
| Memory            | O(K) inner steps  | O(1)          |

### FOMAML Training Step (PyTorch)

```python
# 1. Detach prime MLP weights from init (break gradient through update)
adapted = {n: p.detach().clone().requires_grad_(True)
           for n, p in model.prime_named_params()}

# 2. Inner loop: K steps of SGD on adapted weights (no graph through updates)
for chunk in inner_chunks:
    logits = model.forward_with_primes(chunk, adapted)
    loss = ce_loss(logits, chunk.targets)
    grads = torch.autograd.grad(loss, adapted.values())
    adapted = {n: p - inner_lr * g
               for (n, p), g in zip(adapted.items(), grads)}

# 3. Outer loss: forward with adapted weights, WITH graph for base model
outer_logits = model.forward_with_primes(next_chunk, adapted)
outer_loss = ce_loss(outer_logits, next_chunk.targets)
outer_loss.backward()

# 4. FOMAML: copy adapted gradients to prime init params
for n, p in model.prime_named_params():
    p.grad = adapted[n].grad
```

Both backward passes are vanilla autograd. FA3 works throughout.
The base model (attention, main MLP, embeddings) gets correct gradients
through the forward pass. Only the prime MLP init gradient is approximated.

## Training Strategy

### Revised overhead estimate

Prefix (layers 0-7) outputs are cached during the inner loop — only suffix
(layers 8-10, 3/11 of the network) is re-processed. Real FOMAML overhead:

| Component | Relative cost |
|---|---|
| Full forward (prefix + suffix → inner loss) | 1.0× forward |
| Inner backward (suffix only → prime gradient) | ~0.27× backward |
| Suffix forward (cached prefix → outer loss) | ~0.27× forward |
| Full backward (outer gradient for all params) | 1.0× backward |
| Total | ~1.27× one standard step |

Only ~27% overhead per step, not 2×.

### Option A: FOMAML from scratch (preferred)

Train with FOMAML and prime MLPs from step 1.
- ~111 ms/step (87.7 × 1.27)
- ~5400 steps in 600s (79% of normal 6840)
- The model learns to delegate to prime MLPs throughout training
- No Phase transition artifacts

### Option B: Two-phase

Phase 1 — Standard training (no prime MLPs):
- Full speed (~87.7 ms/step), ~4500 steps in ~394s

Phase 2 — FOMAML meta-fine-tuning:
- Add prime MLPs (zero-init down_proj), switch to FOMAML
- ~111 ms/step, ~1850 steps in ~206s

Total: ~6350 effective steps. More steps but Phase transition risk.

### Option C: Reptile from scratch

Use weight displacement as meta-gradient (even simpler than FOMAML).
K=2 inner steps: ~1.54× overhead → ~4400 steps.
Less theoretically grounded but competitive empirically.

## GPTQ + Prime MLP Interaction

No interaction:
- GPTQ applied to entire main model after Phase 2 (mixed int5/int6, as before)
- Prime MLPs stored separately in bf16, never quantized
- At eval: main model frozen (GPTQ'd), prime MLPs adapted (bf16 gradients)
- Forward pass: prefix blocks use GPTQ'd weights; suffix blocks use GPTQ'd main
  MLP + bf16 prime MLP

The AR self-gen calibration for GPTQ runs with prime MLPs at their meta-learned
init (W_0). The calibration captures the model's behavior INCLUDING the prime MLP
contribution — accurate for the first eval mini-batch. As prime MLPs adapt during
eval, GPTQ'd layers see slightly different inputs, but the frozen main MLP's
features remain relevant (they were trained with prime MLPs present in Phase 2).

## Eval-Time Inner Loop

For each mini-batch of b tokens on the val shard:
1. Forward pass (eval mode, gradients enabled) -> logits
2. Score: compute BPB from logits -> record
3. Compute CE loss from same logits
4. Backward through prime MLP layers only -> gradients
5. SGD update on prime MLP weights
6. Next mini-batch

One forward + one backward per chunk (scoring and training use same forward pass).

Properties:
- Score-before-update: Condition 3 satisfied (score locked before weight update)
- Single left-to-right pass: Condition 4 satisfied
- Causal: Condition 1 satisfied (W_i built from x_1..x_{i*b} only)
- Full distribution: Condition 2 satisfied (standard softmax over full vocab)
- Track B (Adaptive Compression) per issue #1017

Compute cost:
- Prime MLP backward through 3 layers x r=256: ~0.1ms per update
- Val shard ~60M tokens / b=1024 = ~58K updates
- Total overhead: ~5.8s (negligible in 10-min eval budget)
- One forward pass is shared with scoring (no extra forward cost)

No per-sequence reset — prime MLP weights accumulate across entire val shard.
The frozen GPTQ'd main model anchors predictions, preventing catastrophic drift.

### Why the Paper's Architecture Insight Matters Here

"The aggregated advantage of TTT-E2E over full attention mostly comes from the
earlier tokens." And: "Before t=1K (first TTT gradient step), curves differ only
in weights, yet weights of TTT-E2E produce much lower losses."

The meta-learned W_0 is better than naively trained weights EVEN BEFORE any TTT
update. The meta-learning shapes W_0 to be good at the current mini-batch
(specialization) rather than average across all data (generalization). For a tiny
30M-param model, this specialization vs generalization tradeoff is enormous.

## Custom Kernels

The prime MLP is small (512x256 GEMMs). For eval-time TTT, kernel launch overhead
dominates. Unfused PyTorch is likely fast enough (~0.1ms per update, 5.8s total).

For training (Phase 2), the bottleneck is two full forward-backward passes. The
existing fused MLP kernels (Triton TMA + CUTLASS EVT) work unchanged for the main
MLP. The prime MLP forward is tiny and doesn't need fusion.

Potential optimizations if needed:
- Fuse prime MLP forward into suffix block (avoid kernel launch overhead)
- Fuse prime MLP backward + SGD update (avoid writing gradients to HBM)
- But: premature optimization. Profile first.

## Three Timescales of Memory

The model would have three memory systems at different timescales:

1. Sliding window attention (~1K-8K tokens): exact recall, sharp boundary
2. Prime MLP weights (~100K tokens): soft compressed memory, exponential decay
   (effective half-life ≈ 1/inner_lr mini-batches; tunable via weight decay)
3. Frozen main model (permanent): all patterns learned during training

The inner loop weight decay controls the prime MLP's memory horizon:
  W_i = (1 - lr*λ) * W_{i-1} - lr * ∇ℓ(W_{i-1})
  λ=0:    infinite memory, risk of drift over 60M tokens
  λ=0.01: ~10M token memory horizon
  λ=0.1:  ~1M token memory horizon

## U-Net Interaction

Our model is a U-Net (encoder layers 0-4, decoder 5-10, skip connections).
Suffix blocks (8-10) are deep decoder layers:
- Layer 8: receives skip from encoder layer 2 (raw + abstract features combined)
- Layers 9-10: have VE128 (value embeddings)

The prime MLP in layer 8 can modulate how skip connection features combine with
the deep representation — shifting the balance based on current context.

## Risks

1. FOMAML untested for LM TTT (proven for few-shot classification, not LM adaptation)
2. ~5400 steps (FOMAML from scratch) vs ~6840 standard — 21% fewer steps
3. Continuous adaptation may drift over 60M tokens (mitigate with inner weight decay)
4. First mini-batch (b=1024 tokens) gets zero adaptation benefit
5. Prime MLP r=256 may be too small for effective adaptation (or too large for budget)
6. Sequential prime MLP may amplify noise from small adapter — parallel is safer fallback
7. GPTQ calibration at W_0 may not generalize to adapted prime MLP weights
8. The "delegation" effect (base model restructuring around prime MLPs) is unproven
   at 30M scale with FOMAML — this is the core bet

## Experiments

### Experiment 0: Naive adapter TTT on existing model (no retraining)
- Take PR 1105 model, add zero-init prime MLPs (r=256) to last 3 layers
- Run score-first TTT on val shard with SGD, sweep inner_lr: 0.001, 0.003, 0.01, 0.03
- Goal: detect whether sequential prime MLP TTT has any signal without meta-learning
- Expected: small improvement or neutral (like prior naive TTT), but validates machinery
- Key difference from prior naive TTT: prime MLP is separate from GPTQ'd weights

### Experiment 1: FOMAML from scratch
- Train with FOMAML + prime MLPs from step 1 (~5400 steps)
- Eval with score-first prime MLP TTT
- Compare to PR 1105 baseline (1.1125 BPB)
- Also compare: disable TTT at eval (just use the meta-learned base model)
  to isolate the meta-learning benefit from the adaptation benefit

### Experiment 1b: Two-phase training
- Phase 1: standard ~4500 steps, Phase 2: FOMAML ~1850 steps
- Compare to Exp 1 (from scratch) to determine if Phase transition hurts

### Experiment 2: Reptile variant
- Same as Exp 1 but Reptile instead of FOMAML
- K=2 inner steps, meta-gradient = W_0 - W_K
- Cheaper (~1.3x vs ~2x overhead), more total training steps

### Experiment 3: Hyperparameter tuning
- Prime MLP rank r: 128, 192, 256, 320
- Mini-batch size b: 256, 512, 1024, 2048
- Inner loop LR: 0.003, 0.01, 0.03
- Inner loop weight decay λ: 0, 0.01, 0.1 (memory horizon control)
- Suffix layers: last 2, 3, 4, 5
- Inner loop optimizer: SGD, SGD+momentum(0.9)
- K (inner steps during training): 1, 2, 3
- Sequential vs parallel prime MLP ordering

## Prior Art in Parameter Golf (all naive TTT, no meta-learning)

- PR #713: LoRA TTT rank-8 on Q/V, 1.1180 BPB (best naive LoRA-TTT result)
- PR #1148: Muon-TTT + entropy-adaptive epochs, 1.1179 BPB
- PR #1128: SLOT + naive TTT, 1.1154 BPB (SLOT legality questionable)
- PR #658: LoRA TTT negative result — "TTT provides no additional benefit on SOTA base"
- PR #1105 (ours): SLOT gave -0.0037 BPB but violates causality, removed

Every submission uses naive TTT. Nobody has tried meta-learned TTT.
PR #658's finding — "TTT only helps models with weak local embeddings" — applies
specifically to naive TTT. Meta-learned TTT trains the model to leave room for
adaptation, fundamentally changing the dynamic.

## References

- TTT-E2E paper: arxiv.org/abs/2512.23675
- TTT-E2E code: github.com/test-time-training/e2e (JAX)
- FOMAML: arxiv.org/abs/1803.02999 (Nichol et al.)
- Parameter Golf rules: github.com/openai/parameter-golf/issues/1017
- PR 1105 (our current SOTA): github.com/openai/parameter-golf/pull/1105
- SLOT headroom: -0.0037 BPB, removed from PR 1105 for causality violation
