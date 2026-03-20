# autoresearch for parameter-golf

Autonomous experiment loop for the OpenAI Parameter Golf challenge.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar19`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The key files for full context:
   - `README.md` — challenge rules (16MB artifact, 10 min on 8xH100s, lowest val_bpb wins)
   - `train_gpt.py` — the original baseline (read-only reference)
   - `experiments/002_autoresearch/train_gpt.py` — **your working copy**. This is the file you modify.
   - `experiments/002_autoresearch/notes.md` — this experiment's log
4. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains train/val shards and `./data/tokenizers/` has the tokenizer. If not, tell the human to run: `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10`
5. **Initialize results.tsv**: Create `experiments/002_autoresearch/results.tsv` with the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Challenge constraints

- **Artifact size**: The final `final_model.int8.ptz` + code must fit in **16MB** total.
- **Compute budget**: Training must complete within **10 minutes** on **8xH100s** (MAX_WALLCLOCK_SECONDS=600).
- **Metric**: Post-quantization `val_bpb` — lower is better. This is the ONLY metric that matters for the leaderboard.
- **Tokenizer**: You may change the tokenizer/vocab size, but the BPB metric is tokenizer-agnostic.
- **No external dependencies**: Only what's already in `requirements.txt`.

## Experimentation

First, detect the GPU count on this machine:
```bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
```

Each experiment runs via torchrun. The training script runs for a **fixed time budget of 10 minutes** wall clock. You launch it as:

```bash
torchrun --standalone --nproc_per_node=$NUM_GPUS experiments/002_autoresearch/train_gpt.py
```

For faster iteration (smoke tests before full runs), you can do a short run:

```bash
ITERATIONS=200 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=60 torchrun --standalone --nproc_per_node=$NUM_GPUS experiments/002_autoresearch/train_gpt.py
```

**IMPORTANT — Hardware migration (2026-03-20)**:
All previous experiments in `results.tsv` (up to ~98 runs) were conducted on a **single A100 GPU**. You are now running on **8xH100 GPUs**, which matches the actual leaderboard hardware. Key implications:
- On 1xA100: `grad_accum_steps=8`, ~378ms/step, ~1,589 steps in 600s.
- On 8xH100: `grad_accum_steps=1` (parallel across GPUs), ~50-60ms/step, ~10,000-12,000 steps in 600s.
- **The model trains 6-8x more steps now.** This means the absolute BPB numbers will be much better than what's in the historical results.
- The best 1xA100 result was post-quant BPB 1.3480. The same config achieved **1.2947** on 1xH100 in a test run. On 8xH100 it should be significantly lower.
- **Hyperparameter tuning from A100 runs is directionally useful** (relative rankings mostly hold), but absolute values like `WARMDOWN_ITERS=220` were tuned for ~1,100 steps and might need retuning for ~10,000+ steps.
- The SOTA target is **1.206 val_bpb**. This is now reachable.

**Note**: The leaderboard target is 8xH100s. You now have the full leaderboard hardware.

**What you CAN do:**
- Modify `experiments/002_autoresearch/train_gpt.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, quantization strategy, etc.

**What you CANNOT do:**
- Install new packages or add dependencies beyond `requirements.txt`.
- Modify the evaluation harness (`eval_val` function). It is the ground truth metric.
- Exceed 16MB for the final artifact (model + code).
- Exceed 10 minutes wall clock on 8xH100s.

**The goal: get the lowest post-quant val_bpb under the 16MB artifact limit.**

Everything is fair game: architecture, optimizer, hyperparameters, vocab size, quantization strategy, model size, sequence length. The only constraints are the artifact size and compute budget.

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. Weigh complexity cost against improvement magnitude.

**The first run**: Always establish the baseline first — run the training script as-is.

## Output format

The script logs val_bpb at validation steps and at the end. The key lines to look for:

```
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX eval_time:XXXms
```

And the artifact size:
```
Total submission size int8+zlib: XXXXXXX bytes
```

Extract the key metrics:
```bash
grep "final_int8_zlib_roundtrip_exact" logs/*.txt
grep "Total submission size int8+zlib" logs/*.txt
```

## Logging results

When an experiment is done, log it to `experiments/002_autoresearch/results.tsv` (tab-separated).

The TSV has a header row and 6 columns:

```
commit	val_bpb	post_quant_bpb	artifact_bytes	status	description
```

1. git commit hash (short, 7 chars)
2. pre-quant val_bpb (e.g. 1.2244) — use 0.0000 for crashes
3. post-quant val_bpb (e.g. 1.2300) — use 0.0000 for crashes
4. artifact size in bytes (final_model.int8.ptz + code) — use 0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_bpb	post_quant_bpb	artifact_bytes	status	description
a1b2c3d	1.2244	1.2300	15200000	keep	baseline
b2c3d4e	1.2100	1.2150	15300000	keep	increase matrix_lr to 0.05
c3d4e5f	1.2400	1.2500	15100000	discard	switch to GeLU activation
d4e5f6g	0.0000	0.0000	0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar19`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. **Read `experiments/002_autoresearch/results.tsv`** to see ALL previous experiments and their outcomes.
3. **Read `experiments/002_autoresearch/learnings.md`** if it exists — this contains accumulated insights from all experiments so far.
4. Analyze patterns: what directions are working (keep pushing), what failed (avoid repeating), what's unexplored.
5. Think about what to try next. Consider:
   - Architecture changes (depth, width, attention heads, MLP ratio, skip connections)
   - Optimizer changes (learning rates, momentum, warmup/warmdown schedules)
   - Quantization improvements (better int8 strategy, fp16 for certain tensors)
   - Sequence length (longer context = more info per token but fewer steps)
   - Vocab size changes
   - Any creative ideas that might compress better or train more efficiently
6. Edit `experiments/002_autoresearch/train_gpt.py` with the experimental idea.
4. git commit the change.
5. Run the experiment: `torchrun --standalone --nproc_per_node=$NUM_GPUS experiments/002_autoresearch/train_gpt.py > run.log 2>&1`
6. Extract results: `grep "final_int8_zlib_roundtrip_exact\|Total submission size int8+zlib" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
8. Record the results in the TSV (do not commit results.tsv, leave it untracked by git)
9. **Check artifact size**: If `final_model.int8.ptz` + code > 16MB (16,777,216 bytes), treat as a failure — discard.
10. If post-quant val_bpb improved AND artifact is under 16MB, keep the commit and advance the branch.
11. **Push + backup**: `git push origin HEAD` and `aws s3 sync . s3://fuelos-autoresearch/latest/ --profile fuelos --exclude 'data/*' --exclude '.git/*' --exclude '__pycache__/*' --exclude '.venv/*'`
12. If val_bpb is equal or worse, `git reset --hard` back to where you started.
13. **Update learnings**: After every experiment (keep, discard, or crash), append insights to `experiments/002_autoresearch/learnings.md`. Record: what you tried, why, what happened, and what it tells you about future directions. Keep it concise — bullet points, not essays.

**Timeout**: Each experiment should take ~10 minutes for training + overhead. If a run exceeds 15 minutes, kill it and treat as a failure.

**Crashes**: If it's a simple bug (typo, import), fix and re-run. If the idea is fundamentally broken, log "crash", revert, and move on.

**NEVER STOP**: Once the experiment loop begins, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the code, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.

## Ideas to explore (starting points)

- Tune learning rates (embed_lr, matrix_lr, scalar_lr, tied_embed_lr)
- Increase sequence length (TRAIN_SEQ_LEN=2048 showed gains on the leaderboard)
- Adjust warmdown schedule (WARMDOWN_ITERS)
- Try different model dimensions/depths within 16MB budget
- Improve quantization (fp16 embeddings, better clipping)
- Adjust MLP multiplier
- Try different vocab sizes
- Adjust number of KV heads
- Logit softcap tuning
- RoPE base frequency tuning
- Batch size / gradient accumulation tweaks

## Advanced ideas (use when confident)

Beyond hyperparameter tuning, there are researched optimization ideas in this repo that you can implement if you're confident they will yield improvements. Read these files for full details:

- **`experiments/optimization_opportunities.md`** — ranked list of optimizations with expected impact:
  - Tier 1 (easy, high impact): `mode="max-autotune"` in torch.compile (one-line change), fused cross-entropy + logit softcap, fused ReLU² MLP
  - Tier 2 (medium): batch size warmup schedule, Triton autotune config, Lookahead/SNOO outer optimizer + LAWA
  - Tier 3+: profiling, seq4096, mixed FP8

- **`experiments/001_combine_three_wins/`** — combines seq2048 + FP16 embed quantization + sliding window eval (the top 3 leaderboard tricks, all orthogonal). Has a ready `train_gpt.py`.

- **`experiments/002_lookahead_snoo_lawa/`** — Lookahead/SNOO outer optimizer + LAWA checkpoint averaging for flatter minima that survive int8 quantization better. Has a ready `train_gpt.py`.

**When to use these:** If hyperparameter sweeps are hitting diminishing returns, try these bigger architectural/training changes. You can copy working code from the experiment `train_gpt.py` files into your working copy. Test one change at a time so you know what helped.
