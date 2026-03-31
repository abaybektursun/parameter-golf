# ACTIVE TODO — Eval-Time Mixing Research

## NEXT ACTION: Deploy and run EXP-11 on 8xH100

When user provides a machine:
1. scp `scripts/eval_ngram_distributed.py` and `scripts/run_exp11.sh` to machine
2. Ensure FA3 is installed: `pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291`
3. Ensure data is downloaded: `python3 data/cached_challenge_fineweb.py --variant sp1024`
4. Ensure model checkpoint exists (`final_model.pt`)
5. Run: `bash experiments/eval_time_mixing/scripts/run_exp11.sh`
6. Download results locally immediately after completion

## Remaining experiments after EXP-11

### P0 — Core paper (partially done)
- [x] EXP-0: Baselines — DONE (7 results)
- [ ] EXP-2: Logistic vs Linear mixing — mix_logistic and mix_geometric KILLED, need to re-run
- [ ] EXP-1: Order scaling law — never ran, need order sweep K=2..20

### P1 — Advanced techniques
- [ ] EXP-3: PPM escape probabilities
- [ ] EXP-8: Suffix-based longest match
- [ ] EXP-5: Hash collision impact

### P2 — Supporting evidence
- [ ] EXP-6: Stride decomposition (partial data from killed runs)
- [ ] EXP-9: Online weight learning (PAQ-style)
- [ ] EXP-4: Full-distribution vs point-probability mixing

### P3 — Characterization
- [ ] EXP-7: Document-boundary analysis
- [ ] EXP-10: FineWeb entropy estimation

## Data we have locally
- `results/exp0/` — 6 JSON files (baseline, fixed_7gram, backoff_7, backoff_7_ent, backoff_9_ent_oadapt, ngram_only_7)
- `results/exp2/` — 1 JSON file (mix_linear)
- `results/ngram_analysis.json` — full 62M-token match analysis
- `server_logs/` — all server logs from first run
- Research agent output on compression literature

## Key findings so far
- Global cache (1 GPU) massively outperforms partitioned (8 GPU): 0.49 vs ~0.97 BPB
- Pure n-gram (no neural model) beats the neural model: 1.06 vs 1.11 BPB
- Entropy-adaptive alpha HURTS with strong cache: 0.65 vs 0.49 BPB
- N-gram matches 86-100% of tokens (orders 2-7)
- Three compression phenomena: subword completion, common phrases, verbatim repetition
