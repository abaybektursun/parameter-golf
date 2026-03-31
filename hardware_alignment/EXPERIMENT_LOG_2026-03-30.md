# Experiment Log — 2026-03-30

Server: root@86.38.238.45, 2×H100 SXM, Python 3.12, venv at /root/pgolf-env

## EXP1: Baseline (BigramHash 3072×112, no Turbo-Muon)
- Script: `train_gpt_no_turbo.py`
- Config: BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 SEED=42 600s wallclock
- Results: 1899 steps, 316ms/step, pre-GPTQ 1.2289, sliding 1.2676, artifact 9.72 MB
- Reference: on 8×H100, this config gave 1.1138 BPB (3-seed mean)

## EXP2: EngramLite (8192 buckets, 2 heads, 2 orders, 32 dim/head) — 600s wallclock
- Script: `train_gpt_engramlite.py` with ENGRAM=1
- Config: NGRAM_BUCKETS=8192 NGRAM_HEADS=2 NGRAM_ORDERS=2 NGRAM_DIM_PER_HEAD=32
- Model params: 27.78M (vs 27.07M baseline = +710K)
- Step time: 315ms (same as baseline)
- Results (1899 steps): pre-GPTQ 1.2247 (**-0.0042 vs baseline**), roundtrip 1.2776
- Sliding window: interrupted

## EXP2B: EngramLite — 7100 steps, no wallclock cap
- Same config as EXP2 but ITERATIONS=7100 MAX_WALLCLOCK_SECONDS=999999
- Running... (~37 min training)

## EXP2B: EngramLite 8192 — 7100 steps, no wallclock
- Config: ENGRAM=1 NGRAM_BUCKETS=8192 NGRAM_HEADS=2 NGRAM_ORDERS=2 NGRAM_DIM_PER_HEAD=32 ITERATIONS=7100
- Model params: 27.78M, step time: 317ms
- Results: 7100 steps, pre-GPTQ 1.1977, post-EMA 1.1572
- **Sliding BPB: 1.1378** (better than BigramHash 3072!)
- **Artifact: 16.67 MB — OVER 16 MB limit** even after pruning 86% of ±1 values
- Conclusion: EngramLite 8192 is quality-positive but artifact too large at int6

## EXP3: EngramLite 4096 — 7100 steps
- Config: ENGRAM=1 NGRAM_BUCKETS=4096, same other params
- Model params: 27.26M, step time: 317ms
- Results: 7100 steps, pre-GPTQ val_bpb 1.2087 (val at 7100)
- **Sliding BPB: 1.1396**
- **Artifact: 16.67 MB — STILL OVER 16 MB** even after 31% ±1 pruning
- Conclusion: EngramLite doesn't fit at int6 even at 4096 buckets. The embedding table compresses poorly.

## EXP4: MLP 3.5x (1792 hidden) — 7100 steps
- Config: MLP_MULT=3.5, BigramHash 3072×112 (no EngramLite)
- Model params: 29.95M (vs 27.07M baseline = +2.88M)
- Step time: 329ms (vs 316ms baseline — 13ms slower from bigger GEMMs)
- Results: 7100 steps, pre-GPTQ val_bpb 1.1861
- **Sliding BPB: 1.1330** (best quality so far!)
- **Artifact: 17.36 MB — OVER by 1.36 MB** even after full ±1 prune
- Conclusion: MLP 3.5x is quality-positive but needs int5 for some layers

## Summary So Far (7100 steps, 2×H100)

| Experiment | Sliding BPB | Artifact | Fits 16MB? |
|---|---|---|---|
| EXP1: Baseline (BigramHash 3072) | ~1.268* | 9.72 MB | YES (6.3MB headroom) |
| EXP2B: EngramLite 8192 | 1.1378 | 16.67 MB | NO |
| EXP3: EngramLite 4096 | 1.1396 | 16.67 MB | NO |
| **EXP4: MLP 3.5x** | **1.1330** | **17.36 MB** | **NO** |

*EXP1 baseline was 600s wallclock (1899 steps), not 7100 steps

Key insight: ALL quality-improving changes push us over the 16MB limit at uniform int6.
Mixed int5/int6 quantization is the critical enabler.

## EXP5: MLP 3.5x + mixed int5/int6 — 7100 steps (LOST: server rebooted)
- Config: MLP_MULT=3.5 MIXED_QUANT=1 N_INT6_LAYERS=10
- Training completed 7100 steps: pre-GPTQ 1.1913, post-EMA 1.1510
- GPTQ was running with mixed quant when server SSH keys changed — results lost
- Server at 86.38.238.45 is now inaccessible (Permission denied)

## Key Findings

1. **EngramLite 8192 and 4096**: Quality-positive (best pre-GPTQ BPB) but artifacts DON'T FIT at uniform int6. EngramLite embeddings compress poorly.

2. **MLP 3.5x**: Best quality model (1.1330 sliding BPB at 7100 steps) but artifact 17.36 MB — 1.36 MB over. Needs mixed quant.

3. **Mixed int5/int6**: Critical enabler for fitting larger models. EXP5 was testing this when server was lost.

4. **Our current best (no turbo, no engramlite, standard MLP 3x)**: 1.1138 BPB on 8×H100 (3-seed mean), all artifacts under 16 MB. This is still our submission.

## New Server: ssh -p 29954 root@ssh2.vast.ai (Massachusetts 2×H100 SXM, $2.14/hr)

## EXP6: Baseline 7100 steps (new server)
- Same config as our 8×H100 submission but 7100 steps on 2×H100
- Results: sliding BPB 1.1361, artifact 16.67 MB (OVER on 2-GPU but 15.51 MB on 8-GPU)
- Key insight: 2×H100 artifact sizes are NOT a valid proxy for 8×H100 (grad_accum=4 vs 1 produces different weight distributions)

## EXP7: MLP 3.5x + mixed int5/int6 + lr_floor=0.05 + warmdown=3500 ✅
- Server: ssh -p 29954 root@ssh2.vast.ai (Massachusetts 2×H100)
- Config: MLP_MULT=3.5 MIXED_QUANT=1 N_INT6_LAYERS=10 LR_FLOOR=0.05 WARMDOWN_ITERS=3500
- Model params: 29.95M, step time: 333ms
- Mixed quant: 10 int6 (MLP proj early layers), 56 int5
- Results: 7100 steps, pre-GPTQ 1.1880, post-EMA 1.1507
- **Sliding BPB: 1.1301** (-0.0060 vs EXP6 baseline)
- **Artifact: 15.94 MB — FITS under 16 MB!** No pruning needed.
- Key win: mixed int5/int6 is the enabler for fitting MLP 3.5x under 16 MB

## EXP9: EngramLite 8192 + MLP 3.5x + mixed int5/int6 + lr_floor=0.05 (Agent 1) — LOST
- Server: ssh -p 29954 root@ssh2.vast.ai (Massachusetts 2×H100 SXM, CUDA 13.0)
- Config: ENGRAM=1 NGRAM_BUCKETS=8192 MLP_MULT=3.5 MIXED_QUANT=1 N_INT6_LAYERS=10 LR_FLOOR=0.05 WARMDOWN_ITERS=3500
- Model params: 30.66M, step time: 334ms
- Partial: reached step ~3500/7100 before instance SSH became unreachable
- Instance reclaimed/broken. Need to re-run on new machine.

## EXP8: MLP 3.5x baseline, uniform int6 (Agent 2) ✅
- Server: ssh -p 23100 root@146.115.17.181 (4×H100, CUDA 12.8)
- Config: MLP_MULT=3.5 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 SEED=42 ITERATIONS=7100
- Model params: 29.95M, step time: 170ms (4×H100 grad_accum=2)
- Results: 7100 steps, pre-GPTQ 1.1906, post-EMA 1.1506
- **Sliding BPB: 1.1316** (consistent with EXP4 on 2×H100: 1.1330)
- **Artifact: 17.33 MB — OVER 16 MB** (same as EXP4, confirms MLP 3.5x needs mixed quant)

## EXP10: ALL WINS STACKED on 4×H100 (Agent 2) ✅
- Server: ssh -p 23100 root@146.115.17.181 (4×H100, CUDA 12.8)
- Config: MLP_MULT=3.5 MIXED_QUANT=1 N_INT6_LAYERS=10 LR_FLOOR=0.05 WARMDOWN_ITERS=3500 SEED=42 ITERATIONS=7100
- Stack: Fused Triton MLP + CUTLASS EVT + Brotli + memmap + MLP 3.5x + mixed int5/int6 + lr_floor
- Model params: 29.95M, step time: 168ms (4×H100 grad_accum=2)
- Mixed quant: 10 int6 (MLP proj layers 0-4...), 56 int5
- Results: 7100 steps, pre-GPTQ 1.1872, post-EMA 1.1505
- **Sliding BPB: 1.1298**
- **Artifact: 15.93 MB — FITS under 16 MB!** No pruning needed.
- Validates EXP7 result on 4×H100. Ready for 8×H100 3-seed submission.
