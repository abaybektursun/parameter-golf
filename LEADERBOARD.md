# Parameter Golf Leaderboard (2026-04-07)

Compiled from open PRs on [openai/parameter-golf](https://github.com/openai/parameter-golf).
Validity assessed against [PR #1017 "A Field Guide to Valid Submissions"](https://github.com/openai/parameter-golf/pull/1017).

## Validity Rules (PR #1017)

1. **Strict Causal Dependence** — predictions depend only on artifact + tokens before position t
2. **Full Normalized Distribution** — complete probability distribution over full vocabulary before scoring
3. **Score-Before-Update** — state updates only after scoring
4. **Single Left-to-Right Pass** — no rescoring, multi-pass, or retrospective revision

---

## Top Record Submissions (Open PRs, sorted by claimed BPB)

### Tier 1: Sub-1.08 BPB (Top Contenders)

| Rank | PR | BPB | Author | Technique | Validity | Notes |
|------|-----|-----|--------|-----------|----------|-------|
| 1 | [#1333](https://github.com/openai/parameter-golf/pull/1333) | 1.0766 | aryanbhosale | SP4096 + Depth Recurrence + Causal SLOT-16 | NEEDS REVIEW | SLOT usage requires scrutiny |
| 2 | [#1423](https://github.com/openai/parameter-golf/pull/1423) | 1.0791 | aryanbhosale | SP8192 + Pre-Quant TTT + QK-Gain 5.0 + Depth Recurrence | INVALID | Pre-quant TTT trains on val data 6 epochs. Rules 1,3,4 violated. Author notified. |
| 3 | [#1416](https://github.com/openai/parameter-golf/pull/1416) | 1.0795 | erichroepke | SP8192 + Pre-Quant TTT | INVALID | Same pre-quant TTT violation. Author acknowledged and withdrew. |
| 4 | [#1408](https://github.com/openai/parameter-golf/pull/1408) | 1.0800 | aamodbhatt | dTTT + BigramHash 3072x112 | INVALID | Pre-quant dTTT trains 10 epochs on val data before quantization. Same violation as #1423/#1416. Rules 1,3,4 violated. |
| 5 | [#1420](https://github.com/openai/parameter-golf/pull/1420) | 1.0801 | abaybektursun | Triple Loop + Fused Kernels + Parallel Residuals + N-gram Tilt | VALID (FIXED) | within_hint/word_hint causal bug fixed in 5e2eff8. |
| 6 | [#1437](https://github.com/openai/parameter-golf/pull/1437) | 1.0809 | dexhunter | SP8192 + Parallel Residuals + 3-Layer Recurrence + N-gram Tilt | NEEDS FIX | Same within_hint/word_hint causal bug as #1420. Shared C++ code. |
| 7 | [#1289](https://github.com/openai/parameter-golf/pull/1289) | 1.0819 | MatoTeziTanka | PROTEUS v1.6 — Scylla + Parallel Residuals | NEEDS REVIEW | |
| 8 | [#1413](https://github.com/openai/parameter-golf/pull/1413) | 1.0828 | dexhunter | SP8192 + QK-Gain 5 + Legal Score-First TTT | VALID | Score-first TTT: chunks scored under no_grad() before training. Clean. |
| 9 | [#1412](https://github.com/openai/parameter-golf/pull/1412) | 1.0835 | Robby955 | Parallel Residuals + Hessian-Aware SDClip | NEEDS REVIEW | |
| 10 | [#1450](https://github.com/openai/parameter-golf/pull/1450) | 1.0848 | andrewbaggio1 | TMA Megakernel + Triple Loop + Parallel Residuals | NEEDS REVIEW | |
| 11 | [#1257](https://github.com/openai/parameter-golf/pull/1257) | 1.0855 | BoxiYu | 11L Complement Training + TTT + No-JEPA | NEEDS REVIEW | |
| 12 | [#1424](https://github.com/openai/parameter-golf/pull/1424) | 1.0858 | OnlyJundong | Extended Compute Scaling Analysis (50K steps) | NEEDS REVIEW | |
| 13 | [#1394](https://github.com/openai/parameter-golf/pull/1394) | 1.0856 | clarkkev | SP8192 + GPTQ Embeddings + Depth Recurrence + MuonEq-R | NEEDS REVIEW | |
| 14 | [#1406](https://github.com/openai/parameter-golf/pull/1406) | 1.0887 | aamodbhatt | 11L Depth Recurrence + Discriminative Pre-Quant TTT | NEEDS REVIEW | |

### Tier 2: 1.08–1.10 BPB (Competitive)

| Rank | PR | BPB | Author | Technique | Validity | Notes |
|------|-----|-----|--------|-----------|----------|-------|
| 15 | [#1445](https://github.com/openai/parameter-golf/pull/1445) | 1.0889 | X-Abhishek-X | 3-Layer Depth Recurrence + EMA 0.9965 | NEEDS REVIEW | |
| 16 | [#1399](https://github.com/openai/parameter-golf/pull/1399) | 1.0898 | AnubhavBharadwaaj | Pre-Quant TTT + ETLB | NEEDS REVIEW | Pre-quant TTT likely invalid |
| 17 | [#1331](https://github.com/openai/parameter-golf/pull/1331) | 1.0900 | dexhunter | MuonEq-R + 3-Layer Recurrence + All-Int6 | NEEDS REVIEW | |
| 18 | [#1285](https://github.com/openai/parameter-golf/pull/1285) | 1.0912 | dexhunter | MuonEq-R + Depth Recurrence + Mixed Int5/Int6 GPTQ | NEEDS REVIEW | |
| 19 | [#1415](https://github.com/openai/parameter-golf/pull/1415) | 1.0913 | bigbag | SP4096 + 3-Layer Recurrence + GPTQ Embeddings + ETLB | VALID | ETLB bias trained on context only. Clean. |
| 20 | [#1344](https://github.com/openai/parameter-golf/pull/1344) | 1.0923 | Omrigotlieb | SP4096 + Polar Express + MuonEq-R | NEEDS REVIEW | |
| 21 | [#1395](https://github.com/openai/parameter-golf/pull/1395) | 1.0924 | dttdrv | SP4096 + Linear LR + Depth Recurrence | NEEDS REVIEW | |
| 22 | [#1421](https://github.com/openai/parameter-golf/pull/1421) | 1.0925 | X-Abhishek-X | 11L Depth Recurrence + EMA Tuning (0.9965) | VALID | Vanilla sliding window. Frozen model. Clean. |
| 23 | [#1291](https://github.com/openai/parameter-golf/pull/1291) | 1.0925 | dentity007 | Vocab4096 + MLP4.0x + SLOT | NEEDS REVIEW | SLOT requires scrutiny |
| 24 | [#1260](https://github.com/openai/parameter-golf/pull/1260) | 1.0929 | dexhunter | MuonEq-R + Depth Recurrence + Mixed Int5/Int6 GPTQ | NEEDS REVIEW | |
| 25 | [#1339](https://github.com/openai/parameter-golf/pull/1339) | 1.0955 | bigbag | SP2048 + 3-Layer Recurrence + SWA + BigramHash | NEEDS REVIEW | |
| 26 | [#1407](https://github.com/openai/parameter-golf/pull/1407) | 1.0960 | OnlyJundong | Extended Compute Scaling Analysis | NEEDS REVIEW | |
| 27 | [#1435](https://github.com/openai/parameter-golf/pull/1435) | 1.0980 | AbhayAnandUCSD | 11L Depth Recurrence + BigramHash + EMA 0.9965 | VALID | Standard sliding window. Frozen model. Clean. |
| 28 | [#1446](https://github.com/openai/parameter-golf/pull/1446) | 1.0960 | LauraGomezjurado | 11L gated Krylov + AR GPTQ int6 + lzma | NEEDS REVIEW | |

### Tier 3: Sub-1.0 BPB (SLOT/N-gram Heavy — Validity Suspect)

| Rank | PR | BPB | Author | Technique | Validity | Notes |
|------|-----|-----|--------|-----------|----------|-------|
| — | [#1430](https://github.com/openai/parameter-golf/pull/1430) | 0.3964 | renqianluo | Per-Sample SLOT + N-gram Order-22 + TTT | INVALID | Improperly normalized n-gram mixer. Below 0.70 theoretical floor. |
| — | [#1379](https://github.com/openai/parameter-golf/pull/1379) | 0.4162 | LucasErcolano | Mixed quant ngram | LIKELY INVALID | Below 0.70 floor. Likely normalization issue. |
| — | [#1329](https://github.com/openai/parameter-golf/pull/1329) | 0.6361 | renqianluo | Per-Sample SLOT + TTT | LIKELY INVALID | Below 0.70 floor. Same author as #1430. |
| — | [#1319](https://github.com/openai/parameter-golf/pull/1319) | 0.6951 | canivel | 11L LeakyReLU² XSA-all GPTQ-AR SLOT64 | SUSPECT | Near theoretical floor. SLOT-64 requires scrutiny. |
| — | [#1376](https://github.com/openai/parameter-golf/pull/1376) | 0.7094 | stukenov | SLOT-24 + Pre-quant TTT | SUSPECT | Pre-quant TTT + SLOT combination. |
| — | [#1324](https://github.com/openai/parameter-golf/pull/1324) | 0.7271 | yahya010 | SLOT-48 + VRL + QK-Gain 4.0 | SUSPECT | Large SLOT window. |
| — | [#1321](https://github.com/openai/parameter-golf/pull/1321) | 0.7406 | anthony-maio | SLOT-48 | SUSPECT | Large SLOT window. |
| — | [#1278](https://github.com/openai/parameter-golf/pull/1278) | 0.7736 | GitGeeks | SLOT-32 + Partial Depth Recurrence | SUSPECT | Large SLOT window. |
| — | [#1368](https://github.com/openai/parameter-golf/pull/1368) | 0.8503 | JKSNS | Mean-delta warm start + depth recurrence | NEEDS REVIEW | |
| — | [#1313](https://github.com/openai/parameter-golf/pull/1313) | 0.8637 | anthony-maio | SLOT-24 Aggressive | SUSPECT | |
| — | [#1263](https://github.com/openai/parameter-golf/pull/1263) | 0.9354 | xexyz | 11L LeakyReLU² + XSA-all + Full GPTQ + SLOT | SUSPECT | |
| — | [#1303](https://github.com/openai/parameter-golf/pull/1303) | 0.9462 | anthony-maio | SLOT + QK-Gain 4.0 + XSA-11 | SUSPECT | |
| — | [#1246](https://github.com/openai/parameter-golf/pull/1246) | 0.9650 | deborahnelson8788726 | Trinity Ternary GPT | NEEDS REVIEW | |
| — | [#1241](https://github.com/openai/parameter-golf/pull/1241) | 0.9901 | aiejvn | MDLM Masked Diffusion | NEEDS REVIEW | Different architecture class |
| — | [#1318](https://github.com/openai/parameter-golf/pull/1318) | 1.0096 | renqianluo | TTT-AdamW + SLOT L-BFGS25 LogitDelta | NEEDS REVIEW | Same author as #1430 |

### Merged Records (Official Leaderboard)

| PR | BPB | Author | Technique |
|----|-----|--------|-----------|
| [#1019](https://github.com/openai/parameter-golf/pull/1019) | 1.1147 | abaybektursun | AR Self-Gen GPTQ + XSA-all + BigramHash 3072x112 |
| [#549](https://github.com/openai/parameter-golf/pull/549) | 1.1194 | abaybektursun | LeakyReLU² + Legal Score-First TTT + Parallel Muon |
| [#414](https://github.com/openai/parameter-golf/pull/414) | 1.1233 | signalrush | 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15 |
| [#315](https://github.com/openai/parameter-golf/pull/315) | 1.1248 | jfprincz | 11L Partial RoPE + LN Scale + EMA + XSA4 |
| [#287](https://github.com/openai/parameter-golf/pull/287) | 1.1271 | jfprincz | 11L XSA + EMA + Int6 MLP3x + WD=0.04 |
| [#265](https://github.com/openai/parameter-golf/pull/265) | 1.1307 | unnir | 11L + Efficient Partial XSA |
| [#180](https://github.com/openai/parameter-golf/pull/180) | 1.1428 | thwu1 | 10L Int5-MLP + BigramHash(10240) + SWA(0.4) |
| [#162](https://github.com/openai/parameter-golf/pull/162) | 1.1483 | raahilshah | Int6 MLP3x + SmearGate + BigramHash + MuonWD + SWA |
| [#65](https://github.com/openai/parameter-golf/pull/65) | 1.1556 | aquariouseworkman | Mixed Quant Int6/FP16 + SmearGate + OrthoInit |
| [#63](https://github.com/openai/parameter-golf/pull/63) | 1.1598 | yahya010 | 10L Int6 QAT + Zstd MLP2.6x |

---

## Verified Valid Submissions (Code-Reviewed)

| PR | BPB | Author | Technique | Key Finding |
|----|-----|--------|-----------|-------------|
| [#1408](https://github.com/openai/parameter-golf/pull/1408) | ~~1.0800~~ | aamodbhatt | dTTT + BigramHash 3072x112 | **MOVED TO INVALID** — pre-quant dTTT trains 10 epochs on val data. |
| [#1420](https://github.com/openai/parameter-golf/pull/1420) | 1.0801 | abaybektursun | Triple Loop + Fused Kernels + N-gram Tilt | Fixed in 5e2eff8. token_hint clean, within/word_hint fixed. |
| [#1413](https://github.com/openai/parameter-golf/pull/1413) | 1.0828 | dexhunter | SP8192 + QK-Gain 5 + Score-First TTT | Chunks scored under no_grad() before training. |
| [#1415](https://github.com/openai/parameter-golf/pull/1415) | 1.0913 | bigbag | SP4096 + 3-Layer Recurrence + ETLB | ETLB bias trained on context tokens only. |
| [#1421](https://github.com/openai/parameter-golf/pull/1421) | 1.0925 | X-Abhishek-X | 11L Depth Recurrence + EMA | Vanilla sliding window. No eval-time adaptation. |
| [#1435](https://github.com/openai/parameter-golf/pull/1435) | 1.0980 | AbhayAnandUCSD | 11L Depth Recurrence + BigramHash | BigramHash is trained component. Standard frozen eval. |

## Confirmed Invalid Submissions (Code-Reviewed)

| PR | BPB | Author | Violation |
|----|-----|--------|-----------|
| [#1430](https://github.com/openai/parameter-golf/pull/1430) | 0.3964 | renqianluo | Rule 2: N-gram mixer not normalized over full vocab. Explains impossible 0.40 BPB. |
| [#1423](https://github.com/openai/parameter-golf/pull/1423) | 1.0791 | aryanbhosale | Rules 1,3,4: Pre-quant TTT trains on val data 6 epochs before scoring same data. |
| [#1416](https://github.com/openai/parameter-golf/pull/1416) | 1.0795 | erichroepke | Rules 1,3: Same pre-quant TTT pattern. Author acknowledged and withdrew. |
| [#1408](https://github.com/openai/parameter-golf/pull/1408) | 1.0800 | aamodbhatt | Rules 1,3,4: Pre-quant dTTT trains 10 epochs on val data. Artifact encodes val token info. |

---

## Key Observations

1. **Best verified-valid score: 1.0801 BPB** (PR #1420, abaybektursun — after causal fix)
2. **Sub-0.70 BPB submissions are almost certainly invalid** — the theoretical entropy floor for web text is ~0.70 BPB
3. **Pre-quantization TTT** (training on val data before quantizing into artifact) is the most common serious violation
4. **SLOT** (Score-Optimized Last-layer Tuning) is a gray area — small SLOT windows are generally accepted, but large windows (SLOT-32+) produce suspiciously low scores
5. **Score-first TTT** (score chunk, then train on it) appears to be valid when correctly implemented
6. **N-gram tilt** is valid when properly normalized and causal, but easy to get wrong

---

*Generated 2026-04-07. Validity assessments based on source code review against PR #1017 rules.*
