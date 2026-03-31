# I Kept Almost Winning

Three days into OpenAI's Parameter Golf challenge, I had a model scoring 1.1318 bits per byte. The leaderboard SOTA was 1.1307. Eleven ten-thousandths apart. I needed three seed runs for statistical significance before I could submit a PR. By the time I had them, someone else had posted 1.1271.

This happened to me four times.

Parameter Golf, if you haven't seen it, is a challenge to train the best language model that fits in 16 megabytes. You get 10 minutes on 8×H100 GPUs. That's it. Your score is bits per byte on a held-out FineWeb validation set — lower is better. The baseline starts at 1.2244. It launched March 18th and the leaderboard turned into a knife fight almost immediately.

I want to tell this story because I think there's something interesting in it that isn't about the final score. I independently discovered techniques that made it onto the leaderboard under other people's names, I built systems that nobody else in the competition built, and I found a hardware artifact in my benchmarks that I'm pretty sure has fooled every CUTLASS benchmark ever published on H100. I did eventually submit a PR. But getting there is the story.

## The first 48 hours

The challenge dropped and I was on it within hours. My first real result: sequence length 2048 with FP16 tied embeddings, 1.2067 BPB. Not bad — the baseline was 1.2244. I pushed it to the leaderboard and felt good for about six hours. Then sliding window evaluation showed up as a technique (someone scores each token with nearly full context by sliding a window at stride 64 instead of chunking), and suddenly everyone's numbers dropped by 0.02. The game had changed while I was sleeping.

This became the pattern. I'd identify an improvement, validate it, start collecting multi-seed runs, and somewhere during those 30 minutes of training time somebody else would submit a PR with the same idea. Or a better one.

March 19th was the worst day for this. I had a stack working — Int6 quantization-aware training, 3× MLP width, SmearGate token blending, orthogonal init, weight decay 0.04 — that scored 1.1507. The leaderboard at that moment showed 1.1502. Three thousandths of a BPB apart. I go to prepare my submission, and in that window two more PRs land, one at 1.1458 and another at 1.1428. My number isn't even interesting anymore.

## I started building machines instead

Somewhere around midnight on March 19th I made a decision that changed the rest of my week. I stopped trying to compete manually and started building a system to compete for me.

The logic was simple and slightly desperate: there are people in this competition who are better and faster at manual iteration than me. I can't beat their intuition. But I can run more experiments per hour than any human if I automate the whole loop.

So I built a multi-agent system. Multiple Claude Code instances, each running on a different rented GPU machine, coordinated through DynamoDB. Each agent runs autonomously in a loop:

1. Check the current SOTA by scraping PRs on the official repo
2. If SOTA changed, fetch the new leading code and run it as a baseline on this machine
3. Read a research document I maintain about what's been tried and what works
4. Propose an architectural change (not a hyperparameter tweak — I was strict about this)
5. Implement it, smoke test it (10 iterations, 30 seconds, does it crash, is the artifact under 16MB)
6. Full training run. Ten minutes. Parse the logs.
7. If it beats the baseline, notify me on my phone. If not, record what happened and loop.

A cron-based watchdog checks every 5 minutes whether the agent process is still alive. If it died, restart it. The agent writes heartbeats to the database; other agents check if heartbeats go stale, mark the dead agent's experiment as available, and can pick it up. I wrote a full smoke test suite — 10 test cases, 635 lines — because I was going to let this thing spend real money on GPU hours and I didn't want it burning cycles on broken experiments.

The git log from this period is something. My agent on a remote machine committed every experiment result, and it ran experiments roughly every 12 minutes around the clock. There are 60+ commits that look like this:

```
revert: tied_embed_lr 0.03786 experiment (restore 0.03784)
revert: tied_embed_lr 0.03788 experiment (restore 0.03784)
revert: matrix_lr 0.05100 experiment (restore 0.05000)
revert: matrix_lr 0.04900 experiment (restore 0.05000)
```

That's a machine grinding through hyperparameter space at 3am while I sleep. It found some things. It found that QK gain init of 1.70 was better than 1.50. It found precise learning rate values for tied embeddings. Small wins, but they compound.

And yes, I know I said the agent should propose architectural changes, not hyperparam sweeps. The agent had its own ideas about that.

## The techniques I found (that other people also found)

I want to be specific about what I actually achieved, because "I almost won" is a meaningless claim without numbers.

**11 layers instead of 9.** The baseline used 9 transformer layers. Int6 quantization compresses weights enough that you can fit two more layers under the 16MB cap. This was worth about 0.03 BPB. I had this working on March 20th. By then, three other people had submitted 11-layer models.

**Mixed Int5/Int6 quantization.** Int5 for MLP weights (they're more compressible), Int6 for attention weights (they need the precision), FP16 for embeddings (most sensitive tensor). The Int5 savings bought room for a 10th layer plus bigger bigram hash tables. I expanded BigramHash from 4096 to 10240 buckets. Score: 1.1428 BPB.

**Efficient GQA-aware XSA.** This one I'm actually proud of. Exclusive Self-Attention (from a March 2026 paper) removes self-attention bias in deep layers by projecting out the self-value component. The standard implementation with Grouped Query Attention requires `repeat_interleave` to expand value vectors, which doubles memory per layer. I found you can do it with a reshape and a broadcast instead:

```python
# The standard way: expensive, allocates memory
v_expanded = v.repeat_interleave(group_size, dim=-2)
vn = normalize(v_expanded)
y = y - dot(y, vn) * vn

# What I did: free reshape, zero allocation
y_grouped = y.reshape(B, T, Hkv, group_size, D)
vn = normalize(v).unsqueeze(-2)
y = (y_grouped - dot(y_grouped, vn) * vn).reshape(B, T, H, D)
```

Overhead went from ~7ms/step to ~2ms/step. Applied to the deepest 3 layers. Score: 1.1307. Someone else posted 1.1271 the same day with XSA on 4 layers instead of 3.

## Then I went to the metal

By March 20th the SOTA was 1.1248 and architectural improvements were getting thin. Everyone was converging on the same stack: 11 layers, Int6 QAT, EMA, XSA, Muon optimizer, relu² activation, Flash Attention 3. The differences between top entries were in the fourth decimal place.

So I figured: if I can't find a better model, I can train longer. The SOTA trains at ~85ms per step, getting about 7,050 steps in the 600-second budget. If I cut that to 38ms, I'd get 15,000 steps. That's like doubling the compute budget without paying for another machine.

I rented an H100, spun up a new repo (`parameter-golf-kernels`), and spent 12 hours writing CUDA.

My first target was the matrix multiplications. GEMMs dominate the training loop — about 70ms out of 85ms is matmul in the forward and backward pass. I started with CUTLASS, NVIDIA's template library for custom GEMM kernels. I wrote 12 different tile and schedule configurations targeting our specific shapes (M=98304, K=512, N=512/256/1536). My first benchmarks showed CUTLASS beating cuBLAS by 15-53%.

I was thrilled. For about two hours.

Then I changed the benchmark data from CUTLASS's default `BlockFillRandom` to `torch.randn` and the speedup vanished.

## The H100 memory compression thing

This is the part I think actually matters outside of this competition.

H100 GPUs have hardware-level memory compression that nobody talks about. It's transparent — you don't enable it, you don't configure it, it just happens. And it silently inflates benchmark results for any data with low entropy.

CUTLASS's default benchmark initializer, `BlockFillRandom`, generates values in a narrow range with limited exponent diversity. `cudaMemset` is even worse. These data patterns compress well in hardware, which means memory transfers are effectively faster, which means your GEMM benchmark reports numbers that don't correspond to anything you'll see in real training.

Here's what I measured on the exact same GEMM (shape: 98304×512×512):

| Data initialization | Time | Reported efficiency |
|---|---|---|
| `torch.randn` (what training actually looks like) | 0.130ms | 47% of peak |
| `BlockFillRandom[-2,2]` (CUTLASS default) | 0.109ms | 57% of peak |
| `cudaMemset(0x3C)` (constant) | 0.098ms | 63% of peak |
| All zeros | 0.085ms | 73% of peak |

That's a 1.5× inflation from zeros to real data. Every benchmark that doesn't use `torch.randn` or actual training tensors is reporting inflated numbers. I'm pretty sure this affects published CUTLASS benchmarks, Triton benchmark suites, and most academic GEMM papers that test on H100.

After I corrected for this, I tested everything. CUTLASS TMA+WGMMA in 12 configurations. Triton autotune. cuBLASLt exhaustive algorithm search. Stream-K scheduling. L2 cache persistence hints. QKV fusion. Nothing beats cuBLAS on real data for our shapes. Not by a single percent.

The reason is a pipeline depth problem. H100 tensor cores use software-pipelined WGMMA instructions where each BK=64 tile of the K-dimension is one pipeline stage. With K=512 (our model dimension), that's only 8 pipeline iterations. Not enough to hide memory latency. The ~48% efficiency ceiling isn't a software problem. It's physics.

I wrote 2,459 lines of CUDA to prove that the library everyone's already using is the best option. Twelve hours of work to conclude there's nothing to do. But now I know *why* there's nothing to do, and the finding about model dimension is actually useful: going from dim=512 to dim=768 improves GEMM efficiency by 10-13% because you get 12 pipeline iterations instead of 8. If anyone in this competition wants to make their model faster, make it wider, don't touch the kernel.

## So I stopped trying to beat the model and started trying to beat the clock

The GEMM investigation killed the fantasy of a 2× training speedup through custom kernels. But it taught me where the real waste was: the optimizer.

The Muon optimizer runs Newton-Schulz orthogonalization on every weight matrix independently. That's 66 separate small GEMMs, launched one at a time — 120 kernel launches per step, each one touching barely any of the GPU. The entire optimizer step took about 19.7ms, and for most of that time the H100 was sitting at under 1% utilization waiting for the next kernel to launch.

I restructured the model's 66 separate `nn.Linear` weight tensors into 4 contiguous 3D parameter banks — one bank per weight shape. So instead of 11 separate query projection matrices, there's one tensor of shape `(11, 512, 512)`. Forward pass uses `F.linear(x, bank[layer_idx])`. Same math, different memory layout.

Then I replaced the standard Newton-Schulz iteration with Polar Express — minimax-optimal polynomial coefficients from a May 2025 paper that converge 35% tighter in BF16. Running batched Polar Express across the parameter banks instead of iterating over individual matrices: optimizer time dropped from 19.7ms to 1.31ms. Fifteen times faster.

But there was a catch. DDP doesn't understand parameter banks. When your weight tensor is a 3D bank covering all 11 layers, the gradient aggregates across the entire backward pass and only becomes available at the very end. DDP's whole trick is overlapping communication with backward computation — it fires off gradient syncs as each layer finishes. With banks, that overlap is destroyed. I measured a +4ms regression from DDP alone, which ate most of my optimizer savings.

The fix: rip out DDP entirely for bank parameters. Handle the communication yourself. Async reduce-scatter the bank gradients, then while that's in flight, run AdamW on the small 1D parameters (biases, norms, gates — these are tiny and fast). When the reduce-scatter finishes, each rank runs Polar Express on its local shard only. Then async all-gather the orthogonalized updates, overlapping that communication with the next bank's Polar Express. This is the Parallel Muon approach from a November 2025 paper, adapted to work with parameter banking.

I got it working on March 21st. Head-to-head against the SOTA (PR #315) on 8×H100:

| | PR #315 Baseline | Mine |
|--|---|---|
| step time | 84.76 ms | 82.13 ms |
| steps in 600s | 7,079 | 7,306 |
| val_bpb | 1.1253 | 1.1238 |

Same model. Same hyperparameters. Same architecture. 227 extra training steps, and a val_bpb of 1.1238 — better than the current SOTA — from pure systems optimization. The model doesn't know anything changed. It just got to train longer.

I submitted [PR #399](https://github.com/openai/parameter-golf/pull/399) at 4:52am.

## The timeline, compressed

I want to lay this out because I think the pace tells its own story.

March 18: challenge launches. I start training that night.

March 19: I hit 1.1507. Leaderboard shows 1.1502. I blink, and two more PRs land ahead of me. I start building the multi-agent system at midnight.

March 20: I hit 1.1307 with efficient XSA. Someone posts 1.1271. I hit 1.1271 myself later that day. Someone posts 1.1248. I read all 300+ PRs and write a 400-line research synthesis. I start kernel optimization that evening. 60+ automated experiment commits pile up from the agent running on a remote GPU.

March 21: 12 hours of CUDA. The GEMM investigation, the H100 memory compression discovery, 2,459 lines of kernel code. I close the GEMM workstream, pivot to the optimizer, implement parameter banking + Polar Express (15× faster optimizer), then Parallel Muon. The full stack hits 82ms/step at 11:48pm.

March 22, 4:52am: I submit PR #399.

Four days. Sixty automated experiments. 2,459 lines of CUDA. A multi-agent research system. And a PR that takes the current SOTA model and makes it train 3.1% faster without changing a single hyperparameter.

The artifact is 140KB over the 16MB limit right now. I'm working on that. But the speed is real, the improvement is lossless, and I finally have my name on a PR.

I also read every single PR on the competition repo — all 300+ — and wrote a synthesis document mapping which techniques work, which don't, and which ones interact badly. The biggest surprise: XSA and test-time training are mutually destructive. XSA removes self-attention bias at training time; TTT adapts weights to local patterns at eval time. They're targeting the same signal. Combining them is 0.016 BPB *worse* than XSA alone. Nobody had documented this. People were wasting GPU hours trying to stack them.

## What I think about now

The people who topped the leaderboard are good. Really good, and fast. I'm not going to pretend I was robbed. They got there first because they moved faster, or started earlier, or had better intuitions from day one.

But I played a different game by the end. Instead of chasing the fourth decimal place on model quality, I built infrastructure — an autonomous agent system, a kernel library, a hardware investigation that I haven't seen written up anywhere else. The GEMM memory compression finding alone is something I think matters beyond this competition. And the kernel work actually paid off: 82ms/step, lossless, submitted.

The competition runs through April 30th. The agent system is still running. The kernel library has more optimizations queued — FP8 forward pass if I can stomach the 0.002 BPB quality tradeoff, fused cross-entropy, FA3 tile autotuning. 38ms/step is still the target.

If you want to look at any of this, the repos are public. The agent code, the kernel library, the 2,459 lines of CUDA that prove cuBLAS wins. Take what's useful.
