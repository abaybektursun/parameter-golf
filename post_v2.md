# I've slept 12 hours in 4 days because of a competition to train the smallest good language model

OpenAI put out this challenge called Parameter Golf last Tuesday. Train the best language model that fits in 16 megabytes. You get 10 minutes on 8×H100s. Lowest bits-per-byte on FineWeb wins. Sounds simple. I have not been able to turn my brain off since.

The current SOTA is 1.1248 BPB. I'm at 1.1238. I think that puts me top 3, maybe better — hard to say for sure because OpenAI starts reviewing PRs on Monday and the queue is backed up. I submitted [PR #399](https://github.com/openai/parameter-golf/pull/399) at 5am this morning. My hands were shaking a little from the coffee. Four days of work in that PR.

Here's how I got here, because the path was dumb and I want to tell someone about it.

## I kept getting scooped

Day one I scored 1.2067. Felt great. Went to sleep. Woke up and the whole leaderboard had shifted because someone figured out sliding window evaluation and everyone's numbers dropped by 0.02 overnight.

Day two I put together a stack — Int6 quantization, 3× MLP, SmearGate, weight decay — and hit 1.1507. The leaderboard at that exact moment: 1.1502. I'm three ten-thousandths away. I start my three-seed validation runs (the rules require statistical significance to submit). While those are training, two PRs land ahead of me. My number doesn't even qualify anymore.

This happened four separate times. I'd independently find the same technique as someone else, sometimes within hours, and they'd submit first. 11 layers instead of 9. Mixed Int5/Int6 quantization. Efficient XSA. Every time I'm right there and every time I'm too slow.

## So I started letting the machines cook

Around midnight on day two I stopped trying to out-iterate humans and started building agents to do it for me. I set up Claude Code and Codex instances running on rented GPUs, coordinated through DynamoDB. Each agent loops autonomously — scrapes the latest SOTA PR, fetches the code, proposes a change, smoke tests, trains, logs the result, notifies my phone if it beats baseline. A watchdog cron restarts them if they die.

Honestly? The agents have gotten really good. Not at the flashy stuff — they're not going to invent a new architecture. But for ideation and systematic experimentation they're relentless. My git log has 60+ commits from agents grinding at 3am while I'm passed out for my 4 hours:

```
revert: tied_embed_lr 0.03786 experiment (restore 0.03784)
revert: tied_embed_lr 0.03788 experiment (restore 0.03784)
revert: matrix_lr 0.05100 experiment (restore 0.05000)
```

They found real things. QK gain init of 1.70 beats 1.50. Precise learning rate values. Small wins that stack.

## I wrote 2,459 lines of CUDA to prove nothing works

By day three everyone had converged on the same model. Same 11 layers, same Int6, same Muon optimizer, same everything. Differences in the fourth decimal. So I thought: forget the model, make the training faster. If I halve the step time I double the training steps in the same 600 seconds.

I rented an H100 (more on the bill later) and spent 12 hours writing custom GEMM kernels in CUTLASS. My first benchmarks showed 15-53% speedup over cuBLAS.

Then I switched the test data from CUTLASS's synthetic default to `torch.randn` and the speedup vanished.

Turns out H100 has hardware memory compression that silently inflates benchmarks for low-entropy data. Same GEMM, same shape, different data:

| Data | Time | "Efficiency" |
|---|---|---|
| `torch.randn` (real training data) | 0.130ms | 47% |
| CUTLASS default (`BlockFillRandom`) | 0.109ms | 57% |
| Constant data | 0.098ms | 63% |
| Zeros | 0.085ms | 73% |

1.5× inflation from zeros to real data. I'm pretty sure this is fooling published benchmarks and academic papers and nobody's talking about it. 2,459 lines of CUDA and 12 hours to prove cuBLAS is already at the hardware limit for dim=512. The ~48% efficiency is a pipeline depth problem — 8 WGMMA iterations can't hide memory latency. Physics, not software.

## Where the actual speed came from

GEMMs were a dead end. But the Muon optimizer was embarrassingly wasteful — 120 tiny kernel launches per step, 19.7ms, GPU sitting at 1% utilization most of the time.

I packed the model's 66 separate weight matrices into 4 contiguous 3D parameter banks. Replaced Newton-Schulz with Polar Express (better polynomials, converge 35% tighter). Batched the whole thing. Optimizer: 19.7ms → 1.31ms. Fifteen times faster.

Then DDP broke because it can't overlap communication on banked parameters. So I ripped it out and wrote the communication by hand — async reduce-scatter, overlap with AdamW on small params, local Polar Express on each rank's shard, async all-gather overlapped with the next bank. Parallel Muon.

End result on 8×H100, head to head with the SOTA:

| | SOTA (PR #315) | Mine |
|--|---|---|
| ms/step | 84.76 | **82.13** |
| steps in 600s | 7,079 | **7,306** |
| val_bpb | 1.1253 | **1.1238** |

Same model. Same hyperparameters. Zero architecture changes. It just trains 227 more steps because the optimizer isn't wasting time. The model doesn't know anything changed. Pure systems work.

## The other thing I did that I think matters

I read all 300+ PRs on the competition repo and wrote a synthesis of what works and what doesn't. The most interesting finding: XSA and test-time training destroy each other. They target the same signal — local context modeling — from different angles. Combining them is 0.016 BPB *worse* than XSA alone. People were burning GPU hours trying to stack them. Nobody had documented the interaction.

## Where I'm at right now

It's 5am. I submitted the PR. The artifact is 140KB over the 16MB limit which I need to fix before it can be accepted. The competition runs through April 30th and I have more optimizations queued — FP8 forward pass, fused cross-entropy, FA3 tile autotuning. 38ms/step is the real target. I'm at 82.

I've spent more money renting 8×H100s this week than I want to think about. If anyone reading this has access to compute and wants to see where this goes, seriously, reach out. I will put it to work.

The repos are public — the multi-agent system, the kernel library, the CUDA investigation. The competition is still on and I don't plan on sleeping much until April.

[PR #399](https://github.com/openai/parameter-golf/pull/399) | [parameter-golf-kernels](https://github.com/abaybektursun/parameter-golf-kernels)
