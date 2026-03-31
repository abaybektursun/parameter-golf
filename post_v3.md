# parameter golf is ruining my life (in a good way)

I have a Hyrox race this week and instead of training for it I've been up until 5am writing CUDA kernels for a competition about fitting a language model into 16 megabytes. I keep forgetting to eat. I think I'm top 3.

[Parameter Golf](https://github.com/openai/parameter-golf) if you haven't seen it — OpenAI challenge, train the best LM under 16MB, 10 minutes on 8×H100s, scored on bits-per-byte. The leaderboard has been an absolute bloodbath since it launched on the 18th.

I got scooped four times. Not "someone had a similar idea" — I mean I'd have the exact same technique working, be running my validation seeds, and someone would submit a PR with the same thing before my training finished. 1.1507 when the board showed 1.1502. 1.1307 when someone else posted 1.1271 hours later. Every time, right there, just not fast enough.

So I did what any reasonable person would do at midnight on a Wednesday. I stopped doing the work myself and built a system to do it for me.

I've been running Claude Code and Codex agents on rented GPU machines, coordinated through DynamoDB. They loop autonomously — scrape the current SOTA, fetch the code, propose changes, smoke test, train, ping my phone if they beat baseline. They've gotten genuinely good at this. Not "replace a researcher" good, but "run 60 experiments while I sleep for four hours" good. My git log is unhinged:

```
revert: tied_embed_lr 0.03786 experiment (restore 0.03784)
revert: tied_embed_lr 0.03788 experiment (restore 0.03784)
revert: matrix_lr 0.05100 experiment (restore 0.05000)
```

That's a machine doing its thing at 3am. It found stuff. Small wins that stack.

Meanwhile I went after training speed. If you can't beat the model, make the model train longer in the same 10 minutes. I spent 12 hours writing custom GEMM kernels in CUTLASS targeting our exact shapes. My benchmarks showed 15-53% speedup over cuBLAS.

All fake.

H100 has hardware memory compression that nobody talks about. It inflates benchmark numbers for synthetic data by up to 50%. I switched from CUTLASS's default test data to `torch.randn` — which is what real training looks like — and the speedup went to zero. 2,459 lines of CUDA to prove the library everyone's already using is optimal. Twelve hours to learn there's nothing to do. But at least now I know *why* there's nothing to do, and I'm pretty sure this artifact is fooling published academic benchmarks too.

The optimizer though. That was real.

Muon does 120 tiny kernel launches per step. GPU sitting at 1% utilization most of the time. I packed 66 weight matrices into 4 contiguous banks, replaced Newton-Schulz with Polar Express (better convergence polynomials), batched everything. Optimizer went from 19.7ms to 1.31ms. Then I ripped out DDP and wrote async communication by hand because DDP doesn't understand banked parameters.

Result on 8×H100, same model as current SOTA, zero architecture changes:

| | SOTA | Mine |
|--|---|---|
| ms/step | 84.76 | **82.13** |
| val_bpb | 1.1253 | **1.1238** |

227 extra training steps. Better score. The model doesn't know anything changed, it just got to train longer. Pure systems optimization.

I submitted [PR #399](https://github.com/openai/parameter-golf/pull/399) at 5am. OpenAI starts reviewing the queue Monday. The artifact is 140KB over the 16MB limit which I need to shave before it can be accepted but the speed is real.

I've spent more money on 8×H100 rentals this week than I want to admit. If anyone has compute they'd be willing to share I will absolutely put it to work — I have more optimizations queued and the competition runs through April 30th. DMs open.

Now I need to go eat something and remember I have a race to not die in.

[PR #399](https://github.com/openai/parameter-golf/pull/399) | [kernel repo](https://github.com/abaybektursun/parameter-golf-kernels) | [agent system](https://github.com/abaybektursun/parameter-golf)
