# parameter golf is ruining my life (in a good way)

I have a Hyrox race this week. I have not trained for it. What I've been doing instead is staying up until 5am trying to shave 2.6 milliseconds off a training loop, and honestly I think that's a reasonable trade.

OpenAI's running this [competition](https://github.com/openai/parameter-golf) — best language model that fits in 16 megabytes, 10 minutes on 8×H100s. I'm at 1.1238 bits-per-byte. I think that's top 3. OpenAI starts approving PRs Monday and the queue is stacked, so we'll see, but the number is real and I submitted [my PR](https://github.com/openai/parameter-golf/pull/399) at 5am this morning on maybe four hours of sleep.

Getting here was stupid. I got scooped four times — not "similar approach" scooped, I mean I'd have the identical technique working, running my validation seeds, and someone would submit a PR with the same result before my training run finished. 1.1507 when the board showed 1.1502. I'd go to submit and find two new PRs ahead of me. Four times.

At some point I stopped trying to outrun people and started building systems. I've got Claude Code and Codex agents running on rented GPUs through DynamoDB, looping autonomously — they scrape the leaderboard, fetch the current best code, propose changes, train, and ping my phone if they beat baseline. They're not going to invent architectures but they'll grind sixty experiments overnight while I'm getting my four hours. The git log is unhinged. Sixty-something commits at 3am, each one a twelve-minute experiment.

But the thing I actually want to talk about is the CUDA.

Everyone in this competition converged on the same model. Same architecture, same optimizer, same everything. So I figured I'd make it train faster — more steps in the same 10 minutes, better score. I wrote 2,459 lines of custom GEMM kernels in CUTLASS. Benchmarks showed 15-53% speedup over cuBLAS.

All of it was a benchmarking artifact. H100 has hardware memory compression that inflates numbers on synthetic data by up to 50%. Switch to `torch.randn` — which is what real training actually looks like — and the "speedup" goes to zero. Twelve hours of work. I think this artifact is quietly fooling published academic benchmarks too, and nobody's written it up.

The optimizer was real though. Muon does 120 kernel launches per step, GPU at 1% utilization most of the time. I packed 66 weight matrices into 4 contiguous banks, swapped in Polar Express for Newton-Schulz, ripped out DDP and wrote the communication by hand. Optimizer went from 19.7ms to 1.31ms.

Same model as current SOTA, zero changes to the architecture or hyperparameters, 227 extra training steps in 600 seconds. 1.1238 BPB. The model doesn't know anything changed. It just trains longer.

I've spent more on H100 rentals this week than I want to think about. The competition runs through April and I've got more optimizations queued. If anyone has compute to share — genuinely, DMs open, I will put it to work.

Now I need to eat something. And probably train for the race I'm about to suffer through.

[PR #399](https://github.com/openai/parameter-golf/pull/399) | [kernels](https://github.com/abaybektursun/parameter-golf-kernels) | [agent system](https://github.com/abaybektursun/parameter-golf)
