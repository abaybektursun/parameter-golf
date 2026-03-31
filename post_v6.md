parameter golf is ruining my life (in a good way)

Just submitted a PR and I am now #1 on OpenAI's AI research challenge. 1.1213 BPB. The next closest legal submission is 1.1231. I've been sleeping about 4 hours a night since this started and I keep forgetting to eat, which is a problem because I have a Hyrox race in a few days that I have done absolutely zero prep and recovery for.

Parameter Golf is OpenAI's new challenge. Best language model, 16 megabytes, 10 minutes of training. The people in this competition are ML PhDs from fancy unis and actual research scientists. I got scooped four times. I'd have the same technique working as the person who just submitted, sitting there watching my validation run finish, and by the time it's done there's a new PR ahead of me. This happened four times in three days. I started to lose it a little.

What I ended up doing was building a fleet of Claude Code and Codex agents on rented GPUs that just run experiments autonomously. I split them into three tracks. One track watches the actual leaderboard and PRs obsessively — what's really SOTA right now, what empirical data do we have, where's the leverage. Second track does deep research into weirder directions that nobody in the competition is exploring yet, stuff like state space models, esoteric architectures. Third track is hardware — kernel optimization and training speed on 8×H100 specifically. That third one is where my actual edge is right now because nobody else is going after it.

The agents scrape the leaderboard, propose changes, train, text me if they beat baseline. They're genuinely good at this  not "replace a researcher" good but "tireless, will try everything, never complain" good.

I also implemented 2,459 lines of custom CUDA kernels that turned out to be completely useless because H100 has a hardware memory compression feature that was inflating all my benchmarks by 50%. 6 hours of work for a negative result. I think this artifact is fooling some other benchmarks too but that's a separate post.

Pointed me somewhere cool though. The optimizer was doing 120 tiny kernel launches per step, GPU basically idle. I packed the weights into contiguous banks, swapped in better math, rewrote the distributed communication by hand. 19.7ms down to 1.31ms on a single GPU setting. That got me a [faster training PR](https://github.com/openai/parameter-golf/pull/399) — same model, just more steps in the same 10 minutes. Then I stacked test-time training on top of the faster model and that's the 1.1213 submission.

OpenAI starts reviewing Monday. I'll keep winning until I run out of compute quotas, which at the rate I'm burning through H100 rentals might be soon. Competition runs through April and I have a whole queue of optimizations I haven't tried yet. If anyone out there has compute to spare, seriously, come talk to me. I will use every hour of it.

Going to go try to remember how legs work before my hyrox.
