parameter golf is ruining my life (in a good way)

I have a Hyrox race this week that I haven't trained for because I've been up until 5am writing GPU kernels for a language model competition. I keep forgetting to eat. I've slept maybe 12 hours total since Tuesday. I regret nothing.

So OpenAI dropped this thing called Parameter Golf — fit the best language model into 16 megabytes, train it in 10 minutes on 8×H100s. I'm mass-coordinating Claude Code and Codex agents across rented GPU machines and they just run experiments all night while I sleep. Sixty experiments overnight. The git log looks insane.

I got scooped four times. Like, I'd have the exact same technique working, be doing my validation runs, and someone submits a PR with it before mine finishes training. Four separate times my result goes from competitive to irrelevant while I wait for a training run.

That broke something in my brain. I stopped trying to beat the model and started trying to beat the training speed instead — if you train more steps in the same 10 minutes, you win, and nobody else was going after that angle.

I spent 12 hours writing custom CUDA kernels trying to speed up the matrix multiplications. All of it turned out to be a benchmarking artifact — H100 has hardware memory compression that inflates numbers on synthetic data by up to 50%, and I'm fairly sure this is silently fooling published academic benchmarks too. Twelve hours for a negative result. But it pointed me at the optimizer, which was actually wasteful. Got it from 19.7ms down to 1.31ms per step.

End result: same model as the current #1, zero changes to architecture, just faster training. 227 more steps. Better score. Submitted my PR at 5am.

I think I'm top 3 unofficially. OpenAI starts reviewing submissions Monday so we'll find out.

I've spent way more money renting H100s this week than I'm comfortable sharing. Competition goes through April 30 and I have more ideas. If anyone has compute they'd share I'm very serious about putting it to work — DMs open.

Going to go eat something and try to remember how to run.

https://github.com/openai/parameter-golf/pull/399
