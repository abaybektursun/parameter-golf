# Kernel Speedrun Learnings

- 2026-03-20: The previously committed `experiments/004_kernel_speedrun/results.tsv` and `learnings.md` belonged to an older 9-layer kernel-speedrun script. The new staged 11-layer / 3x-MLP baseline was committed as `d506433` before continuing experiments.
- 2026-03-20: Commit `d506433` (`kernel speedrun: import 11-layer baseline`) reached `step:84/150 val_loss:5.6079 val_bpb:3.3213 train_time:120669ms step_avg:1436.53ms` on 1xA100 with `ITERATIONS=150`, `VAL_LOSS_EVERY=0`, `MAX_WALLCLOCK_SECONDS=120`.
- 2026-03-20: This script's nominal speed benchmark path is slow operationally because, after the 120-second training cap is reached, it still performs a full validation pass before printing the final `val_loss/step_avg` line, and then falls through into serialization and roundtrip eval. For iteration, it is sufficient to watch the in-training `step_avg` lines and stop after a stable reading near the cap.
- 2026-03-20: Commit `d9eb73d` (`kernel speedrun: restore inductor gemm autotune`) re-enabled `torch._inductor.config.max_autotune=True` and `max_autotune_gemm=True`. Startup/autotune time increased materially on 1xA100, but steady-state training improved to `1423.03ms/step` by step 80 (vs. `1436.53ms` baseline), an improvement of about `13.50ms` / `0.94%`. Kept.
- 2026-03-20: The best current speed benchmark for the new 11-layer kernel-speedrun model is `1423.03ms/step` on 1xA100 at commit `d9eb73d`.
- Next kernel-focused candidates for this new baseline:
- Fused ReLU² MLP for the 1536-hidden MLP path, ideally with backward recomputation.
- Fused softcap + cross-entropy to avoid materializing the full vocab logits tensor.
- Profile the compiled graph (`TORCH_COMPILE_DEBUG=1`) before porting a custom kernel, to verify whether Inductor is already fusing the ReLU² epilogue and where the remaining unfused chains are.
