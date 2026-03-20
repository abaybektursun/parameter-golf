# Kernel Speedrun Learnings

- 2026-03-20: Baseline speed benchmark on `experiments/004_kernel_speedrun/train_gpt.py` (1xA100, `ITERATIONS=150`, `VAL_LOSS_EVERY=0`) reached `step_avg=726.37ms` at step 150.
- The script currently does not log MFU%, so `mfu_percent` is set to `0.0` in results until MFU logging is added.
- Next Tier-1 optimization to test: `torch.compile(..., mode="max-autotune")`.
- 2026-03-20: Commit `0522fb9` (`torch.compile(..., mode="max-autotune")`) failed to reach `warmup_step` on 1xA100; after extensive autotune output, the process stayed CPU-bound with no new log lines for >3 minutes (`run.log` frozen at 21,881 bytes). Recorded as `crash` and reverted for subsequent experiments.
- 2026-03-20: Commit `b68ddf6` (`mode="reduce-overhead"`) crashed immediately with `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`, traced to mutable RoPE cache tensors in `Rotary.forward`. Reverted.
- 2026-03-20: Commit `91d4e5f` added `torch.compiler.cudagraph_mark_step_begin()` before model invocations, but `mode="reduce-overhead"` still crashed with the same Rotary cache overwrite error. Marker insertion alone is insufficient; would require restructuring/removing mutable RoPE cache for CUDA-graph compatibility.
