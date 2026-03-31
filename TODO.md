# TODO: H100 Deployment Readiness

- [ ] Memory utilization: add guidance in program.md for sizing batches to available VRAM (don't overcorrect after OOM)
- [ ] S3 log upload: add to `complete_run()` and `fail_run()` so logs survive instance termination
- [ ] `fetch_sota_code` robustness: PR file matching grabbed wrong file once — harden the `records/` path logic
- [ ] Crash recovery: orphaned runs stay `status=running` forever if agent dies mid-training — watchdog should clean up
- [ ] Verify `code_hash` and `artifact_bytes` populate correctly in SOTA table on next detection
- [ ] Verify `detect_machine_config()` works correctly on multi-GPU H100 nodes
- [ ] Smoke test wallclock: agent used 60s instead of 30s — enforce in program.md
