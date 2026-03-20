# Experiments

Each experiment lives in its own folder named `NNN_short_name/` (e.g. `001_combine_three_wins/`).

Inside each folder, maintain a single `notes.md` with the following sections:

---

## Template (`notes.md`)

```markdown
# Experiment: <short descriptive title>

## Hypothesis

What do you expect to happen and why? One or two sentences.

## Plan

- Step-by-step list of what you will change
- Specific config/flags/code diffs
- What metric you will measure and what counts as success

## Observations

Raw results as they come in. Timestamped if useful.

| Run | Seed | Steps | Pre-quant BPB | Post-quant BPB | Artifact Size | Notes |
|-----|------|-------|---------------|----------------|---------------|-------|
|     |      |       |               |                |               |       |

## Post-mortem

- Did the hypothesis hold? Why or why not?
- What was learned?
- Next steps or follow-up experiments
```
