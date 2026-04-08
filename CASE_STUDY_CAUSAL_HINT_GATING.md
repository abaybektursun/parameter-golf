# Case Study: Target-Dependent Hint Gating in N-gram Tilt

**PR:** [#1420](https://github.com/openai/parameter-golf/pull/1420) (Triple Loop + Fused Kernels + N-gram Tilt)
**Violation:** Rule 1 — Strict Causal Dependence
**Severity:** Minor (auxiliary channels only, primary channel clean)
**Status:** Fixed in commit `5e2eff8`

---

## 1. The Setup

The n-gram tilt system has three hint channels that can boost the probability
of a predicted token:

| Channel | Context Used | Orders | Contribution |
|---------|-------------|--------|-------------|
| `token_hint` | Hash of prefix tokens `[p-1, p-2, ..., p-16]` | 8–16 | ~85% of BPB gain |
| `within_hint` | Within-word position + prefix hash | 1–3 | ~10% |
| `word_hint` | Ring buffer of previous 4 word hashes | 4-gram | ~5% |

All three channels use hash tables updated in score-before-update order.
The tilt formula is properly normalized:

```
P_tilt(v) = P_model(v) * exp(beta * 1[v == hint]) / Z
Z = 1 + P_model(hint) * (exp(beta) - 1)
```

This sums to 1 over the full vocabulary. Rules 2, 3, and 4 all pass.

---

## 2. The Violation

In `fused_expert_blend.cpp`, the `get_hints_batch` loop:

```cpp
for (int i = 0; i < n; i++) {
    int64_t p = pos[i];
    auto tok = uint16_t(tokens_[p]);           // reads TARGET token
    bool is_bnd = is_bnd_ && is_bnd_[tok];     // flag from TARGET
    bool is_ws  = has_ls_ && has_ls_[tok];      // flag from TARGET
    ...
    within_hint(is_bnd, is_ws, ...);   // gated by TARGET properties
    word_hint(is_ws, ...);             // gated by TARGET properties
}
```

`tokens_[p]` is the token at position `p` — the **target being predicted**.
The `is_bnd` (boundary) and `is_ws` (whitespace/leading-space) flags are
static lookup-table properties of that token. These flags control which
hint channels fire:

- `within_hint`: suppressed if `is_bnd || is_ws`
- `word_hint`: only fires if `is_ws`

### Why this violates Rule 1

Rule 1 requires: *"The predictive distribution P_t(.) must be a function
only of the artifact and the strict prefix x_1, ..., x_{t-1}."*

At position `p`, two different target tokens produce two different
probability distributions:

**Scenario A** — target is a regular token (`is_ws=false`, `is_bnd=false`):
- `within_hint` may fire (boosting some token)
- `word_hint` suppressed
- Distribution: `P_tilt_A(v)`

**Scenario B** — target is a whitespace token (`is_ws=true`):
- `within_hint` suppressed
- `word_hint` may fire (boosting a different token)
- Distribution: `P_tilt_B(v)`

Same prefix, different distributions depending on `x_p`. This is invalid —
the distribution must be fixed before observing the target.

### Concrete example

Position `p` has prefix "the quick brown f". The model must assign
probabilities to all possible next tokens before seeing which one it is.

- If the next token happens to be "ox" (not whitespace): `within_hint`
  fires, boosting "ox" based on within-word patterns
- If the next token happens to be " jumped" (whitespace): `word_hint`
  fires instead, boosting " jumped" based on word-level patterns

The scorer is **choosing which expert to consult based on the answer**.
This is free information — it gets to pick the more favorable hint
channel after seeing the target.

---

## 3. Why It Was Hard to Spot

### 3a. The primary channel was clean

`token_hint` — the dominant contributor (~85% of BPB improvement) — uses
only prefix-derived hashes:

```cpp
static void compute_hashes(const int64_t* tokens, int64_t pos, ...) {
    for (int k = 0; k < lim; k++) {
        h ^= uint64_t(tokens[pos - k - 1]) * PRIMES[k];  // pos - k - 1
        hashes[k] = h;
    }
}
```

This reads `tokens[pos-1], tokens[pos-2], ...` — never `tokens[pos]`.
A reviewer checking only this function would conclude "causal, looks fine."

### 3b. The variable name `tok` was used for two purposes

`tok = tokens_[p]` is used for both:
1. **Hint gating** (before scoring) — the violation
2. **Hash table updates** (after scoring) — correct and necessary

Since updates legitimately need the target token, seeing `tok = tokens_[p]`
in the loop doesn't immediately raise alarms. The dual use obscures the
data flow.

### 3c. The flags are "metadata", not the full token

`is_bnd` and `is_ws` are coarse binary properties — essentially character
class membership. They don't reveal the token's identity. This makes the
violation feel less severe and easier to dismiss as "just optimization".
But under a strict reading of Rule 1, even 1 bit of information about
`x_p` is prohibited from influencing the distribution.

### 3d. The tilt formula is correctly normalized

Rule 2 (Full Normalized Distribution) passes cleanly. The violation is
entirely in Rule 1 — *which* distribution gets applied, not *how* it's
normalized. Reviewers checking normalization math would find nothing wrong.

---

## 4. Detection Checklist

When reviewing n-gram or eval-time scoring code, check each of these:

### Step 1: Identify all reads of the token array at the current position

Search for patterns like:
```
tokens[p]      tokens[pos]      tokens[i]      val_tokens[t]
target          y[i]             tgt             y_batch[...]
```

Any read of the token at the position being scored is a red flag.

### Step 2: Trace the data flow

For each read of `tokens[p]`, trace where the value flows:
- Into **scoring/hint logic** (before the score is locked) → **VIOLATION**
- Into **update logic** (after the score is locked) → **LEGAL**
- Into **both** → needs separation

### Step 3: Check for indirect information leakage

Even if `tokens[p]` isn't read directly, check for:
- Lookup tables indexed by `tokens[p]` (like `is_bnd_[tok]`)
- Conditional branches that depend on `tokens[p]`
- Hash functions that include `tokens[p]` as input
- Any function argument derived from `tokens[p]`

### Step 4: Verify the distribution is prefix-determined

Ask: "If I change the target token at position `p` while keeping the
prefix `x_1...x_{p-1}` identical, does the probability distribution
change?" If yes, Rule 1 is violated.

### Step 5: Check the update/score boundary

Verify that the code has a clear boundary:
1. **SCORE PHASE**: compute distribution, emit score — uses only prefix
2. **UPDATE PHASE**: modify state for future positions — may use target

If both phases share variables derived from `tokens[p]`, verify the
variables are computed separately for each phase, or that the shared
variable only flows into the update phase.

---

## 5. The Fix

**Before (violated Rule 1):**
```cpp
auto tok = uint16_t(tokens_[p]);
bool is_bnd = is_bnd_ && is_bnd_[tok];     // from target
bool is_ws  = has_ls_ && has_ls_[tok];      // from target
...
within_hint(is_bnd, is_ws, ...);            // tainted
word_hint(is_ws, ...);                      // tainted
...
within_update(tok, is_bnd, is_ws);          // same flags reused
word_update(tok, is_bnd, is_ws);
```

**After (causal):**
```cpp
auto tok = uint16_t(tokens_[p]);

// HINT GATING: prefix-only flags
auto prev_tok = (p > 0) ? uint16_t(tokens_[p - 1]) : uint16_t(0);
bool is_bnd = is_bnd_ && is_bnd_[prev_tok];
bool is_ws  = has_ls_ && has_ls_[prev_tok];
...
within_hint(is_bnd, is_ws, ...);            // clean
word_hint(is_ws, ...);                      // clean
...
// UPDATES: target's own flags (post-scoring, correct)
bool tok_is_bnd = is_bnd_ && is_bnd_[tok];
bool tok_is_ws  = has_ls_ && has_ls_[tok];
within_update(tok, tok_is_bnd, tok_is_ws);
word_update(tok, tok_is_bnd, tok_is_ws);
```

Key insight: **two separate sets of flags** — one for hint gating (prefix),
one for updates (target). Using the same flags for both is the root cause.

---

## 6. Lessons Learned

1. **"Baked into the artifact" is not a defense for using target data.**
   Whether the information flows through model weights (pre-quant TTT) or
   through hint channel selection (this case), if the scoring distribution
   depends on `x_p`, it's invalid.

2. **Audit data flow, not just algorithmic structure.** The algorithm was
   sound (score-before-update, properly normalized tilt). The bug was in
   *which data* fed into the algorithm.

3. **Separate scoring and update phases explicitly.** Use different variable
   names, or compute flags separately for each phase. Reusing a variable
   across the score/update boundary invites exactly this kind of bug.

4. **Test with the "swap test".** Mentally swap the target token at position
   `p` with a different token while keeping the prefix fixed. If any part
   of the scoring logic would change, you have a causal violation.

5. **The dominant component being clean doesn't validate the whole system.**
   `token_hint` was fully causal, but `within_hint` and `word_hint` were
   not. Review each component independently.
