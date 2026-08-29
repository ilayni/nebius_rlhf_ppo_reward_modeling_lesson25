# Assignment Plan — Section 14 (Student Exercises)

Notebook: `ppo_glassbox_llm-final.ipynb` (Glass-Box PPO on GPT-2).
Goal: complete all 5 exercises in **§14** as a "predict → turn the dial → verify"
submission, with real curves/numbers to back each prediction.

---

## 0. Environment (already verified — no VM/GPU needed)

- **Machine:** Apple M4 Pro, 48 GB RAM. Runs natively on `mps` (Apple GPU).
- **Interpreter:** use **Anaconda** Python, which already has the deps:
  - `$HOME/anaconda3/bin/python` → torch 2.11.0 (mps=True), transformers 5.5.4, matplotlib
  - ⚠️ Do NOT use `/opt/homebrew/bin/python3` — it lacks torch/transformers.
  - In Jupyter, select the **Anaconda** kernel.
- **First run** downloads gpt2 (~548 MB) from HF once, then caches. The
  "unauthenticated HF Hub" warning is harmless.
- **Note on numbers:** device is `mps`, so absolute rewards/KL/clipfrac will
  differ slightly from the notebook's baked-in outputs. Exercises grade on
  **trends**, not exact digits (see `ipynb_gaps.md` #4).

## 1. Baseline run (do this first)

Run the notebook top-to-bottom once (§1–§12) to:
1. Load model + define all helpers (`rollout`, `aligned_logps_values`,
   `compute_gae`, `ppo_update`, `train`, `plot_hist`).
2. Capture the **baseline** curve (§9–§10): confirm reward climbs while KL stays
   bounded at default dials (`clip_eps=0.2, gae_lambda=0.95, kl_coef=0.05`).
3. Confirm `policy_fresh_state` is saved (§9) — every exercise resets to it so
   runs are comparable. **Always `policy.load_state_dict(copy.deepcopy(policy_fresh_state))`
   before each experiment.**

Keep `iters` modest (default 60, or 40 like the sweep) so each run is ~1–3 min.

## 2. Deliverable format (per exercise)

For each of the 5, produce a short block containing:
1. **Prediction** (written BEFORE running — that's the pedagogical point).
2. **Exact change** made (config field or code edit).
3. **Result**: the `plot_hist` figure + final logged line (reward, KL/seq,
   clipfrac, ent). Save each figure to `figs/ex<N>_<variant>.png`.
4. **Verify**: did the observed trend match the prediction? 1–2 sentences.

## 3. Per-exercise run recipes

All use `c = replace(cfg, ...)` then reset weights and `train(c)`.
Exercises 1, 2, 4 are **config-only**. Exercise 3 swaps the reward fn.
Exercise 5 needs a **one-line code edit** in the update.

### Ex 1 — Clip ε (config-only)
Run `clip_eps ∈ {0.05, 0.2 (baseline), 0.6}`. Overlay reward + clipfrac curves.
- Predict: lower ε → clip bites more → **higher** clipfrac, smaller/safer steps,
  slower; higher ε → **lower** clipfrac, bigger riskier steps.
- Answer the prompt: **0.6 = faster but riskier**.

### Ex 2 — GAE λ (config-only)
Run `gae_lambda ∈ {0.0, 0.95 (baseline), 1.0}`. Focus on **reward-curve noise**.
- Predict: λ=0 (one-step TD) = biased but smooth; λ=1 (Monte-Carlo) = unbiased
  but **noisier** reward curve.

### Ex 3 — β vs reward scale (swap reward_fn)
Uncomment the `!`-counting `reward_fn` in §3 (much larger magnitude). Re-run the
β sweep from §12 (`run_sweep([0.0, 0.05, 0.2, 1.0])`).
- Predict: because reward is larger, the "good" β range **shifts upward** — old
  β values now behave like "too small" (KL blows up). Lesson: β is
  **scale-dependent** (same point as DPO's β). **Restore the positive-word
  reward_fn afterward.**

### Ex 4 — PPO epochs (config-only)
Run `ppo_epochs ∈ {4 (baseline), 8}`. Watch clipfrac.
- Predict: more reuse → policy drifts further from the **fixed** old log-probs
  within one rollout → ratio moves off 1 → clip bites more → **clipfrac rises**.
  Data reuse hurts once the rollout is too stale (too off-policy).

### Ex 5 — Kill the critic (code edit)
Make a copy of `ppo_update` (e.g. `ppo_update_nocritic`) that replaces the
GAE advantage with a plain return baseline:
```python
adv = ret - ret.mean()        # instead of adv_n from compute_gae
# and drop / zero the value-function loss term (vf_coef effectively 0)
```
Run vs the baseline.
- Predict: no critic baseline → **higher-variance advantages** → noisier, slower,
  less stable learning. That variance reduction is **what GAE bought you**.

## 4. Reproducibility

- `torch.manual_seed(0)` is set in the notebook; keep it.
- Create a `figs/` folder; `plt.savefig(...)` each figure before `plt.show()`.
- Log the final metric line of every run into a small results table.

## 5. Assembly (final submission)

Compile into a submission doc (new markdown file or a copy of the notebook with
an added "§14 Answers" section) containing, for all 5 exercises: prediction →
change → figure → observed metrics → verification. Include the **Dial → effect
cheat-sheet** from §14 as the summary.

## 6. Time estimate

~10–15 short training runs total (some exercises need 2–3 variants). At ~1–3 min
each on the M4 Pro → roughly **20–45 min** of compute, plus write-up.

---

## Open question for the user (before executing)

- **How to run?** Two options — tell me which:
  - **(A)** I write a single parameterized driver script (Anaconda Python) that
    runs all 5 experiments unattended, saves every figure to `figs/`, and prints
    a results table. Fastest path to a complete set of artifacts.
  - **(B)** You run it interactively in Jupyter cell-by-cell; I prepare the exact
    edited cells + predictions for you to paste and execute.
- **Submission format?** Markdown report, or an annotated copy of the `.ipynb`?
- Confirm `iters` (default 60) is fine, or lower to 40 for speed.
